"""GPU-native YUV ↔ RGB color conversion + a vsrife wrapper that uses
it instead of `core.resize.Bicubic`.

Why this exists
---------------
On DGX Spark (Grace + Blackwell, NVLink-C2C unified memory), the CPU
is the bottleneck for 4K vapoursynth-RIFE playback — not the GPU. A
single `core.resize.Bicubic(format=vs.RGBH, matrix_in_s="709")` at 4K
costs ~33 ms per frame on the CPU. Round-trip (YUV → RGBH → RIFE →
RGBH → YUV) is ~60 ms per real frame. The RIFE engine itself is only
~24 ms (4.6 + scale=0.5 + 4K). So the bicubic on either side eats the
60 fps frame budget before RIFE even starts.

This module replaces the two `core.resize.Bicubic` calls around `rife`
with a unified GPU pipeline: upload YUV planes, run YUV→RGB matrix +
chroma upsample on the GPU, hand RGB to the cached vsrife flownet
TRT engine, run RGB→YUV on the GPU, write YUV planes back. With
unified memory the upload/download is essentially free (CPU and GPU
share LPDDR5X), so the saving is the full ~60 ms / frame of CPU
bicubic — at 4K that's the difference between stutter and smooth.

Public API
----------
  rife_yuv(clip, model="4.26", scale=1.0, factor_num=2, ...)
      Drop-in for vsrife.rife() that takes YUV input directly and
      keeps the whole color-convert + RIFE pipeline on the GPU.

  fsrcnnx_yuv_gpu(clip, weights_npz, variant, ...)
      Drop-in for fsrcnnx_cudnn.vsfunc.fsrcnnx_yuv() with chroma
      upscaling on GPU instead of `core.resize.<Bicubic>`. Saves
      another ~5–20 ms / frame at 4K source.

Limitations
-----------
- Input must be YUV420P10. Other YUV bit-depths and subsamplings
  (YUV420P8/16, YUV422, YUV444) should be normalised upstream via
  `core.resize.Bicubic(format=vs.YUV420P10)` — rife.vpy does this
  before calling us, so the function stays strict on its own input.
- Color matrices supported: BT.709, BT.2020 NCL, BT.601 (smpte170m).
  Chosen per-frame from the source's `_Matrix` prop. HDR PQ/HLG
  works because the matrix is the same in non-linear domain — RIFE
  has always operated on PQ-encoded RGB, not linearized light.
- Both limited (TV) and full (PC) ranges are supported via the
  `color_range` kwarg to `rife_yuv`. Game streaming (Moonlight,
  NVENC), Parsec, ShadowPlay recordings typically encode full-range
  YUV; the rife.vpy reads `_ColorRange` and passes it through.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from collections import OrderedDict
from threading import Condition, Lock

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import vapoursynth as vs


# Make the fsrcnnx-cudnn bundle importable from here (matches the
# resolution logic in sr_keys_helper). We need `chroma_krig` for the
# luma-guided kriging chroma upsample inside `yuv_p10_to_rgb`. Lazy:
# the first call to `_chroma_upsample` tries to import, caches the
# result (or the failure), and falls back to bilinear otherwise.
_MPV_HOME = Path(
    os.environ.get("MPV_HOME") or
    (os.environ.get("XDG_CONFIG_HOME") or
     os.path.expanduser("~/.config")) + "/mpv"
)
_FSRCNNX_BUNDLE = _MPV_HOME / "fsrcnnx-cudnn"
if _FSRCNNX_BUNDLE.exists() and str(_FSRCNNX_BUNDLE) not in sys.path:
    sys.path.insert(0, str(_FSRCNNX_BUNDLE))

# Make `ninja` discoverable by torch's load_inline JIT compiler. mpv-conda
# wrapper sets PYTHONHOME but not PATH, so `shutil.which("ninja")` returns
# None and the chroma_krig kernel compile aborts. Same workaround
# worker.py applies for the dual-machine remote process.
_PY_BIN = str(Path(sys.executable).parent)
if _PY_BIN not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _PY_BIN + os.pathsep + os.environ.get("PATH", "")

_krig_fn: "callable | None | bool" = None    # None = not tried, False = unavailable


def _chroma_upsample(
    y_full: torch.Tensor,    # (H, W) fp32, range-normalised luma in [0, 1]
    u_lo:   torch.Tensor,    # (cH, cW) fp32, range-normalised chroma
    v_lo:   torch.Tensor,    # (cH, cW) fp32, range-normalised chroma
) -> tuple[torch.Tensor, torch.Tensor]:
    """Upsample (u_lo, v_lo) to y_full's shape. Tries kriging (~3 ms
    on GB10, +6-7 dB U/V over bilinear) first; falls back to bilinear
    if `fsrcnnx_cudnn.chroma_krig` is not importable or its kernel
    JIT-compile fails. Inputs are range-normalised per the source's
    _ColorRange (limited stretches Y to [0,1]; chroma sits in ~[-0.5,
    0.5]); krig is linear in chroma, so negative values pass through
    correctly. Returns (u_full, v_full) in the same range as inputs."""
    global _krig_fn
    h, w = y_full.shape
    if _krig_fn is None:
        try:
            from fsrcnnx_cudnn.chroma_krig import krig_bilateral_chroma
            _krig_fn = krig_bilateral_chroma
        except Exception as _e:
            sys.stderr.write(
                f"[vs_gpu_helpers] chroma_krig unavailable "
                f"({type(_e).__name__}: {_e}) — falling back to bilinear\n")
            _krig_fn = False
    if _krig_fn is not False:
        try:
            u_out = torch.empty((1, 1, h, w),
                                  device=y_full.device, dtype=torch.float16)
            v_out = torch.empty_like(u_out)
            _krig_fn(
                y_full[None, None].to(torch.float16),
                u_lo[None, None].to(torch.float16),
                v_lo[None, None].to(torch.float16),
                u_out=u_out, v_out=v_out,
            )
            return (u_out[0, 0].to(torch.float32),
                    v_out[0, 0].to(torch.float32))
        except Exception as _e:
            sys.stderr.write(
                f"[vs_gpu_helpers] krig launch failed "
                f"({type(_e).__name__}: {_e}) — falling back to bilinear\n")
            _krig_fn = False
    u_full = F.interpolate(u_lo[None, None], size=(h, w),
                            mode="bilinear", align_corners=False)[0, 0]
    v_full = F.interpolate(v_lo[None, None], size=(h, w),
                            mode="bilinear", align_corners=False)[0, 0]
    return u_full, v_full


# =============================================================================
# Color conversion math
# =============================================================================

# YUV → RGB coefficients (non-linear, as used in transmission). For each
# matrix the rows are (R, G, B) and the columns are (Y, Cb, Cr) where
# Y, Cb, Cr are already normalized: Y in [0, 1], Cb, Cr in [-0.5, 0.5].
_YUV2RGB_MATRIX = {
    "709":      torch.tensor(
        [[1.0,  0.0,           1.5748],
         [1.0, -0.187324,     -0.468124],
         [1.0,  1.8556,        0.0]],
        dtype=torch.float32),
    "2020ncl":  torch.tensor(
        [[1.0,  0.0,           1.4746],
         [1.0, -0.164553,     -0.571353],
         [1.0,  1.881400,      0.0]],
        dtype=torch.float32),
    "170m":     torch.tensor(   # BT.601 NTSC
        [[1.0,  0.0,           1.402],
         [1.0, -0.344136,     -0.714136],
         [1.0,  1.772,         0.0]],
        dtype=torch.float32),
}

# RGB → YUV: inverse of the above, computed once at import.
_RGB2YUV_MATRIX = {k: torch.linalg.inv(v) for k, v in _YUV2RGB_MATRIX.items()}


# 10-bit YUV scale/offset constants by `_ColorRange` semantics.
# Tuple layout: (y_scale, y_offset, uv_scale, uv_offset).
#   limited (TV): Y in [64, 940], U/V in [64, 960] centred at 512
#   full (PC) : Y in [0, 1023], U/V in [0, 1023]            centred at 512
# Game streaming (Moonlight, NVENC etc.) and many screen recordings encode
# full-range YUV; container metadata sets _ColorRange=0 for those.
_RANGE_CONSTS_10 = {
    "limited": (876.0, 64.0, 896.0, 512.0),
    "full":    (1023.0, 0.0, 1023.0, 512.0),
}

# `_ColorRange` prop → constants key. Missing prop on YUV defaults to
# limited (BT.709/2020 spec); explicit 0 means full (PC).
_RANGE_FROM_PROP = {0: "full", 1: "limited"}


# Map vapoursynth `_Matrix` integer values → matrix name. mpv
# populates this from the container; defaults vary by source.
_MATRIX_FROM_PROP = {
    1: "709",        # BT.709, the default for HD/SDR
    5: "170m",       # BT.470 BG (PAL) — close enough to 170m
    6: "170m",       # smpte170m, NTSC SD
    7: "240m",       # smpte240m — fallback to 709 (rare in 2025)
    9: "2020ncl",    # BT.2020 non-constant-luminance, HDR/UHD
    10: "2020ncl",   # BT.2020 constant — extremely rare; same matrix
}


@torch.inference_mode()
def yuv_p10_to_rgb(
    y_plane: torch.Tensor,          # (H, W) uint16, 10-bit data in [0, 1023]
    u_plane: torch.Tensor,          # (cH, cW) uint16; cH/cW infer subsampling
    v_plane: torch.Tensor,          # (cH, cW) uint16
    matrix: str,
    *,
    color_range: str = "limited",
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Convert 10-bit YUV planes (any subsampling) to RGB in [0, 1].

    Returns (1, 3, H, W) tensor on the same device as the inputs.
    `color_range` is "limited" (TV; default — matches streaming /
    Blu-ray) or "full" (PC; game streaming / screen recordings).
    Subsampling is inferred from u_plane.shape: shape == y_plane.shape
    skips upsampling (4:4:4); otherwise the chroma planes are
    upsampled via `_chroma_upsample` (krig with luma guide, bilinear
    fallback).
    """
    device = y_plane.device
    y_scale, y_offset, uv_scale, uv_offset = _RANGE_CONSTS_10[color_range]
    y = (y_plane.to(torch.float32) - y_offset)  / y_scale
    u = (u_plane.to(torch.float32) - uv_offset) / uv_scale
    v = (v_plane.to(torch.float32) - uv_offset) / uv_scale

    # Chroma upsample to 4:4:4 only if input chroma is subsampled.
    # 4:4:4 (cH==H, cW==W) hits the no-op fast path. Krig sees
    # range-normalized inputs so its luma-similarity weights line up
    # with the same color_range semantics applied to the matrix mul.
    h, w = y.shape
    if u_plane.shape != y_plane.shape:
        u, v = _chroma_upsample(y, u, v)

    # Matrix multiply. Stack into (H, W, 3) [Y, U, V], then dot.
    yuv = torch.stack([y, u, v], dim=-1)  # (H, W, 3)
    m = _YUV2RGB_MATRIX[matrix].to(device)  # (3, 3) [R,G,B] x [Y,U,V]
    rgb = torch.einsum("hwc,rc->hwr", yuv, m)  # (H, W, 3)

    # Don't clamp here. Wide-gamut BT.709/BT.2020 chroma legitimately
    # produces RGB outside [0, 1]; the CPU path (vs.RGBH = fp16 RGB)
    # preserves these negatives and the inverse matrix recovers Y
    # exactly. RIFE itself does an internal `clamp(0, 1)`, so we don't
    # need to enforce it here. The fp16 storage caps the range at
    # roughly ±65k which is far outside any plausible color value.
    rgb = rgb.to(out_dtype)
    # vsrife/RIFE want NCHW.
    return rgb.permute(2, 0, 1).unsqueeze(0).contiguous()


@torch.inference_mode()
def rgb_to_yuv_p10(
    rgb: torch.Tensor,           # (1, 3, H, W) float16/float32 in [0, 1]
    matrix: str,
    *,
    color_range: str = "limited",
    sub_h: int = 1,              # chroma subsampling shift (1 = 4:2:0)
    sub_w: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert RGB to YUV 10-bit planes (int16, bit-equivalent to uint16
    for 10-bit values in [0, 1023]).

    Returns (y, u, v) tensors on the same device as `rgb`. Chroma
    subsampling is configurable via (sub_h, sub_w):
      (1, 1) — 4:2:0 (default; 2×2 avg pool)
      (0, 1) — 4:2:2 (horizontal 1×2 avg pool)
      (0, 0) — 4:4:4 (no chroma downsample — preserves full-res chroma)

    The 4:4:4 path is useful when the consumer (e.g. FSRCNNX SR with
    chroma_h_out == proc_h) will not need to upsample chroma back; it
    avoids one lossy downsample-then-reupsample round-trip.

    `color_range` should match whatever the source had — limited stays
    limited, full stays full — so mpv's downstream display path
    interprets it correctly."""
    device = rgb.device
    rgb_f = rgb[0].permute(1, 2, 0).to(torch.float32)  # (H, W, 3)
    m = _RGB2YUV_MATRIX[matrix].to(device)             # (3, 3) [Y,U,V] x [R,G,B]
    yuv = torch.einsum("hwc,rc->hwr", rgb_f, m)        # (H, W, 3)
    y, u, v = yuv[..., 0], yuv[..., 1], yuv[..., 2]

    if sub_h == 1 and sub_w == 1:
        # 4:2:0 — 2×2 mean pool
        u = F.avg_pool2d(u[None, None], kernel_size=2, stride=2)[0, 0]
        v = F.avg_pool2d(v[None, None], kernel_size=2, stride=2)[0, 0]
    elif sub_h == 0 and sub_w == 1:
        # 4:2:2 — horizontal 1×2 mean pool only
        u = F.avg_pool2d(u[None, None], kernel_size=(1, 2), stride=(1, 2))[0, 0]
        v = F.avg_pool2d(v[None, None], kernel_size=(1, 2), stride=(1, 2))[0, 0]
    elif sub_h == 0 and sub_w == 0:
        # 4:4:4 — no downsample
        pass
    else:
        raise ValueError(f"unsupported chroma subsampling ({sub_h}, {sub_w})")

    # Quantize to 10-bit using the matching range's constants.
    y_scale, y_offset, uv_scale, uv_offset = _RANGE_CONSTS_10[color_range]
    y_q = (y * y_scale  + y_offset ).round_().clamp_(0, 1023).to(torch.int16)
    u_q = (u * uv_scale + uv_offset).round_().clamp_(0, 1023).to(torch.int16)
    v_q = (v * uv_scale + uv_offset).round_().clamp_(0, 1023).to(torch.int16)
    return y_q, u_q, v_q


# Backwards-compatibility alias — existing callers want the 4:2:0 default.
def rgb_to_yuv420p10(
    rgb: torch.Tensor, matrix: str, *, color_range: str = "limited",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return rgb_to_yuv_p10(rgb, matrix, color_range=color_range,
                           sub_h=1, sub_w=1)


# =============================================================================
# Frame ↔ tensor helpers (CPU-side glue)
# =============================================================================

def _frame_yuv420p10_to_planes(
    frame: vs.VideoFrame, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pull the three planes off a YUV420P10 frame as uint16 tensors
    on the GPU. With Grace-Blackwell unified memory the .to(device)
    call is essentially a metadata move — no PCIe traffic involved."""
    y = torch.from_numpy(np.asarray(frame[0])).to(device, non_blocking=True)
    u = torch.from_numpy(np.asarray(frame[1])).to(device, non_blocking=True)
    v = torch.from_numpy(np.asarray(frame[2])).to(device, non_blocking=True)
    return y, u, v


def _planes_to_frame_yuv420p10(
    y: torch.Tensor, u: torch.Tensor, v: torch.Tensor, dst_frame: vs.VideoFrame,
) -> vs.VideoFrame:
    """Write three int16 plane tensors back into a YUV420P10 frame.

    Same int16-buffer-alias trick fsrcnnx_cudnn.vsfunc uses: vapoursynth
    plane buffers are uint16 numpy views; we reinterpret each as int16
    (same bit layout for 10-bit YUV's [0, 1023] range) and wrap in a
    torch CPU tensor that aliases the destination buffer. Then a single
    `copy_` from GPU int16 directly writes into the vapoursynth frame —
    no `.cpu().numpy() + np.copyto(dst, ...)` chain that bounces 4K
    plane data through CPU memory twice. Saves ~3-5 ms / output frame
    at 4K resolution on the 20-core ARM."""
    y_dst_t = torch.from_numpy(np.asarray(dst_frame[0]).view(np.int16))
    u_dst_t = torch.from_numpy(np.asarray(dst_frame[1]).view(np.int16))
    v_dst_t = torch.from_numpy(np.asarray(dst_frame[2]).view(np.int16))
    y_dst_t.copy_(y, non_blocking=False)
    u_dst_t.copy_(u, non_blocking=False)
    v_dst_t.copy_(v, non_blocking=False)
    return dst_frame


def _detect_matrix(clip: vs.VideoNode) -> str:
    """Pick a matrix name from the first frame's `_Matrix` prop."""
    try:
        m = int(clip.get_frame(0).props.get("_Matrix", 1))
    except Exception:
        m = 1
    return _MATRIX_FROM_PROP.get(m, "709")


# =============================================================================
# rife_yuv: vsrife wrapper with GPU color conversion
# =============================================================================

def _vsrife_engine_paths(model: str, ph: int, pw: int, scale: float,
                         fp16: bool, device: torch.device) -> tuple[Path, Path | None]:
    """Reproduce vsrife's cached-engine filename. Returns (flownet, encode_or_none)."""
    import vsrife
    import tensorrt
    model_dir = Path(vsrife.__file__).parent / "models"
    gpu_name = torch.cuda.get_device_name(device)
    base = (f"flownet_v{model}.pkl_{pw}x{ph}"
            f"_{'fp16' if fp16 else 'fp32'}_scale-{scale}"
            f"_ensemble-False_{gpu_name}_trt-{tensorrt.__version__}.ts")
    flow = model_dir / base
    enc = model_dir / (base + ".encode")
    return flow, (enc if enc.exists() else None)


def _ensure_engines(model: str, h: int, w: int, scale: float,
                    fp16: bool, device: torch.device) -> tuple[Path, Path | None]:
    """If the engines aren't cached, run a 1-frame clip through vsrife
    to trigger compile. With install.sh's warmup step this should be a
    cache hit for production shapes; first-time custom shapes pay the
    one-time TRT compile (~30-60s)."""
    import math
    # Determine modulo + padded shape (matches vsrife)
    if model in ("4.25", "4.25.lite", "4.25.heavy", "4.26", "4.26.heavy"):
        modulo_base = 64
    else:
        modulo_base = 32
    tmp = max(modulo_base, int(modulo_base / scale))
    pw = math.ceil(w / tmp) * tmp
    ph = math.ceil(h / tmp) * tmp

    flow_path, enc_path = _vsrife_engine_paths(model, ph, pw, scale, fp16, device)
    if not flow_path.exists():
        # Trigger build via vsrife on a dummy 1-frame clip at the right
        # native resolution. vsrife's static-shape mode bakes the engine
        # to (pw, ph) and writes it to its cache dir.
        from vsrife import rife as _rife
        dummy_format = vs.RGBH if fp16 else vs.RGBS
        dummy = vs.core.std.BlankClip(width=w, height=h, format=dummy_format,
                                       length=2, fpsnum=30, fpsden=1)
        clip = _rife(dummy, model=model, scale=scale,
                     factor_num=2, factor_den=1,
                     auto_download=True, trt=True)
        clip.get_frame(0)
        flow_path, enc_path = _vsrife_engine_paths(
            model, ph, pw, scale, fp16, device,
        )
        if not flow_path.exists():
            raise vs.Error(f"rife_yuv: engine compile didn't produce expected "
                           f"file: {flow_path}")
    return flow_path, enc_path


# Cache loaded engines + configs across calls.
_engine_cache: dict[tuple, dict] = {}
_engine_lock = Lock()


def _load_engines(model: str, h: int, w: int, scale: float,
                  fp16: bool, device: torch.device) -> dict:
    key = (model, h, w, scale, fp16, str(device))
    with _engine_lock:
        cached = _engine_cache.get(key)
        if cached is not None:
            return cached

        # Make sure torch_tensorrt is imported so the custom op class
        # loads before torch.jit.load() of the .ts file.
        import torch_tensorrt  # noqa: F401

        flow_path, enc_path = _ensure_engines(model, h, w, scale, fp16, device)

        # Compute padded shape (same calculation as _ensure_engines).
        if model in ("4.25", "4.25.lite", "4.25.heavy", "4.26", "4.26.heavy"):
            modulo_base = 64
        else:
            modulo_base = 32
        tmp = max(modulo_base, int(modulo_base / scale))
        pw = math.ceil(w / tmp) * tmp
        ph = math.ceil(h / tmp) * tmp

        flownet = torch.jit.load(str(flow_path)).eval()
        encode = (torch.jit.load(str(enc_path)).eval()
                  if enc_path is not None else None)

        # encode_channel — the dim we have to allocate f0/f1 with.
        # vsrife hardcodes this per-model; we replicate the table.
        ec_table = {
            "4.0": 0, "4.1": 0, "4.2": 0, "4.3": 0, "4.4": 0, "4.5": 0, "4.6": 0,
            "4.7": 4, "4.8": 4, "4.9": 4,
            "4.10": 8, "4.11": 8, "4.12": 8, "4.12.lite": 4,
            "4.13": 8, "4.13.lite": 4, "4.14": 8, "4.14.lite": 8,
            "4.15": 8, "4.15.lite": 4, "4.16.lite": 4,
            "4.17": 8, "4.17.lite": 4,
            "4.18": 8, "4.19": 8, "4.20": 8, "4.21": 8, "4.22": 8, "4.22.lite": 4,
            "4.23": 8, "4.24": 8,
            "4.25": 4, "4.25.lite": 4, "4.25.heavy": 4,
            "4.26": 4, "4.26.heavy": 16,
        }
        encode_channel = ec_table.get(model, 0)

        # Pre-build the warp grid (same per resolution).
        dtype_t = torch.float
        tenFlow_div = torch.tensor([(pw - 1.0) / 2.0, (ph - 1.0) / 2.0],
                                    dtype=dtype_t, device=device)
        tenH = torch.linspace(-1.0, 1.0, pw, dtype=dtype_t, device=device)
        tenH = tenH.view(1, 1, 1, pw).expand(-1, -1, ph, -1)
        tenV = torch.linspace(-1.0, 1.0, ph, dtype=dtype_t, device=device)
        tenV = tenV.view(1, 1, ph, 1).expand(-1, -1, -1, pw)
        backwarp_tenGrid = torch.cat([tenH, tenV], 1).contiguous()

        cached = {
            "flownet": flownet, "encode": encode,
            "encode_channel": encode_channel,
            "ph": ph, "pw": pw, "h": h, "w": w,
            "padding": (0, pw - w, 0, ph - h),
            "need_pad": (pw != w or ph != h),
            "tenFlow_div": tenFlow_div, "backwarp_tenGrid": backwarp_tenGrid,
            "stream_inf": torch.cuda.Stream(device),
            "stream_io": torch.cuda.Stream(device),
            "lock_inf": Lock(), "lock_io": Lock(),
        }
        _engine_cache[key] = cached
        return cached


@torch.inference_mode()
def rife_yuv(
    clip: vs.VideoNode,
    *,
    model: str = "4.26",
    scale: float = 1.0,
    factor_num: int = 2,
    factor_den: int = 1,
    device_index: int = 0,
    color_range: str = "limited",
    target_subsample: str = "420",
) -> vs.VideoNode:
    """Frame-doubling RIFE with GPU color conversion. Drop-in for the
    sequence

        clip = core.resize.Bicubic(clip, format=vs.RGBH, matrix_in_s=_mtx)
        clip = vsrife.rife(clip, model=..., scale=..., trt=True)
        clip = core.resize.Bicubic(clip, format=vs.YUV420P10, matrix_s=_mtx)

    with the two CPU bicubics moved onto the GPU. Engine selection +
    cache filename match vsrife exactly, so existing cached engines
    (built by install.sh's warmup step) are reused as-is.

    target_subsample ("420" default, "444" optional): when "444" the
    output clip is YUV444P10 — chroma is kept at full luma resolution
    instead of avg-pooled back to 4:2:0. Useful when the SR consumer
    (apply_fsrcnnx with chroma_h_out == clip.height) can leverage the
    fully-resolved chroma without a downsample-then-upsample round-trip.
    """
    if clip.format is None or clip.format.color_family != vs.YUV:
        raise vs.Error("rife_yuv: input must be YUV")
    if clip.format.id != int(vs.YUV420P10):
        raise vs.Error("rife_yuv: only YUV420P10 supported "
                        f"(got {clip.format.name})")
    if target_subsample not in ("420", "444"):
        raise vs.Error(
            f"rife_yuv: target_subsample must be '420' or '444' "
            f"(got {target_subsample!r})")
    _OUT_SUB = {"420": (1, 1), "444": (0, 0)}[target_subsample]
    _OUT_FORMAT = {"420": vs.YUV420P10, "444": vs.YUV444P10}[target_subsample]

    # Don't sniff the matrix at init — mpv's vapoursynth integration
    # disallows frame requests before the filter graph is fully wired,
    # and `clip.get_frame(0).props["_Matrix"]` would trigger one. We
    # read `_Matrix` per frame inside the ModifyFrame callbacks below.
    device = torch.device("cuda", device_index)
    fp16 = True   # we always feed fp16 RGB to the engine

    cfg = _load_engines(model, clip.height, clip.width, scale, fp16, device)
    flownet = cfg["flownet"]; encode = cfg["encode"]
    ec = cfg["encode_channel"]
    ph, pw = cfg["ph"], cfg["pw"]
    h, w = cfg["h"], cfg["w"]
    padding = cfg["padding"]; need_pad = cfg["need_pad"]
    tenFlow_div = cfg["tenFlow_div"]
    backwarp_tenGrid = cfg["backwarp_tenGrid"]
    stream_inf = cfg["stream_inf"]; stream_io = cfg["stream_io"]
    lock_inf = cfg["lock_inf"]; lock_io = cfg["lock_io"]

    # Engine cuDNN-graph warmup: feed zeros through flownet (+ encode
    # head when present) once at construction time so the graph
    # compile cost is paid here instead of on mpv's first inference
    # request. Pure synthetic input — never touches mpv's source clip.
    # Avoids the failure mode where mpv vf_vapoursynth returns
    # uninitialised black frames during script load and a
    # `clip.get_frame()`-based prewarm caches those.
    import time as _warm_time
    _t_warm0 = _warm_time.perf_counter()
    with torch.inference_mode(), torch.cuda.stream(stream_inf):
        _zero_rgb = torch.zeros(1, 3, ph, pw, dtype=torch.float16, device=device)
        _zero_t = torch.zeros(1, 1, ph, pw, dtype=torch.float16, device=device)
        _f0_warm = encode(_zero_rgb) if encode is not None else None
        _f1_warm = encode(_zero_rgb) if encode is not None else None
        if encode is not None:
            flownet(_zero_rgb, _zero_rgb, _zero_t, tenFlow_div,
                    backwarp_tenGrid, _f0_warm, _f1_warm)
        else:
            flownet(_zero_rgb, _zero_rgb, _zero_t, tenFlow_div,
                    backwarp_tenGrid)
        stream_inf.synchronize()
    del _zero_rgb, _zero_t, _f0_warm, _f1_warm
    import sys as _sys
    _sys.stderr.write(
        f"[rife_yuv] engine warm {model}@{w}x{h} (pad {pw}x{ph}) in "
        f"{(_warm_time.perf_counter()-_t_warm0)*1000:.0f}ms\n")
    _sys.stderr.flush()

    # Per-source-frame caches: rgb tensor (img0/img1 input to flownet),
    # f0 (encode output), and the matrix string (so the reverse RGB→YUV
    # uses the same matrix the input was decoded with). Keyed by
    # source-frame index.
    #
    # Bounded LRU eviction: when an offline / async consumer requests
    # many frames in parallel, the encoding pass can run far ahead of
    # inference. A naive "evict if k < n - W" policy on the encoding
    # side then deletes entries the inference pass still needs (KeyError,
    # or — worse — silent miss-and-re-encode). We use OrderedDict with
    # a hard cap on size and evict-on-insert so reads never observe a
    # stale-eviction race. cache_lock serialises mutation; reads are
    # safe under the GIL because dict access is atomic per-op.
    # Maximum cached source-frame entries. With factor_num=2 and mpv's
    # buffered-frames=12 / concurrent-frames=24 (saturates at ~20 in
    # flight), at most ~20 distinct source frames are referenced
    # concurrently; 64 leaves plenty of headroom. At 1080p RGB16 a
    # single entry is ~12 MB → ~770 MB GPU cap (similar for f0_cache);
    # fits well inside our 100+ GB unified memory budget.
    CACHE_MAX = 64
    rgb_cache: "OrderedDict[int, torch.Tensor]" = OrderedDict()
    f0_cache: "OrderedDict[int, torch.Tensor]" = OrderedDict()
    matrix_cache: "OrderedDict[int, str]" = OrderedDict()
    cache_lock = Lock()
    in_progress: set[int] = set()       # frames currently being encoded
    progress_cv = Condition(cache_lock)

    has_head = (encode is not None)

    def _frame_matrix(src_frame: vs.VideoFrame) -> str:
        try:
            m = int(src_frame.props.get("_Matrix", 1))
        except Exception:
            m = 1
        return _MATRIX_FROM_PROP.get(m, "709")

    def _cache_get(n: int) -> tuple[torch.Tensor | None,
                                     torch.Tensor | None, str | None]:
        """Atomic 3-cache lookup. Returns (rgb, f0, matrix) or Nones.
        Touch-on-read so an entry actively being consumed by a slow
        inference pass doesn't get LRU-evicted out from under it."""
        with cache_lock:
            if n in rgb_cache:
                rgb_cache.move_to_end(n)
                if n in f0_cache:
                    f0_cache.move_to_end(n)
                if n in matrix_cache:
                    matrix_cache.move_to_end(n)
                return rgb_cache[n], f0_cache.get(n), matrix_cache.get(n)
            return None, None, None

    def _cache_put(n: int, rgb: torch.Tensor,
                   f0: torch.Tensor | None, matrix: str) -> None:
        """Atomic insert + LRU evict. Wakes any thread waiting on n."""
        with cache_lock:
            rgb_cache[n] = rgb
            if f0 is not None:
                f0_cache[n] = f0
            matrix_cache[n] = matrix
            while len(rgb_cache) > CACHE_MAX:
                k, _ = rgb_cache.popitem(last=False)
                f0_cache.pop(k, None)
                matrix_cache.pop(k, None)
            in_progress.discard(n)
            progress_cv.notify_all()

    @torch.inference_mode()
    def _convert_and_encode(n: int, src_frame: vs.VideoFrame) -> None:
        """YUV → padded RGB → encode(f0). Stores both in caches under
        key n. Idempotent: if another thread is already encoding `n`,
        wait for it instead of redoing the work; if `n` is already
        cached, no-op. If the in-progress encoder fails, the waiter
        falls through and retries the encode itself."""
        with cache_lock:
            while True:
                if n in rgb_cache:
                    return
                if n not in in_progress:
                    in_progress.add(n)
                    break
                # Another thread is encoding `n` — wait for it. On
                # wake, re-check: if it succeeded the entry is in
                # rgb_cache; if it failed, in_progress no longer
                # contains `n` and we can claim it ourselves.
                progress_cv.wait()

        try:
            matrix = _frame_matrix(src_frame)
            with lock_io, torch.cuda.stream(stream_io):
                yp, up, vp = _frame_yuv420p10_to_planes(src_frame, device)
                rgb = yuv_p10_to_rgb(yp, up, vp, matrix,
                                       color_range=color_range,
                                       out_dtype=torch.float16)
                if need_pad:
                    rgb = F.pad(rgb, padding)
                stream_io.synchronize()

            f0 = None
            if has_head:
                with lock_inf, torch.cuda.stream(stream_inf):
                    f0 = encode(rgb)
                    stream_inf.synchronize()

            _cache_put(n, rgb, f0, matrix)
        except BaseException:
            with cache_lock:
                in_progress.discard(n)
                progress_cv.notify_all()
            raise

    @torch.inference_mode()
    def _encoding_pass(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        """ModifyFrame callback for the encoding pass. Triggers
        _convert_and_encode (idempotent) and returns the source frame
        unchanged. No eviction here — the LRU cap inside _cache_put
        handles memory pressure without racing against concurrent
        inference reads."""
        rgb, _, _ = _cache_get(n)
        if rgb is None:
            _convert_and_encode(n, f)
        return f

    @torch.inference_mode()
    def _inference_pass(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        """ModifyFrame callback for an output frame. n is the output-clip
        index (factor_num × source); f is [clip0[n], clip1[n], dst]."""
        real_n = n * factor_den // factor_num
        real_n_next = min(real_n + 1, clip.num_frames - 1)
        t = (n * factor_den) % factor_num / factor_num

        # Pass-through real frames (t == 0): convert YUV → RGB just to
        # cache (for subsequent interpolated frames). For 4:2:0 output
        # we can return the source frame directly; for 4:4:4 output
        # we have to materialise a chroma-upsampled version because
        # the source frame's YUV420P10 layout doesn't match the
        # out_skeleton's YUV444P10. We synthesise it from the cached
        # RGB (which already holds chroma at full luma resolution
        # because yuv_p10_to_rgb upsampled it during the encoding
        # pass) by running the reverse with sub_h=sub_w=0.
        if t == 0:
            rgb, _, matrix = _cache_get(real_n)
            if rgb is None:
                _convert_and_encode(real_n, f[0])
                rgb, _, matrix = _cache_get(real_n)
            if target_subsample == "420":
                return f[0]
            rgb_unpadded = rgb[:, :, :h, :w] if need_pad else rgb
            with lock_inf, torch.cuda.stream(stream_inf):
                yq, uq, vq = rgb_to_yuv_p10(rgb_unpadded, matrix,
                                              color_range=color_range,
                                              sub_h=_OUT_SUB[0],
                                              sub_w=_OUT_SUB[1])
                stream_inf.synchronize()
            return _planes_to_frame_yuv420p10(yq, uq, vq, f[2])

        # Interpolated frame: ensure both source frames are in cache,
        # then take a strong reference to the cache entries before we
        # release the lock — that way an LRU eviction triggered by a
        # concurrent encode of frame N+K can't free the GPU memory we're
        # about to read in flownet().
        img0, f0, matrix = _cache_get(real_n)
        if img0 is None:
            _convert_and_encode(real_n, f[0])
            img0, f0, matrix = _cache_get(real_n)
        img1, f1, _ = _cache_get(real_n_next)
        if img1 is None:
            _convert_and_encode(real_n_next, f[1])
            img1, f1, _ = _cache_get(real_n_next)

        timestep = torch.full([1, 1, ph, pw], t,
                               dtype=torch.float16, device=device)

        with lock_inf, torch.cuda.stream(stream_inf):
            if has_head:
                out = flownet(img0, img1, timestep, tenFlow_div,
                              backwarp_tenGrid, f0, f1)
            else:
                out = flownet(img0, img1, timestep, tenFlow_div,
                              backwarp_tenGrid)
            if need_pad:
                out = out[:, :, :h, :w]
            # Use the matrix the source frame came in with — videos with
            # mid-stream matrix changes are extremely rare; for the common
            # case both source frames share a matrix.
            yq, uq, vq = rgb_to_yuv_p10(out, matrix, color_range=color_range,
                                          sub_h=_OUT_SUB[0], sub_w=_OUT_SUB[1])
            stream_inf.synchronize()

        # Write GPU result directly into the BlankClip dst frame (no
        # .copy() — saves a 6 MB memset per interp frame at 1080p).
        # `_MP_IMAGE` and other mpv frame props are restored by the
        # CopyFrameProps wrap below, not per-frame here.
        return _planes_to_frame_yuv420p10(yq, uq, vq, f[2])

    # Build the output clip skeleton (factor_num × source FPS).
    # Wire `_encoding_pass` as a ModifyFrame on the SOURCE clip so the
    # vapoursynth scheduler can pre-fetch and encode source frames in
    # parallel with inference work. Without this, encoding happens
    # synchronously inside `_inference_pass`, eating the interp-frame
    # budget — interp frames spike to ~20 ms and miss vsync at 60 Hz.
    encoded = clip.std.ModifyFrame(clips=clip, selector=_encoding_pass)
    base = encoded
    interleaved = vs.core.std.Interleave([base] * factor_num)
    next_clip = base.std.DuplicateFrames(base.num_frames - 1)[1:]
    next_clip = vs.core.std.Interleave([next_clip] * factor_num)
    if factor_den > 1:
        interleaved = interleaved[::factor_den]
        next_clip = next_clip[::factor_den]

    # When target_subsample == "444", switch the output skeleton from
    # YUV420P10 (inherited from interleaved) to YUV444P10 so the
    # _planes_to_frame writer sees the correct plane dimensions.
    #
    # keep=False is mandatory here: keep=True makes BlankClip return
    # the SAME VSFrame for every output index, and our _inference_pass
    # writes through f[2] in-place. Under vapoursynth concurrent-frames
    # > 1, two worker threads write through the same physical buffer
    # at the same time → chroma drift across runs (Y often lands
    # bit-identical because the GPU copy is plane-ordered Y→U→V and
    # the last writer's Y often overlaps the first writer's Y at the
    # same content, while U/V vary). Same root cause as the
    # fsrcnnx-cudnn upstream fix e1344b8 — see [[fsrcnnx-runner-race-fix]]
    # memory. Per-request fresh buffer eliminates the aliasing.
    if target_subsample == "444":
        out_skeleton = interleaved.std.BlankClip(keep=False, format=vs.YUV444P10)
    else:
        out_skeleton = interleaved.std.BlankClip(keep=False)
    out = out_skeleton.std.ModifyFrame(
        clips=[interleaved, next_clip, out_skeleton],
        selector=_inference_pass,
    )
    # Propagate frame props from `interleaved` (matches output frame
    # count): preserves mpv's `_MP_IMAGE`, `_Matrix`, HDR metadata,
    # etc. on output frames. Without this, mpv's vf chain crashes
    # on the missing `_MP_IMAGE` prop. Cheaper than f[0].copy() in
    # the selector — clip-level filter, no per-frame plane memcpy.
    return out.std.CopyFrameProps(prop_src=interleaved)


# =============================================================================
# FSRCNNX integration: use `fsrcnnx_cudnn.vsfunc.fsrcnnx_yuv` directly.
# Override `chroma_kernel="Bicubic"` on this hardware — DGX Spark's
# 20-core ARM gets CPU-bound when zimg Lanczos chroma stacks on top
# of rife_yuv's plane I/O at 60 fps. Bicubic moves chroma onto the
# GPU and recovers ~5% headroom. (Upstream default is Lanczos for
# the typical desktop case; see fsrcnnx-cudnn PERFORMANCE.md.)
#
# Example:
#     from vsrife import rife
#     from vs_gpu_helpers import rife_yuv
#     from fsrcnnx_cudnn.vsfunc import fsrcnnx_yuv
#     clip = rife_yuv(video_in, model="4.6", scale=1.0,
#                     factor_num=2, factor_den=1)
#     clip = fsrcnnx_yuv(clip, weights_npz=..., variant="...",
#                        chroma_kernel="Bicubic")
#     clip.set_output()
