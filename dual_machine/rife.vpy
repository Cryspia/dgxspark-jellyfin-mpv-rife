# Unified RIFE + FSRCNNX pipeline.
#
# F8 cycles FSRCNNX variant (16x4 → 16x3 → 16x2 → 8x2 → OFF → loop).
# F9 toggles RIFE on/off independently. Shift+F9 toggles the dual-machine
# offload (only effective when DUAL_WORKER_HOST is set). Shift+F8 toggles
# the KrigBilateral chroma GLSL shader (mpv's display-stage chroma
# upsample; orthogonal to in-vapoursynth krig).
# Bound by scripts/sr_keys.lua.
#
# Single-machine presets (dual offline / disabled):
#   h ≤ 720         RIFE 4.26 + scale=1.0  + FSRCNNX family=16-layer
#   720 < h ≤ 1080  RIFE 4.6  + scale=1.0  + FSRCNNX family=8-layer
#   1080 < h ≤ 2160 mixed mode — original 4K real frames passthrough;
#                   interp frames go through downsample → RIFE 4.26 →
#                   FSRCNNX 16-layer → upsample back to 4K
#   fps > 30        RIFE skipped, FSRCNNX still runs if ratio merits
#
# Dual-machine preset (DUAL_WORKER_HOST=worker, not toggled off):
#   1080p YUV420P10, fps ≤ 30:
#     RIFE 4.26 factor=2 + FSRCNNX 16-layer x2 (fixed; host CCSR +
#     guest INTERP in parallel hold 60 fps output).
#   Other resolutions / fps fall back to the single-machine chain.

import os, sys
from pathlib import Path

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path(os.environ.get("MPV_HOME") or
                 os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
                 + "/mpv").resolve()
sys.path.insert(0, str(HERE))
# dual_machine/ subdir contains the remote-SR client + common helpers
_DM = HERE / "dual_machine"
if _DM.is_dir() and str(_DM) not in sys.path:
    sys.path.insert(0, str(_DM))
# dual_machine/native_filter/ holds the native-filter Python helper
# (dgxspark_native_helper.py). Add it to the import path so the
# bridge module is resolvable from this vpy.
_NF = _DM / "native_filter"
if _NF.is_dir() and str(_NF) not in sys.path:
    sys.path.insert(0, str(_NF))

import vapoursynth as vs
import vsrife
from vs_gpu_helpers import rife_yuv
from sr_keys_helper import apply_fsrcnnx, rife_disabled

core = vs.core
clip = video_in

# Default the dual-state file to "0" up front. The inner block flips it
# to "1" only on a successful dual-machine activation; otherwise (single
# path, fallback, unsupported codec hitting the outer guard) the file
# stays at "0", which is what scripts/sr_keys.lua's Shift+F9 reads to
# decide which direction to toggle next.
try:
    with open("/tmp/dual_machine_active", "w") as _f:
        _f.write("0")
except OSError:
    pass

# Defensive guard: if anything in chain construction fails — RGB / 4:2:2 /
# 4:4:4 / float YUV / Dolby Vision ICtCp / missing weights / cuDNN graph
# build error — pass the source through unmodified. mpv would auto-disable
# the filter on failure anyway and play the bare source; doing it here
# explicitly avoids the scary "filter failed" log line and keeps the
# user's playback uninterrupted. For exotic formats the interpolation
# wouldn't have kept up at vsync anyway, so the trade is fine.
try:
    # Whitelist the colour properties we actually support. The chain's
    # YUV↔RGB math hard-codes BT.709/601/2020 matrices; anything else
    # (Dolby Vision ICtCp, niche HDR variants) would silently produce
    # wrong colours rather than crash. Catch those at init and bail to
    # passthrough. Both limited (TV, default for streaming/Blu-ray)
    # and full (PC, common for game streaming / screen recordings)
    # ranges are supported via the color_range parameter to rife_yuv.
    _KNOWN_MATRICES = {1, 5, 6, 7, 9, 10}    # 709, 170m, 240m, 2020ncl
    _MATRIX_S = {1: "709", 5: "470bg", 6: "170m", 7: "240m",
                 9: "2020ncl", 10: "2020cl"}
    _src_props = clip.get_frame(0).props
    _matrix = int(_src_props.get("_Matrix", 1))
    _color_range_int = int(_src_props.get("_ColorRange", 1))
    if _matrix not in _KNOWN_MATRICES:
        raise RuntimeError(f"unsupported _Matrix={_matrix} "
                             f"(known: {sorted(_KNOWN_MATRICES)})")
    if _color_range_int not in (0, 1):
        raise RuntimeError(f"unsupported _ColorRange={_color_range_int} "
                             f"(need 0=full or 1=limited)")
    color_range = "full" if _color_range_int == 0 else "limited"
    matrix_s = _MATRIX_S[_matrix]

    h = clip.height
    w = clip.width
    fps = (clip.fps_num / clip.fps_den) if clip.fps_den else 24.0
    # Cinema-rate (≤24 fps) sources have a 20.8 ms/frame budget at 2×
    # output; 25–30 fps sources tighten to 16.7 ms/frame at 60 fps. The
    # heavy chain (4.26 + 16-layer) fits the cinema budget but blows
    # past 16.7 ms — fall back to lighter (4.6 + 8-layer) above 24 fps.
    heavy_fps = fps < 25

    # Pick the working format: keep 4:2:0/4:2:2/4:4:4 as-is to preserve
    # chroma fidelity all the way through CC + RIFE; non-standard layouts
    # (1-plane GRAY, RGB, 8-bit YUV) get coerced to YUV420P10. rife_yuv's
    # GPU YUV↔RGB path is 4:2:0-only, so 4:2:2 / 4:4:4 routes through
    # vsrife direct with zimg YUV↔RGB; native_dispatcher accepts any
    # of the three subsamplings.
    _sw, _sh = clip.format.subsampling_w, clip.format.subsampling_h
    _bits = clip.format.bits_per_sample
    _is_p10_yuv = (clip.format.num_planes == 3
                   and clip.format.sample_type == vs.INTEGER
                   and _bits == 10)
    if not _is_p10_yuv or (_sw, _sh) not in {(0, 0), (1, 0), (1, 1)}:
        clip = core.resize.Bicubic(clip, format=vs.YUV420P10)
        _sw, _sh = 1, 1

    def _rife(c, model, scale=1.0, factor_num=2, target_subsample="420"):
        """4:2:0: rife_yuv (GPU YUV↔RGB, fast path). 4:2:2: vsrife
        direct with zimg YUV↔RGB. Both produce a YUV clip with the
        subsampling controlled by `target_subsample` ("420" matches
        input, "444" keeps RIFE-decoded chroma at full luma res so
        downstream SR can skip the round-trip)."""
        if c.format.id == int(vs.YUV420P10):
            return rife_yuv(c, model=model, scale=scale,
                            factor_num=factor_num, factor_den=1,
                            color_range=color_range,
                            target_subsample=target_subsample)
        rgb = core.resize.Bicubic(c, format=vs.RGBH,
                                   matrix_in_s=matrix_s, range_in_s=color_range)
        rgb = vsrife.rife(rgb, model=model, scale=scale,
                           factor_num=factor_num, factor_den=1, trt=True)
        out_fmt = (vs.YUV444P10 if target_subsample == "444"
                   else c.format.id)
        return core.resize.Bicubic(rgb, format=out_fmt,
                                    matrix_s=matrix_s, range_s=color_range)

    # ──────────────────────────────────────────────────────────────────
    # Dual-machine offload: when DUAL_WORKER_HOST is set and the source
    # is 1080p YUV420P10 ≤ 30 fps, route through the native C++ filter
    # → native_dispatcher (SPLIT-task + CCSR pipeline). Anything else
    # (other resolutions, missing worker, runtime toggle off) falls
    # through to the single-machine path below.
    # ──────────────────────────────────────────────────────────────────
    dual_active = False
    apply_sr = apply_fsrcnnx          # local default
    rife_factor = 2
    # Single-mode x3 / x4 (AB-comparison only — perf is poor).
    # Effective only when DUAL_WORKER_HOST is unset (i.e. we're
    # already on the single-machine path). When DUAL_WORKER_HOST is
    # set we leave rife_factor at 2 here and let dual_active wiring
    # decide. Dual mode uses DUAL_INTERP_MULT (see below). Production
    # UX caps single mode at 2x; AB env override allows 3/4 for PSNR
    # tooling.
    if not os.environ.get("DUAL_WORKER_HOST", "").strip():
        _single_mult_env = os.environ.get("SINGLE_INTERP_MULT", "2")
        if _single_mult_env in ("3", "4"):
            rife_factor = int(_single_mult_env)
    rife_model_pref = None             # if set, override the model picker
    _DUAL_DISABLED_FLAG = "/tmp/dual_machine_disabled"
    _DUAL_MULT_OVERRIDE = "/tmp/dual_machine_mult_override"
    _DUAL_SR_DISABLED = "/tmp/dual_machine_sr_disabled"
    # Per-source default DUAL_INTERP_MULT. Mirrors the rules in
    # scripts/dual_fps_override.lua — the lua sidecar normally
    # writes the chosen mult to _DUAL_MULT_OVERRIDE before this
    # block runs; this is the fallback for sources the lua skipped
    # (network streams, --no-config benches, clips with fps_den=0).
    #   src_fps > 30 fps          → mult=1 (no temporal upscale)
    #   ≤ 720p + ≤ 25 fps         → mult=4
    #   ≤ 720p + 26-30 fps        → mult=3
    #   1080p,4K + ≤ 25 fps       → mult=3
    #   1080p,4K + 26-30 fps      → mult=2
    try:
        _src_fps = clip.fps_num / clip.fps_den if clip.fps_den else 0.0
    except Exception:
        _src_fps = 0.0
    _src_h = int(clip.height or 0)
    if _src_fps <= 0.0:
        _default_dual_mult = 2  # fps unknown — conservative
    elif _src_fps > 30.0:
        _default_dual_mult = 1
    else:
        _max_mult = 4 if _src_h <= 720 else 3
        _default_dual_mult = _max_mult if _src_fps <= 25.0 else _max_mult - 1
    if not os.environ.get("DUAL_INTERP_MULT", "").strip():
        os.environ["DUAL_INTERP_MULT"] = str(_default_dual_mult)
    # F9 dual cycle writes "1"/"2"/"3"/"4" here (1 = no_interp / SR-only).
    # Override file takes precedence over the resolution-derived default.
    try:
        with open(_DUAL_MULT_OVERRIDE) as _f:
            _mult_ov = _f.read().strip()
        if _mult_ov in ("1", "2", "3", "4"):
            os.environ["DUAL_INTERP_MULT"] = _mult_ov
    except OSError:
        pass
    # F8 dual SR toggle. When this flag exists, the dual session asks
    # the worker to skip FSRCNNX SR (RIFE INTERP at source dim, no
    # upscale). The 4K-DS bucket rejects no_sr because RIFE there runs
    # at half-res and needs SR to upscale back to source dim — falls
    # back to single chain instead in that case.
    _dual_no_sr = os.path.exists(_DUAL_SR_DISABLED)
    # Resolution bucket → (h_max, w_max, FSRCNNX variant, downsample_pre).
    # Buckets gate BOTH single and dual: source outside any bucket → the
    # entire chain passes through unmodified (rife.vpy returns the source
    # clip as-is). h tolerance ±48 px around canonical 480/720/1080/2160;
    # w covers the canonical 16:9 width + a tolerance that admits the
    # common cinema variants (DCI 2K = 2048 for 1080p, DCI 4K = 4096 for
    # 4K). Width gate is necessary because slot.dst sizes scale with W;
    # ultrawide / unusual ARs (e.g. 7680×1080 32:9) would otherwise pass
    # the height gate and balloon slot.dst on the dual path. Single mode
    # uses the same gate for consistency — a >4K or 32:9 source isn't
    # something either chain can process in real-time anyway.
    # Output scale = fsrcnnx_scale / downsample_pre, so the C++ SplitDual
    # filter (and apply_sr's target_w/h) sees the correct dst dim.
    #   480p → ×4 (FSRCNNX_x4)            out ~1920p
    #   720p → ×3 (FSRCNNX_x3)            out 4K
    #   1080p → ×2 (FSRCNNX_x2_16)        out 4K (baseline)
    #   2160p → 4K-DS path (x2 + ds=2)    out 4K (luma-only DS preserves
    #                                            chroma 4:4:4 at 1080p)
    # slot.dst budget for the worst case in each bucket (verified to hold
    # both the SR YUV output planes and the dense-pack INTERP-mult=4
    # overlay; assertion in mp_pipeline.task_dst_ranges catches anything
    # that would overflow at runtime):
    #   480p MAX (528×928)    slot ~31 MB,  interp-x4 ~12 MB ✓
    #   720p MAX (768×1328)   slot ~42 MB,  interp-x4 ~25 MB ✓
    #   1080p MAX (1128×2048) slot ~61 MB,  interp-x4 ~55 MB ✓ (tightest)
    #   4K-DS MAX (2208×4144) slot ~62 MB,  interp-x4 ~55 MB ✓
    _RES_BUCKETS = [
        (528,  928,  "FSRCNNX_x4_16-0-4-1", 1),   # 480p
        (768,  1328, "FSRCNNX_x3_16-0-4-1", 1),   # 720p
        (1128, 2048, "FSRCNNX_x2_16-0-4-1", 1),   # 1080p (+DCI 2K)
        (2208, 4144, "FSRCNNX_x2_16-0-4-1", 2),   # 4K-DS (+DCI 4K)
    ]
    _bucket_idx = None
    _bucket_variant = None
    _bucket_ds = None
    _bucket_out_scale = None
    for _i, (_h_max, _w_max, _v, _ds) in enumerate(_RES_BUCKETS):
        if h <= _h_max and w <= _w_max:
            _bucket_idx = _i
            _bucket_variant = _v
            _bucket_ds = _ds
            _bucket_out_scale = int(_v.split("_x")[1].split("_")[0]) // _ds
            break
    if _bucket_idx is None:
        sys.stderr.write(
            f"[rife.vpy] source {w}x{h} outside processing buckets "
            f"(max h=2208 / w=4144) — passing through unmodified\n")
        sys.stderr.flush()
    # Dual-machine accepts 10-bit YUV at 4:2:0 / 4:2:2 / 4:4:4. CC
    # kernel + cc_cache size chroma planes off subsampling_w/h; the
    # rgb_padded / rgb_interp path is subsampling-agnostic (always
    # full-luma-res 4:4:4 RGB).
    _dual_ok_sub = (clip.format.subsampling_w in (0, 1)
                    and clip.format.subsampling_h in (0, 1)
                    and clip.format.num_planes == 3
                    and clip.format.bits_per_sample == 10
                    and clip.format.sample_type == vs.INTEGER)
    _dual_variant = None
    _dual_ds = None
    _dual_out_scale = None
    # F8 OFF + 4K-DS bucket = inadmissible (RIFE-DS produces 1080p
    # interp frames; without SR they can't be Interleaved back into a
    # 4K real-frame stream). Drop dual in that case; single chain
    # handles 4K-DS path natively even with SR off (its inner apply_sr
    # picks up the off variant from /tmp/fsrcnnx_variant).
    _dual_no_sr_admissible = not (_dual_no_sr and _bucket_idx == 3)
    # F9 cycle's "OFF" maps to interp_mult=1 (no_interp): dual stays
    # active for CCSR/SR_SRC, INTERP+SR_INTERP are skipped. Therefore
    # /tmp/rife_disabled is NOT a dual-gating signal anymore — only the
    # single chain consults rife_disabled() (around line ~325).
    if (os.environ.get("DUAL_WORKER_HOST")
            and not os.path.exists(_DUAL_DISABLED_FLAG)
            and _dual_no_sr_admissible
            and _dual_ok_sub
            and fps <= 30
            and _bucket_idx is not None):
        _dual_variant = _bucket_variant
        _dual_ds = _bucket_ds
        _dual_out_scale = _bucket_out_scale
        # F8 OFF (no_sr) forces output dim = source dim. native_dispatcher's
        # output_scale also gets overridden to 1, but SplitDual's `scale=`
        # param is passed separately here, so we have to mirror the
        # override. Without this, SplitDual allocates a 2x output frame
        # while the worker writes a 1x payload → output is 4K-sized but
        # only the top-left 1080p region is valid.
        if _dual_no_sr:
            _dual_out_scale = 1
    _dual_eligible = _dual_variant is not None
    if _dual_eligible:
        try:
            # Dual-machine fixes RIFE 4.26 + the bucket-selected
            # FSRCNNX variant. The host CCSR + guest INTERP parallel
            # arrangement holds 60 fps output for source ≤ 30 fps
            # across all resolution buckets.
            from native_dispatcher import make_dispatcher
            import dgxspark_native_helper as _nhelper
            _native_so = (HERE / "dual_machine" / "native_filter" /
                           "build" / "libdgxspark_split_dual.so")
            core.std.LoadPlugin(str(_native_so))
            # DUAL_INTERP_MULT controls temporal multiplier in the
            # native SplitDual filter (2 = x2, 3 = x3, 4 = x4). F9
            # dual cycle in sr_keys.lua writes
            # /tmp/dual_machine_mult_override; we already mirrored
            # that into the env above.
            _dual_mult_for_modeb = int(os.environ.get(
                "DUAL_INTERP_MULT", "2"))
            if _dual_mult_for_modeb not in (1, 2, 3, 4):
                _dual_mult_for_modeb = 2
            _compute, _cleanup = make_dispatcher(
                clip, variant=_dual_variant,
                matrix_s=matrix_s, color_range=color_range,
                rife_model="4.26", chroma_kernel="bicubic",
                downsample_pre=_dual_ds,
                interp_mult=_dual_mult_for_modeb,
                no_sr=_dual_no_sr,
            )
            _cb_id = _nhelper.register_callback(_compute)
            clip = core.dgxspark_split_dual.SplitDual(
                clip, scale=_dual_out_scale, compute_id=_cb_id,
                interp_mult=_dual_mult_for_modeb)
            dual_active = True
            fsrcnnx_applied = True
            sys.stderr.write(
                f"[rife.vpy] dual-machine active src={clip.width}x{h} "
                f"variant={_dual_variant} downsample_pre={_dual_ds} "
                f"out_scale={_dual_out_scale}\n")
        except Exception as _be:
            sys.stderr.write(
                f"[rife.vpy] dual-machine setup failed: "
                f"{type(_be).__name__}: {_be} — falling back to "
                "single-machine\n")

    # Surface DUAL_INTERP_MULT so the worker (which reads it in
    # _run_interp) and AB scripts agree on the active mult. End-to-end
    # dual x3 / x4 is fully wired through queue_mgr, cc_cache,
    # SplitDual, guest_mp, and N-stage early delivery.
    _dual_mult = int(os.environ.get("DUAL_INTERP_MULT", "2"))
    if _dual_mult not in (1, 2, 3, 4):
        _dual_mult = 2

    # Publish the resolved dual state to /tmp so scripts/sr_keys.lua's
    # Shift+F9 toggle can decide direction from "what we're actually
    # running" rather than the disable-flag (which is stale on a
    # fallback path — flag absent but we couldn't connect, so we
    # ended up single anyway). Lua reads this on the NEXT press,
    # so a connect failure remains retriable from a single key.
    try:
        with open("/tmp/dual_machine_active", "w") as _f:
            _f.write("1" if dual_active else "0")
        with open("/tmp/dual_machine_mult", "w") as _f:
            _f.write(str(_dual_mult))
        # F8 dual SR state. When dual_active=1 + sr_active=0, the dual
        # session is running RIFE INTERP but no FSRCNNX SR. lua reads
        # this for the F8 OSD label (so the next press knows what to
        # cycle to).
        with open("/tmp/dual_machine_sr_active", "w") as _f:
            _f.write("0" if (dual_active and _dual_no_sr) else
                     ("1" if dual_active else "0"))
    except OSError:
        pass

    # dual_active short-circuits the rest of the chain — the native
    # dispatcher already produced the final 4K output. Don't re-run.
    if not dual_active:
        fsrcnnx_applied = False
    # ≤1080p single-machine path applies the 4:4:4 chroma trick:
    # _rife emits 4:4:4 at the input luma dim (chroma stays full-res
    # through RIFE instead of round-tripping through half-res 4:2:0).
    # apply_sr then runs with output_subsampling=(1,1) + GPU chroma
    # kernel so FSRCNNX's cuDNN fused path folds the 4:4:4 → 4:2:0
    # chroma resize into the same F.interpolate pass as the SR. When
    # input chroma already sits at the output's chroma resolution
    # (1080p source → 4K 4:2:0 chroma = 1080p) the resize becomes
    # identity. Net: one chroma op total (the original YUV→RGB
    # upsample) for interp frames, vs three in a naive chain
    # (upsample → downsample → upsample).
    _SR_OUT_SUB = (1, 1)
    # KrigBilateral: luma-guided kriging chroma resize, +6-7 dB U/V over
    # Bicubic. For the common 1080p→4K 4:2:0 path with the 4:4:4 trick
    # this is identity (input chroma = output chroma dim), so krig does
    # not run; for 720p sources where 4:4:4-trick chroma actually has
    # to resize (720p→2160p in 4:2:0) we get the quality win for free.
    # Lanczos is overkill on CPU; krig stays on the GPU.
    # KRIG_CHROMA=0 reverts to the pre-krig (Bicubic) baseline for
    # benchmarking; default = krig on.
    _SR_CHROMA_KERNEL = ("KrigBilateral"
                         if os.environ.get("KRIG_CHROMA", "1") != "0"
                         else "Bicubic")
    # Bucket-gated single chain: same (h_max, w_max) classification as
    # dual, so a >4K / ultrawide source skips processing on BOTH paths.
    # bucket 0/1 (480p/720p) and bucket 2 (1080p) → _rife direct at src
    # dim + post-fsrcnnx. Bucket 3 (4K-DS) → DS-to-half + RIFE +
    # SR-back-to-src + Interleave. _bucket_idx is None means passthrough.
    if (not dual_active and not rife_disabled() and fps <= 30
            and _bucket_idx is not None):
        if _bucket_idx <= 1:  # 480p / 720p
            clip = _rife(clip, rife_model_pref or "4.26",
                         factor_num=rife_factor, target_subsample="444")
        elif _bucket_idx == 2:  # 1080p
            _m = rife_model_pref or ("4.26" if heavy_fps else "4.6")
            clip = _rife(clip, _m, factor_num=rife_factor,
                         target_subsample="444")
        else:  # bucket 3 — 4K-DS (real frame passthrough + interp DS+SR).
               # Full src-dim RIFE is too heavy on GB10 (4K: ~25 fps
               # pipelined; even scale=0.5 half-flow can't sustain 48 fps
               # at ~38 fps). So we downsample to 1080p, run RIFE there,
               # take ONLY the interp frames, SR them back to src dim,
               # and Interleave with the original src. Real frames stay
               # bit-exact; only the synthesized in-between frames pay
               # the downsample+SR cost.
               #
               # Instead of bicubic-resizing the whole 4:2:0 frame
               # (which downsamples chroma 2× from its native 1080p
               # down to 540p, losing detail), we ask zimg to land at
               # 1080p 4:4:4. zimg sees: luma src→1080p, chroma
               # already at 1080p for 4K 4:2:0 native (or resized
               # otherwise) → no avoidable chroma resize. Result is
               # the source's full-resolution chroma preserved
               # through RIFE.
               #
               # Target = src dim (not 4K env default), so non-
               # standard resolutions (1440p, 1620p, …) work the same
               # way as 4K — the interp pipeline always returns to
               # src dim. Also: interp_4k.format is coerced to match
               # src_4k.format so the Interleave keeps src chroma
               # subsampling (4:2:0 → 4:2:0, 4:2:2 → 4:2:2, 4:4:4 →
               # 4:4:4); Blackwell can hwdec 4:2:2 directly so don't
               # discard it downstream.
               #
               # On the half-the-frames budget, ≤24 fps sources can afford
               # the heavier interp chain (RIFE 4.26 + 16-layer FSRCNNX);
               # 25+ fps falls back to the lighter (4.6 + 8-layer) variant
               # to stay under the tighter 60 fps output budget.
            src_4k = clip
            # Downsample by exactly 2 (uniform across 4K / 1440p / 1620p
            # / any non-standard res — same logic as 4K). The +1 before
            # //2 rounds up so odd src dims still yield even-aligned
            # targets after the & ~1 mask (4:2:0 / 4:2:2 chroma needs
            # even dims).
            target_h = ((clip.height + 1) // 2) & ~1
            target_w = ((clip.width + 1) // 2) & ~1
            down = core.resize.Bicubic(clip,
                                         width=target_w, height=target_h,
                                         format=vs.YUV444P10)
            rife_model  = "4.26" if heavy_fps else "4.6"
            sr_family_4k = "16-layer" if heavy_fps else "8-layer"
            # 4K-like mode keeps factor=2 — the src_4k/interp_4k
            # Interleave only handles 1:1 patterns cleanly. Higher
            # factors would need bespoke frame-mapping logic.
            rife_low = _rife(down, rife_model, factor_num=2)
            rife_interp_1080p = rife_low.std.SelectEvery(2, [1])
            # SR target = src dim. apply_sr's min_ratio gate refuses SR
            # when target/in ≤ 1.2 (e.g. src very close to 1080p); when
            # that happens interp_sr stays at 1080p and we zimg-resize
            # it up to src dim before interleaving.
            interp_sr = apply_sr(rife_interp_1080p, family=sr_family_4k,
                                 target_w=src_4k.width,
                                 target_h=src_4k.height)
            if (interp_sr.width != src_4k.width
                    or interp_sr.height != src_4k.height):
                interp_sr = core.resize.Bicubic(interp_sr,
                                                  width=src_4k.width,
                                                  height=src_4k.height)
            # Match interp format to src so Interleave succeeds, keeping
            # src chroma subsampling end-to-end.
            if interp_sr.format.id != src_4k.format.id:
                interp_4k = core.resize.Bicubic(interp_sr,
                                                  format=src_4k.format.id)
            else:
                interp_4k = interp_sr
            clip = core.std.Interleave([src_4k, interp_4k])
            fsrcnnx_applied = True

    # bucket 0/1 (480p/720p) always use 16-layer (select_variant refuses
    # 8-layer at ratio ≥ 2.5); bucket 2 (1080p) picks 16-layer at cinema
    # rates, 8-layer at 25+ fps. Out-of-bucket sources skipped (already
    # in passthrough mode — apply_sr would try to upscale a non-standard
    # res and either pick a wrong variant or hit min_ratio refusal).
    if not fsrcnnx_applied and _bucket_idx is not None:
        if _bucket_idx <= 1:
            family = "16-layer"
        else:
            family = "16-layer" if heavy_fps else "8-layer"
        _sr_kw = {}
        if (clip.format is not None
                and clip.format.subsampling_w == 0
                and clip.format.subsampling_h == 0):
            _sr_kw["chroma_kernel"] = _SR_CHROMA_KERNEL
            _sr_kw["output_subsampling"] = _SR_OUT_SUB
        clip = apply_sr(clip, family=family, **_sr_kw)
except Exception as _e:
    fmt_name = getattr(video_in.format, "name", "?")
    sys.stderr.write(
        f"[rife.vpy] chain setup failed for source format {fmt_name}: "
        f"{type(_e).__name__}: {_e} — passing through unmodified\n")
    clip = video_in

# Engine warmup happens inside vs_gpu_helpers.rife_yuv and
# fsrcnnx_yuv._load_cudnn_runner — both feed synthetic zeros through
# their cuDNN runners at filter-load time so graph compilation
# (~50-100 ms each) is paid before mpv's first inference request.
# The dual path runs its own engine warmup inside make_dispatcher.
#
# We deliberately do NOT call clip.get_frame() here. mpv's
# vf_vapoursynth returns uninitialised black source frames during
# script load (mpv's decoder hasn't produced video yet); a frame-
# based prewarm would feed those into rife_yuv's rgb_cache, VS would
# cache the (black) output, and mpv would see the first ~5 frames
# as pure black.

clip.set_output()
