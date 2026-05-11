# dgxspark-jellyfin-mpv-rife

[简体中文](./README.zh-CN.md)

Reproducible installer for a **Jellyfin client + RIFE realtime frame
interpolation + FSRCNNX luma upscaling** stack on **NVIDIA DGX Spark**
(GB10, ARM64, CUDA 13, Ubuntu 24.04, GNOME Wayland).

## Stack

- **mpv 0.41** built from source — vapoursynth + vulkan + wayland + x11
  + lua. The conda-forge aarch64 mpv is a headless library build with
  no display backends, so the installer builds its own.
- **vsrife + TensorRT** RIFE picked per source resolution: 4.26 for
  ≤720p, 4.6 for 720p<h≤1080p; off above 1080p (GB10 doesn't have the
  headroom for 4K RIFE) and off above 30fps (already smooth). Mixed
  precision (fp16 weights, fp32 accumulators); pure fp16 flickers on
  fast motion. The installer patches vsrife to pin
  `enabled_precisions={fp16, fp32}`.
- **GPU YUV↔RGB color conversion** (`vs_gpu_helpers.rife_yuv`) — the
  CPU `core.resize.Bicubic(format=vs.RGBH)` round-trip costs ~30 ms /
  frame at 4K and was the actual bottleneck on the 20-core Grace CPU.
  We do the matrix multiply + chroma resample on GPU instead. With
  Grace+Blackwell unified memory there's no PCIe upload cost.
- **FSRCNNX cuDNN super-resolution** — installed from the upstream
  [`Cryspia/fsrcnnx-cudnn`](https://github.com/Cryspia/fsrcnnx-cudnn)
  release bundle (variants x2_8 / x2_16 / x3_16 / x4_16). The chain
  picks one automatically based on source ↔ target ratio; F8 lets you
  cycle through them at runtime. See that repo's README for the
  per-variant benchmarks and family / scale rules.
- **jellyfin-mpv-shim** — Python client that drives mpv via IPC. The
  installer patches it for mpv 0.41's removed `osc=False` option
  (translated to `script-opts=osc-visibility=...` driven by shim's
  `enable_osc` field).

## Requirements

- DGX Spark (or any NVIDIA aarch64 system with CUDA 13). x86_64 is
  **not** supported.
- Ubuntu 24.04 (other distros likely work but untested).
- GNOME Wayland session for the AppIndicator tray icon path.
- `sudo` for the apt-install step. ~5 GB free disk.

## Usage

```bash
./install.sh install                 # full clean install
./install.sh install --no-mirrors    # outside China — skip USTC mirrors
./install.sh install --no-danmaku    # skip the bullet-chat plugin
./install.sh install --rebuild-trt   # wipe + rebuild TRT engine cache
./install.sh install --set-default-video   # promote mpv-conda to GNOME default
./install.sh status                  # show what's currently installed
./install.sh uninstall               # remove (keeps apt + miniforge)
```

`./install.sh install --help` lists every flag. Log out / back in after
a fresh install so GNOME picks up the new launchers + autostart entry.
First playback at a new resolution JIT-compiles a TensorRT engine
(~30–60 s once, then cached at
`~/miniforge3/envs/vsmpv/lib/python3.12/site-packages/vsrife/models/`).

## Default config

| Source | Frame rate | RIFE | FSRCNNX (auto) |
|---|---|---|---|
| ≤ 720p | ≤ 30 fps | 4.26 @ scale=1.0 | x3_16 / x4_16 (ratio-dependent) |
| 720p < h ≤ 1080p | ≤ 30 fps | 4.6 @ scale=1.0 | x2_8 |
| 1080p < h ≤ 2160p | ≤ 30 fps | downsample → 4.6 → SR | x2_8 (recovers 4K) |
| any | > 30 fps | off | runs if ratio merits |

The 4K path downsamples to 1080p, runs RIFE there, and lets FSRCNNX
upsample back to 4K — the only RIFE config GB10 can sustain at 4K.
Benched on two native 4K clips (240 frames each, pipelined via
`get_frame_async`):

| 4K config | sample-01 fps | sample-02 fps | 48 fps budget |
|---|---|---|---|
| downsample → RIFE 4.6 → SR (current) | **65.7** | **68.6** | ✅ ~28% headroom |
| RIFE 4.6 @ scale=0.5 (half-flow) | 41.1 | 41.2 | ❌ −17% |
| RIFE 4.6 @ scale=1.0 (full-flow) | 27.1 | 27.2 | ❌ −77% |

Y PSNR:

| comparison | sample-01 | sample-02 |
|---|---|---|
| current path real frames vs original 4K | 45.0 dB (39.0–47.4) | 48.2 dB (47.0–48.8) |
| current path interp frames vs full-flow | 38.5 dB | 34.8 dB |
| half-flow interp frames vs full-flow | 39.1 dB | 35.1 dB |

The current 4K config trades ~3–6 dB of real-frame Y PSNR vs original
(visible as slightly softer detail on high-frequency textures) for the
headroom to actually run. Interp-frame quality is within 0.3–0.6 dB of
the half-flow path — the flow happens at 1080p in both cases, so the
4K-vs-1080p warp difference is dominated by RIFE's inherent temporal
uncertainty rather than spatial flow accuracy.

Display target defaults to 4K. If you have a smaller screen, export
`FSRCNNX_TARGET_W` / `FSRCNNX_TARGET_H` before launching mpv.

## Keybindings

- **F8** — cycle FSRCNNX variant: `16x4 → 16x3 → 16x2 → 8x2 → OFF →
  loop`. Default starts at the auto-picked variant for the current
  source. Each press triggers a vapoursynth filter reload (~1–3 s
  freeze — no runtime parameter switch in the cuDNN runner). The
  cycle resets to auto on next file load.
- **F9** — toggle RIFE on/off independently. Persists across files.
- All mpv defaults intact (`i` for stats, `s` for screenshot, etc.).

## Danmaku (bullet chat)

Provided by [**Cryspia/mpv-dandanplay-danmaku**](https://github.com/Cryspia/mpv-dandanplay-danmaku).
The installer clones it to `~/src/mpv-dandanplay-danmaku/` and runs
its `install.py`. See that repo's README for keybinds, settings,
config reference, and troubleshooting.

## What lives where after install

| Path | Purpose |
|------|---------|
| `~/miniforge3/envs/vsmpv/` | Conda env: python, mpv, vapoursynth, vsrife, shim, tensorrt |
| `~/.config/mpv/{mpv,input}.conf, rife.vpy, vs_gpu_helpers.py, sr_keys_helper.py` | Our config + helpers |
| `~/.config/mpv/scripts/sr_keys.lua` | F8 / F9 keybind logic |
| `~/.config/mpv/scripts/fsrcnnx-cudnn/` | Upstream fsrcnnx-cudnn bundle (Python pkg + .npz weights) |
| `~/.config/mpv/scripts/dandanplay/` | Danmaku script bundle |
| `~/.config/jellyfin-mpv-shim/conf.json` | Shim's own config (server creds, etc.) |
| `~/.config/jellyfin-mpv-shim/{mpv,input}.conf, rife.vpy, …` | Symlinks to `~/.config/mpv/` |
| `~/.local/bin/{mpv-conda,jellyfin-mpv-shim}` | Wrappers (PYTHONHOME / GI_TYPELIB_PATH) |
| `~/.local/share/applications/*.desktop` | App-launcher entries |
| `~/.config/autostart/jellyfin-mpv-shim.desktop` | Auto-start shim on login |
| `~/src/mpv/`, `~/src/mpv-dandanplay-danmaku/` | Source checkouts (safe to delete; re-cloned on next install) |

## Project layout

```
dgxspark-jellyfin-mpv-rife/
├── install.sh           # installer / status / uninstall
├── rife.vpy*            # generated by install.sh
├── vs_gpu_helpers.py    # rife_yuv: GPU YUV↔RGB + vsrife wrapper
├── sr_keys_helper.py    # F8/F9 side-channel + apply_fsrcnnx
└── icons/               # app icons for shim + mpv-conda launchers
```

`*` `rife.vpy` and the lua scripts are emitted by `install.sh` into
`~/.config/mpv/`; they don't live in the repo.

## Notes worth remembering

- **fp16 mixed-precision** is critical for vsrife — pure fp16 overflows
  flow-vector accumulators on fast motion (visible flicker).
- **PYTHONHOME** is set by the `mpv-conda` wrapper. Without it, mpv
  launched outside conda activation can't find the embedded Python's
  stdlib → vapoursynth filter init aborts.
- **`gpu-context=waylandvk`** is ~1 ms lower latency than `x11vk` but
  loses GNOME-drawn window decorations (mpv 0.41 has no libdecor).
- **`hidpi-window-scale=yes`** is what makes FSRCNNX trigger on a 4K
  HiDPI display — without it mpv renders at logical 1080p and the
  ratio gate stays at 1.0.
- **F8 / F9 reload cost** — both cycle by re-creating the vapoursynth
  filter. The cuDNN runner builds at filter creation, so the press is
  followed by a 1–3 s freeze. Acceptable for occasional manual
  override.
