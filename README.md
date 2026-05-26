# dgxspark-jellyfin-mpv-rife

[简体中文](./README.zh-CN.md)

Reproducible installer for a **Jellyfin client + RIFE realtime frame
interpolation + FSRCNNX luma upscaling** stack on **NVIDIA DGX Spark**
(GB10, ARM64, CUDA 13, Ubuntu 24.04, GNOME Wayland).

Optional **dual-machine mode** (experimental) splits the chain across
two Sparks over a 200 G RoCE link for roughly 2× the steady-state fps.

## Usage

```bash
./install.sh install                 # single-machine, full clean install
./install.sh install --no-mirrors    # outside China — skip USTC mirrors
./install.sh install --no-danmaku    # skip the bullet-chat plugin
./install.sh install --dual-host     # primary box for dual mode
./install.sh install --dual-secondary # worker box for dual mode (worker.py + tray only)
./install.sh status                  # show what's currently installed
./install.sh uninstall               # remove (keeps apt + miniforge)
```

`./install.sh install --help` lists every flag. Log out / back in after
a fresh install so GNOME picks up the new launchers + autostart entry.

## Stack (single-machine)

- **mpv 0.41** built from source — vapoursynth + vulkan + wayland + x11
  + lua. The conda-forge aarch64 mpv is a headless library build with
  no display backends, so the installer builds its own.
- **vsrife + TensorRT** RIFE picked per source resolution **and** fps,
  with the budget split engineered so each combination just fits the
  GB10's per-frame envelope. Full table in [Default config](#default-config).
- **GPU YUV↔RGB color conversion** (`vs_gpu_helpers.rife_yuv`) — matrix
  multiply + chroma resample on GPU instead of zimg's CPU round-trip,
  which costs ~30 ms / frame at 4K on the Grace CPU. Supports YUV
  4:2:0 / 4:2:2 / 4:4:4 at 8 / 10 / 12 / 16 bit, BT.709 / 601 / 2020 NCL,
  limited or full range.
- **FSRCNNX cuDNN super-resolution** — installed from the upstream
  [`Cryspia/fsrcnnx-cudnn`](https://github.com/Cryspia/fsrcnnx-cudnn)
  release bundle (variants x2_8 / x2_16 / x3_16 / x4_16). The chain
  picks one automatically based on source ↔ target ratio.
- **KrigBilateral chroma kriging** — luma-guided chroma upsampling at
  every chroma-resize step. Internal CUDA kernel in `fsrcnnx-cudnn`
  plus mpv display GLSL ([igv's port](https://gist.github.com/igv/a015fc885d5c22e6891820ad89555637)).
- **jellyfin-mpv-shim** — Python client that drives mpv via IPC.
- **Danmaku** (bullet chat) — [Cryspia/mpv-dandanplay-danmaku](https://github.com/Cryspia/mpv-dandanplay-danmaku).

## Default config

| Source | Frame rate | RIFE | FSRCNNX (auto) |
|---|---|---|---|
| ≤ 720p | ≤ 30 fps | 4.26 @ scale=1.0 | x3_16 / x4_16 (ratio-dependent) |
| 720p < h ≤ 1080p | < 25 fps (cinema) | 4.26 @ scale=1.0 | x2_16 |
| 720p < h ≤ 1080p | 25–30 fps | 4.6 @ scale=1.0 | x2_8 |
| 1080p < h ≤ 2160p | < 25 fps (cinema) | mixed mode, 4.26 interp | x2_16 on interp |
| 1080p < h ≤ 2160p | 25–30 fps | mixed mode, 4.6 interp | x2_8 on interp |
| any | > 30 fps | off | runs if ratio merits |

The 25 fps threshold is the budget split: ≤24 fps source × 2 → 48 fps
output → 20.8 ms/frame, which the heavy chain (RIFE 4.26 + 16-layer
FSRCNNX) just fits. At 25+ fps the output budget drops to 16.7 ms, which
the heavy chain misses — so 25/29.97/30 fps content uses the lighter
(4.6 + 8-layer) variant.

For 4K source the chain runs **mixed mode**: real frames pass through at
original 4K (bit-exact), only the synthesized in-between frames take
the downsample → RIFE → SR upscale path. Per-frame budget roughly doubles,
fitting the heavier 4.26 + 16-layer family on interp frames.

Display target defaults to 4K. Override with `FSRCNNX_TARGET_W` /
`FSRCNNX_TARGET_H` before launching mpv for smaller screens.

## Dual-machine mode (experimental)

Two DGX Spark boxes over a 200 G RoCE link split CCSR / INTERP /
SR_INTERP across both GB10 GPUs. Roughly 1.95× sustained throughput vs
single-machine, with byte-identical real frames and visually-identical
interpolated frames.

Install:
```bash
# Primary box (has mpv + shim + dual host service)
./install.sh install --dual-host

# Worker box (worker.py + tray quit app only, no mpv main)
./install.sh install --dual-secondary
```

Both runs prompt for cluster networking (RDMA device, port, self IP,
peer IP); defaults come from `ip link` / `ip a`. Pre-set the env vars
listed by `--help` to script-install without prompts.

In playback: **Shift+F9** toggles dual offload on/off. Connection
failure falls back silently to single-machine mode (press again to
retry).

→ Full design + per-component breakdown: [`dual_machine/README.md`](./dual_machine/README.md).

## Performance and color accuracy

→ [`docs/performance.md`](./docs/performance.md) — fps, GPU util,
per-task timing.
→ [`docs/color-accuracy.md`](./docs/color-accuracy.md) — PSNR vs single
reference, byte-identical determinism guarantees.

## Benchmarks

→ [`bench/README.md`](./bench/README.md) — color / fps / per-task
timing scripts. Test clips synthesized via `bench/gen_clips.sh` (no
local-path dependencies).

## Keybindings

- **F8** — cycle FSRCNNX variant (`16x4 → 16x3 → 16x2 → 8x2 → OFF →
  loop`); each press triggers a vapoursynth filter reload (1–3 s
  freeze). Cycle resets to auto on next file load.
- **F9** — single: toggle RIFE on/off. Dual: cycle interp multiplier
  (4 → 3 → 2 → 1 → off).
- **Shift+F8** — toggle the KrigBilateral display-stage chroma GLSL.
  Instant (no vf reload); persists across files.
- **Shift+F9** — toggle dual-machine offload (no-op without
  `--dual-host` install). Falls back silently on connection failure.
- All mpv defaults intact (`i` for stats, `s` for screenshot, etc.).

## What lives where after install

| Path | Purpose |
|---|---|
| `~/miniforge3/envs/vsmpv/` | Conda env: python, mpv, vapoursynth, vsrife, shim, tensorrt |
| `~/.config/mpv/{mpv,input}.conf, rife.vpy, vs_gpu_helpers.py, sr_keys_helper.py` | Single-machine config + helpers |
| `~/.config/mpv/scripts/{sr_keys,warmup}.lua` | F8 / F9 / Shift+F8 / Shift+F9 keybind logic + post-load warmup |
| `~/.config/mpv/shaders/KrigBilateral.glsl` | Display-stage chroma kriging shader |
| `~/.config/mpv/fsrcnnx-cudnn/` | Upstream fsrcnnx-cudnn bundle (includes `chroma_krig` CUDA kernel) |
| `~/.config/mpv/scripts/dandanplay/` | Danmaku script bundle |
| `~/.config/jellyfin-mpv-shim/conf.json` | Shim's own config (server creds, etc.) |
| `~/.config/dgxspark-mpv/dual.conf` | Dual-machine cluster config (host/worker IPs, RDMA dev) — only present after `--dual-host` / `--dual-secondary` |
| `~/.local/bin/{mpv-conda,jellyfin-mpv-shim}` | Wrappers (PYTHONHOME / GI_TYPELIB_PATH) |
| `~/.local/share/applications/*.desktop` | App-launcher entries |
| `~/.config/autostart/jellyfin-mpv-shim.desktop` | Auto-start shim on login |
| `~/src/{mpv,mpv-dandanplay-danmaku}/` | Source checkouts (safe to delete; re-cloned on next install) |

## Requirements

- DGX Spark (or any NVIDIA aarch64 system with CUDA 13). x86_64 is
  **not** supported.
- Ubuntu 24.04 (other distros likely work but untested).
- GNOME Wayland session for the AppIndicator tray icon path.
- `sudo` for the apt-install step. ~5 GB free disk per box.
- For dual mode: two boxes + 200 G RoCE link with static IPs on the
  RDMA-capable interface.

## Project layout

```
dgxspark-jellyfin-mpv-rife/
├── install.sh           # installer / status / uninstall (single + dual variants)
├── vs_gpu_helpers.py    # rife_yuv: GPU YUV↔RGB + vsrife wrapper
├── sr_keys_helper.py    # F8/F9 side-channel + apply_fsrcnnx
├── scripts/             # sr_keys.lua + warmup.lua mpv plugins
├── shaders/             # vendored mpv GLSL shaders (KrigBilateral)
├── dual_machine/        # dual-machine offload — see its README
├── docs/                # performance + color-accuracy data tables
├── bench/               # color / fps / timing bench scripts
└── icons/               # app icons for shim + mpv-conda launchers
```

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

## Help / feedback

`./install.sh status` first, then file at the project's GitHub if
something doesn't make sense.
