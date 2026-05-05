# dgxspark-jellyfin-mpv-rife

[简体中文](./README.zh-CN.md)

Reproducible installer for a **Jellyfin client + RIFE realtime frame
interpolation + FSRCNNX luma upscaling** stack on **NVIDIA DGX Spark**
(GB10, ARM64, CUDA 13, Ubuntu 24.04, GNOME Wayland).

The whole pipeline runs in a single Miniforge conda environment:

- **mpv** — built from source (mpv 0.41) with vapoursynth + vulkan +
  wayland + x11 + lua all enabled. The conda-forge aarch64 mpv is a
  headless library build with no display backends, so the installer
  builds its own.
- **vsrife + TensorRT** — RIFE 4.26 model running on TRT mixed precision
  (fp16 weights, fp32 accumulators). The installer patches vsrife to use
  `enabled_precisions={fp16, fp32}` instead of `use_explicit_typing=True`,
  which would otherwise cause flow-vector overflow → flicker on fast
  motion.
- **FSRCNNX** glsl shader — 2x luma upscale, only activates when render
  target is >1.3x source (e.g. 1080p source on a 4K HiDPI monitor).
- **jellyfin-mpv-shim** — Python client that connects to a Jellyfin
  server and drives mpv via IPC. Installer patches it for mpv 0.41
  compatibility (`--osc=no` cmdline removal).

## Requirements

- DGX Spark (or any NVIDIA aarch64 system with CUDA 13). x86_64 is **not**
  supported (some choices like building mpv from source are aarch64-specific
  workarounds for missing conda-forge packages).
- Ubuntu 24.04 (other distros likely work but untested).
- GNOME Wayland session for the AppIndicator tray icon path.
- `sudo` access for the apt-install step.
- ~5 GB of free disk for the conda env + TRT engines.

## Usage

```bash
# Full install on a clean machine (USTC mirrors + danmaku ON by default)
./install.sh install

# Outside China — skip the USTC conda-forge / PyPI mirror config
./install.sh install --no-mirrors

# Skip the danmaku (bullet-chat) plugin step
./install.sh install --no-danmaku

# Both
./install.sh install --no-mirrors --no-danmaku

# Show what's currently installed
./install.sh status

# Remove everything we installed (keeps apt packages + miniforge for fast re-install)
./install.sh uninstall
```

Run `./install.sh install --help` for the canonical flag list.

After `install`, log out and back in so GNOME picks up the new launchers,
autostart entry, and icons. First playback will JIT-compile a TensorRT
RIFE engine (~30-60s once); after that, startup is fast and the engine
is cached at `~/miniforge3/envs/vsmpv/lib/python3.12/site-packages/vsrife/models/`.

## Default config

RIFE and FSRCNNX both auto-apply per source resolution + frame rate.
Empirical testing on the GB10 showed the heavy 4.26 model can't sustain
2× interpolation at 1080p without contending with the libplacebo
compositor, so the RIFE band depends on input height:

| Source | Frame rate | RIFE | FSRCNNX |
|---|---|---|---|
| ≤ 720p | ≤ 30 fps | **4.26** (heavy, scale=1.0) | on |
| > 720p, ≤ 1080p | ≤ 30 fps | **4.6** (light, scale=1.0) | on |
| > 1080p | any | off (would peg the GB10) | off |
| any | > 30 fps | off (already smooth) | on |

Other relevant settings:

- vo=gpu-next, gpu-context=waylandvk (no decorations on GNOME, but ~1ms
  less frame-presentation latency than Xwayland)
- hidpi-window-scale=yes (so FSRCNNX activates on 4K HiDPI)
- Both RIFE models use TRT mixed precision (fp16 weights, fp32
  accumulators) per a vsrife patch the installer applies; pure fp16
  flickers on fast motion because flow-vector accumulators overflow.

Tuning knobs are inline in `~/.config/mpv/rife.vpy`,
`~/.config/mpv/rife-light.vpy`, and `~/.config/mpv/mpv.conf`.
Manual override mid-playback: F8 toggles FSRCNNX, F9 cycles the active
RIFE config (4.26 → 4.6 → off → loop) regardless of which band the
profile-cond logic chose.

## Keybindings

- **F8** — toggle FSRCNNX shader
- **F9** — cycle RIFE: 4.26 (default) → 4.6 (light) → off
- **F10** — toggle danmaku visibility (also: click the 弹 icon top-right)
- **Shift+F10** — danmaku settings panel (opacity / font size / speed /
  density / area / render mode / anti-overlap / source filter /
  per-episode time offset / dedup / 繁简转换 / etc.)
- **Ctrl+F10** — danmaku manual fuzzy search (text input → anime picker
  → episode picker)
- **i** / **Shift+I** — mpv default stats overlay
- **Ctrl+S** — mpv default screenshot
- All other mpv defaults intact

## Danmaku (bullet chat)

Provided by [**Cryspia/mpv-dandanplay-danmaku**](https://github.com/Cryspia/mpv-dandanplay-danmaku),
a standalone mpv script for dandanplay-driven bullet-chat overlay.
This installer clones the project to `~/src/mpv-dandanplay-danmaku/`
and invokes its `install.py` during step 8b. See that repo's
[README](https://github.com/Cryspia/mpv-dandanplay-danmaku) for the
full feature list, configuration reference, and troubleshooting.

Quick summary of what you get:

- **Auto-match** by Jellyfin metadata (when launched via shim) or
  filename, with smart-alias fallback for shows whose library name
  differs from dandanplay's.
- **Manual fuzzy search** (Ctrl+F10) — two-stage picker: anime → episode.
- **In-player settings panel** (Shift+F10) covering opacity, font size,
  speed, density, area, render mode, anti-overlap, source filter
  (B站/巴哈/弹弹/其他), traditional↔simplified, dedup, per-episode
  time offset.
- **CJK-aware lane allocation** + banded scroll layout — no overlapping
  text on a row, comments cluster naturally near the top.
- **Default uses upstream's CORS proxy**; switching to your own
  dandanplay AppId is a `cp + edit` away. Strongly recommended; see
  the danmaku project's README for details.

## What lives where after install

| Path | Purpose |
|------|---------|
| `~/miniforge3/envs/vsmpv/` | Conda env: python, mpv, vapoursynth, vsrife, shim, tensorrt |
| `~/.config/mpv/{mpv,input}.conf, rife.vpy, shaders/` | User mpv config (also used by shim via symlinks) |
| `~/.config/jellyfin-mpv-shim/conf.json` | Shim's own config (server creds, etc.) |
| `~/.config/jellyfin-mpv-shim/{mpv.conf,input.conf,rife.vpy,shaders,scripts}` | Symlinks to `~/.config/mpv/` |
| `~/.local/bin/{mpv-conda,jellyfin-mpv-shim}` | Wrappers that set PYTHONHOME / GI_TYPELIB_PATH |
| `~/.local/share/applications/*.desktop` | App-launcher entries |
| `~/.config/autostart/jellyfin-mpv-shim.desktop` | Auto-start shim on login |
| `~/.local/share/icons/hicolor/*/apps/{mpv-conda,jellyfin-mpv-shim}.png` | App icons |
| `~/.config/mpv/scripts/dandanplay/` | Danmaku script bundle (installed by [Cryspia/mpv-dandanplay-danmaku](https://github.com/Cryspia/mpv-dandanplay-danmaku)) |
| `~/.config/mpv/danmaku-{config,credentials,settings}.json` | Danmaku proxy / credentials / panel settings |
| `~/.cache/mpv-danmaku/{matches,offsets,aliases}.json` | Danmaku match cache, time offsets, smart-match aliases |
| `~/src/mpv/` | mpv source checkout for re-builds |
| `~/src/mpv-dandanplay-danmaku/` | Danmaku project clone (re-pulled on each install) |

## Project layout

```
dgxspark-jellyfin-mpv-rife/
├── README.md
├── install.sh                  # installer / status / uninstall entry
├── shaders/
│   └── FSRCNNX_x2_8-0-4-1.glsl
└── icons/
    ├── shim/{16,32,48,64,128,256}.png
    └── mpv/{16,32,64,128}.png + scalable.svg
```

The danmaku plugin formerly lived inline (`danmaku/`) but has graduated
into its own repo at [Cryspia/mpv-dandanplay-danmaku](https://github.com/Cryspia/mpv-dandanplay-danmaku).
This installer fetches it (pinned to the `main` branch — bump
`DANMAKU_REF` in `install.sh` to a tag for reproducible builds) and
runs its installer; you don't need to clone the danmaku repo manually.

Cloning this dgxspark repo onto a fresh DGX Spark and running
`./install.sh install` reproduces the entire setup.

## Notes worth remembering

- **fp16 mixed-precision is critical.** Pure fp16 (which vsrife uses by
  default with `use_explicit_typing=True`) flickers on fast motion
  because flow-vector accumulators overflow.
- **PYTHONHOME is critical.** Without it, mpv launched outside conda
  activation can't find the embedded Python's stdlib and vapoursynth
  filter init aborts → playback EOFs immediately with audio only.
- **gpu-context=waylandvk** has lower latency than `x11vk` but loses
  GNOME-drawn window decorations (mpv 0.41 has no libdecor support).
- **hidpi-window-scale=yes** is what makes FSRCNNX activate on a 4K
  HiDPI display — without it, mpv renders at logical 1080p and the
  shader's `//!WHEN OUTPUT/LUMA > 1.3` gate stays false.
- **NVENC streaming (e.g. Sunshine/Moonlight) costs ~20% GPU.** If you
  see drops with this config, the streaming pipeline is most likely
  the culprit, not RIFE itself. Local 4K display has plenty of headroom
  for RIFE 4.26 + scale=1.0 + FSRCNNX with zero drops.
