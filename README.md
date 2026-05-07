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
- **vsrife + TensorRT** — RIFE picked per source resolution: 4.26
  (heavy, best quality) for ≤720p, 4.6 (light) for 720p<h≤1080p,
  4.6 @ scale=0.5 (half-flow) for 1080p<h≤2160p, off above that.
  All run TRT mixed precision (fp16 weights, fp32 accumulators). The
  installer patches vsrife to use `enabled_precisions={fp16, fp32}`
  instead of `use_explicit_typing=True`, which would otherwise cause
  flow-vector overflow → flicker on fast motion. See
  [Default config](#default-config) for the per-band rules.
- **FSRCNNX** glsl shader — 2x luma upscale. Loaded by default for
  ≤1080p sources; explicitly cleared for >1080p so mpv doesn't pile
  shader work onto already-large frames. The shader's own
  `//!WHEN OUTPUT/LUMA > 1.3` gate self-disables when source ≈ display.
- **jellyfin-mpv-shim** — Python client that connects to a Jellyfin
  server and drives mpv via IPC. Installer patches it for mpv 0.41
  compatibility (the legacy `osc=False` mpv option is gone — translated
  to `script-opts=osc-visibility=auto|never` driven by shim's own
  `enable_osc` config field).

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

# Wipe + rebuild the TRT engine cache (after a driver/CUDA/TRT
# upgrade caused stale cached engines to produce broken video)
./install.sh install --rebuild-trt

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
| > 1080p, ≤ 2160p | ≤ 30 fps | **4.6** at **scale=0.5** (half-flow, ~1080p compute) | off |
| > 2160p | any | off (above 4K is unreasonable for realtime) | off |
| any | > 30 fps | off (already smooth) | on (≤1080p) / off (>1080p) |

Half-flow (`scale=0.5`) for the 4K band runs flow estimation at half
resolution internally — total compute drops to roughly the same as
full-resolution 1080p, so the GB10 keeps up. Quality is noticeably
softer on fast motion than full-res 4.6 (less detail in the flow
field) but RIFE still does *something*, which beats a stuttering 4K
source with no interpolation at all.

Other relevant settings:

- vo=gpu-next, gpu-context=waylandvk (no decorations on GNOME, but ~1ms
  less frame-presentation latency than Xwayland)
- hidpi-window-scale=yes (so FSRCNNX activates on 4K HiDPI)
- Both RIFE models use TRT mixed precision (fp16 weights, fp32
  accumulators) per a vsrife patch the installer applies; pure fp16
  flickers on fast motion because flow-vector accumulators overflow.

Tuning knobs are inline in `~/.config/mpv/rife.vpy`,
`~/.config/mpv/rife-light.vpy`, `~/.config/mpv/rife-half.vpy`, and
`~/.config/mpv/mpv.conf`. Manual override mid-playback: F8 toggles
FSRCNNX, F9 cycles the active RIFE config (4.26 → 4.6 → 4.6 @
scale=0.5 → off → loop) regardless of which band the profile-cond
logic chose.

### TRT engine cache

The installer pre-compiles RIFE engines for the common shapes (720p +
1080p × RIFE 4.26 + 4.6) so the first playback at those resolutions is
instant. Cache lives at
`~/miniforge3/envs/vsmpv/lib/python3.12/site-packages/vsrife/models/`,
~250 MB total. **Engines are keyed on (model, padded_shape, fp16,
scale, GPU model, TRT version)**, where padded_shape rounds the source
height up to a multiple of 32 (RIFE 4.6) or 64 (RIFE 4.26).

If you play a video at an unusual resolution (a lot of films are
1920×800, 1920×816, 1920×1036, …), the first play of that exact shape
will JIT-compile a new engine — **the player window will appear to
freeze for ~30–60 s while it does**. Bigger resolution = longer freeze.
Subsequent plays of the same shape are instant. This is normal.

**If a cached engine starts producing broken video** (corrupt frames,
wrong colors, etc.) it usually means the NVIDIA driver / CUDA / TensorRT
was updated in place — vsrife keys cache files on GPU model and TRT
version but **not** driver build, so cached engines that match by name
can be incompatible at runtime after a driver upgrade. Fix:

```bash
./install.sh install --rebuild-trt
```

That wipes every `*.ts` file in the cache dir before re-warming the
common shapes, forcing TRT to recompile against the new driver/runtime
stack. Adds ~2–3 minutes to install vs the normal idempotent path.

## Keybindings

- **F8** — toggle FSRCNNX shader
- **F9** — cycle RIFE: 4.26 (default) → 4.6 (light) → off
- All mpv defaults intact (`i` for stats, `s` for screenshot, etc.)

Danmaku has its own bindings — see the [Danmaku](#danmaku-bullet-chat)
section below.

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
| `~/.config/mpv/{mpv,input}.conf, rife.vpy, rife-light.vpy, rife-half.vpy, shaders/` | User mpv config (also used by shim via symlinks) |
| `~/.config/jellyfin-mpv-shim/conf.json` | Shim's own config (server creds, OSC visibility, etc.) |
| `~/.config/jellyfin-mpv-shim/{mpv.conf,input.conf,rife.vpy,rife-light.vpy,rife-half.vpy,shaders,scripts}` | Symlinks to `~/.config/mpv/` |
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

The danmaku plugin lives in its own repo at
[Cryspia/mpv-dandanplay-danmaku](https://github.com/Cryspia/mpv-dandanplay-danmaku).
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
- **RIFE inference contends with libplacebo's compositor on the GB10.**
  Pure RIFE 4.26 throughput is ~48 fps on a 24fps source (verified
  with `--vo=null` so no compositor in the loop). Once the FSRCNNX
  shader starts running in mpv's gpu-next compositor, the same source
  drops to ~41 fps — the GPU's compute units split between TRT
  inference and the shader pass. The per-resolution band is the
  workaround: 1080p sources get the lighter 4.6 model so RIFE's
  compute share is smaller and the compositor doesn't starve.
- **A Vulkan-backend RIFE is not viable on this hardware.** We tried
  ncnn-Vulkan to share the GPU API with libplacebo (avoiding
  CUDA↔Vulkan handoffs); it ran ~14 fps in the same pipeline because
  ncnn's Vulkan kernels are far slower than TRT's tensor-core kernels
  on Blackwell, and same-Vulkan contention turned out to be worse
  than cross-API contention. Sticking with TRT.
