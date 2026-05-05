#!/usr/bin/env bash
# install.sh — dgxspark-jellyfin-mpv-rife
#
# Reproducible installer for: jellyfin-mpv-shim + mpv (custom build) +
# RIFE realtime interpolation (vsrife/TensorRT) + FSRCNNX shader upscale,
# all running in a single Miniforge conda environment, on DGX Spark
# (NVIDIA GB10, ARM64, CUDA 13, Ubuntu 24.04 + GNOME Wayland).
#
# Optionally also installs the danmaku (bullet-chat) plugin from
# https://github.com/Cryspia/mpv-dandanplay-danmaku.
#
# Usage:
#   ./install.sh install [--no-mirrors] [--no-danmaku]
#                          full install on a clean system
#                          --no-mirrors: skip USTC mirror config
#                                        (use outside China)
#                          --no-danmaku: skip the danmaku plugin step
#   ./install.sh status     show what's currently installed and where
#   ./install.sh uninstall  remove everything we installed except apt
#                           packages and miniforge itself (so re-install
#                           is fast)
#
# Idempotent: re-running `install` only does work that's still missing.

set -euo pipefail

# ============================================================================
# Constants & paths
# ============================================================================
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="vsmpv"
FORGE_DIR="$HOME/miniforge3"
ENV_PREFIX="$FORGE_DIR/envs/$ENV_NAME"
MPV_SRC_DIR="$HOME/src/mpv"
MPV_VERSION="v0.41.0"

# Danmaku plugin — fetched from a sibling project, installed via its own
# install.py. We pin to main; bump DANMAKU_REF to a tag/commit if you
# want a reproducible build.
DANMAKU_REPO_URL="https://github.com/Cryspia/mpv-dandanplay-danmaku.git"
DANMAKU_SRC_DIR="$HOME/src/mpv-dandanplay-danmaku"
DANMAKU_REF="main"

# User-visible install locations
MPV_CFG_DIR="$HOME/.config/mpv"
SHIM_CFG_DIR="$HOME/.config/jellyfin-mpv-shim"
WRAPPER_DIR="$HOME/.local/bin"
APPS_DIR="$HOME/.local/share/applications"
ICON_ROOT="$HOME/.local/share/icons/hicolor"
AUTOSTART_DIR="$HOME/.config/autostart"

# Default install flags. Toggled by `install --no-mirrors` /
# `--no-danmaku`. Defaulting both ON because this script targets DGX
# Spark + China deployments where USTC mirrors are dramatically faster
# and most users want danmaku.
USE_MIRRORS=1
INSTALL_DANMAKU=1

# Mirrors (USTC for China)
USTC_CONDA_FORGE="https://mirrors.ustc.edu.cn/anaconda/cloud"
USTC_PYPI="https://pypi.mirrors.ustc.edu.cn/simple/"
USTC_PYPI_HOST="pypi.mirrors.ustc.edu.cn"

# PyTorch CUDA 13 wheel index (NOT mirrored on USTC)
PYTORCH_INDEX="https://download.pytorch.org/whl/cu130"
PYTORCH_NIGHTLY_INDEX="https://download.pytorch.org/whl/nightly/cu130"

# ============================================================================
# Logging helpers
# ============================================================================
log()    { printf "\033[1;36m[+] %s\033[0m\n" "$*"; }
warn()   { printf "\033[1;33m[!] %s\033[0m\n" "$*"; }
fatal()  { printf "\033[1;31m[x] %s\033[0m\n" "$*" >&2; exit 1; }
section() { printf "\n\033[1;34m=== %s ===\033[0m\n" "$*"; }

# Run a command in the conda env (requires env created)
in_env() {
  # shellcheck disable=SC1091
  source "$FORGE_DIR/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME" >/dev/null 2>&1
  "$@"
}

# ============================================================================
# Environment detection
# ============================================================================
detect_environment() {
  section "detect environment"
  [[ "$(uname -m)" == aarch64 ]] || fatal "this installer targets aarch64 (got $(uname -m))"
  [[ -f /etc/os-release ]] || fatal "/etc/os-release missing"
  . /etc/os-release
  log "OS: $PRETTY_NAME"
  if ! command -v nvidia-smi >/dev/null; then
    warn "nvidia-smi not found — no NVIDIA driver? RIFE TRT will not work."
  else
    local cuda
    cuda=$(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader 2>/dev/null | head -1)
    log "GPU: $cuda"
  fi
  if ! command -v sudo >/dev/null; then
    fatal "sudo required for apt install steps"
  fi
}

# ============================================================================
# Step 1: apt packages (system-side dependencies)
# ============================================================================
install_apt_packages() {
  section "step 1/10: apt packages"
  local needed=(
    # Build essentials for from-source mpv
    build-essential git pkg-config
    # mpv X11 build dep (only x11-related .pc not on conda-forge)
    libxpresent-dev
    # AppIndicator GI typelib for shim's tray icon. Conda-forge has no
    # libayatana-appindicator on aarch64; we use the system library and
    # inject GI_TYPELIB_PATH from the wrapper.
    gir1.2-ayatanaappindicator3-0.1
    libayatana-appindicator3-1
    # GNOME extension for showing tray icons in the top bar
    gnome-shell-extension-appindicator
  )
  local missing=()
  for p in "${needed[@]}"; do
    dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p")
  done
  if [[ ${#missing[@]} -eq 0 ]]; then
    log "all apt packages already installed"
    return
  fi
  log "installing: ${missing[*]}"
  sudo apt-get update -qq
  sudo apt-get install -y --no-install-recommends "${missing[@]}"
}

# ============================================================================
# Step 2: miniforge + USTC mirror config
# ============================================================================
install_miniforge() {
  section "step 2/10: miniforge + USTC mirrors"
  if [[ -x "$FORGE_DIR/bin/conda" ]]; then
    log "miniforge already present at $FORGE_DIR"
  else
    log "downloading + installing miniforge to $FORGE_DIR"
    cd /tmp
    curl -fL --retry 3 -O \
      https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
    bash Miniforge3-Linux-aarch64.sh -b -p "$FORGE_DIR"
    rm -f Miniforge3-Linux-aarch64.sh
  fi

  if (( USE_MIRRORS )); then
    log "writing ~/.condarc with USTC conda-forge mirror"
    cat > "$HOME/.condarc" <<EOF
channels:
  - conda-forge
custom_channels:
  conda-forge: $USTC_CONDA_FORGE
channel_priority: strict
show_channel_urls: true
EOF

    log "writing ~/.config/pip/pip.conf with USTC PyPI mirror"
    mkdir -p "$HOME/.config/pip"
    cat > "$HOME/.config/pip/pip.conf" <<EOF
[global]
index-url = $USTC_PYPI
trusted-host = $USTC_PYPI_HOST
EOF
  else
    log "skipping mirror config (--no-mirrors); using upstream conda-forge + PyPI"
  fi
}

# ============================================================================
# Step 3: create conda env with all build + runtime deps
# ============================================================================
create_conda_env() {
  section "step 3/10: create conda env '$ENV_NAME'"
  # shellcheck disable=SC1091
  source "$FORGE_DIR/etc/profile.d/conda.sh"

  if mamba env list 2>/dev/null | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log "env '$ENV_NAME' already exists"
    return
  fi

  # When mirrors are enabled, pass the USTC URL explicitly via -c. The
  # `custom_channels` rewrite in ~/.condarc is unreliable with mamba 2.x
  # — `-c conda-forge` can bypass it and hit conda.anaconda.org directly,
  # which the China GFW kills mid-TLS-stream (SSL_read unexpected eof).
  local fchannel="conda-forge"
  if (( USE_MIRRORS )); then
    fchannel="https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge"
  fi

  log "creating env (will pull ~1GB of conda-forge packages)"
  log "  channel: $fchannel"
  mamba create -n "$ENV_NAME" -y -c "$fchannel" \
    python=3.12 \
    `# media + filter chain` \
    vapoursynth ffmpeg \
    `# GTK stack for shim's tray icon (PyGObject loads system AppIndicator typelib at runtime)` \
    pygobject gtk3 librsvg gobject-introspection pillow \
    `# mpv from-source build deps` \
    meson ninja cython pkg-config \
    wayland-protocols libvulkan-headers \
    freetype expat \
    'lua=5.1'  # mpv 0.41 requires lua 5.1/5.2; rejects 5.4
}

# ============================================================================
# Step 4: build mpv from source
# ============================================================================
build_mpv() {
  section "step 4/10: build mpv $MPV_VERSION from source"
  # shellcheck disable=SC1091
  source "$FORGE_DIR/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"

  # If a previous build already produced mpv with the right features, skip.
  if "$ENV_PREFIX/bin/mpv" -v 2>&1 | sed -n 's/.*enabled features:[[:space:]]*//p' | head -1 \
       | { read -r feats; for f in vapoursynth wayland x11 vulkan lua; do
             grep -qw "$f" <<<"$feats" || exit 1; done; } 2>/dev/null; then
    log "mpv already built with vapoursynth + wayland + x11 + vulkan + lua"
    return
  fi

  mkdir -p "$(dirname "$MPV_SRC_DIR")"
  if [[ ! -d "$MPV_SRC_DIR/.git" ]]; then
    log "cloning mpv $MPV_VERSION"
    git clone --depth 1 --branch "$MPV_VERSION" \
      https://github.com/mpv-player/mpv.git "$MPV_SRC_DIR"
  fi

  cd "$MPV_SRC_DIR"
  # Conda env first (so it wins on shared deps), system path appended only
  # for `xpresent.pc` which conda-forge doesn't ship on aarch64.
  export PKG_CONFIG_PATH="$ENV_PREFIX/lib/pkgconfig:$ENV_PREFIX/share/pkgconfig:/usr/lib/aarch64-linux-gnu/pkgconfig:/usr/share/pkgconfig"
  export LIBRARY_PATH="$ENV_PREFIX/lib"
  export CPATH="$ENV_PREFIX/include"
  export LDFLAGS="-L$ENV_PREFIX/lib -Wl,-rpath,$ENV_PREFIX/lib"

  rm -rf build
  log "meson setup (target prefix: $ENV_PREFIX)"
  meson setup build \
    --prefix="$ENV_PREFIX" --buildtype=release \
    -Dvapoursynth=enabled -Dvulkan=enabled \
    -Dwayland=enabled -Dx11=enabled \
    -Degl=enabled -Degl-wayland=enabled -Degl-x11=enabled \
    -Ddrm=disabled \
    -Dlcms2=enabled \
    -Dlua=enabled \
    -Djavascript=disabled \
    -Dmanpage-build=disabled

  log "compiling (~30s)"
  meson compile -C build
  log "installing into $ENV_PREFIX"
  meson install -C build

  unset PKG_CONFIG_PATH LIBRARY_PATH CPATH LDFLAGS
  cd - >/dev/null

  # Verify
  local feats
  feats=$("$ENV_PREFIX/bin/mpv" -v 2>&1 | sed -n 's/.*enabled features:[[:space:]]*//p' | head -1)
  for need in vapoursynth wayland x11 vulkan lua; do
    grep -qw "$need" <<<"$feats" || fatal "mpv build is missing feature: $need"
  done
  log "mpv build OK: features include vapoursynth + wayland + x11 + vulkan + lua"
}

# ============================================================================
# Step 5: pip packages (PyTorch CUDA 13, vsrife, shim, TensorRT)
# ============================================================================
install_pip_packages() {
  section "step 5/10: pip packages"
  # shellcheck disable=SC1091
  source "$FORGE_DIR/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"

  pip install --upgrade pip

  # PyTorch — its CUDA wheels live on pytorch.org (not mirrored on USTC).
  if ! python -c "import torch" 2>/dev/null; then
    log "installing torch (CUDA 13 wheel from pytorch.org)"
    pip install torch --index-url "$PYTORCH_INDEX" || \
      pip install --pre torch --index-url "$PYTORCH_NIGHTLY_INDEX"
  else
    log "torch already installed: $(python -c 'import torch; print(torch.__version__)')"
  fi

  # vsrife + jellyfin-mpv-shim (from USTC PyPI mirror, default in pip.conf)
  pip install vsrife "jellyfin-mpv-shim[gui]"

  # TensorRT for the high-perf RIFE backend
  pip install tensorrt torch_tensorrt

  log "verifying versions:"
  python - <<'PY'
import importlib, vapoursynth as vs
print(f"  vapoursynth : {vs.core.version().splitlines()[0]}")
import torch; print(f"  torch       : {torch.__version__}  (cuda available: {torch.cuda.is_available()})")
import tensorrt; print(f"  tensorrt    : {tensorrt.__version__}")
import torch_tensorrt; print(f"  torch_trt   : {torch_tensorrt.__version__}")
import vsrife
v = importlib.metadata.version("vsrife"); print(f"  vsrife      : {v}")
v = importlib.metadata.version("jellyfin-mpv-shim"); print(f"  shim        : {v}")
PY
}

# ============================================================================
# Step 6: source patches (shim osc-removal compat, vsrife mixed precision)
# ============================================================================
apply_patches() {
  section "step 6/10: patch shim + vsrife"

  # The conda env has both `python3.12` and `python3.1` (a symlink) under
  # lib/, so a `python*` glob hits the same files twice. Iterate concrete
  # directories only.
  local pydir f
  for pydir in "$ENV_PREFIX"/lib/python*; do
    [[ -L "$pydir" ]] && continue
    [[ -d "$pydir" ]] || continue

    # Patch jellyfin-mpv-shim 2.9.0 for mpv 0.41 compatibility:
    # mpv 0.41 removed the `osc` cmdline option; shim still passes
    # --osc=no which makes mpv exit "Fatal error" before the IPC
    # handshake. Replace with `script-opts=osc-visibility=never`.
    f="$pydir/site-packages/jellyfin_mpv_shim/player.py"
    if [[ -f "$f" ]]; then
      if grep -q '"osc"\] = False' "$f"; then
        cp -n "$f" "${f}.bak"
        # mpv 0.41 removed the legacy `--osc` flag (and `_player.osc`
        # property). Replace shim's `mpv_options["osc"] = False` with a
        # script-opts setting that maps the user's `settings.enable_osc`
        # config field to the new `osc-visibility=auto|never` property.
        # Hard-coding "never" (our previous patch) ignored the user's
        # preference and made the OSC permanently invisible in shim mode.
        sed -i 's|mpv_options\["osc"\] = False|mpv_options["script_opts"] = "osc-visibility=" + ("auto" if settings.enable_osc else "never")  # mpv 0.41+ removed --osc|' "$f"
        log "patched $f for mpv 0.41 osc-removal"
      else
        log "shim player.py already patched (or different version)"
      fi
    fi

    # Patch vsrife for mixed-precision TRT compile. Default vsrife passes
    # use_explicit_typing=True which forces the whole graph (incl.
    # accumulators) to a single dtype. With fp16 inputs (RGBH),
    # accumulators go fp16 too → flow-vector overflow on fast motion →
    # flicker. Switch to use_explicit_typing=False +
    # enabled_precisions={fp16,fp32} so TRT auto-promotes overflow-prone
    # ops (grid_sampler, big reductions) to fp32.
    f="$pydir/site-packages/vsrife/__init__.py"
    if [[ -f "$f" ]]; then
      if grep -q 'use_explicit_typing=True,' "$f"; then
        cp -n "$f" "${f}.bak"
        sed -i 's|use_explicit_typing=True,|use_explicit_typing=False, enabled_precisions={torch.float16, torch.float32},|g' "$f"
        log "patched $f for TRT mixed-precision"
      else
        log "vsrife __init__.py already patched (or different version)"
      fi
    fi
  done
}

# ============================================================================
# Step 7: write user-facing config files (mpv.conf, input.conf, rife.vpy)
#         and copy the FSRCNNX shader from project assets
# ============================================================================
install_configs() {
  section "step 7/10: configs + shader"
  mkdir -p "$MPV_CFG_DIR/shaders" "$MPV_CFG_DIR/scripts"

  # mpv.conf — overwrite (we own it)
  cat > "$MPV_CFG_DIR/mpv.conf" <<'EOF'
vo=gpu-next
gpu-api=vulkan
# HiDPI: render at the physical-pixel resolution (e.g. 4K on a 200%-scaled
# GNOME monitor) instead of the logical surface resolution. Without this,
# mutter compositor 2x-upscales mpv's logical surface and FSRCNNX won't
# trigger because OUTPUT/LUMA stays at 1.0.
hidpi-window-scale=yes
# Native Wayland: ~1ms less frame-presentation latency than Xwayland on
# NVIDIA, fewer vsync misses under heavy RIFE load. GNOME mutter doesn't
# do SSD and mpv 0.41 has no libdecor → window has no title bar / min /
# close. Switch to `x11vk` if you want GNOME-drawn decorations back.
gpu-context=waylandvk
hwdec=auto-safe
profile=gpu-hq

# Frame timing — RIFE generates the extra frames, so mpv's own temporal
# interpolation must be off (otherwise we double-interpolate).
video-sync=display-resample
interpolation=no
tscale=oversample

scale=ewa_lanczossharp
cscale=ewa_lanczossharp
dscale=mitchell
correct-downscaling=yes
linear-downscaling=yes
sigmoid-upscaling=yes
deband=yes

target-colorspace-hint=yes
dither-depth=auto

# Danmaku: our generated ASS encodes every comment with explicit \move
# \pos \1c overrides. mpv's secondary-sub-ass-override defaults to
# "strip" which throws those tags away — collapsing all comments to a
# top-stack. "no" tells mpv to honor the script verbatim. secondary-
# sub-pos=0 prevents an additional vertical shift.
secondary-sub-ass-override=no
secondary-sub-pos=0

# FSRCNNX luma upscale, loaded globally — the shader self-gates at
# `//!WHEN OUTPUT/LUMA > 1.3`, so it only kicks in when the display
# output is meaningfully larger than the source. Toggle with F8.
# (See [no-fsrcnnx-large] below for the >1080p-source override.)
glsl-shaders=~~/shaders/FSRCNNX_x2_8-0-4-1.glsl

# RIFE realtime 2x interpolation, applied per resolution band so the
# GB10 isn't asked to do something it can't sustain:
#   ≤30fps and h ≤ 720          → 4.26 (heavy quality, small frames cheap)
#   ≤30fps and 720 < h ≤ 1080   → 4.6  (lighter, balanced for 1080p)
#   ≤30fps and h > 1080          → no RIFE (4K interpolation pegs SMs;
#                                  also UHD content is generally HFR
#                                  natively and clips our 30fps gate)
#   >30fps                       → no RIFE (already smooth; would only
#                                  produce frames mpv drops at vsync)
#
# `or 999` / `or 0` sentinels: at the very first profile-cond evaluation
# (before the demuxer fills container-fps and video-params/h), those
# properties are nil. Defaulting to values that fail the gate prevents
# a ~half-second window where the vf gets attached then ripped off
# when the real numbers land. profile-cond re-evaluates reactively
# whenever the referenced properties change.
[rife-heavy]
profile-cond=(p["container-fps"] or 999)<=30 and 0<(p["video-params/h"] or 0) and (p["video-params/h"] or 0)<=720
profile-restore=copy-equal
vf=vapoursynth=~~/rife.vpy

[rife-light]
profile-cond=(p["container-fps"] or 999)<=30 and (p["video-params/h"] or 0)>720 and (p["video-params/h"] or 0)<=1080
profile-restore=copy-equal
vf=vapoursynth=~~/rife-light.vpy

# For >1080p sources, force FSRCNNX off. The shader's own //!WHEN gate
# at 1.3× wouldn't activate on a 4K source playing on a 4K display
# (output/luma ≈ 1.0), but a HiDPI-scaled display surface above 4K
# could push the ratio over the gate at 4K source — and we don't want
# to pile FSRCNNX shader work on top of an already-large frame.
[no-fsrcnnx-large]
profile-cond=(p["video-params/h"] or 0)>1080
profile-restore=copy-equal
glsl-shaders=
EOF
  log "wrote $MPV_CFG_DIR/mpv.conf"

  # input.conf — F8 / F9 toggles with on-screen feedback
  cat > "$MPV_CFG_DIR/input.conf" <<'EOF'
# F8: toggle FSRCNNX shader on/off, with on-screen feedback so you can
# see immediately whether the binding fired.
F8 cycle-values glsl-shaders "~~/shaders/FSRCNNX_x2_8-0-4-1.glsl" ""; show-text "FSRCNNX: ${glsl-shaders}" 2000

# F9: cycle RIFE between three states:
#   4.26 (default, best quality) → 4.6 (lighter fallback) → off → loop
# Useful when an extreme scene briefly pushes 4.26 over budget; F9 once
# to drop down to 4.6, F9 again to disable, F9 again to come back.
F9 cycle-values vf "vapoursynth=~~/rife.vpy" "vapoursynth=~~/rife-light.vpy" ""; show-text "RIFE: ${vf}" 2500
EOF
  log "wrote $MPV_CFG_DIR/input.conf"

  # rife.vpy — RIFE 4.26 + scale=1.0 + TRT mixed-precision (RGBH triggers
  # vsrife's fp16 path, the patch above makes TRT use mixed precision).
  cat > "$MPV_CFG_DIR/rife.vpy" <<'EOF'
# RIFE realtime frame interpolation — vapoursynth + vsrife (TRT mixed precision).
#
# vsrife internally branches on input bit-depth: bits_per_sample==16 → fp16
# weights/inputs. With the project's vsrife patch (use_explicit_typing=False
# + enabled_precisions={fp16,fp32}), TRT auto-promotes overflow-prone ops
# (grid_sampler, reductions) to fp32 → no flicker, ~30% faster than fp32.
#
# === tuning knobs (light → heavy) ===
#   model: "4.6" (fastest) / "4.25.lite" / "4.25" / "4.26" (heaviest, current)
#   scale: 1.0 (full-res flow, current) / 0.5 / 0.25 (lower res, faster)
#   trt=True: TensorRT engine; first run JIT-compiles ~30-60s, cached.
#
# Default is tuned for 1080p source on a 4K HiDPI display, no streaming.

import vapoursynth as vs
from vsrife import rife

core = vs.core
clip = video_in  # supplied by mpv

## Auto-detect source color matrix from frame metadata so HDR (bt.2020)
## content doesn't get its colors mangled when going through RIFE. mpv
## populates _Matrix from the container: 1=bt.709 (SDR HD), 9=bt.2020nc
## (HDR/UHD wide gamut), 6=bt.601 (SD). Hardcoding "709" was wrong on HDR.
_MATRIX_NAME = {1: "709", 6: "170m", 7: "240m", 9: "2020ncl"}
try:
    _m = int(clip.get_frame(0).props.get("_Matrix", 1))
except Exception:
    _m = 1
_mtx = _MATRIX_NAME.get(_m, "709")

clip = core.resize.Bicubic(clip, format=vs.RGBH, matrix_in_s=_mtx)
clip = rife(
    clip,
    model="4.26",
    scale=1.0,
    factor_num=2,
    factor_den=1,
    auto_download=True,
    trt=True,
)
clip = core.resize.Bicubic(clip, format=vs.YUV420P10, matrix_s=_mtx)

clip.set_output()
EOF
  log "wrote $MPV_CFG_DIR/rife.vpy"

  # rife-light.vpy — fallback config the F9 cycle drops to when 4.26 is
  # briefly too heavy (rare, but useful safety net). Same color-matrix
  # auto-detect as rife.vpy. Engine for 4.6+scale=1.0 JIT-compiles on
  # first F9 cycle into it.
  cat > "$MPV_CFG_DIR/rife-light.vpy" <<'EOF'
# RIFE realtime interpolation — LIGHT FALLBACK (model 4.6).
# Cycled to via F9 when 4.26 is too heavy. ~30% less GPU than 4.26.

import vapoursynth as vs
from vsrife import rife

core = vs.core
clip = video_in

_MATRIX_NAME = {1: "709", 6: "170m", 7: "240m", 9: "2020ncl"}
try:
    _m = int(clip.get_frame(0).props.get("_Matrix", 1))
except Exception:
    _m = 1
_mtx = _MATRIX_NAME.get(_m, "709")

clip = core.resize.Bicubic(clip, format=vs.RGBH, matrix_in_s=_mtx)
clip = rife(
    clip,
    model="4.6",
    scale=1.0,
    factor_num=2,
    factor_den=1,
    auto_download=True,
    trt=True,
)
clip = core.resize.Bicubic(clip, format=vs.YUV420P10, matrix_s=_mtx)

clip.set_output()
EOF
  log "wrote $MPV_CFG_DIR/rife-light.vpy"

  # FSRCNNX shader from project assets
  cp -f "$PROJECT_DIR/shaders/FSRCNNX_x2_8-0-4-1.glsl" "$MPV_CFG_DIR/shaders/"
  log "copied FSRCNNX shader to $MPV_CFG_DIR/shaders/"
}

# ============================================================================
# Step 8: warm the TRT engine cache for both rife configs
#
# Without this, the user's first F9 press (or first playback) triggers a
# 30-60s JIT compile mid-action. Doing it now upfront costs the same time
# but at install — invisible to the user later. Idempotent: vsrife
# detects existing cache files and skips re-compile.
# ============================================================================
warm_trt_cache() {
  section "step 8/10: warm TRT engine cache (4.26 + 4.6, 1080p + 4K)"
  # shellcheck disable=SC1091
  source "$FORGE_DIR/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"

  # vsrife's TRT static-shape mode uses the input clip's actual dimensions
  # to build the engine, so feeding it a dummy at the right resolution
  # produces an engine matching real playback. Cache filenames embed the
  # padded shape (e.g. 1920x1088, 3840x2176) so 1080p and 4K engines
  # coexist independently.
  #
  # Total disk: ~150 MB for all 4 engines. Total time on a clean install:
  # ~2-4 minutes (each engine ~30-60s); cache hits return in <1s.
  python - <<'PY'
import time, vapoursynth as vs
from vsrife import rife

core = vs.core

def warm(label, model, width, height):
    dummy = core.std.BlankClip(width=width, height=height,
                               format=vs.YUV420P8,
                               length=2, fpsnum=30, fpsden=1)
    dummy = core.resize.Bicubic(dummy, format=vs.RGBH, matrix_in_s="709")
    t0 = time.time()
    clip = rife(dummy, model=model, scale=1.0,
                factor_num=2, factor_den=1,
                auto_download=True, trt=True)
    # Force frame request so vsrife actually compiles or loads the engine
    clip.get_frame(0)
    print(f"  {label}: ready in {time.time()-t0:.1f}s")

# 1080p (the common case — most Jellyfin libraries)
warm("RIFE 4.26 @ 1080p (default)",       "4.26", 1920, 1080)
warm("RIFE 4.6  @ 1080p (light fallback)", "4.6",  1920, 1080)

# 4K (modern streaming releases — UHD Blu-rays, HDR remuxes)
warm("RIFE 4.26 @ 4K    (default)",       "4.26", 3840, 2160)
warm("RIFE 4.6  @ 4K    (light fallback)", "4.6",  3840, 2160)
PY
}

# ============================================================================
# Step 8b: danmaku (bullet-chat) plugin   [optional, --no-danmaku to skip]
#
# Delegates to the standalone Cryspia/mpv-dandanplay-danmaku project.
# We clone (or fast-forward an existing checkout) to $DANMAKU_SRC_DIR
# and run its install.py — that project's installer handles the bundle
# layout (~/.config/mpv/scripts/dandanplay/), seeds JSON config files,
# and writes the credentials .example.
# ============================================================================
install_danmaku() {
  section "step 8b/10: danmaku plugin (Cryspia/mpv-dandanplay-danmaku)"

  # Backwards-compat: sweep up the legacy single-file install layout
  # from pre-extraction dgxspark builds. New bundle layout lives at
  # scripts/dandanplay/ instead, so the old paths would just be stale.
  rm -f "$MPV_CFG_DIR/scripts/danmaku.lua" \
        "$WRAPPER_DIR/danmaku_helper.py"

  # Clone or fast-forward the danmaku project.
  mkdir -p "$(dirname "$DANMAKU_SRC_DIR")"
  if [[ -d "$DANMAKU_SRC_DIR/.git" ]]; then
    log "updating existing $DANMAKU_SRC_DIR"
    git -C "$DANMAKU_SRC_DIR" fetch --quiet origin "$DANMAKU_REF"
    git -C "$DANMAKU_SRC_DIR" checkout --quiet "$DANMAKU_REF"
    git -C "$DANMAKU_SRC_DIR" pull --ff-only --quiet
  else
    log "cloning $DANMAKU_REPO_URL → $DANMAKU_SRC_DIR"
    git clone --quiet --branch "$DANMAKU_REF" \
        "$DANMAKU_REPO_URL" "$DANMAKU_SRC_DIR"
  fi
  log "  HEAD: $(git -C "$DANMAKU_SRC_DIR" rev-parse --short HEAD)"

  # Run the project's installer using our conda env's Python (so we
  # know urllib/json/hmac are present even on stock systems with a
  # stripped-down system Python).
  in_env python3 "$DANMAKU_SRC_DIR/install.py"
}

# ============================================================================
# Step 9: shim config + symlinks (so shim shares ~/.config/mpv/ contents)
# ============================================================================
install_shim_config() {
  section "step 9/10: shim config + symlinks to share with ~/.config/mpv/"
  mkdir -p "$SHIM_CFG_DIR"

  # conf.json — set external mpv to our env's mpv binary
  python3 - <<PY
import json, os
p = "$SHIM_CFG_DIR/conf.json"
data = {}
if os.path.exists(p):
    try:
        with open(p) as f: data = json.load(f)
    except Exception:
        data = {}
data["mpv_ext"] = True
data["mpv_ext_path"] = "$ENV_PREFIX/bin/mpv"
with open(p, "w") as f:
    json.dump(data, f, indent=2)
print("wrote", p)
PY

  # shim insists on its own --config-dir, isolating mpv from ~/.config/mpv/.
  # Symlink the user-facing files back so one set of config rules both.
  local name src dst
  for name in mpv.conf input.conf rife.vpy rife-light.vpy \
              danmaku-config.json danmaku-credentials.json danmaku-settings.json; do
    src="$MPV_CFG_DIR/$name"
    dst="$SHIM_CFG_DIR/$name"
    [[ -L "$dst" ]] && continue
    [[ -e "$src" ]] || continue
    rm -f "$dst"
    ln -s "$src" "$dst"
    log "linked $dst → $src"
  done
  for name in shaders scripts; do
    src="$MPV_CFG_DIR/$name"
    dst="$SHIM_CFG_DIR/$name"
    [[ -L "$dst" ]] && continue
    [[ -d "$src" ]] || continue
    rmdir "$dst" 2>/dev/null || rm -rf "$dst"
    ln -s "$src" "$dst"
    log "linked $dst/ → $src/"
  done
}

# ============================================================================
# Step 9: launcher wrappers + .desktop entries + autostart + icons
# ============================================================================
install_launchers() {
  section "step 10/10: wrappers + .desktop + autostart + icons"
  mkdir -p "$WRAPPER_DIR" "$APPS_DIR" "$AUTOSTART_DIR"

  # ---------- mpv-conda wrapper ----------
  # PYTHONHOME so mpv's embedded Python (used by vapoursynth/RIFE) can
  # locate its stdlib without conda activation. Without this, vapoursynth
  # filter init aborts and video stream goes straight to EOF.
  cat > "$WRAPPER_DIR/mpv-conda" <<EOF
#!/usr/bin/env bash
# Wrapper: sets PYTHONHOME so mpv's embedded Python (vapoursynth/RIFE)
# finds its stdlib when launched without conda activation.
export PYTHONHOME="$ENV_PREFIX"
exec "$ENV_PREFIX/bin/mpv" "\$@"
EOF
  chmod +x "$WRAPPER_DIR/mpv-conda"
  log "wrote $WRAPPER_DIR/mpv-conda"

  # ---------- jellyfin-mpv-shim wrapper ----------
  # GI_TYPELIB_PATH so PyGObject finds system's AppIndicator3 typelib
  # (conda-forge has no libayatana-appindicator on aarch64).
  # PYTHONHOME so the mpv shim spawns can boot vapoursynth.
  # NOTE: do NOT set LD_LIBRARY_PATH — that would put system libs ahead of
  # conda's RUNPATH, and conda-built mpv would load ABI-incompatible
  # system libass and fail with `undefined symbol` at startup.
  cat > "$WRAPPER_DIR/jellyfin-mpv-shim" <<EOF
#!/usr/bin/env bash
SYS_GIR="/usr/lib/aarch64-linux-gnu/girepository-1.0"
export GI_TYPELIB_PATH="\${GI_TYPELIB_PATH:+\$GI_TYPELIB_PATH:}\$SYS_GIR"
export PYTHONHOME="$ENV_PREFIX"
exec "$ENV_PREFIX/bin/jellyfin-mpv-shim" "\$@"
EOF
  chmod +x "$WRAPPER_DIR/jellyfin-mpv-shim"
  log "wrote $WRAPPER_DIR/jellyfin-mpv-shim"

  # ---------- icons (from project assets) ----------
  local size src dst_dir
  for size in 16 32 48 64 128 256; do
    src="$PROJECT_DIR/icons/shim/$size.png"
    [[ -f "$src" ]] || continue
    dst_dir="$ICON_ROOT/${size}x${size}/apps"
    mkdir -p "$dst_dir"
    cp -f "$src" "$dst_dir/jellyfin-mpv-shim.png"
  done
  for size in 16 32 64 128; do
    src="$PROJECT_DIR/icons/mpv/$size.png"
    [[ -f "$src" ]] || continue
    dst_dir="$ICON_ROOT/${size}x${size}/apps"
    mkdir -p "$dst_dir"
    cp -f "$src" "$dst_dir/mpv-conda.png"
  done
  if [[ -f "$PROJECT_DIR/icons/mpv/scalable.svg" ]]; then
    mkdir -p "$ICON_ROOT/scalable/apps"
    cp -f "$PROJECT_DIR/icons/mpv/scalable.svg" "$ICON_ROOT/scalable/apps/mpv-conda.svg"
  fi
  log "icons installed under $ICON_ROOT"

  # ---------- .desktop entries ----------
  cat > "$APPS_DIR/mpv-conda.desktop" <<EOF
[Desktop Entry]
Name=mpv
GenericName=Media Player
Comment=Play movies and songs
Exec=$WRAPPER_DIR/mpv-conda --player-operation-mode=pseudo-gui -- %U
Icon=mpv-conda
Type=Application
Categories=AudioVideo;Audio;Video;Player;TV;
MimeType=video/mp4;video/x-matroska;video/webm;application/x-mpegURL;
Terminal=false
StartupWMClass=mpv
EOF
  cat > "$APPS_DIR/jellyfin-mpv-shim.desktop" <<EOF
[Desktop Entry]
Name=Jellyfin MPV Shim
Comment=Cast Jellyfin media to mpv (with RIFE + FSRCNNX)
Exec=$WRAPPER_DIR/jellyfin-mpv-shim
Icon=jellyfin-mpv-shim
Type=Application
Categories=AudioVideo;Player;
Terminal=false
StartupWMClass=jellyfin-mpv-shim
EOF
  log "wrote .desktop entries to $APPS_DIR"

  # ---------- autostart shim on login ----------
  cat > "$AUTOSTART_DIR/jellyfin-mpv-shim.desktop" <<EOF
[Desktop Entry]
Name=Jellyfin MPV Shim
Comment=Cast Jellyfin media to mpv (with RIFE + FSRCNNX)
Exec=$WRAPPER_DIR/jellyfin-mpv-shim
Icon=jellyfin-mpv-shim
Type=Application
Categories=AudioVideo;Player;
Terminal=false
X-GNOME-Autostart-enabled=true
StartupWMClass=jellyfin-mpv-shim
EOF
  log "wrote autostart entry to $AUTOSTART_DIR"

  update-desktop-database "$APPS_DIR" 2>/dev/null || true
  gtk-update-icon-cache "$ICON_ROOT" 2>/dev/null || true
}

# ============================================================================
# Final summary
# ============================================================================
print_install_summary() {
  cat <<EOF

================================================================
  Install complete.
================================================================
  conda env       : $ENV_PREFIX
  mpv binary      : $ENV_PREFIX/bin/mpv (built from $MPV_VERSION source)
  shim binary     : $ENV_PREFIX/bin/jellyfin-mpv-shim
  mpv wrapper     : $WRAPPER_DIR/mpv-conda
  shim wrapper    : $WRAPPER_DIR/jellyfin-mpv-shim
  user mpv config : $MPV_CFG_DIR/{mpv.conf,input.conf,rife.vpy,shaders/}
  shim config     : $SHIM_CFG_DIR/  (symlinks to ~/.config/mpv/)
  app launchers   : $APPS_DIR/{mpv-conda,jellyfin-mpv-shim}.desktop
  autostart       : $AUTOSTART_DIR/jellyfin-mpv-shim.desktop

  Log out and back in (or run 'gtk-update-icon-cache' + restart GNOME
  Shell) so the new launchers + autostart + icons are picked up.

  TensorRT engines pre-compiled at install time (instant F9 cycling /
  first playback at 1080p and 4K):
    - RIFE 4.26 + scale=1.0 @ 1080p  (default)
    - RIFE 4.26 + scale=1.0 @ 4K     (default, UHD content)
    - RIFE 4.6  + scale=1.0 @ 1080p  (light fallback)
    - RIFE 4.6  + scale=1.0 @ 4K     (light fallback, UHD content)
  Cache lives at:
    $ENV_PREFIX/lib/python*/site-packages/vsrife/models/

  Other shapes (720p, 1440p) or different scales will JIT-compile a
  matching engine on first use (~30-60s once).

  Verify status with: $0 status
================================================================
EOF
}

# ============================================================================
# install — orchestrate all steps
# ============================================================================
cmd_install() {
  # Parse install-specific flags. All optional; defaults preserve the
  # original behavior (mirrors ON, danmaku ON).
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --no-mirrors|--no-mirror|--no-ustc) USE_MIRRORS=0; shift ;;
      --no-danmaku|--skip-danmaku)        INSTALL_DANMAKU=0; shift ;;
      -h|--help)
        cat <<EOF
Usage: $0 install [--no-mirrors] [--no-danmaku]

  --no-mirrors   Don't write USTC mirrors into ~/.condarc and
                 ~/.config/pip/pip.conf. Use this outside China where
                 the USTC endpoints are slow / unreachable.
  --no-danmaku   Skip the danmaku (bullet-chat) plugin step. The rest
                 of the stack (mpv + RIFE + FSRCNNX + shim) installs
                 normally.
EOF
        return 0
        ;;
      *) fatal "unknown install flag: $1 (try $0 install --help)" ;;
    esac
  done

  log "install flags: USE_MIRRORS=$USE_MIRRORS  INSTALL_DANMAKU=$INSTALL_DANMAKU"

  detect_environment
  install_apt_packages
  install_miniforge
  create_conda_env
  build_mpv
  install_pip_packages
  apply_patches
  install_configs
  warm_trt_cache
  if (( INSTALL_DANMAKU )); then
    install_danmaku
  else
    section "step 8b/10: danmaku plugin (skipped — --no-danmaku)"
  fi
  install_shim_config
  install_launchers
  print_install_summary
}

# ============================================================================
# status — show current installation state, versions, paths
# ============================================================================
cmd_status() {
  section "system"
  if [[ -f /etc/os-release ]]; then . /etc/os-release; echo "  OS: $PRETTY_NAME ($(uname -m))"; fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "  GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"
  fi

  section "apt packages"
  for p in libxpresent-dev gir1.2-ayatanaappindicator3-0.1 libayatana-appindicator3-1 \
           gnome-shell-extension-appindicator build-essential; do
    if dpkg -s "$p" >/dev/null 2>&1; then
      printf "  %-42s %s\n" "$p" "$(dpkg-query -W -f='${Version}' "$p" 2>/dev/null)"
    else
      printf "  %-42s %s\n" "$p" "(not installed)"
    fi
  done

  section "miniforge + conda env"
  if [[ -x "$FORGE_DIR/bin/conda" ]]; then
    echo "  miniforge: $FORGE_DIR ($("$FORGE_DIR/bin/conda" --version 2>/dev/null))"
  else
    echo "  miniforge: (not installed)"
    return
  fi
  if [[ -d "$ENV_PREFIX" ]]; then
    echo "  env: $ENV_PREFIX"
  else
    echo "  env '$ENV_NAME': (not created)"
    return
  fi

  section "key versions inside env"
  in_env python - <<'PY' 2>/dev/null || echo "  (env exists but Python failed to start)"
import importlib.metadata as m
def v(p):
    try: return m.version(p)
    except Exception: return "(missing)"
print(f"  python      : {__import__('sys').version.split()[0]}")
import vapoursynth as vs
print(f"  vapoursynth : {vs.core.version().splitlines()[0]}")
print(f"  torch       : {v('torch')}")
print(f"  tensorrt    : {v('tensorrt')}")
print(f"  torch_tensorrt : {v('torch_tensorrt')}")
print(f"  vsrife      : {v('vsrife')}")
print(f"  jellyfin-mpv-shim : {v('jellyfin-mpv-shim')}")
import torch
print(f"  CUDA / GPU  : available={torch.cuda.is_available()} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
PY

  section "mpv build"
  if [[ -x "$ENV_PREFIX/bin/mpv" ]]; then
    "$ENV_PREFIX/bin/mpv" --version 2>/dev/null | head -1 | sed 's/^/  /'
    echo "  features (relevant):"
    "$ENV_PREFIX/bin/mpv" -v 2>&1 | sed -n 's/.*enabled features:[[:space:]]*//p' | head -1 | tr ' ' '\n' | \
      grep -E '^(vapoursynth|wayland|x11|vulkan|lua|egl-wayland|egl-x11|libplacebo)$' | \
      sed 's/^/    /'
  else
    echo "  (not built)"
  fi

  section "patches applied"
  # Resolve through any pythonX.Y symlink (env has both python3.1 → python3.12)
  local pydir
  for pydir in "$ENV_PREFIX"/lib/python*; do
    [[ -L "$pydir" ]] && continue
    [[ -d "$pydir" ]] || continue
    local f="$pydir/site-packages/jellyfin_mpv_shim/player.py"
    if [[ -f "$f" ]]; then
      if grep -q 'osc-visibility=" + (' "$f" 2>/dev/null; then
        echo "  shim/player.py : osc-removal patch applied (honors settings.enable_osc)"
      elif grep -q 'osc-visibility=never' "$f" 2>/dev/null; then
        echo "  shim/player.py : older osc-removal patch (always-off — re-run install to upgrade)"
      else
        echo "  shim/player.py : NOT patched (would break mpv 0.41 playback)"
      fi
    fi
    f="$pydir/site-packages/vsrife/__init__.py"
    if [[ -f "$f" ]]; then
      if grep -q 'enabled_precisions={torch.float16, torch.float32}' "$f" 2>/dev/null; then
        echo "  vsrife/__init__.py : mixed-precision patch applied"
      else
        echo "  vsrife/__init__.py : NOT patched (fp16 would flicker)"
      fi
    fi
  done

  section "config files"
  for f in "$MPV_CFG_DIR/mpv.conf" "$MPV_CFG_DIR/input.conf" \
           "$MPV_CFG_DIR/rife.vpy" "$MPV_CFG_DIR/rife-light.vpy" \
           "$MPV_CFG_DIR/shaders/FSRCNNX_x2_8-0-4-1.glsl" \
           "$MPV_CFG_DIR/scripts/dandanplay/main.lua" \
           "$MPV_CFG_DIR/scripts/dandanplay/danmaku_helper.py" \
           "$MPV_CFG_DIR/danmaku-config.json" \
           "$MPV_CFG_DIR/danmaku-credentials.json" \
           "$SHIM_CFG_DIR/conf.json"; do
    if [[ -L "$f" ]]; then
      printf "  %-60s -> %s\n" "${f/#$HOME/~}" "$(readlink "$f")"
    elif [[ -e "$f" ]]; then
      printf "  %-60s (%s)\n" "${f/#$HOME/~}" "$(stat -c%s "$f") bytes"
    else
      printf "  %-60s (missing)\n" "${f/#$HOME/~}"
    fi
  done

  section "danmaku plugin"
  local helper="$MPV_CFG_DIR/scripts/dandanplay/danmaku_helper.py"
  if [[ -f "$helper" ]]; then
    if [[ -d "$DANMAKU_SRC_DIR/.git" ]]; then
      echo "  source: $DANMAKU_SRC_DIR @ $(git -C "$DANMAKU_SRC_DIR" rev-parse --short HEAD 2>/dev/null)"
    fi
    in_env python3 "$helper" check 2>&1 \
      | grep -E "^(\[danmaku\]|OK|HTTP_|NO_CONFIG|ERROR|NETWORK)" \
      | sed 's/^/  /'
  else
    echo "  (helper not installed)"
  fi
  if [[ -d "$HOME/.cache/mpv-danmaku" ]]; then
    local nm na
    nm=$(test -f "$HOME/.cache/mpv-danmaku/matches.json" \
         && python3 -c "import json; d=json.load(open('$HOME/.cache/mpv-danmaku/matches.json')); print(len(d))" 2>/dev/null \
         || echo 0)
    na=$(test -f "$HOME/.cache/mpv-danmaku/aliases.json" \
         && python3 -c "import json; d=json.load(open('$HOME/.cache/mpv-danmaku/aliases.json')); print(len(d))" 2>/dev/null \
         || echo 0)
    echo "  cached matches: $nm  aliases: $na"
  fi

  section "wrappers + launchers"
  for f in "$WRAPPER_DIR/mpv-conda" "$WRAPPER_DIR/jellyfin-mpv-shim" \
           "$APPS_DIR/mpv-conda.desktop" "$APPS_DIR/jellyfin-mpv-shim.desktop" \
           "$AUTOSTART_DIR/jellyfin-mpv-shim.desktop"; do
    [[ -e "$f" ]] && echo "  ${f/#$HOME/~}" || echo "  ${f/#$HOME/~} (missing)"
  done

  section "TRT engine cache"
  local cache_dir=""
  for d in "$ENV_PREFIX"/lib/python*; do
    [[ -L "$d" ]] && continue
    [[ -d "$d/site-packages/vsrife/models" ]] && cache_dir="$d/site-packages/vsrife/models" && break
  done
  if [[ -n "$cache_dir" ]]; then
    local n
    n=$(find "$cache_dir" -maxdepth 1 -name "*.ts" 2>/dev/null | wc -l)
    echo "  $cache_dir/"
    echo "  $n compiled TRT engine(s):"
    find "$cache_dir" -maxdepth 1 -name "*.ts" -printf "    %f (%s bytes)\n" 2>/dev/null
  else
    echo "  (no engine cache yet — will be built on first playback)"
  fi

  section "GNOME tray icon support"
  local ext_cmd ext_list
  if command -v gnome-extensions >/dev/null 2>&1; then
    ext_list=$(gnome-extensions list 2>/dev/null | grep -iE 'appindicator' || true)
    if [[ -n "$ext_list" ]]; then
      echo "  AppIndicator extension installed:"
      echo "$ext_list" | sed 's/^/    /'
      gnome-extensions info "$ext_list" 2>/dev/null | grep -E '^State:' | sed 's/^/    /' || true
    else
      echo "  AppIndicator extension NOT installed — tray icon won't show"
    fi
  fi
}

# ============================================================================
# uninstall — remove everything except apt packages and miniforge itself
# ============================================================================
cmd_uninstall() {
  section "uninstall"
  log "stopping any running shim/mpv processes (from this env)"
  pkill -f "$ENV_PREFIX/bin/jellyfin-mpv-shim" 2>/dev/null || true
  pkill -f "$ENV_PREFIX/bin/mpv" 2>/dev/null || true
  pkill -f "$WRAPPER_DIR/jellyfin-mpv-shim" 2>/dev/null || true
  pkill -f "$WRAPPER_DIR/mpv-conda" 2>/dev/null || true
  sleep 1

  # Selective cleanup: preserve user-supplied state across reinstalls.
  # Specifically keep dandanplay AppId (registration takes 1-3 days) and
  # Jellyfin server credentials (so the user doesn't have to re-pair).
  log "removing files we created in $MPV_CFG_DIR (preserving user creds)"
  for f in mpv.conf input.conf rife.vpy rife-light.vpy \
           danmaku-config.json danmaku-credentials.json.example; do
    rm -f "$MPV_CFG_DIR/$f"
  done
  rm -rf "$MPV_CFG_DIR/scripts" "$MPV_CFG_DIR/shaders"
  # Preserved (not deleted):
  #   $MPV_CFG_DIR/danmaku-credentials.json   ← dandanplay AppId/Secret
  #   $MPV_CFG_DIR/danmaku-settings.json      ← user's panel choices
  rmdir "$MPV_CFG_DIR" 2>/dev/null && \
      log "  (config dir was empty, removed it)" || \
      log "  preserved: $(ls "$MPV_CFG_DIR" 2>/dev/null | tr '\n' ' ')"

  log "removing our symlinks in $SHIM_CFG_DIR (preserving cred.json + user prefs)"
  for f in mpv.conf input.conf rife.vpy rife-light.vpy \
           danmaku-config.json danmaku-credentials.json danmaku-settings.json \
           shaders scripts; do
    if [[ -L "$SHIM_CFG_DIR/$f" ]]; then
      rm -f "$SHIM_CFG_DIR/$f"
    fi
  done
  # Preserved (not deleted):
  #   $SHIM_CFG_DIR/cred.json    ← Jellyfin server URL + access token
  #   $SHIM_CFG_DIR/conf.json    ← shim's own settings (audio device, fs, etc.)
  #     (we wrote mpv_ext_path into it, but leaving the stale path doesn't
  #      break anything — reinstall rewrites it)
  if [[ -d "$SHIM_CFG_DIR" ]]; then
    log "  preserved: $(ls "$SHIM_CFG_DIR" 2>/dev/null | tr '\n' ' ')"
  fi

  log "removing wrappers"
  rm -f "$WRAPPER_DIR/mpv-conda" "$WRAPPER_DIR/jellyfin-mpv-shim"
  # Legacy single-file helper from earlier dgxspark builds (the new
  # bundle keeps the helper inside scripts/dandanplay/ instead).
  rm -f "$WRAPPER_DIR/danmaku_helper.py"

  # Delegate danmaku uninstall to the danmaku project's own installer
  # (it preserves cache, credentials, and user-modified settings).
  if [[ -f "$DANMAKU_SRC_DIR/install.py" ]]; then
    log "running danmaku uninstall"
    in_env python3 "$DANMAKU_SRC_DIR/install.py" --uninstall 2>&1 | sed 's/^/  /'
  else
    # Fallback: tear down the bundle dir directly if the source tree
    # was already removed (e.g. user deleted ~/src/).
    rm -rf "$MPV_CFG_DIR/scripts/dandanplay"
  fi

  log "removing danmaku cache (preserving matches.json + offsets.json + aliases.json — re-install uses them)"
  if [[ -d "$HOME/.cache/mpv-danmaku" ]]; then
    find "$HOME/.cache/mpv-danmaku" -mindepth 1 -maxdepth 1 \
        ! -name 'matches.json' ! -name 'offsets.json' ! -name 'aliases.json' \
        -exec rm -rf {} +
  fi

  log "removing .desktop entries + autostart"
  rm -f "$APPS_DIR/mpv-conda.desktop" "$APPS_DIR/jellyfin-mpv-shim.desktop"
  rm -f "$AUTOSTART_DIR/jellyfin-mpv-shim.desktop"

  log "removing icons we installed"
  local size
  for size in 16x16 32x32 48x48 64x64 128x128 256x256; do
    rm -f "$ICON_ROOT/$size/apps/jellyfin-mpv-shim.png"
    rm -f "$ICON_ROOT/$size/apps/mpv-conda.png"
  done
  rm -f "$ICON_ROOT/scalable/apps/mpv-conda.svg"

  log "removing conda env (keeping miniforge itself)"
  if [[ -x "$FORGE_DIR/bin/conda" ]]; then
    # shellcheck disable=SC1091
    source "$FORGE_DIR/etc/profile.d/conda.sh"
    if conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$ENV_NAME"; then
      mamba env remove -n "$ENV_NAME" -y 2>&1 | tail -3 || true
    fi
  fi

  log "removing mpv source tree at $MPV_SRC_DIR"
  rm -rf "$MPV_SRC_DIR"

  log "removing danmaku source tree at $DANMAKU_SRC_DIR"
  rm -rf "$DANMAKU_SRC_DIR"

  update-desktop-database "$APPS_DIR" 2>/dev/null || true
  gtk-update-icon-cache "$ICON_ROOT" 2>/dev/null || true

  cat <<EOF

================================================================
  Uninstall complete.

  KEPT (so re-install is fast and you don't lose state):
    - apt packages          (system-wide; remove with apt manually)
    - miniforge at $FORGE_DIR  (other envs may use it)
    - ~/.condarc, ~/.config/pip/pip.conf  (USTC mirror config)
    - ~/.config/mpv/danmaku-credentials.json  (dandanplay AppId/AppSecret)
    - ~/.config/mpv/danmaku-settings.json     (your panel preferences)
    - ~/.config/jellyfin-mpv-shim/cred.json   (Jellyfin server URL + token)
    - ~/.config/jellyfin-mpv-shim/conf.json   (shim preferences — fullscreen,
                                               audio device, key bindings...)

  Re-run \`$0 install\` to rebuild from scratch — these are picked back up.
================================================================
EOF
}

# ============================================================================
# Dispatcher
# ============================================================================
case "${1:-}" in
  install)   shift; cmd_install "$@" ;;
  status)    cmd_status ;;
  uninstall) cmd_uninstall ;;
  ""|-h|--help)
    cat <<EOF
$(basename "$0") — dgxspark-jellyfin-mpv-rife

Usage:
  $0 install [flags]   install everything on a clean DGX Spark + GNOME system
  $0 status            show what's currently installed and where
  $0 uninstall         remove everything we installed except apt + miniforge

Install flags (all optional, defaults preserve original behavior):
  --no-mirrors   Don't write USTC mirrors to ~/.condarc + pip.conf
                 (use outside China)
  --no-danmaku   Skip the danmaku (bullet-chat) plugin step

See README.md for details.
EOF
    ;;
  *)
    fatal "unknown command: $1 (try: install / status / uninstall)"
    ;;
esac
