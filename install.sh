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
#   ./install.sh install [--no-mirrors] [--no-danmaku] [--rebuild-trt]
#                        [--set-default-video]
#                          full install on a clean system
#                          --no-mirrors: skip USTC mirror config
#                                        (use outside China)
#                          --no-danmaku: skip the danmaku plugin step
#                          --rebuild-trt: wipe TRT engine cache before warm
#                          --set-default-video: register mpv-conda as the
#                                        GNOME default video player for the
#                                        common video MIME types
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

# Python stack pins. Bumping any of these is a deliberate act: re-run
# install, re-check that apply_patches() still reports "patched … for
# TRT mixed-precision" (it hard-fails if the patch context vanished),
# and re-run bench/fps.sh + bench/color.sh.
TORCH_VERSION="2.12.0"
VSRIFE_VERSION="5.7.0"
TENSORRT_VERSION="10.16.1.11"
TORCH_TRT_VERSION="2.12.0"
SHIM_VERSION="2.10.0"
CUDNN_FE_VERSION="1.24.0"

# Miniforge installer pin + checksum (aarch64). `releases/latest` made
# every fresh install a moving target. Find sha256 values in the
# release's *.sha256 assets when bumping.
MINIFORGE_VERSION="26.3.2-3"
MINIFORGE_SHA256="2c113a69297e612b01ca0f320c22a3107a11f2ab9b573d79ac868a175945ce29"

# Danmaku plugin — fetched from a sibling project, installed via its own
# install.py. Pinned to a commit for reproducibility (an upstream push
# used to change what a re-install produced); bump after testing.
DANMAKU_REPO_URL="https://github.com/Cryspia/mpv-dandanplay-danmaku.git"
DANMAKU_SRC_DIR="$HOME/src/mpv-dandanplay-danmaku"
DANMAKU_REF="1d0e5061ebfce3ca3a2a0c65d95797509dc027b2"

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
REBUILD_TRT=0
SET_DEFAULT_VIDEO=0

# Common video MIME types we promote mpv-conda to default for, when
# --set-default-video is passed. Subset of the full MimeType list in
# mpv-conda.desktop — these are the ones GNOME Files / nautilus-open
# consults when the user double-clicks a video. Setting all of them
# means mpv-conda owns the obvious cases (mp4, mkv, webm, etc.) and
# the long tail (ms-asf, vivo, divx, …) inherits via the desktop
# file's MimeType= registration without us having to enumerate.
DEFAULT_VIDEO_MIMES=(
  video/mp4
  video/x-matroska
  video/webm
  video/quicktime
  video/x-msvideo
  video/mpeg
  video/x-m4v
  video/x-flv
  video/3gpp
  video/x-ms-wmv
  video/ogg
  video/x-ms-asf
  application/x-matroska
  application/vnd.apple.mpegurl
  application/x-mpegURL
)

# fsrcnnx-cudnn release tag pulled by install_configs. Bumping this
# fetches a different bundle from
# https://github.com/Cryspia/fsrcnnx-cudnn/releases/download/<tag>/fsrcnnx-cudnn-bundle.tar.gz
FSRCNNX_CUDNN_VERSION="v0.2.2"
# sha256 of the release asset above — recompute when bumping the tag.
FSRCNNX_CUDNN_SHA256="b32bd5c7ec668f31e27d3daa7091f3596706078d6ecad135bd3773ca64b2708a"

# Dual-machine layout. The host's mpv reads DUAL_* from this file (via
# the mpv-conda wrapper); the secondary box's systemd worker service
# reads it via EnvironmentFile.
DUAL_CFG_DIR="$HOME/.config/dgxspark-mpv"
DUAL_CFG_FILE="$DUAL_CFG_DIR/dual.conf"
DUAL_WORKER_DIR="$HOME/.local/share/dgxspark-mpv/worker"
# Old install location, retained for migration / uninstall cleanup
# of installs predating the move to ~/.local/share/.
DUAL_WORKER_DIR_LEGACY="$HOME/dual_machine"
DUAL_WORKER_SERVICE="dgxspark-dual-worker.service"
DUAL_TRAY_DESKTOP="dgxspark-dual-secondary.desktop"
DUAL_TRAY_AUTOSTART="$HOME/.config/autostart/$DUAL_TRAY_DESKTOP"

# Default cluster config — used as fallback when neither env nor
# interactive input provides a value. Override per-install via
# DUAL_HOST_IP / DUAL_WORKER_IP / DUAL_RDMA_DEV / etc env vars.
DUAL_DEFAULT_RDMA_PORT=29900
DUAL_DEFAULT_LIVENESS_PORT=29905

# Install-mode flags (set by cmd_install argv parsing). At most one of
# the two dual flags can be set at a time.
INSTALL_DUAL_HOST=0
INSTALL_DUAL_SECONDARY=0

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
  # A missing env must be loud — silently falling through used to run
  # the command against base/system python instead.
  conda activate "$ENV_NAME" >/dev/null 2>&1 \
    || fatal "conda env '$ENV_NAME' not found — run install first"
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
    log "downloading + installing miniforge $MINIFORGE_VERSION to $FORGE_DIR"
    cd /tmp
    local mf_installer="Miniforge3-${MINIFORGE_VERSION}-Linux-aarch64.sh"
    curl -fL --retry 3 -O \
      "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${mf_installer}"
    echo "$MINIFORGE_SHA256  $mf_installer" | sha256sum -c - \
      || fatal "miniforge installer checksum mismatch — refusing to run it"
    bash "$mf_installer" -b -p "$FORGE_DIR"
    rm -f "$mf_installer"
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
    `# media + filter chain (pinned: the vapoursynth C ABI + ffmpeg` \
    `#  major drive the whole vf chain; bump deliberately + re-bench)` \
    'vapoursynth=72' 'ffmpeg=8.1' \
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

  # If a previous build already produced mpv at the pinned version with
  # the right features, skip. The version gate matters: without it,
  # bumping MPV_VERSION and re-running install silently kept the old
  # binary (the feature list alone always matched).
  if "$ENV_PREFIX/bin/mpv" --version 2>/dev/null | head -1 \
       | grep -q "mpv ${MPV_VERSION#v}" && \
     "$ENV_PREFIX/bin/mpv" -v 2>&1 | sed -n 's/.*enabled features:[[:space:]]*//p' | head -1 \
       | { read -r feats; for f in vapoursynth wayland x11 vulkan lua; do
             grep -qw "$f" <<<"$feats" || exit 1; done; } 2>/dev/null; then
    log "mpv ${MPV_VERSION} already built with vapoursynth + wayland + x11 + vulkan + lua"
    return
  fi

  mkdir -p "$(dirname "$MPV_SRC_DIR")"
  if [[ ! -d "$MPV_SRC_DIR/.git" ]]; then
    log "cloning mpv $MPV_VERSION"
    git clone --depth 1 --branch "$MPV_VERSION" \
      https://github.com/mpv-player/mpv.git "$MPV_SRC_DIR"
  else
    # Existing shallow checkout may predate a bumped MPV_VERSION —
    # fetch the pinned tag and hard-switch to it.
    log "updating mpv checkout to $MPV_VERSION"
    git -C "$MPV_SRC_DIR" fetch --depth 1 origin tag "$MPV_VERSION" || true
    git -C "$MPV_SRC_DIR" checkout --quiet "$MPV_VERSION" || \
      fatal "cannot checkout mpv $MPV_VERSION in $MPV_SRC_DIR"
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
    pip install "torch==$TORCH_VERSION" --index-url "$PYTORCH_INDEX" || \
      pip install --pre torch --index-url "$PYTORCH_NIGHTLY_INDEX"
  else
    log "torch already installed: $(python -c 'import torch; print(torch.__version__)')"
  fi

  # Pinned application stack, one resolver run. Pins matter doubly here:
  # (a) reproducible installs; (b) apply_patches() sed-patches vsrife by
  # exact text match — an unpinned re-run could silently float vsrife to
  # a release where the patch no longer applies (= fp16 flow overflow
  # flicker with only a soft log line to show for it). Skip entirely
  # when every pin is already satisfied so re-runs are offline-fast.
  # Bump deliberately, re-test, and re-verify the vsrife patch context.
  if python - <<PY
import importlib.metadata as md, sys
pins = {"vsrife": "$VSRIFE_VERSION", "tensorrt": "$TENSORRT_VERSION",
        "torch_tensorrt": "$TORCH_TRT_VERSION",
        "jellyfin-mpv-shim": "$SHIM_VERSION",
        "nvidia-cudnn-frontend": "$CUDNN_FE_VERSION"}
try:
    sys.exit(0 if all(md.version(p) == v for p, v in pins.items()) else 1)
except md.PackageNotFoundError:
    sys.exit(1)
PY
  then
    log "python stack already at pinned versions — skipping pip"
  else
    pip install \
      "vsrife==$VSRIFE_VERSION" \
      "jellyfin-mpv-shim[gui]==$SHIM_VERSION" \
      "tensorrt==$TENSORRT_VERSION" \
      "torch_tensorrt==$TORCH_TRT_VERSION" \
      `# fsrcnnx-cudnn imports the Python frontend ('import cudnn') at` \
      `# runtime — distinct from the runtime library nvidia-cudnn-cu*.` \
      "nvidia-cudnn-frontend==$CUDNN_FE_VERSION"
  fi

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

    # jellyfin-mpv-shim's `mpv_options["osc"] = False` is INTENTIONAL and
    # works correctly on mpv 0.41 — verified via `mpv --no-osc` accepted
    # + `mpv.MPV(osc=False).osc == False` round-tripped. An earlier patch
    # here mistakenly rewrote it to `script_opts="osc-visibility=auto"`,
    # which kept mpv's built-in osc.lua loaded ALONGSIDE shim's
    # trickplay-osc.lua → both scripts force-bind the "input" section →
    # trickplay-osc's mouse bindings get shadowed → seekbar/play/pause
    # unclickable (with hover thumbnails missing if you then disabled
    # trickplay-osc to fix the click bug). Net: don't patch this. The
    # shim's original line correctly disables the built-in OSC so
    # trickplay-osc owns the "input" section uncontested. If a future
    # mpv release actually removes `--osc`, revisit then. (Reverted in
    # the commit that ships this comment.)
    f="$pydir/site-packages/jellyfin_mpv_shim/player.py"
    if [[ -f "$f" ]]; then
      # If a prior install of this script applied the bad sed, restore
      # the original line so trickplay-osc starts working again. cp -n
      # in the old patch saved player.py.bak before mangling.
      if [[ -f "${f}.bak" ]] && grep -q '"script_opts"\] = "osc-visibility' "$f"; then
        cp -f "${f}.bak" "$f"
        log "restored $f from .bak (undoing prior bad OSC patch)"
      fi
      # Same idea for the second bad patch (gated trickplay on
      # thumbnail_osc_builtin) — restoring .bak above already covers it
      # since .bak predates both seds; .bak2 (from the second patch)
      # already has the first bad patch applied, so prefer .bak.
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
      fi
      # Verify the patch actually landed. The old "already patched (or
      # different version)" message conflated success with a vsrife
      # release where the sed context no longer matches — and a missing
      # patch means fp16 flow-vector overflow flicker at playback time
      # with nothing pointing back here. Fail loudly instead.
      grep -q 'enabled_precisions={torch.float16, torch.float32}' "$f" \
        || fatal "vsrife mixed-precision patch did NOT apply to $f — \
vsrife $VSRIFE_VERSION layout changed; update apply_patches()"
      log "vsrife mixed-precision patch verified in $f"
    fi
  done
}

# ============================================================================
# Step 7: write user-facing config files (mpv.conf, input.conf, rife.vpy,
#         sr_keys.lua, sr_keys_helper.py) and fetch the upstream
#         fsrcnnx-cudnn release bundle (Python pkg + .npz weights).
# ============================================================================
install_configs() {
  section "step 7/10: configs + fsrcnnx-cudnn bundle"
  mkdir -p "$MPV_CFG_DIR/scripts"

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
# auto-copy-safe (not auto-safe): the [rife] profile keeps a vapoursynth
# filter attached for ≤4K sources, and vf chains need software frames —
# under auto-safe mpv silently falls back to pure software decode in
# that configuration. copy-back NVDEC keeps the decoder on the GPU; on
# GB10 unified memory the copy is cheap, and the freed CPU/membw goes
# to the RIFE chain (matters most for 4K HEVC/AV1 sources).
hwdec=auto-copy-safe
# gpu-hq is a deprecated alias since mpv 0.40 — high-quality is the
# successor profile (same intent; scale/dscale/deband overridden below).
profile=high-quality

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

# Network streaming (Jellyfin via shim): deeper demuxer cache than the
# ~150 MiB default. The filter chain makes seeks/underruns expensive
# (vf rebuild = 1-3 s freeze), and 4K remux bitrates (60-100 Mbps)
# drain the default forward window in seconds. ~1 GiB resident worst
# case — noise against 128 GB unified memory.
cache=yes
demuxer-max-bytes=768MiB
demuxer-max-back-bytes=256MiB
demuxer-readahead-secs=30

# Danmaku: our generated ASS encodes every comment with explicit \move
# \pos \1c overrides. mpv's secondary-sub-ass-override defaults to
# "strip" which throws those tags away — collapsing all comments to a
# top-stack. "no" tells mpv to honor the script verbatim. secondary-
# sub-pos=0 prevents an additional vertical shift.
secondary-sub-ass-override=no
secondary-sub-pos=0

# FSRCNNX SR is now done inside vapoursynth (chained after rife_yuv in
# rife.vpy via fsrcnnx_cudnn.vsfunc.fsrcnnx_yuv_auto) — the legacy
# FSRCNNX GLSL is intentionally disabled to avoid double-upscale.
#
# KrigBilateral.glsl: luma-guided kriging chroma upsampler (Shiandow).
# Runs in mpv's display chroma stage — orthogonal to the in-vapoursynth
# krig: that one upsamples src chroma to luma dim BEFORE FSRCNNX; this
# one upsamples the FSRCNNX-output chroma (or any 4:2:0 source's chroma)
# to display luma dim BEFORE the YUV→RGB matrix. Quality > the default
# bilinear chroma scaler. Shift+F8 toggles at runtime.
glsl-shaders=~~/shaders/KrigBilateral.glsl

# RIFE + FSRCNNX. Single .vpy. F8 toggles SR (single: cycle variant /
# dual: on/off). F9 toggles INTERP (single: on/off / dual: cycle
# 4 -> 3 -> 2 -> 1[SR-only] -> 4). See scripts/sr_keys.lua and
# scripts/warmup.lua for the keybinds + cold-start pre-roll.
#
#   h ≤ 720         RIFE 4.26 + FSRCNNX family=16-layer (auto x3/x4)
#   720 < h ≤ 1080  RIFE 4.6  + FSRCNNX family=8-layer  (auto x2_8)
#   1080 < h ≤ 2160 mixed mode: original 4K real frames passthrough,
#                   interp frames go through downsample → RIFE 4.26 →
#                   FSRCNNX 16-layer → upsample back to 4K
#   fps > 30        RIFE skipped, FSRCNNX still runs
#
# `or 0` sentinel: at the first profile-cond evaluation (before the
# demuxer fills video-params/h) the property is nil; defaulting to 0
# fails the gate so the vf isn't briefly attached then ripped off.
# `concurrent-frames=24 / buffered-frames=12` — must be ≥ ~16 or mpv's
# vsapi per-call overhead (~16 ms gap between consecutive compute_callable
# invocations) caps real-playback throughput at ~30 fps regardless of how
# fast the filter chain itself is. With CF=24 mpv saturates at its
# internal cap of 20 concurrent requests, which matches the bench config
# the steady-state numbers in docs/performance.md were measured under.
# bf=12 (not 24): a higher buffered-frames pushes more dual-pipeline
# slots in flight, which pressures GPU memory and (empirically) makes
# the dual chain's intermittent mpv crash fire sooner; bf=12 was the
# stable value. At 4K source bf>16 also costs ~6 fps of steady
# throughput. bf=12 keeps a healthy ~200 ms pre-buffer for smoothness
# without those downsides.
[rife]
profile-cond=0<(p["video-params/h"] or 0) and (p["video-params/h"] or 0)<=2160
profile-restore=copy-equal
vf=vapoursynth=~~/rife.vpy:buffered-frames=12:concurrent-frames=24
EOF
  log "wrote $MPV_CFG_DIR/mpv.conf"

  # input.conf — F8 / F9 are bound by scripts/sr_keys.lua (cycle FSRCNNX
  # variant / toggle RIFE). Header here is just documentation; the keys
  # themselves come from the lua script.
  cat > "$MPV_CFG_DIR/input.conf" <<'EOF'
# Keybinds bound by scripts/sr_keys.lua:
#   F8:        SR toggle.
#                * single mode: cycle FSRCNNX variant (x4_16 → x3_16 →
#                  x2_16 → x2_8 → off → loop). Per-file.
#                * dual mode:   on/off only (the bucket auto-picks
#                  variant). 4K-DS source rejects OFF (RIFE-DS needs SR
#                  to upscale back).
#   F9:        interp toggle / multiplier.
#                * single mode: toggle RIFE on/off.
#                * dual mode:   cycle 4 → 3 → 2 → off → 4. OFF disables
#                  the whole dual chain and drops to single SR-only.
#   Shift+F8:  toggle KrigBilateral chroma GLSL (display-stage chroma
#              upsample). Persists across files. Instant — no vf reload.
#   Shift+F9:  toggle dual-machine offload (no-op when DUAL_WORKER_HOST
#              isn't configured). Persists across files.
#
# F8 / F9 / Shift+F9 force a vapoursynth filter reload (~1–3 s freeze)
# because there's no runtime parameter switch inside our chain — we
# rebuild it. Shift+F8 only flips an mpv property and is instant.
EOF
  log "wrote $MPV_CFG_DIR/input.conf"

  # sr_keys.lua — keybind handler for F8 / F9 / Shift+F8 / Shift+F9.
  # See its own header for the full mapping; install.sh just installs.
  cp -f "$PROJECT_DIR/scripts/sr_keys.lua" "$MPV_CFG_DIR/scripts/sr_keys.lua"
  log "copied scripts/sr_keys.lua to $MPV_CFG_DIR/scripts/"

  # warmup.lua — pause + pre-render 1 s of frames at file-loaded so the
  # cold cuDNN / RIFE / FSRCNNX graphs land before playback starts.
  # Both single and dual paths benefit. WARMUP_DISABLE=1 in env → no-op.
  cp -f "$PROJECT_DIR/scripts/warmup.lua" "$MPV_CFG_DIR/scripts/warmup.lua"
  log "copied scripts/warmup.lua to $MPV_CFG_DIR/scripts/"

  # dual_seek_flush.lua — bumps /tmp/dual_machine_seek_epoch on every
  # mpv seek so the dual-machine dispatcher's epoch watcher can flush
  # stale in-flight queue_mgr state. Without this, any seek freezes
  # dual-mode mpv for 30-40 s while wait_phase_done sits on an mpv VA
  # gate that the seek made unreachable. DUAL_SEEK_FLUSH_DISABLE=1 to
  # opt out.
  cp -f "$PROJECT_DIR/scripts/dual_seek_flush.lua" \
        "$MPV_CFG_DIR/scripts/dual_seek_flush.lua"
  log "copied scripts/dual_seek_flush.lua to $MPV_CFG_DIR/scripts/"

  # dual_fps_override.lua — pre-vpy ffprobe sidecar that writes
  # /tmp/dual_machine_mult_override based on real source fps + height.
  # Works around mpv's vsapi rate-propagation bug where some MKV/MP4
  # containers hand fps_num=0/fps_den=0 to vapoursynth, forcing
  # rife.vpy's in-vpy heuristic into the conservative mult=2 branch.
  # DUAL_FPS_OVERRIDE_DISABLE=1 to opt out.
  cp -f "$PROJECT_DIR/scripts/dual_fps_override.lua" \
        "$MPV_CFG_DIR/scripts/dual_fps_override.lua"
  log "copied scripts/dual_fps_override.lua to $MPV_CFG_DIR/scripts/"


  # sr_keys_helper.py — Python helper imported by the rife*.vpy files.
  # Reads the side-channel files written by sr_keys.lua and applies
  # the corresponding override (FSRCNNX variant or RIFE skip). Also
  # writes /tmp/fsrcnnx_active_variant so the lua F8 cycle knows
  # where it is.
  cp -f "$PROJECT_DIR/sr_keys_helper.py" "$MPV_CFG_DIR/"
  log "copied sr_keys_helper.py to $MPV_CFG_DIR/"

  # vs_gpu_helpers.py — provides rife_yuv() that takes YUV420P10 directly
  # and runs YUV↔RGB on GPU instead of the CPU bicubic round-trip. On 4K
  # this saves ~30 ms / frame, turning [rife-half] from "stutters" into
  # "smooth" since the per-frame budget shrinks below 60 fps display.
  cp -f "$PROJECT_DIR/vs_gpu_helpers.py" "$MPV_CFG_DIR/"
  log "copied vs_gpu_helpers.py to $MPV_CFG_DIR/"

  # rife.vpy — unified RIFE + FSRCNNX pipeline. Internal branch on
  # clip.height picks the right RIFE model and FSRCNNX family.
  # F8 / F9 keybinds (scripts/sr_keys.lua) override variant / RIFE-on
  # at runtime via /tmp/fsrcnnx_variant and /tmp/rife_disabled.
  cp -f "$PROJECT_DIR/dual_machine/rife.vpy" "$MPV_CFG_DIR/"
  log "wrote $MPV_CFG_DIR/rife.vpy"

  # dual_machine/*.py + native_filter — host-only. rife.vpy (run by
  # mpv) adds $MPV_CFG_DIR/dual_machine to sys.path and imports
  # native_dispatcher / queue_mgr / etc. The native_filter directory
  # holds a vapoursynth C++ plugin (libdgxspark_split_dual.so) that
  # rife.vpy LoadPlugin()s when dual mode is active. The secondary
  # box doesn't run mpv → skip the whole block there.
  if (( INSTALL_DUAL_SECONDARY == 0 )); then
    local native_so="$PROJECT_DIR/dual_machine/native_filter/build/libdgxspark_split_dual.so"
    if [[ ! -f "$native_so" ]]; then
      section "step 7b/10: cmake build native_filter"
      # pybind11 is a build-time dep for the .so — pip-install into the
      # conda env if missing (mamba create doesn't pull it in by default).
      in_env bash -c '
        python -c "import pybind11" 2>/dev/null \
          || pip install --quiet pybind11
      ' || warn "pybind11 install failed; cmake may not find it"
      in_env bash -c "
        cd '$PROJECT_DIR/dual_machine/native_filter'
        mkdir -p build && cd build
        cmake -DCMAKE_BUILD_TYPE=Release .. >/tmp/native_filter_cmake.log 2>&1
        make -j$(nproc) >>/tmp/native_filter_cmake.log 2>&1
      " || fatal "native_filter build failed — see /tmp/native_filter_cmake.log"
      [[ -f "$native_so" ]] || fatal "build claimed success but $native_so is missing"
      log "built $(ls -la "$native_so" | awk '{print $5}') bytes"
    fi
    install -d "$MPV_CFG_DIR/dual_machine" "$MPV_CFG_DIR/dual_machine/native_filter"
    # --exclude='vs_gpu_helpers.py': a dev checkout may carry a
    # local absolute-path symlink there (gitignored); rsync -a would
    # ship the symlink itself, which dangles on any other machine.
    # The real module is installed from the repo root right below.
    rsync -a --delete --exclude='__pycache__' --exclude='native_filter' \
          --exclude='vk_priority_layer' --exclude='vs_gpu_helpers.py' \
          --exclude='*.sh' --exclude='*.md' --exclude='*.in' --exclude='rife.vpy' \
          "$PROJECT_DIR/dual_machine/" "$MPV_CFG_DIR/dual_machine/"
    install -m 0644 "$PROJECT_DIR/vs_gpu_helpers.py" \
            "$MPV_CFG_DIR/dual_machine/vs_gpu_helpers.py"
    rsync -a --delete --exclude='__pycache__' \
          "$PROJECT_DIR/dual_machine/native_filter/" \
          "$MPV_CFG_DIR/dual_machine/native_filter/"
    log "wrote $MPV_CFG_DIR/dual_machine/ ($(ls "$MPV_CFG_DIR/dual_machine"/*.py 2>/dev/null | wc -l) py modules + native_filter/)"

    # step 7c: VK_LAYER_PRIORITY_BOOST. On a single GB10 the Vulkan
    # present pass time-shares the GPU with the dual chain's CUDA work
    # (RIFE post-flownet + FSRCNNX + krig + CCSR) and loses enough
    # vsyncs to drag a 60 Hz panel down to ~50 fps on 1080p24-x3
    # content. mpv/libplacebo doesn't expose VK_EXT_global_priority,
    # so we ship a small instance layer that injects
    # VkDeviceQueueGlobalPriorityCreateInfoKHR=HIGH at vkCreateDevice;
    # combined with `setcap cap_sys_nice+ep` on the mpv binary this
    # restores ~+10% display fps. Layer manifest + .so go to
    # /usr/local/lib + /usr/share/vulkan because the Vulkan loader
    # ignores user-local paths under AT_SECURE=1. Both are reverted
    # in dual_uninstall_pieces.
    section "step 7c/10: VK_LAYER_PRIORITY_BOOST (Vulkan present-queue priority)"
    if command -v sudo >/dev/null \
       && [[ -x "$PROJECT_DIR/dual_machine/vk_priority_layer/build.sh" ]]; then
      MPV_BIN="$ENV_PREFIX/bin/mpv" \
      VK_INC="$ENV_PREFIX/include" \
        bash "$PROJECT_DIR/dual_machine/vk_priority_layer/build.sh" install \
          2>&1 | sed 's/^/  /' \
        || warn "vk_priority_layer install failed — playback will still work, just at the baseline contended fps"
    else
      warn "skipping vk_priority_layer (no sudo or build.sh missing)"
    fi
  fi

  # Old per-band .vpy files / legacy FSRCNNX glsl shaders / in-tree
  # weights / old bundle location under scripts/ have all been
  # superseded — clean stale copies from previous installs. The old
  # `scripts/fsrcnnx-cudnn/` path caused mpv to log "Cannot find main.*
  # in scripts/<subdir>" on every startup because mpv's multi-file-
  # script convention wants `main.{lua,js,py,mjs}` as the entry, which
  # the bundle doesn't ship. We host the bundle outside `scripts/` now
  # to dodge that.
  rm -f  "$MPV_CFG_DIR/rife-light.vpy" "$MPV_CFG_DIR/rife-half.vpy"
  rm -rf "$MPV_CFG_DIR/weights" \
         "$MPV_CFG_DIR/fsrcnnx_cudnn" "$MPV_CFG_DIR/scripts/fsrcnnx-cudnn"
  rm -f  "$MPV_CFG_DIR/shaders/FSRCNNX_"*.glsl 2>/dev/null

  # Shaders: install KrigBilateral.glsl (Shiandow's luma-guided chroma
  # upsampler, vendored under shaders/). mpv.conf points glsl-shaders
  # at this path by default; Shift+F8 toggles at runtime.
  mkdir -p "$MPV_CFG_DIR/shaders"
  if compgen -G "$PROJECT_DIR/shaders/*.glsl" > /dev/null; then
    cp -f "$PROJECT_DIR/shaders/"*.glsl "$MPV_CFG_DIR/shaders/"
    log "copied $(ls "$PROJECT_DIR/shaders/"*.glsl | wc -l) glsl shader(s) to $MPV_CFG_DIR/shaders/"
  fi

  # FSRCNNX cuDNN super-resolution: pull the upstream release bundle
  # (Python pkg + .npz weights) and extract to ~/.config/mpv/. Pinned
  # by tag so re-installing reproduces a known-good version. See
  # https://github.com/Cryspia/fsrcnnx-cudnn for the source / weights /
  # benchmarks — fsrcnnx_yuv_auto, family, ratio gating etc. all live
  # there.
  local fsrcnnx_bundle_url="https://github.com/Cryspia/fsrcnnx-cudnn/releases/download/${FSRCNNX_CUDNN_VERSION}/fsrcnnx-cudnn-bundle.tar.gz"
  local fsrcnnx_dir="$MPV_CFG_DIR/fsrcnnx-cudnn"
  if [[ -f "$fsrcnnx_dir/.installed-version" ]] && \
     [[ "$(cat "$fsrcnnx_dir/.installed-version" 2>/dev/null)" == "$FSRCNNX_CUDNN_VERSION" ]]; then
    log "fsrcnnx-cudnn $FSRCNNX_CUDNN_VERSION already installed at $fsrcnnx_dir/"
  else
    log "fetching fsrcnnx-cudnn $FSRCNNX_CUDNN_VERSION bundle from GitHub releases"
    local tmp_bundle="/tmp/fsrcnnx-cudnn-bundle-$$.tar.gz"
    curl -fsSL -o "$tmp_bundle" "$fsrcnnx_bundle_url" || \
      fatal "failed to download $fsrcnnx_bundle_url"
    echo "$FSRCNNX_CUDNN_SHA256  $tmp_bundle" | sha256sum -c - \
      || fatal "fsrcnnx-cudnn bundle checksum mismatch — refusing to extract"
    rm -rf "$fsrcnnx_dir"
    tar -xzf "$tmp_bundle" -C "$MPV_CFG_DIR/"
    rm -f "$tmp_bundle"

    # Strip the upstream stand-alone entry points. We use the Python
    # package + weights directly from rife.vpy / sr_keys_helper.py;
    # the lua + companion .vpy are for fsrcnnx-cudnn's solo install
    # mode (which `vf-add`s its own vapoursynth filter — would
    # double-stack on top of our rife.vpy if it ever fired).
    # Both names handled: pre-v0.1.1 bundle had `fsrcnnx_auto.lua`,
    # v0.1.1+ renamed it to `main.lua`.
    rm -f "$fsrcnnx_dir/main.lua" \
          "$fsrcnnx_dir/fsrcnnx_auto.lua" \
          "$fsrcnnx_dir/fsrcnnx_sr.vpy"

    echo "$FSRCNNX_CUDNN_VERSION" > "$fsrcnnx_dir/.installed-version"
    log "installed fsrcnnx-cudnn → $fsrcnnx_dir/ (package + weights only)"
  fi
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
  section "step 8/10: warm TRT engine cache (480p / 720p / 1080p shapes)"
  # shellcheck disable=SC1091
  source "$FORGE_DIR/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"

  # If --rebuild-trt was passed, wipe every cached engine first so the
  # warm step rebuilds from scratch. Use case: NVIDIA driver / CUDA /
  # TensorRT was upgraded in place and the existing engines (which
  # match by GPU model + TRT version but not driver build) now produce
  # corrupt frames. vsrife happily uses them anyway because the cache
  # filename matches; only a forced rebuild fixes it.
  local cache_dir
  cache_dir=$(ls -d "$ENV_PREFIX"/lib/python*/site-packages/vsrife/models 2>/dev/null | head -1)
  # Engine backup outside the conda env. uninstall deletes the whole
  # env — taking 4-7 min of TRT compiles with it while carefully
  # preserving credentials; vsrife upgrades change site-packages paths
  # with the same effect. Engine filenames embed GPU model + TRT
  # version, so a flat dir is safe: non-matching engines are simply
  # never opened.
  local engine_backup_dir="$HOME/.cache/dgxspark-mpv/trt-engines"
  if (( REBUILD_TRT )); then
    log "wiping existing TRT engine cache (--rebuild-trt)"
    [[ -d "$cache_dir" ]] &&       find "$cache_dir" -maxdepth 1 -name "*.ts*" -print -delete | sed 's/^/  removed: /'
    [[ -d "$engine_backup_dir" ]] &&       find "$engine_backup_dir" -maxdepth 1 -name "*.ts*" -print -delete | sed 's/^/  removed backup: /'
  fi
  # Restore any backed-up engines before warming — turns a post-
  # uninstall reinstall's warm step into a <1 s cache hit per shape.
  if [[ -d "$engine_backup_dir" && -d "$cache_dir" ]]; then
    local n_restored=0 ef
    for ef in "$engine_backup_dir"/*.ts*; do
      [[ -e "$ef" ]] || continue
      [[ -e "$cache_dir/$(basename "$ef")" ]] && continue
      cp "$ef" "$cache_dir/" && n_restored=$((n_restored + 1))
    done
    (( n_restored )) && log "restored $n_restored TRT engine file(s) from $engine_backup_dir"
  fi

  # vsrife's TRT static-shape mode uses the input clip's actual dimensions
  # to build the engine, so feeding it a dummy at the right resolution
  # produces an engine matching real playback. Cache filenames embed the
  # padded shape (e.g. 1920x1088, 3840x2176) so 1080p and 4K engines
  # coexist independently.
  #
  # Total disk: ~350 MB for all 7 engines. Total time on a clean install:
  # ~4-7 minutes (each engine ~30-60s); cache hits return in <1s.
  python - <<'PY'
import time, vapoursynth as vs
from vsrife import rife

core = vs.core

def warm(label, model, width, height, scale=1.0):
    dummy = core.std.BlankClip(width=width, height=height,
                               format=vs.YUV420P8,
                               length=2, fpsnum=30, fpsden=1)
    dummy = core.resize.Bicubic(dummy, format=vs.RGBH, matrix_in_s="709")
    t0 = time.time()
    clip = rife(dummy, model=model, scale=scale,
                factor_num=2, factor_den=1,
                auto_download=True, trt=True)
    # Force frame request so vsrife actually compiles or loads the engine
    clip.get_frame(0)
    print(f"  {label}: ready in {time.time()-t0:.1f}s")

# 480p sources — bucket 0 runs 4.26 at the source's own resolution,
# so each common SD shape needs its own static-shape engine. Without
# these, the first SD file blocks mpv's filter init for the 30-60 s
# TRT compile with no OSD feedback (looks like a hang).
warm("RIFE 4.26 @ 854x480 (SD 16:9)",  "4.26",  854, 480)
warm("RIFE 4.26 @ 720x480 (NTSC DVD)", "4.26",  720, 480)
warm("RIFE 4.26 @ 640x480 (4:3)",      "4.26",  640, 480)

# 720p / 540p sources — auto-band uses 4.26 (rife.vpy heavy branch).
warm("RIFE 4.26 @ 720p",  "4.26", 1280,  720)

# 1080p sources — cinema rates (<25 fps) use 4.26, 25-30 fps uses 4.6.
warm("RIFE 4.26 @ 1080p", "4.26", 1920, 1080)
warm("RIFE 4.6  @ 1080p", "4.6",  1920, 1080)

# 4K sources — mixed mode downsamples to 1080p then reuses the
# 4.26 @ 1080p engine above (and 4.6 @ 1080p for 25-30 fps sources);
# no separate 4K engine is built or needed.
PY

  # Back up freshly-warmed engines so they survive uninstall / env
  # rebuild / vsrife path changes.
  if [[ -d "$cache_dir" ]]; then
    mkdir -p "$engine_backup_dir"
    local n_backed=0 ef
    for ef in "$cache_dir"/*.ts*; do
      [[ -e "$ef" ]] || continue
      [[ -e "$engine_backup_dir/$(basename "$ef")" ]] && continue
      cp "$ef" "$engine_backup_dir/" && n_backed=$((n_backed + 1))
    done
    (( n_backed )) && log "backed up $n_backed TRT engine file(s) → $engine_backup_dir"
  fi

  # Pre-compile the chroma kriging CUDA extension. Without this the
  # first frame using KrigBilateral chroma (yuv_p10_to_rgb in
  # vs_gpu_helpers + fsrcnnx_yuv's fused path) pays ~18 s of nvcc/ninja
  # JIT mid-playback. Idempotent: the second call short-circuits on the
  # cached .so under ~/.cache/torch_extensions/. Same env + bundle path
  # logic as sr_keys_helper / native_dispatcher.
  log "pre-compiling chroma_krig CUDA extension"
  # Older fsrcnnx-cudnn release tarballs (≤ v0.1.1) ship without
  # chroma_krig.py / csrc/. Falling back to a runtime JIT in that
  # case just costs ~18 s on first frame and otherwise plays fine.
  if ! PYTHONPATH="$MPV_CFG_DIR/fsrcnnx-cudnn:${PYTHONPATH:-}" \
       python - <<'PY'
import time
t0 = time.time()
from fsrcnnx_cudnn.chroma_krig import precompile
precompile()
print(f"  chroma_krig: ready in {time.time()-t0:.1f}s")
PY
  then
    warn "chroma_krig precompile skipped (bundle missing the module — first frame will JIT)"
  fi
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
    git -C "$DANMAKU_SRC_DIR" fetch --quiet origin || true
    git -C "$DANMAKU_SRC_DIR" checkout --quiet "$DANMAKU_REF" || \
      fatal "cannot checkout danmaku ref $DANMAKU_REF"
  else
    log "cloning $DANMAKU_REPO_URL → $DANMAKU_SRC_DIR"
    # DANMAKU_REF is a commit SHA — clone the default branch, then
    # detach to the pin (git clone --branch doesn't accept SHAs).
    git clone --quiet "$DANMAKU_REPO_URL" "$DANMAKU_SRC_DIR"
    git -C "$DANMAKU_SRC_DIR" checkout --quiet "$DANMAKU_REF" || \
      fatal "cannot checkout danmaku ref $DANMAKU_REF"
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
# Force thumbnail_osc_builtin=True (= shim default) so trickplay-osc.lua
# owns the seekbar with hover BIF thumbnails. An earlier install of this
# script set this to False to work around a click-handling bug; that bug
# was actually caused by a bad sed in apply_patches (now removed) that
# kept mpv's built-in osc.lua loaded alongside trickplay-osc.lua, double-
# binding the "input" section and shadowing trickplay-osc's mouse events.
# With shim's original `mpv_options["osc"] = False` restored, the
# built-in OSC is genuinely disabled and trickplay-osc clicks work again.
data["thumbnail_osc_builtin"] = True
# Shim's default remote_kbps=10000 (= 10 Mbps cap) is sent to the
# Jellyfin server as MaxStreamingBitrate in the device profile and
# forces transcode for any > 10 Mbps remote content — typical
# 1080p BD remux is 30-40 Mbps, 4K is 60-100 Mbps; the cap blows
# them all and loses HDR/HDR10+/DV metadata in the process. The
# detection in shim's is_local_domain() flips to "remote" whenever
# Jellyfin is reached via a public domain (reverse proxy / hairpin
# NAT miss), so anyone with `https://jellyfin.example.com` is hit
# by this by default. Mirror local_kbps's effective-unlimited 2147483
# (= 2 Tbps) so direct-play decides the bitrate, not the device
# profile. (Cross-pollinated from windows-jellyfin-mpv-rife.)
#
# setdefault: respect a user-chosen cap if one is already in conf.json
# from a previous in-mpv menu pick (c → Video Preferences → Remote
# Transcode Quality writes to this same file). Only seeds on first
# install or when no prior setting exists; user-set caps survive
# reinstalls.
data.setdefault("remote_kbps", 2147483)
with open(p, "w") as f:
    json.dump(data, f, indent=2)
print("wrote", p)
PY

  # shim insists on its own --config-dir, isolating mpv from ~/.config/mpv/.
  # Symlink the user-facing files back so one set of config rules both.
  local name src dst
  # Clean up symlinks for files that previous installs created but the
  # current layout no longer ships (rife-light.vpy / rife-half.vpy
  # were folded into rife.vpy).
  for name in rife-light.vpy rife-half.vpy; do
    [[ -L "$SHIM_CFG_DIR/$name" ]] && { rm -f "$SHIM_CFG_DIR/$name"; \
      log "removed stale symlink $SHIM_CFG_DIR/$name"; }
  done

  for name in mpv.conf input.conf rife.vpy \
              vs_gpu_helpers.py sr_keys_helper.py \
              danmaku-config.json danmaku-credentials.json danmaku-settings.json; do
    src="$MPV_CFG_DIR/$name"
    dst="$SHIM_CFG_DIR/$name"
    # Idempotency must compare the link TARGET — a dangling or stale
    # link from an older layout used to be skipped just for being a
    # symlink.
    [[ -L "$dst" && "$(readlink "$dst")" == "$src" ]] && continue
    [[ -e "$src" ]] || continue
    rm -f "$dst"
    ln -s "$src" "$dst"
    log "linked $dst → $src"
  done
  for name in scripts shaders; do
    src="$MPV_CFG_DIR/$name"
    dst="$SHIM_CFG_DIR/$name"
    [[ -L "$dst" && "$(readlink "$dst")" == "$src" ]] && continue
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
# finds its stdlib when launched without conda activation. Also
# sources the dual-machine cluster config if --dual-host was installed,
# so DUAL_WORKER_HOST + DUAL_RDMA_* are visible to rife.vpy at
# playback time without any per-user env setup.
export PYTHONHOME="$ENV_PREFIX"
# Put the env's bin first on PATH so torch's cpp_extension JIT finds
# \`ninja\` (conda-installed) when it (re)compiles the chroma_krig CUDA
# kernel on first frame; without it the build aborts with "Ninja is
# required" and krig falls back to a slow per-frame path. nvcc comes
# from the system CUDA toolkit already on PATH.
export PATH="$ENV_PREFIX/bin:\$PATH"
if [[ -f "$DUAL_CFG_FILE" ]]; then
  set -a; . "$DUAL_CFG_FILE"; set +a
fi
# VK_LAYER_PRIORITY_BOOST opt-in (installed system-wide by --dual-host).
# Requests VK_QUEUE_GLOBAL_PRIORITY_HIGH on mpv's Vulkan present queue
# so it preempts CUDA on the shared GB10. No-op when the layer or the
# binary's cap_sys_nice are absent.
export VK_PRIORITY_BOOST_LEVEL=high
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
# Env bin first on PATH so mpv's embedded torch cpp_extension finds
# \`ninja\` for the chroma_krig CUDA-kernel JIT (same reason as the
# mpv-conda wrapper); also makes the shim's ffprobe sidecar use the
# env's ffmpeg.
export PATH="$ENV_PREFIX/bin:\$PATH"
# Source the dual-machine cluster config so DUAL_WORKER_HOST + DUAL_RDMA_*
# are visible to rife.vpy when the shim spawns mpv; without these in env,
# rife.vpy short-circuits to single-machine mode (see rife.vpy:155).
if [[ -f "$DUAL_CFG_FILE" ]]; then
  set -a; . "$DUAL_CFG_FILE"; set +a
fi
# Match mpv-conda: opt into VK_LAYER_PRIORITY_BOOST=high so the mpv
# subprocess the shim launches gets HIGH-priority Vulkan present.
export VK_PRIORITY_BOOST_LEVEL=high
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
  # MimeType list mirrors GNOME Videos (totem) so GNOME's Settings →
  # Default Applications surfaces mpv-conda as a video-player option for
  # every format totem handles. Without a wide list, GNOME's heuristic
  # for "is this a video player?" doesn't classify mpv as one.
  cat > "$APPS_DIR/mpv-conda.desktop" <<EOF
[Desktop Entry]
Name=mpv
GenericName=Media Player
Comment=Play movies and songs
Exec=$WRAPPER_DIR/mpv-conda --player-operation-mode=pseudo-gui -- %U
Icon=mpv-conda
Type=Application
Categories=AudioVideo;Audio;Video;Player;TV;
MimeType=application/mxf;application/ram;application/sdp;application/vnd.apple.mpegurl;application/vnd.ms-asf;application/vnd.ms-wpl;application/vnd.rn-realmedia;application/vnd.rn-realmedia-vbr;application/x-extension-m4a;application/x-extension-mp4;application/x-flash-video;application/x-matroska;application/x-mpegURL;application/x-netshow-channel;application/x-quicktimeplayer;application/x-shorten;application/smil;application/smil+xml;application/x-quicktime-media-link;application/x-smil;image/vnd.rn-realpix;image/x-pict;misc/ultravox;text/google-video-pointer;text/x-google-video-pointer;video/3gp;video/3gpp;video/3gpp2;video/dv;video/divx;video/fli;video/flv;video/mp2t;video/mp4;video/mp4v-es;video/mpeg;video/mpeg-system;video/msvideo;video/ogg;video/quicktime;video/vivo;video/vnd.divx;video/vnd.mpegurl;video/vnd.rn-realvideo;video/vnd.vivo;video/webm;video/x-anim;video/x-avi;video/x-flc;video/x-fli;video/x-flic;video/x-flv;video/x-m4v;video/x-matroska;video/x-mjpeg;video/x-mpeg;video/x-mpeg2;video/x-ms-asf;video/x-ms-asf-plugin;video/x-ms-asx;video/x-msvideo;video/x-ms-wm;video/x-ms-wmv;video/x-ms-wmx;video/x-ms-wvx;video/x-nsv;video/x-ogm+ogg;video/x-theora;video/x-theora+ogg;x-content/video-dvd;x-scheme-handler/pnm;x-scheme-handler/mms;x-scheme-handler/net;x-scheme-handler/rtp;x-scheme-handler/rtmp;x-scheme-handler/rtsp;x-scheme-handler/mmsh;x-scheme-handler/uvox;x-scheme-handler/icy;x-scheme-handler/icyx;
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
  # Wait for the network to come up before launching shim — without
  # this, shim races NetworkManager and fails to reach the Jellyfin
  # server on the first try (visible as "no server configured" or a
  # connection-refused error). `nm-online -q -t 30` blocks until NM
  # reports a connection is up, or gives up after 30 s. The extra
  # `sleep 3` covers DNS / mDNS / route-table settling after that.
  # `X-GNOME-Autostart-Delay=10` is a belt-and-braces — GNOME defers
  # the autostart trigger 10 s into the user session, giving the
  # desktop time to settle before we even start waiting for the NIC.
  cat > "$AUTOSTART_DIR/jellyfin-mpv-shim.desktop" <<EOF
[Desktop Entry]
Name=Jellyfin MPV Shim
Comment=Cast Jellyfin media to mpv (with RIFE + FSRCNNX)
Exec=sh -c 'nm-online -q -t 30 2>/dev/null; sleep 3; exec $WRAPPER_DIR/jellyfin-mpv-shim'
Icon=jellyfin-mpv-shim
Type=Application
Categories=AudioVideo;Player;
Terminal=false
X-GNOME-Autostart-enabled=true
X-GNOME-Autostart-Delay=10
StartupWMClass=jellyfin-mpv-shim
EOF
  log "wrote autostart entry to $AUTOSTART_DIR"

  update-desktop-database "$APPS_DIR" 2>/dev/null || true
  gtk-update-icon-cache "$ICON_ROOT" 2>/dev/null || true

  # ---------- promote to default video player (opt-in) ----------
  # Only runs with --set-default-video. The desktop file's MimeType=
  # list already registers mpv-conda as a candidate ("Open With…"
  # entry) for every video format totem handles, so users who don't
  # pass this flag still get the option in nautilus' submenu — they
  # just don't get mpv-conda auto-promoted over totem.
  if (( SET_DEFAULT_VIDEO )); then
    if command -v xdg-mime >/dev/null 2>&1; then
      local m
      for m in "${DEFAULT_VIDEO_MIMES[@]}"; do
        xdg-mime default mpv-conda.desktop "$m" 2>/dev/null || true
      done
      log "set mpv-conda as default video player for ${#DEFAULT_VIDEO_MIMES[@]} MIME types"
    else
      warn "xdg-mime not found; --set-default-video skipped"
    fi
  fi
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

  TensorRT engines pre-compiled at install time (instant first
  playback for the common source shapes):
    - RIFE 4.26 @ 854x480 / 720x480 / 640x480  (SD bucket)
    - RIFE 4.26 @ 720p                          (HD bucket)
    - RIFE 4.26 @ 1080p  (cinema-rate 1080p + 4K mixed-mode interp)
    - RIFE 4.6  @ 1080p  (25-30 fps 1080p + 4K mixed-mode interp)
  4K sources reuse the 1080p engines (mixed mode runs RIFE at the
  downsampled dim) — no separate 4K engine exists.
  Cache lives at:
    $ENV_PREFIX/lib/python*/site-packages/vsrife/models/

  Other shapes (non-standard SD, 1440p, …) or different scales will
  JIT-compile a matching engine on first use (~30-60s once; mpv shows
  a "compiling TRT engine" OSD note while it runs).

  Verify status with: $0 status
================================================================
EOF
}

# ============================================================================
# install — orchestrate all steps
# ============================================================================
cmd_install() {
  # Parse install-specific flags. All optional; defaults preserve the
  # original behavior (mirrors ON, danmaku ON, single-machine install).
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --no-mirrors|--no-mirror|--no-ustc) USE_MIRRORS=0; shift ;;
      --no-danmaku|--skip-danmaku)        INSTALL_DANMAKU=0; shift ;;
      --rebuild-trt)                      REBUILD_TRT=1; shift ;;
      --set-default-video|--default-video) SET_DEFAULT_VIDEO=1; shift ;;
      --dual-host)                        INSTALL_DUAL_HOST=1; shift ;;
      --dual-secondary)                   INSTALL_DUAL_SECONDARY=1; shift ;;
      -h|--help)
        cat <<EOF
Usage: $0 install [--no-mirrors] [--no-danmaku] [--rebuild-trt]
                  [--set-default-video]
                  [--dual-host | --dual-secondary]

  --no-mirrors    Don't write USTC mirrors into ~/.condarc and
                  ~/.config/pip/pip.conf. Use this outside China
                  where the USTC endpoints are slow / unreachable.
  --no-danmaku    Skip the danmaku (bullet-chat) plugin step.
  --rebuild-trt   Wipe the TRT engine cache before re-warming.
  --set-default-video
                  Promote mpv-conda to the GNOME default video player.

  --dual-host     Run the regular (single-machine) install AND set up
                  this box as the primary side of a dual-machine
                  cluster. Writes $DUAL_CFG_FILE with the cluster
                  config (interactive; or set DUAL_RDMA_DEV /
                  DUAL_HOST_IP / DUAL_PEER_IP env vars to skip prompts).
                  Pre-compiles the chroma_krig CUDA kernel.

  --dual-secondary
                  Worker-only install for the OTHER box of the cluster.
                  Skips the mpv build, jellyfin-mpv-shim, danmaku and
                  default-video registration. Installs the conda env +
                  vsrife + fsrcnnx-cudnn + worker.py code, registers a
                  systemd --user service (started + enabled), and a
                  GNOME autostart entry for a tray app that exposes a
                  Quit menu. Writes the same $DUAL_CFG_FILE.
EOF
        return 0
        ;;
      *) fatal "unknown install flag: $1 (try $0 install --help)" ;;
    esac
  done

  if (( INSTALL_DUAL_HOST && INSTALL_DUAL_SECONDARY )); then
    fatal "--dual-host and --dual-secondary are mutually exclusive"
  fi

  local mode="single"
  (( INSTALL_DUAL_HOST ))      && mode="dual-host"
  (( INSTALL_DUAL_SECONDARY )) && mode="dual-secondary"
  log "install mode: $mode  flags: mirrors=$USE_MIRRORS danmaku=$INSTALL_DANMAKU rebuild-trt=$REBUILD_TRT default-video=$SET_DEFAULT_VIDEO"

  detect_environment
  install_apt_packages
  install_miniforge
  create_conda_env
  install_pip_packages
  apply_patches

  if (( INSTALL_DUAL_SECONDARY )); then
    # Worker-only path. Skip mpv build / shim / danmaku / launchers /
    # set-default-video — none of these are useful on a box that only
    # serves RDMA from worker.py. Configs + warm_trt are still wanted
    # since the worker needs vsrife engines and the fsrcnnx-cudnn
    # bundle (chroma_krig CUDA kernel).
    install_configs
    warm_trt_cache
    dual_install_secondary_pieces
  else
    # Single-machine OR dual-host: full stack.
    build_mpv
    install_configs
    warm_trt_cache
    if (( INSTALL_DANMAKU )); then
      install_danmaku
    else
      section "step 8b/10: danmaku plugin (skipped — --no-danmaku)"
    fi
    install_shim_config
    install_launchers
    if (( INSTALL_DUAL_HOST )); then
      dual_install_host_pieces
    fi
  fi

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

  section "dual-machine"
  if [[ -f $DUAL_CFG_FILE ]]; then
    echo "  config:  $DUAL_CFG_FILE"
    sed 's/^/    /' "$DUAL_CFG_FILE" | head -12
    if systemctl --user list-unit-files 2>/dev/null \
        | grep -q "$DUAL_WORKER_SERVICE"; then
      local st
      st=$(systemctl --user is-active "$DUAL_WORKER_SERVICE" 2>/dev/null || true)
      echo "  worker service: $DUAL_WORKER_SERVICE  ($st)"
    fi
    [[ -e $DUAL_TRAY_AUTOSTART ]] && echo "  tray autostart: $DUAL_TRAY_AUTOSTART"
    [[ -d $DUAL_WORKER_DIR ]] && echo "  worker dir: $DUAL_WORKER_DIR"
  else
    echo "  (not configured — run install with --dual-host or --dual-secondary)"
  fi

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
           "$MPV_CFG_DIR/rife.vpy" "$MPV_CFG_DIR/sr_keys_helper.py" \
           "$MPV_CFG_DIR/vs_gpu_helpers.py" \
           "$MPV_CFG_DIR/scripts/sr_keys.lua" \
           "$MPV_CFG_DIR/fsrcnnx-cudnn/.installed-version" \
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
  log "stopping any running shim/mpv/worker processes"
  pkill -f "$ENV_PREFIX/bin/jellyfin-mpv-shim" 2>/dev/null || true
  pkill -f "$ENV_PREFIX/bin/mpv" 2>/dev/null || true
  pkill -f "$WRAPPER_DIR/jellyfin-mpv-shim" 2>/dev/null || true
  pkill -f "$WRAPPER_DIR/mpv-conda" 2>/dev/null || true
  sleep 1

  # Drop dual-machine pieces too (no-op if they were never installed).
  dual_uninstall_pieces

  # Selective cleanup: preserve user-supplied state across reinstalls.
  # Specifically keep dandanplay AppId (registration takes 1-3 days) and
  # Jellyfin server credentials (so the user doesn't have to re-pair).
  log "removing files we created in $MPV_CFG_DIR (preserving user creds)"
  for f in mpv.conf input.conf rife.vpy rife-light.vpy rife-half.vpy \
           sr_keys_helper.py vs_gpu_helpers.py \
           danmaku-config.json danmaku-credentials.json.example; do
    rm -f "$MPV_CFG_DIR/$f"
  done
  rm -rf "$MPV_CFG_DIR/scripts" "$MPV_CFG_DIR/shaders" \
         "$MPV_CFG_DIR/weights" "$MPV_CFG_DIR/fsrcnnx_cudnn" \
         "$MPV_CFG_DIR/fsrcnnx-cudnn" "$MPV_CFG_DIR/dual_machine" \
         "$MPV_CFG_DIR/__pycache__"
  # Preserved (not deleted):
  #   $MPV_CFG_DIR/danmaku-credentials.json   ← dandanplay AppId/Secret
  #   $MPV_CFG_DIR/danmaku-settings.json      ← user's panel choices
  rmdir "$MPV_CFG_DIR" 2>/dev/null && \
      log "  (config dir was empty, removed it)" || \
      log "  preserved: $(ls "$MPV_CFG_DIR" 2>/dev/null | tr '\n' ' ')"

  log "removing our symlinks in $SHIM_CFG_DIR (preserving cred.json + user prefs)"
  for f in mpv.conf input.conf rife.vpy rife-light.vpy rife-half.vpy \
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
    - ~/.cache/dgxspark-mpv/trt-engines/      (compiled TRT engines — the
                                               next install's warm step
                                               restores these instead of
                                               recompiling 4-7 minutes)

  Re-run \`$0 install\` to rebuild from scratch — these are picked back up.
================================================================
EOF
}

# ============================================================================
# Dual-machine install pieces
# ============================================================================

# Default high-speed iface: first UP enp1s0* / enP2p1s0* device. The
# Spark CX7 NIC presents two PCIe ports that bond into one 200 G link;
# both names start with the same prefix, so picking either rail is
# enough to pin a default — link aggregation pairs are derived below.
dual_detect_iface_default() {
  ip -o link show up 2>/dev/null \
    | awk -F': ' '/enp1s0|enP2p1s0/ {print $2; exit}' \
    | awk '{print $1}'
}

dual_detect_local_ip() {
  local iface=$1
  ip -4 -o addr show dev "$iface" 2>/dev/null \
    | awk '{print $4}' | cut -d/ -f1 | head -1
}

# Probe one interactive value with env-pre-set short-circuit. $1 = env
# var name (used as the override channel); $2 = prompt label; $3 = default.
# Prints the resolved value on stdout (no other noise — caller captures).
_dual_prompt() {
  local env_name=$1 label=$2 default=$3
  local pre="${!env_name:-}"
  if [[ -n $pre ]]; then
    printf '%s\n' "$pre"
    return 0
  fi
  if [[ -t 0 ]]; then
    local answer
    read -r -p "  $label [$default]: " answer < /dev/tty
    if [[ -z $answer ]]; then printf '%s\n' "$default"
    else                       printf '%s\n' "$answer"
    fi
  else
    printf '%s\n' "$default"
  fi
}

# Gather DUAL_* values from env / interactive / defaults, write to
# $DUAL_CFG_FILE in /etc/environment-style key=value form so it works
# both as systemd EnvironmentFile and as `set -a; source ...; set +a`
# for the mpv-conda wrapper.
dual_configure() {
  local role=$1   # "host" or "secondary"
  section "step D1: dual-machine config (role=$role)"

  cat <<EOF
A dual-machine install needs a 200 G RoCE link between this box and
the other Spark, with static IPs (not DHCP) on the high-speed
interface. Link aggregation across both PCIe ports of the CX7 NIC is
enabled by default — bench numbers in docs/performance.md assume
this. Press <Enter> to accept any default, or set the corresponding
env before re-running to skip prompts (DUAL_RDMA_DEV / DUAL_HOST_IP /
DUAL_WORKER_IP / etc).

EOF

  local iface_default; iface_default=$(dual_detect_iface_default)
  iface_default=${iface_default:-enp1s0f0np0}

  local rdma_dev_default; rdma_dev_default=$(printf 'rocep1s0f%s' "$(printf '%s' "$iface_default" | sed -nE 's/.*enp1s0f([0-9]+)np[0-9]+/\1/p')")
  rdma_dev_default=${rdma_dev_default:-rocep1s0f0}

  local self_ip_default; self_ip_default=$(dual_detect_local_ip "$iface_default")
  self_ip_default=${self_ip_default:-10.200.128.1}

  local peer_ip_default
  if [[ $role == "host" ]]; then peer_ip_default="10.200.128.2"
  else                            peer_ip_default="10.200.128.1"
  fi

  local IFACE     RDMA_DEV  SELF_IP   PEER_IP   RDMA_PORT  WORKER_USER
  IFACE=$(_dual_prompt DUAL_IFACE       "high-speed interface"      "$iface_default")
  RDMA_DEV=$(_dual_prompt DUAL_RDMA_DEV "RDMA device"               "$rdma_dev_default")
  SELF_IP=$(_dual_prompt DUAL_SELF_IP   "this box's IP on the link" "$self_ip_default")
  PEER_IP=$(_dual_prompt DUAL_PEER_IP   "peer's IP on the link"     "$peer_ip_default")
  RDMA_PORT=$(_dual_prompt DUAL_RDMA_PORT "RDMA port"               "$DUAL_DEFAULT_RDMA_PORT")
  if [[ $role == "host" ]]; then
    WORKER_USER=$(_dual_prompt DUAL_WORKER_USER \
                  "guest user (for ssh + rsync from bench)" "ubuntu")
  fi
  # MASTER_PORT / NCCL_IB_HCA / NCCL_SOCKET_IFNAME used to be written
  # here for torch.distributed rendezvous. Since the drop-NCCL refactor
  # (control plane is now a single TCP socket on DUAL_LIVENESS_PORT),
  # none of those are read by anything. The single RDMA device on
  # DUAL_RDMA_DEV is enough for pyverbs; multi-rail aggregation, if
  # ever needed, would be configured at the pyverbs layer directly.

  # Role-specific host/worker IP mapping. From the HOST's perspective
  # DUAL_WORKER_HOST is the peer. From the SECONDARY's perspective
  # DUAL_HOST_IP is the peer.
  local HOST_IP WORKER_HOST
  if [[ $role == "host" ]]; then
    HOST_IP="$SELF_IP"
    WORKER_HOST="$PEER_IP"
  else
    HOST_IP="$PEER_IP"
    WORKER_HOST="$SELF_IP"
  fi

  mkdir -p "$DUAL_CFG_DIR"
  cat > "$DUAL_CFG_FILE" <<EOF
# dgxspark-jellyfin-mpv-rife — dual-machine cluster config.
# Source of truth for both the mpv-conda launcher (host) and the
# systemd worker unit (secondary). Regenerate by re-running
# install.sh with --dual-host or --dual-secondary.

DUAL_ROLE=$role
DUAL_IFACE=$IFACE
DUAL_RDMA_DEV=$RDMA_DEV
DUAL_RDMA_PORT=$RDMA_PORT
DUAL_HOST_IP=$HOST_IP
DUAL_WORKER_HOST=$WORKER_HOST
DUAL_WORKER_DIR=$DUAL_WORKER_DIR
${WORKER_USER:+DUAL_WORKER_USER=$WORKER_USER}
EOF
  log "wrote $DUAL_CFG_FILE:"
  sed 's/^/    /' "$DUAL_CFG_FILE"
}

# Copy worker.py + supporting modules to $DUAL_WORKER_DIR so the
# systemd unit has a stable path (independent of where the repo lives).
dual_install_worker_files() {
  section "step D2: install worker code → $DUAL_WORKER_DIR"
  # Migrate from the pre-2026-05 install layout: drop the orphaned
  # $HOME/dual_machine tree once the new path is populated. Skip the
  # rm if it's somehow the same directory (defensive — should never
  # happen since DUAL_WORKER_DIR moved under .local).
  if [[ -d "$DUAL_WORKER_DIR_LEGACY" \
        && "$DUAL_WORKER_DIR_LEGACY" != "$DUAL_WORKER_DIR" ]]; then
    log "migrating: removing legacy $DUAL_WORKER_DIR_LEGACY"
    rm -rf "$DUAL_WORKER_DIR_LEGACY"
  fi
  mkdir -p "$DUAL_WORKER_DIR"
  # Module list shared with bench/_common.sh — edit
  # dual_machine/WORKER_MODULES, not this loop.
  local _wm
  while IFS= read -r _wm; do
    [[ -z "$_wm" || "$_wm" == \#* ]] && continue
    install -m 0644 "$PROJECT_DIR/dual_machine/$_wm" "$DUAL_WORKER_DIR/"
  done < "$PROJECT_DIR/dual_machine/WORKER_MODULES"
  install -m 0644 "$PROJECT_DIR/vs_gpu_helpers.py"               "$DUAL_WORKER_DIR/"
  log "installed worker files: $(ls $DUAL_WORKER_DIR/*.py | wc -l) modules"
}

# Render worker.service.in → systemd-user unit dir.
dual_install_secondary_service() {
  section "step D3: install $DUAL_WORKER_SERVICE (systemd --user)"
  local unit_dir="$HOME/.config/systemd/user"
  mkdir -p "$unit_dir"
  sed -e "s|@DUAL_CONFIG_FILE@|$DUAL_CFG_FILE|g" \
      -e "s|@PYTHON_BIN@|$ENV_PREFIX/bin/python|g" \
      -e "s|@WORKER_PY@|$DUAL_WORKER_DIR/worker.py|g" \
      "$PROJECT_DIR/dual_machine/worker.service.in" \
    > "$unit_dir/$DUAL_WORKER_SERVICE"
  systemctl --user daemon-reload
  systemctl --user enable "$DUAL_WORKER_SERVICE" 2>&1 \
    | grep -v 'Created symlink' || true
  # `restart` (not `start`) so re-running install on an upgraded
  # codebase actually picks up the new worker.py — `start` is a no-op
  # if the unit is already active and the user would be silently left
  # on the old binary.
  systemctl --user restart "$DUAL_WORKER_SERVICE" || \
    warn "systemctl --user restart failed — start it manually after login"
  log "installed + (re)started $unit_dir/$DUAL_WORKER_SERVICE"
}

# Tray app (PyGObject + AppIndicator). Drops the script next to the
# worker, registers a .desktop in autostart.
dual_install_secondary_tray() {
  section "step D4: install secondary tray + launcher + autostart"
  if [[ ! -f "$PROJECT_DIR/dual_machine/secondary_tray.py" ]]; then
    warn "secondary_tray.py missing in project — skipping tray install"
    return 0
  fi
  install -m 0755 "$PROJECT_DIR/dual_machine/secondary_tray.py" \
                  "$DUAL_WORKER_DIR/secondary_tray.py"

  # Install the mpv icon on this box too (the single-machine flow
  # does it in install_launchers, which --dual-secondary skips). The
  # tray uses `Icon=mpv-conda`, the same name single-machine apps use.
  for size in 16 32 64 128; do
    local src="$PROJECT_DIR/icons/mpv/$size.png"
    [[ -f $src ]] || continue
    local dst_dir="$ICON_ROOT/${size}x${size}/apps"
    mkdir -p "$dst_dir"
    cp -f "$src" "$dst_dir/mpv-conda.png"
  done
  if [[ -f "$PROJECT_DIR/icons/mpv/scalable.svg" ]]; then
    mkdir -p "$ICON_ROOT/scalable/apps"
    cp -f "$PROJECT_DIR/icons/mpv/scalable.svg" \
          "$ICON_ROOT/scalable/apps/mpv-conda.svg"
  fi
  gtk-update-icon-cache -f -t "$ICON_ROOT" 2>/dev/null || true

  # Wrapper sets GI_TYPELIB_PATH so the conda Python can find the
  # system's AyatanaAppIndicator3 typelib (conda-forge has none on
  # aarch64) — same trick the jellyfin-mpv-shim wrapper uses.
  mkdir -p "$WRAPPER_DIR"
  cat > "$WRAPPER_DIR/dgxspark-dual-tray" <<EOF
#!/usr/bin/env bash
SYS_GIR="/usr/lib/aarch64-linux-gnu/girepository-1.0"
export GI_TYPELIB_PATH="\${GI_TYPELIB_PATH:+\$GI_TYPELIB_PATH:}\$SYS_GIR"
export PYTHONHOME="$ENV_PREFIX"
exec "$ENV_PREFIX/bin/python" "$DUAL_WORKER_DIR/secondary_tray.py" "\$@"
EOF
  chmod +x "$WRAPPER_DIR/dgxspark-dual-tray"

  local desktop_body
  desktop_body="[Desktop Entry]
Type=Application
Name=DGX Spark Dual Worker
GenericName=Dual-machine worker tray
Comment=Status + Quit menu for the dual-machine worker service
Icon=mpv-conda
Exec=$WRAPPER_DIR/dgxspark-dual-tray
Terminal=false
Categories=AudioVideo;Player;
StartupNotify=false"

  # GNOME Activities launcher — what the user sees in the app grid /
  # Alt-F2. Without this only the autostart entry below exists, and
  # the user has no way to start the tray after manually quitting it.
  mkdir -p "$APPS_DIR"
  printf '%s\n' "$desktop_body" > "$APPS_DIR/$DUAL_TRAY_DESKTOP"
  update-desktop-database "$APPS_DIR" 2>/dev/null || true
  log "launcher at $APPS_DIR/$DUAL_TRAY_DESKTOP"

  # Autostart on login — the launcher is just for manual restart.
  mkdir -p "$(dirname "$DUAL_TRAY_AUTOSTART")"
  printf '%s\nX-GNOME-Autostart-enabled=true\n' \
    "$desktop_body" > "$DUAL_TRAY_AUTOSTART"
  log "tray autostart at $DUAL_TRAY_AUTOSTART"

  # Linger keeps the user session alive at boot (before any GUI login),
  # so the worker.service starts when the box powers on instead of
  # waiting for someone to log into GNOME. Requires sudo.
  if loginctl show-user "$USER" 2>/dev/null | grep -q '^Linger=yes$'; then
    log "user session lingering already enabled"
  else
    log "enabling user session lingering (sudo loginctl enable-linger $USER)"
    if sudo loginctl enable-linger "$USER" 2>/dev/null; then
      log "  ✓ worker.service will start at boot"
    else
      warn "  could not enable linger — worker.service will only start on login"
    fi
  fi
}

# Compile chroma_krig CUDA kernel at install time so first-frame
# playback doesn't pay the ~30 s JIT cost. Imports fsrcnnx_cudnn.chroma_krig
# inside conda env; load_inline + ninja compiles to ~/.cache/torch_extensions.
dual_precompile_krig() {
  section "step D5: pre-compile chroma_krig CUDA kernel"
  if ! "$ENV_PREFIX/bin/python" -c \
       "import sys; sys.path.insert(0, '$MPV_CFG_DIR/fsrcnnx-cudnn'); \
        import fsrcnnx_cudnn.chroma_krig as _; print('chroma_krig:', _.__file__)"; then
    warn "chroma_krig pre-compile failed; first frame at runtime will JIT"
  else
    log "chroma_krig compiled into ~/.cache/torch_extensions"
  fi
}

dual_install_host_pieces() {
  dual_configure "host"
  dual_precompile_krig
  log "host-side dual install done. To enable dual mode in mpv, your"
  log "playback session must source $DUAL_CFG_FILE before launching mpv-conda."
  log "(The default mpv-conda wrapper already does this when the file exists.)"
}

dual_install_secondary_pieces() {
  dual_configure "secondary"
  dual_install_worker_files
  dual_install_secondary_service
  dual_install_secondary_tray
  dual_precompile_krig
  log "secondary-side dual install done."
  log "  service: systemctl --user status $DUAL_WORKER_SERVICE"
  log "  log:     $HOME/.cache/dgxspark-dual-worker.log (or journalctl --user -u $DUAL_WORKER_SERVICE)"
}

dual_uninstall_pieces() {
  section "step D-X: uninstall dual-machine pieces"
  if systemctl --user list-unit-files 2>/dev/null | grep -q "$DUAL_WORKER_SERVICE"; then
    systemctl --user disable --now "$DUAL_WORKER_SERVICE" 2>/dev/null || true
    rm -f "$HOME/.config/systemd/user/$DUAL_WORKER_SERVICE"
    systemctl --user daemon-reload
  fi
  rm -f "$DUAL_TRAY_AUTOSTART"
  rm -f "$DUAL_CFG_FILE"
  [[ -d $DUAL_CFG_DIR && -z "$(ls -A "$DUAL_CFG_DIR" 2>/dev/null)" ]] && rmdir "$DUAL_CFG_DIR"
  rm -rf "$DUAL_WORKER_DIR" "$DUAL_WORKER_DIR_LEGACY"
  # If $DUAL_WORKER_DIR was the only thing under
  # ~/.local/share/dgxspark-mpv/, prune the empty parent so an
  # uninstall doesn't leave breadcrumbs behind.
  local dual_share_parent; dual_share_parent="$(dirname "$DUAL_WORKER_DIR")"
  [[ -d $dual_share_parent && -z "$(ls -A "$dual_share_parent" 2>/dev/null)" ]] && rmdir "$dual_share_parent"

  # Mirror of step 7c: drop the cap from mpv + remove the system layer
  # manifest + .so. The build.sh uninstall path is sudo-aware. Safe even
  # when the layer was never installed (uninstall is idempotent).
  if [[ -x "$PROJECT_DIR/dual_machine/vk_priority_layer/build.sh" ]] \
     && command -v sudo >/dev/null; then
    log "reverting vk_priority_layer (setcap + system manifest + .so)"
    MPV_BIN="$ENV_PREFIX/bin/mpv" \
      bash "$PROJECT_DIR/dual_machine/vk_priority_layer/build.sh" uninstall \
        2>&1 | sed 's/^/  /' || true
  fi
  log "removed dual config + service + tray autostart + worker dir."
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
  $0 install [flags]   install (single-machine by default; --dual-host /
                       --dual-secondary for the two roles of a dual cluster)
  $0 status            show what's currently installed and where
  $0 uninstall         remove everything we installed except apt + miniforge

Install flags (all optional, defaults preserve original behavior):
  --no-mirrors    Don't write USTC mirrors to ~/.condarc + pip.conf
                  (use outside China)
  --no-danmaku    Skip the danmaku (bullet-chat) plugin step
  --rebuild-trt   Wipe + recompile the TRT engine cache (use after
                  driver/CUDA/TensorRT upgrade if cached engines
                  produce broken video)
  --set-default-video
                  Make mpv-conda the GNOME default video player
                  (xdg-mime default for mp4/mkv/webm/...)

See README.md for details.
EOF
    ;;
  *)
    fatal "unknown command: $1 (try: install / status / uninstall)"
    ;;
esac
