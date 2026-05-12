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
FSRCNNX_CUDNN_VERSION="v0.1.1"

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

    # Patch jellyfin-mpv-shim for mpv 0.41 compatibility. Verified
    # against shim 2.9.0 and 2.10.0; both ship `mpv_options["osc"] =
    # False` which mpv 0.41 rejects (the legacy --osc flag and the
    # _player.osc property are both gone). Replace with a script-opts
    # setting that maps the user's `settings.enable_osc` config field
    # to the new `osc-visibility=auto|never` property. Hard-coding
    # "never" (our previous patch) ignored the user's preference and
    # made the OSC permanently invisible in shim mode.
    #
    # Note: shim 2.10.0's enable_osc() method internally uses the
    # property-based path on mpv 0.41, but it's gated behind
    # `if hasattr(self._player, "osc")` (False on 0.41), so the call
    # never reaches it at runtime. The construction-time arg below is
    # therefore the only thing that actually controls visibility.
    f="$pydir/site-packages/jellyfin_mpv_shim/player.py"
    if [[ -f "$f" ]]; then
      if grep -q '"osc"\] = False' "$f"; then
        cp -n "$f" "${f}.bak"
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

# FSRCNNX SR is now done inside vapoursynth (chained after rife_yuv
# in each rife*.vpy via fsrcnnx_cudnn.vsfunc.fsrcnnx_yuv_auto). The
# old GLSL shader path is disabled here to avoid double-upscale.
# Restore by removing the leading `#` if you want to A/B test.
#glsl-shaders=~~/shaders/FSRCNNX_x2_8-0-4-1.glsl

# RIFE + FSRCNNX. Single .vpy. F8 cycles FSRCNNX variant, F9 toggles RIFE.
# See scripts/sr_keys.lua for the keybinds.
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
# `buffered-frames=12 / concurrent-frames=4` — deeper than vapoursynth's
# defaults so the vs scheduler can overlap encode → infer → SR work.
[rife]
profile-cond=0<(p["video-params/h"] or 0) and (p["video-params/h"] or 0)<=2160
profile-restore=copy-equal
vf=vapoursynth=~~/rife.vpy:buffered-frames=12:concurrent-frames=4
EOF
  log "wrote $MPV_CFG_DIR/mpv.conf"

  # input.conf — F8 / F9 are bound by scripts/sr_keys.lua (cycle FSRCNNX
  # variant / toggle RIFE). Header here is just documentation; the keys
  # themselves come from the lua script.
  cat > "$MPV_CFG_DIR/input.conf" <<'EOF'
# F8 / F9 — bound by scripts/sr_keys.lua:
#   F8: cycle FSRCNNX variant (16x4 → 16x3 → 16x2 → 8x2 → loop). Default
#       starting point is whatever fsrcnnx_yuv_auto picks for the current
#       source on this display (4K by default; override via env vars
#       FSRCNNX_TARGET_W / FSRCNNX_TARGET_H). Per-file, returns to auto
#       on the next file-load.
#   F9: toggle RIFE on/off independent of FSRCNNX. Persists across files.
#
# Both keys force a vapoursynth filter reload (~1–3 s freeze) — there's
# no runtime parameter switch inside the cuDNN runner.
EOF
  log "wrote $MPV_CFG_DIR/input.conf"

  # sr_keys.lua — F8 cycles FSRCNNX variant via /tmp/fsrcnnx_variant
  # override file, F9 toggles RIFE on/off via /tmp/rife_disabled. Both
  # force a vf reload so the .vpy re-reads the side-channel files.
  cat > "$MPV_CFG_DIR/scripts/sr_keys.lua" <<'EOF'
-- F8 — cycle FSRCNNX variant.        F9 — toggle RIFE on/off.
--
-- Both keys force a vapoursynth filter reload (~1–3 s freeze) because
-- there's no runtime parameter switch inside our chain — we re-build it.
--
-- Communication with the .vpy chain (file-based, so any vf reload picks
-- up the new state):
--   /tmp/fsrcnnx_variant         — override variant. Empty/missing =
--                                  auto. Otherwise "x2_8" / "x2_16" /
--                                  "x3_16" / "x4_16". Written by F8.
--                                  Cleared on file-loaded (return to
--                                  auto for each new video).
--   /tmp/fsrcnnx_active_variant  — what the .vpy is currently running
--                                  (auto-resolved). Written by the
--                                  .vpy. Read by F8 to know the cycle's
--                                  current position.
--   /tmp/rife_disabled           — touch-file. If present, .vpy skips
--                                  rife_yuv. Toggled by F9. Persists
--                                  across files.
--
-- Default state on first file load: F8 = auto, F9 = enabled.

local mp = require "mp"

local OVERRIDE_FILE = "/tmp/fsrcnnx_variant"
local ACTIVE_FILE   = "/tmp/fsrcnnx_active_variant"
local RIFE_OFF_FILE = "/tmp/rife_disabled"

-- Cycle includes "off" as the final stop so F8 can fully bypass FSRCNNX.
-- When the .vpy reads /tmp/fsrcnnx_variant == "off", it skips the SR pass
-- entirely and outputs the post-RIFE clip (or the source if F9 is also
-- off). Pressing F8 from "off" wraps back to "x4_16".
local CYCLE = { "x4_16", "x3_16", "x2_16", "x2_8", "off" }

local function cycle_index(variant)
  for i, v in ipairs(CYCLE) do
    if v == variant then return i end
  end
  return nil
end

local function read_file(path)
  local fh = io.open(path, "r")
  if not fh then return nil end
  local s = fh:read("*all")
  fh:close()
  return s and s:match("^%s*(.-)%s*$") or nil
end

local function write_file(path, s)
  local fh = io.open(path, "w")
  if not fh then return false end
  fh:write(s); fh:close()
  return true
end

local function file_exists(path)
  local fh = io.open(path, "r")
  if fh then fh:close(); return true end
  return false
end

local function reload_vf()
  -- Clear-then-restore is the most reliable way to force a re-exec
  -- of the .vpy. `vf-command` doesn't reach inside vapoursynth.
  local current = mp.get_property("vf")
  if not current or current == "" then return end
  mp.set_property("vf", "")
  mp.set_property("vf", current)
end

local function cycle_fsrcnnx()
  local active = read_file(ACTIVE_FILE)
  if not active or active == "" or active == "none" then
    -- Chain is currently bypassing FSRCNNX (e.g. 4K → 4K, ratio < 1.3).
    -- Start the cycle from the front rather than refusing.
    active = CYCLE[#CYCLE]   -- so next = CYCLE[1] = x4_16
  end
  local idx = cycle_index(active)
  local next_idx = (idx and (idx % #CYCLE) + 1) or 1
  local next_variant = CYCLE[next_idx]

  write_file(OVERRIDE_FILE, next_variant)
  local label = (next_variant == "off") and "OFF" or next_variant
  mp.osd_message(string.format("FSRCNNX → %s (reloading…)", label), 2)
  reload_vf()
end

local function toggle_rife()
  if file_exists(RIFE_OFF_FILE) then
    os.remove(RIFE_OFF_FILE)
    mp.osd_message("RIFE: ON (reloading…)", 2)
  else
    write_file(RIFE_OFF_FILE, "")
    mp.osd_message("RIFE: OFF (reloading…)", 2)
  end
  reload_vf()
end

local function reset_on_file_load()
  -- F8 (FSRCNNX variant) resets per file — each video gets its own
  -- auto-pick starting point. F9 (RIFE on/off) persists, since it's
  -- a global preference rather than per-source tuning.
  os.remove(OVERRIDE_FILE)
  os.remove(ACTIVE_FILE)
end

mp.add_key_binding("F8", "fsrcnnx-cycle", cycle_fsrcnnx)
mp.add_key_binding("F9", "rife-toggle",   toggle_rife)
mp.register_event("file-loaded", reset_on_file_load)
EOF
  log "wrote $MPV_CFG_DIR/scripts/sr_keys.lua"

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
  cat > "$MPV_CFG_DIR/rife.vpy" <<'EOF'
# Unified RIFE + FSRCNNX pipeline.
#
# F8 cycles FSRCNNX variant (16x4 → 16x3 → 16x2 → 8x2 → OFF → loop).
# F9 toggles RIFE on/off independently. Both bound by scripts/sr_keys.lua.
#
#   h ≤ 720         RIFE 4.26 + scale=1.0  + FSRCNNX family=16-layer
#   720 < h ≤ 1080  RIFE 4.6  + scale=1.0  + FSRCNNX family=8-layer
#   1080 < h ≤ 2160 mixed mode — original 4K real frames passthrough;
#                   interp frames go through downsample → RIFE 4.26 →
#                   FSRCNNX 16-layer → upsample back to 4K
#   fps > 30        RIFE skipped, FSRCNNX still runs if ratio merits

import os, sys
from pathlib import Path

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path(os.environ.get("MPV_HOME") or
                 os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
                 + "/mpv").resolve()
sys.path.insert(0, str(HERE))

import vapoursynth as vs
import vsrife
from vs_gpu_helpers import rife_yuv
from sr_keys_helper import apply_fsrcnnx, rife_disabled

core = vs.core
clip = video_in

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
    fps = (clip.fps_num / clip.fps_den) if clip.fps_den else 24.0
    # Cinema-rate (≤24 fps) sources have a 20.8 ms/frame budget at 2×
    # output; 25–30 fps sources tighten to 16.7 ms/frame at 60 fps. The
    # heavy chain (4.26 + 16-layer) fits the cinema budget but blows
    # past 16.7 ms — fall back to lighter (4.6 + 8-layer) above 24 fps.
    heavy_fps = fps < 25

    # Pick the working format: 4:2:2 / 4:4:4 inputs stay (or collapse to)
    # 4:2:2 to keep the extra chroma fidelity; everything else lands at
    # 4:2:0. rife_yuv's GPU YUV↔RGB path is 4:2:0-only, so the 4:2:2
    # branch falls back to vsrife direct with zimg YUV↔RGB — measured
    # within ~0.4% of the 4:2:0 fast path on this hardware because zimg
    # runs concurrently with the GPU rife/SR work.
    _sw, _sh = clip.format.subsampling_w, clip.format.subsampling_h
    keep_422 = (_sw == 1 and _sh == 0) or (_sw == 0 and _sh == 0)
    target_fmt = vs.YUV422P10 if keep_422 else vs.YUV420P10
    if clip.format.id != int(target_fmt):
        clip = core.resize.Bicubic(clip, format=target_fmt)

    def _rife(c, model, scale=1.0):
        """4:2:0: rife_yuv (GPU YUV↔RGB, fast path). 4:2:2: vsrife
        direct with zimg YUV↔RGB. Both produce a YUV clip with the same
        subsampling as the input."""
        if c.format.id == int(vs.YUV420P10):
            return rife_yuv(c, model=model, scale=scale,
                            factor_num=2, factor_den=1, color_range=color_range)
        rgb = core.resize.Bicubic(c, format=vs.RGBH,
                                   matrix_in_s=matrix_s, range_in_s=color_range)
        rgb = vsrife.rife(rgb, model=model, scale=scale,
                           factor_num=2, factor_den=1, trt=True)
        return core.resize.Bicubic(rgb, format=c.format.id,
                                    matrix_s=matrix_s, range_s=color_range)

    fsrcnnx_applied = False
    if not rife_disabled() and fps <= 30:
        if h <= 720:
            clip = _rife(clip, "4.26")
        elif h <= 1080:
            clip = _rife(clip, "4.26" if heavy_fps else "4.6")
        else:  # 1080 < h ≤ 2160 — 4K mixed mode.
               # Full 4K RIFE is too heavy on GB10 (~25 fps pipelined); even
               # scale=0.5 half-flow can't sustain 48fps (~38 fps). So we
               # downsample to 1080p, run RIFE there, take ONLY the interp
               # frames, SR them back to 4K, and Interleave with the
               # original 4K source. Real frames stay bit-exact; only the
               # synthesized in-between frames pay the downsample+SR cost.
               #
               # On the half-the-frames budget, ≤24 fps sources can afford
               # the heavier interp chain (RIFE 4.26 + 16-layer FSRCNNX);
               # 25+ fps falls back to the lighter (4.6 + 8-layer) variant
               # to stay under the tighter 60 fps output budget.
            src_4k = clip
            target_h = 1080
            target_w = ((clip.width * target_h) // clip.height) & ~1
            down = core.resize.Bicubic(clip, width=target_w, height=target_h)
            rife_model  = "4.26" if heavy_fps else "4.6"
            sr_family_4k = "16-layer" if heavy_fps else "8-layer"
            rife_low = _rife(down, rife_model)
            rife_interp_1080p = rife_low.std.SelectEvery(2, [1])
            interp_4k = apply_fsrcnnx(rife_interp_1080p, family=sr_family_4k)
            if interp_4k.width == src_4k.width and interp_4k.height == src_4k.height:
                clip = core.std.Interleave([src_4k, interp_4k])
            else:
                # F8 forced FSRCNNX off / x3 / x4 — interp_4k isn't at 4K
                # so we can't Interleave. Fall back to the SR chain
                # directly (no original-frame preservation; mpv display
                # path handles fit).
                clip = interp_4k
            fsrcnnx_applied = True

    # h ≤ 720 always uses 16-layer (select_variant refuses 8-layer at
    # ratio ≥ 2.5). 1080p picks 16-layer at cinema rates, 8-layer at
    # 25+ fps. 4K mixed-mode handled its own FSRCNNX call above.
    if not fsrcnnx_applied:
        if h <= 720:
            family = "16-layer"
        else:
            family = "16-layer" if heavy_fps else "8-layer"
        clip = apply_fsrcnnx(clip, family=family)
except Exception as _e:
    fmt_name = getattr(video_in.format, "name", "?")
    sys.stderr.write(
        f"[rife.vpy] chain setup failed for source format {fmt_name}: "
        f"{type(_e).__name__}: {_e} — passing through unmodified\n")
    clip = video_in

clip.set_output()

# Pre-warm: kick frame 0 off on a background thread so the FSRCNNX
# cuDNN graph build + RIFE TRT engine load overlap with the rest of
# mpv's startup (audio init, OSD layout, seek-to-start). vapoursynth
# caches the computed frame, so mpv's own first-frame fetch is usually
# an instant hit. daemon=True so we don't keep the process alive if
# the filter is replaced (F8/F9 reload) before frame 0 finishes.
import threading as _t
def _prewarm(_clip=clip):
    try:
        _clip.get_frame(0)
    except Exception:
        pass
_t.Thread(target=_prewarm, daemon=True, name="rife-prewarm").start()
EOF
  log "wrote $MPV_CFG_DIR/rife.vpy"

  # Old per-band .vpy files / shader dir / in-tree weights / old bundle
  # location under scripts/ have all been superseded — clean stale
  # copies from previous installs. The old `scripts/fsrcnnx-cudnn/`
  # path caused mpv to log "Cannot find main.* in scripts/<subdir>"
  # on every startup because mpv's multi-file-script convention wants
  # `main.{lua,js,py,mjs}` as the entry, which the bundle doesn't
  # ship. We host the bundle outside `scripts/` now to dodge that.
  rm -f  "$MPV_CFG_DIR/rife-light.vpy" "$MPV_CFG_DIR/rife-half.vpy"
  rm -rf "$MPV_CFG_DIR/shaders" "$MPV_CFG_DIR/weights" \
         "$MPV_CFG_DIR/fsrcnnx_cudnn" "$MPV_CFG_DIR/scripts/fsrcnnx-cudnn"

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
  section "step 8/10: warm TRT engine cache (4.26 + 4.6 @ 720p / 1080p)"
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
  if (( REBUILD_TRT )) && [[ -d "$cache_dir" ]]; then
    log "wiping existing TRT engine cache (--rebuild-trt)"
    find "$cache_dir" -maxdepth 1 -name "*.ts" -print -delete | sed 's/^/  removed: /'
  fi

  # vsrife's TRT static-shape mode uses the input clip's actual dimensions
  # to build the engine, so feeding it a dummy at the right resolution
  # produces an engine matching real playback. Cache filenames embed the
  # padded shape (e.g. 1920x1088, 3840x2176) so 1080p and 4K engines
  # coexist independently.
  #
  # Total disk: ~250 MB for all 5 engines. Total time on a clean install:
  # ~3-5 minutes (each engine ~30-60s); cache hits return in <1s.
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

# 720p / 540p sources — auto-band uses 4.26 (rife.vpy heavy branch).
warm("RIFE 4.26 @ 720p",  "4.26", 1280,  720)

# 1080p sources — auto-band uses 4.6.
warm("RIFE 4.6  @ 1080p", "4.6",  1920, 1080)

# 4K sources — mixed mode downsamples to 1080p then runs 4.26.
warm("RIFE 4.26 @ 1080p (for 4K mixed-mode interp)", "4.26", 1920, 1080)
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
    [[ -L "$dst" ]] && continue
    [[ -e "$src" ]] || continue
    rm -f "$dst"
    ln -s "$src" "$dst"
    log "linked $dst → $src"
  done
  for name in scripts; do
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
      --rebuild-trt)                      REBUILD_TRT=1; shift ;;
      --set-default-video|--default-video) SET_DEFAULT_VIDEO=1; shift ;;
      -h|--help)
        cat <<EOF
Usage: $0 install [--no-mirrors] [--no-danmaku] [--rebuild-trt]
                  [--set-default-video]

  --no-mirrors    Don't write USTC mirrors into ~/.condarc and
                  ~/.config/pip/pip.conf. Use this outside China
                  where the USTC endpoints are slow / unreachable.
  --no-danmaku    Skip the danmaku (bullet-chat) plugin step. The
                  rest of the stack (mpv + RIFE + FSRCNNX + shim)
                  installs normally.
  --rebuild-trt   Wipe the TRT engine cache before re-warming. Use
                  this if you've upgraded the NVIDIA driver / CUDA
                  / TensorRT and your existing cached engines now
                  produce broken video — vsrife keys engines on GPU
                  model + TRT version but not driver build, so an
                  in-place driver upgrade can leave stale-but-
                  matching cache files. Adds ~2-3 min to install.
  --set-default-video
                  Promote mpv-conda to the GNOME default video
                  player (xdg-mime default for mp4/mkv/webm/mov/
                  avi/mpeg/m4v/flv/3gp/wmv/ogg/asf/hls). Without
                  this flag mpv-conda is still registered as a
                  candidate (shows up in "Open With…") but totem
                  remains the default.
EOF
        return 0
        ;;
      *) fatal "unknown install flag: $1 (try $0 install --help)" ;;
    esac
  done

  log "install flags: USE_MIRRORS=$USE_MIRRORS  INSTALL_DANMAKU=$INSTALL_DANMAKU  REBUILD_TRT=$REBUILD_TRT  SET_DEFAULT_VIDEO=$SET_DEFAULT_VIDEO"

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
  for f in mpv.conf input.conf rife.vpy rife-light.vpy rife-half.vpy \
           sr_keys_helper.py vs_gpu_helpers.py \
           danmaku-config.json danmaku-credentials.json.example; do
    rm -f "$MPV_CFG_DIR/$f"
  done
  rm -rf "$MPV_CFG_DIR/scripts" "$MPV_CFG_DIR/shaders" \
         "$MPV_CFG_DIR/weights" "$MPV_CFG_DIR/fsrcnnx_cudnn" \
         "$MPV_CFG_DIR/fsrcnnx-cudnn"
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
