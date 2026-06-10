#!/usr/bin/env bash
# bench/display_fps.sh — real-VO playback throughput on the actual display.
#
# Unlike fps.sh (vo=null, --untimed: pure compute ceiling) this launches
# the *installed* mpv with the production ~/.config/mpv config — real
# vo=gpu-next/vulkan, real KrigBilateral.glsl, timed display-resample —
# on the live Wayland session, so it measures what the user actually
# sees: the rate at which upscaled frames make it through the Vulkan
# render+present pipeline while CUDA (RIFE+FSRCNNX) runs concurrently.
#
# It reports two numbers:
#   • vf-fps (mean/median) — mpv's measured output rate of the vapoursynth
#     chain, sampled by display_fps_probe.lua. THIS is the on-screen FPS.
#   • dispatcher fps — the dual compute() return rate (DUAL_REPORT_FPS),
#     a cross-check on whether compute or render is the limiter.
#
# Env knobs:
#   CLIP        source file        (default: 1080p24 HDR10 sample in /tmp)
#   DURATION    seconds to play    (default 26)
#   WARMUP_S    steady-window skip (default 6)
#   KRIG        on|off             (default on) — off clears glsl-shaders
#   FULLSCREEN  1|0                (default 1)
#   MULT        1..4               (optional, forces dual interp mult)
#   LABEL       log/tag suffix     (default derived from KRIG)
#
# Any DUAL_* env already exported is passed through, so pacing experiments
# just prefix e.g. DUAL_RENDER_PACING=1 DUAL_RENDER_BUDGET_MS=16.7 …
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

MPV="${MPV:-$HOME/miniforge3/envs/vsmpv/bin/mpv}"
DUAL_CONF="${DUAL_CONF:-$HOME/.config/dgxspark-mpv/dual.conf}"
PROBE="$HERE/display_fps_probe.lua"

CLIP="${CLIP:-/tmp/sample-1080p24-hdr10.mp4}"   # NOT produced by gen_clips.sh — supply your own HDR10 clip via CLIP=
DURATION="${DURATION:-26}"
WARMUP_S="${WARMUP_S:-6}"
KRIG="${KRIG:-on}"
FULLSCREEN="${FULLSCREEN:-1}"
LABEL="${LABEL:-krig-$KRIG}"

[[ -f "$CLIP" ]] || { echo "ERROR: clip missing: $CLIP"; exit 1; }
[[ -f "$DUAL_CONF" ]] || { echo "ERROR: dual.conf missing — run install --dual-host"; exit 1; }

# Talk to the live GNOME/Wayland session (this shell has no display env).
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

# Replicate the mpv-conda wrapper environment.
export PYTHONHOME="$HOME/miniforge3/envs/vsmpv"
export PATH="$HOME/miniforge3/envs/vsmpv/bin:$PATH"
set -a; . "$DUAL_CONF"; set +a

export DUAL_REPORT_FPS="${DUAL_REPORT_FPS:-120}"
export DUAL_REPORT_WARMUP="${DUAL_REPORT_WARMUP:-200}"
export DISPLAYPROBE_WARMUP_S="$WARMUP_S"

# Optional forced interp multiplier (bypasses the ffprobe heuristic).
if [[ -n "${MULT:-}" ]]; then
  echo "$MULT" > /tmp/dual_machine_mult_override
  echo "  [forced mult=$MULT via /tmp/dual_machine_mult_override]"
fi

LOG="/tmp/display_fps_${LABEL}.log"

extra_args=()
if [[ "$KRIG" = off ]]; then
  extra_args+=(--glsl-shaders="")    # clear KrigBilateral.glsl from the chain
fi
if [[ "$FULLSCREEN" = 1 ]]; then
  extra_args+=(--fullscreen=yes)
else
  extra_args+=(--geometry=1920x1080 --fullscreen=no)
fi
# EXTRA_MPV: space-separated raw mpv flags appended last (override config).
if [[ -n "${EXTRA_MPV:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args+=($EXTRA_MPV)
fi

echo "=========================================================="
echo "=== display_fps — LABEL=$LABEL  KRIG=$KRIG  FS=$FULLSCREEN"
echo "===   clip=$CLIP  dur=${DURATION}s warmup=${WARMUP_S}s"
echo "===   DUAL_WORKER_HOST=$DUAL_WORKER_HOST mult=${MULT:-auto}"
echo "===   render-pacing=${DUAL_RENDER_PACING:-0} pace-ms=${DUAL_RENDER_PACE_MS:-} reactive=${DUAL_RENDER_PACE_REACTIVE:-0} budget=${DUAL_RENDER_BUDGET_MS:-}"
echo "=========================================================="

"$MPV" \
  --length="$DURATION" \
  --keep-open=no \
  --osd-level=0 \
  --script="$PROBE" \
  "${extra_args[@]}" \
  "$CLIP" 2> "$LOG" >/dev/null

echo
echo "--- dispatcher fps (compute() return rate) ---"
grep -E '^\[native_dispatcher fps\]' "$LOG" | tail -1 || echo "  (none — single-machine fallback? check $LOG)"
echo "--- vf-fps summary (on-screen FPS) ---"
grep -E '^\[displayprobe\] SUMMARY' "$LOG" || echo "  (no summary — check $LOG)"
echo "--- last 3 probe samples ---"
grep -E '^\[displayprobe\] t=' "$LOG" | tail -3
echo
echo "Full log: $LOG"
