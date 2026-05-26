#!/usr/bin/env bash
# bench/fps.sh — steady-state fps for the six production modes.
#
# wall-clock fps includes ~3 s of engine + cuDNN warm overhead, so
# the single-machine numbers read ~10-20% low vs steady state at the
# default N=400. Bump N for tighter estimates.
#
# Dual modes additionally print the dispatcher's rolling fps (last
# 30-frame window) which already excludes init.

set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/_common.sh"

N="${N:-400}"
CF="${CF:-24}"

require_clips "$CLIP_120" || exit 1

TMPDIR=$(mktemp -d /tmp/bench_fps.XXXXXX)
flags_save_dir "$TMPDIR/saved_flags"
trap 'cleanup_all; flags_restore "$TMPDIR/saved_flags"; echo; echo "Artefacts: $TMPDIR"' EXIT

_extract_dispatcher_fps() {
  grep -E '^\[native_dispatcher fps\]' "$1" | tail -1 \
    | grep -oE '=> [0-9.]+ fps' | grep -oE '[0-9.]+'
}

run_one() {
  local mode=$1 needs_worker=$2
  local log="$TMPDIR/${mode}_fps.log"
  kill_local_mpv
  if [[ $needs_worker = 1 ]]; then
    start_worker || { echo "    $mode: worker startup failed"; return 1; }
  fi
  local t0=$EPOCHREALTIME
  run_mpv "" "$mode" "$CLIP_120" "$N" "$CF" "$log"
  local t1=$EPOCHREALTIME
  local wc=$(awk "BEGIN{printf \"%.1f\", $N / ($t1 - $t0)}")
  local disp=""
  [[ $needs_worker = 1 ]] && disp=$(_extract_dispatcher_fps "$log")
  printf "    %-18s wall=%s fps" "$mode" "$wc"
  [[ -n $disp ]] && printf "  dispatcher=%s fps (steady-state)" "$disp"
  printf "\n"
}

echo
echo "=========================================================="
echo "=== fps — six modes, CF=$CF, N=$N"
echo "===       src=$CLIP_120"
echo "=========================================================="
cleanup_all

if [[ -n "${SKIP_SINGLE:-}" ]]; then
  echo "[single chain skipped (SKIP_SINGLE set)]"
else
  echo "[single chain]"
  run_one single         0
  run_one single_no_sr   0
  run_one single_no_rife 0
fi

if [[ -n "${SKIP_DUAL:-}" ]]; then
  echo
  echo "[dual chain skipped (SKIP_DUAL set)]"
else
  echo
  echo "[dual chain — worker reused across all three]"
  needs_dual || exit 1
  run_one dual           1
  run_one dual_no_sr     1
  run_one dual_no_interp 1
  stop_worker
fi
