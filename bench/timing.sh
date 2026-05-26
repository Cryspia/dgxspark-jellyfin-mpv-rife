#!/usr/bin/env bash
# bench/timing.sh — per-task GPU compute, idle, RDMA transit, queue
# depth + GPU utilisation for the dual chain. Runs ONE mode (default
# `dual`) with DUAL_PROFILE=1 + DUAL_RDMA_PROF=1 + DUAL_GPU_IDLE_DBG=1
# enabled and tail-parses the worker / host logs into a summary table.
#
# Usage:
#   bench/timing.sh                     # mode=dual
#   MODE=dual_no_sr bench/timing.sh
#   MODE=dual_no_interp N=800 bench/timing.sh
#
# Output groups (all averaged over the steady-state window):
#   • worker: per-task-type GPU kernel_ms / idle_before_ms / n
#   • worker: rolling GPU utilisation (kernel / (kernel + idle))
#   • worker: per-task-type RDMA send / round-trip latency
#   • host:   per-thread pop wait_avg, queue depth at pop, lock-hold
#
# This is a diagnostic bench, not a regression metric — the absolute
# numbers shift with kernel changes; what matters is the relative
# breakdown (is the bottleneck compute, comm, or scheduler?).

set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/_common.sh"

MODE="${MODE:-dual}"
N="${N:-600}"
CF="${CF:-24}"

case "$MODE" in
  dual|dual_no_sr|dual_no_interp) ;;
  *) echo "MODE must be dual / dual_no_sr / dual_no_interp"; exit 1 ;;
esac

require_clips "$CLIP_120" || exit 1
needs_dual || exit 1

TMPDIR=$(mktemp -d /tmp/bench_timing.XXXXXX)
flags_save_dir "$TMPDIR/saved_flags"
trap 'cleanup_all; flags_restore "$TMPDIR/saved_flags"; echo; echo "Artefacts: $TMPDIR"' EXIT

cleanup_all

export WORKER_EXTRA_ENV="DUAL_PROFILE=1 DUAL_RDMA_PROF=1 DUAL_GPU_IDLE_DBG=1"
start_worker || exit 1

HOST_EXTRA="DUAL_PROFILE=1"

echo
echo "=========================================================="
echo "=== timing — $MODE, CF=$CF, N=$N"
echo "===          src=$CLIP_120"
echo "=========================================================="

log_host="$TMPDIR/host.log"
log_worker="$TMPDIR/worker.log"

echo "  rendering..."
kill_local_mpv
run_mpv "" "$MODE" "$CLIP_120" "$N" "$CF" "$log_host" "$HOST_EXTRA"

# Pull worker log down for offline parsing.
ssh -n "$WORKER_USER@$WORKER_IP" "cat /tmp/dual_worker.log" > "$log_worker" 2>/dev/null

stop_worker

echo
echo "─── WORKER: GPU compute + idle totals (last sample) ───────"
grep -E 'PROFILE wall=' "$log_worker" 2>/dev/null | tail -1 \
  | sed 's/^/  /' || echo "  (no PROFILE wall lines)"

echo
echo "─── WORKER: per-task GPU breakdown (last sample, one per type) ─"
# `PROFILE  TYPE: n=X kernel=X.XXms idle_before=X.XXms ...`
# A new sample starts when "PROFILE wall=" fires; we want the last
# block of per-type lines, so pull the 5 lines immediately before
# end-of-log that match the per-type pattern.
grep -E '^.*PROFILE +[A-Z_]+:' "$log_worker" 2>/dev/null \
  | tail -4 | sed 's/^[^:]*:/  /' || echo "  (no per-type lines)"

echo
echo "─── WORKER: GPU stream util (last 3 samples, DUAL_GPU_IDLE_DBG) ─"
grep -E 'GPU stream: kernel=' "$log_worker" 2>/dev/null \
  | tail -3 | sed 's/^[^:]*:/  /' || echo "  (no GPU stream lines)"

echo
echo "─── WORKER: RDMA transit per task type (last sample) ──────"
grep -E 'RDMA-PROF' "$log_worker" 2>/dev/null | tail -1 \
  | sed 's/ | /\n    /g;s/^[^:]*:/  /' || echo "  (no RDMA-PROF lines)"

echo
echo "─── HOST: pop wait + queue depth (last sample) ────────────"
grep -E '\[profile\] (HOST pops|GUEST pops|DEPTH NOW|LOCK-HELD)' "$log_host" 2>/dev/null \
  | tail -4 | sed 's/.*\[profile\] /  /' || echo "  (no [profile] lines)"

echo
echo "─── steady-state dispatcher fps ───────────────────────────"
grep -E '^\[native_dispatcher fps\]' "$log_host" 2>/dev/null \
  | tail -1 | sed 's/^/  /' || echo "  (no dispatcher fps line)"
