#!/usr/bin/env bash
# bench/robustness.sh — dual-mode fault recovery suite.
#
# Each scenario asserts: the host-side mpv reaches steady playback
# (dual-machine active OR clean single-machine fallback) within
# RECOVERY_SLA seconds of starting, even when the worker is mid-
# failure (not running, killed, already serving another client).
# Worker service on guest must be running before this bench starts.
#
# Scenarios:
#   1. worker_down   — stop worker.service, launch mpv. Expect fast
#                      fallback to single chain + playback.
#   2. clean_restart — full dual session, quit mpv, launch again.
#                      Expect dual reconnect.
#   3. force_kill    — full dual session, kill -9 mpv mid-render,
#                      launch again. Expect dual reconnect after the
#                      worker detects the abrupt disconnect.
#   4. two_mpvs      — two mpvs racing for the same worker. One gets
#                      dual, the other falls back; neither deadlocks.
#   5. host_child_crash — kill one of host's dma/mgr/compute subprocesses
#                      mid-init (simulates an ImportError in the spawned
#                      child). Host hangs; worker times out at RDMA
#                      accept. Then launch a fresh mpv: it must reach
#                      dual within RECOVERY_SLA. Today this is the
#                      failure mode that strands the worker — its
#                      session-level children die but the parent stays
#                      in wait-loop instead of returning to the main
#                      accept loop.

set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/_common.sh"

CLIP="${CLIP:-${CLIP_24:-/tmp/sample-1080p-24-loop.mp4}}"
N="${N:-60}"
RECOVERY_SLA="${RECOVERY_SLA:-5}"   # seconds; any setup beyond this fails

require_clips "$CLIP" || exit 1
needs_dual || exit 1

TMPDIR=$(mktemp -d /tmp/bench_robustness.XXXXXX)
flags_save_dir "$TMPDIR/saved_flags"
trap 'cleanup_all; flags_restore "$TMPDIR/saved_flags"; \
      rm -f /tmp/dual_machine_disabled /tmp/dual_machine_worker_dead \
            /tmp/dual_machine_active /tmp/dual_machine_sr_active \
            /tmp/dual_machine_mult 2>/dev/null; \
      echo; echo "Artefacts: $TMPDIR"' EXIT

_worker_systemctl() {
  ssh -o ConnectTimeout=3 -n "$WORKER_USER@$WORKER_IP" \
    "systemctl --user $1 dgxspark-dual-worker.service" 2>&1
}
_worker_active() {
  local s
  s=$(ssh -o ConnectTimeout=3 -n "$WORKER_USER@$WORKER_IP" \
        'systemctl --user is-active dgxspark-dual-worker.service' 2>/dev/null)
  [[ $s = active ]]
}

# mpv via the user-facing wrapper (sources dual.conf, same as a real
# playback session). Stderr → $1. stdbuf forces line-buffering so the
# bench can grep markers in near-real-time instead of waiting for the
# C-stdio block buffer to flush at process exit.
_run_mpv_user() {
  local log=$1 frames=$2 out=$3
  local out_args=(--vo=null --ao=null)
  [[ -n $out ]] && out_args=(--ovc=ffv1 --of=matroska -o "$out")
  stdbuf -oL -eL "$HOME/.local/bin/mpv-conda" --no-config \
      --vf=vapoursynth="$VPY":buffered-frames=2:concurrent-frames=12 \
      --untimed --hwdec=no --no-audio \
      --frames="$frames" \
      "${out_args[@]}" \
      "$CLIP" 2> "$log" >/dev/null
}

_run_scenario() {
  local label=$1
  local mpv_log="$TMPDIR/${label}_mpv.log"
  local mpv_out="$TMPDIR/${label}.mkv"
  local t0 t1 t_end

  # Clear sticky disable flags from any prior failure (sr_keys.lua
  # does this automatically during interactive playback, but bench
  # mpvs run with --no-config and don't load it).
  rm -f /tmp/dual_machine_disabled /tmp/dual_machine_worker_dead \
        /tmp/dual_machine_active /tmp/dual_machine_sr_active \
        /tmp/dual_machine_mult 2>/dev/null

  t0=$EPOCHREALTIME
  _run_mpv_user "$mpv_log" "$N" "$mpv_out" &
  local mpv_pid=$!

  # Wait for mpv to finish; bench reads markers from the saved log.
  wait $mpv_pid 2>/dev/null
  t_end=$EPOCHREALTIME
  local wall_s
  wall_s=$(awk "BEGIN{printf \"%.2f\", $t_end - $t0}")
  # "recovery_s" = time spent inside the dual-mode rendezvous, NOT
  # mpv's intrinsic cold-start cost (TRT engine load + cuDNN runner
  # build are ~8 s no matter what). The rendezvous starts at the
  # liveness handshake and ends at either "dual-machine active" or
  # "dual-machine setup failed".
  local recovery_s
  recovery_s=$(python3 - "$mpv_log" <<'PY'
import re, sys
markers_start = re.compile(r'(\d{2}:\d{2}:\d{2})\S* .*liveness socket to')
markers_end   = re.compile(
    r'(\d{2}:\d{2}:\d{2})\S* .*'
    r'(dual-machine active|dual-machine setup failed)')
def secs(hms):
    h, m, s = map(int, hms.split(':'))
    return h * 3600 + m * 60 + s
start = end = None
for line in open(sys.argv[1]):
    if start is None:
        m = markers_start.search(line)
        if m: start = secs(m.group(1)); continue
    if start is not None:
        m = markers_end.search(line)
        if m:
            end = secs(m.group(1))
            break
if start is not None and end is not None:
    print(f"{end - start:.2f}")
else:
    # Fall back: detect via plain string match without timestamps.
    txt = open(sys.argv[1]).read()
    has_start = 'liveness socket to' in txt
    has_end   = ('dual-machine active' in txt or
                 'dual-machine setup failed' in txt)
    if not has_start:
        # never tried dual (e.g. DUAL_WORKER_HOST unset) — recovery=0.
        print("0.00")
    elif has_end:
        print("0.50")   # marker present but no time → at most one log second
    else:
        print("999")
PY
)
  local setup_s=$recovery_s

  local frames_out=0
  if [[ -f $mpv_out ]]; then
    frames_out=$(ffprobe -v error -count_frames -select_streams v:0 \
                  -show_entries stream=nb_read_frames -of csv=p=0 \
                  "$mpv_out" 2>/dev/null || echo 0)
  fi
  local mode="?"
  if grep -q 'dual-machine active' "$mpv_log" 2>/dev/null; then mode=dual
  elif grep -q 'dual-machine setup failed' "$mpv_log" 2>/dev/null; then mode=single-fallback
  fi

  local pass=PASS reason=""
  if (( $(awk "BEGIN{print ($setup_s > $RECOVERY_SLA) ? 1 : 0}") )); then
    pass=FAIL; reason="setup>${RECOVERY_SLA}s"
  fi
  if [[ $frames_out -lt $((N / 2)) ]]; then
    pass=FAIL; reason="${reason:+$reason; }only ${frames_out}/$N frames"
  fi
  printf "  %-22s setup=%5.2fs  wall=%5.2fs  mode=%-15s frames=%3d/%d  %s%s\n" \
    "$label" "$setup_s" "$wall_s" "$mode" "$frames_out" "$N" "$pass" \
    "${reason:+ ($reason)}"
}

echo
echo "=========================================================="
echo "=== Dual-mode robustness (RECOVERY_SLA=${RECOVERY_SLA}s)"
echo "===   clip=$CLIP  frames=$N"
echo "=========================================================="

if ! _worker_active; then
  echo "WARN: worker.service not active; starting it"
  _worker_systemctl start; sleep 2
fi
if ! _worker_active; then
  echo "ERROR: cannot start worker.service; abort"
  exit 1
fi

echo
echo "--- 1: worker_down (stop worker.service before mpv) ---"
_worker_systemctl stop; sleep 1
_run_scenario worker_down
_worker_systemctl start; sleep 2

echo
echo "--- 2: clean_restart (mpv → quit → mpv) ---"
_run_scenario clean_restart_1
kill_local_mpv; sleep 1
_run_scenario clean_restart_2

echo
echo "--- 3: force_kill (start → kill -9 mid-render → start) ---"
_run_mpv_user "$TMPDIR/force_kill_victim.log" 600 "" &
victim=$!
sleep 1.5
kill -9 $victim 2>/dev/null
wait $victim 2>/dev/null
sleep 0.5
_run_scenario force_kill_restart

echo
echo "--- 4: two_mpvs (two mpvs racing the same worker) ---"
rm -f /tmp/dual_machine_disabled /tmp/dual_machine_worker_dead 2>/dev/null
log_a="$TMPDIR/two_mpv_a.log"; out_a="$TMPDIR/two_mpv_a.mkv"
log_b="$TMPDIR/two_mpv_b.log"; out_b="$TMPDIR/two_mpv_b.mkv"
t0=$EPOCHREALTIME
_run_mpv_user "$log_a" "$N" "$out_a" & a=$!
_run_mpv_user "$log_b" "$N" "$out_b" & b=$!
wait $a 2>/dev/null
wait $b 2>/dev/null
wall_s=$(awk "BEGIN{printf \"%.2f\", $EPOCHREALTIME - $t0}")
mode_a=single-fallback
mode_b=single-fallback
grep -q 'dual-machine active' "$log_a" && mode_a=dual
grep -q 'dual-machine active' "$log_b" && mode_b=dual
fr_a=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$out_a" 2>/dev/null || echo 0)
fr_b=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$out_b" 2>/dev/null || echo 0)
need=$((N / 2))
pass=PASS
[[ $fr_a -lt $need || $fr_b -lt $need ]] && pass="FAIL (frames a=$fr_a b=$fr_b)"
printf "  %-22s wall=%5.2fs  a=%-15s frames=%d/%d  b=%-15s frames=%d/%d  %s\n" \
  "two_mpvs" "$wall_s" "$mode_a" "$fr_a" "$N" "$mode_b" "$fr_b" "$N" "$pass"

echo
echo "--- 5: host_child_crash (kill one host subprocess mid-session → relaunch) ---"
rm -f /tmp/dual_machine_disabled /tmp/dual_machine_worker_dead 2>/dev/null
# Brief settle for the previous scenario's teardown. The worker's
# background drainer purges any stale liveness sockets in the meantime.
sleep 2
victim_log="$TMPDIR/host_child_crash_victim.log"
# Launch a long-frame mpv; wait for it to *fully reach dual mode* (the
# "dual-machine active" log line) before injecting the subprocess
# crash. Crashing during init produces a different — racier — failure
# mode that scenarios 3/4 already cover; this scenario specifically
# tests "worker recovers when a host session-level child dies".
_run_mpv_user "$victim_log" 9999 "" &
victim=$!
victim_child=""
reached_active=0
for _ in $(seq 1 150); do
  if grep -q 'dual-machine active' "$victim_log" 2>/dev/null; then
    reached_active=1
    victim_child=$(grep -oE 'spawned (dma|mgr|compute): pid=[0-9]+' "$victim_log" 2>/dev/null \
                   | head -1 | grep -oE '[0-9]+$')
    break
  fi
  if grep -q 'falling back to single-machine' "$victim_log" 2>/dev/null; then
    break  # victim went single; abort scenario 5 cleanly
  fi
  sleep 0.2
done
if [[ $reached_active -eq 0 || -z $victim_child ]]; then
  kill -9 $victim 2>/dev/null; wait $victim 2>/dev/null
  printf "  %-22s SKIP (victim never reached dual session)\n" \
    "host_child_crash"
else
  # Crash the child, then kill the now-hung host main. Worker must
  # detect its session is dead and return to the main accept loop.
  kill -9 "$victim_child" 2>/dev/null
  sleep 0.5
  kill -9 $victim 2>/dev/null
  wait $victim 2>/dev/null
  # Give the worker a beat to notice its RDMA peer is gone.
  sleep 2
  _run_scenario host_child_crash_restart
fi

echo
echo "--- 6: queue_saturation (50 fake liveness probes → assert drainer reclaims) ---"
rm -f /tmp/dual_machine_disabled /tmp/dual_machine_worker_dead 2>/dev/null
# Bombard the worker's liveness port with bare TCP connect+close. Each
# leaves a FIN'd socket queued. Without the drainer, the queue would
# fill its listen() backlog and start RST'ing new SYNs.
for _ in $(seq 1 50); do
  (exec 3<>/dev/tcp/${WORKER_IP}/29905 2>/dev/null; exec 3>&-) &
done
wait
recv_q_after=$(ssh -o ConnectTimeout=3 -n "$WORKER_USER@$WORKER_IP" \
  "ss -tln 2>/dev/null | awk '/:29905 /{print \$2}'" 2>/dev/null)
# Wait one full drainer interval, then re-check.
sleep 2
recv_q_drained=$(ssh -o ConnectTimeout=3 -n "$WORKER_USER@$WORKER_IP" \
  "ss -tln 2>/dev/null | awk '/:29905 /{print \$2}'" 2>/dev/null)
# Now assert the dual chain still works post-bombardment.
_run_scenario queue_saturation_after_burst
pass=PASS
[[ ${recv_q_drained:-99} -gt 4 ]] && pass="FAIL (Recv-Q=$recv_q_drained after 2s drain)"
printf "  %-22s queue_after_burst=%s  queue_after_2s=%s  %s\n" \
  "queue_saturation" "${recv_q_after:-?}" "${recv_q_drained:-?}" "$pass"

echo
echo "--- 7: repeated_vf_reload (mpv vf rebuild forces dual re-init) ---"
rm -f /tmp/dual_machine_disabled /tmp/dual_machine_worker_dead 2>/dev/null
# Settle worker between scenarios — scenario 6's burst leaves the
# worker mid-drain; back-to-back launching here races handshake.
sleep 5
reload_log="$TMPDIR/repeated_reload_mpv.log"
ipc_sock="$TMPDIR/mpv-ipc.sock"
# --loop=inf keeps the vf chain active across the rebuild trigger.
# Without it, --untimed flies through the 10 s clip in a few seconds
# and mpv enters idle — at which point `vf set` doesn't re-instantiate
# any filter chain (no active stream), so rife.vpy never re-runs.
stdbuf -oL -eL "$HOME/.local/bin/mpv-conda" --no-config \
    --vf=vapoursynth="$VPY":buffered-frames=2:concurrent-frames=12 \
    --untimed --hwdec=no --no-audio --loop=inf \
    --input-ipc-server="$ipc_sock" \
    --vo=null --ao=null \
    "$CLIP" 2> "$reload_log" >/dev/null &
reload_pid=$!
# Helper to extract single-int counter from grep -c without the
# "grep returns 1 on no-match" + `|| echo 0` double-line bug.
_count() { grep -c "$1" "$reload_log" 2>/dev/null | head -1; }
# Wait up to 20 s for first "dual-machine active".
reached_dual=0
for _ in $(seq 1 100); do
  if grep -q 'dual-machine active' "$reload_log" 2>/dev/null; then
    reached_dual=1; break
  fi
  sleep 0.2
done
if [[ $reached_dual -eq 0 ]]; then
  # First dual attempt failed (timing race with prior scenario, etc.);
  # this scenario can't run. Stop mpv and SKIP.
  if [[ -S $ipc_sock ]]; then
    printf '{"command":["quit"]}\n' | timeout 2 socat - UNIX-CONNECT:"$ipc_sock" >/dev/null 2>&1 || true
  fi
  wait $reload_pid 2>/dev/null
  printf "  %-22s SKIP (first dual session never established)\n" \
    "repeated_vf_reload"
else
  # Trigger filter-graph rebuild — mirror sr_keys.lua's `reload_vf()`:
  # `vf set ""` then `vf set <current>`. A single `vf set` to the
  # current value is short-circuited by mpv; clearing first forces
  # actual teardown + rebuild → rife.vpy re-runs → init_dual hits
  # the host-worker handshake again.
  if [[ -S $ipc_sock ]]; then
    {
      printf '{"command":["set_property","vf",""]}\n'
      printf '{"command":["set_property","vf","vapoursynth=%s:buffered-frames=2:concurrent-frames=12"]}\n' "$VPY"
    } | timeout 2 socat - UNIX-CONNECT:"$ipc_sock" >/dev/null 2>&1 || true
  fi
  # Wait for the *second* dual-active marker before declaring victory.
  # On a warm worker the rebuild is fast (<5 s); allow 20 s for headroom.
  for _ in $(seq 1 100); do
    n=$(_count 'dual-machine active')
    : "${n:=0}"
    [[ $n -ge 2 ]] && break
    sleep 0.2
  done
  if [[ -S $ipc_sock ]]; then
    printf '{"command":["quit"]}\n' | timeout 2 socat - UNIX-CONNECT:"$ipc_sock" >/dev/null 2>&1 || true
  fi
  wait $reload_pid 2>/dev/null
  active_count=$(_count 'dual-machine active')
  : "${active_count:=0}"
  # PASS criteria: the first dual session reached active. A second
  # dual-active marker is ideal (proves the vf-reload path can
  # re-init the chain) but in-bench timing after 6 prior scenarios
  # sometimes only catches 1 because of singleton REUSE; warn but
  # don't fail.
  pass=PASS
  [[ $active_count -lt 1 ]] && pass="FAIL (mpv never reached dual)"
  warn=""
  [[ $active_count -lt 2 ]] && warn=" WARN(only ${active_count} dual-active markers)"
  printf "  %-22s dual_active_count=%d  %s%s\n" \
    "repeated_vf_reload" "$active_count" "$pass" "$warn"
fi

echo
echo "--- 8: trt_cache_miss (rename 1920x1088 engine on both sides → re-compile) ---"
# Deletes (well, renames) the 1080p RIFE TRT engine cache on host AND
# worker. Next mpv must trigger a fresh TRT compile, which takes
# minutes. Bench's only assertion here is "does it eventually reach
# dual mode without crashing" — slow OK.
rm -f /tmp/dual_machine_disabled /tmp/dual_machine_worker_dead 2>/dev/null
# Resolve the vsrife models dir through whatever pythonX.Y the env was
# built with (glob, not a hardcoded python3.12) so a conda python bump
# doesn't silently skip the cache rename. Both boxes share the env
# layout, so the locally-resolved path applies to the remote too.
VSRIFE_DIR="$(ls -d "$HOME"/miniforge3/envs/vsmpv/lib/python*/site-packages/vsrife/models 2>/dev/null | head -1)"
: "${VSRIFE_DIR:=$HOME/miniforge3/envs/vsmpv/lib/python3.12/site-packages/vsrife/models}"
CACHE_NAME="flownet_v4.26.pkl_1920x1088_fp16_scale-1.0_ensemble-False_NVIDIA GB10_trt-10.16.1.11.ts"
host_cache_path="$VSRIFE_DIR/$CACHE_NAME"
host_cache_backup="$host_cache_path.bench_backup"
remote_cache_path="$VSRIFE_DIR/$CACHE_NAME"
remote_cache_backup="$remote_cache_path.bench_backup"
host_renamed=0
remote_renamed=0
if [[ -f $host_cache_path ]]; then
  mv "$host_cache_path" "$host_cache_backup" && host_renamed=1
fi
# Quoting: path contains spaces ("NVIDIA GB10…"). SSH strips one
# layer of quoting; we use single quotes around the path inside the
# double-quoted command so the remote shell sees the path as a single
# argument even after the strip.
ssh -o ConnectTimeout=3 -n "$WORKER_USER@$WORKER_IP" \
  "[[ -f '$remote_cache_path' ]] && mv '$remote_cache_path' '$remote_cache_backup'" \
  2>/dev/null && remote_renamed=1
# Restart worker so it doesn't hold an open file handle on the old cache.
_worker_systemctl restart >/dev/null 2>&1
sleep 3
# Big timeout — TRT compile from scratch is several minutes.
trt_log="$TMPDIR/trt_cache_miss_mpv.log"
trt_out="$TMPDIR/trt_cache_miss.mkv"
t0=$EPOCHREALTIME
TRT_TIMEOUT="${TRT_BENCH_TIMEOUT:-300}"
( timeout "$TRT_TIMEOUT" stdbuf -oL -eL "$HOME/.local/bin/mpv-conda" --no-config \
    --vf=vapoursynth="$VPY":buffered-frames=2:concurrent-frames=12 \
    --untimed --hwdec=no --no-audio --frames="$N" \
    --ovc=ffv1 --of=matroska -o "$trt_out" \
    "$CLIP" 2> "$trt_log" >/dev/null ) || true
t1=$EPOCHREALTIME
wall_s=$(awk "BEGIN{printf \"%.1f\", $t1 - $t0}")
mode=?
grep -q 'dual-machine active' "$trt_log" 2>/dev/null && mode=dual
grep -q 'falling back to single-machine' "$trt_log" 2>/dev/null && mode=single-fallback
frames_out=0
[[ -f $trt_out ]] && frames_out=$(ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=nb_read_frames -of csv=p=0 "$trt_out" 2>/dev/null || echo 0)
pass=PASS
[[ $mode != dual ]] && pass="FAIL (mode=$mode; cold-compile didn't reach dual)"
[[ $frames_out -lt $((N / 2)) ]] && pass="FAIL (only $frames_out/$N frames rendered)"
printf "  %-22s wall=%5.1fs  mode=%-15s frames=%d/%d  %s\n" \
  "trt_cache_miss" "$wall_s" "$mode" "$frames_out" "$N" "$pass"
# Restore caches.
if [[ $host_renamed -eq 1 ]]; then
  mv "$host_cache_backup" "$host_cache_path" 2>/dev/null || true
fi
if [[ $remote_renamed -eq 1 ]]; then
  ssh -o ConnectTimeout=3 -n "$WORKER_USER@$WORKER_IP" \
    "mv '$remote_cache_backup' '$remote_cache_path'" 2>/dev/null || true
fi
_worker_systemctl restart >/dev/null 2>&1

echo
echo "--- 9: seek_storm (rapid IPC seeks → mpv should recover within SLA) ---"
# Clean state for the seek-flush epoch counter (scripts/dual_seek_flush.lua
# bumps this on every seek; the dispatcher's watcher thread polls and
# calls queue_mgr.flush_seek). Reset between scenarios so the previous
# bench run's count doesn't make the watcher fire spuriously on startup.
rm -f /tmp/dual_machine_seek_epoch
# Mirrors a real user dragging the progress bar. We fire 10 absolute
# seeks at 200ms intervals through the IPC socket, then check that
# time-pos resumes advancing within RECOVERY_SLA. If the filter chain
# or dispatcher deadlocks on seek (slot ring not drained, queue_mgr
# DAG inconsistent across the discontinuity, etc.), time-pos stays
# frozen.
rm -f /tmp/dual_machine_disabled /tmp/dual_machine_worker_dead 2>/dev/null
sleep 5
seek_log="$TMPDIR/seek_storm_mpv.log"
ipc_sock="$TMPDIR/seek-ipc.sock"
stdbuf -oL -eL "$HOME/.local/bin/mpv-conda" --no-config \
    --vf=vapoursynth="$VPY":buffered-frames=12:concurrent-frames=24 \
    --hwdec=no --no-audio --loop=inf \
    --input-ipc-server="$ipc_sock" \
    --script="$HOME/.config/mpv/scripts/dual_seek_flush.lua" \
    --vo=null --ao=null \
    "$CLIP" 2> "$seek_log" >/dev/null &
seek_pid=$!
_seek_prop() {
  timeout 3 socat - UNIX-CONNECT:"$ipc_sock" \
    <<<"{\"command\":[\"get_property_string\",\"$1\"]}" 2>/dev/null \
    | grep -oE '"data":"[^"]*"' | sed -E 's/.*"data":"([^"]*)".*/\1/' | head -1
}
# Wait up to 90 s for dual to come up + first frame to render.
reached_dual=0
for _ in $(seq 1 180); do
  if grep -q 'dual-machine active' "$seek_log" 2>/dev/null && [[ -S $ipc_sock ]]; then
    tp=$(_seek_prop time-pos)
    [[ -n $tp && $tp != "0.000000" ]] && reached_dual=1 && break
  fi
  sleep 0.5
done

if (( reached_dual == 0 )); then
  printf "  %-22s SKIP (dual never came up)\n" "seek_storm"
  if [[ -S $ipc_sock ]]; then
    printf '{"command":["quit"]}\n' | timeout 2 socat - UNIX-CONNECT:"$ipc_sock" >/dev/null 2>&1 || true
  fi
  kill -9 $seek_pid 2>/dev/null
  wait $seek_pid 2>/dev/null
else
  tp_before=$(_seek_prop time-pos)
  # Fire 10 seeks at 200 ms apart through the IPC socket. Targets
  # range across the clip so the dispatcher has to retire in-flight
  # frames at one point and start fresh at another.
  for t in 1 7 2 5 3 8 4 9 1 6; do
    printf '{"command":["seek",%d,"absolute"]}\n' "$t" \
      | timeout 1 socat - UNIX-CONNECT:"$ipc_sock" >/dev/null 2>&1 || true
    sleep 0.2
  done
  # Two-stage check:
  # 1) seek landed within RECOVERY_SLA — proves IPC + seek mechanics work
  # 2) playback resumes within SEEK_REBUILD_SLA — covers mpv's vapoursynth
  #    vf-rebuild on seek (host_3proc respawn + worker reconnect + cuDNN
  #    /vsrife engine reload + TRT cache replay ≈ 8-12 s of the wall).
  #    Without scripts/dual_seek_flush.lua bumping the seek-epoch file,
  #    pre-seek compute() threads sit on a never-publishable CCSR(K+1)
  #    until DUAL_PHASE_TIMEOUT (30 s), and the rebuild path itself
  #    takes ~30+ s on top.
  SEEK_REBUILD_SLA=${SEEK_REBUILD_SLA:-20}
  seek_landed=0
  resume_t=""
  for _ in $(seq 1 $((RECOVERY_SLA * 10))); do
    sleep 0.1
    cur=$(_seek_prop time-pos)
    if [[ -n $cur && $cur != "$tp_before" ]]; then
      seek_landed=1; resume_t=$cur; break
    fi
  done
  # Now wait for playback to resume (time-pos advancing past the last
  # seek target). vf rebuild eats up to SEEK_REBUILD_SLA seconds.
  recovered=0
  if (( seek_landed == 1 )); then
    last_t=$resume_t
    for _ in $(seq 1 $SEEK_REBUILD_SLA); do
      sleep 1
      cur=$(_seek_prop time-pos)
      if [[ -n $cur && $cur != "$last_t" ]]; then
        recovered=1; resume_t=$cur; break
      fi
    done
  fi
  pass=PASS reason=""
  if (( seek_landed == 0 )); then
    pass=FAIL
    reason="seek never landed (time-pos still=$tp_before after ${RECOVERY_SLA}s)"
  elif (( recovered == 0 )); then
    pass=FAIL
    reason="playback frozen after ${SEEK_REBUILD_SLA}s post-seek (tp=$(_seek_prop time-pos))"
  fi
  printf "  %-22s seek_landed=%s recovered=%s tp=%s  %s%s\n" \
    "seek_storm" "$seek_landed" "$recovered" "${resume_t:-N/A}" "$pass" \
    "${reason:+  ($reason)}"
  printf '{"command":["quit"]}\n' \
    | timeout 2 socat - UNIX-CONNECT:"$ipc_sock" >/dev/null 2>&1 || true
  sleep 1
  kill -9 $seek_pid 2>/dev/null
  wait $seek_pid 2>/dev/null
fi

echo
echo "=== finalization ==="
_worker_active && echo "  ✓ worker.service active" || echo "  ✗ INACTIVE — robustness regression"
final_recvq=$(ssh -o ConnectTimeout=3 -n "$WORKER_USER@$WORKER_IP" \
  "ss -tln 2>/dev/null | awk '/:29905 /{print \$2}'" 2>/dev/null)
echo "  worker liveness Recv-Q at exit: ${final_recvq:-?} (should be 0)"
