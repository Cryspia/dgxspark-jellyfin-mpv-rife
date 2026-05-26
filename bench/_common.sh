#!/usr/bin/env bash
# Shared helpers for bench/{color,fps,timing,robustness}.sh — worker
# lifecycle, mpv invocation, flag-file management. Sourced, not
# executed directly.
#
# Defaults read from the per-user dual config written by
# `install.sh install --dual-host` (~/.config/dgxspark-mpv/dual.conf).
# Anything in that file can be overridden per-run via env.

set -uo pipefail

# ─── paths (override via env) ────────────────────────────────────────
ENV_ROOT="${ENV_ROOT:-$HOME/miniforge3/envs/vsmpv}"
MPV="${MPV:-$ENV_ROOT/bin/mpv}"
PY="${PY:-$ENV_ROOT/bin/python}"
MPV_CFG="${MPV_CFG:-$HOME/.config/mpv}"
VPY="${VPY:-$MPV_CFG/rife.vpy}"

if [[ -z "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
DUAL_DIR="${DUAL_DIR:-$REPO_ROOT/dual_machine}"

# ─── dual networking ─────────────────────────────────────────────────
# Source the installed dual config (DUAL_HOST_IP / DUAL_WORKER_HOST /
# DUAL_RDMA_DEV / DUAL_RDMA_PORT). Each can still be overridden via
# the env vars below.
DUAL_CFG_FILE="${DUAL_CFG_FILE:-$HOME/.config/dgxspark-mpv/dual.conf}"
if [[ -r "$DUAL_CFG_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$DUAL_CFG_FILE"
fi

HOST_IP="${HOST_IP:-${DUAL_HOST_IP:-}}"
WORKER_IP="${WORKER_IP:-${DUAL_WORKER_HOST:-}}"
WORKER_USER="${WORKER_USER:-${DUAL_WORKER_USER:-ubuntu}}"
# Relative path — ssh's login shell resolves it under the remote
# user's home, so we don't have to know /home/$WORKER_USER vs
# /Users/$WORKER_USER vs anything else.
WORKER_DIR="${WORKER_DIR:-${DUAL_WORKER_DIR:-.local/share/dgxspark-mpv/worker}}"
RDMA_DEV="${RDMA_DEV:-${DUAL_RDMA_DEV:-rocep1s0f0}}"
RDMA_PORT="${RDMA_PORT:-${DUAL_RDMA_PORT:-29900}}"
MASTER_PORT="${MASTER_PORT:-$((30000 + (RANDOM ^ $$) % 5000))}"

# ─── default clips (override via env) ────────────────────────────────
CLIP_24="${CLIP_24:-$REPO_ROOT/bench/clips/sample-1080p-24.mp4}"
CLIP_120="${CLIP_120:-$REPO_ROOT/bench/clips/sample-1080p-120.mp4}"

# ─── conda activation ────────────────────────────────────────────────
if [[ "${CONDA_DEFAULT_ENV:-}" != "vsmpv" ]]; then
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
  conda activate vsmpv
fi

# ─── /tmp keybind flag-file save / restore ───────────────────────────
FLAG_FILES=(
  /tmp/fsrcnnx_variant
  /tmp/rife_disabled
  /tmp/dual_machine_disabled
  /tmp/dual_machine_sr_disabled
  /tmp/dual_machine_mult_override
)

flags_save_dir() {
  local d=$1
  mkdir -p "$d"
  for f in "${FLAG_FILES[@]}"; do
    [[ -e $f ]] && cp -p "$f" "$d/$(basename "$f")"
  done
}
flags_restore() {
  local d=$1
  for f in "${FLAG_FILES[@]}"; do
    local n=$d/$(basename "$f")
    if [[ -e $n ]]; then cp -p "$n" "$f"
    else rm -f "$f"; fi
  done
}
flags_reset() { for f in "${FLAG_FILES[@]}"; do rm -f "$f"; done; }
flags_setup() {
  local m=$1
  flags_reset
  case "$m" in
    single)         : ;;
    single_no_sr)   echo "off" > /tmp/fsrcnnx_variant ;;
    single_no_rife) touch /tmp/rife_disabled ;;
    dual)           : ;;
    dual_no_sr)     touch /tmp/dual_machine_sr_disabled ;;
    dual_no_interp) echo "1" > /tmp/dual_machine_mult_override ;;
    *) echo "unknown mode $m"; return 1 ;;
  esac
}

# ─── process cleanup (skips own ancestors) ───────────────────────────
_kill_pat() {
  local pat=$1 skip=$$ p=$$
  while [[ $p -gt 1 ]]; do
    p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
    [[ -z $p || $p = 0 || $p = 1 ]] && break
    skip="$skip|$p"
  done
  pgrep -f "$pat" 2>/dev/null \
    | awk -v s="^($skip)\$" '$1 !~ s' \
    | xargs -r kill -9 2>/dev/null
  return 0
}
kill_local_mpv() {
  _kill_pat 'mpv.*rife\.vpy'
  _kill_pat 'python.*from host_3proc'
  rm -f /dev/shm/dgxspark_host_3p_*.dat \
        /dev/shm/dgxspark_host_cc_cache_*.dat 2>/dev/null
}
kill_remote_worker() {
  timeout 10 ssh -n "$WORKER_USER@$WORKER_IP" '
    pkill -9 -f "worker\.py$" 2>/dev/null
    pkill -9 -f "from worker_3proc" 2>/dev/null
    rm -f /dev/shm/dgxspark_worker_3p_*.dat 2>/dev/null
    sleep 0.2
  ' 2>/dev/null || true
}
cleanup_all() { kill_local_mpv; kill_remote_worker; }

# ─── worker launch ────────────────────────────────────────────────
# By default rsyncs the current dual_machine + vs_gpu_helpers tree
# to $WORKER_DIR so the bench tests the dev-tree code, not what the
# guest has installed. Set BENCH_USE_INSTALLED=1 to skip the rsync
# and launch the worker.py that install.sh put in place on the
# guest — useful for verifying a clean install.
_WORKER_STARTED=0
start_worker() {
  [[ $_WORKER_STARTED = 1 ]] && return 0
  kill_remote_worker
  if [[ "${BENCH_USE_INSTALLED:-0}" != "1" ]]; then
    rsync -azq \
      "$DUAL_DIR/worker.py" "$DUAL_DIR/rdma_transport.py" \
      "$DUAL_DIR/worker_3proc.py" "$DUAL_DIR/mp_pipeline.py" \
      "$DUAL_DIR/cc_cache.py" "$DUAL_DIR/queue_mgr.py" \
      "$MPV_CFG/vs_gpu_helpers.py" \
      "$WORKER_USER@$WORKER_IP:$WORKER_DIR/"
    ssh -n "$WORKER_USER@$WORKER_IP" "mkdir -p ~/.config/mpv/fsrcnnx-cudnn"
    rsync -azq --delete --exclude='__pycache__' \
      "$MPV_CFG/fsrcnnx-cudnn/fsrcnnx_cudnn/" \
      "$WORKER_USER@$WORKER_IP:.config/mpv/fsrcnnx-cudnn/fsrcnnx_cudnn/"
  fi
  ssh -n "$WORKER_USER@$WORKER_IP" "
    cd $WORKER_DIR && \
    ( nohup env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        DUAL_WORKER_HOST=$HOST_IP \
        DUAL_RDMA_DEV=$RDMA_DEV DUAL_RDMA_PORT=$RDMA_PORT \
        ${WORKER_EXTRA_ENV:-} \
        PYTHONPATH=/usr/lib/python3/dist-packages \
        $PY -B worker.py </dev/null >/tmp/dual_worker.log 2>&1 & )
  "
  for _ in {1..60}; do
    if ssh -n "$WORKER_USER@$WORKER_IP" \
        "grep -q 'waiting for host' /tmp/dual_worker.log 2>/dev/null"; then
      _WORKER_STARTED=1
      return 0
    fi
    sleep 0.5
  done
  echo "ERROR: worker startup timeout"
  ssh -n "$WORKER_USER@$WORKER_IP" "tail -50 /tmp/dual_worker.log"
  return 1
}
stop_worker() { kill_remote_worker; _WORKER_STARTED=0; }

# ─── mpv launch: $1 output path (empty = vo=null), $2 mode, $3 clip,
#                 $4 N frames, $5 CF, $6 log path, $7 extra env (str) ─
run_mpv() {
  local out=$1 mode=$2 clip=$3 n=$4 cf=$5 log=$6 extra="${7:-}"
  flags_setup "$mode" || return 1
  local out_args=(--vo=null --ao=null)
  [[ -n $out ]] && out_args=(--ovc=ffv1 --of=matroska -o "$out")
  if [[ $mode = single* ]]; then
    env PYTHONPATH=/usr/lib/python3/dist-packages \
        WARMUP_DISABLE=1 \
        $extra \
        "$MPV" --no-config \
          --vf=vapoursynth="$VPY":buffered-frames=2:concurrent-frames=$cf \
          --untimed --hwdec=no --no-audio \
          --frames="$n" \
          "${out_args[@]}" \
          "$clip" 2> "$log" >/dev/null
  else
    local mult=2
    [[ $mode = dual_no_interp ]] && mult=1
    env DUAL_WORKER_HOST="$WORKER_IP" \
        DUAL_RDMA_DEV="$RDMA_DEV" DUAL_RDMA_PORT="$RDMA_PORT" \
        DUAL_INTERP_MULT=$mult \
        DUAL_REPORT_FPS=30 \
        WARMUP_DISABLE=1 \
        $extra \
        PYTHONPATH=/usr/lib/python3/dist-packages \
        "$MPV" --no-config \
          --vf=vapoursynth="$VPY":buffered-frames=2:concurrent-frames=$cf \
          --untimed --hwdec=no --no-audio \
          --frames="$n" \
          "${out_args[@]}" \
          "$clip" 2> "$log" >/dev/null
  fi
}

# Verify both clips exist before running anything.
require_clips() {
  local missing=0
  for c in "$@"; do
    if [[ ! -f $c ]]; then
      echo "ERROR: clip missing: $c"
      missing=1
    fi
  done
  if [[ $missing = 1 ]]; then
    echo "Generate them: $REPO_ROOT/bench/gen_clips.sh"
    return 1
  fi
}

needs_dual() {
  if [[ -z "${WORKER_IP:-}" ]]; then
    echo "ERROR: WORKER_IP unset — dual modes require a reachable worker"
    return 1
  fi
}
