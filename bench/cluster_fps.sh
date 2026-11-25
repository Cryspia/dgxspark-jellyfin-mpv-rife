#!/usr/bin/env bash
# Measure the installed cluster pipeline without stopping worker services.
set -euo pipefail
bench_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$bench_root/scripts/cluster_env.sh"
cluster_normalize_env
declare -A supplied_cluster_env=()
for name in ${!CLUSTER_@}; do supplied_cluster_env[$name]="${!name}"; done

cluster_config="${CLUSTER_CFG:-$HOME/.config/dgxspark-mpv/cluster.conf}"
if [[ ! -r $cluster_config && -z ${CLUSTER_CFG:-} ]]; then
  cluster_config="$HOME/.config/dgxspark-mpv/dual.conf"
fi
if [[ -r $cluster_config ]]; then
  set -a
  source "$cluster_config"
  set +a
  cluster_normalize_env
fi
for name in "${!supplied_cluster_env[@]}"; do
  printf -v "$name" '%s' "${supplied_cluster_env[$name]}"
  export "$name"
done

clip="${CLIP:-$bench_root/bench/clips/sample-1080p-120.mp4}"
env_prefix="${ENV_ROOT:-$HOME/miniforge3/envs/vsmpv}"
mpv_bin="${MPV:-$env_prefix/bin/mpv}"
vpy="${VPY:-$HOME/.config/mpv/rife.vpy}"
frames="${N:-3000}"
concurrent="${CF:-24}"
buffered="${BF:-12}"
mult="${MULT:-2}"
mode="${MODE:-auto}"

[[ -f $clip && -f $vpy && -x $mpv_bin ]] || { echo 'Missing clip, filter, or mpv binary.' >&2; exit 1; }
[[ -n ${CLUSTER_WORKER_HOSTS:-${CLUSTER_WORKER_HOST:-}} ]] || { echo 'No cluster worker configured.' >&2; exit 1; }
[[ $frames =~ ^[1-9][0-9]*$ && $concurrent =~ ^[1-9][0-9]*$ && $buffered =~ ^[1-9][0-9]*$ ]] \
  || { echo 'N, CF and BF must be positive integers.' >&2; exit 1; }
[[ $mult =~ ^[234]$ ]] || { echo 'MULT must be 2, 3 or 4.' >&2; exit 1; }
[[ ! -e /tmp/dual_machine_disabled ]] || { echo 'Cluster offload is disabled; enable it before benchmarking.' >&2; exit 1; }
if [[ -s /tmp/dual_machine_mult_override ]]; then
  read -r override < /tmp/dual_machine_mult_override || true
  [[ ${override:-} == "$mult" ]] || { echo 'The current F9 multiplier differs from MULT; select the requested multiplier first.' >&2; exit 1; }
fi
case "$mode" in
  auto) unset CLUSTER_TICTAC ;;
  normal) export CLUSTER_TICTAC=0 ;;
  tictac) export CLUSTER_TICTAC=1 ;;
  *) echo 'MODE must be auto, normal or tictac.' >&2; exit 1 ;;
esac

bench_dir="$(mktemp -d /tmp/dgxspark-cluster-fps.XXXXXX)"
log="$bench_dir/mpv.log"
export CLUSTER_INTERP_MULT="$mult" CLUSTER_REPORT_FPS=200 CLUSTER_REPORT_WARMUP=200
export WARMUP_DISABLE=1
export PYTHONHOME="${PYTHONHOME:-$env_prefix}"
export PATH="$env_prefix/bin:$PATH"
export PYTHONPATH="/usr/lib/python3/dist-packages${PYTHONPATH:+:$PYTHONPATH}"
if ! "$mpv_bin" --no-config --vf="vapoursynth=$vpy:buffered-frames=$buffered:concurrent-frames=$concurrent" \
     --untimed --hwdec=no --no-audio --frames="$frames" --vo=null --ao=null "$clip" \
     > "$log" 2>&1; then
  echo "mpv failed; log: $log" >&2
  exit 1
fi
if ! grep -q 'dual-machine active' "$log"; then
  echo "Cluster processing was not active; log: $log" >&2
  exit 1
fi
if [[ $mode == tictac ]] && ! grep -q '\[tictac\] ready with' "$log"; then
  echo "Tic-tac was requested but not active; log: $log" >&2
  exit 1
fi
result="$(grep -oE '\[(tictac|native_dispatcher) fps\].*' "$log" | tail -1 || true)"
[[ -n $result ]] || { echo "No steady-state measurement; increase N. Log: $log" >&2; exit 1; }
printf '%s\nLog: %s\n' "$result" "$log"
