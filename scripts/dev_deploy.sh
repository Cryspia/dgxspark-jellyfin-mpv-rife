#!/usr/bin/env bash
# scripts/dev_deploy.sh — push the working tree's runtime files into the
# live install locations (host: ~/.config/mpv; worker: via bench's
# start_worker rsync). Mirrors install.sh's copy steps for the files
# that change during development, without re-running the full install.
#
# Usage: scripts/dev_deploy.sh            # host-side only
#        scripts/dev_deploy.sh --worker   # also rsync worker modules to
#                                         # configured secondary boxes
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/cluster_env.sh"
cluster_normalize_env
MPV_CFG="${MPV_CFG:-$HOME/.config/mpv}"

cp -f "$ROOT/vs_gpu_helpers.py"      "$MPV_CFG/vs_gpu_helpers.py"
cp -f "$ROOT/sr_keys_helper.py"      "$MPV_CFG/sr_keys_helper.py"
cp -f "$ROOT/dual_machine/rife.vpy"  "$MPV_CFG/rife.vpy"
mkdir -p "$MPV_CFG/scripts"
cp -f "$ROOT"/scripts/{sr_keys,warmup,dual_fps_override,dual_seek_flush}.lua \
      "$MPV_CFG/scripts/"

if [[ -d "$MPV_CFG/dual_machine" ]]; then
  rsync -a --delete --exclude='__pycache__' --exclude='native_filter' \
        --exclude='vk_priority_layer' --exclude='vs_gpu_helpers.py' \
        --exclude='*.sh' --exclude='*.md' --exclude='*.in' --exclude='rife.vpy' \
        "$ROOT/dual_machine/" "$MPV_CFG/dual_machine/"
  install -m 0644 "$ROOT/vs_gpu_helpers.py" \
          "$MPV_CFG/dual_machine/vs_gpu_helpers.py"
  # keep the native_filter build output; only sync helper sources
  rsync -a --exclude='build' --exclude='__pycache__' \
        "$ROOT/dual_machine/native_filter/" \
        "$MPV_CFG/dual_machine/native_filter/"
fi
echo "host deploy → $MPV_CFG done"

if [[ "${1:-}" == "--worker" ]]; then
  # shellcheck disable=SC1090
  config="${CLUSTER_CFG:-$HOME/.config/dgxspark-mpv/cluster.conf}"
  if [[ ! -f "$config" && -z "${CLUSTER_CFG:-}" ]]; then
    config="$HOME/.config/dgxspark-mpv/dual.conf"
  fi
  source "$config"
  cluster_normalize_env
  worker_list="${CLUSTER_WORKER_HOSTS:-${CLUSTER_WORKER_HOST:-}}"
  [[ -n "$worker_list" ]] || { echo "no workers configured" >&2; exit 1; }
  IFS=, read -r -a rdma_hosts <<< "$worker_list"
  ssh_list="${CLUSTER_WORKER_SSH_HOSTS:-$worker_list}"
  IFS=, read -r -a worker_hosts <<< "$ssh_list"
  if (( ${#worker_hosts[@]} != ${#rdma_hosts[@]} )); then
    echo "CLUSTER_WORKER_SSH_HOSTS must have one address per worker" >&2
    exit 1
  fi
  declare -A seen_workers=()
  workers=()
  for host in "${worker_hosts[@]}"; do
    host="${host#"${host%%[![:space:]]*}"}"
    host="${host%"${host##*[![:space:]]}"}"
    if [[ ! "$host" =~ ^[a-zA-Z0-9._-]+$ || -n "${seen_workers[$host]:-}" ]]; then
      echo "invalid or duplicate worker in configuration: $host" >&2
      exit 1
    fi
    seen_workers[$host]=1
    workers+=("$host")
  done
  WORKER_DIR="${CLUSTER_WORKER_DIR:-.local/share/dgxspark-mpv/worker}"
  mods=()
  while IFS= read -r m; do
    [[ -z "$m" || "$m" == \#* ]] && continue
    mods+=("$ROOT/dual_machine/$m")
  done < "$ROOT/dual_machine/WORKER_MODULES"
  for host in "${workers[@]}"; do
    worker="${CLUSTER_WORKER_USER:-${USER:-$(id -un)}}@$host"
    rsync -azq "${mods[@]}" "$ROOT/vs_gpu_helpers.py" "$worker:$WORKER_DIR/"
    echo "worker deploy → $worker:$WORKER_DIR done"
  done
  echo "restart dgxspark-cluster-worker.service on each worker to load the new code"
fi
