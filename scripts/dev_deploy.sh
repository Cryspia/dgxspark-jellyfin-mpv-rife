#!/usr/bin/env bash
# scripts/dev_deploy.sh — push the working tree's runtime files into the
# live install locations (host: ~/.config/mpv; worker: via bench's
# start_worker rsync). Mirrors install.sh's copy steps for the files
# that change during development, without re-running the full install.
#
# Usage: scripts/dev_deploy.sh            # host-side only
#        scripts/dev_deploy.sh --worker   # also rsync worker modules to
#                                         # the secondary box (reads
#                                         # ~/.config/dgxspark-mpv/dual.conf)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
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
  source "$HOME/.config/dgxspark-mpv/dual.conf"
  WORKER="${DUAL_WORKER_USER:-spark}@${DUAL_WORKER_HOST:?}"
  WORKER_DIR="${DUAL_WORKER_DIR:-.local/share/dgxspark-mpv/worker}"
  mods=()
  while IFS= read -r m; do
    [[ -z "$m" || "$m" == \#* ]] && continue
    mods+=("$ROOT/dual_machine/$m")
  done < "$ROOT/dual_machine/WORKER_MODULES"
  rsync -azq "${mods[@]}" "$ROOT/vs_gpu_helpers.py" "$WORKER:$WORKER_DIR/"
  echo "worker deploy → $WORKER:$WORKER_DIR done"
fi
