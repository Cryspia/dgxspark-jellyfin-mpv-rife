#!/bin/bash
# Bring up a DGX Spark as the SECONDARY (worker) for the dual-machine
# pipeline. Run THIS on the worker box (e.g. spark02), after the primary
# has copied this directory + the fsrcnnx weights over.
#
# What it does:
#   1. Ensures miniforge3 env "vsmpv" with torch + cuda is present
#      (rsync'd from primary, but you can re-run if you want a refresh).
#   2. Installs /etc/default/dual_machine_worker — operator-edit file
#      with DUAL_HOST_IP (the primary's first-rail RoCE IP).
#   3. Installs + enables the systemd unit so the worker auto-starts
#      and survives reboots.
#
# Usage:
#   ./setup_secondary.sh PRIMARY_RAIL0_IP
#
# Example:
#   ./setup_secondary.sh 10.200.128.1
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 PRIMARY_RAIL0_IP" >&2
  echo "  PRIMARY_RAIL0_IP must be reachable from this box over RoCE." >&2
  exit 2
fi
PRIMARY_IP="$1"
HERE="$(cd "$(dirname "$0")" && pwd)"

# 1. Sanity check torch/cuda
if [ ! -x /home/spark/miniforge3/envs/vsmpv/bin/python ]; then
  echo "fatal: /home/spark/miniforge3/envs/vsmpv/bin/python missing." >&2
  echo "       rsync the primary's miniforge3 env here first, e.g.:" >&2
  echo "       rsync -aH spark@primary:/home/spark/miniforge3/ /home/spark/miniforge3/" >&2
  exit 1
fi

# 2. Operator env file
sudo install -d -m 755 /etc/default
sudo tee /etc/default/dual_machine_worker >/dev/null <<EOF
# Primary host's RoCE rail-0 IP. Edit if the primary changes interface.
DUAL_HOST_IP=$PRIMARY_IP
DUAL_MASTER_PORT=29500
EOF

# 3. systemd unit
sudo install -m 644 "$HERE/worker.service" /etc/systemd/system/dual_machine_worker.service
sudo systemctl daemon-reload
sudo systemctl enable --now dual_machine_worker.service

echo
echo "Worker service installed and started."
echo "Check status:  systemctl status dual_machine_worker"
echo "Tail logs:     journalctl -u dual_machine_worker -f"
echo
echo "Test from the primary:"
echo "  DUAL_WORKER_HOST=$(hostname -I | awk '{print $1}') \\"
echo "    mpv your-video.mp4"
