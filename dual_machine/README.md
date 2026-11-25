# Multi-machine RIFE + FSRCNNX

[简体中文](./README.zh-CN.md)

The cluster pipeline runs interpolation and super-resolution across DGX Spark
machines connected by RoCE. The leader runs mpv and jellyfin-mpv-shim; each
worker runs the cluster worker service. The local single-machine pipeline
remains available.

## Install and network

Configure RoCE addresses before installation. The installer detects usable
interfaces and GID indices. It can use multiple RDMA rails; using both PCIe
paths is necessary to use the full 200 Gb/s interconnect. Depending on the
fabric, those paths can use one 200G cable or two 100G cables. Each rail must
be reachable between the leader and its workers.

```bash
# Leader: player, shim and cluster configuration
./install.sh install --cluster-host

# Each worker: worker service and tray application
./install.sh install --cluster-worker
```

The leader prompts for its rail-0 address and a comma-separated list of worker
rail-0 addresses. An optional SSH address list, in the same order, is used for
deployment and benchmarks. Each worker prompts for its own address and the
leader address. Interface names, RDMA devices and ports can also be supplied
via environment variables. For unattended installation, set
`CLUSTER_SELF_IP`, `CLUSTER_WORKER_HOSTS` and optionally
`CLUSTER_WORKER_SSH_HOSTS` and `CLUSTER_WORKER_USER` on the leader; set
`CLUSTER_SELF_IP` and `CLUSTER_PEER_IP` on each worker.

Settings are stored in `~/.config/dgxspark-mpv/cluster.conf`; do not publish
this file. The leader launches workers through the addresses in
`CLUSTER_WORKER_HOSTS`, falling back to `CLUSTER_WORKER_HOST` for a single
worker. `CLUSTER_RDMA_DEVS` and `CLUSTER_RDMA_GIDS` list the local rails and
GID indices in matching order. The default RDMA and liveness ports are 29900
and 29905; additional rails use consecutive RDMA ports.

The current service is `dgxspark-cluster-worker.service`. Existing `dual.conf`,
`DUAL_*` environment variables, old install flags and the old service name
remain compatible. A `CLUSTER_*` environment value takes precedence over its
`DUAL_*` counterpart. Reinstalling with `--reuse-cluster-config` preserves an
existing cluster configuration.

## Operating modes

- **Normal:** the leader participates in GPU computation. This is the default
  with one worker. Its priority queues distribute work between the leader and
  available workers.
- **Tic-tac:** the leader handles playback while workers dynamically claim
  source frames and process their interpolation intervals. Two or more workers
  are required. This is the default with three or more configured machines and
  an interpolation multiplier of 2, 3 or 4. Set `CLUSTER_TICTAC=0` to force
  normal mode, or `CLUSTER_TICTAC=1` to request tic-tac when eligible.

Three-machine normal mode had higher average throughput in a measured workload;
tic-tac had less variation and avoids leader GPU contention between playback
and compute. Results depend on the video and system load.

In tic-tac, each worker prioritizes available source-frame work. Workers send
prepared frame features directly to the worker that owns the preceding interval.
Source work, peer transfers and output delivery can overlap. The leader pulls
finished results in playback order by default. Normal mode also uses leader
pull for final results. `CLUSTER_TICTAC_PULL=0` selects worker-initiated output
for tic-tac. A worker failure triggers recovery with the remaining workers;
with one worker the pipeline uses normal mode, and with none it falls back to
local processing. Recovery and real-time playback depend on the workload.

**Shift+F9** toggles cluster offload. **F9** cycles interpolation multipliers
4 → 3 → 2 → 1. **F8** toggles cluster super-resolution where supported. A
filter rebuild is required for these changes.

## Default interpolation

Unknown source frame rates use ×2. `CLUSTER_INTERP_MULT` pins the multiplier;
F9 can change it during playback. For two or three machines, or explicitly
selected normal mode:

| Source | Source fps | Default |
|---|---:|---:|
| ≤720p | ≤25 | ×4 |
| ≤720p | >25–30 | ×3 |
| 1080p or 4K | ≤25 | ×3 |
| 1080p or 4K | >25–30 | ×2 |
| Any | >30 | ×1 |

With four or more configured machines in tic-tac mode:

| Source | Source fps | Default |
|---|---:|---|
| ≤720p | ≤30 | ×4, native-resolution optical flow |
| 1080p | ≤24 | ×4, native-resolution optical flow |
| 1080p | >24–30 | ×3, native-resolution optical flow |
| 4K | ≤24 | ×2, native-resolution optical flow |
| 4K | >24–30 | ×3, downsampled optical flow |
| Supported resolution | >30–60 | ×2; downsampled at 4K |
| Any | >60 | ×1 |

The low-resolution bucket is at most 1328×720; the 4K bucket starts above
2048×1128. Set `CLUSTER_TICTAC_4K_NATIVE=0` to disable automatic native 4K
flow, or `=1` to request it explicitly. Native 4K ×2 has limited measured
headroom. Defaults are not a guarantee of real-time playback.

## Development

`scripts/dev_deploy.sh --worker` installs the current host files and copies the
worker modules to every configured worker. Restart the worker services after a
copy. `tests/` contains scheduler, transport, fallback and installer tests;
`bench/README.md` describes the hardware benchmarks.
