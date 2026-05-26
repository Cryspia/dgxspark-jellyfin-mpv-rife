# bench/

Color / fps / per-task timing benches for the `dgxspark-jellyfin-mpv-rife`
single and dual paths. All scripts share `_common.sh` and pull paths
+ IPs from environment variables (defaults assume a fresh
`install.sh install --dual-host` on this box).

[简体中文](./README.zh-CN.md)

→ [`results.md`](./results.md) holds the latest measurements from the
reference DGX Spark, kept under VC so it doubles as a regression
baseline.

## Prerequisites

- `install.sh install` (or `--dual-host`) has been run; mpv + vapoursynth
  + `rife.vpy` + `vs_gpu_helpers.py` + `fsrcnnx-cudnn` are all on disk.
- For dual benches: a second DGX Spark reachable at `WORKER_IP` over a
  200G RoCE link, with the same conda env. The bench rsync's the worker
  scripts to `$WORKER_DIR` and launches `worker.py` over ssh.
- `ffmpeg` available on `$PATH` (used to synthesize clips + measure PSNR).

## One-time: generate test clips

```bash
bench/gen_clips.sh
```

Writes two synthetic mandelbrot clips into `bench/clips/`:

| file | size | use |
|---|---|---|
| `sample-1080p-24.mp4`  | 1080p / 24 fps / 10 s / yuv420p10le | color + PSNR (low rate, deterministic) |
| `sample-1080p-120.mp4` | 1080p / 120 fps / 5 s  / yuv420p10le | fps bench (source rate ≥ output) |

Override with `CLIP_24` / `CLIP_120` if you'd rather use real footage.

## Scripts

### `bench/color.sh` — single vs dual PSNR

```bash
bench/color.sh                  # N=20 frames, CF=2, 3 pairs
N=40 bench/color.sh             # more frames for stability
```

For each of `full` / `no_sr` / `no_interp`, renders both single and
dual outputs to FFV1, then runs `ffmpeg psnr`. `SR_SRC` (real) frames
should be byte-identical (`psnr_y=inf`). `SR_INTERP` frames carry
RIFE cross-GPU non-determinism on luma (~40-65 dB).

### `bench/fps.sh` — steady-state throughput

```bash
bench/fps.sh                    # N=400 frames, CF=24, 6 modes
SKIP_SINGLE=1 bench/fps.sh      # dual chain only
N=800 bench/fps.sh              # tighter steady-state estimate
```

For dual modes, the worker is started **once** and reused across all
three variants. The dispatcher fps (rolling 30-frame window) excludes
init overhead; the wall-clock fps includes it.

### `bench/robustness.sh` — dual-mode fault recovery

```bash
bench/robustness.sh                # N=60 frames, RECOVERY_SLA=5 s
RECOVERY_SLA=3 N=40 bench/robustness.sh
```

Four scenarios, each asserting the dual chain recovers (or cleanly
falls back to single) within `RECOVERY_SLA` seconds:

1. `worker_down`   — stop worker.service before mpv. Expect fast
   liveness-probe fail → single-machine fallback.
2. `clean_restart` — full dual session, exit mpv, immediately launch
   a second mpv. Expect dual reconnect.
3. `force_kill`    — `kill -9` mpv mid-render, immediately relaunch.
   Expect the worker to detect the dead host via TCP keepalive (≈ 3 s
   on 200 G RoCE) and accept the new mpv.
4. `two_mpvs`      — two mpvs racing for the same worker. One gets
   dual, the other falls back; neither deadlocks.
5. `host_child_crash` — kill one of host's dma/mgr/compute subprocesses
   mid-init (simulates an ImportError-style crash in the spawned
   bootstrap), then kill host mpv. A fresh mpv must reach dual within
   `RECOVERY_SLA`. Covers the failure where host session-level
   children die but the worker is stuck in wait-loop.
6. `queue_saturation` — bombard worker's liveness port with 50 bare
   TCP connects+closes, then assert the background drainer reclaims
   the queue (Recv-Q→0) and dual still works after.
7. `repeated_vf_reload` — single mpv with IPC; reach dual, then send
   `vf set` to force a filter-graph rebuild → second dual handshake
   in the same mpv process. Asserts the chain survives the rebuild
   and ≥ 1 dual-active marker.
8. `trt_cache_miss` — rename the 1080p RIFE TRT engine cache on
   both host and worker, restart worker, launch mpv: TRT must
   recompile from scratch (~30-60 s on this stack). Asserts dual
   eventually reaches active + frames render. Caches restored on
   exit. Tunable via `TRT_BENCH_TIMEOUT` (default 300 s).

Requires the worker systemd service running on the guest box before
the bench launches. The bench stops/starts the service as part of
scenario 1.

### `bench/timing.sh` — per-task GPU + comm breakdown

```bash
bench/timing.sh                       # mode=dual
MODE=dual_no_sr bench/timing.sh
MODE=dual_no_interp N=800 bench/timing.sh
```

Runs one dual mode with `DUAL_PROFILE=1` + `DUAL_RDMA_PROF=1` +
`DUAL_GPU_IDLE_DBG=1` on the worker, then tail-parses the host +
worker logs into:

- worker totals: `kernel_total` / `idle_total` / `GPU_util`
- worker per-task-type: `kernel` / `idle_before` / `claim_wait`
  for INTERP / SR_INTERP / CCSR
- worker RDMA: per-type `send` and `recv-rtt` latency
- host: pop wait, queue depth, lock-hold breakdown

Diagnostic tool — absolute numbers shift with kernel changes; the
useful signal is the relative breakdown (is compute, comm, or scheduler
the bottleneck?).

## Environment overrides

| var | default | meaning |
|---|---|---|
| `ENV_ROOT`   | `~/miniforge3/envs/vsmpv` | conda env prefix |
| `MPV` / `PY` | `$ENV_ROOT/bin/mpv` / `python` | binaries |
| `MPV_CFG`    | `~/.config/mpv` | mpv config dir (has `rife.vpy`, `vs_gpu_helpers.py`, `fsrcnnx-cudnn/`) |
| `VPY`        | `$MPV_CFG/rife.vpy` | vapoursynth entry |
| `DUAL_CFG_FILE` | `~/.config/dgxspark-mpv/dual.conf` | sourced for `DUAL_HOST_IP`/`DUAL_WORKER_HOST`/`DUAL_WORKER_USER`/`DUAL_RDMA_*` defaults (written by `install.sh install --dual-host`) |
| `HOST_IP`    | from `$DUAL_HOST_IP` | this box on the RoCE link |
| `WORKER_IP`  | from `$DUAL_WORKER_HOST` | worker box on the RoCE link |
| `WORKER_USER`| from `$DUAL_WORKER_USER`, else `ubuntu` | ssh user for worker |
| `WORKER_DIR` | from `$DUAL_WORKER_DIR`, else `~/.local/share/dgxspark-mpv/worker` on the worker | rsync target on worker |
| `RDMA_DEV`   | from `$DUAL_RDMA_DEV`, else `rocep1s0f0` | local RDMA device |
| `RDMA_PORT`  | from `$DUAL_RDMA_PORT`, else `29900` | RDMA listen port (worker side) |
| `CLIP_24` / `CLIP_120` | `bench/clips/sample-1080p-{24,120}.mp4` | inputs |

## Modes (single + dual share the same key)

| mode | single chain | dual chain |
|---|---|---|
| `single` / `dual`             | RIFE × 2 + FSRCNNX  | same, RIFE on guest |
| `single_no_sr` / `dual_no_sr` | RIFE × 2 only       | same, no SR  (`F8` off) |
| `single_no_rife` / `dual_no_interp` | FSRCNNX only  | dual SR-only path (mult=1) |

Flag-file convention: bench writes to `/tmp/{fsrcnnx_variant,rife_disabled,dual_machine_*}`
to switch modes; pre-bench state is saved + restored on exit.
