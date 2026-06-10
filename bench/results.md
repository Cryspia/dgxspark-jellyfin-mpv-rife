# Bench results — DGX Spark reference machines

Snapshot of the most recent full run, kept under version control so it
doubles as a regression baseline.

- **Date:** 2026-06-10
- **Hardware:** 2 × DGX Spark (GB10, ARM64, CUDA 13), 200 G RoCE link
- **Software:** `main`
- **Test clips:** `bench/clips/sample-1080p-24.mp4` for color,
  `bench/clips/sample-1080p-120.mp4` for fps — both committed in-repo
  (synthesised by `bench/gen_clips.sh`), so anyone can reproduce.

Reproduce with:
```bash
bench/color.sh
bench/fps.sh
MODE=dual bench/timing.sh
bench/robustness.sh
```

## bench/color.sh — single vs dual PSNR

`N=20`, `CF=2`, comparing the FFV1-encoded outputs of each pair.

| pair | single bytes | dual bytes | Y avg | U avg | V avg | overall avg |
|---|---:|---:|---:|---:|---:|---:|
| full        (RIFE + FSRCNNX) | 53,356,785 | 53,331,494 | 58.31 | 49.93 | 49.03 | 53.22 |
| no\_sr      (RIFE only)       | 26,235,552 | 12,739,136 | 61.46 | 47.14 | 47.38 | 51.71 |
| no\_interp  (FSRCNNX only)    | 52,623,730 | 54,733,061 | 63.79 | 37.69 | 36.56 | 41.84 |

`SR_SRC` (real-frame) outputs are **byte-identical** between single
and dual (`psnr_y=inf` every frame). `SR_INTERP` frames carry RIFE
cross-GPU non-determinism: 54–61 dB Y on the full pair in this run.

PSNR here measures single-vs-dual *consistency* and is strongly
content-dependent — the synthetic clip's sharp, saturated chroma reads
much lower on U/V than real video does. Numbers are only comparable
across runs of the **same** clip; re-baseline whenever the clip
changes.

## bench/fps.sh — sustained throughput

`N=400`, `CF=24`, bf=2, 1080p source → 4K output. `wall` includes
~3 s of engine / cuDNN warm overhead (reads low); `dispatcher` is the
rolling 30-frame estimate (steady state). Run-to-run noise is ±3 %;
the very first dual session after a worker cold start can read far
lower — discard the first run when benchmarking.

| chain | mode | wall fps | dispatcher fps |
|---|---|---:|---:|
| single | single          | 29.7 | — |
| single | single\_no\_sr   | 43.8 | — |
| single | single\_no\_rife | 53.0 | — |
| dual   | dual            | 26.2 |  94.6 – 96.4 |
| dual   | dual\_no\_sr     | 31.9 | 152.8 – 156.1 |
| dual   | dual\_no\_interp | 29.1 | 145.1 – 146.6 |

### dual at higher interp_mult

`N=300`, `CF=24`, mult forced via env (`DUAL_INTERP_MULT=N`);
production auto-defaults are listed in dual_machine/README.md.

| `DUAL_INTERP_MULT` | dispatcher fps (steady) |
|---|---:|
| 2 | 94.6 – 96.4 |
| 3 | 86.7 |

`DUAL_MID_DELIVERY=1` (stage early-delivery) measures bit-clean but
throughput-neutral at both mult=2 (94.2 vs 96.4) and mult=3 (86.6 vs
86.7); it stays off by default — see the flag comments in
`worker_3proc.py`.

## bench/timing.sh — per-task GPU + comm (MODE=dual, N=600)

Numbers below are with full instrumentation enabled
(`DUAL_PROFILE=1`, `DUAL_GPU_IDLE_DBG`, `DUAL_RDMA_PROF`), which
costs ~13 % steady-state fps (83 vs 95) — compare within this table,
not against fps.sh. Measured with an idle desktop: no shim, no other
GPU clients on either box.

Worker side per task type (last `PROFILE` sample, GB10):

| task type | kernel (ms) | idle\_before (ms) |
|---|---:|---:|
| INTERP    | 17.6 | 0.3 |
| SR\_INTERP | 8.5  | 0.6 – 6 |
| CCSR      | 11.4 | 7.7 |

Worker GPU stream utilisation (rolling): **80 – 88 %**.

Worker RDMA per task (`DUAL_RDMA_PROF=1`):

| task | send (ms) | recv-rtt (ms) |
|---|---:|---:|
| INTERP    | 4.3 | 15.1 |
| SR\_INTERP | 4.6 | 17.3 |
| CCSR      | 6.4 | 41.3 |

`recv-rtt` is **not** wire latency: it measures "previous send-CQ on
this slot → next recv on this slot", attributed to the *incoming*
task's type — i.e. slot-reuse cadence including all host-side think
time. Task types the guest receives rarely (CCSR is mostly consumed
host-side) therefore read high by construction. Transit pipelines
behind the next task's compute; effective critical-path cost ≈ 0 ms.

Host queue depth + lock-hold (`DUAL_PROFILE=1`):
- HOST pops ≈ 69 / s (cc 29, sr 33, interp 6); GUEST ≈ 57 / s
  (interp 35, cc 13, sr 8). `wait_avg ≈ 4 ms` (dispatchers wait on
  work, not slots).
- Queue depth at pop ≈ 0.6 – 1.5; DAG size ≈ 38 (peak 51).
- Lock-hold: pops ≈ 0.05 ms, task_done ≈ 0.15 ms, submit ≈ 0.08 ms.

## bench/robustness.sh

All **9/9 scenarios PASS**: worker_down (clean single fallback),
clean_restart ×2, force_kill_restart, two_mpvs, host_child_crash,
queue_saturation, repeated_vf_reload, trt_cache_miss (full recompile,
118.9 s wall), seek_storm.
