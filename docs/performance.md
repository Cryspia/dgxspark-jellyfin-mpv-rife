# Performance

[简体中文](./performance.zh-CN.md)

Throughput, GPU utilisation, and per-task timing on DGX Spark (GB10,
ARM64, CUDA 13). Measurements via `bench/fps.sh` and `bench/timing.sh`
on the committed test clips; last full run 2026-06-10 (raw tables in
[`bench/results.md`](../bench/results.md)).

## Single-machine

1080p YUV420P10 source → 4K, vo=null, concurrent-frames=24, N=400.
Wall fps includes ~3 s of engine/cuDNN warm overhead, so steady state
reads ~10 % higher.

| chain | wall fps (N=400) |
|---|---|
| full (RIFE + FSRCNNX) | 30 – 31 |
| RIFE only             | 44 – 47 |
| FSRCNNX only          | 53 – 55 |

4K source → 4K output, mixed mode (RIFE on interp frames only), N=240
(2026-05 measurement):

| variant | sample-01 fps | sample-02 fps |
|---|---|---|
| RIFE 4.26 + 16-layer FSRCNNX | 63.9 | 65.6 |
| RIFE 4.6 @ scale=0.5         | 37.1 | 41.3 |
| RIFE 4.6 @ scale=1.0         | 25.6 | 27.3 |

## Dual-machine

1080p YUV420P10 source → 4K, vo=null, concurrent-frames=24,
DUAL_INTERP_MULT=2. Dispatcher fps = rolling 30-frame steady state.

| chain | dispatcher fps (steady) |
|---|---|
| full (RIFE + FSRCNNX)          | 95 – 96  |
| RIFE only (no SR)              | 153 – 156 |
| FSRCNNX only (mult=1)          | 145 – 147 |
| full, mult=3                   | 87       |

Roughly 2× the single-machine steady state.

Worker GPU stream utilisation under the full chain: **79 – 89 %**
(with profiling instrumentation enabled; uninstrumented runs higher).

## Per-task timing (worker side, MODE=dual bench/timing.sh)

GB10, 1080p source, CF=24, INTERP_MULT=2. Instrumentation
(`DUAL_PROFILE` + `DUAL_GPU_IDLE_DBG` + `DUAL_RDMA_PROF`) costs
~14 % steady fps — compare within this section only.

| task type | GPU kernel avg (ms) | idle_before avg (ms) |
|---|---|---|
| INTERP    | 17.6 | 0.3 |
| SR_INTERP | 8.5  | 0.6 |
| CCSR      | 11.4 | 7.7 |

## RDMA transit (worker ↔ host)

Each task ships only the byte ranges its type actually produces or
consumes — an INTERP mult=2 response is 12.4 MB (not the slot's 54 MB
worst case), SR_INTERP 24.9 MB, and host→worker CCSR/SR_INTERP
requests ~12.5 MB. INTERP / SR_INTERP request payloads are zero-copy:
the NIC gathers them straight out of the MR-registered cc_cache
region via an SGE list, so dispatch costs no CPU memmove.

| task | post_send (ms) | recv-rtt (ms) |
|---|---|---|
| INTERP    | 4.3 | 15.1 |
| SR_INTERP | 4.6 | 17.3 |
| CCSR      | 6.4 | 41.3 |

(`recv-rtt` is slot-reuse cadence — previous send-CQ on a slot to the
next request landing on it, including all host-side think time — not
wire latency; rarely-guest-dispatched types like CCSR read high by
construction. Transit is fully pipelined behind the next task's
compute, so effective critical-path cost ≈ 0 ms.)

`DUAL_MID_DELIVERY` (stage early-delivery) is on by default and
carried in the handshake so host and worker always agree; A/B-verified
bit-clean and slightly faster at mult=2 (`DUAL_MID_DELIVERY=0` on the
host disables it).

## Fault behaviour

A worker death mid-session is detected by the host's liveness
watchdog (~3 s via TCP keepalive): in-flight guest tasks are failed
and requeued to the host, the guest dispatcher stops, and playback
continues single-machine. The seek-flush sweep frees orphaned
cc_cache slots, so heavy seeking doesn't drain the pool. All 9
`bench/robustness.sh` scenarios pass.

## Theoretical ceiling

RIFE 4.26 is SM-saturated at 2 concurrent INTERPs per GB10 (1 GPU runs
1 INTERP at full speed). Two GPUs running INTERP in parallel + perfect
pipelining of CCSR / SR / wire ≈ **108 fps** ceiling on this stack.
Current sustained 95 – 96 fps = ~89 % of ceiling.
