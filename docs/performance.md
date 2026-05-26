# Performance

[简体中文](./performance.zh-CN.md)

Throughput, GPU utilisation, and per-task timing on DGX Spark (GB10,
ARM64, CUDA 13). Measurements via `bench/fps.sh` and `bench/timing.sh`.

## Single-machine

1080p YUV420P10 source → 4K, vo=null, concurrent-frames=24.

| chain | sustained fps |
|---|---|
| full (RIFE + FSRCNNX) | 48 – 50 |
| RIFE only             | 67 – 71 |
| FSRCNNX only          | 66 – 71 |

4K source → 4K output, mixed mode (RIFE on interp frames only), N=240:

| variant | sample-01 fps | sample-02 fps |
|---|---|---|
| RIFE 4.26 + 16-layer FSRCNNX | 63.9 | 65.6 |
| RIFE 4.6 @ scale=0.5         | 37.1 | 41.3 |
| RIFE 4.6 @ scale=1.0         | 25.6 | 27.3 |

## Dual-machine

1080p YUV420P10 source → 4K, vo=null, concurrent-frames=24,
DUAL_INTERP_MULT=2.

| chain | dispatcher fps (steady) | rough speedup vs single |
|---|---|---|
| full (RIFE + FSRCNNX)          | 96  | 1.95× |
| RIFE only (`F8` off)           | 148 | 2.9×  |
| FSRCNNX only (mult=1)          | 112 | 1.6×  |

GPU utilisation under full chain:

| side | GPU util |
|---|---|
| host  | 94 – 98 % |
| guest | 93 – 98 % |

## Per-task timing (worker side, DUAL_PROFILE=1)

GB10, 1080p source, CF=24, INTERP_MULT=2.

| task type | GPU kernel avg (ms) | idle_before avg (ms) |
|---|---|---|
| INTERP    | 5 – 7  | 0.3 – 0.8 |
| SR_INTERP | 6 – 8  | 0.4 – 1.2 |
| CCSR      | 9 – 12 | 0.2 – 0.6 |

## RDMA transit (worker → host)

| task | post_send (ms) | wire RTT (ms) |
|---|---|---|
| INTERP    | 5.7 | 16 |
| SR_INTERP | 3.1 | 11 |
| CCSR      | 4.2 | 13 |

Transit is fully pipelined behind the next task's compute; effective
critical-path cost is ≈ 0 ms.

## Theoretical ceiling

RIFE 4.26 is SM-saturated at 2 concurrent INTERPs per GB10 (1 GPU runs
1 INTERP at full speed). Two GPUs running INTERP in parallel + perfect
pipelining of CCSR / SR / wire ≈ **108 fps** ceiling on this stack.
Current sustained 96 fps = 89 % of ceiling.

