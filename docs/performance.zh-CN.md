# 性能

[English](./performance.md)

DGX Spark (GB10, ARM64, CUDA 13) 上的吞吐 / GPU 使用率 / 每任务耗时。
数据来自 `bench/fps.sh` 和 `bench/timing.sh`。

## 单机

1080p YUV420P10 源 → 4K, vo=null, concurrent-frames=24.

| 链路 | sustained fps |
|---|---|
| full (RIFE + FSRCNNX) | 48 – 50 |
| 只 RIFE              | 67 – 71 |
| 只 FSRCNNX           | 66 – 71 |

4K 源 → 4K 输出，mixed mode （只在插帧上跑 RIFE）， N=240:

| variant | sample-01 fps | sample-02 fps |
|---|---|---|
| RIFE 4.26 + 16-layer FSRCNNX | 63.9 | 65.6 |
| RIFE 4.6 @ scale=0.5         | 37.1 | 41.3 |
| RIFE 4.6 @ scale=1.0         | 25.6 | 27.3 |

## 双机

1080p YUV420P10 源 → 4K, vo=null, concurrent-frames=24,
DUAL_INTERP_MULT=2.

| 链路 | dispatcher fps （稳态） | 对单机加速比 |
|---|---|---|
| full (RIFE + FSRCNNX)         | 96  | 1.95× |
| 只 RIFE (`F8` off)            | 148 | 2.9×  |
| 只 FSRCNNX (mult=1)           | 112 | 1.6×  |

full 链路 GPU 使用率：

| side | GPU util |
|---|---|
| host  | 94 – 98 % |
| guest | 93 – 98 % |

## 每任务耗时 （worker 端，DUAL_PROFILE=1）

GB10, 1080p 源，CF=24, INTERP_MULT=2.

| task type | GPU kernel 平均 （ms） | idle_before 平均 （ms） |
|---|---|---|
| INTERP    | 5 – 7  | 0.3 – 0.8 |
| SR_INTERP | 6 – 8  | 0.4 – 1.2 |
| CCSR      | 9 – 12 | 0.2 – 0.6 |

## RDMA 传输 （worker → host）

| task | post_send (ms) | wire RTT (ms) |
|---|---|---|
| INTERP    | 5.7 | 16 |
| SR_INTERP | 3.1 | 11 |
| CCSR      | 4.2 | 13 |

传输完全 pipeline 在下一个 task 的 compute 后面；关键路径耗时 ≈ 0 ms.

## 理论上限

RIFE 4.26 在 GB10 上 SM 饱和 = 单 GPU 跑 1 个 INTERP 全速，2 个并发
就 cache thrash。 两块 GPU 各跑一个 INTERP + CCSR / SR / 线 完美
pipeline ≈ 单机能跑到的 **108 fps** 上限。 当前稳态 96 fps = 上限的
89 %.

