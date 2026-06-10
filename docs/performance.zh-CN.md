# 性能

[English](./performance.md)

DGX Spark（GB10、ARM64、CUDA 13）上的吞吐、GPU 利用率与每任务耗时。
用 `bench/fps.sh` 与 `bench/timing.sh` 在仓库自带测试片源上测得；
最近一次完整测量为 2026-06-10（原始表格见
[`bench/results.md`](../bench/results.md)）。

## 单机

1080p YUV420P10 源 → 4K，vo=null，concurrent-frames=24，N=400。
wall fps 含约 3 秒引擎/cuDNN 预热开销（读数偏低约 10%）。

| 链 | wall fps (N=400) |
|---|---|
| 全链（RIFE + FSRCNNX） | 30 – 31 |
| 仅 RIFE                | 44 – 47 |
| 仅 FSRCNNX             | 53 – 55 |

4K 源 → 4K 输出，混合模式（只对插帧跑 RIFE），N=240
（2026-05 测量）：

| 变体 | sample-01 fps | sample-02 fps |
|---|---|---|
| RIFE 4.26 + 16 层 FSRCNNX | 63.9 | 65.6 |
| RIFE 4.6 @ scale=0.5      | 37.1 | 41.3 |
| RIFE 4.6 @ scale=1.0      | 25.6 | 27.3 |

## 双机

1080p YUV420P10 源 → 4K，vo=null，concurrent-frames=24，
DUAL_INTERP_MULT=2。dispatcher fps = 30 帧滚动稳态。

| 链 | dispatcher fps（稳态） |
|---|---|
| 全链（RIFE + FSRCNNX） | 95 – 96 |
| 仅 RIFE（无 SR）       | 153 – 156 |
| 仅 FSRCNNX（mult=1）   | 145 – 147 |
| 全链，mult=3           | 87 |

约为单机稳态的 2 倍。

全链下 worker GPU stream 利用率：**79 – 89%**（带性能探针；
不带探针更高）。

## 每任务耗时（worker 侧，MODE=dual bench/timing.sh）

GB10，1080p 源，CF=24，INTERP_MULT=2。探针
（`DUAL_PROFILE` + `DUAL_GPU_IDLE_DBG` + `DUAL_RDMA_PROF`）约
损耗 14% 稳态 fps —— 本节数据只做内部对比。

| 任务类型 | GPU kernel 均值 (ms) | idle_before 均值 (ms) |
|---|---|---|
| INTERP    | 17.6 | 0.3 |
| SR_INTERP | 8.5  | 0.6 |
| CCSR      | 11.4 | 7.7 |

## RDMA 传输（worker ↔ host）

按任务收发实际字节（2026-06）：每个任务只传它真正产出/消费的
字节 —— INTERP mult=2 回传 12.4 MB（原来整槽 54 MB），SR_INTERP
24.9 MB；host→worker 的 CCSR/SR_INTERP 请求从 58.5 MB 降到约
12.5 MB。INTERP / SR_INTERP 请求负载同时走零拷贝：NIC 以 SGE
gather 直接从注册为 MR 的 cc_cache 区域取数，不再由派发线程经
~5 GB/s 的 cudaHostRegister CPU 读路径 memmove。

| 任务 | post_send (ms) | recv-rtt (ms) |
|---|---|---|
| INTERP    | 4.3 | 15.1 |
| SR_INTERP | 4.6 | 17.3 |
| CCSR      | 6.4 | 41.3 |

（`recv-rtt` 是 slot 复用节奏 —— 该 slot 上一次发送完成到下一个请求
落上来的间隔，含 host 端全部处理/空闲时间，不是网络延迟；guest 很少
被派 CCSR，所以该项天然偏大。传输完全流水在下一个任务的计算后面，
关键路径有效成本 ≈ 0 ms。）

`DUAL_MID_DELIVERY`（stage 早投递）默认关闭：mult=2 与 mult=3 实测
画质干净但吞吐持平 —— 依据见 `dual_machine/worker_3proc.py` 的
flag 注释。

## 容错行为

会话中 worker 死亡由 host 侧 liveness watchdog 检测（TCP keepalive
约 3 秒）：在途 guest 任务标记失败并回流给 host，guest 派发停止，
播放无缝降级单机继续。seek-flush 会释放孤儿 cc_cache 槽位，长会话
反复拖进度条不会耗尽槽池。`bench/robustness.sh` 9/9 场景全部通过。

## 理论天花板

RIFE 4.26 在单 GB10 上 2 个并发 INTERP 即打满 SM（1 GPU 全速跑
1 个 INTERP）。两 GPU 并行 INTERP + CCSR/SR/传输完美流水 ≈ 本栈
**108 fps** 天花板。当前稳态 95 – 96 fps ≈ 天花板的 89%。
