# bench/

`dgxspark-jellyfin-mpv-rife` 单机 + 双机的颜色 / fps / 每任务 timing 基准
脚本。所有脚本共享 `_common.sh`，路径 + IP 全从环境变量读 （默认假设
本机刚跑过 `install.sh install --dual-host`）。

[English](./README.md)

→ [`results.md`](./results.md) 是参考机最新的实测数据，进 VC 当回归基准。

## 前置条件

- 已经跑过 `install.sh install` （或 `--dual-host`）；mpv + vapoursynth
  + `rife.vpy` + `vs_gpu_helpers.py` + `fsrcnnx-cudnn` 都在位。
- 双机 bench： 另一台 DGX Spark 通过 200G RoCE 链路可达 `WORKER_IP`，
  conda env 一致。 bench 会 rsync worker 脚本到 `$WORKER_DIR` 然后 ssh
  启动 `worker.py`。
- `ffmpeg` 在 `$PATH` 里 （用来合成测试片 + 跑 PSNR）。

## 一次性：生成测试片

```bash
bench/gen_clips.sh
```

往 `bench/clips/` 写两个 mandelbrot 合成片：

| 文件 | 规格 | 用途 |
|---|---|---|
| `sample-1080p-24.mp4`  | 1080p / 24 fps / 10 s / yuv420p10le | 颜色 + PSNR （低帧率，确定性） |
| `sample-1080p-120.mp4` | 1080p / 120 fps / 5 s  / yuv420p10le | fps bench （源帧率 ≥ 输出） |

要用真实素材就设 `CLIP_24` / `CLIP_120` 覆盖。

## 脚本

### `bench/color.sh` — 单机 vs 双机 PSNR

```bash
bench/color.sh                  # N=20 帧, CF=2, 3 对比
N=40 bench/color.sh             # 更多帧, 更稳
```

`full` / `no_sr` / `no_interp` 三档，每档分别渲染单机 + 双机到 FFV1，
然后 `ffmpeg psnr`。`SR_SRC` （真实） 帧应该字节相同 （`psnr_y=inf`）。
`SR_INTERP` 帧因为 RIFE 跨 GPU 非确定性，luma 大约 40-65 dB。

### `bench/fps.sh` — 稳态吞吐

```bash
bench/fps.sh                    # N=400 帧, CF=24, 6 模式
SKIP_SINGLE=1 bench/fps.sh      # 只测双机
N=800 bench/fps.sh              # 稳态估计更准
```

双机模式下 worker **只启动一次**，三个变体复用同一个进程。dispatcher
fps （30 帧滑窗） 已排除 init,wall-clock fps 包含 init。

### `bench/robustness.sh` — 双机故障恢复

```bash
bench/robustness.sh                # N=60 帧, RECOVERY_SLA=5 s
RECOVERY_SLA=3 N=40 bench/robustness.sh
```

九个场景，每个验证双机链能在 `RECOVERY_SLA` 秒内恢复（或干净 fallback 到单机）：

1. `worker_down`   — mpv 启动前停掉 worker.service。 期望 liveness probe
   快速失败 → 单机 fallback.
2. `clean_restart` — 完整 dual session，正常退出 mpv，立刻再启动一个。
   期望 dual 重连。
3. `force_kill`    — 渲染中 `kill -9` mpv，立刻再启。 期望 worker 通过
   TCP keepalive（200 G RoCE 上约 3 s）检测到 host 死亡 + 接受新 mpv.
4. `two_mpvs`      — 两个 mpv 抢同一个 worker。 一个走 dual，另一个 fallback；
   都不能 deadlock.
5. `host_child_crash` — 在 dma/mgr/compute 启动中途 SIGKILL 一个 host 子进程
   （模拟 bootstrap 里 ImportError 的崩溃），再杀 host mpv 本体。 之后启动
   一个新 mpv，必须在 `RECOVERY_SLA` 秒内进 dual。 覆盖"host 会话子进程
   死了，worker 卡在 wait-loop 不回主 accept loop"这个场景。
6. `queue_saturation` — 用 50 个裸 TCP connect+close 轰炸 worker liveness
   端口，验证后台 drainer 能把队列清空 （Recv-Q→0），之后 dual 仍能正常用。
7. `repeated_vf_reload` — 单个带 IPC 的 mpv，先进 dual，再用 IPC 触发
   `vf set` 强制重建 filter graph → 同一 mpv 进程内 dual handshake
   重做一次。 验证链路存活 + 至少 1 次 dual-active。
8. `trt_cache_miss` — 把 host 和 worker 双方的 1080p RIFE TRT engine
   cache 都改名，重启 worker 再开 mpv：必须重新编译 TRT （这套栈下
   30–60 秒），验证最终能进 dual 且渲染出帧。 退出时还原 cache。
   用 `TRT_BENCH_TIMEOUT` 调上限 （默认 300 s）。
9. `seek_storm` — 对存活的 dual 会话用 IPC 连发 seek，验证 SLA 内恢复
   播放。同时覆盖 seek-flush 的 cc_cache 槽位回收（高频拖动不允许耗尽
   槽池）。

需要 worker systemd 服务在从机上提前跑起来。 bench 自己会作为场景 1 的
一部分停/启服务。

### `bench/timing.sh` — 每任务 GPU + 通信分解

```bash
bench/timing.sh                       # mode=dual
MODE=dual_no_sr bench/timing.sh
MODE=dual_no_interp N=800 bench/timing.sh
```

worker 端开 `DUAL_PROFILE=1` + `DUAL_RDMA_PROF=1` +
`DUAL_GPU_IDLE_DBG=1` 跑一档，然后 tail-parse 主机 + worker log 输出：

- worker 总量：`kernel_total` / `idle_total` / `GPU_util`
- worker 每任务类型：`kernel` / `idle_before` / `claim_wait`
  （INTERP / SR_INTERP / CCSR）
- worker RDMA： 每类型 `send` 和 `recv-rtt` 延迟
- 主机：pop 等待，队列深度，锁持时间

诊断工具 — 绝对数会随 kernel 变化漂移，有用的信号是相对分解 （瓶颈
在算/通/调度？）。

## 环境变量覆盖

| 变量 | 默认 | 含义 |
|---|---|---|
| `ENV_ROOT`   | `~/miniforge3/envs/vsmpv` | conda env 前缀 |
| `MPV` / `PY` | `$ENV_ROOT/bin/mpv` / `python` | 可执行文件 |
| `MPV_CFG`    | `~/.config/mpv` | mpv 配置目录 （`rife.vpy`, `vs_gpu_helpers.py`, `fsrcnnx-cudnn/`） |
| `VPY`        | `$MPV_CFG/rife.vpy` | vapoursynth 入口 |
| `DUAL_CFG_FILE` | `~/.config/dgxspark-mpv/dual.conf` | source 进来用 `DUAL_HOST_IP`/`DUAL_WORKER_HOST`/`DUAL_WORKER_USER`/`DUAL_RDMA_*` 当默认 （由 `install.sh install --dual-host` 写入） |
| `HOST_IP`    | 来自 `$DUAL_HOST_IP` | RoCE 链路上的本机 |
| `WORKER_IP`  | 来自 `$DUAL_WORKER_HOST` | RoCE 链路上的 worker |
| `WORKER_USER`| 来自 `$DUAL_WORKER_USER`，否则 `ubuntu` | worker ssh 用户 |
| `WORKER_DIR` | 来自 `$DUAL_WORKER_DIR`，否则 `~/.local/share/dgxspark-mpv/worker` | rsync 目标 |
| `RDMA_DEV`   | 来自 `$DUAL_RDMA_DEV`，否则 `rocep1s0f0` | 本机 RDMA 设备 |
| `RDMA_PORT`  | 来自 `$DUAL_RDMA_PORT`，否则 `29900` | worker 端 RDMA 监听端口 |
| `CLIP_24` / `CLIP_120` | `bench/clips/sample-1080p-{24,120}.mp4` | 输入 |

## 模式 （单机 + 双机共享按键）

| mode | 单机链 | 双机链 |
|---|---|---|
| `single` / `dual`             | RIFE × 2 + FSRCNNX  | 同，RIFE 跑 guest |
| `single_no_sr` / `dual_no_sr` | 仅 RIFE × 2         | 同，无 SR (`F8` off) |
| `single_no_rife` / `dual_no_interp` | 仅 FSRCNNX    | 双机仅 SR 路径 （mult=1） |

切档约定：bench 写 `/tmp/{fsrcnnx_variant,rife_disabled,dual_machine_*}`
做切换；退出时恢复 bench 前状态。
