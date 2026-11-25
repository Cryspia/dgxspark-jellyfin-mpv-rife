# 多机 RIFE + FSRCNNX

[English](./README.md)

集群管线通过 RoCE 在多台 DGX Spark 上分配插帧和超分任务。主机运行 mpv 与
jellyfin-mpv-shim，每个 worker 运行集群计算服务。单机管线仍可独立使用。

## 安装与网络

安装前先配置 RoCE 地址。安装器会检测可用网口和 GID 索引；传输支持多条
RDMA rail，要用满 200 Gb/s 互联带宽需要同时使用两条 PCIe 路径。根据网络
连接方式，可以用一根 200G 线或两根 100G 线。主机与各 worker 的对应 rail
必须互通。

```bash
# 主机：播放器、shim 和集群配置
./install.sh install --cluster-host

# 每台 worker：计算服务和托盘
./install.sh install --cluster-worker
```

主机安装时输入自身 rail-0 地址和逗号分隔的 worker rail-0 地址；可另填按
相同顺序排列的 SSH 管理地址，供部署和 benchmark 使用。worker 安装时输入
自身地址和主机地址。网口、RDMA 设备及端口也可通过环境变量指定。非交互
安装时，主机设置 `CLUSTER_SELF_IP`、`CLUSTER_WORKER_HOSTS`，以及可选的
`CLUSTER_WORKER_SSH_HOSTS`、`CLUSTER_WORKER_USER`；每台 worker 设置
`CLUSTER_SELF_IP` 和 `CLUSTER_PEER_IP`。

配置保存在 `~/.config/dgxspark-mpv/cluster.conf`，请勿公开。主机优先使用
`CLUSTER_WORKER_HOSTS`，未设置时使用单个 `CLUSTER_WORKER_HOST`。
`CLUSTER_RDMA_DEVS` 与 `CLUSTER_RDMA_GIDS` 按相同顺序列出本机 rail 和
GID 索引。默认 RDMA 端口为 29900，存活检测端口为 29905；额外 rail 使用
连续的 RDMA 端口。

服务名为 `dgxspark-cluster-worker.service`。旧 `dual.conf`、`DUAL_*`
环境变量、旧安装参数和旧服务名仍可兼容使用；新旧环境变量同时设置时，
以 `CLUSTER_*` 为准。重新安装时可用 `--reuse-cluster-config` 保留配置。

## 运行模式

- **普通模式**：主机参与 GPU 计算，按优先队列把任务分给主机和可用 worker；
  一个 worker 时默认使用。
- **Tic-tac 模式**：主机负责播放，worker 动态领取源帧并处理对应插帧区间。
  至少需要两个 worker；配置三机及以上且插帧倍率为 2、3 或 4 时默认使用。
  `CLUSTER_TICTAC=0` 强制普通模式，`=1` 在满足条件时选择 tic-tac。

三机普通模式在测试中的平均吞吐较高；tic-tac 波动较小，并避免主机播放与
计算竞争 GPU。实际结果取决于素材和系统负载。

Tic-tac 的 worker 优先处理可执行的原始帧任务，并直接向负责前一区间的
worker 发送预处理特征。原始帧计算、worker 间传输和结果回传可以并行。
默认由主机按播放顺序拉取完成的结果；普通模式的最终结果也由主机拉取。
`CLUSTER_TICTAC_PULL=0` 可让 tic-tac worker 主动发送结果。worker 断线时
管线使用剩余 worker 恢复；只剩一个 worker 时使用普通模式，没有可用
worker 时回退本机。恢复和实时播放表现取决于负载。

**Shift+F9** 开关多机 offload；**F9** 按 4 → 3 → 2 → 1 循环插帧倍率；
**F8** 在支持时开关多机超分。这些操作需要重建滤镜。

## 默认插帧倍率

源帧率未知时默认 ×2。可用 `CLUSTER_INTERP_MULT` 指定倍率，或在播放时
使用 F9。双机、三机或显式选择普通模式时：

| 源 | 源帧率 | 默认 |
|---|---:|---:|
| ≤720p | ≤25 | ×4 |
| ≤720p | >25–30 | ×3 |
| 1080p 或 4K | ≤25 | ×3 |
| 1080p 或 4K | >25–30 | ×2 |
| 任意 | >30 | ×1 |

配置四机及以上且使用 tic-tac 时：

| 源 | 源帧率 | 默认 |
|---|---:|---|
| ≤720p | ≤30 | ×4，原始分辨率光流 |
| 1080p | ≤24 | ×4，原始分辨率光流 |
| 1080p | >24–30 | ×3，原始分辨率光流 |
| 4K | ≤24 | ×2，原始分辨率光流 |
| 4K | >24–30 | ×3，降采样光流 |
| 支持的分辨率 | >30–60 | ×2；4K 使用降采样光流 |
| 任意 | >60 | ×1 |

低分辨率档不超过 1328×720；超过 2048×1128 的源进入 4K 档。
`CLUSTER_TICTAC_4K_NATIVE=0` 关闭自动原始 4K 光流，`=1` 显式请求原始
光流。原始 4K ×2 的实测余量较小；默认倍率不保证所有素材都能实时播放。

## 开发

`scripts/dev_deploy.sh --worker` 安装当前主机文件，并向所有配置的 worker
复制运行模块；复制后需要重启 worker 服务。`tests/` 包含调度、传输、故障
恢复及安装器测试；硬件 benchmark 见 `bench/README.zh-CN.md`。
