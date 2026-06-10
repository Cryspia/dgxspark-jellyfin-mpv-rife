# dgxspark-jellyfin-mpv-rife

[English](./README.md)

可复现的 **Jellyfin 客户端 + RIFE 实时插帧 + FSRCNNX luma 超分** 栈，
跑在 **NVIDIA DGX Spark** (GB10, ARM64, CUDA 13, Ubuntu 24.04, GNOME
Wayland) 上。

可选 **双机模式** (experimental)：把链路切到两台 Spark 上通过 200 G
RoCE 互联，稳态 fps 大约翻倍。

## 用法

```bash
./install.sh install                 # 单机, 完整清装
./install.sh install --no-mirrors    # 中国大陆外 — 跳过 USTC 镜像
./install.sh install --no-danmaku    # 跳过弹幕插件
./install.sh install --dual-host     # 双机的主机
./install.sh install --dual-secondary # 双机的从机 (只装 worker.py + 任务栏退出 app)
./install.sh status                  # 查当前装了什么
./install.sh uninstall               # 卸载 (保留 apt + miniforge)
```

`./install.sh install --help` 列出所有参数。清装后注销重登 GNOME 才会
看到新的启动器 + 自启项。

## 单机栈

- **mpv 0.41** 源码编译 — vapoursynth + vulkan + wayland + x11 + lua.
  conda-forge 上的 aarch64 mpv 是无显示后端的 headless library 版，所以
  脚本自己编一份。
- **vsrife + TensorRT** 按源分辨率 + 帧率挑 RIFE 模型，每种组合的预算
  正好够 GB10 的单帧 envelope。 完整表见 [默认配置](#默认配置)。
- **GPU YUV↔RGB 色彩转换** (`vs_gpu_helpers.rife_yuv`) — 矩阵乘 + chroma
  resample 都跑在 GPU 上，而不是 zimg 的 CPU 路径 （后者 4K 单帧 ~30 ms
  在 Grace CPU 上）。 支持 YUV 4:2:0 / 4:2:2 / 4:4:4 / 8 / 10 / 12 / 16 bit,
  BT.709 / 601 / 2020 NCL,limited 或 full range.
- **FSRCNNX cuDNN 超分** — 从上游
  [`Cryspia/fsrcnnx-cudnn`](https://github.com/Cryspia/fsrcnnx-cudnn)
  release 包安装 （variants x2_8 / x2_16 / x3_16 / x4_16）。 链路自动按
  源 ↔ 目标比挑。
- **KrigBilateral chroma kriging** — luma 引导的 chroma 升采样，所有
  chroma resize 步骤都用。 内部 CUDA kernel 在 `fsrcnnx-cudnn` 里 +
  mpv 显示 GLSL ([igv 的移植](https://gist.github.com/igv/a015fc885d5c22e6891820ad89555637)).
- **jellyfin-mpv-shim** — 通过 IPC 驱动 mpv 的 Python 客户端。
- **弹幕** — [Cryspia/mpv-dandanplay-danmaku](https://github.com/Cryspia/mpv-dandanplay-danmaku).

## 默认配置

| 源 | 帧率 | RIFE | FSRCNNX （自动） |
|---|---|---|---|
| ≤ 720p | ≤ 30 fps | 4.26 @ scale=1.0 | x3_16 / x4_16 （按比例选） |
| 720p < h ≤ 1080p | < 25 fps （电影） | 4.26 @ scale=1.0 | x2_16 |
| 720p < h ≤ 1080p | 25–30 fps | 4.6 @ scale=1.0 | x2_8 |
| 1080p < h ≤ 2160p | < 25 fps （电影） | mixed mode, 4.26 插帧 | x2_16 插帧 |
| 1080p < h ≤ 2160p | 25–30 fps | mixed mode, 4.6 插帧 | x2_8 插帧 |
| 任意 | > 30 fps | 关 | 比例值得才跑 |

25 fps 阈值是预算切分：≤24 fps 源 × 2 → 48 fps 输出 → 20.8 ms/帧，重链
（RIFE 4.26 + 16-layer FSRCNNX） 正好够；25+ fps 时输出预算降到 16.7 ms，
重链就跑不动 — 所以 25/29.97/30 fps 内容用轻链 （4.6 + 8-layer）.

4K 源走 **mixed mode**： 真实帧原始 4K 直通 （bit-exact），只在合成的中间
帧上走 降采样 → RIFE → SR 升采样路径。每帧预算大概翻倍，4.26 + 16-layer
重链能跑。

显示目标默认 4K。 屏幕小的设 `FSRCNNX_TARGET_W` / `FSRCNNX_TARGET_H` env
覆盖。

## 双机模式 （experimental）

两台 DGX Spark 通过 200 G RoCE 互联，把 CCSR / INTERP / SR_INTERP 切到
两块 GB10 上跑。 稳态吞吐约单机 1.95×，真实帧 byte-identical，插帧视觉
上一致。

安装：
```bash
# 主机 (mpv + shim + dual host 服务)
./install.sh install --dual-host

# 从机 (只装 worker.py + 任务栏退出 app,无 mpv 主程序)
./install.sh install --dual-secondary
```

两边都会提示输 cluster 网络配置 （RDMA 设备 / 端口 / 自身 IP / 对端 IP）；
默认值从 `ip link` / `ip a` 取。 想脚本化安装就预设 `--help` 列出的 env.

播放时：**Shift+F9** 开关双机 offload。 连不上自动回退到单机模式 （再按一
次重试）；会话中 worker 死亡同样自动回退 —— host 侧 liveness watchdog
~3 秒内感知，回收在途任务，播放无缝降级单机继续。

→ 完整设计 + 组件细节：[`dual_machine/README.zh-CN.md`](./dual_machine/README.zh-CN.md).

## 性能和颜色准度

→ [`docs/performance.zh-CN.md`](./docs/performance.zh-CN.md) — fps,
GPU 使用率，每任务耗时。
→ [`docs/color-accuracy.zh-CN.md`](./docs/color-accuracy.zh-CN.md) — vs
单机基准 PSNR, byte-identical 确定性保证。

## Benchmark

→ [`bench/README.zh-CN.md`](./bench/README.zh-CN.md) — color / fps /
per-task timing 脚本。 测试视频用 `bench/gen_clips.sh` 合成 （不依赖本机
路径）.

## 按键

- **F8** — 切 FSRCNNX 变体 （`16x4 → 16x3 → 16x2 → 8x2 → 关 → 循环`）；
  每按一次触发 vapoursynth filter reload （1–3 s 卡顿）。 下次文件加载重置
  回 auto.
- **F9** — 单机：开关 RIFE。 双机：循环插帧倍数 （4 → 3 → 2 → 1 → off）.
- **Shift+F8** — 开关显示阶段 KrigBilateral chroma GLSL。 即时 （不 reload
  vf）； 跨文件 persist.
- **Shift+F9** — 开关双机 offload （未 `--dual-host` 安装时无效）。 连接失败
  自动回退。
- 所有 mpv 默认键都在 （`i` 看 stats, `s` 截图，等等）。

## 装完后东西在哪

| 路径 | 用途 |
|---|---|
| `~/miniforge3/envs/vsmpv/` | conda env: python, mpv, vapoursynth, vsrife, shim, tensorrt |
| `~/.config/mpv/{mpv,input}.conf, rife.vpy, vs_gpu_helpers.py, sr_keys_helper.py` | 单机配置 + helper |
| `~/.config/mpv/scripts/{sr_keys,warmup}.lua` | F8 / F9 / Shift+F8 / Shift+F9 + 文件加载后 warmup |
| `~/.config/mpv/shaders/KrigBilateral.glsl` | 显示阶段 chroma kriging shader |
| `~/.config/mpv/fsrcnnx-cudnn/` | 上游 fsrcnnx-cudnn 包 （含 `chroma_krig` CUDA kernel） |
| `~/.config/mpv/scripts/dandanplay/` | 弹幕脚本 |
| `~/.config/jellyfin-mpv-shim/conf.json` | shim 自己的配置 （服务器凭据等） |
| `~/.config/dgxspark-mpv/dual.conf` | 双机集群配置 （主/从 IP, RDMA dev） — `--dual-host` / `--dual-secondary` 后才有 |
| `~/.local/bin/{mpv-conda,jellyfin-mpv-shim}` | wrapper (PYTHONHOME / GI_TYPELIB_PATH) |
| `~/.local/share/applications/*.desktop` | 启动器入口 |
| `~/.config/autostart/jellyfin-mpv-shim.desktop` | 登录自启 shim |
| `~/src/{mpv,mpv-dandanplay-danmaku}/` | 源码检出 （能删，下次安装会重 clone） |
| `~/.cache/dgxspark-mpv/trt-engines/` | 编译好的 TRT 引擎备份 —— uninstall / 重建 env 后保留，下次安装直接还原省 4–7 分钟重编译 |

## 系统要求

- DGX Spark （或任意 NVIDIA aarch64 + CUDA 13）。 x86_64 **不支持**。
- Ubuntu 24.04 （其他发行版可能能跑但没测过）。
- GNOME Wayland session — AppIndicator 任务栏 icon 需要。
- apt-install 步骤需要 `sudo`。 每台机要 ~5 GB 空盘。
- 双机要：两台机 + 200 G RoCE 静态 IP 链路。

## 项目结构

```
dgxspark-jellyfin-mpv-rife/
├── install.sh           # 安装器 / 状态 / 卸载 (单机 + 双机变体)
├── vs_gpu_helpers.py    # rife_yuv: GPU YUV↔RGB + vsrife 包装
├── sr_keys_helper.py    # F8/F9 side-channel + apply_fsrcnnx
├── scripts/             # sr_keys.lua + warmup.lua mpv 插件
├── shaders/             # vendored mpv GLSL shader (KrigBilateral)
├── dual_machine/        # 双机 offload — 见其 README
├── docs/                # 性能 + 颜色准度数据表
├── bench/               # color / fps / timing bench 脚本
└── icons/               # shim + mpv-conda 启动器图标
```

## 几个要记住的点

- **fp16 mixed-precision** 对 vsrife 是关键 — 纯 fp16 在快动作时 flow
  向量累加器会溢出 （能看到闪烁）。
- **PYTHONHOME** 由 `mpv-conda` wrapper 设。 不设的话，在 conda activate
  外启动的 mpv 找不到嵌入 Python 的 stdlib → vapoursynth filter init
  失败。
- **`gpu-context=waylandvk`** 比 `x11vk` 低 ~1 ms 延迟，但会丢 GNOME
  绘的窗口装饰 （mpv 0.41 没 libdecor）。
- **`hidpi-window-scale=yes`** 是 4K HiDPI 屏上 FSRCNNX 能触发的前提 —
  不开 mpv 按逻辑 1080p 渲染，比例 gate 一直是 1.0.
- **F8 / F9 reload 成本** — 都靠重建 vapoursynth filter 实现。 cuDNN
  runner 在 filter 创建时构建，所以每次按之后有 1–3 s 卡。 偶尔手动覆盖
  能接受。

## 反馈

先跑 `./install.sh status`，然后到项目 GitHub 提 issue.
