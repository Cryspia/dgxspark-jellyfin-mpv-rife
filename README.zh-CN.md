# dgxspark-jellyfin-mpv-rife

[English](./README.md)

为 **NVIDIA DGX Spark**（GB10，ARM64，CUDA 13，Ubuntu 24.04，GNOME
Wayland）写的可复现一键脚本：**Jellyfin 客户端 + RIFE 实时插帧 +
FSRCNNX 亮度超分**，全部跑在一个 Miniforge conda 环境里。

## 组成

- **mpv 0.41**，从源码编译 —— 启用 vapoursynth + vulkan + wayland +
  x11 + lua。conda-forge aarch64 上的 mpv 是 headless 库构建，没显示
  后端，所以本地重编。
- **vsrife + TensorRT** RIFE 按源分辨率自动选档：≤720p 用 4.26，
  720p<h≤1080p 用 4.6；1080p 以上不开（GB10 在 4K 撑不住），>30fps
  也不开（已经够流畅）。混合精度（fp16 权重 + fp32 累加器）；纯 fp16
  会在快速运动时溢出闪烁，安装脚本 patch 上游 vsrife 固定
  `enabled_precisions={fp16, fp32}`。
- **GPU YUV↔RGB 色转**（`vs_gpu_helpers.rife_yuv`）—— CPU 上
  `core.resize.Bicubic(format=vs.RGBH)` 在 4K 下 ~30 ms/帧，是 20 核
  Grace CPU 真正的瓶颈。我们把矩阵乘 + chroma 重采样放 GPU；
  Grace+Blackwell 一致内存下没有 PCIe 上传开销。
- **FSRCNNX cuDNN 超分** —— 安装时从上游
  [`Cryspia/fsrcnnx-cudnn`](https://github.com/Cryspia/fsrcnnx-cudnn)
  的 release bundle 拉（包含 x2_8 / x2_16 / x3_16 / x4_16 四档）。链
  路按源↔目标比例自动选一档；F8 可以手动切。具体的每档基准、family /
  scale 选档规则，详见上游 README。
- **jellyfin-mpv-shim** —— 连 Jellyfin、用 IPC 控 mpv 的 Python 客户
  端。安装脚本 patch 它适配 mpv 0.41（`osc=False` 已移除，翻译成
  `script-opts=osc-visibility=...` 由 shim 自己的 `enable_osc` 决定）。

## 系统要求

- DGX Spark（或任何 NVIDIA aarch64 + CUDA 13 的系统）。**不支持**
  x86_64。
- Ubuntu 24.04（其他发行版理论上能跑，没测过）。
- GNOME Wayland 会话（AppIndicator 托盘依赖）。
- `sudo`（apt 那一步用）。约 5 GB 空闲磁盘。

## 用法

```bash
./install.sh install                 # 完整干净安装
./install.sh install --no-mirrors    # 国外用：跳过 USTC 镜像
./install.sh install --no-danmaku    # 不装弹幕插件
./install.sh install --rebuild-trt   # 清空并重建 TRT engine cache
./install.sh install --set-default-video   # 把 mpv-conda 设为 GNOME 默认
./install.sh status                  # 查看当前安装状态
./install.sh uninstall               # 卸载（保留 apt + miniforge）
```

`./install.sh install --help` 列出所有 flag。`install` 完后注销重登
一次让 GNOME 刷新启动器和自启动。新分辨率第一次播放会现场编译 TRT
engine（一次 30–60 秒，之后缓存在
`~/miniforge3/envs/vsmpv/lib/python3.12/site-packages/vsrife/models/`）。

## 默认配置

| 源分辨率 | 帧率 | RIFE | FSRCNNX (auto) |
|---|---|---|---|
| ≤ 720p | ≤ 30 fps | 4.26 @ scale=1.0 | x3_16 / x4_16（按比例） |
| 720p < h ≤ 1080p | < 25 fps（电影帧率） | 4.26 @ scale=1.0 | x2_16 |
| 720p < h ≤ 1080p | 25–30 fps | 4.6 @ scale=1.0 | x2_8 |
| 1080p < h ≤ 2160p | < 25 fps（电影帧率） | mixed 模式，interp 用 4.26 | interp 上 x2_16 |
| 1080p < h ≤ 2160p | 25–30 fps | mixed 模式，interp 用 4.6 | interp 上 x2_8 |
| 任意 | > 30 fps | 关 | 比例满足时仍跑 |

25 fps 阈值是预算分界：≤24 fps 源 × 2 → 48 fps 输出 → 20.8 ms/帧，
重链（RIFE 4.26 + 16-layer FSRCNNX）刚好能装下。25+ fps 时输出预算
压到 16.7 ms，重链装不下 —— 所以 25/29.97/30 fps 内容走轻链
（4.6 + 8-layer）。

4K 档走 **mixed 模式**：real 帧原 4K bit-exact passthrough，仅合成的
interp 帧走下采样 → RIFE → SR 升回 4K 这条路。因为链只在一半帧上跑，
单帧预算翻倍，这刚好够用更重的 4.26 模型 + 16-layer FSRCNNX 提升
interp 质量。

在两条原生 4K 视频上（各 240 帧，pipelined）实测：

| 4K 配置 | sample-01 fps | sample-02 fps | 48 fps 预算 |
|---|---|---|---|
| mixed + RIFE 4.26 + 16-layer（现行）| **63.9** | **65.6** | ✅ 余 ~35% |
| RIFE 4.6 @ scale=0.5（half-flow）| 37.1 | 41.3 | ❌ 差 17% |
| RIFE 4.6 @ scale=1.0（full-flow）| 25.6 | 27.3 | ❌ 差 47% |

Y PSNR（vs full-flow baseline）：

| 比较对象 | sample-01 | sample-02 |
|---|---|---|
| real 帧（passthrough）vs 原 4K | 240 dB | 240 dB |
| interp 帧（4.26 @ 1080p + SR）vs full-flow 4.6 | 38.5 dB | 34.8 dB |
| half-flow 的 interp 帧 vs full-flow | 39.1 dB | 35.1 dB |

Real 帧 bit-exact 4K，零损失。Interp 帧 PSNR 比 half-flow 路只低
0.3–0.6 dB —— 两者都在 1080p 估光流，4K vs 1080p 的 warp 差别被 RIFE
时域插帧本身的不确定性主导。Mixed 模式有一个潜在感知风险：sharp-real
和 SR-recovered-interp 交替**可能**在高频纹理上"细节呼吸"（shimmer）。
实测下采样+SR 这条 roundtrip 够温和，real 和 interp 的 Y PSNR 都远超
可感知阈值，我们没看到；如果你看到了请反馈。

显示目标默认 4K。如果是其他分辨率屏幕，启动 mpv 前 export
`FSRCNNX_TARGET_W` / `FSRCNNX_TARGET_H` 即可。

## 快捷键

- **F8** —— 循环切 FSRCNNX 档：`16x4 → 16x3 → 16x2 → 8x2 → OFF →
  loop`。默认从当前源自动选的那一档开始。每按一次会触发 vapoursynth
  filter 重建（~1–3 s 卡顿，cuDNN runner 在 filter 创建时构建，没法
  运行时换档）。换片时回到 auto。
- **F9** —— 独立开关 RIFE。跨片持久化（不会因为换片自动复位）。
- mpv 默认快捷键全保留（`i` 看 stats，`s` 截图等）。

## 弹幕

由 [**Cryspia/mpv-dandanplay-danmaku**](https://github.com/Cryspia/mpv-dandanplay-danmaku)
提供。安装脚本会把它克隆到 `~/src/mpv-dandanplay-danmaku/` 并调
`install.py`。完整快捷键、配置项、问题排查见那个仓库的 README。

## 安装后文件分布

| 路径 | 用途 |
|------|------|
| `~/miniforge3/envs/vsmpv/` | conda 环境：python, mpv, vapoursynth, vsrife, shim, tensorrt |
| `~/.config/mpv/{mpv,input}.conf, rife.vpy, vs_gpu_helpers.py, sr_keys_helper.py` | 本项目的配置 + 辅助模块 |
| `~/.config/mpv/scripts/sr_keys.lua` | F8 / F9 keybind 逻辑 |
| `~/.config/mpv/scripts/fsrcnnx-cudnn/` | 上游 fsrcnnx-cudnn bundle（Python 包 + .npz 权重） |
| `~/.config/mpv/scripts/dandanplay/` | 弹幕脚本 |
| `~/.config/jellyfin-mpv-shim/conf.json` | shim 自己的配置（服务器凭据等） |
| `~/.config/jellyfin-mpv-shim/{mpv,input}.conf, rife.vpy, …` | 指向 `~/.config/mpv/` 的软链 |
| `~/.local/bin/{mpv-conda,jellyfin-mpv-shim}` | 启动 wrapper（设 PYTHONHOME / GI_TYPELIB_PATH） |
| `~/.local/share/applications/*.desktop` | 启动器 |
| `~/.config/autostart/jellyfin-mpv-shim.desktop` | 登录自启动 |
| `~/src/mpv/`, `~/src/mpv-dandanplay-danmaku/` | 源码副本（可删，下次安装重新克隆） |

## 工程结构

```
dgxspark-jellyfin-mpv-rife/
├── install.sh           # 安装 / 状态 / 卸载
├── rife.vpy*            # 由 install.sh 写入
├── vs_gpu_helpers.py    # rife_yuv：GPU YUV↔RGB + vsrife wrapper
├── sr_keys_helper.py    # F8/F9 side-channel + apply_fsrcnnx
└── icons/               # shim + mpv-conda 启动器图标
```

`*` `rife.vpy` 和 lua 脚本由 `install.sh` 在
`~/.config/mpv/` 里生成，仓库里没有。

## 一些值得记住的点

- **fp16 混合精度**对 vsrife 是关键 —— 纯 fp16 会让光流向量累加器溢
  出，快速运动时闪烁。
- **PYTHONHOME** 由 `mpv-conda` wrapper 设。没设的话非 conda 激活态
  下启动 mpv 找不到内嵌 Python 的 stdlib，vapoursynth filter 初始化
  失败。
- **`gpu-context=waylandvk`** 比 `x11vk` 延迟低 ~1 ms，但失去 GNOME
  的窗口装饰（mpv 0.41 没有 libdecor 支持）。
- **`hidpi-window-scale=yes`** 是 4K HiDPI 屏上 FSRCNNX 能触发的关
  键 —— 没开的话 mpv 会按逻辑 1080p 渲染，比例 gate 永远是 1.0。
- **F8 / F9 重建成本** —— 都通过重建 vapoursynth filter 实现。cuDNN
  runner 在 filter 创建时编译，所以按一次后会卡 1–3 s。手动覆盖偶尔
  按一次能接受。
