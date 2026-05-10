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
| ≤ 720p | ≤ 30 fps | 4.26 | x3_16 / x4_16（按比例） |
| 720p < h ≤ 1080p | ≤ 30 fps | 4.6 | x2_8 |
| > 1080p | 任意 | 关 | bypass（ratio < 1.3） |
| 任意 | > 30 fps | 关 | 比例满足时仍跑 |

我们试过 4K 源走 RIFE 4.6 @ scale=0.5（半光流），实测 GB10 撑不住，
丢的帧比加的还多，所以直接关掉了；4K 源现在裸通过。FSRCNNX 在 4K 源
+ 4K 屏的场景下，根据上游
[1.3× WHEN gate](https://github.com/Cryspia/fsrcnnx-cudnn) 也会自动
bypass。

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
