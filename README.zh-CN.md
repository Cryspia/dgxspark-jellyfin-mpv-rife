# dgxspark-jellyfin-mpv-rife

[English](./README.md)

为 **NVIDIA DGX Spark**（GB10，ARM64，CUDA 13，Ubuntu 24.04，GNOME Wayland）写的可复现一键脚本：**Jellyfin 客户端 + RIFE 实时插帧 + FSRCNNX 亮度超分**，整套跑在一个 Miniforge conda 环境里。

整条链路里包含的组件：

- **mpv** —— 从源码编译（mpv 0.41），同时启用 vapoursynth + vulkan + wayland + x11 + lua。conda-forge 上 aarch64 的 mpv 是没有显示后端的 headless 库构建，没法直接当播放器用，所以脚本自己重新编译。
- **vsrife + TensorRT** —— RIFE 4.26 模型，TRT 混合精度（fp16 权重 + fp32 累加器）。脚本会 patch 上游 vsrife，把 `use_explicit_typing=True` 换成 `enabled_precisions={fp16, fp32}`，否则光流向量在快速运动场景下会溢出导致闪烁。
- **FSRCNNX** glsl 着色器 —— 2x 亮度超分，仅在渲染目标 > 源 × 1.3 时才生效（例如 1080p 源播到 4K HiDPI 屏）。
- **jellyfin-mpv-shim** —— 连接 Jellyfin 服务器、通过 IPC 控制 mpv 的 Python 客户端。脚本会 patch 它去掉 `--osc=no` 这个 mpv 0.41 已经移除的命令行参数，否则启动就会报错。

## 系统要求

- DGX Spark（或任何 NVIDIA aarch64 + CUDA 13 的系统）。**不支持** x86_64 —— 例如「从源码编译 mpv」这一步是为了绕过 conda-forge aarch64 上没有显示后端的问题，不适用于其他架构。
- Ubuntu 24.04（其他发行版理论上能跑但没测过）。
- GNOME Wayland 会话（AppIndicator 托盘图标依赖它）。
- `sudo` 权限（apt 那一步要用）。
- 至少 5 GB 空闲磁盘（conda 环境 + TRT engine cache）。

## 用法

```bash
# 干净系统上完整安装（默认开启 USTC 镜像 + 弹幕插件）
./install.sh install

# 在国外用 —— 跳过 USTC conda-forge / PyPI 镜像配置
./install.sh install --no-mirrors

# 不安装弹幕（bullet-chat）插件
./install.sh install --no-danmaku

# 两者都关
./install.sh install --no-mirrors --no-danmaku

# 查看当前安装状态
./install.sh status

# 卸载（保留 apt 包和 miniforge 本身，方便重装）
./install.sh uninstall
```

完整可选参数见 `./install.sh install --help`。

`install` 完成后请注销并重新登录一次，让 GNOME 重新扫描启动器、自启动条目和图标缓存。第一次播放会现场编译 TensorRT RIFE engine（一次大约 30–60 秒），之后启动会非常快，engine 缓存路径在 `~/miniforge3/envs/vsmpv/lib/python3.12/site-packages/vsrife/models/`。

## 默认配置（针对 1080p 源 + 4K HiDPI 屏调好）

- RIFE 4.26，scale=1.0，2/1 倍率，TRT 混合精度
- FSRCNNX 8-0-4-1 亮度超分
- vo=gpu-next，gpu-context=waylandvk（在 GNOME 下没有窗口装饰，但比 Xwayland 少约 1 ms 的呈现延迟）
- hidpi-window-scale=yes（4K HiDPI 显示器上 FSRCNNX 才会触发）

可调项的注释直接写在 `~/.config/mpv/rife.vpy` 和 `~/.config/mpv/mpv.conf` 里。

## 快捷键

- **F8** —— FSRCNNX 着色器开关
- **F9** —— RIFE 切换：4.26（默认）→ 4.6（轻量 fallback）→ 关
- **F10** —— 弹幕开关（也可以点画面右上角的「弹」图标）
- **Shift+F10** —— 弹幕设置面板（透明度、字号、速度、密度、显示区域、渲染模式、防重叠、来源过滤、每集时间偏移、去重、繁简转换 等等）
- **Ctrl+F10** —— 弹幕手动模糊搜索（输入关键词 → 选番剧 → 选剧集）
- **i** / **Shift+I** —— mpv 自带的统计信息浮层
- **Ctrl+S** —— mpv 自带的截图
- 其他 mpv 默认快捷键全部保留

## 弹幕（bullet-chat）

由独立项目 [**Cryspia/mpv-dandanplay-danmaku**](https://github.com/Cryspia/mpv-dandanplay-danmaku) 提供，是一个基于 dandanplay 的 mpv 弹幕脚本。本脚本会在 step 8b 把它 clone 到 `~/src/mpv-dandanplay-danmaku/` 然后调用其 `install.py`。完整功能、配置参考、故障排查都在那个仓库的 [README](https://github.com/Cryspia/mpv-dandanplay-danmaku/blob/main/README.zh-CN.md) 里。

简要功能介绍：

- **自动匹配**：根据文件名，或在通过 jellyfin-mpv-shim 启动时根据 Jellyfin 元数据。当资料库剧名和 dandanplay 上不一致时，第一次手动选择以后会自动记忆映射，下次同剧别集自动 fallback。
- **手动模糊搜索**（Ctrl+F10）—— 两级选择器：输入关键词 → 选番剧 → 选剧集。
- **播放器内设置面板**（Shift+F10）—— 透明度、字号、速度、密度、显示区域、渲染模式、防重叠、来源过滤（B站/巴哈/弹弹/其他）、繁简转换、去重、每集时间偏移等等。
- **CJK 感知的行宽计算 + 分带滚动**：同一行连续两条中文弹幕不会重叠，弹幕较少时聚集在屏幕上 1/4。
- **默认使用上游的 CORS 代理**；强烈建议申请你自己的 dandanplay AppId 走直连，详见弹幕项目的 README。

## 安装后文件路径速查

| 路径 | 用途 |
|------|------|
| `~/miniforge3/envs/vsmpv/` | conda 环境：python、mpv、vapoursynth、vsrife、shim、tensorrt |
| `~/.config/mpv/{mpv,input}.conf, rife.vpy, shaders/` | mpv 用户配置（shim 通过 symlink 共用同一份） |
| `~/.config/jellyfin-mpv-shim/conf.json` | shim 自己的配置（服务器凭证等） |
| `~/.config/jellyfin-mpv-shim/{mpv.conf,input.conf,rife.vpy,shaders,scripts}` | symlink 到 `~/.config/mpv/` |
| `~/.local/bin/{mpv-conda,jellyfin-mpv-shim}` | 启动 wrapper（设置 PYTHONHOME / GI_TYPELIB_PATH） |
| `~/.local/share/applications/*.desktop` | 桌面环境启动器条目 |
| `~/.config/autostart/jellyfin-mpv-shim.desktop` | 登录后 shim 自启动 |
| `~/.local/share/icons/hicolor/*/apps/{mpv-conda,jellyfin-mpv-shim}.png` | 应用图标 |
| `~/.config/mpv/scripts/dandanplay/` | 弹幕脚本包（由 [Cryspia/mpv-dandanplay-danmaku](https://github.com/Cryspia/mpv-dandanplay-danmaku) 安装） |
| `~/.config/mpv/danmaku-{config,credentials,settings}.json` | 弹幕代理 / 凭证 / 面板设置 |
| `~/.cache/mpv-danmaku/{matches,offsets,aliases}.json` | 弹幕匹配缓存、时间偏移、智能别名 |
| `~/src/mpv/` | mpv 源码 checkout（再次构建用） |
| `~/src/mpv-dandanplay-danmaku/` | 弹幕项目的 clone（每次 install 时 fetch 更新） |

## 项目结构

```
dgxspark-jellyfin-mpv-rife/
├── README.md
├── README.zh-CN.md
├── install.sh                  # 安装 / 状态查询 / 卸载入口
├── shaders/
│   └── FSRCNNX_x2_8-0-4-1.glsl
└── icons/
    ├── shim/{16,32,48,64,128,256}.png
    └── mpv/{16,32,64,128}.png + scalable.svg
```

弹幕插件原本是这个仓库的一个子目录（`danmaku/`），现在已经独立到 [Cryspia/mpv-dandanplay-danmaku](https://github.com/Cryspia/mpv-dandanplay-danmaku) 仓库。本脚本会自动 fetch 它（默认追 `main` 分支；如果想锁定到具体版本，把 `install.sh` 里的 `DANMAKU_REF` 改成 tag 即可）然后调用它的 install.py，你不需要手动 clone。

把这个仓库 clone 到一台干净的 DGX Spark，运行 `./install.sh install` 就能完整复现整个安装。

## 一些值得记住的坑

- **fp16 混合精度是关键**。如果用 vsrife 默认的 `use_explicit_typing=True`（即纯 fp16 包括累加器），快速运动场景会因为光流向量溢出而闪烁。
- **PYTHONHOME 是关键**。如果 mpv 不通过 conda activate 启动，找不到内嵌 Python 的标准库，vapoursynth filter 初始化失败，画面立刻 EOF 只剩声音。
- **gpu-context=waylandvk** 比 `x11vk` 延迟低，但失去 GNOME 绘制的窗口装饰（mpv 0.41 不支持 libdecor）。
- **hidpi-window-scale=yes** 是 FSRCNNX 在 4K HiDPI 屏上能触发的关键 —— 否则 mpv 按逻辑 1080p 渲染，着色器里的 `//!WHEN OUTPUT/LUMA > 1.3` 永远不成立。
- **NVENC 串流（Sunshine / Moonlight）大约要吃 20% GPU**。如果你这套配置在串流里掉帧，瓶颈大概率是串流管线，不是 RIFE。本地 4K 屏上 RIFE 4.26 + scale=1.0 + FSRCNNX 完全跑得动，零掉帧。
