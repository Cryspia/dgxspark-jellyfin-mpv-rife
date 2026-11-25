# 性能

[English](./performance.md)

性能取决于素材、插帧倍率、输出分辨率、网络、worker 数量及显示负载。
`--vo=null` 测试反映处理吞吐，不能保证实际显示时播放流畅。

2026-11-25 的四机 tic-tac 测试使用合成的 1080p24 YUV420P 源、4K 输出和
两条 RDMA rail：×2 测得 152.3 fps，×4 测得 108.6 fps。worker GPU 使用率
均值分别约为 96% 和 80–82%。使用率由 `nvidia-smi` 每 500 ms 采样，
不代表 SM 占用率。具体条件见 [benchmark 结果](../bench/results.md)。

四机原始分辨率 4K ×2 的实测余量较小。默认倍率是负载目标，不保证所有素材
都能实时播放。模式与默认策略见[多机说明](../dual_machine/README.zh-CN.md)，
测试命令见 [bench/README.zh-CN.md](../bench/README.zh-CN.md)。
