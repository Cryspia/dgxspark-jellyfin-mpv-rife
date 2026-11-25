# Performance

[简体中文](./performance.zh-CN.md)

Performance depends on the source, interpolation multiplier, output size,
network, number of workers and display load. Measurements with `--vo=null`
reflect processing throughput; they do not guarantee smooth displayed playback.

A 2026-11-25 four-machine tic-tac run on a synthetic 1080p24 YUV420P source,
with 4K output and two RDMA rails, measured 152.3 fps at ×2 and 108.6 fps at
×4. The corresponding worker GPU utilisation means were approximately 96%
for ×2 and 80–82% for ×4. Samples were taken at 500 ms intervals with
`nvidia-smi`; this is not an SM occupancy measurement. See
[benchmark results](../bench/results.md) for the test conditions.

The four-machine native-resolution 4K ×2 path has limited measured headroom.
Default multipliers are workload targets, not guarantees. See the
[multi-machine guide](../dual_machine/README.md) for the policy and modes, and
[bench/README.md](../bench/README.md) for test commands.
