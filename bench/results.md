# Benchmark results

These measurements are workload-specific observations, not real-time playback
guarantees. See [bench/README.md](./README.md) for benchmark prerequisites.

## Four-machine tic-tac, 1080p → 4K

Measured 2026-11-25 with one playback leader, three workers and two RDMA rails.
The input was a synthetic 1080p24 YUV420P clip. mpv used `--vo=null --untimed`,
12 buffered frames and 24 concurrent frames; each run requested 3,000 output
frames. GPU utilisation was sampled every 500 ms with `nvidia-smi`, without
per-task CUDA synchronisation. The means below use the shared steady interval
across all three workers.

| Interpolation | Dispatcher fps | Worker GPU utilisation means |
|---|---:|---|
| ×2 | 152.3 | 95.7%, 95.9%, 96.0% |
| ×4 | 108.6 | 79.6%, 82.0%, 81.4% |

GPU utilisation indicates whether work was present during a sample; it is not
SM occupancy. This comparison uses the same source and output size, but ×4
performs more interpolation work per output frame. Source content, playback,
network, worker count and background load affect results.

In two alternating comparisons of the installed pipeline and the cleaned
development tree under the same conditions, ×2 measured 152.3/153.1 fps before
and 152.7/151.8 fps after. ×4 measured 108.6/109.9 fps before and 109.2/107.8
fps after. These short runs showed no clear systematic regression; they do not
rule out small differences on other workloads.

## Correctness

A 120-frame FFV1 comparison of four-machine ×4 output before and after the
source-frame scheduling change was byte-identical. A separate 60-frame test
that interrupted one worker and recovered with the remaining workers also
matched the reference output frame-for-frame. These checks cover the tested
clips and configurations, not every possible source.
