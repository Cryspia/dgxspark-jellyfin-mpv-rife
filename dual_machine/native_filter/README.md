# Native C++ SplitDual vapoursynth filter

C++ implementation of the vapoursynth filter that mpv's frame_thread
calls into. Owns the pair cache, dispatch shell, and host↔worker
hand-off. Computes nothing itself — it calls a pybind11-registered
`compute_callable` (set up by `native_dispatcher.make_dispatcher`)
which actually drives the GPU kernels and RDMA.

This filter exists because the previous pure-Python `ModifyFrame`
dispatcher capped at ~42 fps on 1080p → 4K: 8 parallel
`ModifyFrame` callbacks all contended on the Python GIL for the
dispatch envelope (cache lock, scheduler pick, frame-prop reads,
vapoursynth metadata I/O) even though each one released the GIL
inside torch ops. The C++ filter keeps the dispatch envelope
GIL-free and only re-acquires the GIL for the registered
`compute_callable`, which itself releases the GIL inside its torch /
pyverbs calls.

## Build

```bash
conda activate vsmpv
cd dual_machine/native_filter
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The plugin lands at `build/libdgxspark_split_dual.so`. The host
install (`install.sh install --dual-host`) compiles + places it in
the right spot automatically.

## API

```python
core.dgxspark_split_dual.SplitDual(clip, scale=2, compute_id=N, interp_mult=2)
```

- `scale` — spatial upscale factor (matches the FSRCNNX variant's
  output scale; usually 2).
- `compute_id` — id returned by `native_helper.register_callback(fn)`
  on the Python side. The filter calls `fn(pair_k, phase, sa_y, …,
  dst_y, …)` for each frame pair.
- `interp_mult` — temporal multiplier (1 / 2 / 3 / 4). Determines
  output VideoInfo's num_frames.

See `dual_machine/native_dispatcher.py` for the full Python side
that constructs the `compute_callable`.
