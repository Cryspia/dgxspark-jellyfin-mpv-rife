"""Bridge between the native C++ Mode-B filter and Python compute.

The C++ filter (libdgxspark_split_dual.so) takes a `compute_id` argument and
resolves it via `get_callback(cb_id)` here. Callers register their
callable with `register_callback(fn) -> int` and pass the returned id
to `core.dgxspark_split_dual.SplitDual(clip, compute_id=cb_id)`.

This pattern sidesteps VapourSynth's VSMap not having a clean way to
carry a real Python `PyObject*` across the C++/Python boundary inside a
filter constructor — using ids keeps the wiring in pure Python.

The callable signature (see split_dual_filter.cpp splitDualGetFrame for the
exact call) is:

    compute(pair_k: int, phase: int,
            sa_y: int, sa_u: int, sa_v: int,
            sb_y: int, sb_u: int, sb_v: int,
            dst_y: int, dst_u: int, dst_v: int,
            sa_y_stride: int, sa_uv_stride: int,
            dst_y_stride: int, dst_uv_stride: int) -> None

The plane pointers are raw addresses (use numpy.frombuffer + ctypes
to wrap into ndarrays at the dimensions implied by the clip). phase=0
means "src_a SR output frame", phase=1 means "interp SR output frame".
"""

from __future__ import annotations
import threading

_lock = threading.Lock()
_callbacks: dict[int, object] = {}


def register_callback(fn) -> int:
    """Register a callable and return its id, suitable for passing as
    the `compute_id` kwarg to core.dgxspark_split_dual.SplitDual."""
    cb_id = id(fn)
    with _lock:
        _callbacks[cb_id] = fn
    return cb_id


def unregister_callback(cb_id: int) -> None:
    with _lock:
        _callbacks.pop(cb_id, None)


def get_callback(cb_id: int):
    """Resolve a registered callable by id, or return None if unknown.
    Invoked by the C++ filter at construction time."""
    with _lock:
        return _callbacks.get(cb_id)
