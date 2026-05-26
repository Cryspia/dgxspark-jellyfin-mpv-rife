"""Native-filter compute callable for the dual-machine pipeline.

The native C++ filter (dgxspark_split_dual.SplitDual) invokes this callable
once per output frame, passing raw plane pointers. Behind the
callable: NCCL handshake → host_3proc (HostMP) + queue_mgr +
guest_mp (RDMAChannel to remote worker_3proc).

Per-frame flow:
  mpv frame_thread → compute() → _compute_split_task →
  queue_mgr.submit_with_phase_dst → mpv blocks on wait_frame_done →
  background _split_dispatch_loop / _guest_dispatch_loop pop tasks
  → host_mp / guest_mp run CCSR / INTERP / SR_INTERP → _hmp_watcher
  memmove's slot.dst into mpv VA + task_done → frame_done unblocks.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
import time as _time
from collections import OrderedDict, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _parse_fsrcnnx_scale(variant: str) -> int:
    """Parse FSRCNNX_xN_... variant string and return the scale N
    (2/3/4). Raises if unrecognised."""
    import re
    m = re.match(r"FSRCNNX_x(\d)_", variant)
    if not m:
        raise ValueError(
            f"can't parse FSRCNNX scale from variant {variant!r}")
    s = int(m.group(1))
    if s not in (2, 3, 4):
        raise ValueError(
            f"unsupported FSRCNNX scale x{s} in variant {variant!r}")
    return s


def make_dispatcher(
    clip,
    *,
    variant: str,
    matrix_s: str,
    color_range: str,
    rife_model: str = "4.26",
    chroma_kernel: str = "bicubic",
    downsample_pre: int = 1,
    pipeline_depth: int = 4,
    device: str = "cuda:0",
    # Legacy NCCL knob; retained for ABI but no longer applies — the
    # control plane is now a single TCP socket (liveness:29905) and
    # the per-stage timeouts are: 0.5 s for the initial connect, then
    # `DUAL_WORKER_READY_TIMEOUT` (default 300 s) for the worker's
    # 'R' byte after engines load.
    init_timeout_s: float = 3.0,
    interp_mult: int = 2,
    no_sr: bool = False,
):
    """Returns (callable, cleanup_fn). The callable is what the native
    filter invokes. cleanup_fn drains in-flight work and tears down
    NCCL/RDMA on shutdown.

    downsample_pre — pre-CC luma downsample factor for the 4K-source
    path (=2 for 4K-in/4K-out via internal-1080p). Default 1 = no
    downsample (1080p→4K, 720p→4K x3, 480p→x4 etc.).

    Output I/O scale = fsrcnnx_scale(variant) // downsample_pre. So
    FSRCNNX_x2 + downsample_pre=1 → ×2 out; FSRCNNX_x2 + downsample_pre=2
    → ×1 out (4K source path); FSRCNNX_x3 + downsample_pre=1 → ×3 out
    (720p→4K); FSRCNNX_x4 + downsample_pre=1 → ×4 out (480p→1920p).
    """
    if clip.format is None or clip.format.bits_per_sample != 10 \
            or clip.format.num_planes != 3 \
            or clip.format.subsampling_w not in (0, 1) \
            or clip.format.subsampling_h not in (0, 1):
        raise RuntimeError(
            "native_dispatcher: only 10-bit 3-plane YUV with subsampling "
            "in {(0,0), (1,0), (0,1), (1,1)} (4:4:4 / 4:2:2 / 4:2:0) "
            "supported"
        )
    src_sub_w = clip.format.subsampling_w
    src_sub_h = clip.format.subsampling_h

    fsrcnnx_scale = _parse_fsrcnnx_scale(variant)
    if fsrcnnx_scale % downsample_pre != 0:
        raise RuntimeError(
            f"FSRCNNX scale x{fsrcnnx_scale} not divisible by "
            f"downsample_pre={downsample_pre}; would give non-integer "
            f"output dims")
    output_scale = fsrcnnx_scale // downsample_pre
    if no_sr:
        # SR disabled: output dim = source dim, regardless of variant.
        # rife.vpy's bucket gate already rejected no_sr for the 4K-DS
        # (downsample_pre=2) path so we can assume downsample_pre=1
        # here — proc dims == source dims and the worker can copy raw
        # Y straight to ov["yao"] instead of running FSRCNNX. The
        # variant string is still sent for handshake compatibility
        # (VARIANT_IDX must be valid) but the worker won't use it.
        if downsample_pre != 1:
            raise RuntimeError(
                f"no_sr=True with downsample_pre={downsample_pre} is "
                f"inadmissible — RIFE-DS path needs SR to upscale back. "
                f"rife.vpy bucket gate should have rejected this.")
        output_scale = 1

    # Output subsampling decision (4K-like preservation):
    #   • output_scale == 1 (= downsample_pre == fsrcnnx_scale, the
    #     real-frame-passthrough variant for 4K and other non-standard
    #     resolutions; also no_sr): output dim == src dim, so preserve
    #     src chroma subsampling end-to-end. Real frames stay byte-
    #     identical; Blackwell can hwdec 4:2:2 / 4:4:4 sources
    #     downstream.
    #   • output_scale > 1 (1080p / 720p / 480p paths upscaling to a
    #     larger target): force 4:2:0 since the consumer is mpv's 4K
    #     display surface where 4:2:0 is the canonical layout.
    if output_scale == 1:
        dst_sub_w, dst_sub_h = src_sub_w, src_sub_h
    else:
        dst_sub_w, dst_sub_h = 1, 1

    H, W = clip.height, clip.width
    cH, cW = H >> src_sub_h, W >> src_sub_w
    proc_h = H // downsample_pre
    proc_w = W // downsample_pre
    oH = H * output_scale
    oW = W * output_scale
    ocH, ocW = oH >> dst_sub_h, oW >> dst_sub_w
    MAX = float((1 << 10) - 1)

    # ── Singleton-reuse fast path ──────────────────────────────────
    # mpv tears down + recreates the vapoursynth vf chain on every
    # seek, which re-runs rife.vpy → calls make_dispatcher() again.
    # Without reuse, each call respawns 3 host child python procs +
    # rebuilds cuDNN runner (200 ms) + reloads TRT engines (1.4 s) +
    # re-handshakes worker (~1 s) = ~3 s of host work per seek (then
    # mpv catches up over another ~5 s). With the same mpv process
    # python module state survives across vf rebuilds — verified by
    # experiment in _persist_probe.py (same module id, same marker,
    # same pid) — so we can stash the host_mp / queue_mgr / guest_mp
    # / compute closure in dual_machine/_dispatcher_singleton.py and
    # re-hand the closure back when the params match.
    try:
        from _dispatcher_singleton import get as _singleton_get, lock as _singleton_lock
    except Exception:
        _singleton_get = _singleton_lock = None
    # params key: shape, layout, and the per-task knobs that affect
    # subprocess wiring / engine selection. Resolution changes (next
    # video) and mult changes (F9) flip this and force a real teardown.
    _params_key = (H, W, src_sub_w, src_sub_h, dst_sub_w, dst_sub_h,
                    output_scale, downsample_pre, interp_mult, no_sr,
                    variant, matrix_s, color_range, rife_model,
                    chroma_kernel, device)
    if _singleton_get is not None:
        with _singleton_lock():
            _st = _singleton_get()
            if (_st.get("params_key") == _params_key
                    and _st.get("compute") is not None
                    and _st.get("host_mp") is not None):
                _cached_compute = _st["compute"]
                _cached_qmgr = _st.get("queue_mgr")
                sys.stderr.write(
                    "[native_dispatcher] singleton REUSE — params match, "
                    "skipping 3-proc spawn + engine reload\n")
                sys.stderr.flush()
                # First-wins flush_seek with the watcher. Read file
                # epoch + compare to last consumed in the singleton
                # dict; whichever path advances last_epoch claims
                # the flush. The outer singleton lock is held so we
                # can't re-acquire it (threading.Lock non-reentrant);
                # GIL-atomic dict access is enough.
                if _cached_qmgr is not None:
                    try:
                        _do_flush = False
                        try:
                            with open("/tmp/dual_machine_seek_epoch", "r") as _f:
                                _cur_file_epoch = int(
                                    _f.read().strip() or "0")
                        except Exception:
                            _cur_file_epoch = -1
                        _last = _st.get("seek_watcher_last_epoch", -1)
                        if _cur_file_epoch >= 0 and _last < _cur_file_epoch:
                            _st["seek_watcher_last_epoch"] = _cur_file_epoch
                            _do_flush = True
                        elif _cur_file_epoch < 0:
                            _do_flush = True   # no file = always flush
                        if _do_flush:
                            _cached_qmgr.flush_seek()
                    except Exception:
                        pass
                # No-op cleanup: don't tear down the singleton on
                # rife.vpy exit. mpv-process exit handles the real
                # OS-level cleanup of child subprocesses + shm.
                return _cached_compute, (lambda: None)
            # Params changed (resolution / mult / variant): tear down
            # old before initialising new.
            if _st.get("host_mp") is not None:
                sys.stderr.write(
                    f"[native_dispatcher] singleton INVALIDATE — params "
                    f"changed {_st.get('params_key')} → {_params_key}\n")
                sys.stderr.flush()
                for _key in ("queue_mgr", "host_mp"):
                    _obj = _st.get(_key)
                    if _obj is not None:
                        try: _obj.shutdown()
                        except Exception: pass
                _st.clear()

    dev = torch.device(device)
    torch.cuda.set_device(dev)

    # Match worker_3proc.py's cuDNN setting so single-proc and 3-proc
    # pick the same conv algorithm (the original split-dual prototype also relies on this).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    import vs_gpu_helpers as vgh
    # Handshake encoding constants — keep in sync with worker.py's
    # VARIANTS / CHROMA_MODES / _MATRIX_BY_IDX.
    _HS_VARIANTS = [
        "FSRCNNX_x2_8-0-4-1",
        "FSRCNNX_x2_16-0-4-1",
        "FSRCNNX_x3_16-0-4-1",
        "FSRCNNX_x4_16-0-4-1",
    ]
    _HS_CHROMA_MODES = ["bilinear", "bicubic", "nearest"]
    # Resolve fsrcnnx-cudnn from the same bundle location sr_keys_helper
    # uses (`~/.config/mpv/fsrcnnx-cudnn/`) so the package source and weights
    # come from the install.sh-managed release bundle, not the dev tree.
    _mpv_home = Path(
        os.environ.get("MPV_HOME") or
        (os.environ.get("XDG_CONFIG_HOME") or
         os.path.expanduser("~/.config")) + "/mpv"
    )
    _fsrcnnx_bundle = _mpv_home / "fsrcnnx-cudnn"
    if _fsrcnnx_bundle.exists() and str(_fsrcnnx_bundle) not in sys.path:
        sys.path.insert(0, str(_fsrcnnx_bundle))
    from fsrcnnx_cudnn.model.cudnn_runner import FSRCNNXcuDNNRunner  # noqa

    # ── Load RIFE engine config first — handshake needs pH/pW ──────
    # _load_engines is cheap when the engine is already cached on disk
    # (just deserialises the metadata); FSRCNNX runner construction is
    # what dominates pre-warm wall time and stays below.
    rife_cfg = vgh._load_engines(rife_model, proc_h, proc_w, 1.0, True, dev)

    # ── Liveness pre-flight ─────────────────────────────────────────
    # Probe the worker BEFORE starting NCCL rendezvous. If the worker
    # isn't serving (port not listening / different host), we want a
    # < 1 s fallback to single-machine, not a 5 s NCCL timeout. This
    # also opens the keep-alive socket the worker uses to detect host
    # crash within ~3 s.
    import socket as _sock
    worker_ip = os.environ.get("DUAL_WORKER_HOST", "127.0.0.1")
    _liveness_port = int(os.environ.get("DUAL_LIVENESS_PORT", "29905"))
    _liveness_sock = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
    _liveness_sock.settimeout(0.5)
    try:
        _liveness_sock.connect((worker_ip, _liveness_port))
    except (OSError, _sock.timeout) as _le:
        _liveness_sock.close()
        raise RuntimeError(
            f"worker liveness probe to {worker_ip}:{_liveness_port} "
            f"failed in <0.5 s ({type(_le).__name__}: {_le}) — worker "
            f"not serving") from _le
    _liveness_sock.setsockopt(_sock.SOL_SOCKET, _sock.SO_KEEPALIVE, 1)
    try:
        _liveness_sock.setsockopt(_sock.IPPROTO_TCP, _sock.TCP_KEEPIDLE, 1)
        _liveness_sock.setsockopt(_sock.IPPROTO_TCP, _sock.TCP_KEEPINTVL, 1)
        _liveness_sock.setsockopt(_sock.IPPROTO_TCP, _sock.TCP_KEEPCNT, 2)
    except OSError:
        pass
    _liveness_sock.settimeout(None)
    sys.stderr.write(
        f"[native_dispatcher] liveness socket to "
        f"{worker_ip}:{_liveness_port} open\n")
    sys.stderr.flush()

    # ── Handshake over the liveness TCP socket ─────────────────────
    # No NCCL: we send the 18-int64 handshake directly on the already-
    # open liveness socket. Mirror the env vars worker_3proc reads;
    # these get inherited by subprocesses spawned downstream.
    if interp_mult not in (1, 2, 3, 4):
        interp_mult = 2
    os.environ["DUAL_INTERP_MULT"] = str(interp_mult)
    os.environ["DUAL_NO_SR"] = "1" if no_sr else "0"
    import struct as _struct
    _VARIANT_IDX = {v: i for i, v in enumerate(_HS_VARIANTS)}
    _CHROMA_IDX = {m: i for i, m in enumerate(_HS_CHROMA_MODES)}
    _MATRIX_PROP = {"709": 1, "470bg": 5, "170m": 6, "240m": 7,
                    "2020ncl": 9, "2020cl": 10}
    _RIFE_IDX = {"4.26": 0, "4.6": 1}
    handshake_bytes = _struct.pack(
        "<18q",
        2, H, W, output_scale,
        _VARIANT_IDX[variant],
        _CHROMA_IDX[chroma_kernel],
        src_sub_w, src_sub_h,
        _MATRIX_PROP.get(matrix_s, 1),
        0 if color_range == "full" else 1,
        _RIFE_IDX.get(rife_model, 0),
        downsample_pre,
        dst_sub_w, dst_sub_h,
        int(rife_cfg["ph"]), int(rife_cfg["pw"]),
        interp_mult,
        1 if no_sr else 0,
    )
    sys.stderr.write("[native_dispatcher] sending handshake (144 B)…\n")
    sys.stderr.flush()
    try:
        _liveness_sock.sendall(handshake_bytes)
    except OSError as _he:
        raise RuntimeError(
            f"handshake send failed ({type(_he).__name__}: {_he})") from _he

    # ── Pre-warm engines ──────────────────────────────────────────
    # Host compute always runs in the 3-proc HostMP subprocess
    # (dma_proc + buffer_mgr_proc + compute_proc). RIFE + FSRCNNX
    # operate at proc dims (= H/downsample_pre) — pre-warm there.
    sys.stderr.write(
        f"[native_dispatcher] pre-warming host engines… "
        f"src={W}x{H} proc={proc_w}x{proc_h} out={oW}x{oH} "
        f"variant=x{fsrcnnx_scale} downsample_pre={downsample_pre}\n")
    sys.stderr.flush()
    _t0 = _time.perf_counter()
    # _mpv_home already computed above; reuse for the weights lookup.
    wpath = _mpv_home / "fsrcnnx-cudnn" / "weights" / f"{variant}.npz"
    # The dispatcher's `local_runner` is only used for this prewarm — the
    # actual compute-proc FSRCNNX runner is built independently inside
    # HostMP's subprocess. Skip both when no_sr (the compute proc reads
    # DUAL_NO_SR via env and skips its own runner too).
    if not no_sr:
        local_runner = FSRCNNXcuDNNRunner(
            wpath, variant, H=proc_h, W=proc_w, device=dev)
        with torch.inference_mode():
            z = torch.zeros(1, 1, proc_h, proc_w, dtype=torch.float32, device=dev)
            local_runner.forward(z).clamp_(0, 1)
            torch.cuda.synchronize(device=dev)
    # Compile the krig CUDA extension up-front. On a cold cache this is
    # ~18 s (one-shot nvcc + ninja); on a warm cache it's a no-op.
    try:
        from fsrcnnx_cudnn.chroma_krig import precompile as _krig_precompile
        _krig_precompile()
    except Exception as _ke:
        sys.stderr.write(
            f"[native_dispatcher] krig precompile failed: "
            f"{type(_ke).__name__}: {_ke}\n")
        sys.stderr.flush()
    sys.stderr.write(
        f"[native_dispatcher] pre-warm done in "
        f"{(_time.perf_counter()-_t0)*1000:.0f}ms; waiting for worker…\n")
    sys.stderr.flush()
    # Wait for worker's 'R' byte on the liveness socket. Worker emits
    # this after compute_proc has loaded its engines (or compiled TRT
    # from scratch on cache miss — minutes). The socket is already
    # keepalive-armed (≈3 s detection of ungraceful disconnect), so a
    # worker crash mid-wait still trips cleanly.
    _ready_timeout = float(os.environ.get("DUAL_WORKER_READY_TIMEOUT", "300"))
    _liveness_sock.settimeout(_ready_timeout)
    try:
        _r = _liveness_sock.recv(1)
    except (OSError, _sock.timeout) as _re:
        raise RuntimeError(
            f"worker ready signal not received in {_ready_timeout}s "
            f"({type(_re).__name__}: {_re})") from _re
    finally:
        _liveness_sock.settimeout(None)
    if not _r:
        raise RuntimeError("worker closed liveness before signaling ready")
    if _r != b"R":
        raise RuntimeError(f"unexpected worker ready byte: {_r!r}")
    sys.stderr.write("[native_dispatcher] worker ready\n"); sys.stderr.flush()

    # Transport config — read once, used by guest_mp's RDMAChannel below.
    # worker_ip already pulled up top for the liveness pre-flight.
    rdma_dev = os.environ.get("DUAL_RDMA_DEV", "rocep1s0f0")
    rdma_port = int(os.environ.get("DUAL_RDMA_PORT", "29900"))
    rdma_gid = int(os.environ.get("DUAL_RDMA_GID", "3"))

    # Liveness socket to worker. Holds open for the lifetime of the
    # session; on host crash / clean exit, the kernel sends FIN to the
    # worker's listening watchdog, which trips its shutdown path. We
    # connect AFTER the worker signaled "ready" (worker spawns its
    # listener in serve_session before signal_ready_cb fires). Failure
    # here doesn't crash the session — we fall through and rely on the
    # NCCL session timeout (10 min) for catastrophic detection.
    # Liveness probe + keepalive already done up top before NCCL init.

    # ── Host 3-proc ──────────────────────────────────────────────
    # Three host children: dma_proc (process_vm_readv from mpv VA
    # into shm), buffer_mgr, compute_proc (RIFE+SR — reuses
    # worker_3proc's body). The dispatcher's compute() hook submits
    # pairs to HostMP via the queue_mgr.
    host_mp = None
    # 2 slots (tick-tac) hide the dma_proc → compute_proc handoff
    # at the SPLIT-task fine-grained pace.
    host_mp_slots = 2
    if True:
        # process_vm_readv from dma_proc (child) into mpv's address
        # space (parent) requires the tracee (mpv) to authorize tracing
        # under yama mode 1 (default on Ubuntu). PR_SET_PTRACER_ANY
        # opens us to ptrace from any process at the same uid — fine
        # for the dma_proc use case; we only fork same-uid children.
        # Without this, process_vm_readv returns EPERM.
        try:
            _PR_SET_PTRACER = 0x59616d61  # 0x59616d61 'Yama' magic
            _PR_SET_PTRACER_ANY = ctypes.c_ulong(-1).value
            _libc_pd = ctypes.CDLL("libc.so.6", use_errno=True)
            rc = _libc_pd.prctl(_PR_SET_PTRACER, _PR_SET_PTRACER_ANY,
                                  0, 0, 0)
            if rc != 0:
                err = ctypes.get_errno()
                sys.stderr.write(
                    f"[native_dispatcher] prctl(PR_SET_PTRACER, ANY) "
                    f"rc={rc} errno={err} — process_vm_readv may fail "
                    f"with EPERM\n")
        except Exception as ex:
            sys.stderr.write(
                f"[native_dispatcher] prctl call failed: {ex}\n")
        from host_3proc import HostSlotRingLayout, HostMP
        h_layout = HostSlotRingLayout(
            n_slots=host_mp_slots, H=H, W=W, scale=output_scale,
            sub_w=src_sub_w, sub_h=src_sub_h,
            dst_sub_w=dst_sub_w, dst_sub_h=dst_sub_h,
            pH=int(rife_cfg["ph"]),
            pW=int(rife_cfg["pw"]),
            enc_ch=int(rife_cfg.get("encode_channel", 0) or 0))
        # cc_cache region: SPLIT-task pipeline always allocates one so
        # CCSR / INTERP outputs can park between stages.
        cc_cache_layout = None
        if True:
            from cc_cache import CCCacheLayout
            # cc_cache slots must accommodate ALL in-flight pair work:
            # 2 CC outputs per pair × ~max-pair-in-flight + a few
            # INTERP-output slots. With concurrent-frames=8 in mpv we
            # can have up to 8 pairs simultaneously, each using 2 CC
            # slots → 16 slots minimum, plus INTERP output spillage.
            # Generously sized — host has ~128 GB LPDDR5x; ~75 MB/slot
            # × 32 ≈ 2.4 GB is comfortable.
            cc_cache_layout = CCCacheLayout(
                n_slots=32,
                H_src=H, W_src=W, H_out=oH, W_out=oW,
                pH=int(rife_cfg["ph"]),
                pW=int(rife_cfg["pw"]),
                enc_channels=int(rife_cfg.get("encode_channel", 0) or 0),
                sub_h=src_sub_h, sub_w=src_sub_w,
            )
        host_mp = HostMP(
            layout=h_layout,
            shm_name=f"dgxspark_host_3p_{os.getpid()}",
            cc_cache_layout=cc_cache_layout,
            cc_cache_name=(f"dgxspark_host_cc_cache_{os.getpid()}"
                            if cc_cache_layout else None),
            variant=variant, rife_model=rife_model,
            matrix_s=matrix_s, color_range=color_range,
            chroma_mode=chroma_kernel, bits=10,
            downsample_pre=downsample_pre,
            weights_dir=str(_mpv_home / "fsrcnnx-cudnn" / "weights"),
            log=lambda m: sys.stderr.write(m + "\n"),
        )
        host_mp.setup()
        host_mp.start()
        sys.stderr.write(
            f"[native_dispatcher] HostMP starting, waiting for "
            f"compute_proc ready…\n")
        sys.stderr.flush()
        host_mp.wait_compute_ready(
            timeout=float(os.environ.get(
                "DUAL_WORKER_READY_TIMEOUT", "300")))
        sys.stderr.write("[native_dispatcher] HostMP compute_proc ready\n")
        sys.stderr.flush()

    # ── QueueManager + host dispatch thread ──────────────────────
    # The host dispatch loop pops tasks from the queue manager, claims
    # a host worker slot, fills RequestMeta, and notifies dma_proc.
    # When the slot finishes, _hmp_watcher (started further down)
    # memmoves slot.dst into mpv VA (for SR_*/CCSR) and calls
    # queue_mgr.task_done(). mpv's submit_with_phase_dst (hot path)
    # injects work into the queue manager from the frame_thread side.
    queue_mgr = None
    if host_mp is not None:
        from queue_mgr import (QueueManager, TaskType, FrameVA)
        queue_mgr = QueueManager(
            cc_cache=host_mp.cc_cache,
            log=lambda m: sys.stderr.write(m + "\n"),
        )

        # atexit fallback. mpv's embedded python doesn't reliably
        # finalize on exit, so the primary close path is the lua
        # `shutdown` hook → marker file → inotify watcher (see
        # _do_shutdown below). atexit only fires from `python -m
        # dual_machine` style entry, kept here so devs running the
        # module directly still get clean teardown.
        import atexit as _atexit_dispatcher
        _hmp_close = host_mp
        _qmgr_close = queue_mgr
        _guest_close_holder: list = [None]
        _inotify_close_holder: list = [None]
        _liveness_close_holder: list = [_liveness_sock]
        def _dispatcher_atexit():
            try: _qmgr_close.shutdown(drain_timeout=0.3)
            except Exception: pass
            try:
                if _guest_close_holder[0] is not None:
                    _guest_close_holder[0].close()
            except Exception: pass
            try: _hmp_close.shutdown()
            except Exception: pass
            try:
                if _liveness_close_holder[0] is not None:
                    _liveness_close_holder[0].close()
            except Exception: pass
            try:
                if _inotify_close_holder[0] is not None:
                    os.close(_inotify_close_holder[0])
            except Exception: pass
        _atexit_dispatcher.register(_dispatcher_atexit)

        # Forward-declared holder for guest_mp (assigned later, after
        # this loop is defined and the thread spawned). The loop reads
        # _guest_mp_holder[0] each iteration so it picks up the
        # assignment when it happens.
        _guest_mp_holder: list = [None]

        def _split_dispatch_loop():
            """Host dispatch loop. Just asks queue_mgr 'give me any
            task I can run' — mgr applies the cc>sr>interp priority
            internally. No fallback logic here, so a temporarily-empty
            cc_q never blocks us out of sr_q work, and a temporarily-
            empty sr_q never blocks us out of cc_q work. Mgr's
            pop_for_host is a single atomic transaction that wakes on
            ANY queue's notify and re-checks the priority order."""
            _prof_on = os.environ.get("DUAL_PROFILE", "0") == "1"
            _prof = {"cpu_ms": 0.0, "claim_retries": 0, "n": 0,
                     "last_report": _time.perf_counter()}
            while True:
                node = queue_mgr.pop_for_host(
                    blocking=True, timeout=0.5)
                if node is None:
                    if queue_mgr._closed:
                        return
                    continue
                _t_pop = _time.perf_counter() if _prof_on else 0.0
                # Strict gating: queue_mgr publishes a node only when
                # all per-task prerequisites are met (CC needs src_y,
                # SR_* needs FrameVA.dst_{a,i}_y). By construction the
                # popped node is publishable.
                sr_dst_y = sr_dst_u = sr_dst_v = 0
                sr_y_stride = sr_uv_stride = 0
                # CCSR is a producer for phase-0 (writes 4K SR to
                # dst_a_*) AND reads raw src YUV like CC. Strict gating
                # guarantees both src_y and dst_a_y are set before
                # publish (see queue_mgr._node_publishable_locked CCSR
                # branch).
                if node.type in (TaskType.SR_INTERP, TaskType.CCSR):
                    va = queue_mgr.get_frame_va(node.frame_idx)
                    if node.type == TaskType.CCSR:
                        sr_dst_y, sr_dst_u, sr_dst_v = (
                            va.dst_a_y, va.dst_a_u, va.dst_a_v)
                    elif node.output_phase == 3:
                        # mult=4 tertiary SR (frame 3 / t=3/4) → dst_i3
                        sr_dst_y, sr_dst_u, sr_dst_v = (
                            va.dst_i3_y, va.dst_i3_u, va.dst_i3_v)
                    elif node.output_phase == 2:
                        # mult ≥ 3 secondary SR (frame 2) → dst_i2
                        sr_dst_y, sr_dst_u, sr_dst_v = (
                            va.dst_i2_y, va.dst_i2_u, va.dst_i2_v)
                    else:
                        sr_dst_y, sr_dst_u, sr_dst_v = (
                            va.dst_i_y, va.dst_i_u, va.dst_i_v)
                    sr_y_stride = va.dst_y_stride
                    sr_uv_stride = va.dst_uv_stride
                    assert sr_dst_y != 0, (
                        f"[split-dispatch] gating violation: SR/CCSR popped "
                        f"without dst VA tid={node.id} K={node.frame_idx} "
                        f"type={node.type} phase={node.output_phase}")
                slot = None
                _claim_retries = 0
                while slot is None:
                    slot = host_mp.claim_slot()
                    if slot is None:
                        _claim_retries += 1
                        # cv-based wait: release_slot() (in
                        # _hmp_watcher) signals the cv; typical wakeup
                        # ~10us. Cap at 5ms as a safety net.
                        host_mp.wait_slot_free(timeout=0.005)
                queue_mgr.set_node_payload(node.id, "worker_slot", slot)
                if os.environ.get("DUAL_SPLIT_DEBUG", "0") == "1":
                    sys.stderr.write(
                        f"[split-dispatch] task_id={node.id} "
                        f"type={TaskType(node.type).name} "
                        f"frame_idx={node.frame_idx} → slot={slot} "
                        f"src_a={node.payload.get('src_cc_slot_a',-1)} "
                        f"src_b={node.payload.get('src_cc_slot_b',-1)} "
                        f"dst={node.payload.get('dst_cc_slot',-1)}\n")
                    sys.stderr.flush()
                req = host_mp.get_request(slot)
                req.pair_k    = node.frame_idx
                req.src_a_idx = node.frame_idx
                req.src_b_idx = node.frame_idx + 1
                req.mpv_pid   = os.getpid()
                req.task_type = int(node.type)
                req.task_id   = node.id
                # Reset everything else; per-task-type fills follow.
                req.sa_y = req.sa_u = req.sa_v = 0
                req.sb_y = req.sb_u = req.sb_v = 0
                req.dst_ya = req.dst_ua = req.dst_va = 0
                req.dst_yi = req.dst_ui = req.dst_vi = 0
                req.sa_y_stride = req.sa_uv_stride = 0
                req.dst_y_stride_a = req.dst_uv_stride_a = 0
                req.dst_y_stride_i = req.dst_uv_stride_i = 0
                req.phases_mask = 0
                req.src_cc_slot_a = node.payload.get("src_cc_slot_a", -1)
                req.src_cc_slot_b = node.payload.get("src_cc_slot_b", -1)
                req.src_cc_field  = node.payload.get("src_cc_field", -1)
                req.dst_cc_slot   = node.payload.get("dst_cc_slot", -1)
                req.dst_cc_field  = node.payload.get("dst_cc_field", -1)
                if node.type == TaskType.CCSR:
                    # mpv VA for this frame's YUV planes — set by
                    # submit_frame(K, va=...) when the mpv frame_thread
                    # first asked for pair K. dma_proc will readv from
                    # these into worker slot.src.
                    req.sa_y = node.payload.get("src_y", 0)
                    req.sa_u = node.payload.get("src_u", 0)
                    req.sa_v = node.payload.get("src_v", 0)
                    req.sa_y_stride = node.payload.get("src_y_stride", 0)
                    req.sa_uv_stride = node.payload.get("src_uv_stride", 0)
                # CC, CCSR, and INTERP set dst_cc_slot via _post_pop_locked
                # (their compute_proc kernels write directly into the
                # cc_cache GPU view at that slot). We've already
                # populated req.dst_cc_slot from node.payload above —
                # do NOT overwrite to -1; dma_proc's _writeback_one
                # will see dst_cc_slot >= 0 and currently does its
                # OWN memcpy slot.dst → cc_cache, which clobbers the
                # kernel's direct write. To prevent that, dispatcher
                # passes dst_cc_field = -2 (sentinel meaning "kernel
                # writes cc_cache itself, dma_proc just mark FREE").
                if node.type in (TaskType.CCSR, TaskType.INTERP):
                    # Keep req.dst_cc_slot as set from node.payload
                    # so the worker compute_proc can find its cc_cache
                    # destination slot.
                    req.dst_cc_field = -2  # "compute_proc wrote it"
                # SR_SRC / SR_INTERP wire output to mpv VA via
                # dma_proc's phases_mask=0x1 path. For mult ≥ 3, all
                # output_phase values (1 → dst_i, 2 → dst_i2, 3 → dst_i3)
                # share the same TT_SR_INTERP wire type; the sr_dst_*
                # values were already routed correctly above.
                elif node.type == TaskType.SR_INTERP:
                    req.dst_ya = sr_dst_y
                    req.dst_ua = sr_dst_u
                    req.dst_va = sr_dst_v
                    req.dst_y_stride_a = sr_y_stride
                    req.dst_uv_stride_a = sr_uv_stride
                    req.phases_mask = 0x1
                # INTERP carries dst_cc_slot_2/3 for additional output
                # frames (mult=3 sets _2; mult=4 sets _2 and _3;
                # -1 means unused).
                if node.type == TaskType.INTERP:
                    req.dst_cc_slot_2 = int(
                        node.payload.get("dst_cc_slot_2", -1))
                    req.dst_cc_slot_3 = int(
                        node.payload.get("dst_cc_slot_3", -1))
                # CCSR writes 4K to slot.dst.yao/uao/vao (like SR_SRC)
                # AND writes rgb_padded + features to cc_cache (like
                # CC). Both writeback paths active.
                if node.type == TaskType.CCSR:
                    req.dst_ya = sr_dst_y
                    req.dst_ua = sr_dst_u
                    req.dst_va = sr_dst_v
                    req.dst_y_stride_a = sr_y_stride
                    req.dst_uv_stride_a = sr_uv_stride
                    req.phases_mask = 0x1   # writeback yao/uao/vao
                    req.dst_cc_field = -2   # kernel wrote cc_cache itself
                del req
                # Publish: ST_FILLING → ST_RECV_PENDING. Has to happen
                # AFTER `del req` so the RequestMeta writes are flushed
                # before submit_listener can race-read this slot.
                host_mp.commit_slot(slot)
                host_mp.notify_submit()
                if _prof_on:
                    _prof["cpu_ms"] += (_time.perf_counter() - _t_pop) * 1000
                    _prof["claim_retries"] += _claim_retries
                    _prof["n"] += 1
                    if (_time.perf_counter() - _prof["last_report"]) >= 2.0:
                        n = _prof["n"]
                        sys.stderr.write(
                            f"[profile] HOST-DISP n={n} "
                            f"cpu_per_task={_prof['cpu_ms']/n:.2f}ms "
                            f"claim_retries={_prof['claim_retries']} "
                            f"(slot full = worker overloaded)\n")
                        sys.stderr.flush()
                        _prof = {"cpu_ms": 0.0, "claim_retries": 0,
                                  "n": 0, "last_report": _time.perf_counter()}

        threading.Thread(target=_split_dispatch_loop, daemon=True,
                          name="split-dispatch").start()
        sys.stderr.write("[native_dispatcher] split-task dispatch thread "
                         "started\n")
        sys.stderr.flush()

        # ── Seek-flush watcher ─────────────────────────────────────
        # Reacts to scripts/dual_seek_flush.lua bumping
        # /tmp/dual_machine_seek_epoch on every mpv seek. Calls
        # queue_mgr.flush_seek() to drop stale per-K state so the
        # post-seek pipeline starts fresh. First-wins coordinated
        # with the make_dispatcher REUSE path via the singleton's
        # `seek_watcher_last_epoch`.
        SEEK_EPOCH_FILE = "/tmp/dual_machine_seek_epoch"
        try:
            from _dispatcher_singleton import (
                get as _w_singleton_get,
                lock as _w_singleton_lock_fn,
            )
            _w_state = _w_singleton_get()
            _w_lock = _w_singleton_lock_fn()
        except Exception:
            _w_state = {}
            _w_lock = threading.Lock()
        try:
            with open(SEEK_EPOCH_FILE, "r") as f:
                _w_state.setdefault(
                    "seek_watcher_last_epoch",
                    int(f.read().strip() or "0"))
        except Exception:
            _w_state.setdefault("seek_watcher_last_epoch", 0)

        # inotify on the file's parent dir (kernel notifies on
        # IN_MODIFY | IN_CLOSE_WRITE | IN_MOVED_TO; ~µs latency vs
        # 100 ms polling). Falls back to time.sleep polling if
        # inotify setup fails.
        def _try_inotify_wait_for_change(epoch_file: str):
            try:
                import ctypes
                _libc = ctypes.CDLL("libc.so.6", use_errno=True)
                fd = _libc.inotify_init1(0x80000)  # IN_CLOEXEC
                if fd < 0:
                    return None
                _dir = os.path.dirname(epoch_file) or "/"
                wd = _libc.inotify_add_watch(
                    fd, _dir.encode(), 0x00000002 | 0x00000008 | 0x00000080)
                if wd < 0:
                    os.close(fd)
                    return None
                return fd
            except Exception:
                return None

        _inotify_fd = _try_inotify_wait_for_change(SEEK_EPOCH_FILE)
        # publish fd to atexit's holder so it can close on exit
        try:
            _inotify_close_holder[0] = _inotify_fd
        except Exception:
            pass

        def _check_and_flush_locked() -> bool:
            """Read file epoch, claim if > last_epoch, return True
            if a flush should fire (and actually fire it)."""
            try:
                with open(SEEK_EPOCH_FILE, "r") as f:
                    epoch = int(f.read().strip() or "0")
            except Exception:
                return False
            _do_flush = False
            with _w_lock:
                if epoch > _w_state.get(
                        "seek_watcher_last_epoch", 0):
                    _w_state["seek_watcher_last_epoch"] = epoch
                    _do_flush = True
            if _do_flush:
                stats = queue_mgr.flush_seek()
                sys.stderr.write(
                    f"[seek-flush] epoch={epoch} "
                    f"failed={stats['failed']} "
                    f"K_count={stats['K_count']} "
                    f"slots_freed={stats['slots_freed']}\n")
                sys.stderr.flush()
                return True
            return False

        # Close path. Lua's `shutdown` event hook writes
        # SHUTDOWN_FILE; this watcher reacts and runs the same
        # cleanup atexit would run, before the OS reaps our FDs.
        SHUTDOWN_FILE = "/tmp/dual_machine_shutdown"
        try: os.unlink(SHUTDOWN_FILE)  # clear any stale marker
        except FileNotFoundError: pass
        except Exception: pass
        _shutdown_fired = [False]
        def _do_shutdown():
            if _shutdown_fired[0]:
                return
            _shutdown_fired[0] = True
            sys.stderr.write("[dual-shutdown] starting fast-close\n")
            sys.stderr.flush()
            _t0 = _time.perf_counter()
            try: queue_mgr.shutdown(drain_timeout=0.3)
            except Exception: pass
            try:
                if _guest_close_holder[0] is not None:
                    _guest_close_holder[0].close()
            except Exception: pass
            try: host_mp.shutdown()
            except Exception: pass
            try:
                if _liveness_close_holder[0] is not None:
                    _liveness_close_holder[0].close()
            except Exception: pass
            sys.stderr.write(
                f"[dual-shutdown] complete in "
                f"{(_time.perf_counter()-_t0)*1000:.1f} ms\n")
            sys.stderr.flush()

        def _seek_flush_watcher():
            if _inotify_fd is not None:
                # read(2) blocks until any inotify event in /tmp;
                # drain the buffer, then check both files.
                _buf_sz = 4096
                while True:
                    try:
                        os.read(_inotify_fd, _buf_sz)
                    except Exception:
                        _time.sleep(0.05)
                    _check_and_flush_locked()
                    try:
                        if os.path.exists(SHUTDOWN_FILE):
                            _do_shutdown()
                            try: os.unlink(SHUTDOWN_FILE)
                            except Exception: pass
                            return
                    except Exception:
                        pass
            else:
                # Polling fallback if inotify isn't available.
                while True:
                    _time.sleep(0.05)
                    _check_and_flush_locked()
                    try:
                        if os.path.exists(SHUTDOWN_FILE):
                            _do_shutdown()
                            try: os.unlink(SHUTDOWN_FILE)
                            except Exception: pass
                            return
                    except Exception:
                        pass
        threading.Thread(target=_seek_flush_watcher, daemon=True,
                          name="seek-flush-watcher").start()

    # ── GuestMP — RDMA wrapper for routing INTERP / SR_INTERP /
    #    SR_SRC tasks to the remote guest worker. A second dispatch
    #    loop pulls from the queue manager in parallel with the host
    #    loop; host claims CCSR, guest claims INTERP (with SR_INTERP
    #    as fallback). Both sides also accept the other's task type
    #    if the priority order is empty on their side.
    guest_mp = None
    if host_mp is not None:
        from guest_mp import GuestMP
        from cc_cache import FIELD_RGB_INTERP

        _guest_port = rdma_port
        # Must match the worker's RDMA n_slots (4 — see worker.py).
        _guest_slots = 4

        # cc_cache writeback callback:
        #   INTERP payload : ("rgb_interp", rgb_addr, rgb_sz)
        #     → memmoves rgb_interp into cc_cache[dst_slot].rgb_interp
        #   CCSR payload   : ("ccsr", rgb_addr, rgb_sz, feat_addr, feat_sz)
        #     → memmoves rgb_padded + rife_features into cc_cache[dst_slot]
        _cc = host_mp.cc_cache
        from cc_cache import (FIELD_RGB_PADDED, FIELD_RIFE_FEATURES,
                              FIELD_RGB_INTERP)
        def _cc_writeback(dst_slot: int, payload) -> None:
            if len(payload) == 3 and payload[0] == "rgb_interp":
                _, rgb_addr, rgb_sz = payload
                rgb_dst = _cc.field_addr(dst_slot, FIELD_RGB_INTERP)
                ctypes.memmove(rgb_dst, rgb_addr, rgb_sz)
                _cc.set_field(dst_slot, FIELD_RGB_INTERP)
            elif len(payload) == 5 and payload[0] == "ccsr":
                _, rgb_addr, rgb_sz, feat_addr, feat_sz = payload
                rgb_dst = _cc.field_addr(dst_slot, FIELD_RGB_PADDED)
                feat_dst = _cc.field_addr(dst_slot, FIELD_RIFE_FEATURES)
                ctypes.memmove(rgb_dst,  rgb_addr,  rgb_sz)
                ctypes.memmove(feat_dst, feat_addr, feat_sz)
                _cc.set_field(dst_slot, FIELD_RGB_PADDED)
                _cc.set_field(dst_slot, FIELD_RIFE_FEATURES)
            else:
                sys.stderr.write(
                    f"[cc_writeback] unknown payload shape "
                    f"len={len(payload)} head={payload[0] if payload else None}\n")

        # task_done callback: queue_mgr is the source of truth. We
        # tag result with executor="guest" so _compute_split_task can
        # recognise that the SR_* output is already in mpv VA (not in
        # host_mp.shm.slot_dst) and skip the host-side memmove.
        def _guest_task_done(task_id: int) -> None:
            if os.environ.get("DUAL_REFCOUNT_DBG", "0") == "1":
                sys.stderr.write(
                    f"[ref-caller] guest_mp-cq task_done({task_id})\n")
                sys.stderr.flush()
            queue_mgr.task_done(task_id, ok=True,
                                 result={"executor": "guest"})

        # Mid SEND arrival → unblock downstream INTERPs gated on
        # CCSR stage 0. Stage idx encoded by guest_mp's CQ thread.
        def _guest_task_stage_done(task_id: int, stage_idx: int) -> None:
            if os.environ.get("DUAL_REFCOUNT_DBG", "0") == "1":
                sys.stderr.write(
                    f"[ref-caller] guest_mp-cq task_stage_done("
                    f"{task_id}, {stage_idx})\n")
                sys.stderr.flush()
            queue_mgr.task_stage_done(task_id, stage_idx)

        # GuestMP reads DUAL_DOWNSAMPLE_PRE env to derive proc dims for
        # the INTERP rgb_interp ferry size. Make sure it sees the same
        # value we passed in handshake.
        os.environ["DUAL_DOWNSAMPLE_PRE"] = str(downsample_pre)
        # GuestMP / _pack_interp_payload compute wire offsets from
        # WMP_SPLIT_PH/PW (= RIFE-padded source dims). Default 1088/1920
        # is for 1080p; on 720p sources the actual rife_cfg pH/pW is
        # 768/1280. Without this override host's pack offsets diverge
        # from the worker's split_src_views layout (worker's WMP_SPLIT_PH
        # comes from layout.pH propagated via _child_env, which IS
        # correct), causing worker to read rgb_padded_b out of the
        # middle of features_a — silent 33-9 dB PSNR drop on INTERP.
        os.environ["WMP_SPLIT_PH"] = str(int(rife_cfg["ph"]))
        os.environ["WMP_SPLIT_PW"] = str(int(rife_cfg["pw"]))
        try:
            guest_mp = GuestMP(
                H=H, W=W, scale=output_scale,
                sub_w=src_sub_w, sub_h=src_sub_h,
                dst_sub_w=dst_sub_w, dst_sub_h=dst_sub_h,
                pH=int(rife_cfg["ph"]),
                pW=int(rife_cfg["pw"]),
                enc_ch=int(rife_cfg.get("encode_channel", 0) or 0),
                n_slots=_guest_slots,
                rdma_dev=rdma_dev,
                rdma_port=1, rdma_gid_index=rdma_gid,
                peer_ip=worker_ip,
                peer_handshake_port=_guest_port,
                cc_cache_writeback_fn=_cc_writeback,
                task_done_fn=_guest_task_done,
                task_stage_done_fn=_guest_task_stage_done,
                log=lambda m: sys.stderr.write(m + "\n"),
            )
            sys.stderr.write(
                f"[native_dispatcher] GuestMP up: slots={_guest_slots} "
                f"peer={worker_ip}:{_guest_port}\n")
            sys.stderr.flush()
            # Publish to the holder so _split_dispatch_loop (already
            # running) starts gating its compute-q steal on guest's
            # slot availability.
            _guest_mp_holder[0] = guest_mp
            # publish to atexit's holder so it can close on exit
            try:
                _guest_close_holder[0] = guest_mp
            except Exception:
                pass
        except Exception as ex:
            sys.stderr.write(
                f"[native_dispatcher] GuestMP setup failed "
                f"({type(ex).__name__}: {ex}); falling back to "
                f"host_mp-only split-task path\n")
            sys.stderr.flush()
            guest_mp = None

        # Second dispatch loop: pulls non-CC tasks from compute_q,
        # packs cc_cache field data into the guest_mp send buffer,
        # and posts via notify_submit. Runs in parallel with the
        # host dispatch loop above; queue_mgr's _cv-locked pop_compute
        # is multi-consumer safe (each task is delivered to exactly
        # one loop). See SPLIT_TASK_ARCHITECTURE.md §13.
        if guest_mp is not None:
            from cc_cache import FIELD_RGB_PADDED, FIELD_RIFE_FEATURES

            def _pack_interp_payload(cc, node, base, poff, gmp):
                """Wire layout (must match worker_3proc.split_src_views):
                  rgb_padded_a, features_a, rgb_padded_b, features_b
                each contiguous, in slot.src after the 128B header."""
                src_a = node.payload["src_cc_slot_a"]
                src_b = node.payload["src_cc_slot_b"]
                rgb_sz = gmp._rgb_padded_bytes
                ft_sz  = gmp._rife_features_bytes
                cur = poff
                for cc_slot in (src_a, src_b):
                    ctypes.memmove(
                        base + cur,
                        cc.field_addr(cc_slot, FIELD_RGB_PADDED),
                        rgb_sz)
                    cur += rgb_sz
                    ctypes.memmove(
                        base + cur,
                        cc.field_addr(cc_slot, FIELD_RIFE_FEATURES),
                        ft_sz)
                    cur += ft_sz

            def _pack_sr_payload(cc, node, base, poff, gmp):
                """Wire layout: rgb_interp (fp16 RGB at proc dims).
                cc_cache.rgb_interp is already laid out as (1, 3, proc_h,
                proc_w) — single contiguous memmove is the whole payload.
                """
                src = node.payload["src_cc_slot_a"]
                total = gmp._interp_dst_rgb_bytes
                ctypes.memmove(base + poff,
                                cc.field_addr(src, FIELD_RGB_INTERP),
                                total)

            # Pack CCSR src payload (1080p YUV) from mpv frame VA into
            # slot.src's ya/ua/va region. Mirrors host dma_proc's
            # behavior but happens in mpv process (UMA shm means a
            # simple memmove handles it). Honor mpv's strides if
            # non-packed (vapoursynth row pad).
            _ccsr_src_offs = {name: off for name, off, _, _, _ in
                               guest_mp.src_layout}
            _ccsr_H, _ccsr_W = guest_mp.H, guest_mp.W
            _ccsr_cH, _ccsr_cW = guest_mp.cH, guest_mp.cW
            def _pack_ccsr_payload(node, base, gmp):
                src_y_va = node.payload.get("src_y", 0)
                src_u_va = node.payload.get("src_u", 0)
                src_v_va = node.payload.get("src_v", 0)
                if not (src_y_va and src_u_va and src_v_va):
                    raise RuntimeError(
                        f"CCSR pack: missing mpv VA "
                        f"(y={src_y_va:#x} u={src_u_va:#x} v={src_v_va:#x})")
                src_y_stride = int(node.payload.get(
                    "src_y_stride", _ccsr_W * 2))
                src_uv_stride = int(node.payload.get(
                    "src_uv_stride", _ccsr_cW * 2))
                y_row = _ccsr_W * 2
                uv_row = _ccsr_cW * 2
                def _copy_plane(local_off, va, rows, row_bytes, stride):
                    if stride == row_bytes:
                        ctypes.memmove(base + local_off, va,
                                        row_bytes * rows)
                    else:
                        for r in range(rows):
                            ctypes.memmove(base + local_off + r * row_bytes,
                                            va + r * stride, row_bytes)
                _copy_plane(_ccsr_src_offs["ya"], src_y_va,
                             _ccsr_H, y_row, src_y_stride)
                _copy_plane(_ccsr_src_offs["ua"], src_u_va,
                             _ccsr_cH, uv_row, src_uv_stride)
                _copy_plane(_ccsr_src_offs["va"], src_v_va,
                             _ccsr_cH, uv_row, src_uv_stride)

            def _guest_dispatch_loop():
                """Guest dispatch loop. Asks mgr for any task it can
                run (CCSR / INTERP / SR_INTERP); priority is set by
                _GUEST_PRIORITY in queue_mgr. Atomic priority
                resolution means a temporarily-empty primary queue
                doesn't lock us into waiting — we wake on any
                submit_frame / task_done notify and re-check all
                eligible queues."""
                _prof_on = os.environ.get("DUAL_PROFILE", "0") == "1"
                _prof = {"cpu_ms": 0.0, "claim_retries": 0,
                          "dst_va_waits": 0, "n": 0,
                          "last_report": _time.perf_counter()}
                while True:
                    node = queue_mgr.pop_for_guest(
                        blocking=True, timeout=0.5)
                    if node is None:
                        if queue_mgr._closed:
                            return
                        continue
                    _t_pop = _time.perf_counter() if _prof_on else 0.0

                    # Strict gating: SR_* + CCSR tasks only enter the
                    # queue after FrameVA.dst_{a,i}_y is set. By
                    # construction va is set when we pop.
                    sr_dst_y = sr_dst_u = sr_dst_v = 0
                    sr_y_stride = sr_uv_stride = 0
                    if node.type in (TaskType.SR_INTERP, TaskType.CCSR):
                        va = queue_mgr.get_frame_va(node.frame_idx)
                        if node.type == TaskType.CCSR:
                            sr_dst_y, sr_dst_u, sr_dst_v = (
                                va.dst_a_y, va.dst_a_u, va.dst_a_v)
                        elif node.output_phase == 3:
                            sr_dst_y, sr_dst_u, sr_dst_v = (
                                va.dst_i3_y, va.dst_i3_u, va.dst_i3_v)
                        elif node.output_phase == 2:
                            sr_dst_y, sr_dst_u, sr_dst_v = (
                                va.dst_i2_y, va.dst_i2_u, va.dst_i2_v)
                        else:
                            sr_dst_y, sr_dst_u, sr_dst_v = (
                                va.dst_i_y, va.dst_i_u, va.dst_i_v)
                        sr_y_stride = va.dst_y_stride
                        sr_uv_stride = va.dst_uv_stride
                        assert sr_dst_y != 0, (
                            f"[guest-dispatch] gating violation: SR/CCSR "
                            f"popped without dst VA tid={node.id} "
                            f"K={node.frame_idx} type={node.type} "
                            f"phase={node.output_phase}")

                    slot = None
                    _claim_retries = 0
                    while slot is None:
                        # cv-based blocking claim. guest_mp.release_slot
                        # already signals its internal cv. Cap timeout
                        # at 5ms as safety net.
                        slot = guest_mp.claim_slot(
                            blocking=True, timeout=0.005)
                        if slot is None:
                            _claim_retries += 1

                    base = guest_mp.get_send_buffer_addr(slot)
                    poff = guest_mp.get_send_payload_offset()
                    try:
                        if node.type == TaskType.INTERP:
                            _pack_interp_payload(_cc, node, base, poff,
                                                  guest_mp)
                            guest_mp.fill_request(slot, node)
                        elif node.type == TaskType.CCSR:
                            _pack_ccsr_payload(node, base, guest_mp)
                            guest_mp.fill_request(
                                slot, node,
                                dst_y_va=sr_dst_y,
                                dst_u_va=sr_dst_u,
                                dst_v_va=sr_dst_v,
                                dst_y_stride=sr_y_stride,
                                dst_uv_stride=sr_uv_stride)
                        else:
                            _pack_sr_payload(_cc, node, base, poff,
                                              guest_mp)
                            guest_mp.fill_request(
                                slot, node,
                                dst_y_va=sr_dst_y,
                                dst_u_va=sr_dst_u,
                                dst_v_va=sr_dst_v,
                                dst_y_stride=sr_y_stride,
                                dst_uv_stride=sr_uv_stride)
                    except Exception as ex:
                        sys.stderr.write(
                            f"[guest-dispatch] pack failed task_id={node.id}"
                            f" type={node.type}: {ex}\n")
                        guest_mp.release_slot(slot)
                        queue_mgr.task_done(node.id, ok=False,
                                             error=f"pack: {ex}")
                        continue

                    if os.environ.get("DUAL_SPLIT_DEBUG", "0") == "1":
                        sys.stderr.write(
                            f"[guest-dispatch] task_id={node.id} "
                            f"type={TaskType(node.type).name} "
                            f"frame_idx={node.frame_idx} → slot={slot}\n")
                        sys.stderr.flush()
                    guest_mp.notify_submit(slot)
                    if _prof_on:
                        _prof["cpu_ms"] += (_time.perf_counter() - _t_pop) * 1000
                        _prof["claim_retries"] += _claim_retries
                        _prof["n"] += 1
                        if (_time.perf_counter() - _prof["last_report"]) >= 2.0:
                            n = _prof["n"]
                            sys.stderr.write(
                                f"[profile] GUEST-DISP n={n} "
                                f"cpu_per_task={_prof['cpu_ms']/n:.2f}ms "
                                f"claim_retries={_prof['claim_retries']} "
                                f"dst_va_waits={_prof['dst_va_waits']}\n")
                            sys.stderr.flush()
                            _prof = {"cpu_ms": 0.0, "claim_retries": 0,
                                      "dst_va_waits": 0, "n": 0,
                                      "last_report": _time.perf_counter()}

            # DUAL_FORCE_HOST=1 disables guest-dispatch so all tasks
            # run on host_mp. Useful for isolating single-machine PSNR
            # behavior from cross-machine races.
            if os.environ.get("DUAL_FORCE_HOST", "0") == "1":
                sys.stderr.write("[native_dispatcher] DUAL_FORCE_HOST=1 — "
                                 "guest-dispatch NOT started (host-only "
                                 "tasks run via host_mp)\n")
                sys.stderr.flush()
            else:
                threading.Thread(target=_guest_dispatch_loop, daemon=True,
                                  name="guest-dispatch").start()
                sys.stderr.write("[native_dispatcher] guest-dispatch thread "
                                 "started\n")
                sys.stderr.flush()

    # ── State ─────────────────────────────────────────────────────
    interp_kw = dict(size=(ocH, ocW), mode=chroma_kernel)
    if chroma_kernel != "nearest":
        interp_kw["align_corners"] = False

    # ── HostMP completion watcher ─────────────────────────────────
    # Reacts to ef_done (dma_proc signals a slot reached SEND_PENDING
    # or FREE). For terminal SR_*/CCSR tasks the kernel result lives
    # in slot.dst — we memmove it into the mpv VA before calling
    # queue_mgr.task_done so wait_frame_done unblocks AFTER the mpv
    # destination frame is fully populated. CC / INTERP output stays
    # in cc_cache; task_done + release_slot only.
    if host_mp is not None:
        from host_3proc import (ST_FREE, ST_SEND_PENDING, ST_MID_READY,
                                 is_mid_ready_state, mid_ready_stage)

        def _hmp_watcher():
            ef_done_fd = host_mp.ef_done_fd
            from worker_3proc import (TT_SR_INTERP, TT_CCSR)
            while True:
                os.eventfd_read(ef_done_fd)
                for s in range(host_mp_slots):
                    state_val = host_mp.state(s)
                    # Same-machine MID_READY → fire
                    # task_stage_done(tid, stage_idx). stage_idx ranges
                    # 0..mult-3 (or 0 for CCSR / INTERP mult=3).
                    # task_stage_done's stages_fired set makes repeat
                    # calls idempotent in case _hmp_watcher wakes again
                    # before compute_proc advances. Do NOT reset
                    # task_id (still needed for full task_done) and do
                    # NOT release slot.
                    if is_mid_ready_state(state_val):
                        # stage_idx encoded in state value itself
                        # (atomic uint32 read, no race).
                        meta = host_mp.shm.slot_state(s)
                        task_id_mid = int(meta.task_id)
                        del meta
                        stage_idx_mid = mid_ready_stage(state_val)
                        if task_id_mid > 0:
                            queue_mgr.task_stage_done(
                                task_id_mid, stage_idx_mid)
                        continue
                    # Only post-writeback states (SEND_PENDING / FREE)
                    # — ST_DST_READY means kernel done but mpv VA
                    # writeback still pending. Firing task_done at
                    # DST_READY would unblock wait_frame_done before
                    # the mpv frame is populated (mpv then recycles the
                    # buffer mid-writev → heap corruption).
                    if state_val not in (ST_SEND_PENDING, ST_FREE):
                        continue
                    meta = host_mp.shm.slot_state(s)
                    task_id = int(meta.task_id)
                    tt = int(meta.task_type)
                    # Reset task_id BEFORE task_done so a subsequent
                    # ef_done wake doesn't re-process this slot.
                    meta.task_id = 0
                    del meta
                    if task_id <= 0:
                        continue
                    # In-mpv-process memmove from shm slot.dst to mpv
                    # VA runs at ~30 GB/s (same-process on shared shm).
                    # Must happen BEFORE task_done so wait_frame_done
                    # returns to mpv only after the dst frame is fully
                    # populated.
                    # All SR_INTERP output_phase values use TT_SR_INTERP
                    # on the wire — dispatcher already routed req.dst_ya
                    # to dst_i_y / dst_i2_y / dst_i3_y based on phase
                    # before sending.
                    # CLAIMED-task writeback after a seek paints
                    # one frame of wrong-K content; mpv's vo
                    # overwrites it within a couple of vsyncs as
                    # the post-seek chain ramps up. Don't gate the
                    # memmove on an "abandoned" flag — that stalls
                    # dispatch (mpv frame pacing treats an
                    # unwritten dst as still-pending).
                    if (tt in (TT_SR_INTERP, TT_CCSR)
                            and state_val == ST_SEND_PENDING):
                        try:
                            host_mp.memmove_dst_to_mpv(s)
                        except Exception as ex:
                            sys.stderr.write(
                                f"[hmp_watcher] memmove slot={s} "
                                f"task={task_id} failed: "
                                f"{type(ex).__name__}: {ex}\n")
                            sys.stderr.flush()
                    queue_mgr.task_done(task_id)
                    host_mp.release_slot(s)
        threading.Thread(target=_hmp_watcher, daemon=True,
                          name="native-hmp-watcher").start()

    _fps_lock = threading.Lock()
    _fps_state = {"count": 0, "t_first": 0.0, "t_last": 0.0,
                   "in_flight": 0, "max_in_flight": 0}
    _report_every = int(os.environ.get("DUAL_REPORT_FPS", "0"))
    _warmup_skip = int(os.environ.get("DUAL_REPORT_WARMUP", "30"))

    # ── Probe: mpv frame_thread gap + compute() breakdown ────────
    # DUAL_PROBE_MPV_GAP=1 enables per-thread "gap" timing (between
    # this thread's _leave() and its next _enter() — the alleged
    # ~16 ms mpv-internal overhead) plus a submit / wait / post
    # breakdown for the split-task path. Aggregates p50/p95/p99/max
    # over a sliding window and dumps every DUAL_PROBE_EVERY frames
    # plus on cleanup. Zero cost when off (one env check at init).
    _probe_on = os.environ.get("DUAL_PROBE_MPV_GAP", "0") == "1"
    _probe_every = int(os.environ.get("DUAL_PROBE_EVERY", "200"))
    _probe_warm = int(os.environ.get("DUAL_PROBE_WARMUP", "30"))
    _probe_buf = int(os.environ.get("DUAL_PROBE_BUF", "4096"))
    _probe_local = threading.local()
    _probe_state = {
        "n": 0,
        "last_exit": {},   # tid -> perf_counter at last _leave
        "gap_ms":    deque(maxlen=_probe_buf),
        "total_ms":  deque(maxlen=_probe_buf),
        "submit_ms": deque(maxlen=_probe_buf),
        "wait_ms":   deque(maxlen=_probe_buf),
        "post_ms":   deque(maxlen=_probe_buf),
    }

    def _probe_pct(arr, p):
        if not arr:
            return float("nan")
        s = sorted(arr)
        i = max(0, min(len(s) - 1, int(len(s) * p / 100)))
        return s[i]

    def _probe_dump_locked(tag="periodic"):
        n = _probe_state["n"]
        if n == 0:
            return
        g  = _probe_state["gap_ms"]
        t  = _probe_state["total_ms"]
        sb = _probe_state["submit_ms"]
        w  = _probe_state["wait_ms"]
        po = _probe_state["post_ms"]
        n_threads = len(_probe_state["last_exit"])
        sys.stderr.write(
            f"[probe mpv gap/{tag}] n={n} threads={n_threads} "
            f"window={len(g)}\n")
        if g:
            sys.stderr.write(
                f"  gap     ms  p50={_probe_pct(g,50):5.2f} "
                f"p95={_probe_pct(g,95):5.2f} "
                f"p99={_probe_pct(g,99):5.2f} max={max(g):5.2f}\n")
        if t:
            sys.stderr.write(
                f"  total   ms  p50={_probe_pct(t,50):5.2f} "
                f"p95={_probe_pct(t,95):5.2f} "
                f"p99={_probe_pct(t,99):5.2f} max={max(t):5.2f}\n")
        if sb:
            sys.stderr.write(
                f"    submit ms p50={_probe_pct(sb,50):5.2f} "
                f"p95={_probe_pct(sb,95):5.2f} "
                f"p99={_probe_pct(sb,99):5.2f}\n"
                f"    wait   ms p50={_probe_pct(w,50):5.2f} "
                f"p95={_probe_pct(w,95):5.2f} "
                f"p99={_probe_pct(w,99):5.2f}\n"
                f"    post   ms p50={_probe_pct(po,50):5.2f} "
                f"p95={_probe_pct(po,95):5.2f} "
                f"p99={_probe_pct(po,99):5.2f}\n")
        if g and t:
            cycle = _probe_pct(g, 50) + _probe_pct(t, 50)
            if cycle > 0 and n_threads > 0:
                sys.stderr.write(
                    f"  per-thread cycle p50≈{cycle:.2f} ms → "
                    f"theoretical {n_threads*1000.0/cycle:.1f} fps with "
                    f"{n_threads} threads\n")
        sys.stderr.flush()
    def _tick_fps():
        if _report_every <= 0:
            return
        with _fps_lock:
            n = _fps_state["count"] + 1
            _fps_state["count"] = n
            now = _time.perf_counter()
            if n == _warmup_skip + 1:
                _fps_state["t_first"] = now
            _fps_state["t_last"] = now
            if n > _warmup_skip and (n - _warmup_skip) % _report_every == 0:
                dt = now - _fps_state["t_first"]
                steady_n = n - _warmup_skip
                fps = steady_n / dt if dt > 0 else 0
                mif = _fps_state["max_in_flight"]
                sys.stderr.write(
                    f"[native_dispatcher fps] +{steady_n} frames in "
                    f"{dt:.2f}s => {fps:.1f} fps steady "
                    f"(peak compute() concurrency = {mif})\n")
                sys.stderr.flush()

    def _enter():
        with _fps_lock:
            _fps_state["in_flight"] += 1
            if _fps_state["in_flight"] > _fps_state["max_in_flight"]:
                _fps_state["max_in_flight"] = _fps_state["in_flight"]
            if _probe_on:
                tid = threading.get_ident()
                now = _time.perf_counter()
                _probe_local.t_enter = now
                _probe_local.bd = None
                last_exit = _probe_state["last_exit"].get(tid)
                if last_exit is not None:
                    _probe_state["gap_ms"].append((now - last_exit) * 1000)
    def _leave():
        # Snap t_exit BEFORE the lock so the post-wait piece doesn't
        # include this lock acquisition. Negligible normally but keeps
        # the breakdown additions consistent.
        t_exit = _time.perf_counter() if _probe_on else 0.0
        with _fps_lock:
            _fps_state["in_flight"] -= 1
            if _probe_on:
                tid = threading.get_ident()
                _probe_state["last_exit"][tid] = t_exit
                t_enter = getattr(_probe_local, "t_enter", t_exit)
                total_ms = (t_exit - t_enter) * 1000
                _probe_state["total_ms"].append(total_ms)
                bd = getattr(_probe_local, "bd", None)
                if bd is not None:
                    sub_ms, wait_ms = bd
                    post_ms = max(0.0, total_ms - sub_ms - wait_ms)
                    _probe_state["submit_ms"].append(sub_ms)
                    _probe_state["wait_ms"].append(wait_ms)
                    _probe_state["post_ms"].append(post_ms)
                _probe_state["n"] += 1
                n = _probe_state["n"]
                if (_probe_every > 0 and n > _probe_warm
                        and (n - _probe_warm) % _probe_every == 0):
                    _probe_dump_locked()

    # Hoist FrameVA + unused imports to module-resolved local refs so
    # the hot-path compute_callable doesn't re-resolve "from X import Y"
    # via the module loader on every call (cached but still a dict
    # lookup + bytecode). Direct local-name lookup is faster.
    from queue_mgr import FrameVA as _FrameVA
    _qm_submit_with_phase_dst = getattr(
        queue_mgr, "submit_with_phase_dst", None)
    _QM_WAIT_ABANDONED = getattr(
        queue_mgr.__class__ if queue_mgr is not None else type(None),
        "WAIT_ABANDONED", "abandoned")

    # On wait_phase_done timeout (the trailing-K corner at end-of-
    # file is the common case — mpv stops issuing K+1 so the last
    # mult-1 phases of K never publish) we used to return silently
    # with the dst VA un-touched; mpv then displayed whatever the
    # buffer pool last held = garbage at file end. Write a
    # known black frame instead so the user sees a brief blackout
    # (1-3 frames, ≤ 84 ms at 24 fps × mult=3) rather than residue.
    #
    # YUV420P10 black: Y = 0 (full range) or 64 (limited range);
    # U = V = 512 (neutral chroma in 10-bit). memset can only write
    # a single byte, but U/V = 512 = 0x0200 doesn't fit a single
    # byte — memsetting U/V to 0 would mean chroma = -512 from
    # neutral, which YUV→RGB renders as bright green. Pre-fill a
    # row buffer with the correct 16-bit-LE pattern once and
    # memmove it into each row.
    _y_pad_row_buf  = (ctypes.c_uint16 * oW)(*([0]   * oW))
    _uv_pad_row_buf = (ctypes.c_uint16 * (oW // 2))(*([512] * (oW // 2)))
    _y_pad_row_n    = ctypes.sizeof(_y_pad_row_buf)
    _uv_pad_row_n   = ctypes.sizeof(_uv_pad_row_buf)
    _memmove = ctypes.memmove

    def _emit_blank_frame(dst_y, dst_u, dst_v, y_stride, uv_stride):
        try:
            for r in range(oH):
                _memmove(dst_y + r * y_stride, _y_pad_row_buf, _y_pad_row_n)
            for r in range(oH // 2):
                _memmove(dst_u + r * uv_stride, _uv_pad_row_buf, _uv_pad_row_n)
                _memmove(dst_v + r * uv_stride, _uv_pad_row_buf, _uv_pad_row_n)
        except Exception:
            pass

    def _compute_split_task(pair_k, phase,
                              sa_y, sa_u, sa_v, sb_y, sb_u, sb_v,
                              dst_y, dst_u, dst_v,
                              sa_y_stride, sa_uv_stride,
                              dst_y_stride, dst_uv_stride):
        """mpv frame_thread submits the pair
        to the queue mgr, blocks on frame_done, then memcpys the
        terminal task's slot.dst (SR_SRC / SR_INTERP) into mpv's dst
        frame VA and releases the worker slot."""
        # Single fused submit call (one _lock acquisition instead of
        # two). Falls back to the split version if queue_mgr doesn't
        # expose the fused API yet.
        _t_sub0 = _time.perf_counter() if _probe_on else 0.0
        if _qm_submit_with_phase_dst is not None:
            _qm_submit_with_phase_dst(
                pair_k, phase,
                sa_y=sa_y, sa_u=sa_u, sa_v=sa_v,
                sb_y=sb_y, sb_u=sb_u, sb_v=sb_v,
                sa_y_stride=sa_y_stride, sa_uv_stride=sa_uv_stride,
                dst_y=dst_y, dst_u=dst_u, dst_v=dst_v,
                dst_y_stride=dst_y_stride, dst_uv_stride=dst_uv_stride)
        else:
            va = _FrameVA(
                pair_k=pair_k,
                sa_y=sa_y, sa_u=sa_u, sa_v=sa_v,
                sb_y=sb_y, sb_u=sb_u, sb_v=sb_v,
                sa_y_stride=sa_y_stride,
                sa_uv_stride=sa_uv_stride,
                dst_y_stride=dst_y_stride,
                dst_uv_stride=dst_uv_stride,
            )
            queue_mgr.set_phase_dst(
                pair_k, phase,
                dst_y=dst_y, dst_u=dst_u, dst_v=dst_v,
                dst_y_stride=dst_y_stride, dst_uv_stride=dst_uv_stride)
            queue_mgr.submit_frame(pair_k, va=va)
        _t_sub1 = _time.perf_counter() if _probe_on else 0.0
        # Wait for THIS phase's terminal task, not the whole pair.
        # mult=3 + mpv CF<3 deadlocks the pair-level wait because phase
        # 2 never gets submitted by mpv until phase 0 or 1 returns;
        # per-phase wait lets each compute() return as soon as its own
        # output is in mpv VA.
        #
        # Cold first pair completes in <250 ms (prewarm already done
        # in make_dispatcher); steady-state is <100 ms. Default 1 s
        # is the safety-net upper bound for the "seek hits a K
        # waiting on its successor's submit AND lua seek-flush hook
        # missed firing" worst case. With CF=24 mpv may park up to
        # ~24 compute() threads in wait_phase_done at seek time;
        # if any of them hit the trailing-K corner case AND the lua
        # flush is late, they each timeout in series — keeping this
        # short minimises the user-visible seek pause. 1 s is well
        # above steady-state (100 ms) and the cold first-pair
        # (250 ms), so legitimate computes never trip it.
        _phase_timeout = float(os.environ.get("DUAL_PHASE_TIMEOUT", "1"))
        _wait_result = queue_mgr.wait_phase_done(
            pair_k, phase, timeout=_phase_timeout)
        if _wait_result == _QM_WAIT_ABANDONED:
            # Seek-flush abandoned the wait. Skip overwriting dst —
            # one frame of buffer-pool residue is acceptable here
            # because the new chain catches up within a couple of
            # vsyncs and overwrites it. Zeroing during seek causes
            # a measurable dispatch stall (mpv vo seems to treat
            # an unwritten dst as "still pending").
            return
        if not _wait_result:
            # Timeout (the trailing-K corner at end-of-file is the
            # common case — mpv stops issuing K+1 so this K's
            # SR_INTERP gates on a CCSR(K+1) that never publishes).
            # Zero the planes so the user sees a black frame instead
            # of buffer-pool residue. Don't raise — pybind11 turns
            # Python exceptions out of compute_callable into a fatal
            # filter error.
            _emit_blank_frame(dst_y, dst_u, dst_v,
                               dst_y_stride, dst_uv_stride)
            sys.stderr.write(
                f"[native_dispatcher] wait_phase_done timeout {_phase_timeout}s "
                f"K={pair_k} phase={phase} — emitted black frame\n")
            sys.stderr.flush()
            return
        if _probe_on:
            _t_wait1 = _time.perf_counter()
            _probe_local.bd = (
                (_t_sub1 - _t_sub0) * 1000,
                (_t_wait1 - _t_sub1) * 1000,
            )
        # Both host (dma_proc phases_mask=0x1 path) and guest (guest_mp
        # CQ thread memmove) write directly to mpv VA before frame_done
        # fires. No post-wait work needed; slot release happens upstream
        # — dma_proc transitions to SEND_PENDING, _hmp_watcher calls
        # task_done + release_slot.

    def compute(pair_k, phase,
                 sa_y, sa_u, sa_v, sb_y, sb_u, sb_v,
                 dst_y, dst_u, dst_v,
                 sa_y_stride, sa_uv_stride,
                 dst_y_stride, dst_uv_stride):
        """The native filter calls this once per output frame.
        pair_k = n // mult, phase = n % mult."""
        _enter()
        if os.environ.get("DUAL_COMPUTE_TRACE", "0") == "1":
            import time as _t
            sys.stderr.write(
                f"[cpu] t={_t.perf_counter():.3f} compute(K={pair_k}, "
                f"phase={phase}) ENTER dst=0x{dst_y:x}\n")
            sys.stderr.flush()
        try:
            _compute_split_task(
                pair_k, phase,
                sa_y, sa_u, sa_v, sb_y, sb_u, sb_v,
                dst_y, dst_u, dst_v,
                sa_y_stride, sa_uv_stride,
                dst_y_stride, dst_uv_stride)
            _tick_fps()
            if os.environ.get("DUAL_COMPUTE_TRACE", "0") == "1":
                import time as _t
                sys.stderr.write(
                    f"[cpu] t={_t.perf_counter():.3f} compute(K={pair_k}, "
                    f"phase={phase}) RETURN\n")
                sys.stderr.flush()
        finally:
            _leave()

    def cleanup():
        # No-op by design: the singleton path
        # (_dispatcher_singleton._state) keeps host_mp / queue_mgr /
        # guest_mp alive across rife.vpy invocations so seek doesn't
        # respawn the 3-proc + reload TRT engines (8-10 s otherwise).
        # Real OS-level cleanup happens at mpv-process exit via the
        # parent-exit SIGHUP path + mp_pipeline's atexit shm unlink.
        # If `init_timeout_s` debug or test code in the future needs
        # an explicit teardown (e.g. before re-init with different
        # params), use `_dispatcher_singleton.reset()` after
        # invoking the saved shutdown methods.
        pass

    # ── Save to singleton so the next make_dispatcher() call (vf
    #    rebuild on seek) can short-circuit ─────────────────────────
    if _singleton_get is not None:
        with _singleton_lock():
            _singleton_get().update({
                "params_key": _params_key,
                "host_mp": host_mp,
                "queue_mgr": queue_mgr,
                "guest_mp": guest_mp,
                "compute": compute,
                "liveness_sock": _liveness_sock,
            })
        sys.stderr.write(
            "[native_dispatcher] singleton SAVE — host_mp+queue_mgr+"
            "guest_mp cached for next vf-rebuild\n")
        sys.stderr.flush()

    return compute, cleanup
