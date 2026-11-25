"""Experimental mpv entry point for frame-owned remote processing."""
from __future__ import annotations

import atexit
import ctypes
import os
import secrets
import socket
import struct
import sys
import threading
import time
import warnings
from types import SimpleNamespace

from guest_mp import GuestMP
from peer_control import PeerControlClient
from remote_endpoint import RemoteEndpoint
from tictac_runtime import FrameSet, TicTacRuntime

_lock = threading.RLock()
_current = None


class InsufficientWorkers(RuntimeError):
    def __init__(self, available):
        super().__init__("fewer than two tic-tac workers are reachable")
        self.available = tuple(available)


def _ready_workers(pairs, handshake, timeout):
    """Keep workers that complete the session handshake and become ready."""
    sent = []
    for host, sock in pairs:
        try:
            sock.sendall(handshake)
        except OSError as exc:
            sys.stderr.write(f"[tictac] worker {host} handshake failed: {exc}\n")
            sock.close()
        else:
            sent.append((host, sock))
    ready = []
    for host, sock in sent:
        try:
            sock.settimeout(timeout)
            if sock.recv(1) != b"R":
                raise RuntimeError("worker did not become ready")
            sock.settimeout(None)
        except (OSError, RuntimeError) as exc:
            sys.stderr.write(f"[tictac] worker {host} unavailable during setup: {exc}\n")
            sock.close()
        else:
            ready.append((host, sock))
    return ready


def _restore_env(saved, applied):
    for key, previous in saved.items():
        if os.environ.get(key) != applied[key]:
            continue
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def _close_current():
    global _current
    with _lock:
        current, _current = _current, None
        if current is not None:
            current.close()
            _restore_env(current._saved_env, current._applied_env)


atexit.register(_close_current)


def make_dispatcher(clip, *, worker_hosts, variant, matrix_s, color_range,
                    rife_model, chroma_kernel, downsample_pre, output_scale,
                    dst_sub_w, dst_sub_h, interp_mult, no_sr):
    """Create a playback callable without loading models on the host."""
    global _current
    import vs_gpu_helpers as vgh
    variants = ["FSRCNNX_x2_8-0-4-1", "FSRCNNX_x2_16-0-4-1",
                "FSRCNNX_x3_16-0-4-1", "FSRCNNX_x4_16-0-4-1"]
    chroma = ["bilinear", "bicubic", "nearest"]
    matrices = {"709": 1, "470bg": 5, "170m": 6, "240m": 7,
                "2020ncl": 9, "2020cl": 10}
    if len(worker_hosts) < 2 or interp_mult not in (2, 3, 4):
        raise ValueError("tic-tac needs two remote workers and interpolation")
    h, w = clip.height, clip.width
    sub_h, sub_w = clip.format.subsampling_h, clip.format.subsampling_w
    # The playback callback outlives the VapourSynth filter. Retaining its
    # input node here prevents mpv from releasing the filter during teardown.
    clip_metadata = SimpleNamespace(height=h, width=w,
        format=SimpleNamespace(bits_per_sample=clip.format.bits_per_sample,
                               num_planes=clip.format.num_planes,
                               subsampling_h=sub_h, subsampling_w=sub_w))
    ch, cw = h >> sub_h, w >> sub_w
    meta = vgh.rife_engine_metadata(rife_model, h // downsample_pre,
                                    w // downsample_pre)
    fused = os.environ.get("CLUSTER_TICTAC_FUSED", "1") == "1" and interp_mult == 2
    overlap = fused and os.environ.get("CLUSTER_TICTAC_OVERLAP", "1") == "1"
    peer_write = overlap and os.environ.get("CLUSTER_TICTAC_PEER_WRITE", "1") == "1"
    balanced_src = os.environ.get("CLUSTER_TICTAC_BALANCED_SRC", "1") == "1"
    pull_results = os.environ.get("CLUSTER_TICTAC_PULL", "1") == "1"
    handshake = (9 if peer_write else 8 if overlap else 7 if fused else 6,
                 h, w, output_scale, variants.index(variant),
                 chroma.index(chroma_kernel), sub_w, sub_h,
                 matrices.get(matrix_s, 1), 0 if color_range == "full" else 1,
                 {"4.26": 0, "4.6": 1}[rife_model], downsample_pre,
                 dst_sub_w, dst_sub_h, meta["ph"], meta["pw"], interp_mult,
                 int(no_sr), meta["encode_channel"], (2 if balanced_src else 1) + (4 if pull_results else 0))
    liveness_port = int(os.environ.get("CLUSTER_LIVENESS_PORT", "29905"))
    rdma_port = int(os.environ.get("CLUSTER_RDMA_PORT", "29900"))
    peer_base = int(os.environ.get("CLUSTER_PEER_BASE_PORT", "30100"))
    rails = len((os.environ.get("CLUSTER_RDMA_DEVS", "") or "one").split(","))
    if not 1024 <= peer_base or peer_base + rails * len(worker_hosts) * (len(worker_hosts) - 1) > 65535:
        raise ValueError("peer port range is invalid")

    with _lock:
        _close_current()
        env = {"CLUSTER_LOCAL_FRAME_CACHE": "1", "CLUSTER_WORKER_FRAME_IO": "1",
               "CLUSTER_PULL_RESULTS": "1" if pull_results else "0",
               "CLUSTER_PULL_NORMAL": "0",
               "CLUSTER_WORKER_PREFETCH": "1", "CLUSTER_MID_DELIVERY": "1",
               "CLUSTER_BALANCED_SRC": "1" if balanced_src else "0",
               "CLUSTER_WORKER_FUSED": "1" if fused else "0",
               "CLUSTER_INTERP_MULT": str(interp_mult),
               "CLUSTER_DOWNSAMPLE_PRE": str(downsample_pre)}
        saved_env = {key: os.environ.get(key) for key in env
                     if key != "CLUSTER_INTERP_MULT"}
        os.environ.update(env)
        sockets = []
        endpoints = {}
        controls = {}
        try:
            reachable = []
            for host in worker_hosts:
                try:
                    sock = socket.create_connection((host, liveness_port), timeout=0.5)
                except OSError as exc:
                    sys.stderr.write(f"[tictac] worker {host} unavailable: {exc}\n")
                    continue
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                try:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 1)
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 1)
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 2)
                except OSError:
                    pass
                sockets.append(sock)
                reachable.append(host)
            if len(reachable) < 2:
                raise InsufficientWorkers(reachable)
            ready = _ready_workers(
                zip(reachable, sockets), struct.pack("<20q", *handshake),
                float(os.environ.get("CLUSTER_WORKER_READY_TIMEOUT", "300")))
            if len(ready) < 2:
                raise InsufficientWorkers(host for host, _ in ready)
            worker_hosts = tuple(host for host, _ in ready)
            sockets = [sock for _, sock in ready]
            for host, sock in zip(worker_hosts, sockets):
                control = PeerControlClient(sock)
                controls[host] = control
                guest = GuestMP(H=h, W=w, scale=output_scale, sub_w=sub_w, sub_h=sub_h,
                                dst_sub_w=dst_sub_w, dst_sub_h=dst_sub_h,
                                pH=meta["ph"], pW=meta["pw"], enc_ch=meta["encode_channel"],
                                n_slots=6 if pull_results else 4 if overlap else 2,
                                peer_ip=host, peer_handshake_port=rdma_port,
                                rdma_dev=os.environ.get("CLUSTER_RDMA_DEV", "rocep1s0f0"),
                                rdma_gid_index=int(os.environ.get("CLUSTER_RDMA_GID", "3")),
                                log=lambda message: sys.stderr.write(message + "\n"))
                endpoints[host] = RemoteEndpoint(guest, control)
            links = {}
            edge = 0
            for src in worker_hosts:
                for dst in worker_hosts:
                    if src == dst:
                        continue
                    name = f"edge-{edge}"
                    port = peer_base + rails * edge
                    token = secrets.token_hex(16)
                    server = controls[src].submit("connect", name, host=dst, port=port,
                                                  server=True, token=token,
                                                  transport="write" if peer_write else "send")
                    client = controls[dst].submit("connect", name, host=src, port=port,
                                                  server=False, token=token,
                                                  transport="write" if peer_write else "send")
                    server.result(30)
                    client.result(30)
                    links[(src, dst)] = name
                    edge += 1
            runtime = TicTacRuntime(endpoints, links, interp_mult=interp_mult,
                                    fused_interp_sr=fused, overlap_output=overlap,
                                    peer_write=peer_write)
            runtime._saved_env = saved_env
            runtime._applied_env = env
            _current = runtime
        except Exception:
            for endpoint in endpoints.values():
                endpoint.close()
            for control in controls.values():
                control.close()
            for sock in sockets:
                sock.close()
            _restore_env(saved_env, env)
            raise

    submit_lock = threading.Lock()
    submitted = FrameSet()
    delivered = [FrameSet() for _ in range(interp_mult)]
    fallback_lock = threading.Lock()
    fallback_compute = None
    recovery_backing = []
    recovery_sources = {}
    fps_lock = threading.Lock()
    fps_count = 0
    fps_start = 0.0
    report_every = int(os.environ.get("CLUSTER_REPORT_FPS", "0"))
    warmup = max(1, int(os.environ.get("CLUSTER_REPORT_WARMUP", "30")))

    def copy_plane(addr, rows, row_bytes, stride):
        if not addr or stride < row_bytes:
            raise ValueError("invalid source frame plane")
        if stride == row_bytes:
            return ctypes.string_at(addr, rows * row_bytes)
        return b"".join(ctypes.string_at(addr + row * stride, row_bytes)
                        for row in range(rows))

    def source(y, u, v, ys, uvs):
        return (copy_plane(y, h, 2 * w, ys), copy_plane(u, ch, 2 * cw, uvs),
                copy_plane(v, ch, 2 * cw, uvs))

    oh, ow = h * output_scale, w * output_scale
    och, ocw = oh >> dst_sub_h, ow >> dst_sub_w
    y_blank = (ctypes.c_uint16 * ow)(*([0 if color_range == "full" else 64] * ow))
    uv_blank = (ctypes.c_uint16 * ocw)(*([512] * ocw))

    def blank(y, u, v, ys, uvs):
        for row in range(oh):
            ctypes.memmove(y + row * ys, y_blank, ow * 2)
        for row in range(och):
            ctypes.memmove(u + row * uvs, uv_blank, ocw * 2)
            ctypes.memmove(v + row * uvs, uv_blank, ocw * 2)

    def recover():
        nonlocal fallback_compute
        with fallback_lock:
            if fallback_compute is not None:
                return fallback_compute
            runtime.failure_identified.wait(3)
            lost = runtime.failed_workers
            remaining = tuple(host for host in worker_hosts if host not in lost)
            if not lost or not remaining:
                raise RuntimeError("cannot identify a surviving worker")
            sys.stderr.write(f"[tictac] lost {len(lost)} worker(s); "
                             f"rebuilding with {len(remaining)} remote worker(s)\n")
            with runtime._cv:
                retained_sources = dict(runtime._sources)
            recovery_sources.update(retained_sources)
            with submit_lock:
                already_delivered = {frame for frame in retained_sources
                                     if frame in delivered[0]}
            _close_current()
            os.environ["CLUSTER_WORKER_HOSTS"] = ",".join(remaining)
            os.environ["CLUSTER_WORKER_HOST"] = remaining[0]
            if len(remaining) >= 2:
                fallback_compute, _ = make_dispatcher(
                    clip_metadata, worker_hosts=remaining, variant=variant, matrix_s=matrix_s,
                    color_range=color_range, rife_model=rife_model,
                    chroma_kernel=chroma_kernel, downsample_pre=downsample_pre,
                    output_scale=output_scale, dst_sub_w=dst_sub_w, dst_sub_h=dst_sub_h,
                    interp_mult=interp_mult, no_sr=no_sr)
                # A few later intervals may already have reached mpv
                # before the worker failed. Replaying the retained input
                # window avoids holes that would otherwise pin both A/B
                # slots while earlier callbacks are retried.
                new_runtime = _current
                for frame in sorted(retained_sources):
                    if frame + 1 in retained_sources:
                        new_runtime.submit_interval(frame, retained_sources[frame],
                                                    retained_sources[frame + 1])
            else:
                from native_dispatcher import make_dispatcher as make_dual
                # VapourSynth's logging bridge cannot handle this PyTorch
                # deprecation warning from a frame callback thread.
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning,
                                            message=".*torch.jit.load.*")
                    fallback_compute, _ = make_dual(
                        clip_metadata, variant=variant, matrix_s=matrix_s,
                        color_range=color_range, rife_model=rife_model,
                        chroma_kernel=chroma_kernel, downsample_pre=downsample_pre,
                        interp_mult=interp_mult, no_sr=no_sr)
                from _dispatcher_singleton import get as singleton_get
                from queue_mgr import FrameVA
                manager = singleton_get()["queue_mgr"]
                # Register the whole retained window atomically. A CCSR
                # must see both neighboring INTERP consumers before it can
                # complete and initialise its cache reference counts.
                with manager._lock:
                    for frame in sorted(retained_sources):
                        first = [ctypes.create_string_buffer(plane)
                                 for plane in retained_sources[frame]]
                        next_planes = retained_sources.get(frame + 1)
                        second = ([ctypes.create_string_buffer(plane)
                                   for plane in next_planes] if next_planes else [])
                        addresses = [ctypes.addressof(p) for p in first]
                        next_addresses = ([ctypes.addressof(p) for p in second]
                                          if second else [0, 0, 0])
                        dst = []
                        if frame in already_delivered:
                            dst = [ctypes.create_string_buffer(n) for n in
                                   (oh * ow * 2, och * ocw * 2, och * ocw * 2)]
                            manager.submit_with_phase_dst(
                                frame, 0, sa_y=addresses[0], sa_u=addresses[1],
                                sa_v=addresses[2], sb_y=next_addresses[0],
                                sb_u=next_addresses[1], sb_v=next_addresses[2],
                                sa_y_stride=w * 2, sa_uv_stride=cw * 2,
                                dst_y=ctypes.addressof(dst[0]),
                                dst_u=ctypes.addressof(dst[1]),
                                dst_v=ctypes.addressof(dst[2]),
                                dst_y_stride=ow * 2, dst_uv_stride=ocw * 2)
                        else:
                            manager.submit_frame(frame, va=FrameVA(
                                pair_k=frame, sa_y=addresses[0], sa_u=addresses[1],
                                sa_v=addresses[2], sb_y=next_addresses[0],
                                sb_u=next_addresses[1], sb_v=next_addresses[2],
                                sa_y_stride=w * 2, sa_uv_stride=cw * 2))
                        # DMA may outlive an individual callback.
                        recovery_backing.append((first, second, dst))
            return fallback_compute

    def compute(k, phase, ay, au, av, by, bu, bv, dy, du, dv,
                src_ys, src_uvs, dst_ys, dst_uvs):
        nonlocal fps_count, fps_start

        def check_recovery_source():
            if os.environ.get("CLUSTER_RECOVERY_DEBUG", "0") != "1":
                return
            for frame, values in ((k, (ay, au, av)), (k + 1, (by, bu, bv))):
                expected = recovery_sources.get(frame)
                if expected is not None:
                    actual = source(*values, src_ys, src_uvs)
                    sys.stderr.write(f"[tictac recovery] frame={frame} "
                                     f"phase={phase} source_match={actual == expected}\n")
        try:
            if fallback_compute is not None:
                check_recovery_source()
                return fallback_compute(k, phase, ay, au, av, by, bu, bv,
                                        dy, du, dv, src_ys, src_uvs, dst_ys, dst_uvs)
            with submit_lock:
                if k not in submitted:
                    runtime.submit_interval_lazy(
                        k, lambda: source(ay, au, av, src_ys, src_uvs),
                        lambda: source(by, bu, bv, src_ys, src_uvs))
                    submitted.add(k)
            out = runtime.wait_output(k, phase,
                                      timeout=float(os.environ.get("CLUSTER_PHASE_TIMEOUT", "5")))
            GuestMP._copy_plane(ctypes.addressof(out.y), dy, oh, ow * 2, dst_ys)
            GuestMP._copy_plane(ctypes.addressof(out.u), du, och, ocw * 2, dst_uvs)
            GuestMP._copy_plane(ctypes.addressof(out.v), dv, och, ocw * 2, dst_uvs)
            with submit_lock:
                delivered[phase].add(k)
                runtime.consume_output(k, phase)
            if report_every > 0:
                with fps_lock:
                    fps_count += 1
                    if fps_count == warmup:
                        fps_start = time.monotonic()
                    count = fps_count - warmup
                    if count > 0 and count % report_every == 0:
                        elapsed = time.monotonic() - fps_start
                        sys.stderr.write(f"[tictac fps] {count / elapsed:.1f} fps "
                                         f"({count} frames in {elapsed:.2f}s)\n")
        except Exception as exc:
            if runtime._error is not None:
                try:
                    fallback = recover()
                    check_recovery_source()
                    return fallback(k, phase, ay, au, av, by, bu, bv,
                                    dy, du, dv, src_ys, src_uvs, dst_ys, dst_uvs)
                except Exception as recovery_error:
                    sys.stderr.write(f"[tictac] recovery failed: {recovery_error}\n")
                    import traceback
                    traceback.print_exc()
            sys.stderr.write(f"[tictac] frame={k} phase={phase} failed: {exc}\n")
            blank(dy, du, dv, dst_ys, dst_uvs)

    sys.stderr.write(f"[tictac] ready with {len(endpoints)} remote workers\n")
    return compute, lambda: None
