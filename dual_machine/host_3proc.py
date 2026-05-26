"""3-process host: dma_proc + buffer_mgr_proc + compute_proc.

Symmetric mirror of worker_3proc.py — same slot ring, same eventfds,
same state machine. The only thing that differs is the data-movement
process: worker's rdma_proc fetches src bytes from the NIC; host's
dma_proc fetches them from mpv's frame memory via process_vm_readv
(likewise process_vm_writev on the writeback).

Almost everything else is reused from worker_3proc:
  - SlotRingShm / SlotMeta / EventFdChannel       (shm + eventfds)
  - ST_* state constants                          (state machine)
  - compute_proc_main, buffer_mgr_proc_main       (process bodies)
  - MPPipelineBase                                (orchestrator base)

This module only adds:
  - HostSlotRingLayout  — extends SlotRingLayout with a RequestMeta table
  - RequestMeta         — per-slot mpv-side pointers + strides
  - dma_proc_main       — process body that does the actual readv/writev
  - HostMP              — MPPipelineBase subclass

Dispatcher integration: native_dispatcher.py spawns HostMP. The host
compute path submits via HostMP.claim_slot() / get_request() /
notify_submit() instead of running compute in-process.
"""
from __future__ import annotations

import ctypes
import os
import sys

from mp_pipeline import (
    SlotRingLayout, SlotRingShm, MPPipelineBase,
    ST_FREE, ST_RECV_PENDING, ST_RECV_DONE, ST_DST_READY, ST_SEND_PENDING,
    ST_FILLING, ST_MID_READY, is_mid_ready_state, mid_ready_stage,
    _align_up, _setup_logging, _ef_from_env,
)
from cc_cache import (
    CCCacheLayout, CCCacheShm,
    FIELD_RGB_PADDED, FIELD_RIFE_FEATURES, FIELD_SR_YUV,
    FIELD_RGB_INTERP, FIELD_RGB_4K,
)

_CC_FIELD_BY_INDEX = (FIELD_RGB_PADDED, FIELD_RIFE_FEATURES,
                       FIELD_SR_YUV, FIELD_RGB_4K, FIELD_RGB_INTERP)


# ──────────────────────────────────────────────────────────────────────
# RequestMeta — per-slot mpv-side pointer block in shm
# ──────────────────────────────────────────────────────────────────────


class RequestMeta(ctypes.Structure):
    """Filled by mpv dispatcher when submitting a pair; read by
    dma_proc to drive process_vm_readv/writev against mpv's PID.

    All addresses are mpv-process virtual addresses (valid until the
    dispatcher returns from compute(), which it won't do until after
    dma_proc has finished the writeback). pair_k / src_a_idx /
    src_b_idx mirror worker's wire header so compute_proc sees the
    same encode-cache keys.

    task_type / task_id drive the split-task pipeline to different
    kernels (CC / SR_SRC / INTERP / SR_INTERP).
    """
    _fields_ = [
        ("pair_k",       ctypes.c_int64),
        ("src_a_idx",    ctypes.c_int64),
        ("src_b_idx",    ctypes.c_int64),
        ("mpv_pid",      ctypes.c_int64),

        ("sa_y",         ctypes.c_uint64),
        ("sa_u",         ctypes.c_uint64),
        ("sa_v",         ctypes.c_uint64),
        ("sb_y",         ctypes.c_uint64),
        ("sb_u",         ctypes.c_uint64),
        ("sb_v",         ctypes.c_uint64),
        ("dst_ya",       ctypes.c_uint64),
        ("dst_ua",       ctypes.c_uint64),
        ("dst_va",       ctypes.c_uint64),
        ("dst_yi",       ctypes.c_uint64),
        ("dst_ui",       ctypes.c_uint64),
        ("dst_vi",       ctypes.c_uint64),

        ("sa_y_stride",   ctypes.c_uint32),
        ("sa_uv_stride",  ctypes.c_uint32),
        ("dst_y_stride_a",  ctypes.c_uint32),
        ("dst_uv_stride_a", ctypes.c_uint32),
        ("dst_y_stride_i",  ctypes.c_uint32),
        ("dst_uv_stride_i", ctypes.c_uint32),

        # bit0 = phase-a writeback, bit1 = phase-i writeback. Dispatcher
        # typically sets both since dst frames for both phases are
        # known upfront (one ModifyFrame per phase, but they read the
        # cached pair result). task_type packed adjacent to keep
        # task_id 8-byte aligned without padding gaps.
        ("phases_mask",   ctypes.c_uint32),
        ("task_type",     ctypes.c_uint32),
        ("task_id",       ctypes.c_uint64),

        # cc_cache reference block. Each slot id is -1 when the
        # corresponding direction is not used (CC sets dst_cc_slot
        # only; INTERP sets src_cc_slot_a/b only; SR_SRC sets
        # src_cc_slot_a only). cc_field is one of CC_FIELD_RGB_PADDED
        # / RIFE_FEATURES / SR_YUV / RGB_4K below.
        ("src_cc_slot_a", ctypes.c_int32),
        ("src_cc_slot_b", ctypes.c_int32),
        ("src_cc_field",  ctypes.c_int32),
        ("dst_cc_slot",   ctypes.c_int32),
        ("dst_cc_field",  ctypes.c_int32),

        # INTERP timestep — unused now that worker reads
        # DUAL_INTERP_MULT env and runs dual flownet inside one task.
        # Kept for binary compat; worker ignores when mult>2.
        ("interp_timestep_num",   ctypes.c_uint32),
        ("interp_timestep_denom", ctypes.c_uint32),
        # INTERP writes mult-1 output frames into cc_cache (dense pack).
        # dst_cc_slot = frame 0; dst_cc_slot_2 = frame 1 (mult ≥ 3);
        # dst_cc_slot_3 = frame 2 (mult == 4). -1 = unused.
        ("dst_cc_slot_2", ctypes.c_int32),
        ("dst_cc_slot_3", ctypes.c_int32),

        # Pad to 256 B (4 cache lines). Used 204 B so far; 52 B pad.
        ("_pad",          ctypes.c_uint8 * 52),
    ]


REQ_META_BYTES = ctypes.sizeof(RequestMeta)
assert REQ_META_BYTES == 256, f"RequestMeta is {REQ_META_BYTES}B, expected 256"

# cc_cache field selector values (must match cc_cache._ALL_FIELDS order)
CC_FIELD_NONE          = -1
CC_FIELD_RGB_PADDED    = 0
CC_FIELD_RIFE_FEATURES = 1
CC_FIELD_SR_YUV        = 2
CC_FIELD_RGB_4K        = 3
CC_FIELD_RGB_INTERP    = 4

_CC_FIELD_NAMES = {
    CC_FIELD_RGB_PADDED:    "rgb_padded",
    CC_FIELD_RIFE_FEATURES: "rife_features",
    CC_FIELD_SR_YUV:        "sr_yuv",
    CC_FIELD_RGB_4K:        "rgb_4K",
    CC_FIELD_RGB_INTERP:    "rgb_interp",
}


# ──────────────────────────────────────────────────────────────────────
# Layout extension — adds a request-meta table after the state table
# ──────────────────────────────────────────────────────────────────────


class HostSlotRingLayout(SlotRingLayout):
    """Extends SlotRingLayout with a per-slot RequestMeta table.

    Layout (page-aligned regions):
      [state_table        : n_slots × 64 B  ]   (from base)
      [request_meta_table : n_slots × 192 B ]   (NEW — extra_prefix)
      [slot 0             : src_a + dst_a   ]
      ...

    The request table lives in the `extra_prefix` region that
    SlotRingLayout reserves before slot 0, so compute_proc (which
    builds its own SlotRingLayout) can match the total_size by
    setting WMP_EXTRA_PREFIX to the same byte count.
    """

    def __init__(self, *, n_slots: int, H: int, W: int,
                 scale: int, sub_w: int, sub_h: int,
                 dst_sub_w: int = 1, dst_sub_h: int = 1,
                 pH: int = 1088, pW: int = 1920, enc_ch: int = 4):
        req_table_size_a = _align_up(n_slots * REQ_META_BYTES)
        super().__init__(n_slots=n_slots, H=H, W=W,
                          scale=scale, sub_w=sub_w, sub_h=sub_h,
                          dst_sub_w=dst_sub_w, dst_sub_h=dst_sub_h,
                          extra_prefix_bytes_aligned=req_table_size_a,
                          pH=pH, pW=pW, enc_ch=enc_ch)
        # Convenience accessors — extra_prefix_off / _size_a from base
        # are the canonical names; req_table_* are aliases for clarity.
        self.req_table_off = self.extra_prefix_off
        self.req_table_size_a = self.extra_prefix_size_a

    def slot_req_off(self, i: int) -> int:
        return self.req_table_off + i * REQ_META_BYTES


def _attach_request(shm: SlotRingShm, layout: HostSlotRingLayout,
                     i: int) -> RequestMeta:
    return RequestMeta.from_buffer(shm.mm, layout.slot_req_off(i))


# ──────────────────────────────────────────────────────────────────────
# process_vm_readv / writev wrappers (one ctypes import for the proc)
# ──────────────────────────────────────────────────────────────────────


class _Iovec(ctypes.Structure):
    _fields_ = [("iov_base", ctypes.c_void_p),
                ("iov_len",  ctypes.c_size_t)]


_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.process_vm_readv.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(_Iovec), ctypes.c_ulong,
    ctypes.POINTER(_Iovec), ctypes.c_ulong,
    ctypes.c_ulong,
]
_libc.process_vm_readv.restype = ctypes.c_ssize_t
_libc.process_vm_writev.argtypes = _libc.process_vm_readv.argtypes
_libc.process_vm_writev.restype = ctypes.c_ssize_t


def _pvm_plane(*, pid: int, local_addr: int, remote_addr: int,
                h: int, w: int, elem_size: int, remote_stride: int,
                write: bool) -> None:
    """Copy one plane via process_vm_{read,write}v. Fast path = stride
    matches packed → single iovec; slow path = row-by-row in batches
    of IOV_MAX. vapoursynth's stride is 64-byte aligned and our plane
    widths are already multiples, so packed is the normal case."""
    row_bytes = w * elem_size
    fn = _libc.process_vm_writev if write else _libc.process_vm_readv
    if remote_stride == row_bytes:
        local = _Iovec(ctypes.c_void_p(local_addr),
                        ctypes.c_size_t(row_bytes * h))
        remote = _Iovec(ctypes.c_void_p(remote_addr),
                         ctypes.c_size_t(row_bytes * h))
        rc = fn(pid, ctypes.byref(local), 1, ctypes.byref(remote), 1, 0)
        if rc != row_bytes * h:
            err = ctypes.get_errno()
            op = "writev" if write else "readv"
            raise RuntimeError(
                f"process_vm_{op}(pid={pid}, n={row_bytes*h}) "
                f"returned {rc} (errno={err})")
        return
    IOV_MAX = 1024
    local_arr = (_Iovec * IOV_MAX)()
    remote_arr = (_Iovec * IOV_MAX)()
    r = 0
    while r < h:
        n = min(IOV_MAX, h - r)
        for i in range(n):
            local_arr[i] = _Iovec(
                ctypes.c_void_p(local_addr + (r + i) * row_bytes),
                ctypes.c_size_t(row_bytes))
            remote_arr[i] = _Iovec(
                ctypes.c_void_p(remote_addr + (r + i) * remote_stride),
                ctypes.c_size_t(row_bytes))
        rc = fn(pid, local_arr, n, remote_arr, n, 0)
        if rc != n * row_bytes:
            err = ctypes.get_errno()
            op = "writev" if write else "readv"
            raise RuntimeError(
                f"process_vm_{op}(pid={pid}, rows={n}) "
                f"returned {rc} (errno={err})")
        r += n


# ──────────────────────────────────────────────────────────────────────
# dma_proc — host equivalent of worker's rdma_proc
# ──────────────────────────────────────────────────────────────────────


def dma_proc_main():
    """Host DMA process. Two threads:
      - submit_listener: blocks on WMP_EF_REQUEST. When notified,
        scans shm for RECV_PENDING slots, reads RequestMeta, runs
        process_vm_readv to pull mpv-side src bytes into the slot's
        src bundle, marks RECV_DONE, signals buffer_mgr.
      - writeback_listener: blocks on WMP_EF_MGR_TO_RDMA. When
        notified, scans shm for DST_READY slots, runs process_vm_writev
        to push the slot's dst bundle into mpv-side dst frames, marks
        FREE, signals WMP_EF_DONE so dispatcher's completion watcher
        wakes the per-pair Event.
    """
    import threading
    log = _setup_logging("dma")
    # dma_proc IS the writer of the RequestMeta table, so it needs the
    # full HostSlotRingLayout (with req_table_off / slot_req_off).
    # compute_proc and buffer_mgr_proc don't touch req_table directly —
    # they just need total_size matching, which WMP_EXTRA_PREFIX gives.
    layout = HostSlotRingLayout(
        n_slots=int(os.environ["WMP_N_SLOTS"]),
        H=int(os.environ["WMP_H"]),
        W=int(os.environ["WMP_W"]),
        scale=int(os.environ["WMP_SCALE"]),
        sub_w=int(os.environ["WMP_SUB_W"]),
        sub_h=int(os.environ["WMP_SUB_H"]),
        pH=int(os.environ.get("WMP_SPLIT_PH", "1088")),
        pW=int(os.environ.get("WMP_SPLIT_PW", "1920")),
        enc_ch=int(os.environ.get("WMP_SPLIT_ENC_CH", "4")),
    )
    shm = SlotRingShm(layout, name=os.environ["WMP_SHM_NAME"])
    shm.open_and_map()

    # cc_cache — optional. When HMP_CC_CACHE_NAME is set, open the
    # cuda-pinned shm region the orchestrator created so we can
    # memcpy between worker slots and cc_cache slots according to
    # RequestMeta's cc_cache fields.
    cc_shm: CCCacheShm | None = None
    cc_name = os.environ.get("HMP_CC_CACHE_NAME")
    if cc_name:
        cc_layout = CCCacheLayout(
            n_slots=int(os.environ["HMP_CC_CACHE_NSLOTS"]),
            H_src=int(os.environ["HMP_CC_CACHE_H_SRC"]),
            W_src=int(os.environ["HMP_CC_CACHE_W_SRC"]),
            H_out=int(os.environ["HMP_CC_CACHE_H_OUT"]),
            W_out=int(os.environ["HMP_CC_CACHE_W_OUT"]),
            pH=int(os.environ["HMP_CC_CACHE_PH"]),
            pW=int(os.environ["HMP_CC_CACHE_PW"]),
            enc_channels=int(os.environ["HMP_CC_CACHE_ENC_CH"]),
            sub_h=int(os.environ["HMP_CC_CACHE_SUB_H"]),
            sub_w=int(os.environ["HMP_CC_CACHE_SUB_W"]),
        )
        cc_shm = CCCacheShm(cc_layout, name=cc_name)
        cc_shm.open()
        log(f"cc_cache opened: {cc_layout.describe()}")

    ef_request   = _ef_from_env("request")
    ef_done      = _ef_from_env("done")
    ef_to_mgr    = _ef_from_env("rdma_to_mgr")
    ef_from_mgr  = _ef_from_env("mgr_to_rdma")

    H, W = layout.H, layout.W
    cH, cW = layout.cH, layout.cW
    oH, oW = layout.oH, layout.oW
    ocH, ocW = layout.ocH, layout.ocW

    # Plane offsets within src bundle (16 int64 header, then 6 planes).
    HDR_BYTES = 16 * 8
    sz_y = H * W * 2
    sz_uv = cH * cW * 2
    off_ya = HDR_BYTES
    off_ua = off_ya + sz_y
    off_va = off_ua + sz_uv
    off_yb = off_va + sz_uv
    off_ub = off_yb + sz_y
    off_vb = off_ub + sz_uv
    # `<=` not `==` because slot.src is sized to the larger of (named
    # YUV planes) and (INTERP overlay = header + 2x(rgb_padded +
    # rife_features)). The named planes always fit at the start; the
    # tail is the INTERP overlay region which dma_proc doesn't touch.
    assert off_vb + sz_uv <= layout.src_size, \
        f"src plane offsets overflow: {off_vb + sz_uv} > {layout.src_size}"

    # Plane offsets within dst bundle.
    # Output is always 4K 4:2:0 (mpv consumer), so only one phase's
    # planes (yao/uao/vao). The dst bundle also carries rgb_padded +
    # rife_features after vao (CCSR mid-out for guest path).
    osz_y = oH * oW * 2
    osz_uv = ocH * ocW * 2
    off_yao = 0
    off_uao = off_yao + osz_y
    off_vao = off_uao + osz_uv

    state_lock = threading.Lock()
    stopping = threading.Event()
    counts = {"readv": 0, "writev": 0}
    # Per-task wall-time accounting (only enabled when
    # DUAL_SPLIT_DEBUG=1). readv = bytes-into-shm cost; wait =
    # time spent blocked on ef_request.wait() between consecutive
    # submits. Healthy wait_avg ≈ compute_proc's per-task kernel
    # time (saturated worker).
    import time as _t_dma
    timing = {"readv_sum_ms": 0.0, "readv_n": 0,
              "writev_sum_ms": 0.0, "writev_n": 0,
              "writev_bytes": 0,
              "wait_sum_ms":  0.0, "wait_n":  0}
    _dma_debug = os.environ.get("DUAL_SPLIT_DEBUG", "0") == "1"
    _io_prof = os.environ.get("DUAL_IO_PROF", "0") == "1"

    # Pre-build the descriptor table so each pair's readv/writev loop
    # just iterates a flat list — keeps the hot path tight.
    SRC_PLANES = [
        ("ya", off_ya, H, W, "sa_y", "sa_y_stride"),
        ("ua", off_ua, cH, cW, "sa_u", "sa_uv_stride"),
        ("va", off_va, cH, cW, "sa_v", "sa_uv_stride"),
        ("yb", off_yb, H, W, "sb_y", "sa_y_stride"),
        ("ub", off_ub, cH, cW, "sb_u", "sa_uv_stride"),
        ("vb", off_vb, cH, cW, "sb_v", "sa_uv_stride"),
    ]
    def _submit_one(i: int) -> None:
        req = _attach_request(shm, layout, i)
        pid = int(req.mpv_pid)
        pair_k = int(req.pair_k)
        src_a_idx = int(req.src_a_idx)
        src_b_idx = int(req.src_b_idx)
        task_type = int(req.task_type)
        task_id   = int(req.task_id)
        # Pick src planes to fetch from mpv. CCSR reads raw YUV from
        # mpv VA (3 planes: ya/ua/va). Other split tasks (INTERP,
        # SR_INTERP) pull from cc_cache GPU views inside compute_proc,
        # so dma_proc does no mpv readv at all for them.
        from worker_3proc import TT_CCSR
        if task_type == TT_CCSR:
            planes_to_fetch = SRC_PLANES[:3]           # 3 (ya, ua, va)
        else:
            planes_to_fetch = []                       # cc_cache only
        plane_args = [(p[0], p[1], p[2], p[3],
                        int(getattr(req, p[4])), int(getattr(req, p[5])))
                       for p in planes_to_fetch]
        del req
        # Write wire-format header. The worker side reads this header
        # via rdma_proc, but on the host the compute_proc reads
        # SlotMeta directly — we still populate the header so the
        # protocol stays symmetric and any debug tool inspecting the
        # src bundle sees consistent metadata.
        hdr_addr = shm.addr + layout.slot_src_off(i)
        hdr = (ctypes.c_int64 * 16).from_address(hdr_addr)
        hdr[0]  = pair_k
        hdr[1]  = 0
        hdr[2]  = src_a_idx
        hdr[3]  = src_b_idx
        hdr[10] = task_type    # _HDR_TASK_TYPE
        hdr[11] = task_id      # _HDR_TASK_ID
        del hdr
        base = shm.addr + layout.slot_src_off(i)
        _t_readv0 = _t_dma.perf_counter() if (_dma_debug or _io_prof) else 0
        for _name, off, ph, pw, addr, stride in plane_args:
            try:
                _pvm_plane(pid=pid, local_addr=base + off, remote_addr=addr,
                            h=ph, w=pw, elem_size=2,
                            remote_stride=stride, write=False)
            except RuntimeError as ex:
                # Re-raise with task context so debugging can identify
                # WHICH task hit the stale address.
                raise RuntimeError(
                    f"{ex} [task_id={task_id} task_type={task_type} "
                    f"pair_k={pair_k} src_a={src_a_idx} src_b={src_b_idx} "
                    f"plane={_name} addr=0x{addr:x} h={ph} w={pw}]"
                ) from ex
        if _dma_debug or _io_prof:
            timing["readv_sum_ms"] += (
                _t_dma.perf_counter() - _t_readv0) * 1000
            timing["readv_n"] += 1
        # cc_cache references — read from RequestMeta and propagate to
        # SlotMeta so compute_proc sees them.
        # dst_cc_slot_2/3 carry the additional cc_cache slots for
        # INTERP's mid frames. -1 for mult=2 / non-INTERP.
        req2 = _attach_request(shm, layout, i)
        src_cc_a = int(req2.src_cc_slot_a)
        src_cc_b = int(req2.src_cc_slot_b)
        src_cc_f = int(req2.src_cc_field)
        dst_cc_s = int(req2.dst_cc_slot)
        dst_cc_f = int(req2.dst_cc_field)
        dst_cc_s2 = int(req2.dst_cc_slot_2)
        dst_cc_s3 = int(req2.dst_cc_slot_3)
        del req2
        with state_lock:
            meta = shm.slot_state(i)
            # Write SlotMeta fields BEFORE transitioning state.
            # compute_proc (different process) reads state lock-free
            # and then reads task_id/dst_cc_slot/etc — if state goes
            # RECV_DONE first, compute_proc can see stale fields from
            # the slot's previous use ("task_id=0 cc_cache unavailable"
            # / refcount underflow cascade).
            meta.n_recv += 1
            meta.pair_k = pair_k
            meta.src_a_idx = src_a_idx
            meta.src_b_idx = src_b_idx
            meta.task_type = task_type
            meta.task_id   = task_id
            meta.src_cc_slot_a = src_cc_a
            meta.src_cc_slot_b = src_cc_b
            meta.src_cc_field  = src_cc_f
            meta.dst_cc_slot   = dst_cc_s
            meta.dst_cc_field  = dst_cc_f
            meta.dst_cc_slot_2 = dst_cc_s2
            meta.dst_cc_slot_3 = dst_cc_s3
            meta.state = ST_RECV_DONE
            del meta
        ef_to_mgr.notify()
        counts["readv"] += 1
        if (_dma_debug or _io_prof) and counts["readv"] % 60 == 0:
            log(f"dma stat: readv_n={timing['readv_n']} "
                f"readv_avg={timing['readv_sum_ms']/max(1,timing['readv_n']):.2f}ms "
                f"wait_avg={timing['wait_sum_ms']/max(1,timing['wait_n']):.2f}ms "
                f"wait_n={timing['wait_n']}")
            if _io_prof:
                timing["readv_sum_ms"] = 0; timing["readv_n"] = 0
                timing["wait_sum_ms"] = 0; timing["wait_n"] = 0
            timing["readv_sum_ms"] = 0; timing["readv_n"] = 0
            timing["wait_sum_ms"] = 0;  timing["wait_n"] = 0

    def _writeback_one(i: int) -> None:
        """Three modes:
          - dst_cc_slot >= 0 (split-task): memmove slot.dst into the
            cc_cache[dst_cc_slot].<dst_cc_field> region, then mark
            FREE + ef_done. The compute_proc kernel wrote its output
            to slot.dst[0 .. field_size]; we ship those bytes over to
            the shared cache so downstream consumers can read.
          - phases_mask == 0, dst_cc_slot == -1: dispatcher will read
            shm dst itself (Grace UMA). dma_proc just signals ef_done
            and leaves state at DST_READY for the dispatcher to
            release on consume.
          - phases_mask != 0: classic worker-style symmetric writeback
            via process_vm_writev into mpv's dst frame memory.
        """
        req = _attach_request(shm, layout, i)
        pid = int(req.mpv_pid)
        phases_mask = int(req.phases_mask)
        dst_cc_slot  = int(req.dst_cc_slot)
        dst_cc_field = int(req.dst_cc_field)
        # Sentinel: dst_cc_slot >= 0 AND dst_cc_field == -2 → kernel
        # writes cc_cache directly (no dma_proc memcpy). Transition
        # the slot to ST_SEND_PENDING so this listener doesn't
        # re-fire ef_done on subsequent scans (compute_proc only
        # writes DST_READY). The dispatcher completion watcher
        # accepts ST_SEND_PENDING + ST_DST_READY + ST_FREE alike,
        # reads task_id, then releases.
        #
        # CCSR uses dst_cc_field == -2 (kernel wrote cc_cache for
        # rgb_padded + features) AND phases_mask=0x1 (writeback slot.dst
        # 4K to mpv VA). For CCSR we take this early return because the
        # memmove writeback to mpv VA happens mpv-side in _hmp_watcher
        # (same path as SR_* memmove mode).
        if dst_cc_slot >= 0 and dst_cc_field == -2:
            del req
            with state_lock:
                meta = shm.slot_state(i)
                meta.state = ST_SEND_PENDING
                meta.n_send += 1
                del meta
            ef_done.notify()
            counts["writev"] += 1
            if os.environ.get("DUAL_SPLIT_DEBUG", "0") == "1":
                log(f"writeback slot={i} sentinel → SEND_PENDING")
            return
        if dst_cc_slot >= 0 and dst_cc_field >= 0:
            if cc_shm is None:
                log(f"slot {i} requests cc_cache writeback (slot={dst_cc_slot}"
                    f" field={dst_cc_field}) but cc_cache not opened — "
                    f"falling back to free-only")
            else:
                field_name = _CC_FIELD_BY_INDEX[dst_cc_field]
                size = cc_shm.layout.field_size(field_name)
                src_addr = shm.addr + layout.slot_dst_off(i)
                dst_addr = cc_shm.field_addr(dst_cc_slot, field_name)
                ctypes.memmove(dst_addr, src_addr, size)
                cc_shm.set_field(dst_cc_slot, field_name)
            del req
            with state_lock:
                meta = shm.slot_state(i)
                meta.state = ST_FREE
                meta.n_send += 1
                del meta
            ef_done.notify()
            counts["writev"] += 1
            if counts["writev"] % 60 == 0:
                log(f"readv={counts['readv']} writev={counts['writev']} "
                    f"(cc_cache writeback)")
            return
        if phases_mask == 0:
            del req
            ef_done.notify()
            return
        # SR_* writeback: mpv-side ctypes.memmove in _hmp_watcher.
        # Same shm is mapped in both processes, so the dst is local
        # memcpy at ~30 GB/s, ~0.8 ms per SR task. dma_proc just
        # transitions state to SEND_PENDING and lets mpv do the actual
        # copy. _hmp_watcher reads req.dst_ya/ua/va and copies from
        # shm.slot.dst into mpv's dst frame.
        del req
        with state_lock:
            meta = shm.slot_state(i)
            meta.state = ST_SEND_PENDING
            meta.n_send += 1
            del meta
        ef_done.notify()
        counts["writev"] += 1
        if counts["writev"] % 60 == 0:
            log(f"readv={counts['readv']} writev={counts['writev']} "
                f"(SR_* writeback handled by mpv-side memmove)")

    def _listener(target_state: int, ef, handler, name: str):
        """Each transition into `target_state` is the dispatcher's
        (RECV_PENDING) or buffer_mgr's (DST_READY) responsibility; we
        are the sole consumer, so a single-threaded scan is race-free."""
        try:
            while not stopping.is_set():
                _t_wait0 = _t_dma.perf_counter() if _dma_debug else 0
                ef.wait()
                if _dma_debug and name == "submit":
                    timing["wait_sum_ms"] += (
                        _t_dma.perf_counter() - _t_wait0) * 1000
                    timing["wait_n"] += 1
                for i in range(layout.n_slots):
                    meta = shm.slot_state(i)
                    s = meta.state
                    del meta
                    # Same-machine CCSR transitions COMPUTING →
                    # MID_READY → DST_READY (compute_proc kernel writes
                    # cc_cache directly via dst_cc_field=-2 sentinel,
                    # then DST_READY for the 4K SR writeback). dma_proc
                    # only acts on DST_READY; forward MID_READY to
                    # _hmp_watcher via ef_done so it can fire
                    # task_stage_done(tid, 0) and unblock INTERP.
                    if is_mid_ready_state(s) and name == "writeback":
                        ef_done.notify()
                        continue
                    if s != target_state:
                        continue
                    try:
                        handler(i)
                    except Exception as ex:
                        log(f"{name} slot {i} failed: "
                            f"{type(ex).__name__}: {ex}")
                        import traceback; log(traceback.format_exc())
                        meta = shm.slot_state(i)
                        meta.state = ST_FREE
                        del meta
                        ef_done.notify()
        except Exception as ex:
            log(f"{name} fatal: {type(ex).__name__}: {ex}")
            import traceback; log(traceback.format_exc())

    submit_t = threading.Thread(
        target=_listener,
        args=(ST_RECV_PENDING, ef_request, _submit_one, "submit"),
        daemon=True, name="dma-submit")
    writeback_t = threading.Thread(
        target=_listener,
        args=(ST_DST_READY, ef_from_mgr, _writeback_one, "writeback"),
        daemon=True, name="dma-writeback")
    submit_t.start()
    writeback_t.start()
    log("dma_proc started (process_vm_readv/writev)")
    try:
        submit_t.join()
        writeback_t.join()
    except KeyboardInterrupt:
        stopping.set()


# ──────────────────────────────────────────────────────────────────────
# Orchestrator (thin subclass of MPPipelineBase)
# ──────────────────────────────────────────────────────────────────────


class HostMP(MPPipelineBase):
    """Host side: dma_proc owns mpv-process memory access via
    process_vm_readv/writev. Adds two eventfds beyond the base set:
      request : dispatcher → dma_proc  ("new RECV_PENDING slot")
      done    : dma_proc   → dispatcher ("slot transitioned to FREE")

    Optional cc_cache. When `cc_cache_layout` is provided, the
    orchestrator allocates a cuda-pinned shm region for CC outputs and
    hands its name + layout to dma_proc via env. Children treat it as
    a second shm region for split-task data flow.
    """

    EF_NAMES = MPPipelineBase._BASE_EF_NAMES + ["request", "done"]

    def __init__(self, *, cc_cache_layout: CCCacheLayout | None = None,
                 cc_cache_name: str | None = None, **kw):
        super().__init__(**kw)
        self.cc_cache_layout = cc_cache_layout
        self.cc_cache_name = cc_cache_name or (
            f"dgxspark_host_cc_cache_{os.getpid()}")
        self.cc_cache: CCCacheShm | None = None
        # cv to wake dispatcher when a slot transitions to FREE
        # (replacing a sleep-poll grain in dispatcher's claim_slot
        # retry loop). Notified by release_slot(); waited on by
        # wait_slot_free(). claim_slot's scan stays lock-free against
        # FILLING / RECV_PENDING etc.
        import threading as _th
        self._slot_cv = _th.Condition()

    def _create_shm(self) -> SlotRingShm:
        return SlotRingShm(self.layout, name=self.shm_name)

    def setup(self):
        super().setup()
        if self.cc_cache_layout is not None:
            self.cc_cache = CCCacheShm(
                self.cc_cache_layout, name=self.cc_cache_name)
            self.cc_cache.create()
            try:
                self.cc_cache.register_cuda_pinned()
            except Exception as ex:
                self._log(f"[hmp] cc_cache cuda pin failed: {ex}")
            self._log(f"[hmp] cc_cache: {self.cc_cache_layout.describe()}")

    def shutdown(self):
        if self.cc_cache is not None:
            try:
                self.cc_cache.close()
                self.cc_cache.unlink()
            except Exception:
                pass
            self.cc_cache = None
        super().shutdown()

    def _extra_child_config(self, cfg: dict, role: str) -> None:
        # compute_proc / buffer_mgr_proc build their own SlotRingLayout
        # from the IPC config; they need to know the size of the
        # RequestMeta prefix region so total_size matches our shm.
        cfg["WMP_EXTRA_PREFIX"] = str(self.layout.req_table_size_a)
        cfg["WMP_SPLIT_PH"] = str(self.layout.pH)
        cfg["WMP_SPLIT_PW"] = str(self.layout.pW)
        cfg["WMP_SPLIT_ENC_CH"] = str(self.layout.enc_ch)
        # cc_cache plumbing. dma_proc + compute_proc use these to
        # open / map the shared CC output region.
        if self.cc_cache_layout is not None:
            L = self.cc_cache_layout
            cfg["HMP_CC_CACHE_NAME"]    = self.cc_cache_name
            cfg["HMP_CC_CACHE_NSLOTS"]  = str(L.n_slots)
            cfg["HMP_CC_CACHE_H_SRC"]   = str(L.H_src)
            cfg["HMP_CC_CACHE_W_SRC"]   = str(L.W_src)
            cfg["HMP_CC_CACHE_H_OUT"]   = str(L.H_out)
            cfg["HMP_CC_CACHE_W_OUT"]   = str(L.W_out)
            cfg["HMP_CC_CACHE_PH"]      = str(L.pH)
            cfg["HMP_CC_CACHE_PW"]      = str(L.pW)
            cfg["HMP_CC_CACHE_ENC_CH"]  = str(L.enc_channels)
            cfg["HMP_CC_CACHE_SUB_H"]   = str(L.sub_h)
            cfg["HMP_CC_CACHE_SUB_W"]   = str(L.sub_w)

    def _post_setup(self):
        # Zero the RequestMeta table so dma_proc never reads stale fields.
        for i in range(self.layout.n_slots):
            req = _attach_request(self.shm, self.layout, i)
            ctypes.memset(ctypes.byref(req), 0, REQ_META_BYTES)
            # cc_cache fields default to -1 (unused); dma_proc treats
            # -1 as "skip cc_cache".
            req.src_cc_slot_a = CC_FIELD_NONE
            req.src_cc_slot_b = CC_FIELD_NONE
            req.src_cc_field  = CC_FIELD_NONE
            req.dst_cc_slot   = CC_FIELD_NONE
            req.dst_cc_field  = CC_FIELD_NONE
            # INTERP extra cc_cache slots default unused.
            req.dst_cc_slot_2 = CC_FIELD_NONE
            req.dst_cc_slot_3 = CC_FIELD_NONE
            del req

    def start(self):
        self._spawn("dma",      "host_3proc",   "dma_proc_main")
        self._spawn("mgr",      "worker_3proc", "buffer_mgr_proc_main")
        self._spawn("compute",  "worker_3proc", "compute_proc_main")

    # ── dispatcher-side helpers ──────────────────────────────────

    def claim_slot(self) -> int | None:
        """Find a FREE slot, mark it ST_FILLING (dispatcher will fill
        RequestMeta then call notify_submit() to transition to
        ST_RECV_PENDING). FILLING is invisible to submit_listener,
        which closes the race where submit_one would otherwise read a
        partially-filled RequestMeta if it was still inside an earlier
        scan-loop iteration."""
        for i in range(self.layout.n_slots):
            meta = self.shm.slot_state(i)
            if meta.state == ST_FREE:
                meta.state = ST_FILLING
                del meta
                return i
            del meta
        return None

    def release_slot(self, i: int) -> None:
        """Dispatcher calls this once it has fully consumed the result
        out of shm. Transitions DST_READY → FREE (or any state → FREE
        if called defensively). Wakes wait_slot_free() so dispatcher
        can claim without burning a 1ms sleep grain."""
        meta = self.shm.slot_state(i)
        meta.state = ST_FREE
        del meta
        with self._slot_cv:
            self._slot_cv.notify_all()

    def wait_slot_free(self, timeout: float = 0.005) -> None:
        """Block up to `timeout` seconds until release_slot is called.
        Dispatcher's claim retry path uses this instead of sleep(1ms)
        — typical wakeup latency drops from ~500us to ~10us."""
        with self._slot_cv:
            self._slot_cv.wait(timeout=timeout)

    def get_request(self, i: int) -> RequestMeta:
        return _attach_request(self.shm, self.layout, i)

    def commit_slot(self, i: int) -> None:
        """Transition the just-filled slot from ST_FILLING to
        ST_RECV_PENDING so submit_listener can pick it up. Caller
        MUST have finished writing the entire RequestMeta first;
        the state transition is the publication barrier."""
        meta = self.shm.slot_state(i)
        meta.state = ST_RECV_PENDING
        del meta

    def notify_submit(self):
        self.efs["request"].notify()

    def state(self, i: int) -> int:
        meta = self.shm.slot_state(i)
        s = int(meta.state)
        del meta
        return s

    @property
    def ef_done_fd(self) -> int:
        return self.efs["done"].fd

    def dst_plane_addrs(self, i: int, phase: int) -> tuple[int, int, int]:
        """Returns (y_addr, u_addr, v_addr) in shm for the dst bundle
        of slot i, phase 0 (src-passthrough) or 1 (interp). dispatcher
        uses these as ctypes.memmove sources when writing out the
        result to mpv frame memory."""
        base = self.shm.addr + self.layout.slot_dst_off(i)
        oH, oW = self.layout.oH, self.layout.oW
        ocH, ocW = self.layout.ocH, self.layout.ocW
        osz_y = oH * oW * 2
        osz_uv = ocH * ocW * 2
        if phase == 0:
            off_y = 0
            off_u = off_y + osz_y
            off_v = off_u + osz_uv
        else:
            off_y = osz_y + 2 * osz_uv
            off_u = off_y + osz_y
            off_v = off_u + osz_uv
        return base + off_y, base + off_u, base + off_v

    def memmove_dst_to_mpv(self, i: int) -> bool:
        """In-mpv-process memmove from slot.dst → mpv frame VA, for
        SR_* tasks. Returns True if a copy happened (req had dst_ya
        set), False otherwise.

        Same-process memmove on shared shm runs at ~30 GB/s vs the
        ~2.5 GB/s of cross-process process_vm_writev that this replaced.

        Source: shm.slot[i].dst (phase-a region — both SR_SRC and
        SR_INTERP land their output here).
        Dest: req.dst_ya/dst_ua/dst_va (mpv frame VA registered by the
        dispatcher via set_phase_dst).
        Stride: req.dst_y_stride_a / dst_uv_stride_a — if equal to
        row_bytes we do a single memmove per plane; else fall back to
        row-by-row.
        """
        req = self.get_request(i)
        dst_ya = int(req.dst_ya)
        if not dst_ya:
            del req
            return False
        dst_ua = int(req.dst_ua)
        dst_va = int(req.dst_va)
        dst_y_stride = int(req.dst_y_stride_a)
        dst_uv_stride = int(req.dst_uv_stride_a)
        del req
        src_y, src_u, src_v = self.dst_plane_addrs(i, 0)
        oH, oW = self.layout.oH, self.layout.oW
        ocH, ocW = self.layout.ocH, self.layout.ocW
        row_y = oW * 2
        row_uv = ocW * 2
        if dst_y_stride == row_y and dst_uv_stride == row_uv:
            ctypes.memmove(dst_ya, src_y, row_y * oH)
            ctypes.memmove(dst_ua, src_u, row_uv * ocH)
            ctypes.memmove(dst_va, src_v, row_uv * ocH)
        else:
            for r in range(oH):
                ctypes.memmove(dst_ya + r * dst_y_stride,
                                src_y + r * row_y, row_y)
            for r in range(ocH):
                ctypes.memmove(dst_ua + r * dst_uv_stride,
                                src_u + r * row_uv, row_uv)
                ctypes.memmove(dst_va + r * dst_uv_stride,
                                src_v + r * row_uv, row_uv)
        return True
