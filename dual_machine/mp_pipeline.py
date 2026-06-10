"""Shared MP pipeline scaffolding: layout, shm, eventfd, base orchestrator.

These are independent of the data-movement backend. worker_3proc.py
implements the RDMA backend (rdma_proc_main + WorkerMP) and the
compute_proc / buffer_mgr_proc that BOTH sides reuse. host_3proc.py
implements the DMA backend (dma_proc_main + HostMP).
"""
import atexit
import ctypes
import errno
import glob
import mmap
import os
import struct
import sys

# Set of shm names registered for atexit cleanup; deduped so multiple
# sessions in one mpv process don't register multiple handlers.
_atexit_registered_shm: set[str] = set()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _reap_stale_shm(log) -> None:
    """Unlink any /dev/shm/dgxspark_*_<pid>.dat where the PID is gone.

    mpv's vapoursynth filter teardown doesn't always reach our cleanup
    path (vsapi quirk; the cleanup_fn returned by make_dispatcher is
    never invoked on vf teardown). So a hard mpv exit / SIGKILL leaves
    the parent's shm leaking. Without periodic reaping, /dev/shm fills
    up (each session is ~225 MB + ~3 GB for cc_cache at 4K) and the
    next ftruncate looks fine but mmap pages SIGBUS on first access.

    Patterns cleaned:
      dgxspark_host_3p_<pid>.dat       — host SlotRingShm
      dgxspark_worker_3p_<pid>.dat     — worker SlotRingShm
      dgxspark_host_cc_cache_<pid>.dat — host cc_cache pool (large!)
    """
    import re
    pat = re.compile(r"^dgxspark_(?:host|worker)_(?:3p|cc_cache)_(\d+)\.dat$")
    reaped = 0
    for path in glob.glob("/dev/shm/dgxspark_*.dat"):
        m = pat.match(os.path.basename(path))
        if not m:
            continue
        pid = int(m.group(1))
        if pid == os.getpid():
            continue
        if _pid_alive(pid):
            continue
        try:
            os.unlink(path)
            reaped += 1
        except OSError:
            pass
    if reaped and log:
        log(f"[mp] reaped {reaped} stale /dev/shm/dgxspark_* file(s)")


def _install_atexit_unlink(shm, log) -> None:
    """Best-effort: unlink shm at interpreter exit even if our shutdown()
    isn't called. Idempotent (atexit runs once per registered closure)."""
    name = getattr(shm, "name", None)
    if not name or name in _atexit_registered_shm:
        return
    _atexit_registered_shm.add(name)

    def _cleanup():
        try:
            shm.close()
        except Exception: pass
        try:
            shm.unlink()
        except Exception: pass

    atexit.register(_cleanup)

class SlotMeta(ctypes.Structure):
    _fields_ = [
        # Atomic uint32 state — read/written via ctypes; transitions are
        # serialised by the producer/consumer pair on each edge (rdma
        # only writes FREE→RECV_PENDING and RECV_DONE→FREE-after-send;
        # compute only writes RECV_DONE→COMPUTING and DST_READY).
        ("state",      ctypes.c_uint32),
        ("generation", ctypes.c_uint32),
        # Pair metadata copied out of the wire header by rdma/dma_proc
        # so compute_proc doesn't need to re-read the (volatile) src_buf
        # header during dispatch.
        ("pair_k",     ctypes.c_int64),
        ("src_a_idx",  ctypes.c_int64),
        ("src_b_idx",  ctypes.c_int64),
        # Counters for debugging — written only by the owning process.
        ("n_recv",     ctypes.c_uint64),
        ("n_compute",  ctypes.c_uint64),
        ("n_send",     ctypes.c_uint64),
        # Split-task fields. task_id round-trips back to the queue
        # manager so it can mark the DAG node done.
        ("task_type",  ctypes.c_uint32),
        # WRITE_WITH_IMM: host's slot index for this request, copied
        # here by rdma_proc on RECV_SRC. send_handler uses it to look
        # up host_dst_remote[host_slot] for the response WRITE.
        # Required because host's guest_mp slot count may not match
        # the worker's layout.n_slots — worker's local slot index
        # (M) is not guaranteed to equal host's slot index (K).
        ("host_slot",  ctypes.c_uint32),
        ("task_id",    ctypes.c_uint64),
        # cc_cache references. compute_proc reads these to find the
        # cuda-pinned cc_cache region for its inputs/outputs.
        ("src_cc_slot_a", ctypes.c_int32),
        ("src_cc_slot_b", ctypes.c_int32),
        ("src_cc_field",  ctypes.c_int32),
        ("dst_cc_slot",   ctypes.c_int32),
        ("dst_cc_field",  ctypes.c_int32),
        # N-stage producer: which stage_idx is the next pending WRITE
        # for this slot. send_handler reads on ST_MID_READY. 0..14
        # valid; 15 reserved for "final" (handled via ST_DST_READY
        # anyway).
        ("pending_stage_idx", ctypes.c_uint32),
        # Per-slot MID/FULL byte range cached here so send_handler
        # (worker) / dma_proc (host) don't need to recompute from
        # task_type at every poll. Set by compute_proc on claim.
        # mid_size == 0 ⇒ single-stage task (no MID send / no MID_READY
        # transition). For INTERP mult ≥ 4 the worker derives
        # intermediate stage offsets (k * frame_sz) on the fly from
        # DUAL_INTERP_MULT; mid_off/mid_size still describe stage 0
        # only.
        ("mid_off",   ctypes.c_uint32),
        ("mid_size",  ctypes.c_uint32),
        ("full_off",  ctypes.c_uint32),
        ("full_size", ctypes.c_uint32),
        # Single INTERP task produces N-1 interp frames per source pair
        # via dual flownet. The worker reads DUAL_INTERP_MULT env at
        # startup; per-task timestep fields kept for binary compat.
        ("interp_timestep_num",   ctypes.c_uint32),
        ("interp_timestep_denom", ctypes.c_uint32),
        # INTERP writes mult-1 frames into cc_cache (dense pack).
        # dst_cc_slot = frame 0 dest; dst_cc_slot_2 = frame 1 (mult ≥ 3);
        # dst_cc_slot_3 = frame 2 (mult == 4). Unused slots == -1.
        ("dst_cc_slot_2", ctypes.c_int32),
        ("dst_cc_slot_3", ctypes.c_int32),
        # 8+24+24+4+4+8+24+16+8+8 = 128 B used; no trailing pad needed.
    ]


SLOT_META_BYTES = ctypes.sizeof(SlotMeta)
assert SLOT_META_BYTES == 128, f"SlotMeta is {SLOT_META_BYTES}B, expected 128"


# Task type wire values. MUST match queue_mgr.TaskType. Duplicated here
# so the worker subprocess doesn't need to import the queue manager.
# CCSR inlines the SR_SRC pass, so standalone SR_SRC is never dispatched.
TT_INTERP      = 3
TT_SR_INTERP   = 4
TT_CCSR        = 5
# mult ≥ 3 produces multiple SR_INTERP tasks per pair, distinguished
# by TaskNode.output_phase (1 = primary → dst_i, 2/3 = secondary →
# dst_i2/dst_i3). Wire format identical, so worker uses the same
# _run_sr_interp; only host-side dispatch differs.

TT_NAMES = {
    TT_INTERP:      "INTERP",
    TT_SR_INTERP:   "SR_INTERP",
    TT_CCSR:        "CCSR",
}


# Slot state transitions:
#
#   FREE → RECV_PENDING        [rdma_proc: post_recv issued]
#   RECV_PENDING → RECV_DONE   [rdma_proc: NIC delivered]
#   RECV_DONE → COMPUTING      [compute_proc: claimed for compute]
#   COMPUTING → MID_READY      [compute_proc: mid stage done, NIC-visible]
#   MID_READY → DST_READY      [compute_proc: full stage done, NIC-visible]
#   DST_READY → SEND_PENDING   [rdma_proc: post_send issued]
#   SEND_PENDING → FREE        [rdma_proc: NIC drained]
#
# MID_READY is skipped by single-stage tasks (mult=2 INTERP, SR_INTERP)
# — they go COMPUTING → DST_READY directly. Multi-stage tasks (CCSR,
# mult ≥ 3 INTERP) fire MID_READY after their mid byte range is
# NIC-visible so the host queue_mgr can unblock downstream tasks that
# gate on mid (e.g. INTERP gates on its two CCSR sources' rgb_padded +
# features, not their SR yuv).
ST_FREE         = 0
ST_RECV_PENDING = 1
ST_RECV_DONE    = 2
ST_COMPUTING    = 3
ST_DST_READY    = 4
ST_SEND_PENDING = 5
# Intermediate state used by host_mp.claim_slot. Dispatcher transitions
# FREE → FILLING during claim_slot, fills RequestMeta, then
# notify_submit transitions FILLING → RECV_PENDING. This closes the
# race where submit_listener (in a wake from an earlier notify)
# scanned a just-claimed slot and called submit_one() while the
# dispatcher was still mid-fill, causing partial RequestMeta reads
# (NEW dst_cc_slot, OLD task_id from the previous task that held
# this slot, and ultimately a duplicate _run_cc invocation).
ST_FILLING      = 6
# Multi-phase producer states.
#   ST_MID_READY  — compute_proc set this after stage A done + NIC-
#                   visible; send_handler will post the mid SEND.
#                   On host's dma_proc, transition fires on receiving
#                   the mid SEND completion.
#   ST_MID_SENT   — send_handler set after posting the mid SEND.
#                   later stages still running on compute_proc; once
#                   finished the slot transitions to ST_DST_READY.
ST_MID_READY    = 7    # Use as ST_MID_READY + stage_idx (0..MID_READY_MAX_STAGE-1)
ST_MID_READY_MAX_STAGE = 8   # supports up to 8 mid stages (mult ≤ 9)
ST_MID_SENT     = 16   # moved above MID_READY range to avoid collision
def is_mid_ready_state(st: int) -> bool:
    """True when state encodes a MID_READY transition. Use this instead
    of `st == ST_MID_READY` to handle multi-stage encoding."""
    return ST_MID_READY <= st < ST_MID_READY + ST_MID_READY_MAX_STAGE
def mid_ready_stage(st: int) -> int:
    """Decode stage_idx from a MID_READY state value."""
    return st - ST_MID_READY

ST_NAMES = {
    ST_FREE: "FREE", ST_RECV_PENDING: "RECV_PENDING",
    ST_RECV_DONE: "RECV_DONE", ST_COMPUTING: "COMPUTING",
    ST_DST_READY: "DST_READY", ST_SEND_PENDING: "SEND_PENDING",
    ST_FILLING: "FILLING", ST_MID_READY: "MID_READY",
    ST_MID_SENT: "MID_SENT",
}


_PAGE = 4096


def _align_up(n, a=_PAGE):
    return (n + a - 1) // a * a


class SlotRingLayout:
    """Computes byte offsets for the slot ring. Pure layout — no I/O,
    no allocation. Both processes construct one of these with identical
    args and they'll agree on the byte map.

    Shm map:
      [state_table : n_slots × 64B]    page-aligned
      [slot 0      : src_aligned + dst_aligned]
      [slot 1      : ...]
      ...
    """

    def __init__(self, *, n_slots: int, H: int, W: int,
                 scale: int, sub_w: int, sub_h: int,
                 dst_sub_w: int = 1, dst_sub_h: int = 1,
                 extra_prefix_bytes_aligned: int = 0,
                 pH: int = 1088, pW: int = 1920, enc_ch: int = 4):
        """`extra_prefix_bytes_aligned` reserves a page-aligned region
        between the state table and slot 0. Subclasses (e.g.
        HostSlotRingLayout) use it to place per-slot metadata that
        rdma_proc / dma_proc want to read alongside the state table.

        slot.dst always carries CCSR mid-out (rgb_padded +
        rife_features) alongside the 4K SR output. pH/pW/enc_ch must
        match between host and guest — they cross the wire in the
        handshake tensor (rife_pH/rife_pW) so both sides compute the
        same slot.src and slot.dst sizes.

        sub_w/sub_h describe the SRC chroma subsampling; the OUTPUT
        goes to mpv which always wants 4:2:0, so dst_sub_w/dst_sub_h
        default to (1, 1). Callers can pass different src and dst
        when input ≠ 4:2:0 (e.g. 4:4:4 source keeps full chroma
        through CC + RIFE, then downsamples once at the SR output
        pass)."""
        # Defer the bundle-layout helpers to runtime (they pull torch).
        from rdma_transport import zc_src_bundle_layout, dst_bundle_layout
        cH, cW = H >> sub_h, W >> sub_w
        oH, oW = H * scale, W * scale
        ocH, ocW = oH >> dst_sub_h, oW >> dst_sub_w
        self.n_slots = n_slots
        self.H, self.W = H, W
        self.cH, self.cW = cH, cW
        self.oH, self.oW = oH, oW
        self.ocH, self.ocW = ocH, ocW
        self.scale = scale
        self.sub_w, self.sub_h = sub_w, sub_h
        self.dst_sub_w, self.dst_sub_h = dst_sub_w, dst_sub_h
        self.pH, self.pW, self.enc_ch = pH, pW, enc_ch

        self.src_size, self.src_layout = zc_src_bundle_layout(
            H, W, cH, cW, pH=pH, pW=pW, enc_ch=enc_ch)
        # max_mult=4 reserves room for the F9 cycle's largest INTERP
        # dense-pack (frame_sz = 3 * H * W * 2 bytes fp16 RGB). Pads
        # dst_size up so no_sr (output dim = source dim, smaller
        # named-region total) still admits mult ≥ 3.
        _MAX_INTERP_MULT = 4
        _interp_overlay = _MAX_INTERP_MULT * 3 * H * W * 2
        self.dst_size, self.dst_layout = dst_bundle_layout(
            oH, oW, ocH, ocW, pH=pH, pW=pW, enc_ch=enc_ch,
            interp_overlay_bytes=_interp_overlay)

        self.src_size_a = _align_up(self.src_size)
        self.dst_size_a = _align_up(self.dst_size)
        self.slot_stride = self.src_size_a + self.dst_size_a

        self.state_table_off = 0
        self.state_table_size_a = _align_up(n_slots * SLOT_META_BYTES)
        self.extra_prefix_off = self.state_table_size_a
        self.extra_prefix_size_a = extra_prefix_bytes_aligned
        self.slots_off = self.state_table_size_a + self.extra_prefix_size_a
        self.total_size = self.slots_off + n_slots * self.slot_stride

    def slot_state_off(self, i: int) -> int:
        return self.state_table_off + i * SLOT_META_BYTES

    def slot_src_off(self, i: int) -> int:
        return self.slots_off + i * self.slot_stride

    def slot_dst_off(self, i: int) -> int:
        return self.slot_src_off(i) + self.src_size_a

    def describe(self) -> str:
        return (
            f"SlotRingLayout(n_slots={self.n_slots} "
            f"src={self.src_size/1e6:.2f}MB[a={self.src_size_a/1e6:.2f}MB] "
            f"dst={self.dst_size/1e6:.2f}MB[a={self.dst_size_a/1e6:.2f}MB] "
            f"total={self.total_size/1e6:.2f}MB)"
        )

    def task_dst_ranges(self, task_type: int, *, interp_mult: int = 2,
                        rgb_interp_size: int = 0,
                        mid_delivery: bool = True
                        ) -> tuple[int, int, int, int]:
        """Per-task split of slot.dst into FIRST-MID + FINAL byte ranges.
        Returns (mid_off, mid_size, full_off, full_size).
            mid_*  = stage 0 (first early-delivery WRITE). 0/0 if none.
            full_* = final WRITE (terminal, fires task_done downstream).
        For INTERP with mult ≥ 4 the worker derives intermediate stages
        (1..mult-3) on the fly from dense-pack offsets (k * frame_sz);
        only the two anchors (stage 0 and final) are stored in meta.

        Every range covers only the bytes the host actually reads for
        that task type — slot.dst is sized for the worst case (named
        regions vs the mult=4 INTERP overlay), so a blanket
        0..dst_size WRITE used to ship up to ~40 MB of dead bytes per
        task over the wire.

        mid_delivery=False (DUAL_MID_DELIVERY off, the current
        default) collapses everything into one terminal WRITE: the
        union of the mid + final regions, since the single response
        carries both.
        """
        off = {name: o for (name, o, *_) in self.dst_layout}
        sz  = {name: n for (name, _, n, *_) in self.dst_layout}
        if task_type == TT_CCSR:
            # MID = rgb_padded + rife_features (contiguous tail).
            # FULL = yao + uao + vao  (contiguous head).
            mid_off  = off["rgb_padded"]
            mid_size = sz["rgb_padded"] + sz["rife_features"]
            full_off  = 0
            full_size = off["rgb_padded"]
            if not mid_delivery:
                # Single terminal WRITE carries SR output + mid-out.
                # no_interp (mult=1) CCSR skips the rgb_padded /
                # rife_features writes entirely (no consumer), so the
                # union shrinks to the SR planes.
                if interp_mult <= 1:
                    return (0, 0, 0, off["rgb_padded"])
                return (0, 0, 0, mid_off + mid_size)
            return (mid_off, mid_size, full_off, full_size)
        if task_type == TT_INTERP and interp_mult >= 2:
            # Dense-pack: frame k at slot.dst[k * frame_sz]. Stage 0 =
            # frame 0; final = frame (mult-2). The layout is decoupled
            # from CCSR named regions so CCSR field changes don't move
            # INTERP frame anchors.
            if rgb_interp_size <= 0:
                raise ValueError(
                    "task_dst_ranges(TT_INTERP) requires "
                    "rgb_interp_size; got "
                    f"{rgb_interp_size}")
            need = interp_mult * rgb_interp_size
            assert need <= self.dst_size, (
                f"INTERP mult={interp_mult} needs {need} B but "
                f"slot.dst is {self.dst_size} B "
                f"(rgb_interp_size={rgb_interp_size})")
            if not mid_delivery or interp_mult == 2:
                # One terminal WRITE with all (mult-1) frames. mult=2
                # has a single frame and no mid stage by construction.
                return (0, 0, 0, (interp_mult - 1) * rgb_interp_size)
            final_off = (interp_mult - 2) * rgb_interp_size
            return (0, rgb_interp_size, final_off, rgb_interp_size)
        # Single-stage default: TT_SR_INTERP (and INTERP mult=1, which
        # never dispatches). Host reads only yao/uao/vao.
        # mid_size=0 means producer skips WRITE_MID / MID_DONE entirely.
        return (0, 0, 0, off["rgb_padded"])


# ──────────────────────────────────────────────────────────────────────
# Shm-backed buffer
# ──────────────────────────────────────────────────────────────────────

class SlotRingShm:
    """A page-aligned anonymous shm region sized for a SlotRingLayout.

    Created via /dev/shm/<name>.dat. Both processes open the same path.
    The creator MUST call .create_truncate_and_map(); attachers use
    .open_and_map().

    After mapping, the caller may:
      - call .register_cuda_pinned() to pin the region for cuda DMA
        (each process must do this independently in its own ctx).
      - call .register_rdma_mr(pd, access) to register the region as
        an RDMA MR (rdma_proc only).

    Slot src/dst views are obtained via .slot_src_view(i) and
    .slot_dst_view(i) which return memoryview slices.
    """

    def __init__(self, layout: SlotRingLayout, name: str):
        self.layout = layout
        self.name = name
        self.path = f"/dev/shm/{name}.dat"
        self.fd: int | None = None
        self.mm: mmap.mmap | None = None
        self.addr: int | None = None  # raw mmap address (for ctypes / pyverbs)
        self._cuda_registered = False
        self._mr = None

    # ── lifecycle ──────────────────────────────────────────────────

    def create_truncate_and_map(self):
        """Used by the creator process (rdma_proc). Truncates any stale
        shm with the same name. After return self.addr is a valid raw
        pointer covering layout.total_size bytes."""
        if os.path.exists(self.path):
            os.unlink(self.path)
        self.fd = os.open(self.path,
                          os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        os.ftruncate(self.fd, self.layout.total_size)
        self._map()
        # Zero-fault every page so we don't pay first-touch cost during
        # the hot path. cuda registration requires resident pages too.
        ctypes.memset(self.addr, 0, self.layout.total_size)

    def open_and_map(self):
        """Used by attacher processes (compute_proc, buffer_mgr_proc)."""
        self.fd = os.open(self.path, os.O_RDWR)
        st = os.fstat(self.fd)
        if st.st_size != self.layout.total_size:
            raise RuntimeError(
                f"shm {self.path} size {st.st_size} ≠ "
                f"expected {self.layout.total_size}")
        self._map()

    def _map(self):
        self.mm = mmap.mmap(self.fd, self.layout.total_size,
                             flags=mmap.MAP_SHARED,
                             prot=mmap.PROT_READ | mmap.PROT_WRITE)
        # Raw address — needed for pyverbs ibv_reg_mr and cudaHostRegister.
        # mmap.mmap doesn't directly expose its base address in Python,
        # but ctypes lets us grab it via a c_char buffer cast.
        # We use the data_ptr-style trick: take ctypes.addressof on a
        # 0-len cast.
        self.addr = ctypes.addressof(
            ctypes.c_char.from_buffer(self.mm))

    def close(self):
        if self._cuda_registered:
            self._cuda_unregister()
        if self._mr is not None:
            try:
                self._mr.close()
            except Exception:
                pass  # MR may already be torn down on shutdown
            self._mr = None
        if self.mm is not None:
            # ctypes views (SlotMeta.from_buffer) hold a reference to
            # the mmap; drop them before close. Best-effort: callers
            # who hold lingering views can ignore the BufferError.
            try:
                self.mm.close()
            except BufferError:
                # Force-close by NULL'ing exposed buffers — Python's
                # mmap will refuse close() if any ctypes-typed buffer
                # was created via from_buffer. Skip cleanup; process
                # exit will reap the mapping.
                pass
            self.mm = None
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None

    def unlink(self):
        """Creator should call this on shutdown."""
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass

    # ── cuda registration ─────────────────────────────────────────

    def register_cuda_pinned(self, *, mapped: bool = True):
        """cudaHostRegister(addr, size, cudaHostRegisterMapped |
        cudaHostRegisterPortable). Each process MUST call this in its
        own cuda context for the GPU to see the region. On Grace SoC
        the device pointer equals the host pointer (unified memory)
        so callers can treat self.addr as a valid GPU address after
        this returns."""
        libcudart = ctypes.CDLL("libcudart.so")
        libcudart.cudaHostRegister.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        libcudart.cudaHostRegister.restype = ctypes.c_int
        # cudaHostRegisterPortable=0x01, cudaHostRegisterMapped=0x02
        flags = 0x01 | (0x02 if mapped else 0)
        rc = libcudart.cudaHostRegister(
            ctypes.c_void_p(self.addr),
            ctypes.c_size_t(self.layout.total_size), flags)
        if rc != 0:
            # cudaErrorHostMemoryAlreadyRegistered=712 — fine if another
            # registration in *this* process already grabbed the range.
            if rc == 712:
                pass
            else:
                raise RuntimeError(
                    f"cudaHostRegister(addr={hex(self.addr)}, "
                    f"size={self.layout.total_size}, flags={flags}) "
                    f"failed: cudaError={rc}")
        self._cuda_registered = True
        self._libcudart = libcudart

    def _cuda_unregister(self):
        if not self._cuda_registered:
            return
        self._libcudart.cudaHostUnregister.argtypes = [ctypes.c_void_p]
        self._libcudart.cudaHostUnregister.restype = ctypes.c_int
        rc = self._libcudart.cudaHostUnregister(
            ctypes.c_void_p(self.addr))
        if rc not in (0, 712):  # already unregistered is fine
            sys.stderr.write(
                f"[SlotRingShm] cudaHostUnregister returned {rc}\n")
        self._cuda_registered = False

    def cuda_device_ptr(self) -> int:
        """Returns the device-side pointer for self.addr via
        cudaHostGetDevicePointer. On Grace this equals self.addr."""
        if not self._cuda_registered:
            raise RuntimeError("register_cuda_pinned() first")
        dev_ptr = ctypes.c_void_p(0)
        self._libcudart.cudaHostGetDevicePointer.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
            ctypes.c_uint]
        self._libcudart.cudaHostGetDevicePointer.restype = ctypes.c_int
        rc = self._libcudart.cudaHostGetDevicePointer(
            ctypes.byref(dev_ptr), ctypes.c_void_p(self.addr), 0)
        if rc != 0:
            raise RuntimeError(
                f"cudaHostGetDevicePointer failed: {rc}")
        return dev_ptr.value or 0

    # ── RDMA MR registration ───────────────────────────────────────

    def register_rdma_mr(self, pd, access: int):
        """Register the entire shm region as an RDMA MR. Only rdma_proc
        should call this. Returns the MR (also stored as self._mr)."""
        sys.path.insert(0, "/usr/lib/python3/dist-packages")
        from pyverbs.mr import MR
        self._mr = MR(pd, self.layout.total_size, access,
                      address=self.addr)
        return self._mr

    @property
    def mr(self):
        return self._mr

    # ── slot accessors ────────────────────────────────────────────

    def slot_state(self, i: int) -> SlotMeta:
        """Returns a SlotMeta ctypes view at the right offset.
        Reads/writes go straight to shm."""
        off = self.layout.slot_state_off(i)
        return SlotMeta.from_buffer(self.mm, off)

    def slot_src_addr(self, i: int) -> int:
        return self.addr + self.layout.slot_src_off(i)

    def slot_dst_addr(self, i: int) -> int:
        return self.addr + self.layout.slot_dst_off(i)

    def slot_src_view_bytes(self, i: int) -> memoryview:
        off = self.layout.slot_src_off(i)
        return memoryview(self.mm)[off:off + self.layout.src_size]

    def slot_dst_view_bytes(self, i: int) -> memoryview:
        off = self.layout.slot_dst_off(i)
        return memoryview(self.mm)[off:off + self.layout.dst_size]


# ──────────────────────────────────────────────────────────────────────
# eventfd-based signalling
# ──────────────────────────────────────────────────────────────────────

# We use Linux eventfd for cross-process wakeups. Each "channel" is one
# eventfd inherited by all processes. The recipient reads (which blocks
# until counter > 0) and then scans shm slot states to find work.
#
# Why eventfd vs semaphores: eventfd integrates with select/poll/epoll
# (we'll want this when multiple input channels need to be merged), is
# a single FD that fork() naturally inherits, and read() releases the
# GIL — the recipient process can block efficiently without burning CPU.
#
# Layout:
#   ef_rdma_to_mgr   : rdma_proc → buffer_mgr  ("a slot transitioned;
#                                                 mgr should scan")
#   ef_mgr_to_compute: buffer_mgr → compute_proc ("a RECV_DONE slot is
#                                                 ready to compute")
#   ef_compute_to_mgr: compute_proc → buffer_mgr ("a DST_READY slot is
#                                                 ready to send")
#   ef_mgr_to_rdma   : buffer_mgr → rdma_proc  ("post_send slot N OR
#                                                 something to repost")
#
# Each is a separate eventfd. Counter is in EFD_SEMAPHORE mode so each
# write(1) corresponds to exactly one read(1) — counts events 1:1.

class EventFdChannel:
    """Single-direction eventfd. Owner-side creates; attachers inherit
    the FD via subprocess.Popen(pass_fds=...)."""

    def __init__(self, name: str, fd: int | None = None):
        self.name = name
        if fd is None:
            # EFD_SEMAPHORE = 1 (each read decrements by 1, blocks if 0)
            # EFD_CLOEXEC   = 0x80000 (close on exec — but we'll override
            #                 with pass_fds when spawning subprocesses)
            self.fd = os.eventfd(0, os.EFD_SEMAPHORE)
        else:
            self.fd = fd

    def notify(self, n: int = 1):
        """Sender side: bump the counter by n."""
        os.eventfd_write(self.fd, n)

    def wait(self) -> int:
        """Recipient side: block until counter > 0, then decrement by 1
        (because EFD_SEMAPHORE). Returns 1."""
        return os.eventfd_read(self.fd)

    def close(self):
        if self.fd >= 0:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = -1


# ──────────────────────────────────────────────────────────────────────
# Process entry points
# ──────────────────────────────────────────────────────────────────────
#
# Each "process" runs as a child of the orchestrator. They communicate
# via the shm slot ring + 4 eventfds:
#
#   ef_rdma_to_mgr   : rdma_proc → buffer_mgr  ("slot transition happened")
#   ef_mgr_to_compute: buffer_mgr → compute_proc ("RECV_DONE slot ready")
#   ef_compute_to_mgr: compute_proc → buffer_mgr ("DST_READY slot ready")
#   ef_mgr_to_rdma   : buffer_mgr → rdma_proc  ("post_send a DST_READY slot")
#
# Each entry function loads the shm + eventfds from argv (passed by the
# orchestrator via subprocess.Popen), then runs its loop until killed.


def _setup_logging(role: str):
    """Light-weight per-process logger. Prepends [role pid] tag.

    Also wires PR_SET_PDEATHSIG so the OS sends SIGTERM to this child
    if the parent (main worker.py) dies — pkill -9 -f worker.py on
    the parent alone would otherwise leave us orphaned (which we
    observed: stale rdma_proc holding port 29900 across runs).
    """
    import signal
    import time
    PR_SET_PDEATHSIG = 1
    libc = ctypes.CDLL("libc.so.6")
    libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    pid = os.getpid()
    def log(msg):
        sys.stderr.write(
            f"[{role} pid={pid} {time.strftime('%H:%M:%S')}] {msg}\n")
        sys.stderr.flush()
    return log


def _parse_layout_from_env() -> "SlotRingLayout":
    return SlotRingLayout(
        n_slots=int(os.environ["WMP_N_SLOTS"]),
        H=int(os.environ["WMP_H"]),
        W=int(os.environ["WMP_W"]),
        scale=int(os.environ["WMP_SCALE"]),
        sub_w=int(os.environ["WMP_SUB_W"]),
        sub_h=int(os.environ["WMP_SUB_H"]),
        dst_sub_w=int(os.environ.get("WMP_DST_SUB_W", "1")),
        dst_sub_h=int(os.environ.get("WMP_DST_SUB_H", "1")),
        extra_prefix_bytes_aligned=int(
            os.environ.get("WMP_EXTRA_PREFIX", "0")),
        pH=int(os.environ.get("WMP_SPLIT_PH", "1088")),
        pW=int(os.environ.get("WMP_SPLIT_PW", "1920")),
        enc_ch=int(os.environ.get("WMP_SPLIT_ENC_CH", "4")),
    )


def _ef_from_env(name: str) -> "EventFdChannel":
    return EventFdChannel(name, fd=int(os.environ[f"WMP_EF_{name.upper()}"]))


# ──────────────────────────────────────────────────────────────────────
# Real process entry points
# ──────────────────────────────────────────────────────────────────────

# wr_id encoding: low 16 bits = slot index, high 16 bits = op type.
_WR3_RECV_SRC = 0x10000  # worker recv'd src bundle
_WR3_SEND_DST = 0x20000  # worker sent dst bundle (FULL range for multi-stage)
# Worker emits an extra mid SEND for multi-stage producers (CCSR,
# INTERP mult ≥ 3). Single-stage tasks (INTERP mult=2, SR_INTERP)
# only emit _WR3_SEND_DST.
_WR3_SEND_MID = 0x30000  # worker sent dst MID range
_WR3_RECV_MID = 0x40000  # host recv'd dst MID range (matches _WR3_SEND_MID)


class MPPipelineBase:
    """Shared scaffolding for the 3-proc pipeline used on BOTH sides:

      worker   = rdma_proc  + buffer_mgr_proc + compute_proc
      host     = dma_proc   + buffer_mgr_proc + compute_proc

    Both share buffer_mgr_proc_main and compute_proc_main (they only
    talk to shm + eventfds, not the NIC). The only thing that differs
    between worker and host is the data-movement process (rdma vs
    dma) and the set of eventfds wired up around it.

    Subclasses override:
      EF_NAMES   — the set of eventfds for this side
      start()    — what to spawn

    Common WMP_* env vars (layout + compute config + eventfd fds) are
    populated by _child_env; subclasses extend via _extra_child_env().
    """

    # Eventfds that BOTH sides always need.
    _BASE_EF_NAMES = ["rdma_to_mgr", "mgr_to_rdma",
                       "mgr_to_compute", "compute_to_mgr",
                       "compute_ready"]
    # Subclasses extend this.
    EF_NAMES: list[str] = list(_BASE_EF_NAMES)

    def __init__(self, *, layout: SlotRingLayout, shm_name: str,
                 log=None,
                 variant: str = "", rife_model: str = "4.26",
                 matrix_s: str = "709", color_range: str = "limited",
                 chroma_mode: str = "bicubic", bits: int = 10,
                 downsample_pre: int = 1,
                 weights_dir: str | None = None):
        self.layout = layout
        self.shm_name = shm_name
        self.variant = variant
        self.rife_model = rife_model
        self.matrix_s = matrix_s
        self.color_range = color_range
        self.chroma_mode = chroma_mode
        self.bits = bits
        self.downsample_pre = downsample_pre
        self.weights_dir = weights_dir
        self._log = log or (lambda m: sys.stderr.write(m + "\n"))
        self.shm: SlotRingShm | None = None
        self.efs: dict[str, EventFdChannel] = {}
        self.procs: dict[str, "subprocess.Popen"] = {}

    def setup(self):
        """Create shm (in parent — children open it) + eventfds.
        Subclasses may override to do extra per-slot initialization
        (e.g. host adds a RequestMeta table)."""
        # Reap stale /dev/shm files from crashed mpv runs. The shm
        # naming pattern is `dgxspark_<role>_3p_<pid>.dat`; if the PID
        # is gone we can safely unlink. Without this each crash leaks
        # ~225 MB; after a dozen sessions tmpfs is full and ftruncate
        # silently succeeds but mmap pages SIGBUS on first access.
        _reap_stale_shm(self._log)
        self.shm = self._create_shm()
        self.shm.create_truncate_and_map()
        for i in range(self.layout.n_slots):
            meta = self.shm.slot_state(i)
            meta.state = ST_FREE
            meta.generation = 0
            del meta
        for name in self.EF_NAMES:
            self.efs[name] = EventFdChannel(name)
        self._post_setup()
        # Register an atexit hook + SIGTERM/SIGINT handler so the shm
        # file is unlinked even if our own shutdown() isn't called
        # (mpv's vapoursynth filter teardown doesn't reach into our
        # cleanup_fn — long-standing mpv embedding quirk).
        _install_atexit_unlink(self.shm, self._log)
        self._log(f"[mp] setup ok. {self.layout.describe()}")

    def _create_shm(self) -> SlotRingShm:
        """Subclasses with extended layouts override this to return a
        SlotRingShm built around their custom layout."""
        return SlotRingShm(self.layout, name=self.shm_name)

    def _post_setup(self):
        """Hook for subclasses to zero/initialize extra shm regions
        after the base setup has mapped + zeroed the shm."""
        pass

    def _child_config(self, role: str) -> dict:
        """Build the IPC config dict for a spawned subprocess. Written
        to the child's stdin as JSON; the child's entry reads it before
        running. Only cluster config that genuinely comes from outside
        (RDMA dev/port/gid, NCCL master addr/port, worker_host, etc.)
        stays as inherited env on the Popen side — everything that
        flows orchestrator→subprocess goes through this dict so the
        IPC channel is explicit instead of polluting os.environ."""
        cfg = {
            "WMP_SHM_NAME":     self.shm_name,
            "WMP_N_SLOTS":      str(self.layout.n_slots),
            "WMP_H":            str(self.layout.H),
            "WMP_W":            str(self.layout.W),
            "WMP_SCALE":        str(self.layout.scale),
            "WMP_SUB_W":        str(self.layout.sub_w),
            "WMP_SUB_H":        str(self.layout.sub_h),
            "WMP_DST_SUB_W":    str(getattr(self.layout, "dst_sub_w", 1)),
            "WMP_DST_SUB_H":    str(getattr(self.layout, "dst_sub_h", 1)),
            "WMP_SPLIT_PH":     str(getattr(self.layout, "pH", 1088)),
            "WMP_SPLIT_PW":     str(getattr(self.layout, "pW", 1920)),
            "WMP_SPLIT_ENC_CH": str(getattr(self.layout, "enc_ch", 4)),
            "WMP_ROLE":         role,
            "WMP_VARIANT":      self.variant,
            "WMP_RIFE_MODEL":   self.rife_model,
            "WMP_MATRIX_S":     self.matrix_s,
            "WMP_COLOR_RANGE":  self.color_range,
            "WMP_CHROMA_MODE":  self.chroma_mode,
            "WMP_BITS":         str(self.bits),
            "WMP_DOWNSAMPLE_PRE": str(self.downsample_pre),
        }
        for name in self.EF_NAMES:
            cfg[f"WMP_EF_{name.upper()}"] = str(self.efs[name].fd)
        if self.weights_dir is not None:
            cfg["WMP_WEIGHTS_DIR"] = self.weights_dir
        self._extra_child_config(cfg, role)
        return cfg

    def _extra_child_config(self, cfg: dict, role: str) -> None:
        """Hook for subclasses to inject side-specific config."""
        pass

    def _spawn(self, role: str, module: str, entry_name: str):
        import subprocess, json
        # Entry reads JSON from stdin first, populates os.environ for
        # the rest of its (env-reading) code, then runs the body. This
        # keeps IPC explicit (the contract is "what's in the JSON
        # payload"), not "what's in the parent's environ when Popen
        # ran".
        bootstrap = (
            "import sys, json, os; "
            "cfg = json.load(sys.stdin); "
            "os.environ.update(cfg); "
            f"from {module} import {entry_name}; {entry_name}()"
        )
        # mpv's embedded interpreter sets sys.executable to /usr/bin/python3
        # (the system python that originally embedded libpython), not the
        # conda env's python3.12. Spawning the system python with the env's
        # PYTHONHOME would load 3.12 stdlib into 3.10/3.11 libpython and
        # crash on _ctypes (undefined symbol _PyErr_SetLocaleString). Use
        # the env's python3 explicitly — sys.prefix is set correctly by
        # PYTHONHOME and points at the conda env.
        py_bin = os.path.join(sys.prefix, "bin", "python3")
        if not os.path.exists(py_bin):
            py_bin = sys.executable  # fallback for non-conda environments
        cmd = [py_bin, "-B", "-c", bootstrap]
        fds = [ef.fd for ef in self.efs.values()]
        # Inherit the parent's env so external cluster config (NCCL_*,
        # DUAL_RDMA_*, MASTER_*) is visible; the IPC config goes via
        # stdin and only that updates os.environ inside the child.
        pythonpath = (
            f"/usr/lib/python3/dist-packages:"
            f"{os.path.dirname(os.path.abspath(__file__))}")
        child_env = {**os.environ, "PYTHONPATH": pythonpath}
        p = subprocess.Popen(cmd, env=child_env, pass_fds=fds,
                              stdin=subprocess.PIPE)
        p.stdin.write(json.dumps(self._child_config(role)).encode())
        p.stdin.close()
        self.procs[role] = p
        self._log(f"[mp] spawned {role}: pid={p.pid} "
                   f"({module}.{entry_name})")

    def start(self):
        raise NotImplementedError

    def wait_compute_ready(self, timeout: float = 60.0) -> None:
        import select
        ef = self.efs["compute_ready"]
        rlist, _, _ = select.select([ef.fd], [], [], timeout)
        if not rlist:
            raise RuntimeError(
                f"compute_proc did not signal ready within {timeout}s")
        os.eventfd_read(ef.fd)

    def join(self, timeout: float | None = None) -> int:
        """Wait until ANY child exits, then return its exit code.

        Old behavior was a sequential `p.wait()` over each child, which
        blocked on the first long-lived child while a sibling crash went
        undetected — see the host_child_crash robustness scenario. Now
        any exit (rdma TimeoutError, compute crash, mgr OOM) immediately
        returns; the caller is expected to invoke shutdown() to reap the
        rest. timeout=None waits forever; on timeout returns 0.
        """
        import time as _t
        deadline = None if timeout is None else _t.monotonic() + timeout
        while True:
            for role, p in self.procs.items():
                ec = p.poll()
                if ec is not None:
                    self._log(f"[mp] {role} exited with code={ec}")
                    return ec
            if deadline is not None and _t.monotonic() >= deadline:
                return 0
            _t.sleep(0.1)

    def shutdown(self):
        # Idempotent: cross-thread shutdown (e.g. watchdog + main finally)
        # used to crash with "NoneType has no attribute 'unlink'" because
        # self.shm = None at the end races with a re-entry. Guard each
        # resource individually.
        #
        # Two-pass: SIGKILL all children first, then poll-wait in
        # parallel. Sequential `kill + wait(timeout=2)` per child
        # added up to 6 s for three children, which blew the 3 s
        # end-to-end user-wait budget on session-end → re-init races.
        # After SIGKILL the OS reaps in milliseconds; a tight poll
        # loop is plenty.
        import time as _t
        for role, p in list(self.procs.items()):
            try:
                if p.poll() is None:
                    self._log(f"[mp] killing {role}")
                    p.kill()
            except Exception:
                pass
        deadline = _t.monotonic() + 1.0
        for role, p in list(self.procs.items()):
            while p.poll() is None and _t.monotonic() < deadline:
                _t.sleep(0.02)
        for ef in list(self.efs.values()):
            try: ef.close()
            except Exception: pass
        if self.shm is not None:
            try: self.shm.close()
            except Exception: pass
            try: self.shm.unlink()
            except Exception: pass
            self.shm = None
