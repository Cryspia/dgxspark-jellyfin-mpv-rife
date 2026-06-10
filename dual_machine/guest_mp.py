"""GuestMP — mpv-side RDMA wrapper for dispatching split-task jobs
to the remote guest worker (worker_3proc).

Mirrors host_mp's slot-pool API so the native_dispatcher can treat
host_mp and guest_mp uniformly when pulling tasks from queue_mgr:

  slot = guest_mp.claim_slot(blocking=True)
  # (caller copies cc_cache payload into send_buf at payload_offset)
  guest_mp.fill_request(slot, node, dst_y_va=..., dst_u_va=..., ...)
  guest_mp.notify_submit(slot)
  ... (later, completion thread handles writeback + task_done)

Same-process advantage over host_3proc: guest_mp runs IN the mpv
process (instantiated by native_dispatcher), so completion writeback
is plain `ctypes.memmove` into FrameVA addresses — no process_vm_writev
syscall needed.

Wire protocol:
  - One RC QP between mpv-side and guest worker.py.
  - Each task carries the 128-B ZC_HEADER + task-specific payload
    in slot.src. Header indices follow rdma_transport._HDR_*;
    task_type / task_id steer the worker's compute_proc to the right
    kernel.
  - Response comes back in slot.dst:
      INTERP → sr_yuv (Y + U + V planes at the start of dst,
                       memmove into host cc_cache.sr_yuv[dst_cc_slot])
      SR_*   → 4K int16 YUV (existing yao/uao/vao layout, memmove
                              to mpv frame VA)

CC is host-exclusive (mpv VA accessible only via process_vm_readv)
and never reaches guest_mp.

Lifecycle / threading:
  - Main dispatcher thread calls claim_slot → fill_request →
    notify_submit. fill_request packs into pre-allocated cuda-pinned
    send buffer; notify_submit issues an RDMA SEND.
  - A daemon CQ poll thread drains completions, calls queue_mgr.task_done,
    handles writeback (cc_cache memmove for INTERP; mpv VA memmove for
    SR_*), and releases the slot back to the free list.

Memory budget per slot:
  - send buf (pinned MR): split INTERP needs 64 MB (rgb_padded+features×2);
                          SR_* needs 6 MB. Sized to the INTERP max.
  - recv buf (pinned MR): SR_* 4K output is ~50 MB; INTERP's 6 MB sr_yuv
                          response fits in the first 6 MB.
  - Total: ~114 MB / slot. 4 slots = ~456 MB on mpv side.
  Buffers use pageable host backing (pyverbs pins for DMA via ibv_reg_mr);
  cuda-pinned isn't needed since the cc_cache → send_buf copy is plain
  CPU memcpy from cuda-pinned cc_cache.
"""
from __future__ import annotations

import ctypes
import os
import struct
import sys
import threading
import time

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, Callable

sys.path.insert(0, "/usr/lib/python3/dist-packages")

from rdma_transport import (
    RDMAContext, RDMAStagingBuffer, RDMAChannel,
    _ZC_HEADER_INT64,
    _HDR_PAIR_K, _HDR_PHASE, _HDR_SRC_A, _HDR_SRC_B,
    _HDR_TASK_TYPE, _HDR_TASK_ID,
    _HDR_CC_SLOT_A, _HDR_CC_SLOT_B, _HDR_CC_FIELD,
    _HDR_DST_CC_SLOT_2, _HDR_HOST_SLOT,
)


# Task type codes (must match worker_3proc.TT_*).
TT_INTERP      = 3
TT_SR_INTERP   = 4
TT_CCSR        = 5
# Secondary SR_INTERP (mult ≥ 3) uses TT_SR_INTERP on the wire too;
# only host-side dispatch picks dst_cc_slot_2 / dst_i2 via
# TaskNode.output_phase before the slot is sent.


# wr_id encoding for guest_mp's CQ poll loop. Low 16 bits = slot id.
# Reverse direction (worker → host) is exclusively WRITE_WITH_IMM —
# slot/stage carried in imm_data, see _imm_encode below. Recv WRs are
# all dummy placeholders against a tiny MR.
_WR_SEND_DONE = 0x10000      # host → worker SRC bundle send completion
_WR_RECV_IMM  = 0x40000      # worker → host WRITE_WITH_IMM completion
_WR_MASK_SLOT = 0xFFFF

# imm_data encoding: bits 0-15 = slot, bits 16-19 = stage_code (4 bits).
# stage_code semantics:
#   0..14  → mid stage_idx k (== stage_code). mult=N uses 0..N-3.
#   15     → final (terminal task write, fires task_done downstream).
# 4 bits gives mult ≤ 16 headroom — well past any practical RIFE
# temporal multiplier.
_STAGE_FINAL = 15

def _imm_encode(slot: int, stage_code: int) -> int:
    return (slot & 0xFFFF) | ((stage_code & 0xF) << 16)

def _imm_decode(imm: int) -> tuple[int, int]:
    return (imm & 0xFFFF, (imm >> 16) & 0xF)


# Slot lifecycle states (just for diagnostics — actual state lives
# in free_slots / inflight bookkeeping).
ST_FREE      = 0
ST_FILLED    = 1   # dispatcher packed data; not yet posted
ST_IN_FLIGHT = 2   # RDMA send posted, waiting for response
ST_RESULT    = 3   # response arrived; CQ thread is writing back


@dataclass
class _SlotTask:
    """Per-slot record of the task currently in flight. Populated by
    fill_request, consumed by the CQ thread on response."""
    task_id:        int = 0
    task_type:      int = 0
    dst_cc_slot:    int = -1   # INTERP — rgb_interp frame 1 destination
    dst_cc_slot_2:  int = -1   # INTERP mult≥3 — frame 2 destination
    dst_cc_slot_3:  int = -1   # INTERP mult=4 — frame 3 destination
    dst_y_va:       int = 0    # for SR_* — mpv VA
    dst_u_va:       int = 0
    dst_v_va:       int = 0
    dst_y_stride:   int = 0
    dst_uv_stride:  int = 0


class GuestMP:
    """RDMA-backed slot pool for split-task dispatch to the guest worker."""

    def __init__(self, *,
                 H: int, W: int,
                 scale: int = 2, sub_w: int = 1, sub_h: int = 1,
                 dst_sub_w: int = 1, dst_sub_h: int = 1,
                 pH: int, pW: int, enc_ch: int,
                 n_slots: int = 4,
                 rdma_dev: str = "rocep1s0f0",
                 rdma_port: int = 1,
                 rdma_gid_index: int = 3,
                 peer_ip: str,
                 peer_handshake_port: int,
                 cc_cache_writeback_fn: Optional[
                     Callable[[int, "memoryview"], None]] = None,
                 task_done_fn: Optional[Callable[[int], None]] = None,
                 task_fail_fn: Optional[Callable[[int, str], None]] = None,
                 task_stage_done_fn: Optional[
                     Callable[[int, int], None]] = None,
                 log=lambda m: None):
        """
        H, W                  source resolution (1080×1920 default)
        pH, pW, enc_ch        RIFE-padded dims + encode channels — must
                              equal the worker's values; sized so both
                              ends compute identical slot.src / slot.dst.
        n_slots               RDMA send/recv ring depth (4 by default)
        peer_ip               guest worker IP
        peer_handshake_port   TCP port for RDMA QP exchange
        cc_cache_writeback_fn callback (dst_cc_slot, recv_buf_view) → None
                              invoked when an INTERP response arrives; the
                              callback memmoves the sr_yuv bytes into the
                              right cc_cache.sr_yuv slot.
        task_done_fn          callback (task_id,) → None; calls
                              queue_mgr.task_done() so dependents become
                              ready. Called AFTER cc_cache_writeback / mpv
                              VA writeback completes.
        """
        self.H, self.W = H, W
        self.scale = scale
        self.sub_w, self.sub_h = sub_w, sub_h
        self.dst_sub_w, self.dst_sub_h = dst_sub_w, dst_sub_h
        self.cH = H >> sub_h
        self.cW = W >> sub_w
        self.oH = H * scale
        self.oW = W * scale
        self.ocH = self.oH >> dst_sub_h
        self.ocW = self.oW >> dst_sub_w
        self.n_slots = n_slots
        self.peer_ip = peer_ip
        self.peer_handshake_port = peer_handshake_port
        self.log = log
        self.cc_cache_writeback_fn = cc_cache_writeback_fn
        self.task_done_fn = task_done_fn
        # (task_id, reason) → None; the dispatcher wires this to
        # queue_mgr.task_done(ok=False) so a dead-worker / QP-error
        # slot doesn't leave its task RUNNING forever.
        self.task_fail_fn = task_fail_fn
        # Set by register_cc_cache(); None = zero-copy sends unavailable.
        self._cc_mr = None
        self.cc_lkey: "int | None" = None
        # Optional callback (task_id, stage_idx) → None. Fires when
        # the mid SEND for a two-stage producer arrives. Lets queue_mgr
        # unblock downstream INTERP before the FULL stage completes.
        # None ⇒ skip the mid path.
        self.task_stage_done_fn = task_stage_done_fn
        # Mid-delivery gating env. When set both worker and host post
        # the split sends/recvs (early delivery of mid stages).
        # Default ON (the designed pipeline behaviour; A/B-verified
        # bit-clean and slightly faster) — native_dispatcher snapshots
        # the value into the env before the handshake and ships it to
        # the worker, so both sides always agree. Full rationale at
        # worker_3proc's twin flag.
        self._mid_delivery_enabled = os.environ.get(
            "DUAL_MID_DELIVERY", "1") == "1"

        # Per-slot dst layout: 4K SR output (yao/uao/vao) plus CCSR
        # mid-out (rgb_padded + rife_features).
        from rdma_transport import dst_bundle_layout, zc_src_bundle_layout
        self._pH = pH
        self._pW = pW
        self._enc_ch = enc_ch
        # Mirror SlotRingLayout's INTERP-overlay padding so host and
        # guest agree on dst_size (the value crosses the wire as part
        # of the shm sizing handshake). max_mult=4 = the largest value
        # the F9 cycle can produce.
        _MAX_INTERP_MULT = 4
        _interp_overlay = _MAX_INTERP_MULT * 3 * H * W * 2
        self.dst_size, self.dst_layout = dst_bundle_layout(
            self.oH, self.oW, self.ocH, self.ocW,
            pH=pH, pW=pW, enc_ch=enc_ch,
            interp_overlay_bytes=_interp_overlay)
        self.src_size, self.src_layout = zc_src_bundle_layout(
            H, W, self.cH, self.cW,
            pH=pH, pW=pW, enc_ch=enc_ch)

        # Byte offsets within slot.src/dst for split-task layouts
        # (must match worker_3proc.compute_proc's split_*_views).
        self._hdr_bytes = _ZC_HEADER_INT64 * 8
        self._rgb_padded_bytes  = 1 * 3        * self._pH * self._pW * 2
        self._rife_features_bytes = 1 * self._enc_ch * self._pH * self._pW * 2
        # Processing dims (= H/downsample_pre); for SR_INTERP input
        # (rgb_interp ferry) and for sizing the RGB writeback.
        self._downsample_pre = int(os.environ.get("DUAL_DOWNSAMPLE_PRE", "1"))
        self._proc_h = H // self._downsample_pre
        self._proc_w = W // self._downsample_pre
        self._sr_y_bytes  = H * W * 2
        self._sr_uv_bytes = self.cH * self.cW * 2

        # INTERP dense-pack frame size (one rgb_interp tensor:
        # 3 × proc² × fp16). Frame k sits at slot.dst[k * frame_sz];
        # see _handle_mid_recv / _handle_recv. SR_* writes yao/uao/vao
        # via self._sr_dst_offs below.
        self._interp_dst_rgb_bytes = 3 * self._proc_h * self._proc_w * 2
        # SR_* dst comes from dst_layout (yao, uao, vao).
        self._sr_dst_offs = {}
        for name, off, nbytes, shape, dtype in self.dst_layout:
            self._sr_dst_offs[name] = (off, nbytes, shape, dtype)

        self._interp_mult = int(os.environ.get("DUAL_INTERP_MULT", "2"))
        if self._interp_mult not in (1, 2, 3, 4):
            self._interp_mult = 2
        # Assert dense-pack INTERP fits in slot.dst (only mult>=2 needs
        # this; mult=1 is no_interp so INTERP tasks aren't scheduled).
        if self._interp_mult >= 2:
            _interp_total = self._interp_mult * self._interp_dst_rgb_bytes
            assert _interp_total <= self.dst_size, (
                f"INTERP mult={self._interp_mult} needs "
                f"{_interp_total} B but slot.dst is {self.dst_size} B "
                f"(proc_h={self._proc_h}, proc_w={self._proc_w}); "
                f"reduce DUAL_INTERP_MULT or use a smaller source")

        # Per-task SEND lengths (host → worker). slot.src is sized for
        # the worst case (the INTERP overlay), but each task type only
        # fills a prefix of it; an RDMA SEND shorter than the worker's
        # pre-posted fixed-size recv is legal, and the worker routes by
        # the header's task_type, not byte_len. Sending the live bytes
        # only cuts CCSR / SR_INTERP wire traffic to roughly a quarter.
        #   CCSR:      header + ya/ua/va (the single source frame).
        #   INTERP:    header + 2 × (rgb_padded + rife_features) — the
        #              full overlay, no saving available.
        #   SR_INTERP: header + rgb_interp at proc dims.
        _src_off = {n: off for n, off, *_ in self.src_layout}
        _src_sz  = {n: sz for n, _, sz, *_ in self.src_layout}
        self._task_send_len = {
            TT_CCSR: _src_off["va"] + _src_sz["va"],
            TT_INTERP: (self._hdr_bytes
                        + 2 * (self._rgb_padded_bytes
                               + self._rife_features_bytes)),
            TT_SR_INTERP: self._hdr_bytes + self._interp_dst_rgb_bytes,
        }

        # ── RDMA setup ──────────────────────────────────────────────
        log(f"[guest_mp] opening RDMA context dev={rdma_dev}")
        self.ctx = RDMAContext(dev_name=rdma_dev, port=rdma_port,
                                gid_index=rdma_gid_index,
                                max_cqe=max(128, 8 * n_slots))

        # Per-slot send + recv buffers (pageable, ibv-pinned for DMA).
        # See class docstring "Memory budget per slot" for rationale.
        self.send_bufs = []
        self.recv_bufs = []
        for s in range(n_slots):
            self.send_bufs.append(RDMAStagingBuffer(self.ctx, self.src_size))
            self.recv_bufs.append(RDMAStagingBuffer(self.ctx, self.dst_size))

        # WRITE_WITH_IMM unified protocol: worker RDMA WRITEs the dst
        # bundle directly into our recv_bufs[i] at known offsets with
        # imm = encode(slot, stage). No SEND/RECV recv-queue ordering
        # involved → multi-slot mixed-task safe. Pack per-slot
        # (addr, rkey) into the handshake extra payload so the worker
        # knows where to write.
        # Wire format: u32 n_slots, then n_slots × (u64 addr, u32 rkey).
        _extra = struct.pack("!I", n_slots)
        for s in range(n_slots):
            _extra += struct.pack("!QI", self.recv_bufs[s].addr,
                                   self.recv_bufs[s].rkey)

        # QP — client (mpv side) connects to guest worker's server.
        log(f"[guest_mp] connecting to {peer_ip}:{peer_handshake_port}")
        self.ch = RDMAChannel(self.ctx, is_server=False,
                               peer_ip=peer_ip,
                               peer_port=peer_handshake_port,
                               max_wr=4 * n_slots,
                               extra_payload=_extra)
        log(f"[guest_mp] RDMA QP RTS")

        # Dummy recv pool. WRITE_WITH_IMM consumes one recv WR per IMM
        # event on this side; the WR's local SGE is unused (data lands
        # at the remote-addr the worker chose, which IS our
        # recv_bufs[slot] + offset). Pre-post against a tiny dummy MR
        # and refill on each completion. Pool size = 2 × n_slots + 2
        # (two-stage tasks have mid + full in flight, plus slack).
        self._dummy_recv_buf = RDMAStagingBuffer(self.ctx, 16)
        self._mid_imm_pool_size = 2 * n_slots + 2
        for _ in range(self._mid_imm_pool_size):
            self.ch.post_recv_at(self._dummy_recv_buf.addr, 16,
                                  self._dummy_recv_buf.lkey,
                                  wr_id=_WR_RECV_IMM)

        # ── slot bookkeeping ────────────────────────────────────────
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._free = deque(range(n_slots))
        self._state = [ST_FREE] * n_slots
        self._task = [_SlotTask() for _ in range(n_slots)]
        self._closed = False
        # Serializes the recv-post-then-send-post pair in notify_submit
        # so the response for our K-th post_send lands in the K-th
        # post_recv buffer (slot K's recv_buf). See note in notify_submit.
        self._io_lock = threading.Lock()
        # Do NOT pre-post recvs at construction time. We post recv right
        # before each post_send (under io_lock) so the recv-WR order
        # in the QP matches the send-WR order — guaranteeing host slot K's
        # response lands in slot K's recv buffer. Mirrors
        # HostPipelineHybridRDMA._send_loop's pattern (rdma_transport.py).

        # CQ poll daemon
        self._cq_thread = threading.Thread(
            target=self._cq_loop, daemon=True, name="guest_mp-cq")
        self._cq_thread.start()

        log(f"[guest_mp] ready: {n_slots} slots, "
            f"send={self.src_size/1e6:.1f}MB recv={self.dst_size/1e6:.1f}MB")

    # ── slot pool API (mirrors host_mp) ────────────────────────────

    def release_slot(self, slot: int) -> None:
        """Return slot directly to the free pool (e.g. when fill_request
        fails). For successful submits the CQ thread releases the slot
        after writeback."""
        with self._cv:
            self._state[slot] = ST_FREE
            self._free.append(slot)
            self._cv.notify()

    def claim_slot(self, *, blocking: bool = True,
                    timeout: Optional[float] = None) -> Optional[int]:
        with self._cv:
            if not self._free and not blocking:
                return None
            deadline = (time.monotonic() + timeout) if timeout else None
            while not self._free:
                if self._closed:
                    return None
                wait_for = (deadline - time.monotonic()) if deadline else None
                if wait_for is not None and wait_for <= 0:
                    return None
                self._cv.wait(timeout=wait_for)
            slot = self._free.popleft()
            self._state[slot] = ST_FILLED
            return slot

    def fill_request(self, slot: int, node,
                     dst_y_va: int = 0, dst_u_va: int = 0, dst_v_va: int = 0,
                     dst_y_stride: int = 0, dst_uv_stride: int = 0) -> None:
        """Pack the wire header for `node` into send_bufs[slot] and
        record dst metadata for the CQ completion thread. NOTE: the
        dispatcher is responsible for copying the per-task payload
        (rgb_padded+features for INTERP, sr_yuv for SR_*) into the
        send buffer at offset get_send_payload_offset() *before*
        calling notify_submit. See get_send_buffer_addr().

        dst_*_va / dst_*_stride apply only to SR_* tasks (4K YUV
        writeback into mpv frame memory). For INTERP, the completion
        thread routes the response into cc_cache via the writeback
        callback registered at construction time; pass 0 here.
        """
        tt = int(node.type)  # TaskType is IntEnum

        hdr_buf = (ctypes.c_int64 * _ZC_HEADER_INT64).from_address(
            self.send_bufs[slot].addr)
        hdr_buf[_HDR_PAIR_K]    = node.frame_idx
        hdr_buf[_HDR_PHASE]     = 0
        hdr_buf[_HDR_SRC_A]     = node.frame_idx
        hdr_buf[_HDR_SRC_B]     = node.frame_idx + 1
        hdr_buf[_HDR_TASK_TYPE] = tt
        hdr_buf[_HDR_TASK_ID]   = node.id
        hdr_buf[_HDR_CC_SLOT_A] = node.payload.get("src_cc_slot_a", -1)
        hdr_buf[_HDR_CC_SLOT_B] = node.payload.get("src_cc_slot_b", -1)
        hdr_buf[_HDR_CC_FIELD]  = -1  # split-mode: payload is inline
        # WRITE_WITH_IMM: tell the worker which host slot this request
        # belongs to. Worker echoes this in imm_data so the response
        # writes land in our recv_bufs[slot] even when host n_slots !=
        # worker n_slots (slot indices not paired by order).
        hdr_buf[_HDR_HOST_SLOT] = slot
        # For INTERP this carries the second cc_cache slot for frame 2
        # (mult≥3); -1 / 0 for mult=2 single-frame. Other task types
        # ignore it.
        hdr_buf[_HDR_DST_CC_SLOT_2] = int(
            node.payload.get("dst_cc_slot_2", -1))

        st = self._task[slot]
        st.task_id       = node.id
        st.task_type     = tt
        st.dst_cc_slot   = node.payload.get("dst_cc_slot", -1)
        st.dst_cc_slot_2 = node.payload.get("dst_cc_slot_2", -1)
        st.dst_cc_slot_3 = node.payload.get("dst_cc_slot_3", -1)
        st.dst_y_va      = dst_y_va
        st.dst_u_va      = dst_u_va
        st.dst_v_va      = dst_v_va
        st.dst_y_stride  = dst_y_stride
        st.dst_uv_stride = dst_uv_stride

    def register_cc_cache(self, addr: int, length: int) -> bool:
        """Register the host cc_cache shm region as an MR so INTERP /
        SR_INTERP payloads can be gathered straight out of it by the
        NIC (notify_submit with payload_sges) instead of the dispatch
        thread memmoving ~58 MB per INTERP through the CPU's slow
        (~5 GB/s) cudaHostRegister'd read path. Returns True on
        success; on failure callers keep using the memmove path."""
        try:
            self._cc_mr = self.ctx.register_region(addr, length)
            self.cc_lkey = self._cc_mr.lkey
            self.log(f"[guest_mp] cc_cache region registered for "
                      f"zero-copy sends ({length/1e6:.0f} MB, "
                      f"lkey={self.cc_lkey})")
            return True
        except Exception as e:
            self._cc_mr = None
            self.cc_lkey = None
            self.log(f"[guest_mp] cc_cache MR registration failed "
                      f"({type(e).__name__}: {e}) — falling back to "
                      f"staged sends")
            return False

    def get_send_buffer_addr(self, slot: int) -> int:
        """Return the GPU/CPU address of the slot's send buffer.
        Header is at off 0..128. Caller writes payload starting at
        off 128 according to the task type's wire layout."""
        return self.send_bufs[slot].addr

    def get_send_payload_offset(self) -> int:
        """Byte offset within send_buf where the per-task payload
        begins (i.e. just past the 128-B header)."""
        return self._hdr_bytes

    def notify_submit(self, slot: int,
                      payload_sges: "list | None" = None) -> None:
        """post_send the slot's SRC bundle (host → worker direction).
        Worker → host direction is exclusively WRITE_WITH_IMM:
        worker RDMA WRITEs the dst bundle straight into recv_bufs[slot]
        at the right offset, with imm = encode(slot, stage). The dummy
        recv WR pool catches the IMM events; no per-slot recv posting
        here.

        Two payload modes:
          payload_sges=None — classic staged path: the dispatcher
            already memmoved the payload into send_buf behind the
            header; SEND length is per-task (see _task_send_len).
          payload_sges=[(addr, len, lkey), ...] — zero-copy path: the
            wire message is the gather of [header (send_buf, 128 B)]
            + the given SGEs (typically cc_cache fields registered via
            register_cc_cache). Wire bytes are identical to the staged
            layout, so the worker can't tell the difference.

        Either way a SEND shorter than the worker's pre-posted
        src_size recv is legal, and the worker routes by the header's
        task_type."""
        with self._cv:
            assert self._state[slot] == ST_FILLED, (
                f"notify_submit on slot {slot} but state={self._state[slot]}")
            self._state[slot] = ST_IN_FLIGHT
            send_len = self._task_send_len.get(
                self._task[slot].task_type, self.src_size)
        # WRITE_WITH_IMM unified protocol: worker writes directly into
        # recv_bufs[slot] at the correct offset; the dummy-recv pool
        # pre-posted at __init__ catches IMM events. notify_submit
        # only needs to post the SRC SEND (host → worker direction).
        with self._io_lock:
            if payload_sges is None:
                self.ch.post_send(self.send_bufs[slot], send_len,
                                   wr_id=_WR_SEND_DONE | slot)
            else:
                sges = [(self.send_bufs[slot].addr, self._hdr_bytes,
                         self.send_bufs[slot].lkey)]
                sges.extend(payload_sges)
                self.ch.post_send_sgl(sges, wr_id=_WR_SEND_DONE | slot)

    # ── CQ poll thread + writeback ─────────────────────────────────

    def _cq_loop(self):
        """Drain CQ. On SEND completion: nothing to do (slot stays
        IN_FLIGHT until matching RECV). On RECV completion: parse
        response, do writeback (cc_cache for INTERP; mpv VA for SR_*),
        invoke task_done callback, re-post recv, return slot to free.

        Mirrors RDMAChannel.poll_cq_blocking's `ctx.cq.poll(n) → (n,wcs)`
        contract, but as a non-blocking spin so we react to self._closed.

        Idle backoff: while tasks are in flight, completions arrive
        every few ms, so the tight 100 µs poll keeps latency low. With
        nothing in flight (pause, single-mode fallback, idle player),
        the same spin needlessly burns a core inside the mpv process —
        back off to 1 ms after ~2 ms of consecutive empties; the first
        completion resets to the tight cadence."""
        idle_polls = 0
        while True:
            if self._closed:
                return
            try:
                n, wcs = self.ctx.cq.poll(8)
            except Exception as exc:
                self.log(f"[guest_mp-cq] poll failed: {exc}")
                time.sleep(0.001)
                continue
            if n == 0:
                idle_polls += 1
                time.sleep(0.0001 if idle_polls < 20 else 0.001)
                continue
            idle_polls = 0
            for wc in wcs[:n]:
                if wc.status != 0:
                    # A failed SEND (QP error / flush) means the slot's
                    # task will never produce a response — recover the
                    # slot and report the task failed so queue_mgr
                    # doesn't leave it RUNNING forever (the old code
                    # just logged; 4 such errors leaked every slot and
                    # wedged the guest dispatcher). RECV_IMM flushes
                    # carry no slot of their own.
                    self.log(f"[guest_mp-cq] wc.status={wc.status} "
                              f"wr_id={hex(wc.wr_id)}")
                    op = wc.wr_id & ~_WR_MASK_SLOT
                    if op == _WR_SEND_DONE:
                        self._fail_slot(int(wc.wr_id & _WR_MASK_SLOT),
                                        "rdma send error "
                                        f"status={wc.status}")
                    continue
                op = wc.wr_id & ~_WR_MASK_SLOT
                if op == _WR_SEND_DONE:
                    continue
                if op == _WR_RECV_IMM:
                    # WRITE_WITH_IMM landed in recv_bufs[slot] at the
                    # offset the worker chose. Decode (slot, stage_code)
                    # from imm_data and refill the dummy-recv pool.
                    # stage_code < _STAGE_FINAL ⇒ a mid stage (stage_idx
                    # == stage_code, 0..N-3 for mult=N). stage_code ==
                    # _STAGE_FINAL ⇒ terminal write → task_done flows.
                    slot, stage_code = _imm_decode(int(wc.imm_data))
                    self.ch.post_recv_at(self._dummy_recv_buf.addr, 16,
                                          self._dummy_recv_buf.lkey,
                                          wr_id=_WR_RECV_IMM)
                    if stage_code == _STAGE_FINAL:
                        self._handle_recv(slot)
                    else:
                        self._handle_mid_recv(slot, stage_code)
                    continue
                self.log(f"[guest_mp-cq] unknown wr_id {hex(wc.wr_id)}")

    def _handle_mid_recv(self, slot: int, stage_idx: int) -> None:
        """A mid-stage WRITE_WITH_IMM landed. Memmove the stage's bytes
        into host cc_cache, then fire task_stage_done so dependents
        can be queued before the next stage / final lands.

        Mid-stage producers and their per-stage interpretation:
          - CCSR (stage 0 only): mid = rgb_padded + rife_features →
            cc_cache[dst_cc_slot].
          - INTERP mult ≥ 3 (stages 0..mult-3): stage k = frame k+1
            at slot.dst[k*frame_sz : (k+1)*frame_sz] (dense pack) →
            cc_cache[dst_cc_slot_{k+1}].rgb_interp. SR_INTERP phase
            k+1 gates on this stage.

        Does NOT release the slot or call task_done — the final WRITE
        still has to land first. _handle_recv handles the final."""
        task = self._task[slot]
        recv_addr = self.recv_bufs[slot].addr
        try:
            if task.task_type == TT_CCSR:
                if stage_idx != 0:
                    self.log(f"[guest_mp-cq] CCSR slot={slot} got "
                              f"unexpected mid stage_idx={stage_idx}")
                if ("rgb_padded" in self._sr_dst_offs
                        and "rife_features" in self._sr_dst_offs
                        and self.cc_cache_writeback_fn is not None
                        and task.dst_cc_slot >= 0):
                    rgb_off, rgb_sz, _, _ = self._sr_dst_offs["rgb_padded"]
                    feat_off, feat_sz, _, _ = self._sr_dst_offs["rife_features"]
                    self.cc_cache_writeback_fn(
                        task.dst_cc_slot,
                        ("ccsr",
                         recv_addr + rgb_off,  rgb_sz,
                         recv_addr + feat_off, feat_sz))
            elif task.task_type == TT_INTERP:
                # Dense-pack: frame k+1 sits at offset k * frame_sz.
                dst_cc = self._interp_stage_dst_cc(task, stage_idx)
                if (self.cc_cache_writeback_fn is not None
                        and dst_cc >= 0):
                    rgb_addr = (recv_addr
                                + stage_idx * self._interp_dst_rgb_bytes)
                    self.cc_cache_writeback_fn(
                        dst_cc,
                        ("rgb_interp", rgb_addr,
                         self._interp_dst_rgb_bytes))
            else:
                self.log(f"[guest_mp-cq] unexpected mid recv slot={slot} "
                          f"task_type={task.task_type} stage={stage_idx}")
        except Exception as e:
            self.log(f"[guest_mp-cq] mid writeback failed slot={slot} "
                      f"tid={task.task_id} type={task.task_type} "
                      f"stage={stage_idx}: {e}")

        # Fire task_stage_done(tid, stage_idx) so queue_mgr can unblock
        # the SR_INTERP gated on this specific stage. Stage 0 covers
        # CCSR → INTERP edge and INTERP → SR_INTERP phase=1 edge
        # (mult ≥ 3); stages 1+ cover INTERP → SR_INTERP phase ≥ 2
        # edges (mult = 4).
        if (self.task_stage_done_fn is not None
                and task.task_id > 0):
            try:
                self.task_stage_done_fn(task.task_id, stage_idx)
            except Exception as e:
                self.log(f"[guest_mp-cq] task_stage_done("
                          f"{task.task_id}, {stage_idx}) failed: {e}")

    @staticmethod
    def _interp_stage_dst_cc(task: "_SlotTask", stage_idx: int) -> int:
        """Map INTERP stage_idx → which cc_cache slot the frame lands in.
        stage 0 → dst_cc_slot, 1 → dst_cc_slot_2, 2 → dst_cc_slot_3."""
        if stage_idx == 0:
            return task.dst_cc_slot
        if stage_idx == 1:
            return task.dst_cc_slot_2
        if stage_idx == 2:
            return task.dst_cc_slot_3
        return -1

    def _handle_recv(self, slot: int) -> None:
        """Response arrived in recv_bufs[slot]. Memmove the relevant
        plane(s) to their destination, then task_done + release slot."""
        task = self._task[slot]
        recv_addr = self.recv_bufs[slot].addr
        try:
            if task.task_type == TT_INTERP:
                # Dense-pack layout: frame k at slot.dst[k * frame_sz].
                # mult=N has frames 0..N-2; the final stage is frame
                # N-2. Mid frames 0..N-3 are written by _handle_mid_recv
                # when their stage's WRITE_WITH_IMM lands. We unconditionally
                # write the final frame here; we also belt-and-suspenders
                # rewrite earlier frames when mid delivery is off (i.e.
                # the worker never issued mid stages).
                if (self.cc_cache_writeback_fn is not None
                        and self._interp_mult >= 2):
                    mult = self._interp_mult
                    frame_sz = self._interp_dst_rgb_bytes
                    final_stage = mult - 2
                    final_dst_cc = self._interp_stage_dst_cc(task,
                                                              final_stage)
                    if final_dst_cc >= 0:
                        self.cc_cache_writeback_fn(
                            final_dst_cc,
                            ("rgb_interp",
                             recv_addr + final_stage * frame_sz,
                             frame_sz))
                    if not self._mid_delivery_enabled:
                        # Mid path off → all mid frames also live in
                        # this single response. Replay each into its
                        # cc_cache slot.
                        for k in range(final_stage):
                            mid_dst_cc = self._interp_stage_dst_cc(task, k)
                            if mid_dst_cc < 0:
                                continue
                            self.cc_cache_writeback_fn(
                                mid_dst_cc,
                                ("rgb_interp",
                                 recv_addr + k * frame_sz,
                                 frame_sz))
            elif task.task_type == TT_SR_INTERP:
                # Memmove yao/uao/vao to mpv VA
                if task.dst_y_va and task.dst_u_va and task.dst_v_va:
                    yao_off, _, _, _ = self._sr_dst_offs["yao"]
                    uao_off, _, _, _ = self._sr_dst_offs["uao"]
                    vao_off, _, _, _ = self._sr_dst_offs["vao"]
                    # Output plane shapes (1, 1, oH, oW) int16 / (1,1,ocH,ocW)
                    y_row_bytes = self.oW * 2
                    uv_row_bytes = self.ocW * 2
                    # Strided memmove if mpv VA stride != row_bytes
                    self._copy_plane(recv_addr + yao_off, task.dst_y_va,
                                      self.oH, y_row_bytes, task.dst_y_stride)
                    self._copy_plane(recv_addr + uao_off, task.dst_u_va,
                                      self.ocH, uv_row_bytes, task.dst_uv_stride)
                    self._copy_plane(recv_addr + vao_off, task.dst_v_va,
                                      self.ocH, uv_row_bytes, task.dst_uv_stride)
            elif task.task_type == TT_CCSR:
                # CCSR has two writeback targets:
                #   4K SR (yao/uao/vao) → mpv VA at dst_a (phase 0)
                #   rgb_padded + rife_features → host cc_cache[dst_cc_slot]
                # Both regions live in recv_bufs[slot].
                # When DUAL_MID_DELIVERY=1, the rgb_padded + features
                # writeback already happened in _handle_mid_recv;
                # skip it here.
                yao_off, _, _, _ = self._sr_dst_offs["yao"]
                uao_off, _, _, _ = self._sr_dst_offs["uao"]
                vao_off, _, _, _ = self._sr_dst_offs["vao"]
                y_row_bytes = self.oW * 2
                uv_row_bytes = self.ocW * 2
                if task.dst_y_va and task.dst_u_va and task.dst_v_va:
                    self._copy_plane(recv_addr + yao_off, task.dst_y_va,
                                      self.oH, y_row_bytes, task.dst_y_stride)
                    self._copy_plane(recv_addr + uao_off, task.dst_u_va,
                                      self.ocH, uv_row_bytes, task.dst_uv_stride)
                    self._copy_plane(recv_addr + vao_off, task.dst_v_va,
                                      self.ocH, uv_row_bytes, task.dst_uv_stride)
                # Mid-out writeback ONLY when mid delivery is OFF
                # (single-recv path carries it; mid-recv path already
                # handled it above).
                if (not self._mid_delivery_enabled
                        and "rgb_padded" in self._sr_dst_offs
                        and "rife_features" in self._sr_dst_offs
                        and self.cc_cache_writeback_fn is not None
                        and task.dst_cc_slot >= 0):
                    rgb_off, rgb_sz, _, _ = self._sr_dst_offs["rgb_padded"]
                    feat_off, feat_sz, _, _ = self._sr_dst_offs["rife_features"]
                    self.cc_cache_writeback_fn(
                        task.dst_cc_slot,
                        ("ccsr",
                         recv_addr + rgb_off,  rgb_sz,
                         recv_addr + feat_off, feat_sz))
            else:
                self.log(f"[guest_mp-cq] slot {slot} unexpected "
                          f"task_type {task.task_type}")
        except Exception as e:
            self.log(f"[guest_mp-cq] writeback failed slot={slot} "
                      f"tid={task.task_id} type={task.task_type}: {e}")

        # task_done + release. Always fire so the queue mgr doesn't
        # stall, even if the writeback errored.
        if self.task_done_fn is not None and task.task_id > 0:
            try:
                self.task_done_fn(task.task_id)
            except Exception as e:
                self.log(f"[guest_mp-cq] task_done({task.task_id}) "
                          f"failed: {e}")

        # NB: we do NOT re-post recv here. Each notify_submit posts
        # recv+send atomically under io_lock, so the next send for
        # this slot will re-post its recv at that point.

        with self._cv:
            self._state[slot] = ST_FREE
            self._free.append(slot)
            self._cv.notify()

    @staticmethod
    def _copy_plane(src_addr: int, dst_addr: int, rows: int,
                     row_bytes: int, dst_stride_bytes: int) -> None:
        if dst_stride_bytes == 0 or dst_stride_bytes == row_bytes:
            ctypes.memmove(dst_addr, src_addr, row_bytes * rows)
        else:
            # Strided destination: one vectorised numpy copy instead of
            # `rows` Python-level memmoves (~2 ms of call overhead for a
            # 4K Y plane). ctypes arrays expose a writable buffer, so
            # the frombuffer views are writable; both views only live
            # for the duration of this call.
            src = np.frombuffer(
                (ctypes.c_char * (row_bytes * rows)).from_address(src_addr),
                dtype=np.uint8).reshape(rows, row_bytes)
            dst_buf = (ctypes.c_char * (
                dst_stride_bytes * (rows - 1) + row_bytes)
            ).from_address(dst_addr)
            dst = np.lib.stride_tricks.as_strided(
                np.frombuffer(dst_buf, dtype=np.uint8),
                shape=(rows, row_bytes), strides=(dst_stride_bytes, 1))
            np.copyto(dst, src)

    def _fail_slot(self, slot: int, reason: str) -> None:
        """Recover one slot whose task can no longer complete: report
        the task failed (so queue_mgr doesn't leave it RUNNING and the
        frame falls back to its timeout path exactly once) and return
        the slot to the free list."""
        if not (0 <= slot < len(self._state)):
            return
        tid = 0
        with self._cv:
            t = self._task[slot]
            if self._state[slot] != ST_FREE:
                tid = t.task_id
                t.task_id = 0
                self._state[slot] = ST_FREE
                self._free.append(slot)
                self._cv.notify()
        if tid > 0:
            self.log(f"[guest_mp] slot {slot} task {tid} failed: {reason}")
            if self.task_fail_fn is not None:
                try:
                    self.task_fail_fn(tid, reason)
                except Exception as e:
                    self.log(f"[guest_mp] task_fail_fn({tid}) raised: {e}")

    def fail_inflight(self, reason: str) -> int:
        """Worker connection lost: fail every slot still carrying an
        unfinished task and free it. Returns the number of recovered
        tasks. Called by the dispatcher's liveness watchdog."""
        n = 0
        for slot in range(len(self._state)):
            if self._state[slot] != ST_FREE:
                self._fail_slot(slot, reason)
                n += 1
        return n

    def close(self):
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        # Daemon thread will exit on next poll iteration.
