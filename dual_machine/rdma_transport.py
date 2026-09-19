"""Raw RDMA (pyverbs) transport — replacement for NCCL P2P.

Why bypass NCCL: PyTorch's NCCL backend registers pending P2P
operations in its caching allocator's event tracker. Any subsequent
GPU sync (F.interpolate, .to('cpu'), .synchronize(), etc.) blocks
until those events fire — which they can't until the remote side
sends matching data. This makes async pipelining on the worker
deadlock. Going through pyverbs directly leaves PyTorch unaware,
so its allocator never blocks on RDMA state.

Architecture:
  RDMAContext       device + PD + CQ (one per process)
  RDMABuffer        pinned CPU MR (staging buffer for GPU↔NIC)
  RDMAChannel       one RC QP between two endpoints (host ↔ worker)

QP setup uses an out-of-band TCP handshake to exchange QPN + GID.
Once in RTS state, post_send / post_recv + poll_completion are the
only operations the user calls.

GPU tensor transfer is done via CPU staging buffers:
  send_tensor:  cudaMemcpy GPU → pinned CPU  →  post_send + poll
  recv_tensor:  post_recv + poll              →  cudaMemcpy pinned CPU → GPU

For top-end bandwidth this should later use DmaBufMR with CUDA-exported
dmabuf fd (GPU-Direct RDMA), but staging gets us functional first.

Requires PYTHONPATH=/usr/lib/python3/dist-packages so the vsmpv conda
env can see system pyverbs (compiled against Python 3.12 aarch64).
"""
from __future__ import annotations
import ctypes as _ctypes
import logging as _logging
import os
import socket
import struct
import sys

# pyverbs.MR.__init__ does logging.debug; under vapoursynth that
# routes through a buggy logging bridge ("no attribute parent"). Set
# the global level above DEBUG so pyverbs' debug messages are filtered
# out before they reach the bridge.
_logging.disable(_logging.INFO)

import numpy as np
import torch

# pyverbs lives in system Python 3.12 dist-packages
_PYVERBS_PATH = "/usr/lib/python3/dist-packages"
if _PYVERBS_PATH not in sys.path:
    sys.path.insert(0, _PYVERBS_PATH)

from pyverbs.device import Context
from pyverbs.pd import PD
from pyverbs.cq import CQ
from pyverbs.qp import QPInitAttr, QPCap, QP, QPAttr
from pyverbs.mr import MR
from pyverbs.wr import SendWR, RecvWR, SGE
from pyverbs.addr import GID, GlobalRoute, AHAttr
import pyverbs.enums as e


# ──────────────────────────────────────────────────────────────────────
# Context
# ──────────────────────────────────────────────────────────────────────


class RDMAContext:
    """One per process. Owns the ibv_context, PD, and a shared CQ
    that all channels poll on."""

    def __init__(self, dev_name: str = "rocep1s0f0", *, port: int = 1,
                 gid_index: int = 3, max_cqe: int = 4096):
        self.dev_name = dev_name
        self.port = port
        self.gid_index = gid_index  # 3 = RoCE v2 IPv4-mapped on this fabric

        self.ctx = Context(name=dev_name)
        self.pd = PD(self.ctx)
        self.cq = CQ(self.ctx, max_cqe)
        self.port_attr = self.ctx.query_port(port)
        self.gid = self.ctx.query_gid(port, gid_index)

    def register_region(self, addr: int, length: int) -> MR:
        """Register an existing memory region (e.g. the cc_cache shm)
        as an MR on this context's PD so its pages can be used as SGEs
        directly — the NIC then DMA-reads them instead of the CPU
        memmoving into a staging buffer (which, for cudaHostRegister'd
        shm on Grace, reads at ~5 GB/s). Caller keeps the returned MR
        alive for as long as the region is used in WRs."""
        access = (e.IBV_ACCESS_LOCAL_WRITE |
                  e.IBV_ACCESS_REMOTE_WRITE |
                  e.IBV_ACCESS_REMOTE_READ)
        return MR(self.pd, length, access, address=addr)

    def __del__(self):
        # pyverbs object dtors clean up automatically
        pass


# ──────────────────────────────────────────────────────────────────────
# Pinned host staging buffer (registered as MR)
# ──────────────────────────────────────────────────────────────────────


class RDMAStagingBuffer:
    """A page-aligned pinned host buffer registered as an MR. Used as
    the bounce buffer for GPU↔NIC transfers when GPU-Direct isn't
    wired up yet."""

    def __init__(self, rdma_ctx: RDMAContext, length: int,
                 *, force_pinned: bool = False):
        self.length = length
        # Plain (pageable) CPU buffer. We don't use pin_memory=True
        # because on Grace (GB10) cuda-pinned memory reads through
        # weakly-ordered uncached paths — fine for cudaMemcpy via DMA
        # but ~5 GB/s for plain CPU loads (vs ~40 GB/s for regular).
        # ibv_reg_mr will pin the pages for the duration of the MR
        # registration regardless, so RDMA still works.
        # `force_pinned=True` (or RDMA_PINNED=1) gives cuda-pinned
        # backing — required when the consumer wants cudaMemcpyAsync
        # straight off the MR (GPU-staging path).
        use_pinned = force_pinned or os.environ.get("RDMA_PINNED", "0") == "1"
        if use_pinned:
            self.cpu_t = torch.empty(length, dtype=torch.uint8,
                                      pin_memory=True)
        else:
            self.cpu_t = torch.empty(length, dtype=torch.uint8)
        addr = self.cpu_t.data_ptr()
        access = (e.IBV_ACCESS_LOCAL_WRITE |
                  e.IBV_ACCESS_REMOTE_WRITE |
                  e.IBV_ACCESS_REMOTE_READ)
        self.mr = MR(rdma_ctx.pd, length, access, address=addr)
        self.addr = addr
        self.lkey = self.mr.lkey
        self.rkey = self.mr.rkey

    def view_as(self, dtype, shape):
        """Return a torch view of the buffer with the given dtype +
        shape. Total bytes must fit in self.length."""
        nbytes = int(np.prod(shape) * torch.tensor([], dtype=dtype).element_size())
        if nbytes > self.length:
            raise ValueError(
                f"view_as: {shape} {dtype} = {nbytes} bytes > buffer {self.length}")
        return self.cpu_t[:nbytes].view(dtype).view(*shape)


# ──────────────────────────────────────────────────────────────────────
# Multi-rail support
#
# A DGX Spark's CX7 reaches 200 Gb/s only when two PCIe paths are driven
# at once (see dual_machine/README.md). One RDMAContext is one PCI
# function, so "use both rails" means: N contexts, N QPs, and every
# logical transfer split across them.
#
# Two asymmetries drive the design:
#
#   worker -> host is RDMA WRITE_WITH_IMM. The sender picks the remote
#   address, so a range can be cut anywhere and the pieces reassemble
#   themselves. Splitting is free; the only bookkeeping is telling the
#   receiver how many pieces to expect, which rides in the imm.
#
#   host -> worker is SEND/RECV. A message lands at the start of the
#   next pre-posted recv WR on that QP, so the split points must be
#   fixed in advance (src_partition) and, critically, the per-rail recv
#   queues must stay in lockstep: if a message puts a piece on rail 0
#   but not rail 1, the queues drift and later messages land in the
#   wrong slot's buffer. Hence every message posts exactly one piece per
#   rail, zero-length when that rail's range holds no bytes.
# ──────────────────────────────────────────────────────────────────────


def resolve_rails(devs: "str | None", gids: "str | None",
                   *, fallback_dev: str, fallback_gid: int) -> list[tuple]:
    """Parse the rail config into [(rdma_dev, gid_index), ...].

    `devs` / `gids` are comma-separated and come from DUAL_RDMA_DEVS /
    DUAL_RDMA_GIDS (host) or WMP_RDMA_DEVS / WMP_RDMA_GIDS (worker).
    A short or missing gid list repeats `fallback_gid`. An empty
    `devs` yields the single-rail config, which is byte-for-byte the
    old behaviour — that is the safety valve when a rail misbehaves.
    """
    if not devs or not devs.strip():
        return [(fallback_dev, fallback_gid)]
    dev_list = [d.strip() for d in devs.split(",") if d.strip()]
    gid_list = [g.strip() for g in (gids or "").split(",") if g.strip()]
    out = []
    for i, d in enumerate(dev_list):
        try:
            g = int(gid_list[i])
        except (IndexError, ValueError):
            g = fallback_gid
        out.append((d, g))
    return out


def src_partition(total: int, n_rails: int, *, align: int = 4096
                   ) -> list[tuple[int, int]]:
    """Fixed [(offset, length), ...] split of a slot's src buffer, one
    entry per rail. Both ends compute this from values exchanged in the
    handshake, so they agree without extra wire traffic.

    Used to place the worker's pre-posted recv WRs. Rail r's recv SGE
    is slot_src_addr + offset_r with length_r, and the host only ever
    sends rail r the bytes that fall in that window.
    """
    if n_rails <= 1:
        return [(0, total)]
    step = max(align, (total // n_rails) // align * align)
    parts = []
    off = 0
    for r in range(n_rails):
        ln = (total - off) if r == n_rails - 1 else min(step, total - off)
        parts.append((off, max(0, ln)))
        off += ln
    return parts


def stripe(off: int, size: int, n_rails: int, *,
            align: int = 4096, min_stripe: int = 1 << 20,
            rotate: int = 0) -> list[tuple[int, int, int]]:
    """Cut [off, off+size) into [(rail, sub_off, sub_size), ...].

    Ranges below `min_stripe` are not worth two completions and two
    doorbells, so they go whole onto one rail, chosen by `rotate` so a
    stream of small writes still spreads across the fabric.
    """
    if n_rails <= 1 or size <= 0:
        return [(0, off, size)]
    if size < min_stripe:
        return [(rotate % n_rails, off, size)]
    step = max(align, (size // n_rails) // align * align)
    out = []
    cur = off
    end = off + size
    for r in range(n_rails):
        sub = (end - cur) if r == n_rails - 1 else min(step, end - cur)
        out.append((r, cur, max(0, sub)))
        cur += sub
    return out


# imm_data layout, shared by guest_mp and worker_3proc. Must stay in
# sync on both ends — that is why it lives here and not in either.
#   bits  0-15  host slot
#   bits 16-19  stage_code (0..14 = mid stage_idx, 15 = final)
#   bits 20-23  part index within the striped write
#   bits 24-27  part count (1 = not striped)
STAGE_FINAL = 15


def imm_encode(slot: int, stage_code: int,
                part: int = 0, n_parts: int = 1) -> int:
    return ((slot & 0xFFFF)
            | ((stage_code & 0xF) << 16)
            | ((part & 0xF) << 20)
            | ((n_parts & 0xF) << 24))


def imm_decode(imm: int) -> tuple[int, int, int, int]:
    """-> (slot, stage_code, part, n_parts). n_parts is clamped to >=1
    so an imm written by a single-rail peer (which leaves bits 24-27
    zero) still decodes as one whole part."""
    return (imm & 0xFFFF,
            (imm >> 16) & 0xF,
            (imm >> 20) & 0xF,
            max(1, (imm >> 24) & 0xF))


class MultiRailBuffer:
    """One host allocation, registered once per rail.

    An MR belongs to exactly one PD, so N rails need N registrations of
    the same pages. The keys differ per rail; the address does not.
    Rail 0's keys are exposed as .lkey / .rkey so single-rail callers
    read unchanged.
    """

    def __init__(self, ctxs: list, length: int, *,
                 force_pinned: bool = False):
        self.length = length
        use_pinned = force_pinned or os.environ.get("RDMA_PINNED", "0") == "1"
        self.cpu_t = torch.empty(length, dtype=torch.uint8,
                                  pin_memory=True) if use_pinned else \
            torch.empty(length, dtype=torch.uint8)
        self.addr = self.cpu_t.data_ptr()
        access = (e.IBV_ACCESS_LOCAL_WRITE |
                  e.IBV_ACCESS_REMOTE_WRITE |
                  e.IBV_ACCESS_REMOTE_READ)
        self.mrs = [MR(c.pd, length, access, address=self.addr)
                    for c in ctxs]
        self.lkeys = [m.lkey for m in self.mrs]
        self.rkeys = [m.rkey for m in self.mrs]
        self.lkey = self.lkeys[0]
        self.rkey = self.rkeys[0]

    def view_as(self, dtype, shape):
        nbytes = int(np.prod(shape) * torch.tensor([], dtype=dtype).element_size())
        if nbytes > self.length:
            raise ValueError(
                f"view_as: {shape} {dtype} = {nbytes} bytes > buffer {self.length}")
        return self.cpu_t[:nbytes].view(dtype).view(*shape)


# ──────────────────────────────────────────────────────────────────────
# Channel: one RC QP between two endpoints
# ──────────────────────────────────────────────────────────────────────


# 24 bytes: u32 qpn + u32 psn + 16 bytes gid_raw, then u32 payload_len +
# `payload_len` bytes of caller-supplied extra blob (used for
# WRITE_WITH_IMM slot info exchange).
_HANDSHAKE_FMT = "!II16sI"  # qpn, psn, gid_raw, extra_len


def _recv_exact(sock, n: int) -> bytes:
    data = b""
    while len(data) < n:
        chunk = sock.recv(min(n - len(data), 65536))
        if not chunk:
            raise RuntimeError(f"handshake: peer closed after {len(data)}/{n}")
        data += chunk
    return data


def _exchange_endpoints(my_qpn: int, my_psn: int, my_gid_raw: bytes,
                         *, peer_ip: str, peer_port: int,
                         is_server: bool, my_extra: bytes = b""
                         ) -> tuple[int, int, bytes, bytes]:
    """TCP-based handshake: trade (qpn, psn, gid, extra_blob) with peer.
    Returns peer's (qpn, psn, gid_raw, extra_blob). `extra` is opaque to
    this function — used by RDMAChannel callers to swap remote MR
    addresses + rkeys (WRITE_WITH_IMM)."""
    head = struct.pack(_HANDSHAKE_FMT, my_qpn, my_psn, my_gid_raw,
                        len(my_extra))
    msg = head + my_extra
    # Both sides must be inside this window for the QPs to pair up.
    # 30 s used to be the fixed budget on each side and it is not
    # enough: with multiple rails each end opens N contexts and
    # registers N MRs over the slot ring before it reaches the first
    # handshake, and a cold worker may still be compiling TRT engines.
    # The failure mode was a bare "cannot reach" that looks like a
    # network fault. Override with DUAL_HANDSHAKE_TIMEOUT_S.
    budget = float(os.environ.get("DUAL_HANDSHAKE_TIMEOUT_S", "120"))
    if is_server:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", peer_port))
        # backlog 1 turns a stray probe (or a peer that reconnects)
        # into RST for everyone behind it; give the queue some slack.
        s.listen(16)
        s.settimeout(budget)
        c, _ = s.accept()
        c.sendall(msg)
        data = _recv_exact(c, struct.calcsize(_HANDSHAKE_FMT))
        peer_qpn, peer_psn, peer_gid_raw, peer_extra_len = \
            struct.unpack(_HANDSHAKE_FMT, data)
        peer_extra = _recv_exact(c, peer_extra_len) if peer_extra_len else b""
        c.close(); s.close()
    else:
        import time as _t
        deadline = _t.monotonic() + budget
        last_err = None
        attempts = 0
        while _t.monotonic() < deadline:
            attempts += 1
            c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                c.settimeout(2.0)
                c.connect((peer_ip, peer_port))
                break
            except OSError as ex:
                last_err = ex
                try: c.close()
                except OSError: pass
                _t.sleep(0.5)
        else:
            raise RuntimeError(
                f"handshake: cannot reach {peer_ip}:{peer_port} after "
                f"{attempts} attempts in {budget:.0f}s "
                f"(last error: {last_err}) — is the peer's worker up, "
                f"and does this rail's subnet route to it?")
        c.sendall(msg)
        data = _recv_exact(c, struct.calcsize(_HANDSHAKE_FMT))
        peer_qpn, peer_psn, peer_gid_raw, peer_extra_len = \
            struct.unpack(_HANDSHAKE_FMT, data)
        peer_extra = _recv_exact(c, peer_extra_len) if peer_extra_len else b""
        c.close()
    return peer_qpn, peer_psn, peer_gid_raw, peer_extra


def _gid_to_raw(gid: GID) -> bytes:
    """Pack a GID's 16 bytes for wire transmission.
    pyverbs exposes only .gid as 'aaaa:bbbb:...' — parse it ourselves."""
    return bytes.fromhex(gid.gid.replace(":", ""))


def _raw_to_gid(raw: bytes) -> GID:
    """Reconstruct a GID from 16 raw bytes."""
    hex_pairs = [raw[i:i+2].hex() for i in range(0, 16, 2)]
    return GID(":".join(hex_pairs))


class RDMAChannel:
    """One RC QP for bidirectional traffic between host and worker.
    Both sides post_send / post_recv as needed; CQ delivers completions."""

    def __init__(self, rdma_ctx: RDMAContext, *, is_server: bool,
                 peer_ip: str, peer_port: int,
                 max_wr: int = 64, my_psn: int = 0,
                 extra_payload: bytes = b""):
        self.rdma_ctx = rdma_ctx
        # max_send_sge=8: post_send_sgl gathers a task header + up to 4
        # cc_cache fields into one SEND (zero-copy dispatch path). mlx5
        # supports 30+; 8 leaves headroom without bloating the WQE.
        cap = QPCap(max_send_wr=max_wr, max_recv_wr=max_wr,
                    max_send_sge=8, max_recv_sge=1)
        init = QPInitAttr(qp_type=e.IBV_QPT_RC,
                          scq=rdma_ctx.cq, rcq=rdma_ctx.cq, cap=cap)
        self.qp = QP(rdma_ctx.pd, init)
        self.my_qpn = self.qp.qp_num
        self.my_psn = my_psn

        # Exchange QPN + PSN + GID (+ optional opaque payload) with peer
        # over TCP. `extra_payload` lets WRITE_WITH_IMM swap remote
        # MR (addr, rkey, size) tuples in the same round-trip.
        peer_qpn, peer_psn, peer_gid_raw, peer_extra = _exchange_endpoints(
            self.my_qpn, self.my_psn, _gid_to_raw(rdma_ctx.gid),
            peer_ip=peer_ip, peer_port=peer_port, is_server=is_server,
            my_extra=extra_payload)
        self.peer_qpn = peer_qpn
        self.peer_psn = peer_psn
        self.peer_gid = _raw_to_gid(peer_gid_raw)
        self.peer_extra = peer_extra

        # State transitions: RESET → INIT → RTR → RTS
        self._init_state()
        self._init_to_rtr()
        self._rtr_to_rts()

    def _init_state(self):
        attr = QPAttr()
        attr.qp_state = e.IBV_QPS_INIT
        attr.pkey_index = 0
        attr.port_num = self.rdma_ctx.port
        attr.qp_access_flags = (e.IBV_ACCESS_LOCAL_WRITE |
                                e.IBV_ACCESS_REMOTE_WRITE |
                                e.IBV_ACCESS_REMOTE_READ)
        mask = (e.IBV_QP_STATE | e.IBV_QP_PKEY_INDEX |
                e.IBV_QP_PORT | e.IBV_QP_ACCESS_FLAGS)
        self.qp.modify(attr, mask)

    def _init_to_rtr(self):
        attr = QPAttr()
        attr.qp_state = e.IBV_QPS_RTR
        attr.path_mtu = self.rdma_ctx.port_attr.active_mtu
        attr.dest_qp_num = self.peer_qpn
        attr.rq_psn = self.peer_psn
        attr.max_dest_rd_atomic = 16
        attr.min_rnr_timer = 12
        # Address handle: RoCE always uses GRH (no LIDs). Set GR
        # fields directly on AHAttr.
        ah = AHAttr()
        ah.is_global = 1
        ah.dlid = 0
        ah.sl = 0
        ah.src_path_bits = 0
        ah.port_num = self.rdma_ctx.port
        # AHAttr.dgid setter wants the string form (calls .split(':')
        # internally), not the GID object. Pass .gid attribute.
        ah.dgid = self.peer_gid.gid
        ah.sgid_index = self.rdma_ctx.gid_index
        ah.hop_limit = 64
        ah.traffic_class = 0
        ah.flow_label = 0
        attr.ah_attr = ah
        mask = (e.IBV_QP_STATE | e.IBV_QP_AV | e.IBV_QP_PATH_MTU |
                e.IBV_QP_DEST_QPN | e.IBV_QP_RQ_PSN |
                e.IBV_QP_MAX_DEST_RD_ATOMIC | e.IBV_QP_MIN_RNR_TIMER)
        self.qp.modify(attr, mask)

    def _rtr_to_rts(self):
        attr = QPAttr()
        attr.qp_state = e.IBV_QPS_RTS
        attr.sq_psn = self.my_psn
        attr.timeout = 14
        attr.retry_cnt = 7
        attr.rnr_retry = 7
        attr.max_rd_atomic = 16
        mask = (e.IBV_QP_STATE | e.IBV_QP_SQ_PSN | e.IBV_QP_TIMEOUT |
                e.IBV_QP_RETRY_CNT | e.IBV_QP_RNR_RETRY |
                e.IBV_QP_MAX_QP_RD_ATOMIC)
        self.qp.modify(attr, mask)

    # ── Posting work requests ─────────────────────────────────────────

    def post_send(self, buf: RDMAStagingBuffer, length: int, wr_id: int = 0):
        sge = SGE(addr=buf.addr, length=length, lkey=buf.lkey)
        wr = SendWR(opcode=e.IBV_WR_SEND, num_sge=1, sg=[sge],
                     wr_id=wr_id, send_flags=e.IBV_SEND_SIGNALED)
        self.qp.post_send(wr)

    def post_recv(self, buf: RDMAStagingBuffer, length: int, wr_id: int = 0):
        sge = SGE(addr=buf.addr, length=length, lkey=buf.lkey)
        wr = RecvWR(num_sge=1, sg=[sge], wr_id=wr_id)
        self.qp.post_recv(wr)

    def post_send_at(self, addr: int, length: int, lkey: int,
                      wr_id: int = 0):
        """Address-based variant — used by worker_3proc.py where the
        buffer lives in shm and is not wrapped in an RDMAStagingBuffer."""
        sge = SGE(addr=addr, length=length, lkey=lkey)
        wr = SendWR(opcode=e.IBV_WR_SEND, num_sge=1, sg=[sge],
                     wr_id=wr_id, send_flags=e.IBV_SEND_SIGNALED)
        self.qp.post_send(wr)

    def post_send_sgl(self, sges: list, wr_id: int = 0):
        """Gathered SEND. `sges` = [(addr, length, lkey), ...] in wire
        order; the peer receives their concatenation as ONE message.
        Lets a dispatcher send header (bounce buffer) + payload fields
        (e.g. cc_cache shm registered as its own MR) without first
        memmoving everything into one staging buffer — the NIC's DMA
        engine does the gather."""
        sg = [SGE(addr=a, length=ln, lkey=k) for (a, ln, k) in sges]
        wr = SendWR(opcode=e.IBV_WR_SEND, num_sge=len(sg), sg=sg,
                     wr_id=wr_id, send_flags=e.IBV_SEND_SIGNALED)
        self.qp.post_send(wr)

    def post_recv_at(self, addr: int, length: int, lkey: int,
                      wr_id: int = 0):
        sge = SGE(addr=addr, length=length, lkey=lkey)
        wr = RecvWR(num_sge=1, sg=[sge], wr_id=wr_id)
        self.qp.post_recv(wr)

    def post_write(self, *, src_addr: int, src_lkey: int, length: int,
                   remote_addr: int, rkey: int, wr_id: int = 0,
                   signaled: bool = True):
        """Submit an RDMA WRITE: copy [src_addr, src_addr+length) into
        the peer's [remote_addr, remote_addr+length) (the peer must
        have pre-registered an MR covering that range with rkey).
        No completion is generated on the peer."""
        sge = SGE(addr=src_addr, length=length, lkey=src_lkey)
        flags = e.IBV_SEND_SIGNALED if signaled else 0
        wr = SendWR(opcode=e.IBV_WR_RDMA_WRITE, num_sge=1, sg=[sge],
                     wr_id=wr_id, send_flags=flags)
        wr.set_wr_rdma(rkey=rkey, addr=remote_addr)
        self.qp.post_send(wr)

    def post_write_with_imm(self, *, src_addr: int, src_lkey: int,
                             length: int, remote_addr: int, rkey: int,
                             imm: int, wr_id: int = 0,
                             signaled: bool = True):
        """RDMA WRITE_WITH_IMM: write [src_addr, +length) to peer's
        [remote_addr, +length) AND deliver the 32-bit `imm` to the
        peer's CQ. Consumes one recv WR on the peer; the recv WR's
        local SGE is ignored (zero-length dummies are fine). Used by
        mid_delivery to dispatch (slot, stage) via imm rather than
        relying on SEND/RECV queue ordering."""
        sge = SGE(addr=src_addr, length=length, lkey=src_lkey)
        flags = e.IBV_SEND_SIGNALED if signaled else 0
        wr = SendWR(opcode=e.IBV_WR_RDMA_WRITE_WITH_IMM, num_sge=1,
                     sg=[sge], wr_id=wr_id, send_flags=flags)
        wr.set_wr_rdma(rkey=rkey, addr=remote_addr)
        wr.imm_data = imm
        self.qp.post_send(wr)

    def post_read(self, *, dst_addr: int, dst_lkey: int, length: int,
                  remote_addr: int, rkey: int, wr_id: int = 0,
                  signaled: bool = True):
        """Submit an RDMA READ: pull [remote_addr, remote_addr+length)
        from the peer (the peer must have registered that range with
        rkey) into our local [dst_addr, dst_addr+length)."""
        sge = SGE(addr=dst_addr, length=length, lkey=dst_lkey)
        flags = e.IBV_SEND_SIGNALED if signaled else 0
        wr = SendWR(opcode=e.IBV_WR_RDMA_READ, num_sge=1, sg=[sge],
                     wr_id=wr_id, send_flags=flags)
        wr.set_wr_rdma(rkey=rkey, addr=remote_addr)
        self.qp.post_send(wr)

    def poll_cq_blocking(self) -> int:
        """Spin-poll until one WC arrives. Returns wr_id."""
        while True:
            n, wcs = self.rdma_ctx.cq.poll(1)
            if n > 0:
                wc = wcs[0]
                if wc.status != e.IBV_WC_SUCCESS:
                    raise RuntimeError(
                        f"WC error: status={wc.status} wr_id={wc.wr_id}")
                return wc.wr_id


# ──────────────────────────────────────────────────────────────────────
# High-level helpers for Mode B pair I/O
#
# Per pair the host sends:
#   header (4 × int64 = 32 B) + 6 source planes (12 MB) packed
#   into one staging buffer → one post_send
# The worker recvs into its own staging buffer, computes, then
# sends 6 result planes (50 MB) packed into one staging buffer back
# to the host (one post_send). The host has one post_recv pending
# for that.
#
# Two unidirectional 'channels' here = two QPs:
#   qp_out  host→worker (host sends src bundles; worker recvs)
#   qp_in   worker→host (worker sends dst bundles; host recvs)
# We use one CQ shared by both — completions distinguished by wr_id.
# ──────────────────────────────────────────────────────────────────────


# Layout of one pair's "src bundle":
#   [0..32)        header   (4 × int64)
#   [32..    )     ya       int16 (1, 1, H, W)
#   [..       )    ua       int16 (1, 1, cH, cW)
#   ... etc
def dst_bundle_layout(out_h: int, out_w: int, oc_h: int, oc_w: int,
                       *,
                       pH: int = 1088, pW: int = 1920,
                       enc_ch: int = 4,
                       interp_overlay_bytes: int = 0
                       ) -> tuple[int, list]:
    """Result planes the worker sends back.

    Layout: 4K SR output (yao/uao/vao) + CCSR mid-output (rgb_padded
    + rife_features for guest CCSR's downstream INTERPs).

    Per-task interpretation:
      - CCSR slot:        writes all 5 fields. Producer splits the
                          slot into MID range (rgb_padded +
                          rife_features, contiguous tail) and FULL
                          range (yao/uao/vao, contiguous head).
      - INTERP slot:      dense-pack — frame k at offset k * frame_sz.
                          mult=2 writes one frame at offset 0
                          (overlaying yao space); mult ≥ 3 packs
                          successive frames.
      - SR_INTERP slot:   writes yao/uao/vao (single-stage, no MID).
    The per-task (mid_off, mid_size, full_off, full_size) split is
    computed by callers (see worker_3proc / host_3proc) using the
    returned layout dict.

    `interp_overlay_bytes` reserves enough tail capacity for the
    dense-pack INTERP overlay (max_mult * rgb_interp_size). When the
    no_sr path makes output dim equal to source dim, the named-region
    total can fall below mult × rgb_interp_size and the dense-pack
    would assert in mp_pipeline.task_dst_ranges. Pad the total so
    INTERP always fits; named offsets are unaffected.

    Returns (total_bytes, [(name, offset, nbytes, shape, dtype), ...]).
    """
    layout = []
    off = 0
    def add(name, shape, dtype=torch.int16):
        nonlocal off
        nbytes = int(torch.empty(shape, dtype=dtype).numel() *
                      torch.empty([], dtype=dtype).element_size())
        layout.append((name, off, nbytes, shape, dtype))
        off += nbytes
    add("yao", (1, 1, out_h, out_w))
    add("uao", (1, 1, oc_h, oc_w))
    add("vao", (1, 1, oc_h, oc_w))
    add("rgb_padded",    (1, 3,      pH, pW), dtype=torch.float16)
    add("rife_features", (1, enc_ch, pH, pW), dtype=torch.float16)
    total = max(off, interp_overlay_bytes)
    return total, layout

_ZC_HEADER_INT64 = 16   # 128-byte header (16 × int64)

def zc_src_bundle_layout(H: int, W: int, cH: int, cW: int,
                          pH: int, pW: int, enc_ch: int
                          ) -> tuple[int, list]:
    """Zero-copy src bundle layout: 128-byte header (pair_k, phase,
    src/dst mpv plane targets, cc_slot a/b/field) followed by 6 YUV
    planes (ya/ua/va/yb/ub/vb).

    INTERP tasks overlay their own per-task structure (rgb_padded +
    rife_features for each of the 2 source frames) on top of slot.src
    starting at the header — header offset disambiguates task type. The
    INTERP overlay can be larger than the named YUV planes, so the
    returned size is max(planes, header + interp_overlay). Both sides
    of the wire MUST compute identical sizes, so pH/pW/enc_ch (which
    cross the wire in the handshake) are inputs rather than defaults.
    """
    layout = []
    off = 0
    def add(name, shape, dtype=torch.int16):
        nonlocal off
        nbytes = int(torch.empty(shape, dtype=dtype).numel() *
                      torch.empty([], dtype=dtype).element_size())
        layout.append((name, off, nbytes, shape, dtype))
        off += nbytes
    add("header", (_ZC_HEADER_INT64,), torch.int64)
    add("ya", (1, 1, H, W))
    add("ua", (1, 1, cH, cW))
    add("va", (1, 1, cH, cW))
    add("yb", (1, 1, H, W))
    add("ub", (1, 1, cH, cW))
    add("vb", (1, 1, cH, cW))
    # INTERP overlay: header + 2 × (rgb_padded + rife_features).
    # Per-frame rgb_padded = 1 × 3 × pH × pW × 2 (fp16 NCHW),
    # rife_features = 1 × enc_ch × pH × pW × 2.
    header_bytes = _ZC_HEADER_INT64 * 8
    rgb_padded_bytes    = 1 * 3      * pH * pW * 2
    rife_features_bytes = 1 * enc_ch * pH * pW * 2
    interp_overlay = header_bytes + 2 * (rgb_padded_bytes + rife_features_bytes)
    if off < interp_overlay:
        off = interp_overlay
    return off, layout


# wr_id encoding for the zerocopy path:
_ZC_WR_SRC_RECV  = 0x10000  # | slot — worker: src bundle arrived
_ZC_WR_WRITE     = 0x20000  # | slot — worker: RDMA WRITE of FULL range completed
_ZC_WR_ACK_SEND  = 0x30000  # | slot — worker: FULL-done SEND completed
_ZC_WR_ACK_RECV  = 0x40000  # | slot — host:   FULL-done arrived (→ DST_READY)
_ZC_WR_SRC_SEND  = 0x50000  # | slot — host:   src bundle SEND completed
# Two-phase early-delivery: producers (CCSR, INTERP mult ≥ 3) that
# have a mid stage emit a WRITE of the mid byte range followed by a
# small SEND ack. Host's dma_proc transitions slot → MID_READY on the
# RECV. Single-stage tasks (INTERP mult=2, SR_INTERP) skip these.
_ZC_WR_WRITE_MID = 0x60000  # | slot — worker: RDMA WRITE of MID range completed
_ZC_WR_MID_SEND  = 0x70000  # | slot — worker: MID-done SEND completed
_ZC_WR_MID_RECV  = 0x80000  # | slot — host:   MID-done arrived (→ MID_READY)


# Header field offsets (int64 indices)
_HDR_PAIR_K    = 0
_HDR_PHASE     = 1
_HDR_SRC_A     = 2
_HDR_SRC_B     = 3
_HDR_HOST_SLOT = 4    # WRITE_WITH_IMM: host's guest_mp slot index
# Slots 5-9 reserved for future use.
# Split-task wire fields.
_HDR_TASK_TYPE = 10
_HDR_TASK_ID   = 11
# cc_cache references. On the wire we encode unset slots as -1
# (sign-extended to int64). The guest's rdma_proc consults these to
# decide whether to RDMA-READ from the host's cc_cache region.
_HDR_CC_SLOT_A = 12
_HDR_CC_SLOT_B = 13
_HDR_CC_FIELD  = 14
# For INTERP mult ≥ 3, the second output frame's cc_cache slot on
# the host side. -1 (or 0) means single-frame INTERP (mult=2). Worker
# reads this and also reads mult from DUAL_INTERP_MULT env at startup
# — the slot value just routes the writeback target.
_HDR_DST_CC_SLOT_2 = 15

