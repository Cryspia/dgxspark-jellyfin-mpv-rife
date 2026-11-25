"""Bounded, one-directional multi-rail transfer between two workers.

This transport is independent of the task scheduler and GPU runtime.
Each endpoint owns one instance per direction and allows one message in
flight; a sender completion permits its input to be reused. The receiver
returns a private copy so its registered receive buffers can be reposted.
"""
from __future__ import annotations

import ctypes
import struct
import threading
import time

_MAGIC = b"PT02"
_HEADER = struct.Struct("!4sQQQII")
_HELLO = struct.Struct("!4sIQI")
_WRITE_HELLO = struct.Struct("!4sIIQQQI")


def _partitions(length: int, n_rails: int, capacity: int | None = None) -> list[tuple[int, int]]:
    capacity = length if capacity is None else capacity
    if length < 0 or n_rails < 1 or capacity < length:
        raise ValueError("invalid payload length or rail count")
    size = (capacity + n_rails - 1) // n_rails
    return [(min(i * size, length), min(size, max(0, length - i * size)))
            for i in range(n_rails)]


class RegisteredBuffer:
    """A caller-owned region registered separately with each rail."""

    def __init__(self, owner: "PeerLink", addr: int, length: int):
        if addr <= 0 or length < owner.max_bytes:
            raise ValueError("registered region is smaller than link capacity")
        self.owner = owner
        self.addr = addr
        self.length = length
        self.mrs = [ctx.register_region(addr, length) for ctx in owner._ctx]
        self.lkeys = [mr.lkey for mr in self.mrs]

    def close(self) -> None:
        with self.owner._lock:
            for mr in self.mrs:
                mr.close()
            self.mrs.clear()
            if self in self.owner._registrations:
                self.owner._registrations.remove(self)


class PeerLink:
    """Synchronous framed SEND/RECV on dedicated RDMA queue pairs.

    Callers can run this object in a network thread while GPU compute and
    host-output transfers proceed on their own threads and queue pairs.
    A timed-out connection must be discarded, not reused.
    """

    def __init__(self, rails: list[tuple[str, int]], *, peer_ip: str,
                 port: int, is_server: bool, max_bytes: int,
                 timeout: float = 30.0, token: bytes = b""):
        if not rails or max_bytes < 1 or timeout <= 0 or len(token) not in (0, 16):
            raise ValueError("invalid rails, capacity or timeout")
        from rdma_transport import RDMAChannel, RDMAContext, RDMAStagingBuffer
        self.max_bytes = max_bytes
        self.timeout = timeout
        self._lock = threading.RLock()
        self._failed = False
        self._closed = False
        self._registrations: list[RegisteredBuffer] = []
        self._ctx = [RDMAContext(dev_name=dev, gid_index=gid, max_cqe=16)
                     for dev, gid in rails]
        max_part = (max_bytes + len(rails) - 1) // len(rails)
        self._buffers = [RDMAStagingBuffer(ctx, _HEADER.size + max_part)
                         for ctx in self._ctx]
        self._channels = []
        try:
            for i, ctx in enumerate(self._ctx):
                hello = _HELLO.pack(b"PH02", len(rails), max_bytes, i) + token
                channel = RDMAChannel(ctx, is_server=is_server,
                                      peer_ip=peer_ip, peer_port=port + i,
                                      max_recv_sge=2,
                                      handshake_timeout=timeout,
                                      extra_payload=hello)
                self._channels.append(channel)
                if channel.peer_extra != hello:
                    raise ValueError("peer rail count, capacity or protocol mismatch")
        except Exception:
            self.close()
            raise

    def _wait(self, rail: int) -> object:
        from pyverbs import enums as e
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            n, completions = self._ctx[rail].cq.poll(1)
            if n:
                wc = completions[0]
                if wc.status != e.IBV_WC_SUCCESS:
                    raise RuntimeError(f"peer RDMA completion failed: {wc.status}")
                return wc
            time.sleep(0.0001)
        raise TimeoutError(f"peer RDMA rail {rail} timed out")

    def register(self, addr: int, length: int) -> RegisteredBuffer:
        """Register existing host-visible memory for direct DMA transfers.

        The caller owns the allocation and must keep both it and the
        returned registration alive until all transfers finish.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("peer link is closed")
            buffer = RegisteredBuffer(self, addr, length)
            self._registrations.append(buffer)
            return buffer

    def _check_registered(self, buffer: RegisteredBuffer, length: int,
                          offset: int = 0) -> None:
        if (buffer.owner is not self or not buffer.mrs or length < 0
                or length > self.max_bytes or offset < 0
                or offset + length > buffer.length):
            raise ValueError("invalid or closed peer registration")

    def send(self, generation: int, frame: int, payload: bytes) -> None:
        if generation < 0 or frame < 0 or len(payload) > self.max_bytes:
            raise ValueError("invalid generation, frame or payload size")
        with self._lock:
            if self._failed:
                raise RuntimeError("peer link is unusable after failure")
            try:
                parts = _partitions(len(payload), len(self._channels), self.max_bytes)
                for i, (offset, size) in enumerate(parts):
                    header = _HEADER.pack(_MAGIC, generation, frame, len(payload),
                                          i, len(parts))
                    buf = self._buffers[i]
                    ctypes.memmove(buf.addr, header, len(header))
                    if size:
                        ctypes.memmove(buf.addr + _HEADER.size,
                                       payload[offset:offset + size], size)
                    self._channels[i].post_send(buf, _HEADER.size + size, wr_id=i)
                for i in range(len(parts)):
                    self._wait(i)
            except Exception:
                self._failed = True
                raise

    def receive(self) -> tuple[int, int, bytes]:
        with self._lock:
            if self._failed:
                raise RuntimeError("peer link is unusable after failure")
            try:
                for i, channel in enumerate(self._channels):
                    channel.post_recv(self._buffers[i], self._buffers[i].length,
                                      wr_id=i)
                records = []
                for i, buf in enumerate(self._buffers):
                    wc = self._wait(i)
                    if wc.byte_len < _HEADER.size:
                        raise ValueError("truncated peer transfer header")
                    fields = _HEADER.unpack(ctypes.string_at(buf.addr, _HEADER.size))
                    magic, generation, frame, total, rail, count = fields
                    if (magic != _MAGIC or rail != i or count != len(self._buffers)
                            or total > self.max_bytes):
                        raise ValueError("invalid peer transfer header")
                    records.append((generation, frame, total, wc.byte_len))
                generation, frame, total, _ = records[0]
                parts = _partitions(total, len(records), self.max_bytes)
                chunks = []
                for i, ((offset, size), record) in enumerate(zip(parts, records)):
                    if record != (generation, frame, total, _HEADER.size + size):
                        raise ValueError("peer transfer parts do not match")
                    chunks.append(ctypes.string_at(self._buffers[i].addr + _HEADER.size, size))
                return generation, frame, b"".join(chunks)
            except Exception:
                self._failed = True
                raise

    def send_registered(self, generation: int, frame: int,
                        buffer: RegisteredBuffer, length: int, *,
                        offset: int = 0) -> None:
        """Send directly from an existing registered buffer, without staging."""
        if generation < 0 or frame < 0 or length < 0:
            raise ValueError("invalid peer message")
        with self._lock:
            if self._failed:
                raise RuntimeError("peer link is unusable after failure")
            self._check_registered(buffer, length, offset)
            try:
                parts = _partitions(length, len(self._channels), self.max_bytes)
                for i, (part_offset, size) in enumerate(parts):
                    header = _HEADER.pack(_MAGIC, generation, frame, length,
                                          i, len(parts))
                    buf = self._buffers[i]
                    ctypes.memmove(buf.addr, header, len(header))
                    sges = [(buf.addr, len(header), buf.lkey)]
                    if size:
                        sges.append((buffer.addr + offset + part_offset, size,
                                     buffer.lkeys[i]))
                    self._channels[i].post_send_sgl(sges, wr_id=i)
                for i in range(len(parts)):
                    self._wait(i)
            except Exception:
                self._failed = True
                raise

    def receive_registered(self, buffer: RegisteredBuffer, *,
                           offset: int = 0) -> tuple[int, int, int]:
        """Receive directly into registered memory and return its metadata."""
        with self._lock:
            if self._failed:
                raise RuntimeError("peer link is unusable after failure")
            self._check_registered(buffer, self.max_bytes, offset)
            try:
                capacity_parts = _partitions(self.max_bytes, len(self._channels),
                                             self.max_bytes)
                for i, (part_offset, size) in enumerate(capacity_parts):
                    hdr = self._buffers[i]
                    sges = [(hdr.addr, _HEADER.size, hdr.lkey)]
                    if size:
                        sges.append((buffer.addr + offset + part_offset, size,
                                     buffer.lkeys[i]))
                    self._channels[i].post_recv_sgl(sges, wr_id=i)
                records = []
                for i, hdr in enumerate(self._buffers):
                    wc = self._wait(i)
                    if wc.byte_len < _HEADER.size:
                        raise ValueError("truncated peer transfer header")
                    magic, generation, frame, total, rail, count = _HEADER.unpack(
                        ctypes.string_at(hdr.addr, _HEADER.size))
                    if (magic != _MAGIC or rail != i or count != len(self._buffers)
                            or total > self.max_bytes):
                        raise ValueError("invalid peer transfer header")
                    records.append((generation, frame, total, wc.byte_len))
                generation, frame, total, _ = records[0]
                for record, (_, size) in zip(
                        records, _partitions(total, len(records), self.max_bytes)):
                    if record != (generation, frame, total, _HEADER.size + size):
                        raise ValueError("peer transfer parts do not match")
                return generation, frame, total
            except Exception:
                self._failed = True
                raise

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._failed = True
            for channel in self._channels:
                channel.qp.close()
            for buffer in tuple(self._registrations):
                buffer.close()
            self._registrations.clear()
            for buffer in self._buffers:
                buffer.mr.close()
            for ctx in self._ctx:
                ctx.cq.close()
                ctx.pd.close()
                ctx.ctx.close()


class FrameCachePeer:
    """Move one prepared frame between worker-owned CC cache slots.

    The scheduler retains both slots until this operation completes.
    A successful receive is the point at which its destination becomes
    safe for the receiving worker's compute process to consume.
    """

    def __init__(self, link: PeerLink, cache):
        layout = cache.layout
        if (layout.n_slots < 1 or layout.off_rgb_padded != 0
                or layout.off_sr_yuv > layout.slot_stride
                or layout.off_sr_yuv != link.max_bytes):
            raise ValueError("incompatible frame cache and peer link")
        self.link = link
        self.cache = cache
        self._registration = link.register(cache.addr, layout.total_size)

    def _offset(self, slot: int) -> int:
        if not isinstance(slot, int) or not 0 <= slot < self.cache.layout.n_slots:
            raise ValueError("invalid frame cache slot")
        return slot * self.cache.layout.slot_stride

    def send(self, generation: int, frame: int, slot: int) -> None:
        self.link.send_registered(generation, frame, self._registration,
                                  self.link.max_bytes, offset=self._offset(slot))

    def receive(self, generation: int, frame: int, slot: int) -> None:
        actual = self.link.receive_registered(self._registration,
                                              offset=self._offset(slot))
        if actual != (generation, frame, self.link.max_bytes):
            # The slot may already contain an unrelated or partial frame.
            # Discard the QP so later commands cannot consume that data.
            self.link.close()
            raise ValueError("unexpected peer frame metadata or size")

    def close(self) -> None:
        self._registration.close()


class FrameCacheWritePeer(PeerLink):
    """Write prepared frames directly into a peer's registered cache.

    Both rails must complete before the source and destination slots may
    be reused. Session shutdown drains or closes these QPs before either
    cache is unmapped; the scheduler owns the per-frame slot reservations.
    """

    def __init__(self, rails, *, cache, peer_ip, port, is_server,
                 token, timeout=10.0):
        if not rails or len(token) != 16 or timeout <= 0:
            raise ValueError("invalid frame-cache peer settings")
        from rdma_transport import RDMAChannel, RDMAContext
        self.cache = cache
        self.max_bytes = cache.layout.off_sr_yuv
        self.timeout = timeout
        self._lock = threading.RLock()
        self._failed = False
        self._closed = False
        self._ctx = []
        self._mrs = []
        self._channels = []
        self._remote = []
        try:
            for dev, gid in rails:
                self._ctx.append(RDMAContext(dev_name=dev, gid_index=gid, max_cqe=16))
            for ctx in self._ctx:
                self._mrs.append(ctx.register_region(cache.addr, cache.layout.total_size))
            for i, ctx in enumerate(self._ctx):
                hello = _WRITE_HELLO.pack(
                    b"PW01", len(rails), cache.layout.n_slots,
                    cache.layout.slot_stride, self.max_bytes, cache.addr,
                    self._mrs[i].rkey) + token
                channel = RDMAChannel(ctx, is_server=is_server, peer_ip=peer_ip,
                                      peer_port=port + i, extra_payload=hello,
                                      handshake_timeout=timeout)
                self._channels.append(channel)
                raw = channel.peer_extra
                if len(raw) != _WRITE_HELLO.size + len(token) or raw[-len(token):] != token:
                    raise ValueError("peer cache protocol or token mismatch")
                magic, nrails, slots, stride, size, address, rkey = _WRITE_HELLO.unpack(
                    raw[:_WRITE_HELLO.size])
                if (magic != b"PW01" or nrails != len(rails)
                        or slots != cache.layout.n_slots
                        or stride != cache.layout.slot_stride
                        or size != self.max_bytes or address == 0):
                    raise ValueError("peer cache layout mismatch")
                self._remote.append((address, rkey))
        except Exception:
            self.close()
            raise

    def write(self, source_slot, destination_slot):
        if (type(source_slot) is not int or type(destination_slot) is not int
                or not 0 <= source_slot < self.cache.layout.n_slots
                or not 0 <= destination_slot < self.cache.layout.n_slots):
            raise ValueError("invalid peer cache slot")
        with self._lock:
            if self._closed or self._failed:
                raise RuntimeError("peer link is closed or failed")
            try:
                for i, (offset, size) in enumerate(
                        _partitions(self.max_bytes, len(self._channels))):
                    address, rkey = self._remote[i]
                    self._channels[i].post_write(
                        src_addr=self.cache.addr + source_slot * self.cache.layout.slot_stride + offset,
                        src_lkey=self._mrs[i].lkey, length=size,
                        remote_addr=address + destination_slot * self.cache.layout.slot_stride + offset,
                        rkey=rkey, wr_id=i)
                for i in range(len(self._channels)):
                    self._wait(i)
            except Exception:
                self._failed = True
                raise

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._failed = True
            for channel in self._channels:
                channel.qp.close()
            for mr in self._mrs:
                mr.close()
            for ctx in self._ctx:
                ctx.cq.close()
                ctx.pd.close()
                ctx.ctx.close()
