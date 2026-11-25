"""CPU-only scheduling state for frame-owned remote processing.

The transport and worker runtime must execute the returned commands and
report completion; this module deliberately does not perform any I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from threading import RLock


class Kind(Enum):
    PREPARE = auto()
    CCSR = auto()
    PREFETCH = auto()
    SUPER_RESOLVE = auto()
    TRANSFER = auto()
    INTERPOLATE = auto()
    SUPER_RESOLVE_INTERPOLATED = auto()
    OUTPUT_FRAME = auto()
    OUTPUT_INTERPOLATED = auto()


@dataclass(frozen=True)
class Command:
    generation: int
    kind: Kind
    frame: int
    worker: str
    peer: str | None = None
    phase: int = 0
    prefetched: bool = False


@dataclass
class _Frame:
    owner: str | None = None
    prefetched: bool = False
    prefetch_running: bool = False
    prepared: bool = False
    sr_done: bool = False
    ccsr_compute_done: bool = False
    neighbor_ready: bool = False
    transfer_running: bool = False
    interpolated: bool = False
    fused_output_pending: bool = False
    interp_sr_done: set[int] = field(default_factory=set)
    frame_output_running: bool = False
    frame_output_done: bool = False
    interp_output_running: set[int] = field(default_factory=set)
    interp_output_done: set[int] = field(default_factory=set)
    released: bool = False


class FrameScheduler:
    """Assign each frame on demand and keep its interval on its owner.

    A worker has independent compute, peer-transfer and host-output
    lanes. TRANSFER copies prepared frame K+1 from its owner to K's
    owner, while either worker may continue computing. OUTPUT commands
    publish completed frames to the playback host.
    A generation reset invalidates late completions after a seek.
    """

    def __init__(self, workers: list[str], *, interp_mult: int = 2,
                 prefetch: bool = False, fused_ccsr: bool = False,
                 frame_slots: int = 2):
        if len(workers) < 2 or len(set(workers)) != len(workers) or any(not w for w in workers):
            raise ValueError("at least two distinct workers are required")
        if interp_mult not in (1, 2, 3, 4):
            raise ValueError("interpolation multiplier must be 1, 2, 3 or 4")
        # Each reservation uses two input slots and up to three output
        # slots in the worker's 32-slot local frame cache.
        if not 2 <= frame_slots <= 6:
            raise ValueError("frame slots must be between 2 and 6")
        self._workers = tuple(workers)
        self._interp_mult = interp_mult
        self._prefetch_enabled = prefetch
        self._frame_slots = frame_slots
        self._fused_ccsr = fused_ccsr
        self._lock = RLock()
        self._generation = 0
        self._frames: dict[int, _Frame] = {}
        self._intervals: set[int] = set()
        self._busy: dict[str, Command] = {}
        self._ahead: dict[str, Command] = {}
        self._finishing: set[Command] = set()
        self._prefetch_busy: dict[str, Command] = {}
        self._prefetch_reserved: dict[str, int] = {}
        self._slots: dict[str, set[int]] = {worker: set() for worker in workers}
        # Each reservation owns its source and a destination for the
        # following frame's features. Neighbor copies cannot overwrite
        # another reserved source frame.
        self._cache_slots: dict[str, dict[int, int]] = {
            worker: {} for worker in workers
        }
        self._peer_busy: dict[str, Command] = {}
        self._output_busy: dict[str, Command] = {}
        self._retired: set[Command] = set()

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def reset(self) -> int:
        with self._lock:
            # An epoch change invalidates results immediately, but it
            # does not stop a NIC or GPU that still owns an old buffer.
            # Do not issue new work until every old operation has been
            # acknowledged complete or cancelled after transport drain.
            for lane in (self._busy, self._ahead, self._prefetch_busy, self._peer_busy,
                         self._output_busy):
                self._retired.update(lane.values())
            self._retired.update(self._finishing)
            self._generation += 1
            self._frames.clear()
            self._intervals.clear()
            self._busy.clear()
            self._ahead.clear()
            self._finishing.clear()
            self._prefetch_busy.clear()
            self._prefetch_reserved.clear()
            for slots in self._slots.values():
                slots.clear()
            for slots in self._cache_slots.values():
                slots.clear()
            self._peer_busy.clear()
            self._output_busy.clear()
            return self._generation

    def submit_interval(self, frame: int) -> None:
        if frame < 0:
            raise ValueError("negative frame")
        with self._lock:
            self._intervals.add(frame)
            self._frames.setdefault(frame, _Frame())
            self._frames.setdefault(frame + 1, _Frame())

    def claim(self, worker: str) -> Command | None:
        with self._lock:
            if worker not in self._workers:
                raise ValueError("unknown worker")
            if self._retired:
                return None
            if worker in self._busy:
                raise RuntimeError("worker already has an outstanding command")
            # Prefer source-frame work across all owned slots.
            for k, f in sorted(self._frames.items()):
                if (f.owner == worker and f.prepared and not f.sr_done
                        and not f.ccsr_compute_done):
                    return self._start(Kind.SUPER_RESOLVE, k, worker)
            # Publish a prefetched frame's features before starting another
            # interpolation so that its predecessor can run on a peer.
            if worker in self._prefetch_reserved:
                k = self._prefetch_reserved[worker]
                if not self._frames[k].prefetch_running:
                    del self._prefetch_reserved[worker]
                    kind = Kind.CCSR if self._fused_ccsr else Kind.PREPARE
                    return self._start(kind, k, worker, prefetched=True)
            if worker not in self._prefetch_reserved:
                source = self._claim_source(worker, ahead=False)
                if source is not None:
                    return source
            for k, f in sorted(self._frames.items()):
                if f.owner == worker and f.interpolated and not f.fused_output_pending:
                    for phase in range(1, self._interp_mult):
                        if phase not in f.interp_sr_done:
                            return self._start(Kind.SUPER_RESOLVE_INTERPOLATED,
                                               k, worker, phase=phase)
            for k in sorted(self._intervals):
                f, nxt = self._frames[k], self._frames[k + 1]
                local_neighbor = nxt.owner == worker and nxt.prepared
                if (self._interp_mult > 1 and f.owner == worker and f.prepared
                        and (f.sr_done or f.ccsr_compute_done)
                        and not f.interpolated and (f.neighbor_ready or local_neighbor)):
                    return self._start(Kind.INTERPOLATE, k, worker)
            if worker in self._prefetch_reserved:
                return None
            for k, f in sorted(self._frames.items()):
                if f.owner is None:
                    if self._can_reserve(worker, k):
                        f.owner = worker
                        if self._prefetch_enabled:
                            self._reserve(worker, k)
                        kind = Kind.CCSR if self._fused_ccsr else Kind.PREPARE
                        return self._start(kind, k, worker)
                    # Do not fill later slots while an earlier source has
                    # no eligible owner; that can trap its two neighbors.
                    return None
            return None

    def claim_prefetch(self, worker: str) -> Command | None:
        """Reserve a nonadjacent next frame while its worker computes.

        PREFETCH only transfers input. PREPARE is issued after the
        prefetch completion and after the compute lane becomes free.
        At most one input transfer is prefetched at a time; completed
        frames may continue to hold other nonadjacent cache slots.
        """
        with self._lock:
            if not self._prefetch_enabled:
                raise RuntimeError("prefetch is not enabled")
            if self._retired:
                return None
            self._check_lane(worker, self._prefetch_busy)
            current = self._busy.get(worker)
            if current is None or worker in self._prefetch_reserved:
                return None
            for k, f in sorted(self._frames.items()):
                if (f.owner is None and abs(k - current.frame) > 1
                        and self._can_reserve(worker, k)):
                    f.owner = worker
                    f.prefetch_running = True
                    self._reserve(worker, k)
                    self._prefetch_reserved[worker] = k
                    command = Command(self._generation, Kind.PREFETCH, k, worker)
                    self._prefetch_busy[worker] = command
                    return command
            return None

    def claim_ahead(self, worker: str) -> Command | None:
        """Queue one independent task behind the worker's running work."""
        with self._lock:
            if worker not in self._workers:
                raise ValueError("unknown worker")
            current = self._busy.get(worker)
            if self._retired or current is None or worker in self._ahead:
                return None
            source = self._claim_source(worker, ahead=True)
            if source is not None:
                return source
            # A multi-output interpolation has independent SR jobs.
            # Queue the next phase while the current result is in transit
            # so the worker does not wait for a host completion round trip.
            if current.kind is Kind.SUPER_RESOLVE_INTERPOLATED:
                frame = self._frames[current.frame]
                for phase in range(1, self._interp_mult):
                    if phase != current.phase and phase not in frame.interp_sr_done:
                        command = Command(self._generation, Kind.SUPER_RESOLVE_INTERPOLATED,
                                          current.frame, worker, phase=phase)
                        self._ahead[worker] = command
                        return command
            for k in sorted(self._intervals):
                f, nxt = self._frames[k], self._frames[k + 1]
                if (k != current.frame and f.owner == worker and f.prepared
                        and (f.sr_done or f.ccsr_compute_done)
                        and not f.interpolated
                        and (f.neighbor_ready or (nxt.owner == worker and nxt.prepared))):
                    command = Command(self._generation, Kind.INTERPOLATE, k, worker)
                    self._ahead[worker] = command
                    return command
            return None

    def _claim_source(self, worker: str, *, ahead: bool) -> Command | None:
        """Reserve the earliest eligible source without skipping a gap."""
        if not (self._prefetch_enabled and self._fused_ccsr):
            return None
        for k, frame in sorted(self._frames.items()):
            if frame.owner is not None:
                continue
            if not self._can_reserve(worker, k):
                return None
            frame.owner = worker
            self._reserve(worker, k)
            if not ahead:
                return self._start(Kind.CCSR, k, worker)
            command = Command(self._generation, Kind.CCSR, k, worker)
            self._ahead[worker] = command
            return command
        return None

    def _can_reserve(self, worker: str, frame: int) -> bool:
        if not self._prefetch_enabled:
            return True
        slots = self._slots[worker]
        if len(slots) >= self._frame_slots or any(abs(frame - other) <= 1 for other in slots):
            return False
        if len(self._workers) == 2:
            # Out-of-order source arrival can reveal a missing run only
            # after one of its endpoints has already been reserved. With
            # two workers, the endpoints must admit alternating ownership
            # across that hole; otherwise an interior frame can fit on
            # neither worker while both endpoints wait for it.
            for owner, held in self._slots.items():
                for other in held:
                    same_parity = (frame - other) % 2 == 0
                    if same_parity == (worker == owner):
                        continue
                    lo, hi = sorted((frame, other))
                    if any(lo < k < hi and f.owner is None
                           for k, f in self._frames.items()):
                        return False
        return True

    def _reserve(self, worker: str, frame: int) -> None:
        slots = self._cache_slots[worker]
        used = set(slots.values())
        pair = next(pair for pair in range(self._frame_slots) if pair not in used)
        slots[frame] = pair
        self._slots[worker].add(frame)

    def cache_slots(self, worker: str, frame: int) -> tuple[int, int]:
        """Return the source and neighbor cache slots held for a frame."""
        with self._lock:
            if not self._prefetch_enabled or worker not in self._cache_slots:
                raise ValueError("unknown worker or prefetch disabled")
            try:
                pair = self._cache_slots[worker][frame]
            except KeyError:
                raise ValueError("frame is not held by this worker") from None
            return 2 * pair, 2 * pair + 1

    def interpolation_slots(self, worker: str, frame: int) -> tuple[int, ...]:
        """Return separate output slots for this frame's interpolated phases."""
        with self._lock:
            source, _ = self.cache_slots(worker, frame)
            pair = source // 2
            return tuple(2 * self._frame_slots + pair * 3 + phase
                         for phase in range(self._interp_mult - 1))

    def transfer_slots(self, command: Command) -> tuple[int, int]:
        """Return the source and destination slots of an in-flight transfer."""
        with self._lock:
            if (command.kind is not Kind.TRANSFER
                    or command.generation != self._generation
                    or self._peer_busy.get(command.worker) != command
                    or command.peer is None):
                raise ValueError("peer transfer is not outstanding")
            source, _ = self.cache_slots(command.worker, command.frame)
            _, destination = self.cache_slots(command.peer, command.frame - 1)
            return source, destination

    def release_frame(self, worker: str, frame: int, *, final: bool = False) -> None:
        """Release a slot after all outputs and neighbor uses finish."""
        with self._lock:
            if not self._prefetch_enabled or worker not in self._slots:
                raise ValueError("unknown worker or prefetch disabled")
            if frame not in self._slots[worker]:
                raise ValueError("frame is not held by this worker")
            f = self._frames[frame]
            if not (f.prepared and f.sr_done and f.frame_output_done):
                raise RuntimeError("real-frame output is not complete")
            phases = set(range(1, self._interp_mult))
            if frame in self._intervals:
                if phases and (not f.interpolated or f.interp_sr_done != phases
                               or f.interp_output_done != phases):
                    raise RuntimeError("interpolated outputs are not complete")
            elif not final:
                raise RuntimeError("future interval ownership is not final")
            if phases and frame - 1 in self._intervals:
                previous = self._frames[frame - 1]
                if previous.owner is None:
                    raise RuntimeError("previous interval has no owner")
                if previous.owner == worker and not previous.interpolated:
                    raise RuntimeError("previous interval still needs this frame")
                if previous.owner != worker and not previous.neighbor_ready:
                    raise RuntimeError("neighbor transfer is not complete")
            for lane in (self._busy, self._ahead, self._prefetch_busy, self._peer_busy,
                         self._output_busy):
                command = lane.get(worker)
                if command is not None and command.frame == frame:
                    raise RuntimeError("frame still has in-flight work")
            self._slots[worker].remove(frame)
            del self._cache_slots[worker][frame]
            f.released = True

    def forget_interval(self, frame: int) -> None:
        """Forget a released interval after its host outputs are consumed."""
        with self._lock:
            f = self._frames.get(frame)
            if f is None or frame not in self._intervals or not f.released:
                raise ValueError("interval has not been released")
            if frame - 1 in self._intervals:
                raise ValueError("previous interval still references this frame")
            # Its interpolation has completed before release, so the
            # successor no longer depends on this frame's bookkeeping.
            del self._frames[frame]
            self._intervals.remove(frame)

    def claim_peer_transfer(self, worker: str) -> Command | None:
        """Claim a prepared neighbor without reserving a compute lane."""
        with self._lock:
            if self._retired:
                return None
            self._check_lane(worker, self._peer_busy)
            if self._interp_mult == 1:
                return None
            for k in sorted(self._intervals):
                f, nxt = self._frames[k], self._frames[k + 1]
                if (nxt.owner == worker and nxt.prepared and f.owner is not None
                        and f.owner != worker and not f.neighbor_ready
                        and not f.transfer_running):
                    f.transfer_running = True
                    cmd = Command(self._generation, Kind.TRANSFER, k + 1,
                                  worker, f.owner)
                    self._peer_busy[worker] = cmd
                    return cmd
            return None

    def claim_host_output(self, worker: str) -> Command | None:
        """Publish finished results independently of peer transfers."""
        with self._lock:
            if self._retired:
                return None
            self._check_lane(worker, self._output_busy)
            for k, f in sorted(self._frames.items()):
                if f.owner != worker:
                    continue
                if f.sr_done and not f.frame_output_done and not f.frame_output_running:
                    f.frame_output_running = True
                    cmd = Command(self._generation, Kind.OUTPUT_FRAME, k, worker)
                else:
                    cmd = None
                    for phase in range(1, self._interp_mult):
                        if (phase in f.interp_sr_done
                                and phase not in f.interp_output_done
                                and phase not in f.interp_output_running):
                            f.interp_output_running.add(phase)
                            cmd = Command(self._generation, Kind.OUTPUT_INTERPOLATED,
                                          k, worker, phase=phase)
                            break
                    if cmd is None:
                        continue
                self._output_busy[worker] = cmd
                return cmd
            return None

    def _check_lane(self, worker: str, busy: dict[str, Command]) -> None:
        if worker not in self._workers:
            raise ValueError("unknown worker")
        if worker in busy:
            raise RuntimeError("worker lane already has an outstanding command")

    def _start(self, kind: Kind, frame: int, worker: str,
               peer: str | None = None, phase: int = 0,
               prefetched: bool = False) -> Command:
        cmd = Command(self._generation, kind, frame, worker, peer, phase, prefetched)
        self._busy[worker] = cmd
        return cmd

    def report_prepared(self, command: Command) -> bool:
        """Publish CCSR's encoded frame before its real-frame SR finishes."""
        with self._lock:
            if command.generation != self._generation:
                return False
            if (command.kind is not Kind.CCSR or
                    (self._busy.get(command.worker) != command
                     and self._ahead.get(command.worker) != command)):
                raise ValueError("CCSR command is not outstanding")
            self._frames[command.frame].prepared = True
            return True

    def complete_fused_interpolation(self, command: Command) -> bool:
        """Complete x2 interpolation and its worker-local SR together."""
        with self._lock:
            if self._interp_mult != 2 or command.kind is not Kind.INTERPOLATE:
                raise ValueError("fused interpolation requires x2")
            if not self.complete(command):
                return False
            self._frames[command.frame].fused_output_pending = False
            self._frames[command.frame].interp_sr_done.add(1)
            return True

    def report_compute_done(self, command: Command, *, fused=False) -> bool:
        """Release the GPU lane while the completed result is in transit."""
        with self._lock:
            if command.generation != self._generation:
                return False
            if self._busy.get(command.worker) != command:
                raise ValueError("compute command is not outstanding")
            if command.kind is Kind.CCSR:
                self._frames[command.frame].ccsr_compute_done = True
            elif command.kind is Kind.INTERPOLATE and fused and self._interp_mult == 2:
                f = self._frames[command.frame]
                f.interpolated = True
                f.fused_output_pending = True
            else:
                raise ValueError("unsupported early compute completion")
            del self._busy[command.worker]
            self._finishing.add(command)
            if command.worker in self._ahead:
                self._busy[command.worker] = self._ahead.pop(command.worker)
            return True

    def complete(self, command: Command) -> bool:
        """Complete an outstanding command; ignore stale generations."""
        with self._lock:
            if command.generation != self._generation:
                self._retired.discard(command)
                return False
            lane = (self._prefetch_busy if command.kind is Kind.PREFETCH else
                    self._peer_busy if command.kind is Kind.TRANSFER else
                    self._output_busy if command.kind in
                    (Kind.OUTPUT_FRAME, Kind.OUTPUT_INTERPOLATED) else self._busy)
            if command in self._finishing:
                self._finishing.remove(command)
            else:
                if lane.get(command.worker) != command:
                    raise ValueError("command is not outstanding")
                del lane[command.worker]
                if lane is self._busy and command.worker in self._ahead:
                    self._busy[command.worker] = self._ahead.pop(command.worker)
            f = self._frames[command.frame]
            if command.kind is Kind.PREPARE:
                f.prepared = True
            elif command.kind is Kind.CCSR:
                f.prepared = True
                f.sr_done = True
            elif command.kind is Kind.PREFETCH:
                f.prefetch_running = False
                f.prefetched = True
            elif command.kind is Kind.SUPER_RESOLVE:
                f.sr_done = True
            elif command.kind is Kind.TRANSFER:
                target = self._frames[command.frame - 1]
                target.transfer_running = False
                target.neighbor_ready = True
            elif command.kind is Kind.INTERPOLATE:
                f.interpolated = True
            elif command.kind is Kind.SUPER_RESOLVE_INTERPOLATED:
                f.interp_sr_done.add(command.phase)
            elif command.kind is Kind.OUTPUT_FRAME:
                f.frame_output_running = False
                f.frame_output_done = True
            elif command.kind is Kind.OUTPUT_INTERPOLATED:
                f.interp_output_running.remove(command.phase)
                f.interp_output_done.add(command.phase)
            return True

    def fail(self, command: Command) -> bool:
        """Invalidate the session after an in-flight command fails.

        Its transport may still be writing, so retrying in the same
        generation could publish stale or partially written data. The
        caller must drain the transports before submitting a new session.
        """
        with self._lock:
            if command.generation != self._generation:
                return False
            if not any(lane.get(command.worker) == command for lane in
                       (self._busy, self._ahead, self._prefetch_busy, self._peer_busy,
                        self._output_busy)):
                raise ValueError("command is not outstanding")
            self.reset()
            return True

    def acknowledge_drained(self, command: Command) -> bool:
        """Release a stale command only after its device/transport is drained."""
        with self._lock:
            if command not in self._retired:
                return False
            self._retired.remove(command)
            return True


class OutputReorder:
    """Keep out-of-order remote results until the next playback phase.

    This object manages ownership of Python payloads, not DMA buffers;
    the caller must keep any backing memory alive until pop_ready returns
    it and the renderer has finished consuming it.
    """

    def __init__(self, interp_mult: int, *, max_pending: int = 32):
        if interp_mult not in (1, 2, 3, 4) or max_pending < 1:
            raise ValueError("invalid multiplier or pending limit")
        self._mult = interp_mult
        self._limit = max_pending
        self._lock = RLock()
        self._generation = 0
        self._next = 0
        self._pending: dict[int, object] = {}

    def reset(self, *, start_frame: int = 0) -> int:
        if start_frame < 0:
            raise ValueError("negative start frame")
        with self._lock:
            self._generation += 1
            self._next = start_frame * self._mult
            self._pending.clear()
            return self._generation

    def submit(self, generation: int, frame: int, phase: int, payload: object) -> bool:
        if frame < 0 or not 0 <= phase < self._mult:
            raise ValueError("invalid output frame or phase")
        seq = frame * self._mult + phase
        with self._lock:
            if generation != self._generation or seq < self._next:
                return False
            if seq in self._pending:
                raise ValueError("duplicate output")
            # Reserve the missing next item even when later frames fill
            # the buffer, or backpressure can deadlock the pipeline.
            if len(self._pending) >= self._limit and seq != self._next:
                raise BufferError("output reorder buffer full")
            self._pending[seq] = payload
            return True

    def pop_ready(self) -> tuple[int, int, object] | None:
        with self._lock:
            if self._next not in self._pending:
                return None
            seq = self._next
            payload = self._pending.pop(seq)
            self._next += 1
            return seq // self._mult, seq % self._mult, payload
