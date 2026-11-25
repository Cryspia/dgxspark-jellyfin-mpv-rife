"""Frame-owned remote scheduling using worker-local caches and peer links."""
from __future__ import annotations

import threading
import os
import sys
import time
from bisect import bisect_left, bisect_right
from concurrent.futures import ThreadPoolExecutor

from frame_scheduler import FrameScheduler, Kind


class FrameSet:
    """Represent long runs of frame numbers without one object per frame."""

    def __init__(self):
        self._starts = []
        self._ends = []

    def __contains__(self, frame):
        i = bisect_right(self._starts, frame) - 1
        return i >= 0 and frame < self._ends[i]

    def add(self, frame):
        if frame in self:
            return
        i = bisect_left(self._starts, frame)
        start, end = frame, frame + 1
        if i and self._ends[i - 1] == frame:
            start = self._starts.pop(i - 1)
            self._ends.pop(i - 1)
            i -= 1
        if i < len(self._starts) and self._starts[i] == end:
            end = self._ends.pop(i)
            self._starts.pop(i)
        self._starts.insert(i, start)
        self._ends.insert(i, end)


class TicTacRuntime:
    """Drive a FrameScheduler against established RemoteEndpoint objects.

    Input planes are packed immutable bytes. Output frames live in owned
    host memory until the caller consumes them, independent of worker slot
    reuse. `links[(source, destination)]` names a directed peer link that
    exists on both endpoints.
    """

    def __init__(self, endpoints, links, *, interp_mult=2, fused_interp_sr=False,
                 overlap_output=False, peer_write=False):
        if len(endpoints) < 2:
            raise ValueError("tic-tac requires at least two remote workers")
        self.endpoints = dict(endpoints)
        self.links = dict(links)
        self._pull_coordinator = None
        if os.environ.get("CLUSTER_PULL_RESULTS", "0") == "1":
            from pull_queue import PullCoordinator
            self._pull_coordinator = PullCoordinator()
            for endpoint in self.endpoints.values():
                endpoint.guest._pull_controller = self._pull_coordinator
        self.mult = interp_mult
        self.fused_interp_sr = fused_interp_sr and interp_mult == 2
        self.overlap_output = overlap_output and self.fused_interp_sr
        self.peer_write = peer_write
        self.scheduler = FrameScheduler(list(endpoints), interp_mult=interp_mult,
                                        prefetch=True, fused_ccsr=True,
                                        frame_slots=3)
        self._cv = threading.Condition(threading.RLock())
        self._sources = {}
        self._results = {}
        self._running = {}
        self._ahead_callbacks = {}
        self._consumed = {}
        self._forgotten = FrameSet()
        self._closed = False
        self._error = None
        self.failed_workers = set()
        self.failure_identified = threading.Event()
        self._profile = os.environ.get("CLUSTER_TICTAC_PROFILE", "0") == "1"
        self._profile_every = max(1, int(os.environ.get("CLUSTER_TICTAC_PROFILE_EVERY", "100")))
        self._profile_start = {}
        self._profile_data = {}
        self._pool = ThreadPoolExecutor(max_workers=max(8, 4 * len(endpoints)),
                                        thread_name_prefix="tictac")
        for name, endpoint in self.endpoints.items():
            endpoint.on_failure = lambda reason, worker=name: self._abort(
                RuntimeError(f"worker {worker}: {reason}"), failed={worker})
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="tictac-scheduler")
        self._thread.start()

    def submit_interval(self, frame, first, second):
        """Retain a pair of packed YUV planes for asynchronous dispatch."""
        if frame < 0 or len(first) != 3 or len(second) != 3:
            raise ValueError("invalid input interval")
        self.submit_interval_lazy(frame, lambda: first, lambda: second)

    def submit_interval_lazy(self, frame, first, second):
        """Copy each input only if it is not already retained by a neighbor."""
        if frame < 0:
            raise ValueError("invalid input interval")
        with self._cv:
            if self._closed:
                raise RuntimeError("tic-tac runtime is closed")
            if frame in self._forgotten:
                raise ValueError("interval was already consumed")
            for index, factory in ((frame, first), (frame + 1, second)):
                if index not in self._sources:
                    planes = factory()
                    if len(planes) != 3:
                        raise ValueError("invalid input interval")
                    self._sources[index] = tuple(bytes(p) for p in planes)
            self.scheduler.submit_interval(frame)
            self._cv.notify_all()

    def wait_output(self, frame, phase, timeout=None):
        if frame < 0 or not 0 <= phase < self.mult:
            raise ValueError("invalid output phase")
        with self._cv:
            ready = self._cv.wait_for(lambda: (frame, phase) in self._results
                                      or self._error is not None or self._closed,
                                      timeout)
            if not ready:
                raise TimeoutError("tic-tac output timed out")
            if (frame, phase) in self._results:
                return self._results[(frame, phase)]
            raise RuntimeError(f"tic-tac stopped: {self._error or 'closed'}")

    def consume_output(self, frame, phase):
        """Release host bookkeeping after a caller has copied an output."""
        with self._cv:
            if (frame, phase) not in self._results:
                raise ValueError("output is not available")
            del self._results[(frame, phase)]
            self._consumed.setdefault(frame, set()).add(phase)
            self._retire_completed()
            self._cv.notify_all()

    def _retire_completed(self):
        for frame, phases in sorted(self._consumed.items()):
            if len(phases) != self.mult:
                continue
            try:
                self.scheduler.forget_interval(frame)
            except ValueError:
                continue
            self._sources.pop(frame, None)
            del self._consumed[frame]
            self._forgotten.add(frame)

    def _wake(self):
        with self._cv:
            self._cv.notify_all()

    def _promote_ahead(self, worker):
        ahead_key = (worker, 'ahead')
        promoted = self._running.get(ahead_key)
        if promoted is None or self.scheduler._busy.get(worker) != promoted:
            return
        self._running.pop(ahead_key)
        self._running[(worker, 'compute')] = promoted
        pending = self._ahead_callbacks.pop(promoted, {})
        if 'compute' in pending:
            key, future = pending['compute']
            self._compute_finished(key, promoted, future)
        if 'finish' in pending:
            key, future = pending['finish']
            self._finish(key, promoted, future)

    def _finish(self, key, command, future):
        try:
            result = future.result()
            with self._cv:
                if self._closed:
                    return
                if self.scheduler._ahead.get(command.worker) == command:
                    self._ahead_callbacks.setdefault(command, {})['finish'] = (key, future)
                    return
                if command.kind is Kind.CCSR:
                    self._results[(command.frame, 0)] = result
                elif command.kind is Kind.SUPER_RESOLVE_INTERPOLATED:
                    self._results[(command.frame, command.phase)] = result
                elif command.kind is Kind.INTERPOLATE and self.fused_interp_sr:
                    self._results[(command.frame, 1)] = result
                if command.kind is Kind.INTERPOLATE and self.fused_interp_sr:
                    self.scheduler.complete_fused_interpolation(command)
                else:
                    self.scheduler.complete(command)
                self._profile_end(command, command.kind.name)
                for running_key, running in tuple(self._running.items()):
                    if running == command:
                        self._running.pop(running_key, None)
                self._promote_ahead(command.worker)
                self._cv.notify_all()
        except Exception as exc:
            self._abort(exc)

    def _prepared(self, command, future):
        try:
            future.result()
            with self._cv:
                if not self._closed:
                    self.scheduler.report_prepared(command)
                    self._cv.notify_all()
        except Exception as exc:
            self._abort(exc)

    def _compute_finished(self, key, command, future):
        try:
            future.result()
            with self._cv:
                if not self._closed:
                    if self.scheduler._ahead.get(command.worker) == command:
                        self._ahead_callbacks.setdefault(command, {})['compute'] = (key, future)
                        return
                    self.scheduler.report_compute_done(
                        command, fused=command.kind is Kind.INTERPOLATE)
                    for running_key, running in tuple(self._running.items()):
                        if running == command:
                            self._running.pop(running_key, None)
                    self._promote_ahead(command.worker)
                    self._cv.notify_all()
        except Exception as exc:
            self._abort(exc)

    def _execute(self, key, command):
        try:
            if os.environ.get("CLUSTER_SPLIT_DEBUG", "0") == "1":
                sys.stderr.write(f"[tictac] worker={command.worker} "
                                 f"task={command.kind.name} frame={command.frame}\n")
            endpoint = self.endpoints[command.worker]
            source, neighbor = self.scheduler.cache_slots(command.worker, command.frame)
            if command.kind is Kind.CCSR:
                task = endpoint.submit_frame(command.frame, self._sources[command.frame],
                                             source)
                task.prepared.add_done_callback(lambda f: self._prepared(command, f))
                if self.overlap_output:
                    task.compute_done.add_done_callback(
                        lambda f: self._compute_finished(key, command, f))
                task.done.add_done_callback(lambda f: self._finish(key, command, f))
            elif command.kind is Kind.INTERPOLATE:
                outputs = self.scheduler.interpolation_slots(command.worker, command.frame)
                if self.fused_interp_sr:
                    task = endpoint.interpolate_and_super_resolve(
                        command.frame, source, neighbor, outputs[0])
                else:
                    task = endpoint.interpolate(command.frame, source, neighbor, outputs)
                if self.overlap_output:
                    task.compute_done.add_done_callback(
                        lambda f: self._compute_finished(key, command, f))
                task.done.add_done_callback(lambda f: self._finish(key, command, f))
            elif command.kind is Kind.SUPER_RESOLVE_INTERPOLATED:
                outputs = self.scheduler.interpolation_slots(command.worker, command.frame)
                task = endpoint.super_resolve(command.frame, outputs[command.phase - 1], phase=command.phase)
                task.done.add_done_callback(lambda f: self._finish(key, command, f))
            elif command.kind is Kind.TRANSFER:
                src, dst = self.scheduler.transfer_slots(command)
                link = self.links[(command.worker, command.peer)]
                if self.peer_write:
                    write = endpoint.control.submit("write", link,
                        generation=command.generation, frame=command.frame,
                        slot=src, destination_slot=dst)
                    write.result(20)
                else:
                    receiver = self.endpoints[command.peer].control
                    receive = receiver.submit("receive", link, generation=command.generation,
                                              frame=command.frame, slot=dst)
                    send = endpoint.control.submit("send", link, generation=command.generation,
                                                   frame=command.frame, slot=src)
                    send.result(20)
                    receive.result(20)
                with self._cv:
                    if not self._closed:
                        self.scheduler.complete(command)
                        self._profile_end(command, command.kind.name)
                        self._running.pop(key, None)
                        self._cv.notify_all()
            else:
                raise RuntimeError(f"unsupported scheduler command {command.kind}")
        except Exception as exc:
            self._abort(exc)

    def _loop(self):
        workers = tuple(self.endpoints)
        cursor = 0
        while True:
            with self._cv:
                if self._closed:
                    return
                made_progress = False
                try:
                    # Rotate the first checked worker when several become
                    # ready together; idle workers still claim immediately.
                    order = workers[cursor:] + workers[:cursor]
                    cursor = (cursor + 1) % len(workers)
                    for worker in order:
                        for lane, claim in (("compute", self.scheduler.claim),
                                            ('ahead', self.scheduler.claim_ahead),
                                            ("peer", self.scheduler.claim_peer_transfer)):
                            key = (worker, lane)
                            if key in self._running:
                                continue
                            command = claim(worker)
                            if command is not None:
                                self._running[key] = command
                                if self._profile:
                                    self._profile_start[command] = time.monotonic()
                                self._pool.submit(self._execute, key, command)
                                made_progress = True
                        output = self.scheduler.claim_host_output(worker)
                        if output is not None:
                            self.scheduler.complete(output)
                            made_progress = True
                        for frame in tuple(self.scheduler._slots[worker]):
                            try:
                                self.scheduler.release_frame(worker, frame)
                                made_progress = True
                            except RuntimeError:
                                pass
                    self._retire_completed()
                except Exception as exc:
                    self._abort(exc)
                    return
                if not made_progress:
                    self._cv.wait(timeout=0.05)

    def _profile_end(self, command, kind):
        if not self._profile:
            return
        start = self._profile_start.pop(command, None)
        if start is None:
            return
        data = self._profile_data.setdefault(kind, [])
        data.append((time.monotonic() - start) * 1000)
        if len(data) == self._profile_every:
            ordered = sorted(data)
            p50 = ordered[(len(ordered) - 1) * 50 // 100]
            p95 = ordered[(len(ordered) - 1) * 95 // 100]
            sys.stderr.write(f"[tictac profile] {kind} n={len(data)} "
                             f"mean={sum(data)/len(data):.2f}ms "
                             f"p50={p50:.2f}ms p95={p95:.2f}ms\n")
            data.clear()

    def _abort(self, error, *, failed=()):
        with self._cv:
            if self._closed:
                self.failed_workers.update(failed)
                return
            self._closed = True
            self._error = error
            self.failed_workers = set(failed) | {
                name for name, endpoint in self.endpoints.items()
                if getattr(endpoint, "_closed", False)
                or getattr(endpoint.control, "_error", None) is not None
            }
            self._cv.notify_all()
        threading.Thread(target=self._diagnose_and_close, daemon=True,
                         name="tictac-abort").start()

    def _diagnose_and_close(self):
        # Control and data QPs can report the same loss in either order.
        # Let the liveness reader identify the failed worker before closing
        # the remaining healthy sessions ourselves.
        time.sleep(0.5)
        with self._cv:
            self.failed_workers.update(
                name for name, endpoint in self.endpoints.items()
                if getattr(endpoint, "_closed", False)
                or getattr(endpoint.control, "_error", None) is not None)
            self.failure_identified.set()
        self._close_endpoints()

    def _close_endpoints(self):
        for endpoint in self.endpoints.values():
            try:
                endpoint.close()
            except Exception:
                pass

    def close(self):
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        self._close_endpoints()
        if self._pull_coordinator is not None:
            self._pull_coordinator.close()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=5)
        self._pool.shutdown(wait=True, cancel_futures=True)
