"""Thread-safe priority queue for completed remote frames awaiting READ."""
from __future__ import annotations

import heapq
import itertools
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field


@dataclass(order=True, frozen=True)
class PullRequest:
    frame: int
    phase: int
    sequence: int
    worker: object = field(compare=False)
    host_slot: int = field(compare=False)
    worker_slot: int = field(compare=False)
    task_id: int = field(compare=False)


class PullQueue:
    """Prioritize ready frames without waiting for missing earlier frames.

    A request remains owned until ``complete``.  A transport must drain its
    READ before completing or discarding an in-flight request.
    """

    def __init__(self, max_inflight=1):
        if max_inflight < 1:
            raise ValueError("max_inflight must be positive")
        self._limit = max_inflight
        self._cv = threading.Condition()
        self._heap = []
        self._active = set()
        self._owned = set()
        self._sequence = itertools.count()
        self._closed = False

    def put(self, frame, phase, worker, host_slot, worker_slot, task_id):
        if min(frame, phase, host_slot, worker_slot) < 0 or task_id <= 0:
            raise ValueError("invalid result coordinates")
        with self._cv:
            if self._closed:
                raise RuntimeError("pull queue is closed")
            key = (worker, task_id)
            if key in self._owned:
                raise ValueError("result already queued")
            req = PullRequest(frame, phase, next(self._sequence), worker,
                              host_slot, worker_slot, task_id)
            self._owned.add(key)
            heapq.heappush(self._heap, req)
            self._cv.notify_all()
            return req

    def get(self, timeout=None):
        with self._cv:
            ready = self._cv.wait_for(
                lambda: self._closed or (self._heap and len(self._active) < self._limit),
                timeout)
            if not ready or self._closed:
                return None
            req = heapq.heappop(self._heap)
            self._active.add(req)
            return req

    def complete(self, req):
        with self._cv:
            if req not in self._active:
                raise ValueError("result is not in flight")
            self._active.remove(req)
            self._owned.remove((req.worker, req.task_id))
            self._cv.notify_all()

    def discard_worker(self, worker):
        """Discard queued requests; active requests must still be drained."""
        with self._cv:
            removed = [req for req in self._heap if req.worker == worker]
            self._heap = [req for req in self._heap if req.worker != worker]
            heapq.heapify(self._heap)
            for req in removed:
                self._owned.remove((req.worker, req.task_id))
            self._cv.notify_all()
            return removed

    def close(self):
        with self._cv:
            self._closed = True
            pending, self._heap = self._heap, []
            for req in pending:
                self._owned.remove((req.worker, req.task_id))
            self._cv.notify_all()
            return pending


class PullCoordinator:
    """One leader-side reader shared by every remote endpoint."""

    def __init__(self):
        self.queue = PullQueue()
        self._gate = threading.RLock()
        self._closed = False
        self._pending_lock = threading.Lock()
        self._pending = {}
        self._writebacks = ThreadPoolExecutor(max_workers=4, thread_name_prefix="pull-copy")
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="result-pull")
        self._thread.start()

    def enqueue(self, guest, host_slot, worker_slot):
        task = guest._task[host_slot]
        self.queue.put(task.frame, task.phase, guest, host_slot, worker_slot,
                       task.task_id)

    def detach(self, guest):
        self.queue.discard_worker(guest)
        with self._gate:
            guest._pull_detached = True
            with self._pending_lock:
                pending = tuple(self._pending.get(guest, ()))
        for future in pending:
            try:
                future.result()
            except Exception:
                # The completion callback reports the error; teardown still
                # has to fence every other writeback before freeing buffers.
                pass

    def _writeback_done(self, guest, future):
        with self._pending_lock:
            jobs = self._pending.get(guest)
            if jobs is not None:
                jobs.discard(future)
                if not jobs:
                    self._pending.pop(guest, None)
        try:
            future.result()
        except Exception as exc:
            if guest.transport_error_fn is not None:
                guest.transport_error_fn(f"result writeback failed: {exc}")

    def _run(self):
        while not self._closed:
            req = self.queue.get()
            if req is None:
                return
            guest = req.worker
            try:
                with self._gate:
                    if guest._pull_detached:
                        continue
                    guest.pull_result(req.host_slot, req.worker_slot, req.task_id)
                    future = self._writebacks.submit(guest.finish_pull, req.host_slot, req.task_id)
                    with self._pending_lock:
                        self._pending.setdefault(guest, set()).add(future)
                    future.add_done_callback(lambda f, g=guest: self._writeback_done(g, f))
            except Exception as exc:
                if guest.transport_error_fn is not None:
                    guest.transport_error_fn(f"result pull failed: {exc}")
            finally:
                self.queue.complete(req)

    def close(self):
        self._closed = True
        self.queue.close()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=20)
            if self._thread.is_alive():
                raise RuntimeError("result pull thread did not stop")
        self._writebacks.shutdown(wait=True)
