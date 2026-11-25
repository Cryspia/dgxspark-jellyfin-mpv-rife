"""Asynchronous frame-owned operations on an established remote worker."""
from __future__ import annotations

import ctypes
import queue
import threading
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass
from types import SimpleNamespace

from guest_mp import TT_CCSR, TT_INTERP, TT_SR_INTERP


def _running_future():
    future = Future()
    # DMA buffers must stay alive until the operation is drained.
    future.set_running_or_notify_cancel()
    return future


@dataclass
class FrameOutput:
    y: object
    u: object
    v: object
    y_stride: int
    uv_stride: int


class RemoteTask:
    def __init__(self, task_id, output, *, prefetched=False, stages=0):
        self.id = task_id
        self.output = output
        self.done = _running_future()
        self.compute_done = _running_future()
        self.prepared = _running_future()
        self.prefetched = _running_future()
        self.stages = tuple(_running_future() for _ in range(stages))
        if not prefetched:
            self.prefetched.set_result(None)


class RemoteEndpoint:
    """Own a GuestMP and its session control connection.

    All output allocations are kept alive through the final RDMA write and
    host copy. Task futures become failed only after the QPs are closed when
    a transport is lost, so callers cannot reuse an in-flight destination.
    """

    def __init__(self, guest, control):
        self.guest = guest
        self.control = control
        guest._pull_control = control
        self._lock = threading.Lock()
        self._submit_lock = threading.RLock()
        self._close_lock = threading.Lock()
        self._notifications = queue.SimpleQueue()
        self._notifier = threading.Thread(target=self._notify_loop, daemon=True,
                                          name="remote-endpoint-notifier")
        self._notifier.start()
        self._tasks = {}
        self._next_id = 1
        self._closed = False
        self.on_failure = None
        guest.task_done_fn = self._done
        guest.task_stage_done_fn = self._stage
        guest.task_prefetch_ready_fn = self._prefetched
        guest.task_compute_done_fn = self._compute_done
        guest.task_fail_fn = self._failed
        guest.transport_error_fn = self._transport_error
        control.on_disconnect = lambda error: self._transport_error(str(error))
        if getattr(control, "_error", None) is not None:
            self._transport_error(str(control._error))

    def _output(self):
        g = self.guest
        return FrameOutput(ctypes.create_string_buffer(g.oH * g.oW * 2),
                           ctypes.create_string_buffer(g.ocH * g.ocW * 2),
                           ctypes.create_string_buffer(g.ocH * g.ocW * 2),
                           g.oW * 2, g.ocW * 2)

    def _make_task(self, output, *, prefetched=False, stages=0):
        with self._lock:
            if self._closed:
                raise RuntimeError("worker endpoint is closed")
            task = RemoteTask(self._next_id, output, prefetched=prefetched,
                              stages=stages)
            self._next_id += 1
            self._tasks[task.id] = task
            return task

    @staticmethod
    def _set(future, value=None, error=None):
        if future.done():
            return
        try:
            if error is None:
                future.set_result(value)
            else:
                future.set_exception(error)
        except InvalidStateError:
            pass

    def _notify_loop(self):
        while True:
            item = self._notifications.get()
            if item is None:
                return
            future, value, error = item
            self._set(future, value=value, error=error)

    def _publish(self, future, value=None, error=None):
        self._notifications.put((future, value, error))

    def _done(self, task_id):
        with self._lock:
            task = self._tasks.pop(task_id, None)
        if task is None:
            return
        self._publish(task.prepared)
        self._publish(task.compute_done)
        for stage in task.stages:
            self._publish(stage)
        self._publish(task.done, task.output)

    def _stage(self, task_id, stage):
        with self._lock:
            task = self._tasks.get(task_id)
        if task is not None:
            self._publish(task.prepared)
            if 0 <= stage < len(task.stages):
                self._publish(task.stages[stage])

    def _prefetched(self, task_id):
        with self._lock:
            task = self._tasks.get(task_id)
        if task is not None:
            self._publish(task.prefetched)

    def _compute_done(self, task_id):
        with self._lock:
            task = self._tasks.get(task_id)
        if task is not None:
            self._publish(task.compute_done)

    def _failed(self, task_id, reason, _source):
        # The caller may not release task buffers until all rails drain.
        self._transport_error(f"task {task_id} failed: {reason}")

    def _transport_error(self, reason):
        with self._lock:
            if self._closed:
                return
            callback = self.on_failure
        if callback is not None:
            callback(reason)
        threading.Thread(target=self.close, args=(reason,), daemon=True,
                         name="remote-endpoint-close").start()

    def _submit(self, kind, frame, payload, *, planes=None, prefetch=False,
                output=False, stages=0):
        with self._submit_lock:
            return self._submit_locked(kind, frame, payload, planes=planes,
                                       prefetch=prefetch, output=output,
                                       stages=stages)

    def _submit_locked(self, kind, frame, payload, *, planes=None, prefetch=False,
                       output=False, stages=0):
        task = self._make_task(self._output() if output else None,
                               prefetched=prefetch, stages=stages)
        slot = None
        posted = False
        try:
            slot = self.guest.claim_slot(timeout=10)
            if slot is None:
                raise TimeoutError("no remote task slot")
            node = SimpleNamespace(id=task.id, type=kind, frame_idx=frame,
                                   payload=payload)
            kwargs = {}
            if task.output is not None:
                out = task.output
                kwargs = dict(dst_y_va=ctypes.addressof(out.y),
                              dst_u_va=ctypes.addressof(out.u),
                              dst_v_va=ctypes.addressof(out.v),
                              dst_y_stride=out.y_stride,
                              dst_uv_stride=out.uv_stride)
            self.guest.fill_request(slot, node, **kwargs)
            if planes is not None:
                if len(planes) != 3:
                    raise ValueError("source frame needs three packed planes")
                layout = {name: (offset, size) for name, offset, size, *_
                          in self.guest.src_layout}
                for name, data in zip(("ya", "ua", "va"), planes):
                    offset, size = layout[name]
                    if len(data) != size:
                        raise ValueError(f"invalid {name} source plane size")
                    ctypes.memmove(self.guest.get_send_buffer_addr(slot) + offset,
                                   data, size)
            posted = True
            self.guest.notify_submit(slot)
        except Exception as exc:
            if posted:
                # A partial post may still write: close QPs before failing.
                self.close(str(exc))
            else:
                if slot is not None:
                    self.guest.release_slot(slot)
                with self._lock:
                    self._tasks.pop(task.id, None)
                for future in (task.done, task.compute_done, task.prepared, task.prefetched, *task.stages):
                    self._publish(future, error=exc)
            raise
        return task

    def submit_frame(self, frame, planes, cache_slot, *, prefetch=False):
        return self._submit(TT_CCSR, frame,
                            {"dst_cc_slot": cache_slot,
                             "worker_keep_data": True,
                             "worker_prefetch": prefetch},
                            planes=planes, prefetch=prefetch, output=True)

    def activate(self, task):
        if not task.prefetched.done() or task.prefetched.exception() is not None:
            raise RuntimeError("prefetch has not completed")
        return self.control.submit("activate", "scheduler", task_id=task.id)

    def interpolate(self, frame, source_slot, neighbor_slot, output_slots):
        if (not 1 <= len(output_slots) <= 3
                or len(output_slots) != self.guest._interp_mult - 1):
            raise ValueError("interpolation output slots do not match multiplier")
        payload = {"worker_local_inputs": True, "worker_keep_data": True,
                   "worker_src_cc_slot_a": source_slot,
                   "worker_src_cc_slot_b": neighbor_slot,
                   "worker_dst_cc_slot": output_slots[0]}
        for i, slot in enumerate(output_slots[1:], start=2):
            payload[f"worker_dst_cc_slot_{i}"] = slot
        return self._submit(TT_INTERP, frame, payload, stages=len(output_slots))

    def super_resolve(self, frame, input_slot, phase=1):
        return self._submit(TT_SR_INTERP, frame,
                            {"worker_local_inputs": True, "output_phase": phase,
                             "worker_src_cc_slot_a": input_slot}, output=True)

    def interpolate_and_super_resolve(self, frame, source_slot, neighbor_slot,
                                      output_slot):
        return self._submit(TT_INTERP, frame,
                            {"worker_local_inputs": True, "worker_keep_data": True,
                             "worker_fused_sr": True,
                             "worker_src_cc_slot_a": source_slot,
                             "worker_src_cc_slot_b": neighbor_slot,
                             "worker_dst_cc_slot": output_slot}, output=True)

    def close(self, reason="worker endpoint closed"):
        with self._submit_lock, self._close_lock:
            with self._lock:
                if self._closed:
                    return
                self._closed = True
            self.guest.close()
            self.control.close()
            with self._lock:
                tasks, self._tasks = self._tasks, {}
            for task in tasks.values():
                error = RuntimeError(reason)
                for future in (task.done, task.compute_done, task.prepared, task.prefetched, *task.stages):
                    self._publish(future, error=error)
            self._notifications.put(None)
        if threading.current_thread() is not self._notifier:
            self._notifier.join(timeout=5)
