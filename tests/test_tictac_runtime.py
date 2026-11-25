import sys
import time
import random
import threading
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dual_machine"))
from tictac_runtime import FrameSet, TicTacRuntime
from frame_scheduler import FrameScheduler


def ready(value=None):
    future = Future()
    future.set_result(value)
    return future


class _Control:
    def submit(self, *_args, **_kwargs):
        return ready()


class _Endpoint:
    def __init__(self, name):
        self.name = name
        self.control = _Control()

    def submit_frame(self, frame, planes, slot, *, prefetch=False):
        task = SimpleNamespace(prefetched=ready(), prepared=Future(), done=Future(),
                               frame=frame)
        if not prefetch:
            self.activate(task)
        return task

    def activate(self, task):
        task.prepared.set_result(None)
        task.done.set_result((self.name, task.frame, 0))
        return ready()

    def interpolate(self, frame, source_slot, neighbor_slot, output_slots):
        return SimpleNamespace(done=ready())

    def super_resolve(self, frame, input_slot, phase=1):
        return SimpleNamespace(done=ready((self.name, frame, input_slot)))

    def close(self):
        pass


class TicTacRuntimeTest(unittest.TestCase):
    def test_runtime_submits_source_without_activation_command(self):
        class NoActivationControl(_Control):
            def submit(self, operation, *args, **kwargs):
                if operation == 'activate':
                    raise AssertionError('runtime sent an activation command')
                return super().submit(operation, *args, **kwargs)

        class DirectEndpoint(_Endpoint):
            def __init__(self, name):
                super().__init__(name)
                self.control = NoActivationControl()
                self.direct_sources = 0

            def submit_frame(self, frame, planes, slot, *, prefetch=False):
                if prefetch:
                    raise AssertionError('runtime submitted a held prefetch')
                self.direct_sources += 1
                return super().submit_frame(frame, planes, slot, prefetch=prefetch)

        endpoints = {name: DirectEndpoint(name) for name in ('w1', 'w2')}
        runtime = TicTacRuntime(endpoints, {('w1', 'w2'): 'a', ('w2', 'w1'): 'b'})
        try:
            planes = (b'a', b'b', b'c')
            for frame in range(12):
                runtime.submit_interval(frame, planes, planes)
            for frame in range(12):
                for phase in (0, 1):
                    runtime.wait_output(frame, phase, timeout=5)
            self.assertGreater(sum(e.direct_sources for e in endpoints.values()), 0)
        finally:
            runtime.close()

    def test_ahead_source_can_prepare_before_predecessor_notice(self):
        scheduler = FrameScheduler(['w1', 'w2'], prefetch=True, fused_ccsr=True)
        scheduler.submit_interval(0)
        scheduler.submit_interval(1)
        current = scheduler.claim('w1')
        scheduler.claim('w2')
        ahead = scheduler.claim_ahead('w1')
        runtime = object.__new__(TicTacRuntime)
        runtime._cv = threading.Condition(threading.RLock())
        runtime._closed = False
        runtime.scheduler = scheduler
        runtime._running = {('w1', 'compute'): current, ('w1', 'ahead'): ahead}
        runtime._ahead_callbacks = {}
        runtime._results = {}
        runtime._profile = False
        runtime.fused_interp_sr = True
        runtime._prepared(ahead, ready())
        self.assertTrue(scheduler._frames[2].prepared)
        runtime._compute_finished(('w1', 'ahead'), ahead, ready())
        runtime._finish(('w1', 'ahead'), ahead, ready('source'))
        self.assertEqual(runtime._results, {})
        runtime._compute_finished(('w1', 'compute'), current, ready())
        self.assertEqual(runtime._results[(2, 0)], 'source')
        self.assertFalse(runtime._running)

    def test_ahead_callbacks_wait_for_predecessor_notification(self):
        scheduler = FrameScheduler(['w1', 'w2'], prefetch=True, fused_ccsr=True)
        scheduler.submit_interval(0)
        scheduler.submit_interval(1)
        first = scheduler.claim('w1')
        prefetch = scheduler.claim_prefetch('w1')
        neighbor = scheduler.claim('w2')
        scheduler.complete(prefetch)
        scheduler.report_prepared(first)
        scheduler.report_prepared(neighbor)
        scheduler.complete(scheduler.claim_peer_transfer('w2'))
        scheduler.report_compute_done(first)
        current = scheduler.claim('w1')
        ahead = scheduler.claim_ahead('w1')

        runtime = object.__new__(TicTacRuntime)
        runtime._cv = threading.Condition(threading.RLock())
        runtime._closed = False
        runtime.scheduler = scheduler
        runtime._running = {('w1', 'compute'): current, ('w1', 'ahead'): ahead}
        runtime._ahead_callbacks = {}
        runtime._results = {}
        runtime._profile = False
        runtime.fused_interp_sr = True
        runtime._compute_finished(('w1', 'ahead'), ahead, ready())
        runtime._finish(('w1', 'ahead'), ahead, ready('result'))
        self.assertEqual(runtime._results, {})
        runtime._compute_finished(('w1', 'compute'), current, ready())
        self.assertEqual(runtime._results[(0, 1)], 'result')
        self.assertFalse(runtime._running)

    def test_lazy_input_reuses_the_shared_neighbor(self):
        endpoints = {name: _Endpoint(name) for name in ("w0", "w1")}
        runtime = TicTacRuntime(endpoints, {("w0", "w1"): "a",
                                            ("w1", "w0"): "b"})
        calls = []

        def source(index):
            def read():
                calls.append(index)
                return (bytes([index]), b"u", b"v")
            return read

        try:
            runtime.submit_interval_lazy(0, source(0), source(1))
            runtime.submit_interval_lazy(1, source(1), source(2))
            self.assertEqual(calls, [0, 1, 2])
            for frame in (0, 1):
                for phase in (0, 1):
                    self.assertEqual(runtime.wait_output(frame, phase, timeout=5)[1], frame)
        finally:
            runtime.close()

    def test_frame_set_compresses_out_of_order_insertions(self):
        frames = FrameSet()
        for frame in (0, 2, 4, 1, 3, 10, 9, 8):
            frames.add(frame)
        self.assertEqual(frames._starts, [0, 8])
        self.assertEqual(frames._ends, [5, 11])
        self.assertTrue(all(frame in frames for frame in (0, 4, 8, 10)))
        self.assertFalse(any(frame in frames for frame in (-1, 5, 7, 11)))

    def test_multiple_workers_drain_all_phases(self):
        for worker_count in (2, 3, 4):
            for mult in (2, 3, 4):
                names = [f"w{i}" for i in range(worker_count)]
                endpoints = {name: _Endpoint(name) for name in names}
                links = {(a, b): f"{a}-{b}" for a in names for b in names if a != b}
                runtime = TicTacRuntime(endpoints, links, interp_mult=mult)
                try:
                    planes = (b"a", b"b", b"c")
                    for frame in range(12):
                        runtime.submit_interval(frame, planes, planes)
                    for frame in range(12):
                        for phase in range(mult):
                            result = runtime.wait_output(frame, phase, timeout=5)
                            self.assertEqual(result[1], frame)
                finally:
                    runtime.close()

    def test_consumed_frames_release_host_and_scheduler_storage(self):
        endpoints = {name: _Endpoint(name) for name in ("w0", "w1")}
        runtime = TicTacRuntime(endpoints, {("w0", "w1"): "a",
                                            ("w1", "w0"): "b"})
        try:
            planes = (b"a", b"b", b"c")
            for frame in range(20):
                runtime.submit_interval(frame, planes, planes)
                for phase in (0, 1):
                    runtime.wait_output(frame, phase, timeout=5)
                    runtime.consume_output(frame, phase)
            deadline = time.monotonic() + 5
            while len(runtime._sources) > 1 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertLessEqual(len(runtime._sources), 1)
            self.assertLessEqual(len(runtime._results), 1)
            self.assertLessEqual(len(runtime.scheduler._frames), 1)
        finally:
            runtime.close()

    def test_out_of_order_interval_submissions_drain(self):
        names = ["w0", "w1"]
        endpoints = {name: _Endpoint(name) for name in names}
        runtime = TicTacRuntime(endpoints, {("w0", "w1"): "a",
                                            ("w1", "w0"): "b"})
        try:
            frames = list(range(20))
            random.Random(7).shuffle(frames)
            for frame in frames:
                runtime.submit_interval(frame, (b"a", b"b", b"c"),
                                         (b"d", b"e", b"f"))
            for frame in frames:
                for phase in (0, 1):
                    runtime.wait_output(frame, phase, timeout=2)
        finally:
            runtime.close()


if __name__ == "__main__":
    unittest.main()
