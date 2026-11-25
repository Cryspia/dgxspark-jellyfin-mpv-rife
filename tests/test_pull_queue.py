import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dual_machine'))
from pull_queue import PullQueue


def test_ready_frames_are_prioritized_without_waiting_for_gaps():
    q = PullQueue()
    q.put(10, 1, 'a', 0, 0, 1)
    q.put(9, 1, 'b', 0, 0, 1)
    q.put(9, 0, 'c', 0, 0, 1)
    first = q.get(0)
    assert (first.frame, first.phase) == (9, 0)
    assert q.get(0) is None
    q.complete(first)
    second = q.get(0)
    assert (second.frame, second.phase) == (9, 1)
    q.complete(second)
    third = q.get(0)
    assert (third.frame, third.phase) == (10, 1)
    q.complete(third)


def test_disconnect_discards_only_queued_requests_and_keeps_active_owned():
    q = PullQueue()
    q.put(0, 0, 'a', 0, 0, 1)
    active = q.get(0)
    q.put(2, 0, 'a', 1, 1, 2)
    q.put(1, 0, 'b', 0, 0, 1)
    assert len(q.discard_worker('a')) == 1
    with pytest.raises(ValueError):
        q.put(3, 0, 'a', 0, 0, 1)
    q.complete(active)
    assert q.get(0).worker == 'b'


def test_close_wakes_waiters_and_keeps_active_completable():
    q = PullQueue()
    q.put(0, 0, 'a', 0, 0, 1)
    active = q.get(0)
    q.put(1, 0, 'b', 0, 0, 1)
    assert len(q.close()) == 1
    assert q.get(0) is None
    q.complete(active)
    with pytest.raises(RuntimeError):
        q.put(2, 0, 'a', 0, 0, 2)


def test_coordinator_detach_waits_for_inflight_read():
    import threading
    from types import SimpleNamespace
    from pull_queue import PullCoordinator

    entered, finish = threading.Event(), threading.Event()
    class Guest:
        _pull_detached = False
        transport_error_fn = None
        _task = [SimpleNamespace(frame=3, phase=1, task_id=8)]
        def pull_result(self, host_slot, worker_slot, task_id):
            entered.set()
            assert finish.wait(2)
        def finish_pull(self, host_slot, task_id):
            pass

    guest = Guest()
    coordinator = PullCoordinator()
    try:
        coordinator.enqueue(guest, 0, 1)
        assert entered.wait(2)
        detached = threading.Event()
        thread = threading.Thread(target=lambda: (coordinator.detach(guest), detached.set()))
        thread.start()
        assert not detached.wait(.05)
        finish.set()
        assert detached.wait(2)
        thread.join(2)
        assert guest._pull_detached
    finally:
        finish.set()
        coordinator.close()


def test_next_read_runs_while_previous_writeback_is_pending():
    import threading
    from types import SimpleNamespace
    from pull_queue import PullCoordinator

    copying, finish_copy, second_read = threading.Event(), threading.Event(), threading.Event()
    class Guest:
        _pull_detached = False
        transport_error_fn = None
        _task = [SimpleNamespace(frame=0, phase=0, task_id=1),
                 SimpleNamespace(frame=1, phase=0, task_id=2)]
        def pull_result(self, slot, worker_slot, task_id):
            if slot == 1:
                second_read.set()
        def finish_pull(self, slot, task_id):
            if slot == 0:
                copying.set()
                assert finish_copy.wait(2)

    guest = Guest()
    coordinator = PullCoordinator()
    try:
        coordinator.enqueue(guest, 0, 0)
        assert copying.wait(2)
        coordinator.enqueue(guest, 1, 1)
        assert second_read.wait(2)
        detached = threading.Event()
        thread = threading.Thread(target=lambda: (coordinator.detach(guest), detached.set()))
        thread.start()
        assert not detached.wait(.05)
        finish_copy.set()
        assert detached.wait(2)
        thread.join(2)
    finally:
        finish_copy.set()
        coordinator.close()
