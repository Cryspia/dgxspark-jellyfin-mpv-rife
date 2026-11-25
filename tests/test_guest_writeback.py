import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dual_machine"))
from guest_mp import GuestMP, TT_SR_INTERP


class GuestWritebackTest(unittest.TestCase):
    def test_copy_error_fails_task_instead_of_publishing_success(self):
        failures, completed = [], []
        task = SimpleNamespace(task_type=TT_SR_INTERP, task_id=8,
                               dst_y_va=1, dst_u_va=1, dst_v_va=1,
                               dst_y_stride=2, dst_uv_stride=2)
        def fail_copy(*_args):
            raise RuntimeError("invalid destination")
        guest = SimpleNamespace(
            _task=[task], recv_bufs=[SimpleNamespace(addr=1)],
            _sr_dst_offs={"yao": (0, None, None, None),
                          "uao": (1, None, None, None),
                          "vao": (2, None, None, None)},
            oW=2, oH=2, ocW=1, ocH=1, _copy_plane=fail_copy,
            log=lambda _message: None,
            _fail_slot=lambda slot, reason: failures.append((slot, reason)),
            task_done_fn=lambda task_id: completed.append(task_id))
        GuestMP._handle_recv(guest, 0)
        self.assertEqual(len(failures), 1)
        self.assertIn("invalid destination", failures[0][1])
        self.assertFalse(completed)


if __name__ == "__main__":
    unittest.main()


def test_normal_pull_waits_for_all_early_stages():
    import threading
    from types import SimpleNamespace
    from guest_mp import GuestMP, TT_INTERP
    guest = object.__new__(GuestMP)
    guest._task = [SimpleNamespace(task_id=7, task_type=TT_INTERP)]
    guest._pull_normal = True
    guest._mid_delivery_enabled = True
    guest._interp_mult = 4
    guest._pull_mid_cv = threading.Condition()
    guest._pull_mid_seen = [set()]
    guest._pull_abort = threading.Event()
    published = threading.Event()
    guest._handle_recv = lambda slot: published.set()
    thread = threading.Thread(target=guest.finish_pull, args=(0, 7))
    thread.start()
    try:
        with guest._pull_mid_cv:
            guest._pull_mid_seen[0].add(1)
            guest._pull_mid_cv.notify_all()
        assert not published.wait(.05)
        with guest._pull_mid_cv:
            guest._pull_mid_seen[0].add(0)
            guest._pull_mid_cv.notify_all()
        assert published.wait(2)
    finally:
        guest._pull_abort.set()
        with guest._pull_mid_cv:
            guest._pull_mid_cv.notify_all()
        thread.join(2)
    assert not thread.is_alive()


def test_pull_priority_uses_queue_node_output_phase():
    from types import SimpleNamespace
    from guest_mp import TT_CCSR, TT_SR_INTERP, _output_phase
    assert _output_phase(TT_SR_INTERP, SimpleNamespace(output_phase=3, payload={})) == 3
    assert _output_phase(TT_SR_INTERP, SimpleNamespace(payload={"output_phase": 2})) == 2
    assert _output_phase(TT_CCSR, SimpleNamespace(output_phase=3, payload={})) == 0
