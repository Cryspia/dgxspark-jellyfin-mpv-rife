import ctypes
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dual_machine"))
from remote_endpoint import RemoteEndpoint


class _Guest:
    def __init__(self):
        self.oH, self.oW, self.ocH, self.ocW = 2, 2, 1, 1
        self._interp_mult = 2
        self.src_layout = [("ya", 128, 4), ("ua", 132, 2), ("va", 134, 2)]
        self.memory = ctypes.create_string_buffer(136)
        self.closed = False
        self.node = None

    def claim_slot(self, timeout):
        return 0

    def fill_request(self, slot, node, **kwargs):
        self.node = node

    def get_send_buffer_addr(self, slot):
        return ctypes.addressof(self.memory)

    def notify_submit(self, slot):
        pass

    def release_slot(self, slot):
        pass

    def close(self):
        self.closed = True


class _Control:
    def close(self):
        pass


class RemoteEndpointTest(unittest.TestCase):
    def test_stages_and_final_output_are_delivered_asynchronously(self):
        guest = _Guest()
        endpoint = RemoteEndpoint(guest, _Control())
        try:
            task = endpoint.submit_frame(4, (b"abcd", b"ef", b"gh"), 2,
                                         prefetch=True)
            self.assertEqual(guest.memory.raw[128:136], b"abcdefgh")
            guest.task_prefetch_ready_fn(task.id)
            task.prefetched.result(2)
            self.assertFalse(task.done.done())
            guest.task_stage_done_fn(task.id, 0)
            task.prepared.result(2)
            callback_thread = []
            called = threading.Event()
            def on_done(_):
                callback_thread.append(threading.current_thread().name)
                called.set()
            task.done.add_done_callback(on_done)
            guest.task_done_fn(task.id)
            self.assertIs(task.done.result(2), task.output)
            self.assertTrue(called.wait(2))
            self.assertEqual(callback_thread, ["remote-endpoint-notifier"])
        finally:
            endpoint.close()

    def test_close_fences_transport_before_failing_task(self):
        guest = _Guest()
        endpoint = RemoteEndpoint(guest, _Control())
        task = endpoint.submit_frame(1, (b"abcd", b"ef", b"gh"), 0)
        self.assertFalse(task.done.cancel())
        observed = []
        task.done.add_done_callback(lambda _: observed.append(guest.closed))
        endpoint.close("lost")
        with self.assertRaisesRegex(RuntimeError, "lost"):
            task.done.result(2)
        self.assertEqual(observed, [True])

    def test_rejects_wrong_output_phase_count(self):
        endpoint = RemoteEndpoint(_Guest(), _Control())
        try:
            with self.assertRaises(ValueError):
                endpoint.interpolate(1, 0, 1, (2, 3))
        finally:
            endpoint.close()


if __name__ == "__main__":
    unittest.main()
