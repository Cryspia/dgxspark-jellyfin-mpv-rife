import socket
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dual_machine"))
from dual_machine.peer_control import (PeerControlClient, WorkerPeerManager,
                                       _read_message, _send_message)


class PeerControlTest(unittest.TestCase):
    def test_completions_can_arrive_out_of_order(self):
        host, worker = socket.socketpair()
        client = PeerControlClient(host)
        try:
            first = client.submit("receive", "a", frame=1, generation=0, slot=2)
            second = client.submit("send", "b", frame=2, generation=0, slot=0)
            requests = [_read_message(worker), _read_message(worker)]
            _send_message(worker, {"id": requests[1]["id"], "ok": True})
            _send_message(worker, {"id": requests[0]["id"], "ok": False,
                                   "error": "transfer failed"})
            self.assertIsNone(second.result(2))
            with self.assertRaisesRegex(RuntimeError, "transfer failed"):
                first.result(2)
        finally:
            worker.close()
            client.close()

    def test_disconnect_fails_pending_commands(self):
        host, worker = socket.socketpair()
        client = PeerControlClient(host)
        pending = client.submit("receive", "a")
        _read_message(worker)
        worker.close()
        with self.assertRaises(EOFError):
            pending.result(2)
        client.close()

    def test_oversized_message_is_rejected(self):
        host, worker = socket.socketpair()
        client = PeerControlClient(host)
        try:
            result = client.submit("connect", "x", host="x" * 5000)
            with self.assertRaises(ValueError):
                result.result(2)
        finally:
            worker.close()
            client.close()

    def test_result_release_rejects_stale_slot_or_task(self):
        from mp_pipeline import ST_PULL_DONE, ST_SEND_PENDING
        meta = SimpleNamespace(state=ST_SEND_PENDING, task_id=42)
        worker = SimpleNamespace(local_cache=object(), layout=SimpleNamespace(n_slots=1),
                                 shm=SimpleNamespace(slot_state=lambda slot: meta))
        manager = WorkerPeerManager(worker, [])
        with self.assertRaisesRegex(ValueError, "stale"):
            manager.execute(dict(op="release_result", link="scheduler", slot=0, task_id=41))
        self.assertEqual(meta.state, ST_SEND_PENDING)
        manager.execute(dict(op="release_result", link="scheduler", slot=0, task_id=42))
        self.assertEqual(meta.state, ST_PULL_DONE)
        with self.assertRaisesRegex(ValueError, "stale"):
            manager.execute(dict(op="release_result", link="scheduler", slot=0, task_id=42))
        manager.close()

    def test_different_peer_links_can_transfer_concurrently(self):
        manager = WorkerPeerManager(SimpleNamespace(local_cache=object()), [])
        barrier = threading.Barrier(2)
        failures = []

        class Link:
            def close(self):
                pass

        class Peer(Link):
            def send(self, generation, frame, slot):
                barrier.wait(2)

        manager.links = {name: (Link(), Peer()) for name in ("a", "b")}

        def send(name):
            try:
                manager.execute(dict(op="send", link=name, generation=0,
                                     frame=1, slot=0))
            except Exception as exc:
                failures.append(exc)

        threads = [threading.Thread(target=send, args=(name,)) for name in ("a", "b")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(3)
        self.assertFalse(failures)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        manager.close()

    def test_concurrent_closers_wait_for_active_transfer(self):
        manager = WorkerPeerManager(SimpleNamespace(local_cache=object()), [])
        entered, finish = threading.Event(), threading.Event()
        closed = []

        class Link:
            def close(self):
                closed.append(True)

        class Peer(Link):
            def send(self, generation, frame, slot):
                entered.set()
                finish.wait(2)

        manager.links["a"] = (Link(), Peer())
        task = threading.Thread(target=manager.execute, args=(dict(
            op="send", link="a", generation=0, frame=1, slot=0),))
        task.start()
        self.assertTrue(entered.wait(2))
        closers = [threading.Thread(target=manager.close) for _ in range(2)]
        for thread in closers:
            thread.start()
        self.assertTrue(all(thread.is_alive() for thread in closers))
        self.assertFalse(closed)
        finish.set()
        task.join(2)
        for thread in closers:
            thread.join(2)
        self.assertFalse(any(thread.is_alive() for thread in closers))
        self.assertEqual(len(closed), 2)


if __name__ == "__main__":
    unittest.main()
