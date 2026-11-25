import socket
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dual_machine"))
from tictac_dispatcher import _ready_workers


class TicTacStartupTest(unittest.TestCase):
    def test_drops_worker_that_disconnects_during_handshake(self):
        pairs = []
        peers = []
        threads = []
        for index in range(3):
            client, peer = socket.socketpair()
            pairs.append((f"worker{index}", client))
            peers.append(peer)

            def reply(connection=peer, good=index != 1):
                try:
                    self.assertEqual(connection.recv(4), b"init")
                    if good:
                        connection.sendall(b"R")
                finally:
                    connection.close()

            thread = threading.Thread(target=reply)
            thread.start()
            threads.append(thread)
        try:
            ready = _ready_workers(pairs, b"init", 2)
            self.assertEqual([host for host, _ in ready], ["worker0", "worker2"])
            self.assertEqual(pairs[1][1].fileno(), -1)
            self.assertTrue(all(sock.gettimeout() is None for _, sock in ready))
        finally:
            for _, client in pairs:
                client.close()
            for thread in threads:
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
