import ctypes
import unittest

from dual_machine.cc_cache import CCCacheLayout, CCCacheShm
from dual_machine.peer_link import FrameCachePeer, PeerLink, RegisteredBuffer


class _Link:
    def __init__(self, capacity):
        self.max_bytes = capacity
        self.sent = []
        self.result = (0, 0, capacity)
        self.closed = False

    def register(self, addr, length):
        self.memory = (addr, length)
        return self

    def send_registered(self, generation, frame, buffer, length, *, offset):
        self.sent.append((generation, frame, length, offset))

    def receive_registered(self, buffer, *, offset):
        self.receive_offset = offset
        return self.result

    def close(self):
        self.closed = True


class FrameCachePeerTest(unittest.TestCase):
    def setUp(self):
        layout = CCCacheLayout(3, 16, 16, 32, 32, 32, 32, 4)
        self.cache = CCCacheShm(layout, name="unused")
        self.memory = ctypes.create_string_buffer(layout.total_size)
        self.cache.addr = ctypes.addressof(self.memory)
        self.link = _Link(layout.off_sr_yuv)
        self.peer = FrameCachePeer(self.link, self.cache)

    def test_uses_distinct_slot_offsets_and_rejects_invalid_slots(self):
        self.peer.send(2, 10, 2)
        self.assertEqual(self.link.sent, [(2, 10, self.link.max_bytes,
                                          2 * self.cache.layout.slot_stride)])
        self.link.result = (2, 11, self.link.max_bytes)
        self.peer.receive(2, 11, 1)
        self.assertEqual(self.link.receive_offset, self.cache.layout.slot_stride)
        for invalid in (-1, 3):
            with self.assertRaises(ValueError):
                self.peer.send(2, 10, invalid)

    def test_unexpected_frame_closes_link(self):
        self.link.result = (7, 5, self.link.max_bytes)
        with self.assertRaises(ValueError):
            self.peer.receive(7, 6, 0)
        self.assertTrue(self.link.closed)

    def test_registered_range_is_bounded(self):
        # Exercise the range check without constructing an RDMA device.
        link = object.__new__(PeerLink)
        link.max_bytes = 100
        buffer = object.__new__(RegisteredBuffer)
        buffer.owner = link
        buffer.mrs = [object()]
        buffer.length = 300
        link._check_registered(buffer, 100, 200)
        for offset, length in ((201, 100), (-1, 1), (0, -1), (0, 101)):
            with self.assertRaises(ValueError):
                link._check_registered(buffer, length, offset)


if __name__ == "__main__":
    unittest.main()
