import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dual_machine'))
from rdma_transport import src_partition, zc_src_bundle_layout
from guest_mp import GuestMP
from worker import _parse_handshake


class SourcePartitionTest(unittest.TestCase):
    def test_ccsr_is_split_and_slot_covered(self):
        for height, width in ((1080, 1920), (2160, 3840)):
            for sub in (0, 1):
                total, layout = zc_src_bundle_layout(height, width, height >> sub,
                                                     width >> sub, height, width, 4)
                fields = {name: (offset, size) for name, offset, size, *_ in layout}
                prefix = sum(fields['va'])
                parts = src_partition(total, 2, balanced_prefix=prefix)
                self.assertEqual(parts[0][0], 0)
                self.assertEqual(parts[1][0], parts[0][1])
                self.assertEqual(sum(length for _, length in parts), total)
                guest = object.__new__(GuestMP)
                guest.src_parts, guest.n_rails = parts, 2
                split = guest._split_wire([(1000, prefix, 7)], {7: [11, 12]})
                lengths = [sum(s[1] for s in rail) for rail in split]
                self.assertEqual(sum(lengths), prefix)
                self.assertLess(abs(lengths[0] - lengths[1]), 8192)
                self.assertEqual(split[0][0][2], 11)
                self.assertEqual(split[1][0][2], 12)
                self.assertEqual(guest._split_wire([(1000, 128, 7)], {})[1], [])
                self.assertEqual(src_partition(total, 1, balanced_prefix=prefix), [(0, total)])
                step = max(4096, (total // 2) // 4096 * 4096)
                self.assertEqual(src_partition(total, 2), [(0, step), (step, total-step)])

    def test_layout_handshake_preserves_early_delivery(self):
        values = [0] * 20
        for mode in (5, 6, 7, 8, 9):
            values[0], values[19] = mode, 2
            result = _parse_handshake(struct.pack('<20q', *values))
            self.assertTrue(result['balanced_src'])
            self.assertEqual(result['mid_delivery'], 1)
        values[19] = 1
        self.assertFalse(_parse_handshake(struct.pack('<20q', *values))['balanced_src'])
        values[0], values[19] = 4, 2
        self.assertIsNone(_parse_handshake(struct.pack('<20q', *values)))
        for mode in (5, 6, 7, 8, 9):
            for value in (5, 6):
                values[0], values[19] = mode, value
                result = _parse_handshake(struct.pack('<20q', *values))
                self.assertTrue(result['pull_results'])
                self.assertEqual(result['balanced_src'], value == 6)
                self.assertEqual(result['mid_delivery'], 1)
        values[0], values[19] = 4, 6
        self.assertIsNone(_parse_handshake(struct.pack('<20q', *values)))
        for mode, early_values in ((10, (0, 1)), (11, (1,))):
            for early in early_values:
                values[0], values[19] = mode, early
                result = _parse_handshake(struct.pack('<20q', *values))
                self.assertTrue(result['pull_results'])
                self.assertTrue(result['normal_pull'])
                self.assertEqual(result['mid_delivery'], early)
        values[0], values[19] = 11, 0
        self.assertIsNone(_parse_handshake(struct.pack('<20q', *values)))
        for mode in (2, 3, 4):
            values[0], values[19] = mode, 1
            self.assertIsNone(_parse_handshake(struct.pack('<20q', *values)))
        values[0], values[19] = 10, 5
        self.assertIsNone(_parse_handshake(struct.pack('<20q', *values)))


if __name__ == '__main__':
    unittest.main()
