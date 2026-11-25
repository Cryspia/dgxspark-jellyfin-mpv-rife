"""Check warp equivalence and safe, repeatable vsrife cache patching."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("patch_vsrife_warp", ROOT / "scripts/patch_vsrife_warp.py")
patcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(patcher)


class WarpPatchTest(unittest.TestCase):
    def test_reference_warp_and_broadcast_match(self):
        torch.manual_seed(27)
        for dtype in (torch.float16, torch.float32):
            for height, width in ((16, 24), (64, 128)):
                image = torch.rand(1, 7, height, width, dtype=dtype).float()
                flow = (torch.randn(1, 2, height, width, dtype=dtype) * 35).float()
                flow[:, :, 0, 0] = 5000
                divisor = torch.tensor([(width - 1) / 2, (height - 1) / 2])
                x = torch.linspace(-1, 1, width).view(1, 1, 1, width).expand(1, 1, height, width)
                y = torch.linspace(-1, 1, height).view(1, 1, height, 1).expand(1, 1, height, width)
                base = torch.cat((x, y), 1)
                reference = torch.cat((flow[:, :1] / divisor[0], flow[:, 1:2] / divisor[1]), 1)
                candidate = flow / divisor.view(1, 2, 1, 1)
                self.assertTrue(torch.equal(reference, candidate))
                a = F.grid_sample(image, (base + reference).permute(0, 2, 3, 1),
                                  mode="bilinear", padding_mode="border", align_corners=True).to(dtype)
                b = F.grid_sample(image, (base + candidate).permute(0, 2, 3, 1),
                                  mode="bilinear", padding_mode="border", align_corners=True).to(dtype)
                self.assertTrue(torch.equal(a, b))

    def test_patch_is_repeatable_and_versions_engine_names(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            (package / "warplayer.py").write_text(patcher.OLD_WARP + "\n")
            (package / "__init__.py").write_text(patcher.MODEL_DIR + "\n" + patcher.OLD_NAME + "\n")
            patcher.patch(package)
            first = [(package / name).read_text() for name in ("warplayer.py", "__init__.py")]
            patcher.patch(package)
            self.assertEqual(first, [(package / name).read_text() for name in ("warplayer.py", "__init__.py")])
            self.assertIn(patcher.NEW_WARP, first[0])
            self.assertIn(patcher.CACHE_TAG, first[1])
            self.assertIn(patcher.NEW_NAME, first[1])

    def test_unexpected_upstream_layout_does_not_modify_files(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            before = {"warplayer.py": "unexpected\n", "__init__.py": patcher.MODEL_DIR + "\n" + patcher.OLD_NAME}
            for name, content in before.items():
                (package / name).write_text(content)
            with self.assertRaises(RuntimeError):
                patcher.patch(package)
            self.assertEqual(before, {name: (package / name).read_text() for name in before})


if __name__ == "__main__":
    unittest.main()
