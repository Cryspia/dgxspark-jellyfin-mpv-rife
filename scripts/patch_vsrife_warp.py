#!/usr/bin/env python3
"""Apply the equivalent two-channel warp normalization and version its cache."""

import sys
from pathlib import Path


OLD_WARP = '    tenFlow = torch.cat([tenFlow[:, 0:1] / tenFlow_div[0], tenFlow[:, 1:2] / tenFlow_div[1]], 1)'
NEW_WARP = '    tenFlow = tenFlow / tenFlow_div.view(1, 2, 1, 1)'
MODEL_DIR = 'model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")'
CACHE_TAG = '_WARP_CACHE_TAG = "_warpvec1"'
OLD_NAME = 'f"{model_name}"'
NEW_NAME = 'f"{model_name}{_WARP_CACHE_TAG}"'


def update_once(text: str, before: str, after: str, label: str) -> str:
    if text.count(after) == 1 and text.count(before) == int(before in after):
        return text
    if text.count(before) != 1 or after in text:
        raise RuntimeError(f"Unexpected vsrife {label} layout")
    return text.replace(before, after, 1)


def patch(package: Path) -> None:
    warp_path = package / "warplayer.py"
    init_path = package / "__init__.py"
    warp = update_once(warp_path.read_text(encoding="utf-8"), OLD_WARP, NEW_WARP, "warp")
    init = init_path.read_text(encoding="utf-8")
    init = update_once(init, MODEL_DIR, MODEL_DIR + "\n" + CACHE_TAG, "cache tag")
    init = update_once(init, OLD_NAME, NEW_NAME, "engine name")
    # Check both files before modifying either of them.
    warp_path.write_text(warp, encoding="utf-8")
    init_path.write_text(init, encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: patch_vsrife_warp.py <vsrife-package-directory>")
    patch(Path(sys.argv[1]))
