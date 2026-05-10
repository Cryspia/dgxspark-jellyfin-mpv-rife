"""Helper for rife.vpy: apply FSRCNNX with F8 override + read F9 RIFE
toggle. Files written by scripts/sr_keys.lua are read here so the chain
reflects the current keybind state on every filter reload.

Side-channel files (under /tmp, kernel-cleaned across reboots):
  /tmp/rife_disabled         touch-file → skip rife_yuv
  /tmp/fsrcnnx_variant       override variant ("x2_8" / "x2_16" /
                              "x3_16" / "x4_16" / "off"); empty → auto
  /tmp/fsrcnnx_active_variant  written by us; lua reads to know the
                                cycle's current position

The fsrcnnx-cudnn package + weights are installed by install.sh from
the upstream release bundle to ~/.config/mpv/scripts/fsrcnnx-cudnn/
(matches the layout documented in the upstream README). We add that
directory to sys.path here so `from fsrcnnx_cudnn.vsfunc import …`
resolves regardless of how the package was installed.
"""

import os
import sys
from pathlib import Path

OVERRIDE_FILE      = Path("/tmp/fsrcnnx_variant")
ACTIVE_FILE        = Path("/tmp/fsrcnnx_active_variant")
RIFE_DISABLED_FILE = Path("/tmp/rife_disabled")

_MPV_HOME = Path(
    os.environ.get("MPV_HOME") or
    (os.environ.get("XDG_CONFIG_HOME") or
     os.path.expanduser("~/.config")) + "/mpv"
)
FSRCNNX_BUNDLE = _MPV_HOME / "scripts" / "fsrcnnx-cudnn"
if FSRCNNX_BUNDLE.exists() and str(FSRCNNX_BUNDLE) not in sys.path:
    sys.path.insert(0, str(FSRCNNX_BUNDLE))

from fsrcnnx_cudnn.vsfunc import (
    fsrcnnx_yuv, fsrcnnx_yuv_auto, select_variant,
)


def rife_disabled() -> bool:
    return RIFE_DISABLED_FILE.exists()


def _publish_active(name: str) -> None:
    try:
        ACTIVE_FILE.write_text(name)
    except OSError:
        pass


def apply_fsrcnnx(
    clip,
    *,
    family: str,
    weights_dir=None,
    target_w: int | None = None,
    target_h: int | None = None,
    chroma_kernel: str = "Bicubic",
):
    """Run FSRCNNX with F8 override support.

    `family` is the auto path's family ("8-layer" or "16-layer") —
    bands with smaller sources need 16-layer because select_variant
    refuses 8-layer at ratio ≥ 2.5.

    `weights_dir` defaults to the bundle's weights/ directory.

    `target_w` / `target_h` default to env vars FSRCNNX_TARGET_W /
    FSRCNNX_TARGET_H, then to 3840 / 2160 (4K display). Override the
    env vars in mpv.conf via wrapper / launcher when running on a
    non-4K screen.
    """
    if weights_dir is None:
        weights_dir = FSRCNNX_BUNDLE / "weights"
    if target_w is None:
        target_w = int(os.environ.get("FSRCNNX_TARGET_W", "3840"))
    if target_h is None:
        target_h = int(os.environ.get("FSRCNNX_TARGET_H", "2160"))

    override = ""
    if OVERRIDE_FILE.exists():
        try:
            override = OVERRIDE_FILE.read_text().strip().lower()
        except OSError:
            override = ""

    if override == "off":
        # F8 cycled to the explicit-off stop. Skip FSRCNNX entirely.
        _publish_active("off")
        return clip

    if override and override != "auto":
        weights_npz = Path(weights_dir) / f"FSRCNNX_{override}-0-4-1.npz"
        variant_full = f"FSRCNNX_{override}-0-4-1"
        clip = fsrcnnx_yuv(
            clip, weights_npz=str(weights_npz), variant=variant_full,
            target_width=target_w, target_height=target_h,
            chroma_kernel=chroma_kernel, min_ratio=0.0,
        )
        _publish_active(override)
        return clip

    picked = select_variant(
        target_w, target_h, clip.width, clip.height, family=family,
    )
    if picked is None:
        _publish_active("none")
        return fsrcnnx_yuv_auto(
            clip, weights_dir=str(weights_dir),
            target_width=target_w, target_height=target_h,
            family=family, chroma_kernel=chroma_kernel,
        )

    short = picked.replace("FSRCNNX_", "").replace("-0-4-1", "")
    _publish_active(short)
    return fsrcnnx_yuv_auto(
        clip, weights_dir=str(weights_dir),
        target_width=target_w, target_height=target_h,
        family=family, chroma_kernel=chroma_kernel,
    )
