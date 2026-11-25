import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def run_bench(tmp_path, output, mode="tictac"):
    if any(Path(p).exists() for p in ("/tmp/dual_machine_disabled",
                                      "/tmp/dual_machine_mult_override")):
        pytest.skip("playback has a global cluster override")
    clip = tmp_path / "clip.mkv"
    vpy = tmp_path / "rife.vpy"
    config = tmp_path / "cluster.conf"
    mpv = tmp_path / "mpv"
    clip.touch()
    vpy.touch()
    config.write_text("CLUSTER_WORKER_HOSTS=configured-worker\n")
    mpv.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$MOCK_BENCH_OUTPUT" >&2\n')
    mpv.chmod(0o755)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("CLUSTER_", "DUAL_"))}
    env.update(CLUSTER_CFG=str(config), CLUSTER_WORKER_HOSTS="explicit-worker",
               CLIP=str(clip), VPY=str(vpy), MPV=str(mpv), MODE=mode,
               MOCK_BENCH_OUTPUT=output, N="600")
    return subprocess.run([str(ROOT / "bench/cluster_fps.sh")], env=env,
                          text=True, capture_output=True)


def test_cluster_bench_accepts_active_tictac_output(tmp_path):
    result = run_bench(tmp_path, "dual-machine active\n[tictac] ready with 2 remote workers\n"
                       "[tictac fps] 100.0 fps (400 frames in 4.00s)")
    assert result.returncode == 0, result.stderr
    assert "100.0 fps" in result.stdout


def test_cluster_bench_rejects_silent_fallback(tmp_path):
    result = run_bench(tmp_path, "local fallback\n[tictac fps] 100.0 fps")
    assert result.returncode != 0
    assert "not active" in result.stderr
