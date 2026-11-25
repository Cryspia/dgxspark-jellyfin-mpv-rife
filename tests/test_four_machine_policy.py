import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dual_machine"))

import pytest
from common import default_interpolation_multiplier, four_machine_policy, use_tictac


@pytest.mark.parametrize('fps,height,width,expected', [
    (0,2160,3840,2), (23.976,2160,3840,2), (24,2160,3840,2),
    (24.001,2160,3840,3), (25,2160,3840,3), (30,2160,3840,3),
    (30.001,2160,3840,2), (60,2160,3840,2), (60.001,2160,3840,1),
    (22,1080,1920,4), (24,1080,1920,4), (24.001,1080,1920,3),
    (30,1080,1920,3), (30.001,1080,1920,2), (60,1080,1920,2),
    (30,720,1280,4), (30,720,1920,3), (30.001,720,1280,2), (24,1080,2560,2),
])
def test_four_machine_thresholds(fps, height, width, expected):
    assert four_machine_policy()
    assert default_interpolation_multiplier(fps, height, width, 3) == expected


@pytest.mark.parametrize('workers,setting', [(1, None), (2, None), (3, '0')])
def test_other_modes_keep_existing_policy(workers, setting):
    choose = lambda f,h: default_interpolation_multiplier(f, h, 1920, workers, setting)
    assert choose(24,1080) == 3
    assert choose(30,1080) == 2
    assert choose(31,1080) == 1
    assert choose(24,720) == 4
    assert choose(30,720) == 3
    assert choose(24,2160) == 3


def test_invalid_policy_falls_back(monkeypatch):
    monkeypatch.setattr('common.four_machine_policy', lambda: {})
    assert default_interpolation_multiplier(24,2160,3840,3) == 3
    assert default_interpolation_multiplier(60,1080,1920,3) == 1


def test_native_4k_opt_out_uses_downsampled_policy():
    assert default_interpolation_multiplier(24,2160,3840,3,native_4k="0") == 3
    assert default_interpolation_multiplier(24,2160,3840,3,native_4k="1") == 2


def test_source_fps_ignores_stale_or_invalid_sidecar(tmp_path, monkeypatch):
    import os
    from pathlib import Path
    import common
    process = tmp_path / 'process'
    process.mkdir()
    rate_file = tmp_path / 'rate'
    real_path = Path
    def select_path(value):
        if value == '/proc/self':
            return process
        if value == f'/tmp/dual_machine_source_fps_{os.getpid()}':
            return rate_file
        return real_path(value)
    monkeypatch.setattr(common, 'Path', select_path)
    assert common.source_fps_from_sidecar() == 0
    rate_file.write_text('23.976023976')
    assert common.source_fps_from_sidecar() == pytest.approx(23.976023976)
    rate_file.write_text('nan')
    assert common.source_fps_from_sidecar() == 0
    rate_file.write_text('24')
    os.utime(rate_file, (1, 1))
    assert common.source_fps_from_sidecar() == 0
