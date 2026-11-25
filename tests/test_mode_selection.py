import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dual_machine"))

from common import use_tictac


def test_mode_selection_by_worker_count_and_override():
    for setting in (None, ""):
        assert not use_tictac(1, 2, setting)
        assert use_tictac(2, 2, setting)
        assert use_tictac(3, 4, setting)
    assert not use_tictac(3, 2, "0")
    assert use_tictac(2, 2, "1")
    assert not use_tictac(1, 2, "1")
    assert not use_tictac(3, 1, "1")
    assert not use_tictac(3, 2, "false")
