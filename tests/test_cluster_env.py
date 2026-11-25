import os
import subprocess
from pathlib import Path

from dual_machine.cluster_env import normalize_cluster_environment


def test_legacy_environment_is_accepted_and_new_names_win():
    env = {"DUAL_WORKER_HOSTS": "old", "DUAL_TICTAC": "1",
           "CLUSTER_WORKER_HOSTS": "new", "UNRELATED": "value"}
    normalize_cluster_environment(env)
    assert env["CLUSTER_WORKER_HOSTS"] == "new"
    assert env["CLUSTER_TICTAC"] == "1"
    assert env["UNRELATED"] == "value"


def test_shell_environment_accepts_legacy_and_preserves_empty_new_value():
    helper = Path(__file__).resolve().parents[1] / "scripts/cluster_env.sh"
    env = dict(os.environ, DUAL_TEST="old", CLUSTER_TEST="", DUAL_OTHER="legacy")
    result = subprocess.run(
        ["bash", "-uc", 'source "$1"; cluster_normalize_env; printf "%s|%s" "$CLUSTER_TEST" "$CLUSTER_OTHER"',
         "bash", str(helper)], env=env, text=True, capture_output=True, check=True)
    assert result.stdout == "|legacy"
