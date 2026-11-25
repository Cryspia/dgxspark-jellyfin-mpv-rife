import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_configure(tmp_path, role="host", **settings):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("CLUSTER_", "DUAL_"))}
    env.update(CLUSTER_SELF_IP="198.51.100.10", CLUSTER_IFACE="test0",
               CLUSTER_RDMA_DEV="rdma0", CLUSTER_RDMA_DEVS="rdma0,rdma1",
               CLUSTER_RDMA_GIDS="3,3", CLUSTER_WORKER_USER="worker")
    env.update(settings)
    script = r'''
source "$1/install.sh" --help >/dev/null
CLUSTER_CFG_DIR="$2"
CLUSTER_CFG_FILE="$2/cluster.conf"
dual_detect_iface_default() { printf test0; }
dual_detect_rdma_dev() { printf rdma0; }
dual_detect_local_ip() { printf 198.51.100.10; }
dual_detect_gid_index() { printf 3; }
dual_detect_rails() { printf 'test0 rdma0 3 198.51.100.10\ntest1 rdma1 3 203.0.113.10\n'; }
dual_configure "$3"
'''
    return subprocess.run(["bash", "-c", script, "bash", str(ROOT), str(tmp_path), role],
                          env=env, text=True, capture_output=True)


def test_configure_accepts_multiple_workers_without_separate_peer(tmp_path):
    result = run_configure(tmp_path, CLUSTER_WORKER_HOSTS="worker-a,worker-b",
                           CLUSTER_WORKER_SSH_HOSTS="ssh-a,ssh-b")
    assert result.returncode == 0, result.stderr
    config = (tmp_path / "cluster.conf").read_text()
    assert "CLUSTER_WORKER_HOST=worker-a\n" in config
    assert "CLUSTER_WORKER_HOSTS=worker-a,worker-b\n" in config
    assert "CLUSTER_WORKER_SSH_HOSTS=ssh-a,ssh-b\n" in config
    assert (tmp_path / "cluster.conf").stat().st_mode & 0o777 == 0o600


def test_configure_rejects_duplicate_workers_and_mismatched_ssh_list(tmp_path):
    duplicate = run_configure(tmp_path, CLUSTER_WORKER_HOSTS="worker-a,worker-a")
    assert duplicate.returncode != 0
    mismatch = run_configure(tmp_path, CLUSTER_WORKER_HOSTS="worker-a,worker-b",
                             CLUSTER_WORKER_SSH_HOSTS="ssh-a")
    assert mismatch.returncode != 0


def test_worker_configuration_uses_leader_and_local_addresses(tmp_path):
    result = run_configure(tmp_path, role="secondary", CLUSTER_PEER_IP="leader-node")
    assert result.returncode == 0, result.stderr
    config = (tmp_path / "cluster.conf").read_text()
    assert "CLUSTER_HOST_IP=leader-node\n" in config
    assert "CLUSTER_WORKER_HOST=198.51.100.10\n" in config
    assert "CLUSTER_WORKER_HOSTS=" not in config


def test_configure_rejects_unsafe_or_invalid_network_settings(tmp_path):
    for values in ({"CLUSTER_SELF_IP": "999.0.0.1"},
                   {"CLUSTER_PEER_IP": "worker;bad"},
                   {"CLUSTER_RDMA_PORT": "0"},
                   {"CLUSTER_RDMA_GIDS": "3"}):
        result = run_configure(tmp_path, CLUSTER_WORKER_HOSTS="worker-a", **values)
        assert result.returncode != 0
        assert not (tmp_path / "cluster.conf").exists()
