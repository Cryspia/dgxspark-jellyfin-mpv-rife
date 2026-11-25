import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class DevDeployTest(unittest.TestCase):
    def run_deploy(self, config):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            (temp / "mpv").mkdir()
            (temp / "bin").mkdir()
            script = temp / "bin" / "rsync"
            script.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$MOCK_RSYNC_OUT"\n')
            script.chmod(0o755)
            (temp / "dual.conf").write_text(config)
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith(("CLUSTER_", "DUAL_"))}
            env.update(MPV_CFG=str(temp / "mpv"),
                       CLUSTER_CFG=str(temp / "dual.conf"), USER="testuser",
                       MOCK_RSYNC_OUT=str(temp / "rsync.log"),
                       PATH=str(temp / "bin") + os.pathsep + os.environ["PATH"])
            result = subprocess.run([str(root / "scripts/dev_deploy.sh"), "--worker"],
                                    env=env, text=True, capture_output=True)
            log = temp / "rsync.log"
            calls = log.read_text().splitlines() if log.exists() else []
            return result, calls

    def test_deploys_every_configured_worker(self):
        result, calls = self.run_deploy(
            'CLUSTER_WORKER_HOSTS="alpha, beta,gamma"\nCLUSTER_WORKER_USER=tester\n')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 3)
        for call, host in zip(calls, ("alpha", "beta", "gamma")):
            self.assertTrue(call.endswith(f"tester@{host}:.local/share/dgxspark-mpv/worker/"))

    def test_single_worker_fallback(self):
        result, calls = self.run_deploy("CLUSTER_WORKER_HOST=legacy\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 1)
        self.assertIn("testuser@legacy:", calls[0])

    def test_separate_management_addresses(self):
        result, calls = self.run_deploy(
            'CLUSTER_WORKER_HOSTS="rail-a,rail-b"\n'
            'CLUSTER_WORKER_SSH_HOSTS="manage-a,manage-b"\n')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 2)
        self.assertIn("testuser@manage-a:", calls[0])
        self.assertIn("testuser@manage-b:", calls[1])

    def test_management_address_count_must_match(self):
        result, calls = self.run_deploy(
            'CLUSTER_WORKER_HOSTS="rail-a,rail-b"\nCLUSTER_WORKER_SSH_HOSTS="manage-a"\n')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls, [])

    def test_duplicate_worker_is_rejected_before_remote_copy(self):
        result, calls = self.run_deploy('CLUSTER_WORKER_HOSTS="alpha,alpha"\n')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
