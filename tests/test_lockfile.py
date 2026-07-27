import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fpl_robo.runtime.lockfile import RunLock


class RunLockTests(unittest.TestCase):
    def test_dead_pid_lock_is_taken_over(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.lock"
            path.write_text(json.dumps({"pid": 999999, "ts": 10**12}), encoding="utf-8")
            lock = RunLock(path)
            with mock.patch("fpl_robo.runtime.lockfile._pid_alive", return_value=False):
                self.assertTrue(lock.acquire(stale_seconds=3600))
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["pid"], os.getpid())
            lock.release()
            self.assertFalse(path.exists())

    def test_live_pid_lock_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.lock"
            path.write_text(json.dumps({"pid": 1, "ts": 10**12}), encoding="utf-8")
            lock = RunLock(path)
            with mock.patch("fpl_robo.runtime.lockfile._pid_alive", return_value=True):
                self.assertFalse(lock.acquire(stale_seconds=3600))


if __name__ == "__main__":
    unittest.main()
