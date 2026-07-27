"""Process lockfile helpers for unattended Pi runs."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we can't signal it — treat as alive.
        return True
    except OSError:
        return False
    return True


@dataclass
class RunLock:
    path: Path
    acquired: bool = False

    def acquire(self, *, stale_seconds: int = 3600) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                pid = int(data.get("pid") or 0)
                ts = float(data.get("ts") or 0)
                # Dead holder → take over immediately (don't block the execute window).
                if pid and not _pid_alive(pid):
                    pass
                elif time.time() - ts < stale_seconds:
                    return False
            except Exception:
                # Corrupt lock — take over.
                pass
        payload = {"pid": os.getpid(), "ts": time.time()}
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        self.acquired = True
        return True

    def held_by_live_process(self) -> bool:
        if not self.path.exists():
            return False
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            pid = int(data.get("pid") or 0)
            return _pid_alive(pid)
        except Exception:
            return False

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            if self.path.exists():
                self.path.unlink()
        finally:
            self.acquired = False
