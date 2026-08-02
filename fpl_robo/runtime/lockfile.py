"""Process lockfile helpers for unattended Pi runs."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore


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
    _fh: TextIO[str] | None = field(default=None, repr=False, compare=False)

    def acquire(self, *, stale_seconds: int = 3600) -> bool:
        """Acquire exclusive lock (fcntl when available) + pid stamp."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Open for read/write; create if missing. Keep FD for flock lifetime.
        fh = open(self.path, "a+", encoding="utf-8")
        try:
            if fcntl is not None:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    fh.close()
                    return False
                except OSError:
                    fh.close()
                    return False

            fh.seek(0)
            raw = fh.read()
            if raw.strip():
                try:
                    data = json.loads(raw)
                    pid = int(data.get("pid") or 0)
                    ts = float(data.get("ts") or 0)
                    if pid and pid != os.getpid() and _pid_alive(pid):
                        if time.time() - ts < stale_seconds:
                            if fcntl is not None:
                                try:
                                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                                except OSError:
                                    pass
                            fh.close()
                            return False
                except Exception:
                    # Corrupt lock — take over.
                    pass

            payload = {"pid": os.getpid(), "ts": time.time()}
            fh.seek(0)
            fh.truncate()
            fh.write(json.dumps(payload))
            fh.flush()
            os.fsync(fh.fileno())
            self._fh = fh
            self.acquired = True
            return True
        except Exception:
            try:
                fh.close()
            except Exception:
                pass
            return False

    def touch(self) -> None:
        """Refresh lock stamp timestamp so healthy long runs are not stolen."""
        if not self.acquired or self._fh is None:
            return
        try:
            payload = {"pid": os.getpid(), "ts": time.time()}
            self._fh.seek(0)
            self._fh.truncate()
            self._fh.write(json.dumps(payload))
            self._fh.flush()
            os.fsync(self._fh.fileno())
        except Exception:
            pass

    def held_by_live_process(self) -> bool:
        if not self.path.exists():
            return False
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            pid = int(data.get("pid") or 0)
            return bool(pid) and pid != os.getpid() and _pid_alive(pid)
        except Exception:
            return False

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            if self._fh is not None:
                if fcntl is not None:
                    try:
                        fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
                    except OSError:
                        pass
                try:
                    self._fh.close()
                except Exception:
                    pass
                self._fh = None
            if self.path.exists():
                try:
                    data = json.loads(self.path.read_text(encoding="utf-8"))
                    if int(data.get("pid") or 0) == os.getpid():
                        self.path.unlink()
                except Exception:
                    try:
                        self.path.unlink()
                    except Exception:
                        pass
        finally:
            self.acquired = False
