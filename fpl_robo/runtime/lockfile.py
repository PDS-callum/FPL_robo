"""Process lockfile helpers for unattended Pi runs."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunLock:
    path: Path
    acquired: bool = False

    def acquire(self, *, stale_seconds: int = 3600) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                ts = float(data.get("ts") or 0)
                if time.time() - ts < stale_seconds:
                    return False
            except Exception:
                # Corrupt lock — take over.
                pass
        payload = {"pid": os.getpid(), "ts": time.time()}
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        self.acquired = True
        return True

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            if self.path.exists():
                self.path.unlink()
        finally:
            self.acquired = False
