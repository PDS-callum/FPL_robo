"""Detect when the long-running daemon is serving stale code on disk.

Python imports are fixed at process start. After a git pull / edit, the daemon
must exit so systemd can restart it — otherwise CLI runs (new code) get
overwritten by the next daemon cycle (old code).
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Iterable

logger = logging.getLogger("fpl_robo.season")

# Key modules whose mtime changes imply a restart is required.
_WATCH_RELPATHS = (
    "fpl_robo/config.py",
    "fpl_robo/runtime/season_runner.py",
    "fpl_robo/optimize/transfers.py",
    "fpl_robo/optimize/attack_floor.py",
    "fpl_robo/predict/baseline.py",
    "fpl_robo/signals/minutes_policy.py",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def watched_paths(root: Path | None = None) -> list[Path]:
    base = root or repo_root()
    return [base / rel for rel in _WATCH_RELPATHS]


def code_mtime_fingerprint(paths: Iterable[Path] | None = None) -> float:
    """Max mtime across watched source files (0 if none exist)."""
    best = 0.0
    for path in paths or watched_paths():
        try:
            best = max(best, float(path.stat().st_mtime))
        except OSError:
            continue
    return best


def read_disk_feature_version(config_path: Path | None = None) -> str | None:
    """Parse KG_ENRICH feature_version from config.py without importing it."""
    path = config_path or (repo_root() / "fpl_robo" / "config.py")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.search(
        r'["\']feature_version["\']\s*:\s*["\']([^"\']+)["\']',
        text,
    )
    return m.group(1) if m else None


class CodeFreshnessGuard:
    """Exit the process when watched source files change after start."""

    def __init__(
        self,
        *,
        feature_version: str | None,
        paths: list[Path] | None = None,
        started_at: float | None = None,
    ) -> None:
        self.feature_version = str(feature_version or "")
        self.paths = list(paths or watched_paths())
        self.started_at = float(time.time() if started_at is None else started_at)
        self.baseline_mtime = code_mtime_fingerprint(self.paths)

    def stale_reason(self) -> str | None:
        now_mtime = code_mtime_fingerprint(self.paths)
        if now_mtime > self.baseline_mtime + 0.5:
            return (
                f"source mtime advanced ({self.baseline_mtime:.0f}->{now_mtime:.0f}); "
                "restart required"
            )
        disk_feature = read_disk_feature_version()
        if (
            disk_feature
            and self.feature_version
            and disk_feature != self.feature_version
        ):
            return (
                f"feature_version drift process={self.feature_version!r} "
                f"disk={disk_feature!r}; restart required"
            )
        return None

    def check_or_exit(self, *, exit_code: int = 0) -> None:
        reason = self.stale_reason()
        if not reason:
            return
        logger.warning(
            "Stale daemon code detected: %s — exiting for systemd restart", reason
        )
        raise SystemExit(exit_code)
