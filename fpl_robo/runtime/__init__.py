"""Runtime package for unattended season operation."""

from __future__ import annotations

from typing import Any

__all__ = ["SeasonRunResult", "run_week", "run_daemon"]


def __getattr__(name: str) -> Any:
    if name in {"SeasonRunResult", "run_week", "run_daemon"}:
        from fpl_robo.runtime import season_runner

        return getattr(season_runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
