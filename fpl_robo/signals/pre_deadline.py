"""Pre-deadline signal scrape gate (~2h before GW lock)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fpl_robo.config import SIGNAL_INGEST
from fpl_robo.data.fpl_client import FPLClient
from fpl_robo.runtime.atomic import atomic_write_json
from fpl_robo.runtime.deadline import hours_to_deadline, next_deadline

logger = logging.getLogger("fpl_robo.signals.pre_deadline")


def load_pre_deadline_stamp(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path or SIGNAL_INGEST["pre_deadline_stamp_path"])
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def write_pre_deadline_stamp(
    *,
    gameweek: int,
    hours_to_deadline_value: float,
    path: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "gameweek": int(gameweek),
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "hours_to_deadline": round(float(hours_to_deadline_value), 3),
        **(extra or {}),
    }
    atomic_write_json(path or SIGNAL_INGEST["pre_deadline_stamp_path"], payload)


def should_run_pre_deadline_ingest(
    *,
    client: FPLClient | None = None,
    force: bool = False,
    hours_horizon: float | None = None,
) -> dict[str, Any]:
    """Decide whether to run the final pre-deadline scrape for the next GW.

    Returns a decision dict with keys: run, reason, gameweek, hours_to_deadline.
    """
    horizon = float(
        SIGNAL_INGEST["pre_deadline_hours"] if hours_horizon is None else hours_horizon
    )
    client = client or FPLClient()
    bootstrap = client.bootstrap_static()
    deadline = next_deadline(bootstrap)
    if deadline is None:
        return {
            "run": False,
            "reason": "no_upcoming_deadline",
            "gameweek": None,
            "hours_to_deadline": None,
        }
    gw = int(deadline["id"])
    hours = hours_to_deadline(deadline["deadline_utc"])
    if force:
        return {
            "run": True,
            "reason": "forced",
            "gameweek": gw,
            "hours_to_deadline": hours,
            "deadline_time": deadline.get("deadline_time"),
        }
    if hours <= 0:
        return {
            "run": False,
            "reason": "deadline_passed",
            "gameweek": gw,
            "hours_to_deadline": hours,
        }
    if hours > horizon:
        return {
            "run": False,
            "reason": f"outside_window(>{horizon}h)",
            "gameweek": gw,
            "hours_to_deadline": hours,
            "deadline_time": deadline.get("deadline_time"),
        }
    stamp = load_pre_deadline_stamp()
    if int(stamp.get("gameweek") or -1) == gw:
        return {
            "run": False,
            "reason": "already_ran_this_gw",
            "gameweek": gw,
            "hours_to_deadline": hours,
            "stamp": stamp,
            "deadline_time": deadline.get("deadline_time"),
        }
    return {
        "run": True,
        "reason": "within_pre_deadline_window",
        "gameweek": gw,
        "hours_to_deadline": hours,
        "deadline_time": deadline.get("deadline_time"),
    }


def maybe_run_pre_deadline_ingest(
    *,
    client: FPLClient | None = None,
    force: bool = False,
    apply: bool = True,
) -> dict[str, Any]:
    """Run lineup-focused ingest once when inside the ~2h pre-deadline window."""
    from fpl_robo.signals.ingest import ingest_signals

    decision = should_run_pre_deadline_ingest(client=client, force=force)
    if not decision.get("run"):
        logger.info(
            "Pre-deadline ingest skipped: %s (gw=%s hours=%.2f)",
            decision.get("reason"),
            decision.get("gameweek"),
            float(decision.get("hours_to_deadline") or 0.0),
        )
        return {"ok": True, "skipped": True, **decision}

    gw = int(decision["gameweek"])
    hours = float(decision["hours_to_deadline"])
    logger.info(
        "Running pre-deadline signal ingest for GW%s (%.2fh to deadline)",
        gw,
        hours,
    )
    result = ingest_signals(
        dry_run=not apply,
        apply=apply,
        client=client,
        pre_deadline=True,
        gameweek=gw,
    )
    if apply and result.ok:
        write_pre_deadline_stamp(
            gameweek=gw,
            hours_to_deadline_value=hours,
            extra={
                "proposed_news": len(result.proposed_news),
                "proposed_lineups": len(getattr(result, "proposed_lineups", []) or []),
                "reason": result.reason,
            },
        )
    out = {
        "ok": result.ok,
        "skipped": False,
        **decision,
        "ingest": result.to_dict(),
    }
    return out
