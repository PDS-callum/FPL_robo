"""Scheduled signal ingest gates (daily / pre-match / pre-deadline)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fpl_robo.config import SIGNAL_INGEST
from fpl_robo.data.fpl_client import FPLClient
from fpl_robo.runtime.atomic import atomic_write_json
from fpl_robo.runtime.cadence import decide_ingest, record_ingest
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

    Kept for CLI/tests; daily / pre-match use `decide_ingest` via the daemon.
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
    """Back-compat: final pre-deadline ingest only."""
    return maybe_run_scheduled_ingest(
        client=client,
        force=force,
        apply=apply,
        kinds={"pre_deadline", "forced"} if not force else None,
    )


def maybe_run_scheduled_ingest(
    *,
    client: FPLClient | None = None,
    force: bool = False,
    apply: bool = True,
    kinds: set[str] | None = None,
    fixtures: list[dict[str, Any]] | None = None,
    bootstrap: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run daily / pre-match / pre-deadline ingest when due."""
    from fpl_robo.signals.ingest import ingest_signals

    client = client or FPLClient()
    bootstrap = bootstrap or client.bootstrap_static()
    fixtures = fixtures if fixtures is not None else client.fixtures()
    decision = decide_ingest(
        bootstrap=bootstrap,
        fixtures=fixtures,
        force=force,
    )
    kind = decision.get("kind")
    if kinds is not None and kind not in kinds and not force:
        return {
            "ok": True,
            "skipped": True,
            "reason": f"kind_filtered({kind})",
            **{k: decision.get(k) for k in ("gameweek", "hours_to_deadline")},
        }
    if not decision.get("run"):
        logger.info(
            "Scheduled ingest skipped: %s (gw=%s hours=%s)",
            decision.get("reason"),
            decision.get("gameweek"),
            decision.get("hours_to_deadline"),
        )
        return {"ok": True, "skipped": True, **decision}

    gw = decision.get("gameweek")
    hours = decision.get("hours_to_deadline")
    lineup_focus = bool(decision.get("lineup_focus"))
    logger.info(
        "Running %s signal ingest (gw=%s hours_to_deadline=%s)",
        kind,
        gw,
        hours,
    )
    result = ingest_signals(
        dry_run=not apply,
        apply=apply,
        client=client,
        pre_deadline=lineup_focus,
        gameweek=int(gw) if gw is not None else None,
    )
    if apply and result.ok:
        record_ingest(
            kind=str(kind or "unknown"),
            gameweek=int(gw) if gw is not None else None,
            hours_to_deadline_value=float(hours) if hours is not None else None,
            extra={
                "proposed_news": len(result.proposed_news),
                "proposed_lineups": len(getattr(result, "proposed_lineups", []) or []),
                "reason": result.reason,
                "content_changed": bool(getattr(result, "content_changed", True)),
            },
        )
    return {
        "ok": result.ok,
        "skipped": False,
        **decision,
        "ingest": result.to_dict(),
    }
