"""Daemon cadence: when to scrape news and when to re-optimise."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from fpl_robo.config import KG_ENRICH, SEASON_RUNNER, SIGNAL_INGEST
from fpl_robo.runtime.deadline import hours_to_deadline, next_deadline

logger = logging.getLogger("fpl_robo.runtime.cadence")


@dataclass
class CadenceDecision:
    gameweek: int | None
    hours_to_deadline: float | None
    should_ingest: bool = False
    ingest_kind: str | None = None  # daily | pre_deadline | forced
    ingest_reason: str | None = None
    should_optimise: bool = False
    optimise_triggers: list[str] = field(default_factory=list)
    skip_reason: str | None = None
    sleep_seconds: int = 900
    bootstrap_fingerprint: str | None = None
    price_status_fingerprint: str | None = None
    first_kickoff_today: str | None = None
    next_daily_ingest_at: str | None = None
    phase: str = "idle"


def local_tz() -> ZoneInfo:
    name = str(SIGNAL_INGEST.get("timezone") or "Europe/London")
    try:
        return ZoneInfo(name)
    except Exception:
        return ZoneInfo("UTC")


def bootstrap_fingerprint(bootstrap: dict[str, Any]) -> str:
    """Compact hash of prediction-relevant bootstrap fields."""
    # Bump when KG EV feature set changes so cadence re-plans once.
    parts: list[str] = [f"kg_features:{KG_ENRICH.get('feature_version') or 'v2'}"]
    events = bootstrap.get("events") or []
    for ev in events:
        if ev.get("is_next") or ev.get("is_current"):
            parts.append(f"ev:{ev.get('id')}:{ev.get('deadline_time')}")
    for el in sorted(bootstrap.get("elements") or [], key=lambda e: int(e.get("id") or 0)):
        parts.append(
            "|".join(
                str(x)
                for x in (
                    el.get("id"),
                    el.get("now_cost"),
                    el.get("status"),
                    el.get("chance_of_playing_next_round"),
                    el.get("chance_of_playing_this_round"),
                    el.get("ep_next"),
                    el.get("news"),
                    el.get("form"),
                )
            )
        )
    digest = hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()
    return digest[:20]


def price_status_fingerprint(bootstrap: dict[str, Any]) -> str:
    """Hash of price / availability only (explicit price_status_change trigger)."""
    parts: list[str] = []
    for el in sorted(bootstrap.get("elements") or [], key=lambda e: int(e.get("id") or 0)):
        parts.append(
            "|".join(
                str(x)
                for x in (
                    el.get("id"),
                    el.get("now_cost"),
                    el.get("status"),
                    el.get("chance_of_playing_next_round"),
                    el.get("chance_of_playing_this_round"),
                )
            )
        )
    digest = hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()
    return digest[:20]


def signals_content_fingerprint(payload: dict[str, Any] | None) -> str | None:
    """Stable hash of prediction-relevant signal rows (ignores updated_at / ingest meta)."""
    if not isinstance(payload, dict):
        return None

    def _rows(key: str) -> list[str]:
        items = payload.get(key) or []
        out: list[str] = []
        if not isinstance(items, list):
            return out
        for item in items:
            if not isinstance(item, dict):
                continue
            out.append(
                "|".join(
                    str(item.get(k) if item.get(k) is not None else "")
                    for k in (
                        "player_id",
                        "web_name",
                        "type",
                        "kind",
                        "severity",
                        "exclude",
                        "source",
                        "boost",
                        "minutes",
                        "role",
                        "gameweek",
                        "confidence",
                    )
                )
            )
        return sorted(out)

    parts = (
        [f"news:{x}" for x in _rows("news")]
        + [f"pre:{x}" for x in _rows("preseason")]
        + [f"intl:{x}" for x in _rows("international")]
        + [f"xi:{x}" for x in _rows("predicted_lineups")]
    )
    digest = hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()
    return digest[:20]


def signals_fingerprint(path: str | Path | None = None) -> str | None:
    """Content fingerprint of signals.json on disk."""
    p = Path(path or SIGNAL_INGEST.get("signals_path") or "data/signals.json")
    if not p.exists():
        return None
    try:
        import json

        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return signals_content_fingerprint(raw if isinstance(raw, dict) else None)


def signals_mtime(path: str | Path | None = None) -> float | None:
    p = Path(path or SIGNAL_INGEST.get("signals_path") or "data/signals.json")
    if not p.exists():
        return None
    try:
        return p.stat().st_mtime
    except OSError:
        return None


def load_ingest_stamp(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path or SIGNAL_INGEST.get("ingest_stamp_path") or "data/cache/ingest_schedule.json")
    if not p.exists():
        # Fall back to legacy pre-deadline stamp shape.
        legacy = Path(SIGNAL_INGEST.get("pre_deadline_stamp_path") or "")
        if legacy.exists():
            try:
                import json

                raw = json.loads(legacy.read_text(encoding="utf-8"))
                return raw if isinstance(raw, dict) else {}
            except Exception:
                return {}
        return {}
    try:
        import json

        raw = json.loads(p.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def write_ingest_stamp(payload: dict[str, Any], path: str | Path | None = None) -> None:
    from fpl_robo.runtime.atomic import atomic_write_json

    atomic_write_json(
        path or SIGNAL_INGEST.get("ingest_stamp_path") or "data/cache/ingest_schedule.json",
        payload,
    )


def _parse_kickoff(raw: Any) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def first_kickoff_on_local_day(
    fixtures: list[dict[str, Any]],
    *,
    day: datetime | None = None,
    tz: ZoneInfo | None = None,
) -> datetime | None:
    """Earliest unfinished kickoff on the given local calendar day."""
    tz = tz or local_tz()
    day = day or datetime.now(tz)
    local_date = day.astimezone(tz).date()
    kicks: list[datetime] = []
    for fix in fixtures:
        if fix.get("finished"):
            continue
        ko = _parse_kickoff(fix.get("kickoff_time"))
        if ko is None:
            continue
        if ko.astimezone(tz).date() != local_date:
            continue
        kicks.append(ko)
    return min(kicks) if kicks else None


def next_daily_ingest_utc(now: datetime | None = None) -> datetime:
    """Next (or today's pending) daily ingest instant in UTC."""
    tz = local_tz()
    now = (now or datetime.now(timezone.utc)).astimezone(tz)
    hour = int(SIGNAL_INGEST.get("daily_ingest_hour") or 8)
    minute = int(SIGNAL_INGEST.get("daily_ingest_minute") or 30)
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if now < candidate:
        return candidate.astimezone(timezone.utc)
    return (candidate + timedelta(days=1)).astimezone(timezone.utc)


def _daily_local_today(local_now: datetime) -> datetime:
    hour = int(SIGNAL_INGEST.get("daily_ingest_hour") or 8)
    minute = int(SIGNAL_INGEST.get("daily_ingest_minute") or 30)
    return local_now.replace(hour=hour, minute=minute, second=0, microsecond=0)


def _deadline_before_daily_today(
    deadline: dict[str, Any] | None,
    *,
    local_now: datetime,
    daily_today: datetime,
) -> bool:
    """True when the next FPL deadline is today at/before the daily ingest time."""
    if not deadline or not deadline.get("deadline_utc"):
        return False
    dl_local = deadline["deadline_utc"].astimezone(local_now.tzinfo)
    if dl_local.date() != local_now.date():
        return False
    return dl_local <= daily_today


def decide_ingest(
    *,
    bootstrap: dict[str, Any],
    fixtures: list[dict[str, Any]],
    stamp: dict[str, Any] | None = None,
    now: datetime | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Pick at most one ingest action: pre_deadline or daily.

    Policy:
    - Daily scrape at the configured local time (morning window).
    - If a GW deadline falls that local day, also scrape once ~pre_deadline_hours before it.
    - If that deadline is at/before the daily time, skip daily and only do the pre-deadline scrape.
    """
    now = now or datetime.now(timezone.utc)
    tz = local_tz()
    local_now = now.astimezone(tz)
    stamp = stamp if stamp is not None else load_ingest_stamp()
    deadline = next_deadline(bootstrap)
    gw = int(deadline["id"]) if deadline else None
    hours = hours_to_deadline(deadline["deadline_utc"], now=now) if deadline else None
    daily_today = _daily_local_today(local_now)
    skip_daily = _deadline_before_daily_today(
        deadline, local_now=local_now, daily_today=daily_today
    )

    if force:
        return {
            "run": True,
            "kind": "forced",
            "reason": "forced",
            "gameweek": gw,
            "hours_to_deadline": hours,
            "lineup_focus": True,
        }

    # Priority: pre-deadline (lineups) > daily.
    pre_deadline_h = float(SIGNAL_INGEST.get("pre_deadline_hours") or 0.5)

    if hours is not None and 0 < hours <= pre_deadline_h:
        if int(stamp.get("pre_deadline_gw") or -1) != int(gw or -2):
            return {
                "run": True,
                "kind": "pre_deadline",
                "reason": "within_pre_deadline_window",
                "gameweek": gw,
                "hours_to_deadline": hours,
                "lineup_focus": True,
            }

    window_h = float(SIGNAL_INGEST.get("daily_ingest_window_hours") or 6.0)
    daily_end = daily_today + timedelta(hours=window_h)
    local_date = local_now.date().isoformat()
    if (
        not skip_daily
        and daily_today <= local_now <= daily_end
        and stamp.get("daily_date") != local_date
    ):
        return {
            "run": True,
            "kind": "daily",
            "reason": "daily_window",
            "gameweek": gw,
            "hours_to_deadline": hours,
            "lineup_focus": False,
        }

    reason = "no_ingest_due"
    if skip_daily and stamp.get("daily_date") != local_date and daily_today <= local_now <= daily_end:
        reason = "skip_daily_deadline_earlier_today"

    return {
        "run": False,
        "kind": None,
        "reason": reason,
        "gameweek": gw,
        "hours_to_deadline": hours,
        "next_daily_ingest_at": next_daily_ingest_utc(now).isoformat(),
        "skip_daily_for_deadline": skip_daily,
    }


def decide_optimise(
    *,
    bootstrap: dict[str, Any],
    state: dict[str, Any],
    stamp: dict[str, Any] | None = None,
    now: datetime | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Optimise only when prediction inputs may have changed, or final lock window."""
    now = now or datetime.now(timezone.utc)
    stamp = stamp if stamp is not None else load_ingest_stamp()
    cadence = state.get("cadence") if isinstance(state.get("cadence"), dict) else {}
    deadline = next_deadline(bootstrap)
    gw = int(deadline["id"]) if deadline else None
    hours = hours_to_deadline(deadline["deadline_utc"], now=now) if deadline else None
    fp = bootstrap_fingerprint(bootstrap)
    exec_h = float(SEASON_RUNNER.get("execute_within_hours") or 0.5)
    near_h = float(SEASON_RUNNER.get("near_deadline_hours") or 2.0)

    triggers: list[str] = []
    if force:
        triggers.append("forced")

    last_gw = cadence.get("last_optimise_gw")
    if gw is not None and (last_gw is None or int(last_gw) != int(gw)):
        triggers.append("new_gameweek")

    ps_fp = price_status_fingerprint(bootstrap)
    if cadence.get("last_price_status_fp") != ps_fp:
        triggers.append("price_status_change")

    if cadence.get("last_bootstrap_fp") != fp:
        triggers.append("bootstrap_change")

    sig_fp = signals_fingerprint()
    last_fp = cadence.get("last_signals_fp")
    sig_m = signals_mtime()
    last_sig = cadence.get("last_signals_mtime")
    # Prefer content fingerprint so rewritten metadata / mtime flicker cannot thrash.
    if sig_fp is not None:
        if last_fp is None or str(last_fp) != str(sig_fp):
            triggers.append("new_signals")
    elif sig_m is not None and (last_sig is None or float(sig_m) > float(last_sig) + 0.5):
        triggers.append("new_signals")

    # Fresh ingest kinds that should replan only when content actually moved.
    pending = stamp.get("pending_optimise")
    if pending and stamp.get("content_unchanged") is not True:
        triggers.append(f"after_ingest:{pending}")

    if hours is not None and 0 < hours <= exec_h:
        if not cadence.get("final_optimise_gw") or int(cadence.get("final_optimise_gw") or -1) != int(gw or -2):
            triggers.append("execute_window")
    elif hours is not None and 0 < hours <= near_h:
        # One near-deadline plan if we have not already done a final/near plan this GW.
        if not cadence.get("near_optimise_gw") or int(cadence.get("near_optimise_gw") or -1) != int(gw or -2):
            # Only if something else also moved, or we never planned this GW — avoid spam.
            # Actually user asked: "again finally before the deadline". So once in near window is good.
            if "new_gameweek" not in triggers and last_gw is not None:
                triggers.append("pre_deadline_plan")

    # Deduplicate while preserving order.
    seen: set[str] = set()
    uniq: list[str] = []
    for t in triggers:
        if t not in seen:
            seen.add(t)
            uniq.append(t)

    should = bool(uniq)
    return {
        "run": should,
        "triggers": uniq,
        "reason": ",".join(uniq) if should else "no_new_data",
        "gameweek": gw,
        "hours_to_deadline": hours,
        "bootstrap_fingerprint": fp,
        "price_status_fingerprint": ps_fp,
        "signals_mtime": sig_m,
        "signals_fingerprint": sig_fp,
        "phase": _phase(hours, exec_h=exec_h, near_h=near_h),
    }


def _phase(hours: float | None, *, exec_h: float, near_h: float) -> str:
    if hours is None:
        return "unknown"
    if hours <= 0:
        return "locked"
    if hours <= exec_h:
        return "execute"
    if hours <= near_h:
        return "near_deadline"
    return "cruise"


def evaluate_cadence(
    *,
    bootstrap: dict[str, Any],
    fixtures: list[dict[str, Any]],
    state: dict[str, Any],
    now: datetime | None = None,
    force_ingest: bool = False,
    force_optimise: bool = False,
) -> CadenceDecision:
    now = now or datetime.now(timezone.utc)
    stamp = load_ingest_stamp()
    ingest = decide_ingest(
        bootstrap=bootstrap,
        fixtures=fixtures,
        stamp=stamp,
        now=now,
        force=force_ingest,
    )
    optimise = decide_optimise(
        bootstrap=bootstrap,
        state=state,
        stamp=stamp,
        now=now,
        force=force_optimise,
    )

    hours = ingest.get("hours_to_deadline")
    if hours is None:
        hours = optimise.get("hours_to_deadline")
    near_h = float(SEASON_RUNNER.get("near_deadline_hours") or 2.0)
    near_poll = int(SEASON_RUNNER.get("near_deadline_poll_seconds") or 300)
    heartbeat = int(SEASON_RUNNER.get("heartbeat_seconds") or SEASON_RUNNER.get("poll_seconds") or 900)
    exec_h = float(SEASON_RUNNER.get("execute_within_hours") or 0.5)

    sleep = heartbeat
    if hours is not None and 0 < hours <= near_h:
        sleep = near_poll
    # Wake closer to daily ingest if sooner than heartbeat.
    nxt = next_daily_ingest_utc(now)
    until_daily = max(30, int((nxt - now).total_seconds()))
    if until_daily < sleep and not ingest.get("run"):
        sleep = until_daily

    # Wake for the pre-deadline scrape window.
    pre_deadline_h = float(SIGNAL_INGEST.get("pre_deadline_hours") or 0.5)
    if hours is not None and hours > pre_deadline_h:
        until_pd = max(30, int((hours - pre_deadline_h) * 3600.0))
        if until_pd < sleep and not ingest.get("run"):
            sleep = until_pd

    first_ko = first_kickoff_on_local_day(fixtures, day=now.astimezone(local_tz()), tz=local_tz())

    return CadenceDecision(
        gameweek=ingest.get("gameweek") or optimise.get("gameweek"),
        hours_to_deadline=float(hours) if hours is not None else None,
        should_ingest=bool(ingest.get("run")),
        ingest_kind=ingest.get("kind"),
        ingest_reason=ingest.get("reason"),
        should_optimise=bool(optimise.get("run")),
        optimise_triggers=list(optimise.get("triggers") or []),
        skip_reason=None if optimise.get("run") else str(optimise.get("reason") or "no_new_data"),
        sleep_seconds=int(sleep),
        bootstrap_fingerprint=optimise.get("bootstrap_fingerprint"),
        price_status_fingerprint=optimise.get("price_status_fingerprint"),
        first_kickoff_today=first_ko.isoformat() if first_ko else None,
        next_daily_ingest_at=next_daily_ingest_utc(now).isoformat(),
        phase=_phase(
            float(hours) if hours is not None else None,
            exec_h=exec_h,
            near_h=near_h,
        ),
    )


def record_ingest(
    *,
    kind: str,
    gameweek: int | None,
    hours_to_deadline_value: float | None,
    extra: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    local_date = now.astimezone(local_tz()).date().isoformat()
    stamp = load_ingest_stamp()
    extra = dict(extra or {})
    content_changed = extra.pop("content_changed", True)
    stamp.update(
        {
            "last_kind": kind,
            "ran_at": now.isoformat(),
            "gameweek": gameweek,
            "hours_to_deadline": (
                round(float(hours_to_deadline_value), 3)
                if hours_to_deadline_value is not None
                else None
            ),
            # Only force a follow-up plan when signal content actually moved.
            "pending_optimise": kind if content_changed else None,
            "content_unchanged": (not content_changed),
            **extra,
        }
    )
    if kind == "daily":
        stamp["daily_date"] = local_date
    elif kind in {"pre_deadline", "forced"}:
        if gameweek is not None:
            stamp["pre_deadline_gw"] = int(gameweek)
        # Keep legacy field for older readers.
        stamp["gameweek"] = gameweek
    write_ingest_stamp(stamp)
    # Also mirror legacy pre-deadline stamp for compatibility.
    if kind in {"pre_deadline", "forced"} and gameweek is not None:
        from fpl_robo.signals.pre_deadline import write_pre_deadline_stamp

        write_pre_deadline_stamp(
            gameweek=int(gameweek),
            hours_to_deadline_value=float(hours_to_deadline_value or 0.0),
            extra={"kind": kind, **(extra or {})},
        )
    return stamp


def record_optimise(
    state: dict[str, Any],
    *,
    gameweek: int | None,
    triggers: list[str],
    bootstrap_fp: str | None,
    signals_m: float | None = None,
    hours_to_deadline_value: float | None = None,
    now: datetime | None = None,
    price_status_fp: str | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    cadence = dict(state.get("cadence") or {})
    cadence.update(
        {
            "last_optimise_at": now.isoformat(),
            "last_optimise_gw": gameweek,
            "last_optimise_triggers": list(triggers),
            "last_bootstrap_fp": bootstrap_fp,
            "last_price_status_fp": price_status_fp,
            "last_signals_mtime": signals_m if signals_m is not None else signals_mtime(),
            "last_signals_fp": signals_fingerprint(),
        }
    )
    exec_h = float(SEASON_RUNNER.get("execute_within_hours") or 0.5)
    near_h = float(SEASON_RUNNER.get("near_deadline_hours") or 2.0)
    if hours_to_deadline_value is not None and gameweek is not None:
        if 0 < float(hours_to_deadline_value) <= exec_h:
            cadence["final_optimise_gw"] = int(gameweek)
            cadence["near_optimise_gw"] = int(gameweek)
        elif 0 < float(hours_to_deadline_value) <= near_h:
            cadence["near_optimise_gw"] = int(gameweek)
    state["cadence"] = cadence

    # Clear pending ingest → optimise flag.
    stamp = load_ingest_stamp()
    if stamp.get("pending_optimise"):
        stamp["pending_optimise"] = None
        stamp["pending_cleared_at"] = now.isoformat()
        write_ingest_stamp(stamp)
    return state
