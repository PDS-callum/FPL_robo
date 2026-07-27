"""Lagged rolling minutes/points from FPL event-live endpoints (live EV parity)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fpl_robo.data.fpl_client import FPLClient
from fpl_robo.runtime.atomic import atomic_write_json

logger = logging.getLogger("fpl_robo.data.rolling_history")

DEFAULT_CACHE_PATH = "data/cache/event_live_history.json"
ROLLING_WINDOW = 4


@dataclass
class PlayerHistoryFeatures:
    prior_minutes: float = 0.0
    rolling_minutes: float = 0.0
    prior_points: float = 0.0
    rolling_points: float = 0.0


def finished_gameweeks_before(
    bootstrap: dict[str, Any],
    *,
    before_gw: int,
    limit: int = ROLLING_WINDOW,
) -> list[int]:
    """Return up to `limit` finished GW ids strictly before `before_gw`, ascending."""
    finished: list[int] = []
    for event in bootstrap.get("events") or []:
        try:
            gw = int(event["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if gw >= int(before_gw):
            continue
        if event.get("finished") or event.get("data_checked"):
            finished.append(gw)
    finished = sorted(set(finished))
    if limit > 0:
        finished = finished[-limit:]
    return finished


def _extract_element_stats(payload: dict[str, Any]) -> dict[int, dict[str, float]]:
    """Map element id -> {minutes, total_points} from an event/live payload."""
    out: dict[int, dict[str, float]] = {}
    for row in payload.get("elements") or []:
        if not isinstance(row, dict):
            continue
        try:
            pid = int(row["id"])
        except (KeyError, TypeError, ValueError):
            continue
        stats = row.get("stats") or {}
        out[pid] = {
            "minutes": float(stats.get("minutes") or 0.0),
            "total_points": float(stats.get("total_points") or 0.0),
        }
    return out


def load_cache(path: str | Path = DEFAULT_CACHE_PATH) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"gameweeks": {}}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {"gameweeks": {}}
        gws = raw.get("gameweeks")
        if not isinstance(gws, dict):
            raw["gameweeks"] = {}
        return raw
    except Exception:
        return {"gameweeks": {}}


def save_cache(cache: dict[str, Any], path: str | Path = DEFAULT_CACHE_PATH) -> None:
    atomic_write_json(path, cache)


def ensure_event_live_history(
    *,
    client: FPLClient,
    bootstrap: dict[str, Any],
    decision_gw: int,
    window: int = ROLLING_WINDOW,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
) -> dict[int, dict[int, dict[str, float]]]:
    """Fetch+cache per-player minutes/points for lagged finished GWs.

    Returns: {gameweek: {player_id: {minutes, total_points}}}
    """
    needed = finished_gameweeks_before(bootstrap, before_gw=decision_gw, limit=window)
    cache = load_cache(cache_path)
    gws: dict[str, Any] = cache.setdefault("gameweeks", {})
    result: dict[int, dict[int, dict[str, float]]] = {}

    for gw in needed:
        key = str(gw)
        if key in gws and isinstance(gws[key], dict) and gws[key]:
            # Keys may be strings in JSON; normalise to int player ids.
            parsed: dict[int, dict[str, float]] = {}
            for pid_s, stats in gws[key].items():
                try:
                    parsed[int(pid_s)] = {
                        "minutes": float((stats or {}).get("minutes") or 0.0),
                        "total_points": float((stats or {}).get("total_points") or 0.0),
                    }
                except (TypeError, ValueError):
                    continue
            result[gw] = parsed
            continue
        try:
            payload = client.event_live(gw)
            parsed = _extract_element_stats(payload)
            gws[key] = {str(pid): stats for pid, stats in parsed.items()}
            result[gw] = parsed
            logger.info("Cached event-live stats for GW%s (%s players)", gw, len(parsed))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch event-live for GW%s: %s", gw, exc)
            result[gw] = {}

    cache["decision_gw"] = int(decision_gw)
    save_cache(cache, cache_path)
    return result


def aggregate_player_history(
    gw_stats: dict[int, dict[int, dict[str, float]]],
    *,
    window: int = ROLLING_WINDOW,
) -> dict[int, PlayerHistoryFeatures]:
    """Build prior/rolling minutes+points from lagged GW maps (no same-GW leakage).

    `gw_stats` should only contain GWs strictly before the decision GW.
    Rolling = mean of the most recent `window` GWs present (missing GW => 0 for that slot
    only if the GW was requested; players absent in a GW count as 0).
    """
    if not gw_stats:
        return {}
    ordered = sorted(gw_stats.keys())
    if window > 0:
        ordered = ordered[-window:]

    player_ids: set[int] = set()
    for gw in ordered:
        player_ids.update(gw_stats.get(gw, {}).keys())

    out: dict[int, PlayerHistoryFeatures] = {}
    for pid in player_ids:
        mins_list: list[float] = []
        pts_list: list[float] = []
        prior_m = 0.0
        prior_p = 0.0
        for gw in ordered:
            stats = (gw_stats.get(gw) or {}).get(pid) or {}
            m = float(stats.get("minutes") or 0.0)
            p = float(stats.get("total_points") or 0.0)
            mins_list.append(m)
            pts_list.append(p)
            prior_m += m
            prior_p += p
        n = len(mins_list) or 1
        out[pid] = PlayerHistoryFeatures(
            prior_minutes=prior_m,
            rolling_minutes=sum(mins_list) / float(n),
            prior_points=prior_p,
            rolling_points=sum(pts_list) / float(n),
        )
    return out


def load_rolling_features(
    *,
    client: FPLClient,
    bootstrap: dict[str, Any],
    decision_gw: int,
    window: int = ROLLING_WINDOW,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
) -> dict[int, PlayerHistoryFeatures]:
    gw_stats = ensure_event_live_history(
        client=client,
        bootstrap=bootstrap,
        decision_gw=decision_gw,
        window=window,
        cache_path=cache_path,
    )
    return aggregate_player_history(gw_stats, window=window)


def attach_history_to_graph(
    kg: Any,
    features: dict[int, PlayerHistoryFeatures],
) -> int:
    """Write prior/rolling minutes+points onto player node attrs. Returns count updated."""
    from fpl_robo.graph.schema import NodeType

    if not features:
        return 0
    updated = 0
    for node_id, attrs in kg.nodes_of_type(NodeType.PLAYER):
        pid = int(node_id.key)
        feat = features.get(pid)
        if feat is None:
            continue
        attrs["prior_minutes"] = float(feat.prior_minutes)
        attrs["rolling_minutes"] = float(feat.rolling_minutes)
        attrs["prior_points"] = float(feat.prior_points)
        attrs["rolling_points"] = float(feat.rolling_points)
        attrs["history_attached"] = True
        updated += 1
    return updated
