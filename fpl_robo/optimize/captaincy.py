"""Captain / vice-captain helpers shared by optimiser and team writes."""

from __future__ import annotations

from typing import Any

from fpl_robo.config import CAPTAINCY


def play_factor_for_captain(
    row: dict[str, Any] | Any,
    *,
    cfg: dict[str, Any] | None = None,
) -> float:
    """Minutes / availability factor in [floor, 1] for captain scoring."""
    c = {**CAPTAINCY, **(cfg or {})}
    try:
        floor = float(c.get("play_factor_floor") or 0.35)
    except (TypeError, ValueError):
        floor = 0.35
    factor = 1.0

    def _get(key: str, default: Any = None) -> Any:
        if isinstance(row, dict):
            return row.get(key, default)
        return getattr(row, key, default)

    chance = _get("chance_of_playing_next_round")
    if chance is not None:
        try:
            factor = min(factor, max(0.0, float(chance) / 100.0))
        except (TypeError, ValueError):
            pass

    status = str(_get("status") or "a").lower()
    if status in {"i", "u", "s"}:
        factor = min(factor, float(c.get("status_out_factor") or 0.15))
    elif status == "d":
        factor = min(factor, float(c.get("status_doubt_factor") or 0.70))

    role = str(_get("lineup_role") or "").lower().strip()
    conf = _get("lineup_confidence")
    try:
        conf_v = float(conf) if conf is not None else 0.0
    except (TypeError, ValueError):
        conf_v = 0.0
    min_conf = float(c.get("lineup_min_confidence") or 0.75)
    if conf_v >= min_conf:
        if role in {"bench", "benched", "reserve"}:
            factor = min(factor, float(c.get("lineup_bench_factor") or 0.55))
        elif role in {"doubt", "doubtful", "risk"}:
            factor = min(factor, float(c.get("lineup_doubt_factor") or 0.75))

    return max(floor, min(1.0, float(factor)))


def captain_score(
    row: dict[str, Any] | Any,
    *,
    cfg: dict[str, Any] | None = None,
) -> float:
    """EV × play_factor — used for MIP captain term and VC selection."""

    def _get(key: str, default: Any = None) -> Any:
        if isinstance(row, dict):
            return row.get(key, default)
        return getattr(row, key, default)

    try:
        ev = float(_get("ev") or 0.0)
    except (TypeError, ValueError):
        ev = 0.0
    return ev * play_factor_for_captain(row, cfg=cfg)


def pick_vice_captain_id(
    squad_rows: list[dict[str, Any]],
    captain_id: int | None,
    *,
    cfg: dict[str, Any] | None = None,
) -> int | None:
    """Choose VC: best captain_score starter; prefer other club within tolerance."""
    if not squad_rows:
        return None
    c = {**CAPTAINCY, **(cfg or {})}
    try:
        club_tol = float(c.get("vice_other_club_tolerance") or 0.15)
    except (TypeError, ValueError):
        club_tol = 0.15

    cap = int(captain_id) if captain_id is not None else None
    cap_team = None
    if cap is not None:
        for r in squad_rows:
            try:
                if int(r.get("id")) == cap:
                    cap_team = r.get("team_id")
                    break
            except (TypeError, ValueError):
                continue

    starters = []
    for r in squad_rows:
        if not r.get("start"):
            continue
        try:
            rid = int(r["id"])
        except (TypeError, ValueError, KeyError):
            continue
        if cap is not None and rid == cap:
            continue
        starters.append(r)
    if not starters:
        others = [r for r in squad_rows if cap is None or int(r.get("id") or -1) != cap]
        others.sort(key=lambda r: captain_score(r, cfg=c), reverse=True)
        return int(others[0]["id"]) if others else None

    starters.sort(key=lambda r: captain_score(r, cfg=c), reverse=True)
    best = starters[0]
    best_score = captain_score(best, cfg=c)

    if cap_team is not None and bool(c.get("vice_prefer_other_club", True)):
        try:
            best_team = best.get("team_id")
        except Exception:
            best_team = None
        if best_team is not None and int(best_team) == int(cap_team):
            for alt in starters[1:]:
                try:
                    alt_team = alt.get("team_id")
                except Exception:
                    continue
                if alt_team is None or int(alt_team) == int(cap_team):
                    continue
                if best_score - captain_score(alt, cfg=c) <= club_tol:
                    return int(alt["id"])
    return int(best["id"])
