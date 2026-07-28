"""Suppress TC/BB when predicted lineups contradict the chip thesis."""

from __future__ import annotations

from typing import Any

from fpl_robo.config import SIGNAL_INGEST
from fpl_robo.signals.schema import PredictedLineupSignal


_DOUBT_ROLES = frozenset({"doubt", "doubtful", "bench", "sub", "substitute", "out"})
_START_ROLES = frozenset({"start", "starter", "xi", "likely_start"})


def _role_for_player(
    *,
    player_id: int | None,
    web_name: str | None,
    gameweek: int,
    lineups: list[PredictedLineupSignal],
) -> tuple[str | None, float]:
    """Best matching predicted role for this GW (highest confidence wins)."""
    best_role: str | None = None
    best_conf = -1.0
    want_name = (web_name or "").strip().lower()
    for sig in lineups:
        if sig.gameweek is not None and int(sig.gameweek) != int(gameweek):
            continue
        match = False
        if player_id is not None and sig.player_id is not None and int(sig.player_id) == int(player_id):
            match = True
        elif want_name and (sig.web_name or "").strip().lower() == want_name:
            match = True
        if not match:
            continue
        conf = float(sig.confidence or 0.0)
        if conf >= best_conf:
            best_conf = conf
            best_role = (sig.role or "").strip().lower() or None
    return best_role, best_conf


def apply_lineup_chip_gates(
    chips: dict[str, Any],
    *,
    gameweek: int,
    captain_id: int | None,
    starting_xi: list[dict[str, Any]],
    bench: list[dict[str, Any]],
    predicted_lineups: list[PredictedLineupSignal],
    min_confidence: float | None = None,
    bb_min_startish: int | None = None,
) -> dict[str, Any]:
    """Mutate play_now / reasons when lineup signals argue against TC or BB.

    Returns a small details dict for logging / dashboard.
    """
    conf_floor = float(
        SIGNAL_INGEST.get("lineup_min_confidence")
        if min_confidence is None
        else min_confidence
    )
    need_startish = int(2 if bb_min_startish is None else bb_min_startish)
    play = dict(chips.get("play_now") or {})
    reasons = dict(chips.get("reasons") or {})
    details: dict[str, Any] = {
        "captain_role": None,
        "captain_confidence": None,
        "bench_startish": 0,
        "suppressed": [],
    }
    if not predicted_lineups:
        chips["lineup_gate"] = details
        return details

    captain = None
    if captain_id is not None:
        for p in starting_xi:
            try:
                if int(p.get("id")) == int(captain_id):
                    captain = p
                    break
            except (TypeError, ValueError):
                continue

    cap_role, cap_conf = _role_for_player(
        player_id=int(captain_id) if captain_id is not None else None,
        web_name=(captain or {}).get("web_name"),
        gameweek=gameweek,
        lineups=predicted_lineups,
    )
    details["captain_role"] = cap_role
    details["captain_confidence"] = round(cap_conf, 3) if cap_conf >= 0 else None

    if (
        play.get("triple_captain")
        and cap_role in _DOUBT_ROLES
        and cap_conf >= conf_floor
    ):
        play["triple_captain"] = False
        reasons["triple_captain"] = (
            str(reasons.get("triple_captain") or "")
            + f" | suppressed (lineup:{cap_role} conf={cap_conf:.2f})"
        ).strip(" |")
        details["suppressed"].append(f"tc_lineup_{cap_role}")

    startish = 0
    for p in bench:
        role, rconf = _role_for_player(
            player_id=int(p["id"]) if p.get("id") is not None else None,
            web_name=p.get("web_name"),
            gameweek=gameweek,
            lineups=predicted_lineups,
        )
        if role in _START_ROLES and rconf >= conf_floor:
            startish += 1
        # Predicted to start from the bench is good for BB; predicted bench/doubt is not.
    details["bench_startish"] = startish

    if play.get("bench_boost") and predicted_lineups and startish < need_startish:
        # Only suppress when we have *some* lineup coverage for this GW, else stay quiet.
        covered = 0
        for p in bench:
            role, _ = _role_for_player(
                player_id=int(p["id"]) if p.get("id") is not None else None,
                web_name=p.get("web_name"),
                gameweek=gameweek,
                lineups=predicted_lineups,
            )
            if role:
                covered += 1
        if covered >= 2:
            play["bench_boost"] = False
            reasons["bench_boost"] = (
                str(reasons.get("bench_boost") or "")
                + f" | suppressed (lineup: only {startish}/{need_startish} bench look like starters)"
            ).strip(" |")
            details["suppressed"].append(f"bb_lineup_startish_{startish}")

    chips["play_now"] = play
    chips["reasons"] = reasons
    chips["lineup_gate"] = details
    return details
