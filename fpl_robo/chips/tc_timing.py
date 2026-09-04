"""Forward-looking Triple Captain timing: best owned player × upcoming week."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from fpl_robo.config import CHIP_DEFAULTS
from fpl_robo.predict.horizon import fixture_scale


@dataclass
class TcCandidate:
    gameweek: int
    player_id: int
    web_name: str
    team_id: int | None
    score: float
    projected_ev: float
    base_ev: float
    fixture_scale: float
    fdr: float | None
    n_fixtures: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TcTimingOutlook:
    this_gw: int
    this_best: TcCandidate | None
    best_overall: TcCandidate | None
    play_now: bool
    rank: int
    weeks_scored: int
    candidates: list[TcCandidate]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "this_gw": self.this_gw,
            "this_best": None if self.this_best is None else self.this_best.to_dict(),
            "best_overall": None
            if self.best_overall is None
            else self.best_overall.to_dict(),
            "play_now": self.play_now,
            "rank": self.rank,
            "weeks_scored": self.weeks_scored,
            "candidates": [c.to_dict() for c in self.candidates],
            "reason": self.reason,
            "this_player_id": None
            if self.this_best is None
            else int(self.this_best.player_id),
            "best_player_id": None
            if self.best_overall is None
            else int(self.best_overall.player_id),
            "best_gw": None if self.best_overall is None else int(self.best_overall.gameweek),
        }


def _team_id(player: dict[str, Any]) -> int | None:
    for key in ("team_id", "team"):
        val = player.get(key)
        if val is None:
            continue
        try:
            return int(val)
        except (TypeError, ValueError):
            continue
    return None


def _slot(
    team_fixtures: dict[int, dict[int, dict[str, float]]],
    team_id: int | None,
    gameweek: int,
) -> dict[str, float]:
    if team_id is None:
        return {"n_fixtures": 1.0, "best_fdr": None}
    return (team_fixtures.get(int(team_id)) or {}).get(int(gameweek)) or {
        "n_fixtures": 0.0,
        "best_fdr": None,
    }


def _base_ev(player: dict[str, Any]) -> float:
    for key in ("ev", "ep_anchor", "ep_next", "xP"):
        val = player.get(key)
        if val is None:
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return 0.0


def score_player_tc_week(
    player: dict[str, Any],
    *,
    gameweek: int,
    this_gw: int,
    team_fixtures: dict[int, dict[int, dict[str, float]]],
    config: dict[str, Any] | None = None,
) -> TcCandidate:
    """Project TC extra value for an owned player in a given gameweek.

    Uses current EV as player strength (form already in) and reweights by the
    target week's FDR / blank / DGW scale. Fixture weight is exponentiated so
    easy matchups / DGWs outrank merely high early-season form on a middling FDR.
    """
    cfg = {**CHIP_DEFAULTS, **(config or {})}
    fdr_w = float(cfg.get("tc_fdr_weight") or 0.25)
    power = float(cfg.get("tc_fixture_power") or 1.5)
    tid = _team_id(player)
    base = _base_ev(player)

    tgt = _slot(team_fixtures, tid, gameweek)
    n = int(float(tgt.get("n_fixtures") or 0))
    fdr = tgt.get("best_fdr")
    scale = fixture_scale(
        n_fixtures=n,
        fdr=float(fdr) if fdr is not None else None,
        fdr_weight=fdr_w,
    )
    projected = base * (float(scale) ** power)
    score = 2.0 * projected  # TC extra vs normal captain
    return TcCandidate(
        gameweek=int(gameweek),
        player_id=int(player["id"]),
        web_name=str(player.get("web_name") or player["id"]),
        team_id=tid,
        score=round(score, 3),
        projected_ev=round(projected, 3),
        base_ev=round(base, 3),
        fixture_scale=round(float(scale), 3),
        fdr=float(fdr) if fdr is not None else None,
        n_fixtures=n,
    )


def _tc_pool(
    starting_xi: list[dict[str, Any]],
    bench: list[dict[str, Any]] | None,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Owned players worth considering for TC (starters first, then high-EV bench)."""
    rows = list(starting_xi) + list(bench or [])
    # Skip GKs — TC on a keeper is almost never the plan.
    outfield = [
        r
        for r in rows
        if str(r.get("position") or "").upper() not in {"GK", "GKP"}
    ]
    if not outfield:
        outfield = list(rows)
    outfield.sort(key=_base_ev, reverse=True)
    seen: set[int] = set()
    pool: list[dict[str, Any]] = []
    for r in outfield:
        try:
            pid = int(r["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if pid in seen:
            continue
        seen.add(pid)
        pool.append(r)
        if len(pool) >= limit:
            break
    return pool


def build_tc_timing_outlook(
    *,
    gameweek: int,
    starting_xi: list[dict[str, Any]],
    bench: list[dict[str, Any]] | None = None,
    team_fixtures: dict[int, dict[int, dict[str, float]]] | None,
    window_end: int | None = None,
    config: dict[str, Any] | None = None,
) -> TcTimingOutlook | None:
    """Rank (player, week) TC opportunities and say whether to play this GW.

    Returns None when plan-ahead is off or fixture data is missing (caller may
    fall back to legacy absolute/history gates).
    """
    cfg = {**CHIP_DEFAULTS, **(config or {})}
    if not bool(cfg.get("tc_plan_ahead", True)):
        return None
    if not team_fixtures:
        return None

    look = int(cfg.get("tc_lookahead_gws") or 8)
    end = int(window_end) if window_end is not None else int(gameweek) + look
    end = min(end, int(gameweek) + look)
    pool_n = int(cfg.get("tc_plan_pool") or 8)
    pool = _tc_pool(starting_xi, bench, limit=pool_n)
    if not pool:
        return None

    candidates: list[TcCandidate] = []
    for gw in range(int(gameweek), end + 1):
        covered = any(int(gw) in (by_gw or {}) for by_gw in team_fixtures.values())
        if not covered:
            continue
        for player in pool:
            cand = score_player_tc_week(
                player,
                gameweek=gw,
                this_gw=int(gameweek),
                team_fixtures=team_fixtures,
                config=cfg,
            )
            # Blanks are never TC weeks for that player.
            if cand.n_fixtures <= 0:
                continue
            candidates.append(cand)
    if not candidates:
        return None

    ranked = sorted(candidates, key=lambda c: (-c.score, c.gameweek, -c.base_ev))
    best = ranked[0]
    this_week = [c for c in ranked if c.gameweek == int(gameweek)]
    this_best = this_week[0] if this_week else None
    if this_best is None:
        return TcTimingOutlook(
            this_gw=int(gameweek),
            this_best=None,
            best_overall=best,
            play_now=False,
            rank=len(ranked) + 1,
            weeks_scored=len({c.gameweek for c in candidates}),
            candidates=ranked[:24],
            reason=(
                f"no TC fixture this GW; best={best.web_name}@{best.gameweek} "
                f"score={best.score:.2f}"
            ),
        )

    margin = float(cfg.get("tc_near_best_margin") or 1.0)
    # Play only when this GW is the earliest remaining week that reaches
    # within `margin` of the best (player, week) TC score in the look-ahead.
    best_score = best.score
    best_weeks = sorted(
        {c.gameweek for c in ranked if c.score >= best_score - margin}
    )
    play_now = bool(best_weeks and best_weeks[0] == int(gameweek))

    rank = next(
        i
        for i, c in enumerate(ranked, start=1)
        if c.gameweek == this_best.gameweek and c.player_id == this_best.player_id
    )
    weeks_scored = len({c.gameweek for c in candidates})

    if play_now:
        reason = (
            f"best remaining TC slot this_gw={this_best.gameweek} "
            f"{this_best.web_name} score={this_best.score:.2f} "
            f"(proj_ev={this_best.projected_ev:.2f}, fdr={this_best.fdr}, "
            f"n={this_best.n_fixtures}); "
            f"best={best.web_name}@{best.gameweek}:{best.score:.2f}"
        )
    else:
        reason = (
            f"save TC for better slot: this={this_best.web_name}@"
            f"{this_best.gameweek}:{this_best.score:.2f} "
            f"(fdr={this_best.fdr}); best={best.web_name}@"
            f"{best.gameweek}:{best.score:.2f} "
            f"(fdr={best.fdr}, proj_ev={best.projected_ev:.2f})"
        )

    return TcTimingOutlook(
        this_gw=int(gameweek),
        this_best=this_best,
        best_overall=best,
        play_now=play_now,
        rank=rank,
        weeks_scored=weeks_scored,
        candidates=ranked[:24],
        reason=reason,
    )
