"""Forward-looking Free Hit timing: prefer the best remaining fixture slate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from fpl_robo.config import CHIP_DEFAULTS


@dataclass
class FhWeekScore:
    gameweek: int
    score: float
    owned_blank: int
    market_ease: float
    dgw_teams: int
    league_blanks: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FhTimingOutlook:
    this_gw: int
    this_score: float
    best_gw: int | None
    best_score: float
    rank: int
    weeks_scored: int
    near_best: bool
    blank_force: bool
    candidates: list[FhWeekScore]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "this_gw": self.this_gw,
            "this_score": round(self.this_score, 3),
            "best_gw": self.best_gw,
            "best_score": round(self.best_score, 3),
            "rank": self.rank,
            "weeks_scored": self.weeks_scored,
            "near_best": self.near_best,
            "blank_force": self.blank_force,
            "candidates": [c.to_dict() for c in self.candidates],
            "reason": self.reason,
        }


def _slot(
    team_fixtures: dict[int, dict[int, dict[str, float]]],
    team_id: int,
    gameweek: int,
) -> dict[str, float]:
    return (team_fixtures.get(int(team_id)) or {}).get(int(gameweek)) or {
        "n_fixtures": 0.0,
        "best_fdr": None,
    }


def score_gw_fh_attractiveness(
    *,
    team_fixtures: dict[int, dict[int, dict[str, float]]],
    owned_team_ids: list[int] | set[int],
    gameweek: int,
    config: dict[str, Any] | None = None,
) -> FhWeekScore:
    """Higher score = better Free Hit week (soft market + owned blanks + DGWs)."""
    cfg = {**CHIP_DEFAULTS, **(config or {})}
    owned = {int(t) for t in owned_team_ids if t is not None}
    teams = list(team_fixtures.keys())
    if not teams:
        return FhWeekScore(
            gameweek=int(gameweek),
            score=0.0,
            owned_blank=0,
            market_ease=0.0,
            dgw_teams=0,
            league_blanks=0,
        )

    owned_blank = 0
    for tid in owned:
        n = int(float(_slot(team_fixtures, tid, gameweek).get("n_fixtures") or 0))
        if n <= 0:
            owned_blank += 1

    eases: list[float] = []
    dgw_teams = 0
    league_blanks = 0
    for tid in teams:
        slot = _slot(team_fixtures, tid, gameweek)
        n = int(float(slot.get("n_fixtures") or 0))
        if n <= 0:
            league_blanks += 1
            continue
        if n >= 2:
            dgw_teams += 1
        fdr = slot.get("best_fdr")
        difficulty = 3.0 if fdr is None else float(fdr)
        # FDR 2 → +1 ease, FDR 3 → 0, FDR 4 → -1; DGW adds a small bonus.
        ease = 3.0 - difficulty
        if n >= 2:
            ease += 0.75
        eases.append(ease)

    market_ease = float(sum(eases) / len(eases)) if eases else 0.0
    score = (
        float(cfg.get("fh_owned_blank_weight") or 3.0) * owned_blank
        + float(cfg.get("fh_market_ease_weight") or 1.0) * market_ease
        + float(cfg.get("fh_dgw_weight") or 0.35) * dgw_teams
        + float(cfg.get("fh_league_blank_weight") or 0.12) * league_blanks
    )
    return FhWeekScore(
        gameweek=int(gameweek),
        score=round(score, 3),
        owned_blank=owned_blank,
        market_ease=round(market_ease, 3),
        dgw_teams=dgw_teams,
        league_blanks=league_blanks,
    )


def build_fh_timing_outlook(
    *,
    gameweek: int,
    team_fixtures: dict[int, dict[int, dict[str, float]]] | None,
    owned_team_ids: list[int] | set[int] | None,
    blank_owned: int = 0,
    window_end: int | None = None,
    config: dict[str, Any] | None = None,
) -> FhTimingOutlook | None:
    """Rank this GW vs upcoming GWs for Free Hit attractiveness.

    Returns None when plan-ahead is off or fixture data is missing (caller keeps
    gap-only behaviour).
    """
    cfg = {**CHIP_DEFAULTS, **(config or {})}
    if not bool(cfg.get("fh_plan_ahead", True)):
        return None
    if not team_fixtures:
        return None

    look = int(cfg.get("fh_lookahead_gws") or 6)
    end = int(window_end) if window_end is not None else int(gameweek) + look
    end = min(end, int(gameweek) + look)
    owned = list(owned_team_ids or [])

    candidates: list[FhWeekScore] = []
    for gw in range(int(gameweek), end + 1):
        # Skip weeks with no fixture coverage at all.
        covered = any(int(gw) in (by_gw or {}) for by_gw in team_fixtures.values())
        if not covered:
            continue
        candidates.append(
            score_gw_fh_attractiveness(
                team_fixtures=team_fixtures,
                owned_team_ids=owned,
                gameweek=gw,
                config=cfg,
            )
        )
    if not candidates:
        return None

    this = next((c for c in candidates if c.gameweek == int(gameweek)), candidates[0])
    ranked = sorted(candidates, key=lambda c: (-c.score, c.gameweek))
    best = ranked[0]
    rank = next(i for i, c in enumerate(ranked, start=1) if c.gameweek == this.gameweek)
    margin = float(cfg.get("fh_near_best_margin") or 0.5)
    top_rank = max(1, int(cfg.get("fh_top_rank") or 1))
    others = [c for c in candidates if c.gameweek != this.gameweek]
    if others:
        mean_others = float(sum(c.score for c in others) / len(others))
        schedule_edge = this.score - mean_others
    else:
        mean_others = this.score
        schedule_edge = 0.0
    min_edge = float(cfg.get("fh_min_schedule_edge") or 0.75)
    has_edge = schedule_edge >= min_edge
    near_best = (
        this.score >= best.score - margin
        and rank <= top_rank
        and has_edge
    )
    blank_force_n = int(
        cfg.get("fh_blank_force_owned")
        or cfg.get("fh_blank_squad_threshold")
        or 3
    )
    blank_force = int(blank_owned) >= blank_force_n or this.owned_blank >= blank_force_n

    if blank_force:
        reason = (
            f"blank_force owned_blank={max(int(blank_owned), this.owned_blank)} "
            f"(need>={blank_force_n})"
        )
    elif near_best:
        reason = (
            f"best remaining slate this_gw={this.gameweek} score={this.score:.2f} "
            f"(best={best.gameweek}:{best.score:.2f}, edge={schedule_edge:+.2f}, "
            f"rank={rank}/{len(candidates)})"
        )
    elif not has_edge:
        reason = (
            f"no schedule edge this={this.gameweek}:{this.score:.2f} "
            f"vs look-ahead mean={mean_others:.2f} "
            f"(edge={schedule_edge:+.2f}, need>={min_edge:g}); "
            f"best={best.gameweek}:{best.score:.2f}"
        )
    else:
        reason = (
            f"save for better slate: this={this.gameweek}:{this.score:.2f} "
            f"rank={rank}/{len(candidates)}; best={best.gameweek}:{best.score:.2f}"
        )

    return FhTimingOutlook(
        this_gw=int(gameweek),
        this_score=this.score,
        best_gw=best.gameweek,
        best_score=best.score,
        rank=rank,
        weeks_scored=len(candidates),
        near_best=near_best,
        blank_force=blank_force,
        candidates=candidates,
        reason=reason,
    )


def owned_team_ids_from_squad(
    starting_xi: list[dict[str, Any]],
    bench: list[dict[str, Any]] | None = None,
) -> list[int]:
    """Collect club ids from XI/bench rows."""
    out: list[int] = []
    seen: set[int] = set()
    for p in list(starting_xi) + list(bench or []):
        for key in ("team_id", "team"):
            val = p.get(key)
            if val is None:
                continue
            try:
                tid = int(val)
            except (TypeError, ValueError):
                continue
            if tid not in seen:
                seen.add(tid)
                out.append(tid)
            break
    return out
