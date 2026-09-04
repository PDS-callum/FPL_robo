"""Forward-looking Bench Boost timing: best owned-bench week."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from fpl_robo.config import CHIP_DEFAULTS
from fpl_robo.predict.horizon import fixture_scale


@dataclass
class BbWeekScore:
    gameweek: int
    score: float
    projected_bench_ev: float
    n_playing: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BbTimingOutlook:
    this_gw: int
    this_score: float
    best_gw: int | None
    best_score: float
    play_now: bool
    candidates: list[BbWeekScore]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "this_gw": self.this_gw,
            "this_score": round(self.this_score, 3),
            "best_gw": self.best_gw,
            "best_score": round(self.best_score, 3),
            "play_now": self.play_now,
            "candidates": [c.to_dict() for c in self.candidates],
            "reason": self.reason,
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


def score_bench_week(
    bench: list[dict[str, Any]],
    *,
    gameweek: int,
    team_fixtures: dict[int, dict[int, dict[str, float]]],
    config: dict[str, Any] | None = None,
) -> BbWeekScore:
    cfg = {**CHIP_DEFAULTS, **(config or {})}
    fdr_w = float(cfg.get("bb_fdr_weight") or cfg.get("tc_fdr_weight") or 0.25)
    power = float(cfg.get("bb_fixture_power") or 1.5)
    projected = 0.0
    playing = 0
    for p in bench:
        tid = _team_id(p)
        slot = _slot(team_fixtures, tid, gameweek)
        n = int(float(slot.get("n_fixtures") or 0))
        if n <= 0:
            continue
        playing += 1
        fdr = slot.get("best_fdr")
        scale = fixture_scale(
            n_fixtures=n,
            fdr=float(fdr) if fdr is not None else None,
            fdr_weight=fdr_w,
        )
        projected += _base_ev(p) * (float(scale) ** power)
    return BbWeekScore(
        gameweek=int(gameweek),
        score=round(projected, 3),
        projected_bench_ev=round(projected, 3),
        n_playing=playing,
    )


def build_bb_timing_outlook(
    *,
    gameweek: int,
    bench: list[dict[str, Any]],
    team_fixtures: dict[int, dict[int, dict[str, float]]] | None,
    window_end: int | None = None,
    config: dict[str, Any] | None = None,
) -> BbTimingOutlook | None:
    """Play Bench Boost only on the best remaining owned-bench fixture week."""
    cfg = {**CHIP_DEFAULTS, **(config or {})}
    if not bool(cfg.get("bb_plan_ahead", True)):
        return None
    if not team_fixtures or not bench:
        return None

    look = int(cfg.get("bb_lookahead_gws") or 8)
    end = int(window_end) if window_end is not None else int(gameweek) + look
    end = min(end, int(gameweek) + look)
    margin = float(cfg.get("bb_near_best_margin") or 1.0)

    candidates: list[BbWeekScore] = []
    for gw in range(int(gameweek), end + 1):
        covered = any(int(gw) in (by_gw or {}) for by_gw in team_fixtures.values())
        if not covered:
            continue
        candidates.append(
            score_bench_week(
                bench,
                gameweek=gw,
                team_fixtures=team_fixtures,
                config=cfg,
            )
        )
    if not candidates:
        return None

    ranked = sorted(candidates, key=lambda c: (-c.score, c.gameweek))
    best = ranked[0]
    this = next((c for c in candidates if c.gameweek == int(gameweek)), None)
    if this is None:
        return BbTimingOutlook(
            this_gw=int(gameweek),
            this_score=0.0,
            best_gw=best.gameweek,
            best_score=best.score,
            play_now=False,
            candidates=candidates,
            reason=f"no BB fixtures this GW; best={best.gameweek}:{best.score:.2f}",
        )

    best_weeks = sorted(c.gameweek for c in ranked if c.score >= best.score - margin)
    play_now = bool(best_weeks and best_weeks[0] == int(gameweek))
    if play_now:
        reason = (
            f"best remaining BB week this_gw={this.gameweek} "
            f"bench_ev={this.score:.2f} (n_playing={this.n_playing}); "
            f"best={best.gameweek}:{best.score:.2f}"
        )
    else:
        reason = (
            f"save BB for better bench slate: this={this.gameweek}:{this.score:.2f}; "
            f"best={best.gameweek}:{best.score:.2f}"
        )
    return BbTimingOutlook(
        this_gw=int(gameweek),
        this_score=this.score,
        best_gw=best.gameweek,
        best_score=best.score,
        play_now=play_now,
        candidates=candidates,
        reason=reason,
    )
