"""Forward-looking Wildcard timing: multi-week rebuild value vs current squad."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from fpl_robo.config import CHIP_DEFAULTS
from fpl_robo.predict.horizon import fixture_scale


@dataclass
class WcWeekScore:
    gameweek: int
    gap: float
    rebuild_run: float
    current_run: float
    horizon: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WcTimingOutlook:
    this_gw: int
    this_gap: float
    best_gw: int | None
    best_gap: float
    play_now: bool
    horizon: int
    candidates: list[WcWeekScore]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "this_gw": self.this_gw,
            "this_gap": round(self.this_gap, 3),
            "best_gw": self.best_gw,
            "best_gap": round(self.best_gap, 3),
            "play_now": self.play_now,
            "horizon": self.horizon,
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


def _xi_rows(squad: list[dict[str, Any]]) -> list[dict[str, Any]]:
    starters = [r for r in squad if r.get("start")]
    if len(starters) >= 11:
        return starters[:11]
    # Already an XI list (no start flags).
    if squad and all(r.get("start") is None for r in squad):
        return list(squad)[:11]
    return starters or list(squad)[:11]


def squad_horizon_run_ev(
    squad: list[dict[str, Any]],
    *,
    start_gw: int,
    horizon: int,
    team_fixtures: dict[int, dict[int, dict[str, float]]],
    config: dict[str, Any] | None = None,
) -> float:
    """Weighted multi-GW EV for a squad starting at `start_gw`.

    Reweights each player's current EV by that week's fixture scale so a
    Newcastle-style easy run compounds across the horizon — the point of a WC.
    """
    cfg = {**CHIP_DEFAULTS, **(config or {})}
    fdr_w = float(cfg.get("wc_fdr_weight") or cfg.get("tc_fdr_weight") or 0.25)
    power = float(cfg.get("wc_fixture_power") or 1.25)
    decay = float(cfg.get("wc_week_decay") or 0.85)
    rows = _xi_rows(squad)
    total = 0.0
    for offset in range(max(1, int(horizon))):
        gw = int(start_gw) + offset
        week_w = decay ** offset
        week_sum = 0.0
        for p in rows:
            tid = _team_id(p)
            slot = _slot(team_fixtures, tid, gw)
            n = int(float(slot.get("n_fixtures") or 0))
            fdr = slot.get("best_fdr")
            scale = fixture_scale(
                n_fixtures=n,
                fdr=float(fdr) if fdr is not None else None,
                fdr_weight=fdr_w,
            )
            week_sum += _base_ev(p) * (float(scale) ** power)
        total += week_w * week_sum
    return float(total)


def build_wc_timing_outlook(
    *,
    gameweek: int,
    current_xi: list[dict[str, Any]],
    rebuild_xi: list[dict[str, Any]] | None,
    team_fixtures: dict[int, dict[int, dict[str, float]]] | None,
    window_end: int | None = None,
    config: dict[str, Any] | None = None,
) -> WcTimingOutlook | None:
    """Pick the GW where wildcards capture the best multi-week rebuild edge.

    Compares fixture-weighted horizon EV of the unconstrained rebuild XI vs the
    locked XI for each candidate start week (captures runs like Newcastle's).
    """
    cfg = {**CHIP_DEFAULTS, **(config or {})}
    if not bool(cfg.get("wc_plan_ahead", True)):
        return None
    if not team_fixtures or not rebuild_xi or not current_xi:
        return None

    look = int(cfg.get("wc_lookahead_gws") or 6)
    horizon = int(cfg.get("wc_run_horizon") or 4)
    end = int(window_end) if window_end is not None else int(gameweek) + look
    end = min(end, int(gameweek) + look)
    margin = float(cfg.get("wc_near_best_margin") or 2.0)

    candidates: list[WcWeekScore] = []
    for gw in range(int(gameweek), end + 1):
        covered = any(int(gw) in (by_gw or {}) for by_gw in team_fixtures.values())
        if not covered:
            continue
        # Need enough forward fixtures for a meaningful run score.
        forward_ok = any(
            int(gw) + horizon - 1 in (by_gw or {}) for by_gw in team_fixtures.values()
        )
        if not forward_ok:
            continue
        cur = squad_horizon_run_ev(
            current_xi,
            start_gw=gw,
            horizon=horizon,
            team_fixtures=team_fixtures,
            config=cfg,
        )
        reb = squad_horizon_run_ev(
            rebuild_xi,
            start_gw=gw,
            horizon=horizon,
            team_fixtures=team_fixtures,
            config=cfg,
        )
        candidates.append(
            WcWeekScore(
                gameweek=int(gw),
                gap=round(reb - cur, 3),
                rebuild_run=round(reb, 3),
                current_run=round(cur, 3),
                horizon=horizon,
            )
        )
    if not candidates:
        return None

    ranked = sorted(candidates, key=lambda c: (-c.gap, c.gameweek))
    best = ranked[0]
    this = next((c for c in candidates if c.gameweek == int(gameweek)), None)
    if this is None:
        return WcTimingOutlook(
            this_gw=int(gameweek),
            this_gap=0.0,
            best_gw=best.gameweek,
            best_gap=best.gap,
            play_now=False,
            horizon=horizon,
            candidates=candidates,
            reason=(
                f"no WC fixture coverage this GW; best={best.gameweek} "
                f"horizon_gap={best.gap:.2f}"
            ),
        )

    best_gap = best.gap
    best_weeks = sorted(c.gameweek for c in ranked if c.gap >= best_gap - margin)
    play_now = bool(best_weeks and best_weeks[0] == int(gameweek))

    if play_now:
        reason = (
            f"best remaining WC run this_gw={this.gameweek} "
            f"horizon_gap={this.gap:.2f} over {horizon}gw "
            f"(rebuild={this.rebuild_run:.1f} vs current={this.current_run:.1f}); "
            f"best={best.gameweek}:{best.gap:.2f}"
        )
    else:
        reason = (
            f"save WC for better multi-week run: this={this.gameweek}:"
            f"{this.gap:.2f} over {horizon}gw; best={best.gameweek}:{best.gap:.2f} "
            f"(rebuild={best.rebuild_run:.1f} vs current={best.current_run:.1f})"
        )

    return WcTimingOutlook(
        this_gw=int(gameweek),
        this_gap=this.gap,
        best_gw=best.gameweek,
        best_gap=best.gap,
        play_now=play_now,
        horizon=horizon,
        candidates=candidates,
        reason=reason,
    )
