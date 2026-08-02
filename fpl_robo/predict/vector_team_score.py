"""Non-linear team score over player vectors (XI + geometry terms)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from fpl_robo.config import VECTOR_TEAM
from fpl_robo.graph.schema import STARTING_BOUNDS, STARTING_SIZE


@dataclass
class TeamScoreBreakdown:
    total: float
    xi_points: float
    captain_extra: float
    bench_points: float
    portfolio: float
    attack_coverage: float
    minutes_reliability: float
    club_soft: float
    value: float
    starting_ids: list[int]
    bench_ids: list[int]
    captain_id: int | None


@dataclass(frozen=True)
class PlayerSnap:
    pid: int
    position: str
    team_id: int
    price: float
    base: float
    play_prob: float
    run_ease: float
    threat: float


class FastTeamScorer:
    """Precomputed float snapshots so SA/hill-climb avoid pandas in the hot loop."""

    def __init__(self, preds: pd.DataFrame):
        snaps: dict[int, PlayerSnap] = {}
        for _, r in preds.iterrows():
            pid = int(r["id"])
            pos = str(r.get("position") or "MID").upper()
            xg = float(r.get("v_rolling_xgi") or 0.0)
            if not xg:
                xg = float(r.get("rolling_expected_goals") or 0.0) + float(
                    r.get("rolling_expected_assists") or 0.0
                )
            snaps[pid] = PlayerSnap(
                pid=pid,
                position=pos,
                team_id=int(r.get("team_id") or 0),
                price=max(4.0, float(r.get("price") or 4.5)),
                base=float(r.get("base_points") or r.get("ev") or 0.0),
                play_prob=float(r.get("play_prob") or 0.7),
                run_ease=float(r.get("run_ease") or 0.5),
                threat=xg,
            )
        self.snaps = snaps

    def select_xi(self, squad_ids: Iterable[int]) -> tuple[list[int], list[int]]:
        squad = [int(i) for i in squad_ids if int(i) in self.snaps]
        by_pos: dict[str, list[PlayerSnap]] = {"GK": [], "DEF": [], "MID": [], "FWD": []}
        for pid in squad:
            s = self.snaps[pid]
            by_pos.setdefault(s.position, []).append(s)
        for pos in by_pos:
            by_pos[pos].sort(key=lambda p: p.base, reverse=True)

        gk = by_pos.get("GK") or []
        if not gk:
            ordered = sorted((self.snaps[i] for i in squad), key=lambda p: p.base, reverse=True)
            start = [p.pid for p in ordered[:STARTING_SIZE]]
            return start, [i for i in squad if i not in set(start)]

        best_score = -1e18
        best_start: list[int] = []
        defs, mids, fwds = by_pos.get("DEF") or [], by_pos.get("MID") or [], by_pos.get("FWD") or []
        def_lo, def_hi = STARTING_BOUNDS["DEF"]
        mid_lo, mid_hi = STARTING_BOUNDS["MID"]
        fwd_lo, fwd_hi = STARTING_BOUNDS["FWD"]
        for n_def in range(def_lo, def_hi + 1):
            for n_mid in range(mid_lo, mid_hi + 1):
                n_fwd = STARTING_SIZE - 1 - n_def - n_mid
                if n_fwd < fwd_lo or n_fwd > fwd_hi:
                    continue
                if n_def > len(defs) or n_mid > len(mids) or n_fwd > len(fwds):
                    continue
                rows = [gk[0]] + defs[:n_def] + mids[:n_mid] + fwds[:n_fwd]
                score = sum(p.base for p in rows) + max(p.base for p in rows)
                if score > best_score:
                    best_score = score
                    best_start = [p.pid for p in rows]
        if not best_start:
            ordered = sorted((self.snaps[i] for i in squad), key=lambda p: p.base, reverse=True)
            best_start = [p.pid for p in ordered[:STARTING_SIZE]]
        start_set = set(best_start)
        bench = sorted(
            (i for i in squad if i not in start_set),
            key=lambda i: self.snaps[i].base,
            reverse=True,
        )
        return best_start, bench

    def score(self, squad_ids: Iterable[int], *, geometry: bool = True) -> TeamScoreBreakdown:
        squad = [int(i) for i in squad_ids if int(i) in self.snaps]
        starting_ids, bench_ids = self.select_xi(squad)

        def bp(pid: int) -> float:
            return self.snaps[pid].base

        xi_points = sum(bp(i) for i in starting_ids)
        captain_id = max(starting_ids, key=bp) if starting_ids else None
        captain_extra = bp(captain_id) if captain_id is not None else 0.0
        bench_w = float(VECTOR_TEAM["bench_weight"])
        bench_points = sum(
            bench_w * bp(i) for i in bench_ids if self.snaps[i].position != "GK"
        )

        portfolio = attack = minutes = club = value = 0.0
        if geometry and starting_ids:
            cfg = VECTOR_TEAM
            run_eases = [self.snaps[i].run_ease for i in starting_ids]
            clubs = [self.snaps[i].team_id for i in starting_ids]
            portfolio = float(cfg["portfolio_weight"]) * (
                float(np.mean(run_eases))
                + 0.25 * float(np.std(run_eases))
                + 0.05 * len(set(clubs))
            )
            threats = sorted(
                (
                    self.snaps[i].threat
                    for i in starting_ids
                    if self.snaps[i].position in ("MID", "FWD")
                ),
                reverse=True,
            )
            attack = float(cfg["attack_coverage_weight"]) * sum(threats[:3])
            minutes = -float(cfg["minutes_reliability_weight"]) * sum(
                max(0.0, 0.75 - self.snaps[i].play_prob) for i in starting_ids
            )
            c = Counter(self.snaps[i].team_id for i in squad)
            triples = sum(1 for n in c.values() if n >= 3)
            doubles = sum(1 for n in c.values() if n == 2)
            club = -float(cfg["club_soft_weight"]) * (1.5 * triples + 0.25 * doubles)
            vals = [bp(i) / self.snaps[i].price for i in starting_ids]
            value = float(cfg["value_weight"]) * float(np.mean(vals)) if vals else 0.0

        total = xi_points + captain_extra + bench_points + portfolio + attack + minutes + club + value
        return TeamScoreBreakdown(
            total=float(total),
            xi_points=float(xi_points),
            captain_extra=float(captain_extra),
            bench_points=float(bench_points),
            portfolio=float(portfolio),
            attack_coverage=float(attack),
            minutes_reliability=float(minutes),
            club_soft=float(club),
            value=float(value),
            starting_ids=starting_ids,
            bench_ids=bench_ids,
            captain_id=captain_id,
        )


def _by_id(preds: pd.DataFrame) -> dict[int, pd.Series]:
    return {int(r["id"]): r for _, r in preds.iterrows()}


def select_xi_greedy(squad_ids: Iterable[int], by_id: dict[int, pd.Series]) -> tuple[list[int], list[int]]:
    """Pick a valid 11 maximising sum of base_points under formation bounds."""
    if not by_id:
        return [], []
    return FastTeamScorer(pd.DataFrame(list(by_id.values()))).select_xi(squad_ids)


def score_team(
    squad_ids: Iterable[int],
    preds: pd.DataFrame,
    *,
    geometry: bool = True,
) -> TeamScoreBreakdown:
    """Score a 15-man squad: XI base + captain + optional geometry terms."""
    return FastTeamScorer(preds).score(squad_ids, geometry=geometry)


def player_records(
    ids: list[int],
    preds: pd.DataFrame,
) -> list[dict[str, Any]]:
    by_id = _by_id(preds)
    out: list[dict[str, Any]] = []
    for pid in ids:
        r = by_id.get(int(pid))
        if r is None:
            continue
        out.append(
            {
                "id": int(r["id"]),
                "web_name": str(r.get("web_name") or r["id"]),
                "position": str(r.get("position") or ""),
                "team_id": int(r.get("team_id") or 0),
                "price": float(r.get("price") or 4.5),
                "ev": float(r.get("base_points") or r.get("ev") or 0.0),
                "base_points": float(r.get("base_points") or r.get("ev") or 0.0),
                "play_prob": float(r.get("play_prob") or 0.0),
                "run_ease": float(r.get("run_ease") or 0.0),
            }
        )
    return out
