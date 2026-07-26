"""Baseline tabular expected-points model."""

from __future__ import annotations

from typing import Any

import pandas as pd

from fpl_robo.graph.schema import EdgeType, NodeType
from fpl_robo.graph.store import KnowledgeGraph


class BaselineEVModel:
    """Simple EV from FPL EP / form / PPG, adjusted by next-fixture difficulty.

    This is intentionally tabular and transparent — a GNN can replace the
    scoring function later without changing the MIP interface.
    """

    def __init__(self, difficulty_weight: float = 0.08) -> None:
        self.difficulty_weight = difficulty_weight

    def predict(self, kg: KnowledgeGraph, gameweek: int | None = None) -> pd.DataFrame:
        gw = gameweek or kg.current_gameweek()
        fixture_diff = self._fixture_difficulty_by_team(kg, gw)

        rows: list[dict[str, Any]] = []
        for player in kg.players():
            base = self._base_ev(player)
            team_id = int(player["team_id"])
            diff = fixture_diff.get(team_id)
            # FDR is 1 (easy) … 5 (hard). Centre on 3.
            if diff is None:
                adj = 1.0
                next_fdr = None
            else:
                next_fdr = float(diff)
                adj = 1.0 + self.difficulty_weight * (3.0 - next_fdr)

            minutes_factor = self._minutes_factor(player)
            availability = self._availability_factor(player)
            ev = max(0.0, base * adj * minutes_factor * availability)

            rows.append(
                {
                    "id": player["id"],
                    "web_name": player.get("web_name"),
                    "team_id": team_id,
                    "team_short": player.get("team_short"),
                    "position": player.get("position"),
                    "price": float(player.get("price") or 0.0),
                    "status": player.get("status"),
                    "chance_of_playing_next_round": player.get(
                        "chance_of_playing_next_round"
                    ),
                    "base_ev": round(base, 3),
                    "next_fdr": next_fdr,
                    "minutes_factor": round(minutes_factor, 3),
                    "availability_factor": round(availability, 3),
                    "ev": round(ev, 3),
                    "gameweek": gw,
                }
            )

        df = pd.DataFrame(rows)
        return df.sort_values("ev", ascending=False).reset_index(drop=True)

    def _base_ev(self, player: dict[str, Any]) -> float:
        ep_next = player.get("ep_next")
        form = player.get("form")
        ppg = player.get("points_per_game")

        candidates = [c for c in (ep_next, form, ppg) if c is not None]
        if not candidates:
            return 0.0
        # Prefer official EP when present; otherwise blend form/PPG.
        if ep_next is not None:
            return float(ep_next)
        if form is not None and ppg is not None:
            return 0.6 * float(form) + 0.4 * float(ppg)
        return float(candidates[0])

    def _minutes_factor(self, player: dict[str, Any]) -> float:
        minutes = int(player.get("minutes") or 0)
        # Soft prior: unknown/new players aren't zeroed, just down-weighted.
        if minutes <= 0:
            return 0.55
        if minutes < 90:
            return 0.7
        if minutes < 270:
            return 0.85
        return 1.0

    def _availability_factor(self, player: dict[str, Any]) -> float:
        status = (player.get("status") or "a").lower()
        if status in {"u", "s"}:  # unavailable / suspended
            return 0.0
        if status == "i":
            return 0.15
        if status == "d":
            chance = player.get("chance_of_playing_next_round")
            if chance is None:
                return 0.5
            return max(0.0, min(1.0, float(chance) / 100.0))
        chance = player.get("chance_of_playing_next_round")
        if chance is None:
            return 1.0
        return max(0.0, min(1.0, float(chance) / 100.0))

    def _fixture_difficulty_by_team(
        self, kg: KnowledgeGraph, gameweek: int | None
    ) -> dict[int, int]:
        if gameweek is None:
            return {}
        out: dict[int, int] = {}
        for u, v, _, data in kg.g.edges(keys=True, data=True):
            if data.get("edge_type") != EdgeType.FACES.value:
                continue
            if data.get("gameweek") != gameweek:
                continue
            if u[0] != NodeType.TEAM.value:
                continue
            team_id = int(u[1])
            difficulty = data.get("difficulty")
            if difficulty is None:
                continue
            # Double gameweeks: keep the easier fixture for EV optimism;
            # refine later with proper DGW modelling.
            prev = out.get(team_id)
            if prev is None or int(difficulty) < prev:
                out[team_id] = int(difficulty)
        return out
