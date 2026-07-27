"""Baseline tabular expected-points models."""

from __future__ import annotations

from typing import Any

import pandas as pd

from fpl_robo.config import BENCH_COVER
from fpl_robo.graph.schema import EdgeType, NodeType
from fpl_robo.graph.store import KnowledgeGraph


class BaselineEVModel:
    """Simple EV from FPL EP / form / PPG, adjusted by next-fixture difficulty."""

    def __init__(
        self,
        difficulty_weight: float = 0.08,
        min_cover_ep: float | None = None,
        min_cover_minutes: int | None = None,
        min_cover_chance: int | None = None,
    ) -> None:
        self.difficulty_weight = difficulty_weight
        self.min_cover_ep = (
            BENCH_COVER["min_ep_next"] if min_cover_ep is None else min_cover_ep
        )
        self.min_cover_minutes = (
            BENCH_COVER["min_minutes"] if min_cover_minutes is None else min_cover_minutes
        )
        self.min_cover_chance = (
            BENCH_COVER["min_chance"] if min_cover_chance is None else min_cover_chance
        )

    def predict(self, kg: KnowledgeGraph, gameweek: int | None = None) -> pd.DataFrame:
        gw = gameweek or kg.current_gameweek()
        fixture_diff = self._fixture_difficulty_by_team(kg, gw)

        rows: list[dict[str, Any]] = []
        for player in kg.players():
            base = self._base_ev(player)
            team_id = int(player["team_id"])
            diff = fixture_diff.get(team_id)
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
                    "cover_eligible": self._cover_eligible(player),
                    "gameweek": gw,
                }
            )

        df = pd.DataFrame(rows)
        return df.sort_values("ev", ascending=False).reset_index(drop=True)

    def _cover_eligible(self, player: dict[str, Any]) -> bool:
        """True if the player is plausible auto-sub / minutes cover."""
        status = (player.get("status") or "a").lower()
        if status in {"u", "s", "i"}:
            return False
        chance = player.get("chance_of_playing_next_round")
        if chance is not None and float(chance) < float(self.min_cover_chance):
            return False
        ep_next = player.get("ep_next")
        minutes = int(player.get("minutes") or 0)
        if ep_next is not None and float(ep_next) >= float(self.min_cover_ep):
            return True
        if minutes >= int(self.min_cover_minutes):
            return True
        return False

    def _base_ev(self, player: dict[str, Any]) -> float:
        ep_next = player.get("ep_next")
        form = player.get("form")
        ppg = player.get("points_per_game")

        candidates = [c for c in (ep_next, form, ppg) if c is not None]
        if not candidates:
            return 0.0
        if ep_next is not None:
            return float(ep_next)
        if form is not None and ppg is not None:
            return 0.6 * float(form) + 0.4 * float(ppg)
        return float(candidates[0])

    def _minutes_factor(self, player: dict[str, Any]) -> float:
        minutes = int(player.get("minutes") or 0)
        if minutes <= 0:
            return 0.55
        if minutes < 90:
            return 0.7
        if minutes < 270:
            return 0.85
        return 1.0

    def _availability_factor(self, player: dict[str, Any]) -> float:
        status = (player.get("status") or "a").lower()
        if status in {"u", "s"}:
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
        for u, _, _, data in kg.g.edges(keys=True, data=True):
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
            prev = out.get(team_id)
            if prev is None or int(difficulty) < prev:
                out[team_id] = int(difficulty)
        return out


class PreseasonEVModel(BaselineEVModel):
    """Cold-start EV model for preseason initial squad construction."""

    def __init__(
        self,
        gameweek_weights: list[float],
        fixture_weight: float = 0.12,
        ownership_weight: float = 0.02,
        uncertainty_penalty: float = 0.18,
        signal_factors: dict[int, Any] | None = None,
    ) -> None:
        super().__init__(difficulty_weight=fixture_weight)
        self.gameweek_weights = gameweek_weights
        self.ownership_weight = ownership_weight
        self.uncertainty_penalty = uncertainty_penalty
        self.signal_factors = signal_factors or {}

    def predict(self, kg: KnowledgeGraph, gameweek: int | None = None) -> pd.DataFrame:
        gw = gameweek or kg.current_gameweek() or 1
        team_fdrs = self._team_fdr_horizon(kg, start_gw=gw, horizon=len(self.gameweek_weights))

        rows: list[dict[str, Any]] = []
        for player in kg.players():
            pid = int(player["id"])
            team_id = int(player["team_id"])
            factors = self.signal_factors.get(pid)
            base = self._preseason_base_ev(player)
            if factors is not None:
                base = max(0.0, base + float(getattr(factors, "form_boost", 0.0) or 0.0))

            minutes_factor = self._minutes_factor(player)
            if factors is not None:
                minutes_factor *= float(getattr(factors, "minutes_mult", 1.0) or 1.0)

            availability = self._availability_factor(player)
            if factors is not None and getattr(factors, "exclude", False):
                availability = 0.0

            ownership = self._ownership_factor(player)
            uncertainty = self._uncertainty_factor(player)
            news_mult = float(getattr(factors, "news_mult", 1.0) or 1.0) if factors else 1.0
            fatigue_mult = (
                float(getattr(factors, "fatigue_mult", 1.0) or 1.0) if factors else 1.0
            )

            weighted_fixture_ev = 0.0
            for idx, weight in enumerate(self.gameweek_weights):
                fdr = team_fdrs.get(team_id, {}).get(gw + idx)
                fixture_adj = 1.0 if fdr is None else 1.0 + self.difficulty_weight * (3.0 - fdr)
                # Fatigue bites hardest in early GWs of the horizon.
                early_fatigue = fatigue_mult if idx < 3 else min(1.0, fatigue_mult + 0.08)
                weighted_fixture_ev += weight * base * fixture_adj * early_fatigue

            ev = max(
                0.0,
                weighted_fixture_ev
                * minutes_factor
                * availability
                * ownership
                * uncertainty
                * news_mult,
            )

            status = player.get("status")
            if factors is not None and getattr(factors, "exclude", False):
                status = "u"

            rows.append(
                {
                    "id": pid,
                    "web_name": player.get("web_name"),
                    "team_id": team_id,
                    "team_short": player.get("team_short"),
                    "position": player.get("position"),
                    "price": float(player.get("price") or 0.0),
                    "status": status,
                    "chance_of_playing_next_round": player.get(
                        "chance_of_playing_next_round"
                    ),
                    "preseason_base_ev": round(base, 3),
                    "minutes_factor": round(minutes_factor, 3),
                    "availability_factor": round(availability, 3),
                    "ownership_factor": round(ownership, 3),
                    "uncertainty_factor": round(uncertainty, 3),
                    "signal_news_mult": round(news_mult, 3),
                    "signal_fatigue_mult": round(fatigue_mult, 3),
                    "signal_applied": factors is not None,
                    "cover_eligible": self._cover_eligible(player)
                    and not (factors is not None and getattr(factors, "exclude", False)),
                    "ev": round(ev, 3),
                    "gameweek": gw,
                }
            )

        df = pd.DataFrame(rows)
        return df.sort_values("ev", ascending=False).reset_index(drop=True)

    def _preseason_base_ev(self, player: dict[str, Any]) -> float:
        ep_next = player.get("ep_next")
        ppg = player.get("points_per_game")
        ict = player.get("ict_index")

        if ep_next is not None:
            base = float(ep_next)
        elif ppg is not None:
            base = float(ppg)
        else:
            base = 2.0

        if ict is not None:
            base += min(0.8, max(0.0, float(ict) / 100.0))
        return base

    def _ownership_factor(self, player: dict[str, Any]) -> float:
        selected = player.get("selected_by_percent")
        if selected is None:
            return 1.0
        # Slight preference for popular preseason picks as a stability prior.
        return 1.0 + min(0.08, self.ownership_weight * float(selected))

    def _uncertainty_factor(self, player: dict[str, Any]) -> float:
        minutes = int(player.get("minutes") or 0)
        # Players with little/no prior minutes get a conservative haircut in preseason.
        if minutes <= 0:
            return max(0.6, 1.0 - self.uncertainty_penalty)
        if minutes < 900:
            return max(0.75, 1.0 - self.uncertainty_penalty / 2.0)
        return 1.0

    def _team_fdr_horizon(
        self, kg: KnowledgeGraph, start_gw: int, horizon: int
    ) -> dict[int, dict[int, int]]:
        out: dict[int, dict[int, int]] = {}
        end_gw = start_gw + horizon - 1
        for u, _, _, data in kg.g.edges(keys=True, data=True):
            if data.get("edge_type") != EdgeType.FACES.value:
                continue
            if u[0] != NodeType.TEAM.value:
                continue
            gw = data.get("gameweek")
            diff = data.get("difficulty")
            if gw is None or diff is None:
                continue
            gw = int(gw)
            if gw < start_gw or gw > end_gw:
                continue
            team_id = int(u[1])
            team_map = out.setdefault(team_id, {})
            current = team_map.get(gw)
            if current is None or int(diff) < current:
                team_map[gw] = int(diff)
        return out
