"""Open EV model: lagged involvement / points / price — no ep_next / xP."""

from __future__ import annotations

from typing import Any

import pandas as pd

from fpl_robo.config import BUY_GATES
from fpl_robo.graph.schema import EdgeType, NodeType
from fpl_robo.graph.store import KnowledgeGraph
from fpl_robo.predict.ev_modes import open_ev_from_player_history


class OpenEVModel:
    """Score players from open features only (no FPL expected points)."""

    def __init__(self, difficulty_weight: float = 0.05) -> None:
        self.difficulty_weight = difficulty_weight
        self.min_buy_ev = float(BUY_GATES["min_buy_ev"])
        self.min_prior_minutes = float(BUY_GATES["min_prior_minutes"])
        self.min_rolling_minutes = float(BUY_GATES["min_rolling_minutes"])

    def predict(self, kg: KnowledgeGraph, gameweek: int | None = None) -> pd.DataFrame:
        gw = gameweek or kg.current_gameweek()
        fixture_diff = self._fixture_difficulty_by_team(kg, gw)
        prior_gw = max(0, int(gw or 1) - 1)

        rows: list[dict[str, Any]] = []
        for player in kg.players():
            team_id = int(player["team_id"])
            price = float(player.get("price") or 0.0)
            if player.get("history_attached"):
                season_minutes = int(float(player.get("prior_minutes") or 0))
                rolling_minutes = float(player.get("rolling_minutes") or 0.0)
                rolling_points = float(
                    player.get("rolling_points")
                    if player.get("rolling_points") is not None
                    else (player.get("form") or 0.0)
                )
            else:
                season_minutes = int(player.get("minutes") or 0)
                form = float(player.get("form") or 0.0)
                # Proxy for last-4 mean minutes: season minutes-per-GW (capped at 90).
                rolling_minutes = (
                    min(90.0, float(season_minutes) / float(prior_gw)) if prior_gw > 0 else 0.0
                )
                rolling_points = form
            rolling_xg = float(player.get("expected_goals") or 0.0)
            rolling_xa = float(player.get("expected_assists") or 0.0)
            rolling_ict = float(player.get("ict_index") or 0.0)

            base = open_ev_from_player_history(
                rolling_points=rolling_points,
                rolling_minutes=rolling_minutes,
                price=price,
                rolling_xg=rolling_xg / max(1, prior_gw) if prior_gw else 0.0,
                rolling_xa=rolling_xa / max(1, prior_gw) if prior_gw else 0.0,
                rolling_ict=rolling_ict,
            )

            diff = fixture_diff.get(team_id)
            if diff is None:
                adj = 1.0
                next_fdr = None
            else:
                next_fdr = float(diff)
                adj = 1.0 + self.difficulty_weight * (3.0 - next_fdr)

            availability = self._availability_factor(player)
            ev = max(0.0, base * adj * availability)
            minutes_ok = self._minutes_gate_ok(season_minutes, rolling_minutes)
            buy_eligible = (ev >= self.min_buy_ev) and minutes_ok
            cover_eligible = (ev >= 1.5) and minutes_ok and self._status_ok(player)

            rows.append(
                {
                    "id": player["id"],
                    "web_name": player.get("web_name"),
                    "team_id": team_id,
                    "team_short": player.get("team_short"),
                    "position": player.get("position"),
                    "price": price,
                    "status": player.get("status"),
                    "chance_of_playing_next_round": player.get("chance_of_playing_next_round"),
                    "base_ev": round(base, 3),
                    "next_fdr": next_fdr,
                    "ev": round(ev, 3),
                    "cover_eligible": cover_eligible,
                    "buy_eligible": buy_eligible,
                    "prior_minutes": float(season_minutes),
                    "rolling_minutes": round(rolling_minutes, 2),
                    "cost_change_event": int(player.get("cost_change_event") or 0),
                    "gameweek": gw,
                }
            )

        df = pd.DataFrame(rows)
        return df.sort_values("ev", ascending=False).reset_index(drop=True)

    def _minutes_gate_ok(self, prior: float, rolling: float) -> bool:
        if prior <= 0.0 and rolling <= 0.0:
            return True
        if BUY_GATES.get("require_minutes_or_rolling", True):
            return prior >= self.min_prior_minutes or rolling >= self.min_rolling_minutes
        return prior >= self.min_prior_minutes and rolling >= self.min_rolling_minutes

    def _status_ok(self, player: dict[str, Any]) -> bool:
        status = (player.get("status") or "a").lower()
        if status in {"u", "s", "i"}:
            return False
        chance = player.get("chance_of_playing_next_round")
        if chance is not None and float(chance) < 50:
            return False
        return True

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
