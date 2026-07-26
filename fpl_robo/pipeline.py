"""End-to-end step-1 pipeline: data → graph → EV → MIP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fpl_robo.config import DEFAULT_MANAGER_ID
from fpl_robo.data.fpl_client import FPLClient
from fpl_robo.data.manager import ManagerState, load_manager_state
from fpl_robo.graph.builder import build_knowledge_graph
from fpl_robo.graph.store import KnowledgeGraph
from fpl_robo.optimize.single_gw import SquadSolution, optimise_single_gw
from fpl_robo.predict.baseline import BaselineEVModel


@dataclass
class PipelineResult:
    graph_summary: dict[str, int]
    gameweek: int | None
    solution: SquadSolution
    top_ev: list[dict[str, Any]]
    manager: ManagerState | None = None


def build_squad(
    budget: float | None = None,
    min_play_chance: int | None = 50,
    top_n_preview: int = 10,
    manager_id: int = DEFAULT_MANAGER_ID,
    use_manager_budget: bool = True,
    client: FPLClient | None = None,
) -> PipelineResult:
    client = client or FPLClient()
    manager = load_manager_state(manager_id, client=client)

    if budget is None:
        if use_manager_budget and manager.team_value > 0:
            # From-scratch build uses full squad purchasing power.
            budget = round(manager.team_value + manager.bank, 1)
        else:
            budget = 100.0

    kg: KnowledgeGraph = build_knowledge_graph(client=client)
    model = BaselineEVModel()
    predictions = model.predict(kg)
    solution = optimise_single_gw(
        predictions,
        budget=budget,
        min_play_chance=min_play_chance,
    )
    preview = predictions.head(top_n_preview)[
        ["web_name", "position", "team_short", "price", "ev", "next_fdr"]
    ].to_dict(orient="records")

    return PipelineResult(
        graph_summary=kg.summary(),
        gameweek=kg.current_gameweek(),
        solution=solution,
        top_ev=preview,
        manager=manager,
    )


def show_manager(
    manager_id: int = DEFAULT_MANAGER_ID,
    client: FPLClient | None = None,
) -> ManagerState:
    client = client or FPLClient()
    return load_manager_state(manager_id, client=client)
