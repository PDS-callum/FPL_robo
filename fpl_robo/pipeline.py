"""End-to-end pipelines for squad recommendation and execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from fpl_robo.config import (
    DEFAULT_MANAGER_ID,
    DEFAULT_SIGNALS_PATH,
    BENCH_COVER,
    PRESEASON_GW_WEIGHTS,
    PRESEASON_HORIZON,
    PRESEASON_RISK_PROFILES,
)
from fpl_robo.data.fpl_client import FPLClient
from fpl_robo.data.manager import ManagerState, load_manager_state
from fpl_robo.data.team_updater import apply_initial_squad
from fpl_robo.graph.builder import build_knowledge_graph
from fpl_robo.graph.signals import enrich_graph_with_signals
from fpl_robo.graph.store import KnowledgeGraph
from fpl_robo.optimize.single_gw import SquadSolution, optimise_single_gw
from fpl_robo.predict.baseline import BaselineEVModel, PreseasonEVModel
from fpl_robo.signals import aggregate_player_signals, load_signals


@dataclass
class PipelineResult:
    graph_summary: dict[str, int]
    gameweek: int | None
    solution: SquadSolution
    top_ev: list[dict[str, Any]]
    manager: ManagerState | None = None


@dataclass
class InitialTeamResult:
    graph_summary: dict[str, int]
    gameweek: int | None
    horizon: int
    risk: str
    budget: float
    manager: ManagerState
    top_ev: list[dict[str, Any]]
    solution: SquadSolution
    signals: dict[str, Any] | None = None
    execution: dict[str, Any] | None = None


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
        require_cover_eligible=True,
        bench_ev_weight=BENCH_COVER["bench_ev_weight"],
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


def build_initial_team(
    manager_id: int = DEFAULT_MANAGER_ID,
    budget: float = 100.0,
    horizon: int = PRESEASON_HORIZON,
    risk: str = "balanced",
    top_n_preview: int = 12,
    execute: bool = False,
    signals_path: str | None = DEFAULT_SIGNALS_PATH,
    client: FPLClient | None = None,
) -> InitialTeamResult:
    if risk not in PRESEASON_RISK_PROFILES:
        raise ValueError(f"Unknown risk profile: {risk}")

    cfg = PRESEASON_RISK_PROFILES[risk]
    weights = PRESEASON_GW_WEIGHTS[:horizon]
    if not weights:
        raise ValueError("Horizon must be >= 1")

    client = client or FPLClient()
    manager = load_manager_state(manager_id=manager_id, client=client)

    kg: KnowledgeGraph = build_knowledge_graph(client=client)
    bundle = load_signals(signals_path)
    signal_edges = enrich_graph_with_signals(kg, bundle)

    players = {int(p["id"]): p for p in kg.players()}
    factors = aggregate_player_signals(bundle, players_by_id=players)

    model = PreseasonEVModel(
        gameweek_weights=weights,
        ownership_weight=cfg["ownership_weight"],
        uncertainty_penalty=cfg["uncertainty_penalty"],
        signal_factors=factors,
    )
    predictions = model.predict(kg)

    solution = optimise_single_gw(
        predictions,
        budget=budget,
        min_play_chance=cfg["min_play_chance"],
        exclude_unavailable=True,
        require_cover_eligible=True,
        bench_ev_weight=BENCH_COVER["bench_ev_weight"],
    )

    preview_cols = [
        "web_name",
        "position",
        "team_short",
        "price",
        "ev",
        "preseason_base_ev",
        "uncertainty_factor",
        "signal_applied",
        "signal_news_mult",
        "signal_fatigue_mult",
    ]
    preview = predictions.head(top_n_preview)[preview_cols].to_dict(orient="records")

    execution: dict[str, Any] | None = None
    if execute and solution.status == "Optimal":
        execution = apply_initial_squad(
            client=client,
            manager_id=manager_id,
            squad_rows=solution.squad,
            captain_id=solution.captain_id,
        )

    signals_summary = {
        **bundle.summary(),
        "matched_players": len(factors),
        "graph_edges_added": signal_edges,
    }

    return InitialTeamResult(
        graph_summary=kg.summary(),
        gameweek=kg.current_gameweek(),
        horizon=horizon,
        risk=risk,
        budget=budget,
        manager=manager,
        top_ev=preview,
        solution=solution,
        signals=signals_summary,
        execution=execution,
    )


def initial_team_result_to_dict(result: InitialTeamResult) -> dict[str, Any]:
    return {
        "graph_summary": result.graph_summary,
        "gameweek": result.gameweek,
        "horizon": result.horizon,
        "risk": result.risk,
        "budget": result.budget,
        "manager": {
            "manager_id": result.manager.manager_id,
            "name": result.manager.name,
            "team_name": result.manager.team_name,
        },
        "signals": result.signals,
        "top_ev": result.top_ev,
        "solution": result.solution.to_dict(),
        "execution": result.execution,
    }


def show_manager(
    manager_id: int = DEFAULT_MANAGER_ID,
    client: FPLClient | None = None,
) -> ManagerState:
    client = client or FPLClient()
    return load_manager_state(manager_id, client=client)
