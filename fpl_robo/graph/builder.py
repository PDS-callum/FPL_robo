"""Build a knowledge graph from FPL API payloads."""

from __future__ import annotations

from typing import Any

from fpl_robo.data.fpl_client import FPLClient
from fpl_robo.graph.schema import (
    POSITION_BY_TYPE,
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeId,
    NodeType,
)
from fpl_robo.graph.store import KnowledgeGraph


def build_knowledge_graph(
    bootstrap: dict[str, Any] | None = None,
    fixtures: list[dict[str, Any]] | None = None,
    client: FPLClient | None = None,
) -> KnowledgeGraph:
    """Fetch (if needed) and load FPL entities into a typed graph."""
    client = client or FPLClient()
    bootstrap = bootstrap or client.bootstrap_static()
    fixtures = fixtures if fixtures is not None else client.fixtures()

    kg = KnowledgeGraph()

    for pos_id, code in POSITION_BY_TYPE.items():
        kg.add_node(
            GraphNode(
                NodeId(NodeType.POSITION, code),
                {"element_type": pos_id, "singular_name": code},
            )
        )

    teams_by_id: dict[int, str] = {}
    for team in bootstrap.get("teams", []):
        team_id = int(team["id"])
        teams_by_id[team_id] = team.get("short_name") or team.get("name")
        kg.add_node(
            GraphNode(
                NodeId(NodeType.TEAM, team_id),
                {
                    "name": team.get("name"),
                    "short_name": team.get("short_name"),
                    "strength": team.get("strength"),
                    "strength_attack_home": team.get("strength_attack_home"),
                    "strength_attack_away": team.get("strength_attack_away"),
                    "strength_defence_home": team.get("strength_defence_home"),
                    "strength_defence_away": team.get("strength_defence_away"),
                },
            )
        )

    for event in bootstrap.get("events", []):
        gw = int(event["id"])
        kg.add_node(
            GraphNode(
                NodeId(NodeType.GAMEWEEK, gw),
                {
                    "name": event.get("name"),
                    "deadline_time": event.get("deadline_time"),
                    "finished": bool(event.get("finished")),
                    "is_current": bool(event.get("is_current")),
                    "is_next": bool(event.get("is_next")),
                    "average_entry_score": event.get("average_entry_score"),
                },
            )
        )

    for el in bootstrap.get("elements", []):
        player_id = int(el["id"])
        team_id = int(el["team"])
        element_type = int(el["element_type"])
        position = POSITION_BY_TYPE.get(element_type, "UNK")
        price = float(el.get("now_cost", 0)) / 10.0

        kg.add_node(
            GraphNode(
                NodeId(NodeType.PLAYER, player_id),
                {
                    "web_name": el.get("web_name"),
                    "first_name": el.get("first_name"),
                    "second_name": el.get("second_name"),
                    "birth_date": el.get("birth_date") or None,
                    "age": _age_from_birth_date(el.get("birth_date")),
                    "news": (str(el.get("news") or "").strip() or None),
                    "news_added": el.get("news_added") or None,
                    "team_id": team_id,
                    "team_short": teams_by_id.get(team_id),
                    "position": position,
                    "element_type": element_type,
                    "price": price,
                    "status": el.get("status"),
                    "chance_of_playing_next_round": el.get("chance_of_playing_next_round"),
                    "chance_of_playing_this_round": el.get("chance_of_playing_this_round"),
                    "form": _to_float(el.get("form")),
                    "points_per_game": _to_float(el.get("points_per_game")),
                    "total_points": int(el.get("total_points") or 0),
                    "minutes": int(el.get("minutes") or 0),
                    "ep_next": _to_float(el.get("ep_next")),
                    "ep_this": _to_float(el.get("ep_this")),
                    "ict_index": _to_float(el.get("ict_index")),
                    "expected_goals": _to_float(el.get("expected_goals")),
                    "expected_assists": _to_float(el.get("expected_assists")),
                    "selected_by_percent": _to_float(el.get("selected_by_percent")),
                },
            )
        )
        kg.add_edge(
            GraphEdge(
                EdgeType.PLAYS_FOR,
                NodeId(NodeType.PLAYER, player_id),
                NodeId(NodeType.TEAM, team_id),
            )
        )
        kg.add_edge(
            GraphEdge(
                EdgeType.HAS_POSITION,
                NodeId(NodeType.PLAYER, player_id),
                NodeId(NodeType.POSITION, position),
            )
        )

    for fix in fixtures:
        fixture_id = int(fix["id"])
        event = fix.get("event")
        team_h = int(fix["team_h"])
        team_a = int(fix["team_a"])
        kg.add_node(
            GraphNode(
                NodeId(NodeType.FIXTURE, fixture_id),
                {
                    "event": event,
                    "team_h": team_h,
                    "team_a": team_a,
                    "team_h_difficulty": fix.get("team_h_difficulty"),
                    "team_a_difficulty": fix.get("team_a_difficulty"),
                    "finished": bool(fix.get("finished")),
                    "kickoff_time": fix.get("kickoff_time"),
                },
            )
        )
        kg.add_edge(
            GraphEdge(
                EdgeType.HOME_TEAM,
                NodeId(NodeType.FIXTURE, fixture_id),
                NodeId(NodeType.TEAM, team_h),
                {"difficulty": fix.get("team_h_difficulty")},
            )
        )
        kg.add_edge(
            GraphEdge(
                EdgeType.AWAY_TEAM,
                NodeId(NodeType.FIXTURE, fixture_id),
                NodeId(NodeType.TEAM, team_a),
                {"difficulty": fix.get("team_a_difficulty")},
            )
        )
        if event is not None:
            kg.add_edge(
                GraphEdge(
                    EdgeType.IN_GAMEWEEK,
                    NodeId(NodeType.FIXTURE, fixture_id),
                    NodeId(NodeType.GAMEWEEK, int(event)),
                )
            )
            # Bidirectional "faces" edges for opponent lookup in predictors
            kg.add_edge(
                GraphEdge(
                    EdgeType.FACES,
                    NodeId(NodeType.TEAM, team_h),
                    NodeId(NodeType.TEAM, team_a),
                    {
                        "fixture_id": fixture_id,
                        "gameweek": int(event),
                        "home": True,
                        "difficulty": fix.get("team_h_difficulty"),
                    },
                )
            )
            kg.add_edge(
                GraphEdge(
                    EdgeType.FACES,
                    NodeId(NodeType.TEAM, team_a),
                    NodeId(NodeType.TEAM, team_h),
                    {
                        "fixture_id": fixture_id,
                        "gameweek": int(event),
                        "home": False,
                        "difficulty": fix.get("team_a_difficulty"),
                    },
                )
            )

    return kg


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _age_from_birth_date(value: Any) -> int | None:
    """Compute integer age from an ISO birth_date (YYYY-MM-DD)."""
    if not value:
        return None
    try:
        from datetime import date

        text = str(value)[:10]
        y, m, d = (int(x) for x in text.split("-"))
        born = date(y, m, d)
        today = date.today()
        return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    except Exception:
        return None
