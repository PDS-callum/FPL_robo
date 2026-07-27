"""Attach external signals onto the knowledge graph."""

from __future__ import annotations

from fpl_robo.graph.schema import EdgeType, GraphEdge, GraphNode, NodeId, NodeType
from fpl_robo.graph.store import KnowledgeGraph
from fpl_robo.signals.loader import SignalBundle


def enrich_graph_with_signals(kg: KnowledgeGraph, bundle: SignalBundle) -> int:
    """Add SIGNAL nodes and HAS_SIGNAL edges. Returns number of edges added."""
    if bundle.count == 0:
        return 0

    players_by_name = {
        str(p.get("web_name") or "").strip().lower(): int(p["id"])
        for p in kg.players()
        if p.get("web_name")
    }

    def resolve(player_id: int | None, web_name: str | None) -> int | None:
        if player_id is not None and kg.has_node(NodeId(NodeType.PLAYER, player_id)):
            return player_id
        if web_name:
            return players_by_name.get(web_name.strip().lower())
        return None

    added = 0
    signal_idx = 0

    def add_signal(kind: str, player_id: int, attrs: dict) -> None:
        nonlocal signal_idx, added
        signal_idx += 1
        sid = f"{kind}:{signal_idx}"
        kg.add_node(
            GraphNode(
                NodeId(NodeType.SIGNAL, sid),
                {"kind": kind, "player_id": player_id, **attrs},
            )
        )
        kg.add_edge(
            GraphEdge(
                EdgeType.HAS_SIGNAL,
                NodeId(NodeType.PLAYER, player_id),
                NodeId(NodeType.SIGNAL, sid),
                {"kind": kind},
            )
        )
        added += 1

    for sig in bundle.news:
        pid = resolve(sig.player_id, sig.web_name)
        if pid is None:
            continue
        add_signal(
            "news",
            pid,
            {
                "type": sig.kind,
                "severity": sig.severity,
                "exclude": sig.exclude,
                "note": sig.note,
            },
        )

    for sig in bundle.preseason:
        pid = resolve(sig.player_id, sig.web_name)
        if pid is None:
            continue
        add_signal(
            "preseason",
            pid,
            {
                "minutes": sig.minutes,
                "starts": sig.starts,
                "role": sig.role,
                "boost": sig.boost,
                "note": sig.note,
            },
        )

    for sig in bundle.international:
        pid = resolve(sig.player_id, sig.web_name)
        if pid is None:
            continue
        add_signal(
            "international",
            pid,
            {
                "tournament": sig.tournament,
                "minutes": sig.minutes,
                "matches": sig.matches,
                "fatigue": sig.fatigue,
                "note": sig.note,
            },
        )

    for sig in bundle.predicted_lineups:
        pid = resolve(sig.player_id, sig.web_name)
        if pid is None:
            continue
        add_signal(
            "predicted_lineup",
            pid,
            {
                "role": sig.role,
                "team_short": sig.team_short,
                "gameweek": sig.gameweek,
                "confidence": sig.confidence,
                "note": sig.note,
            },
        )

    return added
