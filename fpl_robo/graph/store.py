"""In-memory knowledge graph backed by NetworkX."""

from __future__ import annotations

from typing import Any, Iterator

import networkx as nx

from fpl_robo.graph.schema import GraphEdge, GraphNode, NodeId, NodeType


class KnowledgeGraph:
    """Typed directed multigraph for FPL entities and relations."""

    def __init__(self) -> None:
        self.g = nx.MultiDiGraph()

    def add_node(self, node: GraphNode) -> None:
        self.g.add_node(node.id.as_tuple(), node_type=node.id.type.value, **node.attrs)

    def add_edge(self, edge: GraphEdge) -> None:
        self.g.add_edge(
            edge.source.as_tuple(),
            edge.target.as_tuple(),
            key=edge.type.value,
            edge_type=edge.type.value,
            **edge.attrs,
        )

    def has_node(self, node_id: NodeId) -> bool:
        return self.g.has_node(node_id.as_tuple())

    def get_attrs(self, node_id: NodeId) -> dict[str, Any]:
        return dict(self.g.nodes[node_id.as_tuple()])

    def nodes_of_type(self, node_type: NodeType) -> Iterator[tuple[NodeId, dict[str, Any]]]:
        for nid, attrs in self.g.nodes(data=True):
            if attrs.get("node_type") == node_type.value:
                yield NodeId(node_type, nid[1]), attrs

    def players(self) -> list[dict[str, Any]]:
        """Convenience: flat player dicts for predictors / MIP."""
        out: list[dict[str, Any]] = []
        for node_id, attrs in self.nodes_of_type(NodeType.PLAYER):
            row = {"id": int(node_id.key), **attrs}
            row.pop("node_type", None)
            out.append(row)
        return out

    def current_gameweek(self) -> int | None:
        """Next unfinished gameweek, else the latest finished one."""
        events = list(self.nodes_of_type(NodeType.GAMEWEEK))
        if not events:
            return None
        next_gw = [
            (int(nid.key), attrs)
            for nid, attrs in events
            if attrs.get("is_next")
        ]
        if next_gw:
            return next_gw[0][0]
        current = [
            (int(nid.key), attrs)
            for nid, attrs in events
            if attrs.get("is_current")
        ]
        if current:
            return current[0][0]
        finished = sorted(
            (
                (int(nid.key), attrs)
                for nid, attrs in events
                if attrs.get("finished")
            ),
            key=lambda x: x[0],
        )
        return finished[-1][0] if finished else None

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, attrs in self.g.nodes(data=True):
            t = attrs.get("node_type", "unknown")
            counts[t] = counts.get(t, 0) + 1
        counts["edges"] = self.g.number_of_edges()
        return counts
