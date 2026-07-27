"""Typed knowledge-graph schema for FPL entities."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    PLAYER = "player"
    TEAM = "team"
    FIXTURE = "fixture"
    GAMEWEEK = "gameweek"
    POSITION = "position"
    SIGNAL = "signal"


class EdgeType(str, Enum):
    PLAYS_FOR = "plays_for"
    HAS_POSITION = "has_position"
    IN_GAMEWEEK = "in_gameweek"
    HOME_TEAM = "home_team"
    AWAY_TEAM = "away_team"
    FACES = "faces"  # player/team faces opponent in a fixture context
    HAS_SIGNAL = "has_signal"


@dataclass(frozen=True)
class NodeId:
    type: NodeType
    key: str | int

    def as_tuple(self) -> tuple[str, str]:
        return (self.type.value, str(self.key))


@dataclass
class GraphNode:
    id: NodeId
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    type: EdgeType
    source: NodeId
    target: NodeId
    attrs: dict[str, Any] = field(default_factory=dict)


# FPL element_type id → short code
POSITION_BY_TYPE: dict[int, str] = {
    1: "GK",
    2: "DEF",
    3: "MID",
    4: "FWD",
}

SQUAD_COUNTS: dict[str, int] = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
STARTING_BOUNDS: dict[str, tuple[int, int]] = {
    "GK": (1, 1),
    "DEF": (3, 5),
    "MID": (2, 5),
    "FWD": (1, 3),
}
MAX_PER_CLUB = 3
SQUAD_SIZE = 15
STARTING_SIZE = 11
