"""Typed signal records for cold-start modelling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NewsSignal:
    """Injury / rotation / delayed-return style news."""

    player_id: int | None
    web_name: str | None
    kind: str  # injury | rotation | delayed_return | suspension | positive
    severity: float = 0.0  # 0..1, higher = worse
    exclude: bool = False
    note: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> NewsSignal:
        return cls(
            player_id=_as_optional_int(raw.get("player_id")),
            web_name=_as_optional_str(raw.get("web_name")),
            kind=str(raw.get("type") or raw.get("kind") or "news").lower(),
            severity=float(raw.get("severity") or 0.0),
            exclude=bool(raw.get("exclude") or False),
            note=str(raw.get("note") or ""),
        )


@dataclass(frozen=True)
class PreseasonSignal:
    """Friendly / training / role confirmation signal."""

    player_id: int | None
    web_name: str | None
    minutes: int = 0
    starts: int = 0
    role: str = "unknown"  # starter | rotation | backup | unknown
    boost: float = 0.0  # additive EV boost (e.g. 0.05)
    note: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PreseasonSignal:
        return cls(
            player_id=_as_optional_int(raw.get("player_id")),
            web_name=_as_optional_str(raw.get("web_name")),
            minutes=int(raw.get("minutes") or 0),
            starts=int(raw.get("starts") or 0),
            role=str(raw.get("role") or "unknown").lower(),
            boost=float(raw.get("boost") or 0.0),
            note=str(raw.get("note") or ""),
        )


@dataclass(frozen=True)
class InternationalSignal:
    """Tournament fatigue / delayed fitness signal."""

    player_id: int | None
    web_name: str | None
    tournament: str
    minutes: int = 0
    matches: int = 0
    fatigue: float = 1.0  # multiplier (<=1 reduces EV)
    note: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> InternationalSignal:
        fatigue = raw.get("fatigue")
        if fatigue is None:
            # Heuristic if not provided: deeper minutes => more fatigue.
            mins = int(raw.get("minutes") or 0)
            if mins >= 540:
                fatigue = 0.82
            elif mins >= 360:
                fatigue = 0.9
            elif mins >= 180:
                fatigue = 0.95
            else:
                fatigue = 1.0
        return cls(
            player_id=_as_optional_int(raw.get("player_id")),
            web_name=_as_optional_str(raw.get("web_name")),
            tournament=str(raw.get("tournament") or "international"),
            minutes=int(raw.get("minutes") or 0),
            matches=int(raw.get("matches") or 0),
            fatigue=float(fatigue),
            note=str(raw.get("note") or ""),
        )


def _as_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
