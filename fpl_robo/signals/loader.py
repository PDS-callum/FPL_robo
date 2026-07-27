"""Load signal bundles from JSON for Pi-friendly offline updates."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fpl_robo.signals.schema import InternationalSignal, NewsSignal, PreseasonSignal


@dataclass
class SignalBundle:
    as_of: str | None = None
    news: list[NewsSignal] = field(default_factory=list)
    preseason: list[PreseasonSignal] = field(default_factory=list)
    international: list[InternationalSignal] = field(default_factory=list)
    source_path: str | None = None

    @property
    def count(self) -> int:
        return len(self.news) + len(self.preseason) + len(self.international)

    def summary(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of,
            "source_path": self.source_path,
            "news": len(self.news),
            "preseason": len(self.preseason),
            "international": len(self.international),
            "total": self.count,
        }


def load_signals(path: str | Path | None) -> SignalBundle:
    """Load signals JSON; missing/empty path returns an empty bundle."""
    if path is None:
        return SignalBundle()
    p = Path(path)
    if not p.exists():
        return SignalBundle(source_path=str(p))

    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"Signals file must be a JSON object: {p}")

    return SignalBundle(
        as_of=_as_optional_str(raw.get("as_of")),
        news=[NewsSignal.from_dict(x) for x in (raw.get("news") or []) if isinstance(x, dict)],
        preseason=[
            PreseasonSignal.from_dict(x)
            for x in (raw.get("preseason") or [])
            if isinstance(x, dict)
        ],
        international=[
            InternationalSignal.from_dict(x)
            for x in (raw.get("international") or [])
            if isinstance(x, dict)
        ],
        source_path=str(p),
    )


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
