"""External preseason / news / international signals."""

from fpl_robo.signals.aggregate import PlayerSignalFactors, aggregate_player_signals
from fpl_robo.signals.loader import SignalBundle, load_signals
from fpl_robo.signals.schema import (
    InternationalSignal,
    NewsSignal,
    PredictedLineupSignal,
    PreseasonSignal,
)

__all__ = [
    "InternationalSignal",
    "IngestResult",
    "NewsSignal",
    "PlayerSignalFactors",
    "PredictedLineupSignal",
    "PreseasonSignal",
    "SignalBundle",
    "aggregate_player_signals",
    "ingest_signals",
    "load_signals",
]


def __getattr__(name: str):
    if name in {"IngestResult", "ingest_signals"}:
        from fpl_robo.signals import ingest as ingest_mod

        return getattr(ingest_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
