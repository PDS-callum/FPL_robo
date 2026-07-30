"""Collapse raw signals into per-player EV multipliers."""

from __future__ import annotations

from dataclasses import dataclass

from fpl_robo.config import SIGNAL_INGEST
from fpl_robo.signals.loader import SignalBundle


@dataclass
class PlayerSignalFactors:
    minutes_mult: float = 1.0
    form_boost: float = 0.0
    fatigue_mult: float = 1.0
    news_mult: float = 1.0
    exclude: bool = False
    notes: list[str] | None = None

    def combined_mult(self) -> float:
        return max(0.0, self.minutes_mult * self.fatigue_mult * self.news_mult)


def aggregate_player_signals(
    bundle: SignalBundle,
    *,
    players_by_id: dict[int, dict],
    players_by_name: dict[str, int] | None = None,
) -> dict[int, PlayerSignalFactors]:
    """Map signal records onto FPL player IDs."""
    name_index = players_by_name or {
        str(p.get("web_name") or "").strip().lower(): int(pid)
        for pid, p in players_by_id.items()
        if p.get("web_name")
    }

    out: dict[int, PlayerSignalFactors] = {
        pid: PlayerSignalFactors(notes=[]) for pid in players_by_id
    }

    def resolve(player_id: int | None, web_name: str | None) -> int | None:
        if player_id is not None and player_id in players_by_id:
            return player_id
        if web_name:
            return name_index.get(web_name.strip().lower())
        return None

    for sig in bundle.news:
        pid = resolve(sig.player_id, sig.web_name)
        if pid is None:
            continue
        factors = out[pid]
        # Exclude only when explicitly flagged (ingest already gates confidence).
        # Do not auto-exclude from severity alone — that was over-aggressive.
        if sig.exclude:
            factors.exclude = True
        # Higher severity => lower news multiplier.
        severity = max(0.0, min(1.0, sig.severity))
        news_floor = float(SIGNAL_INGEST.get("news_mult_floor") or 0.45)
        # Official FPL status/news is already reflected in chance/status EV.
        # Keep exclude/cover gating, but avoid stacking a second hard haircut.
        official = (sig.source == "fpl") or bool(sig.status_align)
        if official and sig.kind in {"injury", "suspension", "delayed_return"}:
            if not sig.exclude and severity >= 0.35:
                factors.news_mult *= max(0.9, 1.0 - 0.15 * severity)
            if sig.note:
                factors.notes.append(f"news:{sig.kind}:{sig.note}")
            continue
        if sig.kind == "positive":
            factors.news_mult *= 1.0 + 0.08 * (1.0 - severity)
        elif sig.kind == "transfer":
            # Availability narrative only; form_boost comes from derived preseason.
            factors.news_mult *= 1.0 + 0.04 * (1.0 - severity)
        elif sig.kind == "form":
            factors.news_mult *= 1.0 + 0.04 * (1.0 - severity)
        elif sig.kind == "rotation":
            factors.news_mult *= max(news_floor, 1.0 - 0.35 * severity)
            factors.minutes_mult *= max(0.85, 1.0 - 0.2 * severity)
        elif factors.exclude:
            factors.news_mult *= max(0.0, 1.0 - 0.7 * severity)
        else:
            factors.news_mult *= max(news_floor, 1.0 - 0.7 * severity)
        if sig.note:
            factors.notes.append(f"news:{sig.kind}:{sig.note}")

    for sig in bundle.preseason:
        pid = resolve(sig.player_id, sig.web_name)
        if pid is None:
            continue
        factors = out[pid]
        role = sig.role
        if role == "starter" or sig.starts >= 2 or sig.minutes >= 150:
            factors.minutes_mult *= 1.08
        elif role == "rotation" or sig.minutes >= 60:
            factors.minutes_mult *= 1.02
        elif role == "backup":
            factors.minutes_mult *= 0.9
        factors.form_boost += max(-0.2, min(0.3, sig.boost))
        if sig.note:
            factors.notes.append(f"preseason:{sig.note}")

    for sig in bundle.international:
        pid = resolve(sig.player_id, sig.web_name)
        if pid is None:
            continue
        factors = out[pid]
        factors.fatigue_mult *= max(0.5, min(1.0, sig.fatigue))
        if sig.note:
            factors.notes.append(f"intl:{sig.tournament}:{sig.note}")
        else:
            factors.notes.append(f"intl:{sig.tournament}")

    start_m = float(SIGNAL_INGEST.get("lineup_start_minutes_mult") or 1.12)
    bench_m = float(SIGNAL_INGEST.get("lineup_bench_minutes_mult") or 0.88)
    doubt_m = float(SIGNAL_INGEST.get("lineup_doubt_minutes_mult") or 0.95)
    min_lineup_conf = float(SIGNAL_INGEST.get("lineup_min_confidence") or 0.65)
    min_start_conf = float(SIGNAL_INGEST.get("lineup_start_min_confidence") or 0.7)
    for sig in bundle.predicted_lineups:
        pid = resolve(sig.player_id, sig.web_name)
        if pid is None:
            continue
        factors = out[pid]
        role = (sig.role or "start").lower()
        conf = max(0.0, min(1.0, float(sig.confidence or 0.5)))
        if conf < min_lineup_conf:
            factors.notes.append(f"lineup:skipped_low_conf:{role}:{sig.web_name}")
            continue
        # Scale effect by confidence so weak predicted XIs do not dominate.
        if role in {"start", "starter", "xi"}:
            if conf < min_start_conf:
                factors.notes.append(f"lineup:skipped_start_conf:{sig.web_name}")
                continue
            mult = 1.0 + (start_m - 1.0) * conf
            factors.minutes_mult *= mult
        elif role in {"bench", "substitute", "sub"}:
            mult = 1.0 + (bench_m - 1.0) * conf
            factors.minutes_mult *= mult
        elif role in {"doubt", "doubtful"}:
            mult = 1.0 + (doubt_m - 1.0) * conf
            factors.minutes_mult *= mult
        gw = f" GW{sig.gameweek}" if sig.gameweek else ""
        factors.notes.append(f"lineup:{role}{gw}:{sig.note or sig.web_name}")

    # Drop untouched default entries to keep maps small.
    return {
        pid: factors
        for pid, factors in out.items()
        if factors.exclude
        or factors.minutes_mult != 1.0
        or factors.form_boost != 0.0
        or factors.fatigue_mult != 1.0
        or factors.news_mult != 1.0
    }
