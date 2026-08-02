"""Minutes policy: what may affect EV vs dashboard-only coverage.

Design goals:
- Fairness: club depth applies to the whole player pool, not only owned players.
- Trust: only explicit transfer/depth/team-news minutes move EV.
- Quiet: near-neutral FPL/depth fills must not thrash the optimiser.
"""

from __future__ import annotations

from typing import Any

from fpl_robo.config import SIGNAL_INGEST

# Sources allowed to change EV via minutes_mult.
EV_TRUSTED_SOURCES = frozenset(
    {
        "transfer_lineup",
        "ollama_lineup",
        "heuristic_lineup",  # only if also "strong" (see below)
    }
)

# Soft / coverage proxies — keep for dashboard, ignore in EV.
EV_IGNORE_SOURCES = frozenset(
    {
        "fpl_implied",
        "depth_model",
        "heuristic_mention",
    }
)

# Role-rule phrases that are too weak / match-report-ish for GW minutes.
SOFT_MINUTE_PHRASES = frozenset(
    {
        "scored",
        "scores",
        "hat-trick",
        "hat trick",
        "brace",
        " returns",
        "return to training",
        "back in training",
        "back available",
        "fit to play",
        "available for",
        "named in the squad",
        "in the squad",
    }
)


def source_trusted_for_ev(source: str | None) -> bool:
    src = str(source or "")
    if src in EV_IGNORE_SOURCES or src == "fpl_implied":
        return False
    return src in EV_TRUSTED_SOURCES or src.startswith("ollama")


def lineup_row_applies_to_ev(row: dict[str, Any] | Any) -> bool:
    """True when a predicted_lineups row should move minutes_mult."""
    if isinstance(row, dict):
        src = str(row.get("source") or "")
        mins = row.get("predicted_minutes")
        role = str(row.get("role") or "").lower()
        note = str(row.get("note") or "").lower()
        evidence = str(row.get("evidence") or "").lower()
        conf = float(row.get("confidence") or 0)
    else:
        src = str(getattr(row, "source", "") or "")
        mins = getattr(row, "predicted_minutes", None)
        role = str(getattr(row, "role", "") or "").lower()
        note = str(getattr(row, "note", "") or "").lower()
        evidence = ""
        conf = float(getattr(row, "confidence", 0) or 0)

    if not source_trusted_for_ev(src):
        return False

    blob = f"{note} {evidence}"
    if any(p in blob for p in SOFT_MINUTE_PHRASES):
        # Match-report form language is not a minutes forecast.
        return False

    # Near-neutral "start ~85m" coverage rows do not help EV.
    try:
        mins_i = int(mins) if mins is not None else None
    except (TypeError, ValueError):
        mins_i = None
    if mins_i is not None and mins_i >= 75 and role in {"start", "starter", "xi"}:
        # Only keep high minutes when confidence is strong team-news grade.
        min_conf = float(SIGNAL_INGEST.get("ev_start_minutes_min_confidence") or 0.8)
        if conf < min_conf and src != "transfer_lineup":
            return False

    # Transfer / explicit bench / low minutes always matter when trusted.
    if src == "transfer_lineup":
        return True
    # Heuristic low-minutes from match reports ("scores… minutes risk") is noise.
    if src == "heuristic_lineup" and mins_i is not None and mins_i <= 55:
        strong_bench = any(
            p in blob
            for p in (
                "bench",
                "unlikely to start",
                "not expected to start",
                "left out",
                "depth",
                "veteran",
                "backup",
            )
        )
        if not strong_bench:
            return False
    if role in {"bench", "doubt"} or (mins_i is not None and mins_i <= 55):
        return True
    if conf >= float(SIGNAL_INGEST.get("ev_lineup_min_confidence") or 0.75):
        return True
    return False


def fwd_absolute_ep_play_prob(ep_next: float | None) -> float:
    """Absolute EP → minutes certainty for forwards (club rank is not enough).

    Every club has an EP#1 FWD. Weak absolute EP (≈2.0 fillers) are not the
    same minutes/output bet as a 2.3+ lead striker. This keeps quality focals
    ahead of cheap club fillers without naming players.
    """
    try:
        ep = float(ep_next) if ep_next is not None else 0.0
    except (TypeError, ValueError):
        ep = 0.0
    if ep >= 2.4:
        return 1.0
    if ep >= 2.25:
        return float(SIGNAL_INGEST.get("fwd_ep_play_high") or 0.98)
    if ep >= 2.1:
        return float(SIGNAL_INGEST.get("fwd_ep_play_mid") or 0.94)
    if ep >= 2.0:
        return float(SIGNAL_INGEST.get("fwd_ep_play_low") or 0.78)
    return float(SIGNAL_INGEST.get("fwd_ep_play_floor") or 0.55)


def mid_absolute_ep_play_prob(ep_next: float | None) -> float:
    """Absolute EP → minutes certainty for midfielders (slightly looser than FWD)."""
    try:
        ep = float(ep_next) if ep_next is not None else 0.0
    except (TypeError, ValueError):
        ep = 0.0
    if ep >= 2.5:
        return 1.0
    if ep >= 2.2:
        return float(SIGNAL_INGEST.get("mid_ep_play_high") or 0.98)
    if ep >= 2.0:
        return float(SIGNAL_INGEST.get("mid_ep_play_mid") or 0.94)
    if ep >= 1.8:
        return float(SIGNAL_INGEST.get("mid_ep_play_low") or 0.85)
    return float(SIGNAL_INGEST.get("mid_ep_play_floor") or 0.65)


def club_rank_play_prob(
    *,
    rank: int | None,
    position: str | None = None,
    ep_gap_to_leader: float | None = None,
    ep_next: float | None = None,
) -> float:
    """Play probability from club×position pecking order (whole market)."""
    pos = (position or "").upper()
    gap = float(ep_gap_to_leader) if ep_gap_to_leader is not None else None

    if rank is None or rank <= 1:
        if pos in {"FWD", "4"}:
            return fwd_absolute_ep_play_prob(ep_next)
        if pos in {"MID", "3"}:
            return mid_absolute_ep_play_prob(ep_next)
        return 1.0

    # Forwards: minutes are more zero-sum than mids.
    if pos in {"FWD", "4"}:
        if rank == 2:
            if gap is not None and gap >= 0.25:
                base = float(SIGNAL_INGEST.get("depth_fwd_rank2_low") or 0.42)
            else:
                base = float(SIGNAL_INGEST.get("depth_fwd_rank2") or 0.55)
        else:
            base = float(SIGNAL_INGEST.get("depth_fwd_rank3") or 0.30)
        return min(base, fwd_absolute_ep_play_prob(ep_next))

    # Midfielders: apply rank haircuts (not blanket near-peer 0.96).
    if pos in {"MID", "3"}:
        if rank == 2:
            base = float(SIGNAL_INGEST.get("depth_mid_rank2") or 0.72)
            # Very close EP to club leader → slight bump toward near-peer.
            if gap is not None and gap < 0.15:
                base = max(base, float(SIGNAL_INGEST.get("depth_mid_near") or 0.96))
        elif rank == 3:
            base = float(SIGNAL_INGEST.get("depth_mid_rank3") or 0.58)
        else:
            base = float(SIGNAL_INGEST.get("depth_mid_rank4") or 0.50)
        return min(base, mid_absolute_ep_play_prob(ep_next))

    # DEF/GK: milder
    if rank == 2:
        return 0.85
    return 0.65
