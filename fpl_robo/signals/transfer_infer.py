"""Infer FPL meaning from confirmed transfer / signing headlines."""

from __future__ import annotations

from typing import Any

from fpl_robo.config import SIGNAL_INGEST

# Destination strength proxies when bootstrap strength is unavailable.
_ELITE = frozenset({"ARS", "CHE", "LIV", "MCI", "MUN", "TOT"})
_STRONG = frozenset({"NEW", "AVL", "BHA", "WHU", "FUL", "CRY", "Bournemouth", "BOU"})
_STARTER_KW = (
    "no 1",
    "no.1",
    "number one",
    "first choice",
    "straight in",
    "expected to come straight",
    "expected to start",
    "nailed",
    "regular starter",
)
_BACKUP_KW = (
    "backup",
    "deputy",
    "cover",
    "squad player",
    "depth",
    "rotation option",
)


def club_tier(team_short: str | None, *, strength_home: float | None = None) -> str:
    """Return elite | strong | mid | lower for a PL club."""
    code = str(team_short or "").strip().upper()
    if strength_home is not None:
        try:
            s = float(strength_home)
            if s >= 4:
                return "elite"
            if s >= 3:
                return "strong"
            if s >= 2:
                return "mid"
            return "lower"
        except (TypeError, ValueError):
            pass
    if code in _ELITE:
        return "elite"
    if code in _STRONG:
        return "strong"
    return "mid"


def infer_transfer_impact(
    *,
    team_short: str | None,
    position: str | None = None,
    element_type: int | None = None,
    evidence: str = "",
    note: str = "",
    strength_home: float | None = None,
) -> dict[str, Any]:
    """Map a confirmed move onto role / minutes / form boost for cold-start EV.

    Heuristic (preseason-friendly):
    - Elite destinations → higher form boost + starter prior
    - Explicit No.1 / straight-in language → starter + minutes
    - Backup/depth language → rotation/backup, smaller boost
    """
    base = float(SIGNAL_INGEST.get("transfer_form_boost") or 0.06)
    elite_b = float(SIGNAL_INGEST.get("transfer_elite_boost") or 0.10)
    strong_b = float(SIGNAL_INGEST.get("transfer_strong_boost") or 0.08)
    blob = f"{evidence} {note}".lower()
    tier = club_tier(team_short, strength_home=strength_home)

    if tier == "elite":
        boost = elite_b
        role = "starter"
    elif tier == "strong":
        boost = strong_b
        role = "starter"
    else:
        boost = base
        role = "rotation"

    likely_starter = any(k in blob for k in _STARTER_KW)
    likely_backup = any(k in blob for k in _BACKUP_KW)

    pos = (position or "").upper()
    is_gk = pos in {"GKP", "GK", "1"} or element_type == 1
    if is_gk and ("sign" in blob or "join" in blob or "no 1" in blob or "no.1" in blob):
        # Club signing a keeper in preseason is usually for minutes.
        likely_starter = True

    minutes = 0
    starts = 0
    if likely_backup and not likely_starter:
        role = "backup"
        boost = max(0.02, boost - 0.04)
    elif likely_starter:
        role = "starter"
        boost += 0.02
        minutes = 60 if is_gk else 45
        starts = 1

    boost = max(-0.05, min(0.18, boost))
    reason_bits = [f"tier={tier}", f"role={role}"]
    if likely_starter:
        reason_bits.append("starter_language")
    if likely_backup:
        reason_bits.append("backup_language")

    return {
        "tier": tier,
        "role": role,
        "boost": round(boost, 3),
        "minutes": minutes,
        "starts": starts,
        "inference": ", ".join(reason_bits),
    }
