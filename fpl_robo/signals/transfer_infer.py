"""Infer FPL meaning from confirmed transfer / signing headlines."""

from __future__ import annotations

import re
from typing import Any

from fpl_robo.config import SIGNAL_INGEST

# Destination strength proxies when bootstrap strength is unavailable.
_ELITE = frozenset({"ARS", "CHE", "LIV", "MCI", "MUN", "TOT"})
_STRONG = frozenset({"NEW", "AVL", "BHA", "WHU", "FUL", "CRY", "Bournemouth", "BOU"})

# Current PL short codes (bootstrap team_short). Used to tell inbound vs exit.
_PL_SHORTS = frozenset(
    {
        "ARS",
        "AVL",
        "BOU",
        "BRE",
        "BHA",
        "CHE",
        "CRY",
        "EVE",
        "FUL",
        "IPS",
        "LEI",
        "LIV",
        "MCI",
        "MUN",
        "NEW",
        "NFO",
        "SOU",
        "TOT",
        "WHU",
        "WOL",
        # 2025-26 / 2026-27 promoted / common variants
        "BUR",
        "LEE",
        "SUN",
        "SHU",
        "LUT",
        "NOT",
    }
)

# Club-name fragments that are clearly not Premier League destinations.
_NON_PL_DEST = (
    "saudi",
    "al-hilal",
    "al hilal",
    "al-nassr",
    "al nassr",
    "al-ittihad",
    "al ittihad",
    "al-ahli",
    "al ahli",
    "qadsiah",
    "real madrid",
    "barcelona",
    "bayern",
    "psg",
    "paris saint",
    "juventus",
    "inter milan",
    "ac milan",
    "napoli",
    "dortmund",
    "atletico",
    "atlético",
    "championship",
    "mls",
    "liga mx",
    "serie a",
    "bundesliga",
    "ligue 1",
    "eredivisie",
    "galatasaray",
    "fenerbahce",
    "fenerbahçe",
    "besiktas",
    "beşiktaş",
    "benfica",
    "porto",
    "ajax",
    "celtic",
    "rangers",
    "riyadh",
    "jeddah",
    "qatar",
    "china",
    "chinese",
    "japan",
    "j-league",
    "major league soccer",
    "la liga",
    "laliga",
)

_EXIT_PHRASE = re.compile(
    r"\b("
    r"leaves?|left\s+(?:for|to)|departs?|departed|"
    r"sold\s+to|sale\s+to|exits?|exit\s+to|"
    r"loaned\s+to|loan\s+move\s+to|permanent\s+move\s+to|"
    r"joins?\s+(?:saudi|mls|real|barcelona|bayern|psg|juventus)|"
    r"moves?\s+(?:abroad|overseas)|"
    r"completes?\s+(?:saudi|mls)\s+"
    r")\b",
    re.IGNORECASE,
)

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
    "squad option",
    "depth",
    "rotation option",
    "depth signing",
    "depth option",
    "veteran",
    "impact sub",
    "from the bench",
    "off the bench",
    "compete for",
    "competition for",
    "support for",
    "to support",
    "increase depth",
    "add depth",
    "squad depth",
)
_DEPTH_AGE_RE = re.compile(r"\b(3[3-9]|[4-9]\d)[-\s]?year[-\s]?old\b", re.IGNORECASE)


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


def _norm_dest(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def destination_is_non_pl(destination_hint: str | None) -> bool:
    """True when the destination string clearly is not a PL club."""
    dest = _norm_dest(destination_hint or "")
    if not dest:
        return False
    # Exact / token match against known PL shorts.
    tokens = {t.strip(".,") for t in dest.upper().replace("-", " ").split()}
    if tokens & _PL_SHORTS:
        return False
    # Common PL full-name fragments (avoid false outbound).
    pl_names = (
        "arsenal",
        "chelsea",
        "liverpool",
        "manchester city",
        "man city",
        "manchester united",
        "man united",
        "tottenham",
        "spurs",
        "newcastle",
        "aston villa",
        "brighton",
        "west ham",
        "fulham",
        "crystal palace",
        "everton",
        "brentford",
        "bournemouth",
        "wolves",
        "wolverhampton",
        "nottingham forest",
        "nott'm forest",
        "ipswich",
        "leicester",
        "southampton",
        "burnley",
        "leeds",
        "sunderland",
    )
    if any(n in dest for n in pl_names):
        return False
    return any(n in dest for n in _NON_PL_DEST)


def looks_like_outbound(
    *,
    evidence: str = "",
    note: str = "",
    destination_hint: str | None = None,
) -> bool:
    """Heuristic: confirmed exit / non-PL destination language."""
    if destination_is_non_pl(destination_hint):
        return True
    blob = f"{evidence} {note} {destination_hint or ''}"
    if _EXIT_PHRASE.search(blob):
        # "leaves training" etc. — require move/club flavour nearby or non-PL dest already handled.
        lower = blob.lower()
        if any(
            k in lower
            for k in (
                "sold",
                "sale",
                "loaned",
                "loan move",
                "permanent",
                "abroad",
                "overseas",
                "saudi",
                "mls",
                "departs",
                "departed",
                "exit",
                "leaves for",
                "left for",
                "moves abroad",
            )
        ):
            return True
        # Bare "leaves Club" / "joins Al Hilal" with non-PL already True above.
        if destination_hint and destination_is_non_pl(destination_hint):
            return True
        # "Player leaves Arsenal" without destination — treat as outbound only with sold/depart/exit.
    if destination_hint and destination_is_non_pl(destination_hint):
        return True
    return False


def extract_destination_hint(*texts: str) -> str | None:
    """Pull a destination club name from transfer copy when regex capture failed."""
    blob = " ".join(str(t or "") for t in texts)
    if not blob.strip():
        return None
    patterns = (
        r"\bjoins?\s+(?P<club>[A-Z][A-Za-z]+(?:\s+(?:United|City|Town|Hotspur|Forest|Palace|Villa))?)\b",
        r"\bsigned\s+for\s+(?P<club>[A-Z][A-Za-z]+(?:\s+(?:United|City|Town|Hotspur|Forest|Palace|Villa))?)\b",
        r"\bmove\s+to\s+(?P<club>[A-Z][A-Za-z]+(?:\s+(?:United|City|Town|Hotspur|Forest|Palace|Villa))?)\b",
        r"\b(?P<club>[A-Z][A-Za-z]+(?:\s+(?:United|City|Town|Hotspur|Forest|Palace|Villa))?)\s+have\s+signed\b",
        r"\b(?P<club>[A-Z][A-Za-z]+(?:\s+(?:United|City|Town|Hotspur|Forest|Palace|Villa))?)\s+signed\b",
        r"\bnew\s+(?P<club>[A-Z][A-Za-z]+)\s+signing\b",
    )
    for pat in patterns:
        m = re.search(pat, blob)
        if not m:
            continue
        club = m.group("club").strip()
        # Skip generic / false positives.
        if club.lower() in {"transfer", "premier", "record", "deal", "move", "season"}:
            continue
        return club
    return None


def infer_transfer_impact(
    *,
    team_short: str | None,
    position: str | None = None,
    element_type: int | None = None,
    evidence: str = "",
    note: str = "",
    strength_home: float | None = None,
    destination_hint: str | None = None,
    age: int | None = None,
) -> dict[str, Any]:
    """Map a confirmed move onto role / minutes / form boost for cold-start EV.

    Heuristic (preseason-friendly):
    - Outbound / non-PL destinations → exclude + negative boost (no minutes)
    - Destination club tier drives form prior (not the selling club)
    - Explicit No.1 / straight-in language → starter + minutes
    - Backup/depth/veteran language → backup with low predicted minutes
    - Always emits predicted_minutes so EV can haircut depth signings
    """
    from fpl_robo.signals.minutes_estimate import destination_to_team_short

    blob = f"{evidence} {note}"
    dest_hint = destination_hint or extract_destination_hint(evidence, note)
    dest_short = destination_to_team_short(dest_hint) or (
        str(team_short or "").strip().upper() or None
    )

    if looks_like_outbound(
        evidence=evidence, note=note, destination_hint=dest_hint
    ):
        haircut = float(SIGNAL_INGEST.get("transfer_exit_boost") or -0.15)
        return {
            "tier": "exit",
            "role": "unavailable",
            "boost": round(max(-0.25, min(-0.05, haircut)), 3),
            "minutes": 0,
            "predicted_minutes": 0,
            "starts": 0,
            "exclude": True,
            "outbound": True,
            "destination_hint": dest_hint,
            "destination_team_short": dest_short,
            "inference": (
                f"outbound_exit,dest={_norm_dest(dest_hint) or 'unknown'}"
            ),
        }

    base = float(SIGNAL_INGEST.get("transfer_form_boost") or 0.06)
    elite_b = float(SIGNAL_INGEST.get("transfer_elite_boost") or 0.10)
    strong_b = float(SIGNAL_INGEST.get("transfer_strong_boost") or 0.08)
    # Prefer destination club for tier; selling-club strength must not inflate role.
    selling = str(team_short or "").strip().upper() or None
    strength_for_tier = strength_home
    if dest_short and selling and dest_short != selling:
        strength_for_tier = None  # do not use seller FDR/strength for buyer role
    tier = club_tier(dest_short or team_short, strength_home=strength_for_tier)

    if tier == "elite":
        boost = elite_b
        role = "starter"
    elif tier == "strong":
        boost = strong_b
        role = "starter"
    else:
        boost = base
        role = "rotation"

    lower = blob.lower()
    likely_starter = any(k in lower for k in _STARTER_KW)
    likely_backup = any(k in lower for k in _BACKUP_KW)
    if _DEPTH_AGE_RE.search(blob):
        likely_backup = True
    if age is not None and int(age) >= 33:
        likely_backup = True
    # Elite clubs signing without starter language → default rotation, not starter.
    if tier == "elite" and not likely_starter:
        role = "rotation"

    pos = (position or "").upper()
    is_gk = pos in {"GKP", "GK", "1"} or element_type == 1
    if is_gk and (
        "sign" in lower or "join" in lower or "no 1" in lower or "no.1" in lower
    ):
        # Club signing a keeper in preseason is usually for minutes.
        likely_starter = True

    minutes = 0
    starts = 0
    predicted_minutes = 45
    if likely_backup and not likely_starter:
        role = "backup"
        boost = max(0.02, boost - 0.05)
        minutes = 20
        predicted_minutes = 20
        starts = 0
    elif likely_starter:
        role = "starter"
        boost += 0.02
        minutes = 60 if is_gk else 70
        predicted_minutes = minutes
        starts = 1
    elif role == "starter":
        # Tier-default starter without explicit language — mild minutes, not 90.
        minutes = 55 if is_gk else 50
        predicted_minutes = minutes
        starts = 1
    else:
        # Rotation default (incl. elite without starter language).
        role = "rotation"
        minutes = 35
        predicted_minutes = 35
        starts = 0

    boost = max(-0.05, min(0.18, boost))
    reason_bits = [f"tier={tier}", f"role={role}"]
    if dest_short:
        reason_bits.append(f"dest={dest_short}")
    if likely_starter:
        reason_bits.append("starter_language")
    if likely_backup:
        reason_bits.append("backup_language")

    return {
        "tier": tier,
        "role": role,
        "boost": round(boost, 3),
        "minutes": minutes,
        "predicted_minutes": predicted_minutes,
        "starts": starts,
        "exclude": False,
        "outbound": False,
        "destination_hint": dest_hint,
        "destination_team_short": dest_short,
        "inference": ", ".join(reason_bits),
    }
