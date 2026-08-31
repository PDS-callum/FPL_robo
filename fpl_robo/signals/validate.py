"""KG-backed evidence checks for LLM news signals."""

from __future__ import annotations

import re
from typing import Any

from fpl_robo.signals.match import alias_in_text, normalize_name

_INJURY_KW = (
    "injur",
    "hamstring",
    "knock",
    "doubt",
    "ruled out",
    "sidelined",
    "out for",
    "miss",
    "fitness",
    "withdrawn",
    "scan",
    "surgery",
    "absent",
    "months out",
    "groin",
)
_SUSPEND_KW = ("suspend", "ban", "red card", "banned")
_ROTATION_KW = ("rotate", "rested", "benched", "dropped", "left out")
_POSITIVE_KW = (
    "return",
    "fit",
    "starting",
    "back in training",
    "available",
    "cleared",
    "hat-trick",
    "hattrick",
    "hat trick",
    "brace",
    "scored",
    "scores",
    "goals",
    "debut",
    "pre-season",
    "preseason",
    "friendly",
)
_TRANSFER_KW = (
    "sign",
    "joins",
    "joined",
    "has joined",
    "completes",
    "complete signing",
    "completes move",
    "sealed",
    "seal deal",
    "transfer",
    "new signing",
    "unveiled",
    "move to",
    "move from",
    "after",
)
_FORM_KW = (
    "hat-trick",
    "hattrick",
    "hat trick",
    "brace",
    "scored",
    "scores",
    "goals",
    "debut",
    "pre-season",
    "preseason",
    "friendly",
    "impressive",
    "on target",
    "lively",
    "sharp",
    "winner",
)


def _contains_norm(haystack_norm: str, needle: str) -> bool:
    """Legacy normalized substring check — prefer `alias_in_text` for player names."""
    key = normalize_name(needle)
    return bool(key) and len(key) >= 3 and key in haystack_norm


def _keyword_hit(text_lower: str, keywords: tuple[str, ...]) -> bool:
    return any(k in text_lower for k in keywords)


def player_mentioned(text: str, player: dict[str, Any]) -> bool:
    """True if article text mentions this player's web/full/surname as whole tokens."""
    from fpl_robo.signals.match import alias_match_spans, player_scan_aliases

    blob = text or ""
    first = str(player.get("first_name") or "").strip().lower()
    for alias in player_scan_aliases(player):
        if not alias_in_text(blob, alias):
            continue
        # Multi-token / dotted aliases ("Bruno Fernandes", "B.Fernandes") are strong.
        if " " in alias or "." in alias:
            return True
        # Bare surname: ignore spans that are clearly "OtherFirst Fernandes".
        if first and not alias_in_text(blob, first):
            clean = False
            for start, _end in alias_match_spans(blob, alias):
                prefix = blob[max(0, start - 24) : start].strip()
                prev = prefix.split()[-1] if prefix.split() else ""
                prev_clean = prev.strip(".,;:'\"“”").lower()
                if (
                    prev_clean
                    and prev_clean != first
                    and prev_clean[:1].isalpha()
                    and len(prev_clean) >= 3
                ):
                    continue
                clean = True
                break
            if clean:
                return True
            continue
        return True
    return False


def team_mentioned(text: str, player: dict[str, Any], team_aliases: dict[int, list[str]]) -> bool:
    from fpl_robo.signals.match import club_hint_in_text

    tid = player.get("team_id")
    blob = normalize_name(text)
    if tid is not None:
        for alias in team_aliases.get(int(tid), []):
            if _contains_norm(blob, alias):
                return True
    short = str(player.get("team_short") or "")
    name = str(player.get("team_name") or "")
    if short and len(short) >= 3 and _contains_norm(blob, short):
        return True
    if name and _contains_norm(blob, name):
        return True
    # Nicknames ("Man Utd", "Spurs") when alias map is missing or incomplete.
    return club_hint_in_text(text, player)


def build_team_aliases(teams: list[dict[str, Any]]) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for t in teams:
        tid = int(t["id"])
        aliases = []
        for key in ("name", "short_name"):
            val = str(t.get(key) or "").strip()
            if val:
                aliases.append(val)
        # Common nicknames / stubs for matching RSS text.
        name = str(t.get("name") or "").lower()
        extras = {
            "manchester united": ["man utd", "man united", "manchester united"],
            "manchester city": ["man city", "manchester city"],
            "tottenham hotspur": ["tottenham", "spurs"],
            "nottingham forest": ["nottingham forest", "forest", "nottingham"],
            "wolverhampton wanderers": ["wolves", "wolverhampton"],
            "brighton & hove albion": ["brighton"],
            "newcastle united": ["newcastle"],
            "west ham united": ["west ham"],
        }
        for full, nick in extras.items():
            if full in name:
                aliases.extend(nick)
        # Dedup preserving order.
        seen: set[str] = set()
        clean: list[str] = []
        for a in aliases:
            k = a.lower()
            if k in seen:
                continue
            seen.add(k)
            clean.append(a)
        out[tid] = clean
    return out


def kind_keywords(kind: str) -> tuple[str, ...]:
    kind = (kind or "").lower()
    if kind in {"injury", "delayed_return"}:
        return _INJURY_KW
    if kind == "suspension":
        return _SUSPEND_KW
    if kind == "rotation":
        return _ROTATION_KW + ("absent", "missed out", "misses", "missed")
    if kind == "positive":
        return _POSITIVE_KW
    if kind == "transfer":
        return _TRANSFER_KW
    if kind == "form":
        return _FORM_KW
    return ()


def score_signal_evidence(
    *,
    article_text: str,
    player: dict[str, Any],
    kind: str,
    team_aliases: dict[int, list[str]],
    llm_note: str = "",
) -> dict[str, Any]:
    """Score how well article+KG support an LLM claim.

    Returns confidence 0..1 plus evidence flags.
    """
    text = article_text or ""
    text_lower = text.lower()
    name_hit = player_mentioned(text, player)
    team_hit = team_mentioned(text, player, team_aliases)
    kw = kind_keywords(kind)
    keyword_hit = _keyword_hit(text_lower, kw) if kw else False

    # Proximity: player token and keyword within a short window.
    proximity = False
    web = str(player.get("web_name") or "")
    second = str(player.get("second_name") or "")
    tokens = [t for t in (web, second) if t and len(t) >= 3]
    if web and " " in web:
        tail = web.split()[-1]
        if len(normalize_name(tail)) >= 4:
            tokens.append(tail)
    if tokens and kw:
        from fpl_robo.signals.match import alias_match_spans

        for tok in tokens:
            for start, end in alias_match_spans(text, tok):
                window = text_lower[max(0, start - 60) : end + 80]
                if _keyword_hit(window, kw):
                    proximity = True
                    break
            if proximity:
                break

    status = str(player.get("status") or "a").lower()
    chance = player.get("chance_of_playing_next_round")
    fpl_news = str(player.get("news") or "").strip()
    fpl_flag = status in {"i", "d", "s"} or (
        chance is not None and float(chance) < 75
    ) or bool(fpl_news)
    # Alignment: FPL already flags issue and LLM says injury/suspension.
    status_align = bool(fpl_flag and kind in {"injury", "suspension", "delayed_return"})

    confidence = 0.0
    if name_hit:
        confidence += 0.35
    if team_hit:
        confidence += 0.25
    if keyword_hit:
        confidence += 0.15
    if proximity:
        confidence += 0.15
    if status_align:
        confidence += 0.15
    # Small bump if LLM note looks grounded (not placeholder).
    note = (llm_note or "").lower()
    if note and "short reason" not in note and len(note) >= 12:
        confidence += 0.05

    confidence = max(0.0, min(1.0, confidence))

    # Hard reject rules for low-certainty junk.
    reject_reason = None
    if not name_hit:
        reject_reason = "player_not_in_article"
    elif not team_hit and not proximity:
        reject_reason = "no_team_or_proximity_evidence"
    elif kind == "form" and not team_hit:
        # Surname collisions (Fernandes → Mateus vs Bruno) need a club cue
        # like "Man Utd" / "Manchester United", or the player's first name.
        first = str(player.get("first_name") or "").strip()
        if not first or not alias_in_text(text, first):
            reject_reason = "form_missing_club_or_firstname"
    elif not keyword_hit:
        reject_reason = "kind_keywords_missing"
    elif confidence < 0.65:
        reject_reason = "confidence_below_threshold"

    return {
        "confidence": round(confidence, 3),
        "name_hit": name_hit,
        "team_hit": team_hit,
        "keyword_hit": keyword_hit,
        "proximity": proximity,
        "status_align": status_align,
        "fpl_status": status,
        "fpl_chance": chance,
        "fpl_news": fpl_news or None,
        "reject_reason": reject_reason,
        "accepted": reject_reason is None,
    }


def player_display_name(player: dict[str, Any]) -> str:
    first = str(player.get("first_name") or "").strip()
    second = str(player.get("second_name") or "").strip()
    full = f"{first} {second}".strip()
    web = str(player.get("web_name") or "").strip()
    if full and web and normalize_name(full) != normalize_name(web):
        return f"{full} / {web}"
    return full or web or "?"


def prioritize_kg_candidates(players: list[dict[str, Any]], *, limit: int = 16) -> list[dict[str, Any]]:
    """Prefer injured/doubtful/suspended players so the LLM prompt stays focused."""

    def _rank(p: dict[str, Any]) -> tuple:
        status = str(p.get("status") or "a").lower()
        status_rank = {"i": 0, "s": 1, "d": 2, "u": 3}.get(status, 4)
        chance = p.get("chance_of_playing_next_round")
        chance_rank = 100 if chance is None else float(chance)
        return (status_rank, chance_rank, str(p.get("web_name") or ""))

    return sorted(players, key=_rank)[:limit]


def kg_candidate_lines(players: list[dict[str, Any]], *, limit: int = 16) -> list[str]:
    """Compact KG identity + availability lines for the LLM prompt."""
    lines: list[str] = []
    for p in prioritize_kg_candidates(players, limit=limit):
        label = player_display_name(p)
        team = p.get("team_short") or "?"
        pos = p.get("position") or "?"
        status = p.get("status") or "a"
        chance = p.get("chance_of_playing_next_round")
        age = p.get("age")
        news = str(p.get("news") or "").strip()
        bits = [str(team), str(pos), f"status={status}"]
        if age is not None:
            bits.append(f"age={age}")
        if chance is not None:
            bits.append(f"chance={chance}")
        line = f"{label} ({', '.join(bits)})"
        if news:
            line += f' news="{news[:80]}"'
        lines.append(line)
    return lines
