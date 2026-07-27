"""Deterministic title/snippet heuristics to seed high-certainty signals."""

from __future__ import annotations

import re
from typing import Any

from fpl_robo.signals.match import normalize_name

_TITLE_INJURY = re.compile(
    r"\b(?P<name>[A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+)?)\b.{0,40}\b(?P<kw>injur(?:y|ed)|doubt|sidelined|ruled out|hamstring|knock)\b",
    re.IGNORECASE,
)
_TITLE_SUSPEND = re.compile(
    r"\b(?P<name>[A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+)?)\b.{0,40}\b(?P<kw>suspend(?:ed|sion)?|banned|ban)\b",
    re.IGNORECASE,
)


def _pick_player(hits: list[dict[str, Any]], *, kind: str) -> dict[str, Any] | None:
    """Disambiguate surname collisions using KG availability when needed."""
    if not hits:
        return None
    if len(hits) == 1:
        return hits[0]
    if kind in {"injury", "suspension"}:
        flagged = [
            p
            for p in hits
            if str(p.get("status") or "a").lower() in {"i", "d", "s"}
            or (
                p.get("chance_of_playing_next_round") is not None
                and float(p.get("chance_of_playing_next_round") or 100) < 75
            )
            or bool(str(p.get("news") or "").strip())
        ]
        if len(flagged) == 1:
            return flagged[0]
    return None


def heuristic_signals_from_title(
    title: str,
    candidate_players: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Seed signals when a headline clearly pairs a candidate with injury/ban language."""
    if not title or not candidate_players:
        return []

    out: list[dict[str, Any]] = []
    patterns = (
        (_TITLE_INJURY, "injury", 0.55),
        (_TITLE_SUSPEND, "suspension", 0.7),
    )
    by_norm: dict[str, list[dict[str, Any]]] = {}
    for p in candidate_players:
        keys = [
            normalize_name(str(p.get("web_name") or "")),
            normalize_name(str(p.get("second_name") or "")),
            normalize_name(f"{p.get('first_name') or ''} {p.get('second_name') or ''}"),
        ]
        first = str(p.get("first_name") or "").strip()
        second = str(p.get("second_name") or "").strip()
        if first and second:
            # Also index "Joe Gomez" style and first-initial forms.
            keys.append(normalize_name(f"{first[0]} {second}"))
            keys.append(normalize_name(f"{first[0]}.{second}"))
        for key in keys:
            if not key:
                continue
            bucket = by_norm.setdefault(key, [])
            if p not in bucket:
                bucket.append(p)

    for cre, kind, severity in patterns:
        for match in cre.finditer(title):
            name = match.group("name")
            key = normalize_name(name)
            hits = by_norm.get(key) or []
            if not hits and " " in name:
                hits = by_norm.get(normalize_name(name.split()[-1])) or []
            player = _pick_player(hits, kind=kind)
            if player is None:
                continue
            status = str(player.get("status") or "a").lower()
            sev = severity
            exclude = False
            if status in {"i", "s"} or (
                player.get("chance_of_playing_next_round") is not None
                and float(player.get("chance_of_playing_next_round") or 100) <= 25
            ):
                sev = max(sev, 0.75)
                exclude = status in {"i", "s"}
            first = str(player.get("first_name") or "").strip()
            web = str(player.get("web_name") or "").strip()
            out.append(
                {
                    "web_name": web,
                    "first_name": first or None,
                    "second_name": player.get("second_name"),
                    "team_short": player.get("team_short"),
                    "type": kind,
                    "severity": sev,
                    "exclude": exclude,
                    "evidence": title,
                    "note": f"[heuristic] {match.group('kw')} in headline → {first} {web}".strip(),
                    "source": "heuristic",
                }
            )
    # Dedupe by player/type
    seen = set()
    uniq = []
    for item in out:
        key = (item.get("web_name"), item.get("team_short"), item.get("type"))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(item)
    return uniq
