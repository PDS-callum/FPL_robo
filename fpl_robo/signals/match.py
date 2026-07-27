"""Match free-text player names onto FPL bootstrap players."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", value or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return _NON_ALNUM.sub("", text.lower())


def build_player_indexes(players: list[dict[str, Any]]) -> dict[str, Any]:
    """Build lookup maps for web_name / surname / full name."""
    by_id: dict[int, dict[str, Any]] = {}
    by_web: dict[str, list[int]] = {}
    by_norm: dict[str, list[int]] = {}

    for p in players:
        pid = int(p["id"])
        by_id[pid] = p
        web = str(p.get("web_name") or "").strip()
        first = str(p.get("first_name") or "").strip()
        second = str(p.get("second_name") or "").strip()
        full = f"{first} {second}".strip()
        if web:
            by_web.setdefault(web.lower(), []).append(pid)
            by_norm.setdefault(normalize_name(web), []).append(pid)
        if second:
            by_norm.setdefault(normalize_name(second), []).append(pid)
        if full:
            by_norm.setdefault(normalize_name(full), []).append(pid)
        if first and second:
            by_norm.setdefault(normalize_name(f"{first[0]}{second}"), []).append(pid)

    return {"by_id": by_id, "by_web": by_web, "by_norm": by_norm}


def resolve_player_name(
    name: str,
    indexes: dict[str, Any],
    *,
    team_short: str | None = None,
    candidate_ids: set[int] | None = None,
) -> dict[str, Any] | None:
    """Resolve a free-text name to a single FPL player when unambiguous.

    Optional `team_short` / `candidate_ids` (usually KG players mentioned in the
    article) are used to break surname ties.
    """
    raw = (name or "").strip()
    if not raw:
        return None
    by_web: dict[str, list[int]] = indexes["by_web"]
    by_norm: dict[str, list[int]] = indexes["by_norm"]
    by_id: dict[int, dict[str, Any]] = indexes["by_id"]

    def _filter(pids: list[int]) -> list[int]:
        out = list(dict.fromkeys(pids))
        if candidate_ids is not None:
            out = [p for p in out if p in candidate_ids]
        if team_short:
            want = team_short.strip().upper()
            team_hits = [
                p
                for p in out
                if str(by_id[p].get("team_short") or "").strip().upper() == want
            ]
            if team_hits:
                out = team_hits
        return out

    web_hits = _filter(list(by_web.get(raw.lower()) or []))
    if len(web_hits) == 1:
        return by_id[web_hits[0]]

    key = normalize_name(raw)
    candidates = _filter(list(by_norm.get(key) or []))
    if len(candidates) == 1:
        return by_id[candidates[0]]

    # Surname fallback: last token.
    parts = [p for p in re.split(r"\s+", raw) if p]
    if parts:
        surname_key = normalize_name(parts[-1])
        surname_hits = _filter(list(by_norm.get(surname_key) or []))
        if len(surname_hits) == 1:
            return by_id[surname_hits[0]]
    return None


def candidate_web_names_for_text(text: str, indexes: dict[str, Any], *, limit: int = 40) -> list[str]:
    """Cheap prefilter: web_names whose normalized form appears in the article text."""
    blob = normalize_name(text)
    hits: list[str] = []
    seen: set[int] = set()
    by_id: dict[int, dict[str, Any]] = indexes["by_id"]
    for norm, pids in indexes["by_norm"].items():
        if len(norm) < 4:
            continue
        if norm not in blob:
            continue
        for pid in pids:
            if pid in seen:
                continue
            seen.add(pid)
            web = str(by_id[pid].get("web_name") or "")
            if web:
                hits.append(web)
            if len(hits) >= limit:
                return hits
    return hits
