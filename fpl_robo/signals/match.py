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


def alias_match_spans(text: str, alias: str) -> list[tuple[int, int]]:
    """Word-boundary spans for `alias` in `text` (case-insensitive).

    Prevents substring false positives such as "Raya" inside "Rayan Cherki".
    Boundaries are alphanumeric (Unicode letters/digits), so punctuation and
    whitespace still split tokens.
    """
    raw = (alias or "").strip()
    blob = text or ""
    if not raw or not blob:
        return []
    pattern = re.compile(
        rf"(?<![0-9A-Za-zÀ-ÖØ-öø-ÿ]){re.escape(raw)}(?![0-9A-Za-zÀ-ÖØ-öø-ÿ])",
        flags=re.IGNORECASE,
    )
    return [(m.start(), m.end()) for m in pattern.finditer(blob)]


def alias_in_text(text: str, alias: str) -> bool:
    """True if `alias` appears as a whole token in `text`."""
    return bool(alias_match_spans(text, alias))


# Common free-text club nicknames keyed by FPL team_short.
CLUB_TEXT_STUBS: dict[str, tuple[str, ...]] = {
    "MUN": ("man utd", "man united", "manchester united"),
    "MCI": ("man city", "manchester city"),
    "TOT": ("tottenham", "spurs"),
    "NFO": ("nottingham", "forest"),
    "WOL": ("wolves", "wolverhampton"),
    "BHA": ("brighton",),
    "NEW": ("newcastle", "toon"),
    "WHU": ("west ham", "hammers"),
    "CRY": ("crystal palace", "palace"),
    "AVL": ("aston villa", "villa"),
    "ARS": ("arsenal", "gunners"),
    "CHE": ("chelsea",),
    "LIV": ("liverpool",),
    "FUL": ("fulham",),
    "BRE": ("brentford",),
    "BOU": ("bournemouth",),
    "EVE": ("everton",),
    "LEE": ("leeds",),
    "BUR": ("burnley",),
    "SUN": ("sunderland",),
}


def club_hint_in_text(text: str, player: dict[str, Any]) -> bool:
    """True if the player's club short/name/nickname appears in free text."""
    lower = (text or "").lower()
    short = str(player.get("team_short") or "").strip()
    name = str(player.get("team_name") or "").strip()
    # Word-bound short codes ("LIV" must not match inside "deliver").
    if short and len(short) >= 3 and alias_in_text(text or "", short):
        return True
    if name and len(name) >= 4 and name.lower() in lower:
        return True
    for stub in CLUB_TEXT_STUBS.get(short.upper(), ()):
        if stub in lower:
            return True
    return False


def player_name_keys(player: dict[str, Any]) -> list[str]:
    """Normalized lookup keys for a player (web / surname / first+surname variants)."""
    keys: list[str] = []
    web = str(player.get("web_name") or "").strip()
    first = str(player.get("first_name") or "").strip()
    second = str(player.get("second_name") or "").strip()
    full = f"{first} {second}".strip()
    if web:
        keys.append(normalize_name(web))
        # Multi-token web names ("João Pedro") → also index trailing token.
        if " " in web:
            keys.append(normalize_name(web.split()[-1]))
    if second:
        keys.append(normalize_name(second))
        # "Borges Fernandes" → also index trailing surname + first+surname.
        if " " in second:
            tail = second.split()[-1]
            if len(normalize_name(tail)) >= 4:
                keys.append(normalize_name(tail))
                if first:
                    keys.append(normalize_name(f"{first} {tail}"))
                    keys.append(normalize_name(f"{first[0]}{tail}"))
                    keys.append(normalize_name(f"{first[0]}.{tail}"))
    if full:
        keys.append(normalize_name(full))
    if first and second:
        keys.append(normalize_name(f"{first[0]}{second}"))
        keys.append(normalize_name(f"{first[0]}.{second}"))
    # Dedup preserving order.
    out: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def player_scan_aliases(player: dict[str, Any]) -> list[str]:
    """Human-readable aliases used to detect a player mention in free text."""
    names: list[str] = []
    web = str(player.get("web_name") or "").strip()
    first = str(player.get("first_name") or "").strip()
    second = str(player.get("second_name") or "").strip()
    if web:
        names.append(web)
        if " " in web:
            names.append(web.split()[-1])
    if second:
        names.append(second)
        if " " in second:
            tail = second.split()[-1]
            if len(normalize_name(tail)) >= 4:
                names.append(tail)
                if first:
                    names.append(f"{first} {tail}")
                    names.append(f"{first[0]}. {tail}")
                    names.append(f"{first[0]}.{tail}")
    if first and second:
        names.append(f"{first} {second}")
        names.append(f"{first[0]}. {second}")
        names.append(f"{first[0]}.{second}")
    uniq: list[str] = []
    seen: set[str] = set()
    for n in sorted(names, key=len, reverse=True):
        key = n.lower()
        if key in seen or len(normalize_name(n)) < 3:
            continue
        seen.add(key)
        uniq.append(n)
    return uniq


def build_player_indexes(players: list[dict[str, Any]]) -> dict[str, Any]:
    """Build lookup maps for web_name / surname / full name."""
    by_id: dict[int, dict[str, Any]] = {}
    by_web: dict[str, list[int]] = {}
    by_norm: dict[str, list[int]] = {}

    for p in players:
        pid = int(p["id"])
        by_id[pid] = p
        web = str(p.get("web_name") or "").strip()
        if web:
            by_web.setdefault(web.lower(), []).append(pid)
        for key in player_name_keys(p):
            bucket = by_norm.setdefault(key, [])
            if pid not in bucket:
                bucket.append(pid)

    return {"by_id": by_id, "by_web": by_web, "by_norm": by_norm}


def resolve_player_name(
    name: str,
    indexes: dict[str, Any],
    *,
    team_short: str | None = None,
    candidate_ids: set[int] | None = None,
    article_text: str | None = None,
) -> dict[str, Any] | None:
    """Resolve a free-text name to a single FPL player when unambiguous.

    Optional `team_short` / `candidate_ids` (usually KG players mentioned in the
    article) are used to break surname ties. `article_text` can also break ties
    via club nicknames (e.g. "Man Utd" → Bruno Fernandes over Mateus).
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
            # Strict: an explicit team hint must not keep the wrong club's player.
            out = [
                p
                for p in out
                if str(by_id[p].get("team_short") or "").strip().upper() == want
            ]
        return out

    def _disambiguate(pids: list[int]) -> list[int]:
        out = list(dict.fromkeys(pids))
        if len(out) <= 1 or not article_text:
            return out
        club_hits = [p for p in out if club_hint_in_text(article_text, by_id[p])]
        if len(club_hits) == 1:
            return club_hits
        if club_hits:
            return club_hits
        # Prefer a longer first+surname form that actually appears in the article.
        specific: list[int] = []
        for pid in out:
            p = by_id[pid]
            first = str(p.get("first_name") or "").strip()
            second = str(p.get("second_name") or "").strip()
            aliases = []
            if first and second:
                aliases.append(f"{first} {second}")
                if " " in second:
                    aliases.append(f"{first} {second.split()[-1]}")
            web = str(p.get("web_name") or "").strip()
            if web and ("." in web or " " in web):
                aliases.append(web)
            if any(alias_in_text(article_text, a) for a in aliases if len(normalize_name(a)) >= 6):
                specific.append(pid)
        if len(specific) == 1:
            return specific
        return out

    web_hits = _filter(list(by_web.get(raw.lower()) or []))
    if len(web_hits) == 1:
        pid = web_hits[0]
        # Exact web_name (Mateus "Fernandes") must not beat a same-surname rival
        # when the article club cue points elsewhere (Man Utd → Bruno).
        if not article_text or club_hint_in_text(article_text, by_id[pid]):
            return by_id[pid]

    key = normalize_name(raw)
    candidates = _disambiguate(
        _filter(
            list(dict.fromkeys((by_web.get(raw.lower()) or []) + (by_norm.get(key) or [])))
        )
    )
    if len(candidates) == 1:
        return by_id[candidates[0]]

    # Surname fallback: last token.
    parts = [p for p in re.split(r"\s+", raw) if p]
    if parts:
        surname_key = normalize_name(parts[-1])
        surname_hits = _disambiguate(_filter(list(by_norm.get(surname_key) or [])))
        if len(surname_hits) == 1:
            return by_id[surname_hits[0]]
    return None


def candidate_web_names_for_text(text: str, indexes: dict[str, Any], *, limit: int = 40) -> list[str]:
    """Cheap prefilter: web_names that appear as whole tokens in the article text."""
    blob = text or ""
    hits: list[str] = []
    seen: set[int] = set()
    by_id: dict[int, dict[str, Any]] = indexes["by_id"]
    # Prefer longer aliases first so "Bruno Fernandes" / "João Pedro" beat bare surnames.
    ranked: list[tuple[str, list[int]]] = []
    for norm, pids in indexes["by_norm"].items():
        if len(norm) < 3:
            continue
        scan_aliases: list[str] = []
        for pid in pids:
            player = by_id[pid]
            for alias in player_scan_aliases(player):
                alias_norm = normalize_name(alias)
                if alias_norm == norm or (
                    " " in alias
                    and normalize_name(alias.split()[-1]) == norm
                    and len(norm) >= 4
                ):
                    scan_aliases.append(alias)
        if not scan_aliases:
            continue
        # Longest alias for ranking; keep every pid sharing this norm key so
        # surname collisions (Fernandes → Bruno + Mateus) stay disambiguatable.
        scan = max(scan_aliases, key=len)
        ranked.append((scan, list(dict.fromkeys(pids))))
    ranked.sort(key=lambda t: len(t[0]), reverse=True)

    for scan, pids in ranked:
        matched = alias_in_text(blob, scan)
        if not matched and " " in scan:
            tail = scan.split()[-1]
            matched = len(normalize_name(tail)) >= 4 and alias_in_text(blob, tail)
        if not matched:
            continue
        for pid in pids:
            if pid in seen:
                continue
            seen.add(pid)
            w = str(by_id[pid].get("web_name") or "")
            if w:
                hits.append(w)
            if len(hits) >= limit:
                return hits
    return hits
