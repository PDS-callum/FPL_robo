"""Deterministic title/snippet heuristics to seed high-certainty signals."""

from __future__ import annotations

import re
from typing import Any

from fpl_robo.signals.match import alias_in_text, alias_match_spans, normalize_name

# Capitalised person name with ≥1 lowercase letter (rejects FPL/BBC acronyms).
_NAME_TOKEN = (
    r"[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ.']*[a-zà-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ.']*"
    r"(?:\s+[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ.']*)?"
)
_NAME = rf"(?P<name>{_NAME_TOKEN})"

_TITLE_INJURY = re.compile(
    rf"\b{_NAME}\b.{{0,28}}\b(?P<kw>injur(?:y|ed)|doubt|sidelined|ruled out|hamstring|knock|months?\s+out|groin|ankle|knee)\b",
    re.UNICODE,
)
_TITLE_ROTATION = re.compile(
    rf"\b{_NAME}\b.{{0,28}}\b(?P<kw>absent|miss(?:es|ed)?(?:\s+out)?)\b",
    re.UNICODE,
)
_TITLE_SUSPEND = re.compile(
    rf"\b{_NAME}\b.{{0,28}}\b(?P<kw>suspend(?:ed|sion)?|banned|ban)\b",
    re.UNICODE,
)
# Player … signs/joins / completes move
_TITLE_TRANSFER = re.compile(
    rf"\b{_NAME}\b.{{0,40}}\b(?P<kw>signs?(?:\s+for)?|joins?|has\s+joined|completes?\s+(?:move|deal|signing)|sealed?\s+(?:a\s+)?deal|unveiled)\b",
    re.UNICODE,
)
# Player leaves / sold to / departs (outbound / exit)
_TITLE_EXIT = re.compile(
    rf"\b{_NAME}\b.{{0,48}}\b(?P<kw>leaves?\s+for|left\s+for|departs?(?:\s+for)?|sold\s+to|loaned\s+to|"
    rf"completes?\s+(?:saudi|mls)\s+(?:move|switch|deal)|moves?\s+abroad)\b",
    re.UNICODE | re.IGNORECASE,
)
# Player joins Destination (capture destination for PL vs non-PL).
_EXIT_TO_CLUB = re.compile(
    rf"\b{_NAME}\b.{{0,28}}\b(?:has\s+)?(?:joined|joins|signed\s+for|signs\s+for|moves?\s+to|sold\s+to|loaned\s+to)\s+"
    rf"(?P<club>{_NAME_TOKEN}(?:\s+(?:United|City|Town|Madrid|Munich|Hilal|Nassr|Ittihad|Ahli))?)\b",
    re.UNICODE,
)
# Club signs [role] Player
_CLUB_SIGNS = re.compile(
    rf"(?P<club>{_NAME_TOKEN})\s+signs?\s+"
    rf"(?:(?:the\s+)?(?:goalkeeper|striker|midfielder|defender|winger|forward|keeper)\s+)?"
    rf"{_NAME}\b",
    re.UNICODE,
)
# Player … after [Club] transfer / Chelsea transfer
_AFTER_TRANSFER = re.compile(
    rf"\b{_NAME}\b.{{0,90}}\b(?P<kw>after\s+(?:an?\s+)?(?:[A-Za-z]+\s+)?transfer|transfer\s+to|moved\s+to)\b",
    re.UNICODE | re.IGNORECASE,
)
_TITLE_FORM = re.compile(
    rf"\b{_NAME}(?:['\u2019]s)?\b.{{0,36}}\b(?P<kw>hat[- ]?trick|brace|scored|scores|goals?|debut|on target|lively|sharp|winner|impressive)\b",
    re.UNICODE,
)
# Solanke + Fernandes missed out / A, B absent — often preseason rest, not injury.
_MULTI_SHARED = re.compile(
    rf"(?P<names>{_NAME_TOKEN}(?:\s*[+&,]\s*{_NAME_TOKEN})+)\s+"
    rf"(?P<kw>missed?\s+out|absent|ruled out|injur(?:y|ed)|doubt)",
    re.UNICODE,
)
# Club completes deal for Player / signing of Player
_DEAL_FOR = re.compile(
    rf"\b(?:complete(?:s|d)?\s+(?:a\s+)?deal\s+for|signing\s+of|signed?)\s+"
    rf"(?:(?:the\s+)?(?:goalkeeper|striker|midfielder|defender|winger|forward)\s+)?"
    rf"{_NAME}\b",
    re.UNICODE | re.IGNORECASE,
)
# Player joins/signs for Club (destination captured).
_JOINED_CLUB = re.compile(
    rf"\b{_NAME}\b.{{0,28}}\b(?:has\s+)?(?:joined|joins|signed\s+for|signs\s+for|moves?\s+to)\s+"
    rf"(?P<club>{_NAME_TOKEN}(?:\s+(?:United|City|Town|Hotspur|Forest|Palace|Villa))?)\b",
    re.UNICODE,
)
# Player completes Club move / seals Club transfer.
_COMPLETES_CLUB_MOVE = re.compile(
    rf"\b{_NAME}\b.{{0,36}}\b(?:completes?|seals?|finalises?|finalizes?)\s+"
    rf"(?:his\s+|a\s+)?"
    rf"(?P<club>{_NAME_TOKEN}(?:\s+(?:United|City|Town|Hotspur|Forest|Palace|Villa))?)\s+"
    rf"(?:move|transfer|switch|deal)\b",
    re.UNICODE | re.IGNORECASE,
)

_CLAUSE_SPLIT = re.compile(r"\s*[,+/|]\s*|\s+-\s+|\s+–\s+")
_RUMOUR_KW = ("linked with", "interested in", "set to bid", "rumour", "hold talks", "talks with")
_CONFIRM_KW = ("signed", "signs", "joins", "joined", "completes", "confirmed", "unveiled", "after")


def _title_clauses(text: str) -> list[str]:
    """Split compound headlines so name↔keyword proximity stays local."""
    parts: list[str] = []
    for chunk in re.split(r":\s*", text or "", maxsplit=1):
        parts.extend(p.strip() for p in _CLAUSE_SPLIT.split(chunk) if p and p.strip())
    return parts or [text]


def _pick_player(
    hits: list[dict[str, Any]],
    *,
    kind: str,
    title: str = "",
) -> dict[str, Any] | None:
    """Disambiguate surname collisions using KG availability / article context."""
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
    blob = (title or "").lower()
    team_hits = []
    for p in hits:
        short = str(p.get("team_short") or "").strip().lower()
        name = str(p.get("team_name") or "").strip().lower()
        if (short and short in blob) or (name and len(name) >= 4 and name in blob):
            team_hits.append(p)
    if len(team_hits) == 1:
        return team_hits[0]
    if kind in {"form", "transfer", "positive"}:
        ranked = sorted(
            hits,
            key=lambda p: (
                float(p.get("selected_by_percent") or 0.0),
                float(p.get("ep_next") or p.get("ep_this") or 0.0),
            ),
            reverse=True,
        )
        if ranked and float(ranked[0].get("selected_by_percent") or 0) > 0:
            return ranked[0]
    return None


def _index_players(candidate_players: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_norm: dict[str, list[dict[str, Any]]] = {}
    for p in candidate_players:
        keys = [
            normalize_name(str(p.get("web_name") or "")),
            normalize_name(str(p.get("second_name") or "")),
            normalize_name(f"{p.get('first_name') or ''} {p.get('second_name') or ''}"),
        ]
        first = str(p.get("first_name") or "").strip()
        second = str(p.get("second_name") or "").strip()
        web = str(p.get("web_name") or "").strip()
        if first and second:
            keys.append(normalize_name(f"{first[0]} {second}"))
            keys.append(normalize_name(f"{first[0]}.{second}"))
        if web and " " in web:
            keys.append(normalize_name(web.split()[-1]))
        for key in keys:
            if not key:
                continue
            bucket = by_norm.setdefault(key, [])
            if p not in bucket:
                bucket.append(p)
    return by_norm


def _lookup_name(
    name: str,
    by_norm: dict[str, list[dict[str, Any]]],
    *,
    kind: str,
    title: str,
) -> dict[str, Any] | None:
    key = normalize_name(name)
    hits = by_norm.get(key) or []
    if not hits and " " in name:
        hits = by_norm.get(normalize_name(name.split()[-1])) or []
    return _pick_player(hits, kind=kind, title=title)


def _signal_for_player(
    player: dict[str, Any],
    *,
    kind: str,
    severity: float,
    kw: str,
    title: str,
) -> dict[str, Any]:
    status = str(player.get("status") or "a").lower()
    sev = severity
    exclude = False
    if kind in {"injury", "suspension"} and (
        status in {"i", "s"}
        or (
            player.get("chance_of_playing_next_round") is not None
            and float(player.get("chance_of_playing_next_round") or 100) <= 25
        )
    ):
        sev = max(sev, 0.75)
        exclude = status in {"i", "s"}
    first = str(player.get("first_name") or "").strip()
    web = str(player.get("web_name") or "").strip()
    return {
        "web_name": web,
        "first_name": first or None,
        "second_name": player.get("second_name"),
        "team_short": player.get("team_short"),
        "element_type": player.get("element_type"),
        "position": player.get("position"),
        "type": kind,
        "severity": sev,
        "exclude": exclude,
        "evidence": title,
        "note": f"[heuristic] {kw} in headline → {first} {web}".strip(),
        "source": "heuristic",
    }


def _signal_for_match(
    *,
    match: re.Match[str],
    kind: str,
    severity: float,
    by_norm: dict[str, list[dict[str, Any]]],
    title: str,
) -> dict[str, Any] | None:
    name = match.group("name")
    player = _lookup_name(name, by_norm, kind=kind, title=title)
    if player is None:
        return None
    return _signal_for_player(
        player,
        kind=kind,
        severity=severity,
        kw=str(match.groupdict().get("kw") or kind),
        title=title,
    )


def _is_rumour_only(blob: str) -> bool:
    lower = blob.lower()
    if any(k in lower for k in _RUMOUR_KW):
        return not any(k in lower for k in _CONFIRM_KW)
    return False


def heuristic_signals_from_title(
    title: str,
    candidate_players: list[dict[str, Any]],
    *,
    summary: str = "",
) -> list[dict[str, Any]]:
    """Seed signals when a headline clearly pairs a candidate with news language."""
    text = (title or "").strip()
    if summary:
        clean = summary.split("The post ")[0].split("appeared first on")[0].strip()
        if clean:
            text = f"{text}. {clean[:220]}"
    if not text or not candidate_players:
        return []

    out: list[dict[str, Any]] = []
    by_norm = _index_players(candidate_players)

    # Multi-name shared keywords on the full headline (before +/, splits).
    for match in _MULTI_SHARED.finditer(text):
        kw = match.group("kw").lower()
        # Preseason "missed out" / bare absent → rotation, not hard injury.
        if any(k in kw for k in ("missed out", "miss out", "absent")) and "injur" not in kw:
            kind, severity = "rotation", 0.25
        elif "doubt" in kw:
            kind, severity = "injury", 0.4
        else:
            kind, severity = "injury", 0.55
        names = re.split(r"\s*[+&,]\s*", match.group("names"))
        for raw_name in names:
            player = _lookup_name(raw_name.strip(), by_norm, kind=kind, title=text)
            if player is None:
                continue
            out.append(
                _signal_for_player(player, kind=kind, severity=severity, kw=kw, title=text)
            )

    # complete deal for / signing of Player
    for match in _DEAL_FOR.finditer(text):
        if _is_rumour_only(match.group(0)):
            continue
        player = _lookup_name(match.group("name"), by_norm, kind="transfer", title=text)
        if player is None:
            continue
        out.append(
            _signal_for_player(
                player,
                kind="transfer",
                severity=0.12,
                kw="deal/signing",
                title=text,
            )
        )

    # Club signs Player (e.g. Ipswich sign goalkeeper Scherpen).
    for clause in _title_clauses(text):
        for match in _CLUB_SIGNS.finditer(clause):
            if _is_rumour_only(match.group(0)):
                continue
            player = _lookup_name(match.group("name"), by_norm, kind="transfer", title=text)
            if player is None:
                continue
            club = match.group("club")
            if str(club).lower() in {"transfer", "premier", "record"}:
                continue
            item = _signal_for_player(
                player,
                kind="transfer",
                severity=0.12,
                kw=f"{club} signs",
                title=text,
            )
            item["destination_hint"] = club
            out.append(item)

    # Player joins / signs for Club — scan clauses so "Transfer news: X joins Y"
    # does not capture name="Transfer".
    for clause in _title_clauses(text):
        for match in _JOINED_CLUB.finditer(clause):
            if _is_rumour_only(match.group(0)):
                continue
            player = _lookup_name(match.group("name"), by_norm, kind="transfer", title=text)
            if player is None:
                continue
            club = match.group("club")
            if str(club).lower() in {"transfer", "premier", "record"}:
                continue
            item = _signal_for_player(
                player,
                kind="transfer",
                severity=0.15,
                kw=f"joins {club}",
                title=text,
            )
            item["destination_hint"] = club
            out.append(item)

    # Player completes Club move.
    for clause in _title_clauses(text):
        for match in _COMPLETES_CLUB_MOVE.finditer(clause):
            if _is_rumour_only(match.group(0)):
                continue
            player = _lookup_name(match.group("name"), by_norm, kind="transfer", title=text)
            if player is None:
                continue
            club = match.group("club")
            item = _signal_for_player(
                player,
                kind="transfer",
                severity=0.15,
                kw=f"completes {club} move",
                title=text,
            )
            item["destination_hint"] = club
            out.append(item)

    # Explicit exit / sold / loaned abroad.
    for match in _TITLE_EXIT.finditer(text):
        if _is_rumour_only(match.group(0)):
            continue
        player = _lookup_name(match.group("name"), by_norm, kind="transfer", title=text)
        if player is None:
            continue
        item = _signal_for_player(
            player,
            kind="transfer",
            severity=0.95,
            kw=match.group("kw"),
            title=text,
        )
        item["exclude"] = True
        item["outbound"] = True
        # Try to capture destination from the same clause.
        for dest_m in _EXIT_TO_CLUB.finditer(match.group(0)):
            item["destination_hint"] = dest_m.group("club")
            break
        out.append(item)

    # Player after Club transfer.
    for match in _AFTER_TRANSFER.finditer(text):
        if _is_rumour_only(text[max(0, match.start() - 20) : match.end() + 20]):
            continue
        player = _lookup_name(match.group("name"), by_norm, kind="transfer", title=text)
        if player is None:
            continue
        out.append(
            _signal_for_player(
                player,
                kind="transfer",
                severity=0.12,
                kw=match.group("kw"),
                title=text,
            )
        )

    patterns = (
        (_TITLE_INJURY, "injury", 0.55),
        (_TITLE_ROTATION, "rotation", 0.25),
        (_TITLE_SUSPEND, "suspension", 0.7),
        (_TITLE_TRANSFER, "transfer", 0.15),
        (_TITLE_FORM, "form", 0.1),
    )
    for cre, kind, severity in patterns:
        for clause in _title_clauses(text):
            for match in cre.finditer(clause):
                if kind == "transfer" and _is_rumour_only(clause):
                    continue
                item = _signal_for_match(
                    match=match,
                    kind=kind,
                    severity=severity,
                    by_norm=by_norm,
                    title=text,
                )
                if item is not None:
                    out.append(item)

    seen: set[tuple] = set()
    uniq: list[dict[str, Any]] = []
    for item in out:
        key = (item.get("web_name"), item.get("team_short"), item.get("type"), item.get("outbound"))
        if key in seen:
            # Prefer the row that already captured a destination club.
            if item.get("destination_hint"):
                for i, prev in enumerate(uniq):
                    if (
                        prev.get("web_name") == item.get("web_name")
                        and prev.get("team_short") == item.get("team_short")
                        and prev.get("type") == item.get("type")
                        and prev.get("outbound") == item.get("outbound")
                        and not prev.get("destination_hint")
                    ):
                        uniq[i] = item
                        break
            continue
        seen.add(key)
        uniq.append(item)

    # Backfill destination from evidence when the join regex missed it.
    from fpl_robo.signals.transfer_infer import extract_destination_hint, looks_like_outbound

    for item in uniq:
        if str(item.get("type") or "") != "transfer":
            continue
        if not item.get("destination_hint"):
            hint = extract_destination_hint(
                str(item.get("evidence") or ""),
                str(item.get("note") or ""),
                title,
                summary,
            )
            if hint:
                item["destination_hint"] = hint
        if item.get("outbound"):
            item["exclude"] = True
            item["severity"] = max(float(item.get("severity") or 0), 0.9)
            continue
        if looks_like_outbound(
            evidence=str(item.get("evidence") or item.get("note") or ""),
            note=str(item.get("note") or ""),
            destination_hint=str(item.get("destination_hint") or "") or None,
        ):
            item["exclude"] = True
            item["outbound"] = True
            item["severity"] = max(float(item.get("severity") or 0), 0.9)
    return uniq


# Role / minutes wording near a player name → predicted lineup rows (no LLM).
_ROLE_RULES: list[tuple[str, str, int, float]] = [
    # phrase, role, predicted_minutes, confidence
    ("not expected to start", "bench", 20, 0.82),
    ("unlikely to start", "bench", 20, 0.8),
    ("won't start", "bench", 15, 0.78),
    ("will not start", "bench", 15, 0.78),
    ("expected to start", "start", 80, 0.84),
    ("set to start", "start", 85, 0.82),
    ("likely to start", "start", 80, 0.8),
    ("tipped to start", "start", 80, 0.78),
    ("should start", "start", 80, 0.76),
    ("assured start", "start", 88, 0.82),
    ("regular starter", "start", 85, 0.8),
    ("regular start", "start", 85, 0.78),
    ("first-choice", "start", 88, 0.82),
    ("first choice", "start", 88, 0.82),
    ("nailed on", "start", 90, 0.85),
    ("lead the line", "start", 85, 0.78),
    ("leads the line", "start", 85, 0.78),
    ("on the bench", "bench", 20, 0.8),
    ("from the bench", "bench", 20, 0.76),
    ("off the bench", "bench", 25, 0.74),
    ("impact sub", "bench", 25, 0.74),
    ("cameo", "bench", 20, 0.72),
    ("left out", "bench", 10, 0.8),
    ("limited minutes", "doubt", 40, 0.78),
    ("fewer minutes", "doubt", 45, 0.74),
    ("share minutes", "doubt", 50, 0.8),
    ("sharing minutes", "doubt", 50, 0.8),
    ("competition for places", "doubt", 55, 0.72),
    ("fight for his place", "doubt", 55, 0.74),
    ("fight for a place", "doubt", 55, 0.74),
    ("understudy", "bench", 25, 0.78),
    ("backup", "bench", 25, 0.76),
    ("back-up", "bench", 25, 0.76),
    ("cover for", "bench", 30, 0.74),
    ("support for", "bench", 35, 0.72),
    ("in support of", "bench", 35, 0.72),
    ("deputis", "bench", 30, 0.74),
    ("squad depth", "bench", 25, 0.7),
    ("deepen the squad", "bench", 25, 0.7),
    ("depth option", "bench", 25, 0.72),
    ("veteran", "bench", 25, 0.72),
    ("35-year-old", "bench", 25, 0.7),
    ("34-year-old", "bench", 25, 0.7),
    ("33-year-old", "bench", 25, 0.7),
    ("increase depth", "bench", 20, 0.74),
    ("add depth", "bench", 20, 0.74),
    ("squad option", "bench", 25, 0.72),
    ("to support", "bench", 30, 0.7),
    ("behind ", "bench", 30, 0.68),
    ("rotated", "doubt", 45, 0.74),
    ("rotation", "doubt", 50, 0.7),
    # Explicit XI / availability — not match-report "scores/hat-trick" noise.
    ("starting xi", "start", 85, 0.74),
    ("starting 11", "start", 85, 0.74),
    ("in the xi", "start", 85, 0.72),
    ("in the 11", "start", 85, 0.72),
    ("rested", "bench", 20, 0.74),
    ("given a rest", "bench", 15, 0.76),
    ("cup tied", "bench", 0, 0.8),
    ("suspended", "bench", 0, 0.85),
]


def _club_hint_in_text(text: str, player: dict[str, Any]) -> bool:
    """True if the player's club short/name appears in the article text."""
    lower = (text or "").lower()
    short = str(player.get("team_short") or "").strip()
    name = str(player.get("team_name") or "").strip()
    # Word-bound short codes ("LIV" must not match inside "deliver").
    if short and len(short) >= 3 and alias_in_text(text or "", short):
        return True
    if name and len(name) >= 4 and name.lower() in lower:
        return True
    # Common stubs without relying on the full alias map.
    stubs = {
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
    for stub in stubs.get(short.upper(), ()):
        if stub in lower:
            return True
    return False


def _player_aliases(player: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for key in ("web_name", "second_name"):
        val = str(player.get(key) or "").strip()
        if val:
            names.append(val)
    first = str(player.get("first_name") or "").strip()
    second = str(player.get("second_name") or "").strip()
    if first and second:
        names.append(f"{first} {second}")
        names.append(f"{first[0]}. {second}")
        names.append(f"{first[0]}.{second}")
    # Longest first so "João Pedro" wins over "Pedro".
    uniq: list[str] = []
    seen: set[str] = set()
    for n in sorted(names, key=len, reverse=True):
        key = n.lower()
        if key in seen or len(n) < 3:
            continue
        seen.add(key)
        uniq.append(n)
    return uniq


def heuristic_lineup_signals_from_text(
    text: str,
    candidate_players: list[dict[str, Any]],
    *,
    prefer_ids: set[int] | None = None,
    source: str = "heuristic_lineup",
    gameweek: int | None = None,
    article_title: str = "",
) -> list[dict[str, Any]]:
    """Extract start/bench/doubt (+ minutes) from role wording near player names."""
    blob = text or ""
    if not blob or not candidate_players:
        return []
    lower = blob.lower()
    prefer_ids = prefer_ids or set()

    # Rank owned players first so surname collisions favour the squad.
    ordered = sorted(
        candidate_players,
        key=lambda p: (0 if int(p.get("id") or 0) in prefer_ids else 1, -len(str(p.get("web_name") or ""))),
    )

    # Alias → how many candidates share it (case-insensitive). Ambiguous names need a club hint.
    alias_owners: dict[str, set[int]] = {}
    for player in ordered:
        pid = int(player["id"])
        for alias in _player_aliases(player):
            alias_owners.setdefault(alias.lower(), set()).add(pid)

    out: list[dict[str, Any]] = []
    claimed: set[int] = set()
    window = 70  # chars either side of a name mention — keep cues local to the player

    negative_start = (
        "not expected to start",
        "unlikely to start",
        "won't start",
        "will not start",
        "on the bench",
        "from the bench",
        "left out",
    )

    for player in ordered:
        pid = int(player["id"])
        if pid in claimed:
            continue
        aliases = _player_aliases(player)
        best: dict[str, Any] | None = None
        for alias in aliases:
            ambiguous = len(alias_owners.get(alias.lower(), {pid})) > 1
            # Ambiguous surnames (Gabriel, etc.) need a club cue; unique names do not.
            if ambiguous and not _club_hint_in_text(blob, player):
                continue
            # Word-boundary match — never "Raya" inside "Rayan".
            for start, end in alias_match_spans(blob, alias):
                ctx = lower[max(0, start - window) : min(len(lower), end + window)]
                hits: list[tuple[float, int, dict[str, Any]]] = []
                for phrase, role, mins, conf in _ROLE_RULES:
                    if phrase not in ctx:
                        continue
                    cand = {
                        "player_id": pid,
                        "web_name": player.get("web_name"),
                        "first_name": player.get("first_name"),
                        "second_name": player.get("second_name"),
                        "team_short": player.get("team_short"),
                        "role": role,
                        "predicted_minutes": mins,
                        "confidence": conf,
                        "gameweek": gameweek,
                        "evidence": ctx.strip()[:220],
                        "note": f"[{source}] {role} (~{mins}m): {player.get('web_name')}",
                        "source": source,
                        "article_title": article_title,
                    }
                    # Longer phrases first, then confidence.
                    hits.append((conf, len(phrase), cand))
                if not hits:
                    continue
                hits.sort(key=lambda t: (t[0], t[1]), reverse=True)
                cand = hits[0][2]
                roles = {h[2]["role"] for h in hits}
                if any(p in ctx for p in negative_start):
                    # Explicit non-start cue wins over a neighbouring "expected to start".
                    bench_hit = next(
                        (h[2] for h in hits if h[2]["role"] in {"bench", "doubt"}),
                        None,
                    )
                    if bench_hit is not None:
                        cand = bench_hit
                elif "start" in roles and roles & {"bench", "doubt"}:
                    # Start + minutes risk in the same clause → start with haircut.
                    lower_mins = min(int(h[2]["predicted_minutes"]) for h in hits)
                    start_hit = next(h[2] for h in hits if h[2]["role"] == "start")
                    cand = dict(start_hit)
                    cand["predicted_minutes"] = min(
                        int(start_hit["predicted_minutes"]), max(45, lower_mins)
                    )
                    cand["note"] = (
                        f"[{source}] start (~{cand['predicted_minutes']}m, minutes risk): "
                        f"{player.get('web_name')}"
                    )
                    cand["role"] = "start"
                if best is None or float(cand["confidence"]) > float(best["confidence"]):
                    best = cand
            if best is not None:
                break
        if best is None:
            continue
        claimed.add(pid)
        out.append(best)

    # Soft path: owned players named in the article with no role phrase still get a
    # mention row so they don't wait solely on the FPL proxy fill.
    if prefer_ids:
        from fpl_robo.signals.minutes_estimate import implied_minutes_point

        for player in ordered:
            pid = int(player["id"])
            if pid not in prefer_ids or pid in claimed:
                continue
            # Need at least one FPL volume/EP cue — otherwise leave to fpl_implied fill.
            if (
                player.get("ep_next") in (None, "")
                and not player.get("minutes")
                and not player.get("starts")
            ):
                continue
            # Soft mentions always need a club hint (blocks Arsenal Gabriel on Man Utd news).
            if not _club_hint_in_text(blob, player):
                continue
            aliases = _player_aliases(player)
            named = False
            for alias in aliases:
                ambiguous = len(alias_owners.get(alias.lower(), {pid})) > 1
                if ambiguous and len(alias) <= 8 and not _club_hint_in_text(blob, player):
                    continue
                if alias_in_text(blob, alias):
                    named = True
                    break
            if not named:
                continue
            mins, role, _ = implied_minutes_point(
                player.get("ep_next"),
                player.get("chance_of_playing_next_round"),
                player.get("status"),
                player.get("minutes"),
                starts=player.get("starts"),
            )
            conf = 0.67  # above default min_confidence; below strong role phrases
            out.append(
                {
                    "player_id": pid,
                    "web_name": player.get("web_name"),
                    "first_name": player.get("first_name"),
                    "second_name": player.get("second_name"),
                    "team_short": player.get("team_short"),
                    "role": role,
                    "predicted_minutes": mins,
                    "confidence": conf,
                    "gameweek": gameweek,
                    "evidence": f"named in article; FPL-soft minutes (~{mins}m)",
                    "note": f"[heuristic_mention] {role} (~{mins}m): {player.get('web_name')}",
                    "source": "heuristic_mention",
                    "article_title": article_title,
                }
            )
            claimed.add(pid)

    return out
