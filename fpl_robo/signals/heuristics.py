"""Deterministic title/snippet heuristics to seed high-certainty signals."""

from __future__ import annotations

import re
from typing import Any

from fpl_robo.signals.match import normalize_name

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
    for match in _CLUB_SIGNS.finditer(text):
        if _is_rumour_only(match.group(0)):
            continue
        player = _lookup_name(match.group("name"), by_norm, kind="transfer", title=text)
        if player is None:
            continue
        club = match.group("club")
        item = _signal_for_player(
            player,
            kind="transfer",
            severity=0.12,
            kw=f"{club} signs",
            title=text,
        )
        item["destination_hint"] = club
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
        key = (item.get("web_name"), item.get("team_short"), item.get("type"))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(item)
    return uniq
