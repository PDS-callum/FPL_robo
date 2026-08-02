"""Implied next-GW minutes for squad coverage when news has gaps.

Also applies club-depth competition so backups behind a clear starter
(e.g. Welbeck behind João Pedro) do not inherit prior-club starter minutes.
"""

from __future__ import annotations

from typing import Any

from fpl_robo.config import SIGNAL_INGEST

_NEWS_SOURCES = {
    "heuristic_lineup",
    "heuristic_mention",
    "ollama_lineup",
    "heuristic",
    "ollama",
    "auto",
    "transfer_lineup",
    "depth_model",
}

# Map common destination strings → FPL team_short.
_DEST_TO_SHORT: dict[str, str] = {
    "arsenal": "ARS",
    "chelsea": "CHE",
    "liverpool": "LIV",
    "manchester city": "MCI",
    "man city": "MCI",
    "manchester united": "MUN",
    "man united": "MUN",
    "man utd": "MUN",
    "tottenham": "TOT",
    "spurs": "TOT",
    "newcastle": "NEW",
    "aston villa": "AVL",
    "villa": "AVL",
    "brighton": "BHA",
    "west ham": "WHU",
    "fulham": "FUL",
    "crystal palace": "CRY",
    "palace": "CRY",
    "everton": "EVE",
    "brentford": "BRE",
    "bournemouth": "BOU",
    "wolves": "WOL",
    "wolverhampton": "WOL",
    "nottingham forest": "NFO",
    "nottingham": "NFO",
    "forest": "NFO",
    "burnley": "BUR",
    "leeds": "LEE",
    "sunderland": "SUN",
}


def destination_to_team_short(hint: str | None) -> str | None:
    """Resolve a free-text destination club to an FPL short code."""
    raw = str(hint or "").strip()
    if not raw:
        return None
    upper = raw.upper()
    if len(upper) <= 3 and upper.isalpha():
        return upper
    lower = raw.lower()
    if lower in _DEST_TO_SHORT:
        return _DEST_TO_SHORT[lower]
    for name, short in sorted(_DEST_TO_SHORT.items(), key=lambda kv: -len(kv[0])):
        if name in lower:
            return short
    return None


def implied_minutes_point(
    ep_next: Any,
    chance: Any,
    status: Any,
    hist_minutes: Any,
    *,
    starts: Any = None,
    ignore_hist_minutes: bool = False,
) -> tuple[int, str, float]:
    """Return (predicted_minutes, role, confidence) from FPL public signals.

    FPL does not publish official predicted minutes; this is a conservative
    point estimate so owned players never lack a minutes figure.
    """
    st = str(status or "a").lower()
    conf = float(SIGNAL_INGEST.get("fpl_implied_minutes_confidence") or 0.66)
    if st in {"u", "s"}:
        return 0, "bench", conf
    if st == "i":
        return 10, "bench", conf
    try:
        ch = float(chance) if chance is not None and chance != "" else None
    except (TypeError, ValueError):
        ch = None
    if ch is not None and ch <= 25:
        return 15, "bench", conf
    if ch is not None and ch <= 50:
        return 40, "doubt", conf

    try:
        ep = float(ep_next) if ep_next is not None and ep_next != "" else 0.0
    except (TypeError, ValueError):
        ep = 0.0
    try:
        mins = 0 if ignore_hist_minutes else int(hist_minutes or 0)
    except (TypeError, ValueError):
        mins = 0
    try:
        st_n = 0 if ignore_hist_minutes else int(starts or 0)
    except (TypeError, ValueError):
        st_n = 0

    # Prior-season volume + EP are the main cold-start proxies.
    # When ignore_hist_minutes (recent inbound transfer), trust EP only so
    # prior-club starter volume does not invent 90m at the new club.
    if ep >= 4.0 or mins >= 2500 or st_n >= 30:
        return 90, "start", conf
    if ep >= 3.0 or mins >= 1500 or st_n >= 20:
        return 85, "start", conf
    if ep >= 2.0 or mins >= 600 or st_n >= 8:
        return 65, "doubt", conf
    if ep >= 1.5 or mins >= 90 or st_n >= 2:
        return 45, "doubt", conf
    return 20, "bench", conf


def club_depth_minutes(
    player: dict[str, Any],
    *,
    elements: list[dict[str, Any]] | None = None,
    team_short: str | None = None,
    team_id: int | None = None,
    element_type: int | None = None,
) -> tuple[int, str, float, str] | None:
    """Haircut minutes from same-club same-position competition.

    Rank 1 (by ep_next, then prior minutes) keeps a starter prior; rank 2+
    are treated as rotation/bench depth.
    Returns (minutes, role, confidence, note) or None if not enough peers.
    """
    if not elements:
        return None
    pos = element_type if element_type is not None else player.get("element_type")
    try:
        pos_i = int(pos) if pos is not None else None
    except (TypeError, ValueError):
        pos_i = None
    if pos_i is None:
        return None

    short = (team_short or player.get("team_short") or "").strip().upper() or None
    tid = team_id if team_id is not None else player.get("team_id")
    try:
        tid_i = int(tid) if tid is not None else None
    except (TypeError, ValueError):
        tid_i = None

    peers: list[dict[str, Any]] = []
    for el in elements:
        if not isinstance(el, dict) or el.get("id") is None:
            continue
        try:
            if int(el.get("element_type") or -1) != pos_i:
                continue
        except (TypeError, ValueError):
            continue
        el_tid = el.get("team") if el.get("team") is not None else el.get("team_id")
        el_short = str(el.get("team_short") or "").strip().upper()
        team_ok = False
        if tid_i is not None and el_tid is not None:
            try:
                team_ok = int(el_tid) == tid_i
            except (TypeError, ValueError):
                team_ok = False
        if not team_ok and short and el_short:
            team_ok = el_short == short
        if not team_ok:
            # Bootstrap elements use numeric team id only — resolve via short map
            # when caller passes team_short and elements lack team_short.
            if short and el_short == short:
                team_ok = True
            elif tid_i is not None and el_tid is not None:
                try:
                    team_ok = int(el_tid) == tid_i
                except (TypeError, ValueError):
                    team_ok = False
        if not team_ok:
            continue
        st = str(el.get("status") or "a").lower()
        if st in {"u", "s"}:
            continue
        peers.append(el)

    if len(peers) < 2:
        return None

    def _ep(el: dict[str, Any]) -> float:
        try:
            return float(el.get("ep_next") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _mins(el: dict[str, Any]) -> int:
        try:
            return int(el.get("minutes") or 0)
        except (TypeError, ValueError):
            return 0

    peers.sort(key=lambda el: (_ep(el), _mins(el), -int(el.get("now_cost") or 0)), reverse=True)
    pid = int(player["id"]) if player.get("id") is not None else None
    rank = None
    for i, el in enumerate(peers, start=1):
        if int(el["id"]) == pid:
            rank = i
            break

    conf = float(SIGNAL_INGEST.get("depth_minutes_confidence") or 0.72)
    leader = peers[0]
    leader_name = leader.get("web_name") or leader.get("second_name") or "club leader"

    # Recent transfers often still show the selling club in bootstrap — treat the
    # player as a new arrival behind the destination pecking order.
    if rank is None:
        return 25, "bench", conf, f"new arrival behind {leader_name} (~25m)"

    gap = _ep(leader) - _ep(next(el for el in peers if int(el["id"]) == pid))

    if rank == 1:
        mins = 85 if _ep(leader) >= 2.0 else 70
        return mins, "start", conf, f"depth rank1 ahead of peers (~{mins}m)"
    if rank == 2:
        # Clear gap to the focal starter → cameo / impact sub.
        if gap >= 0.3 or _mins(leader) >= 1500:
            return 25, "bench", conf, f"depth behind {leader_name} (~25m)"
        return 40, "doubt", conf, f"rotation behind {leader_name} (~40m)"
    # Rank 3+
    return 15, "bench", conf, f"depth rank{rank} behind {leader_name} (~15m)"


def fpl_fallback_row(
    player: dict[str, Any],
    element: dict[str, Any] | None = None,
    *,
    gameweek: int | None = None,
    elements: list[dict[str, Any]] | None = None,
    destination_team_short: str | None = None,
    destination_team_id: int | None = None,
    recent_transfer: bool = False,
) -> dict[str, Any]:
    el = element or player
    # Prefer club-depth model when we can see the destination pecking order.
    depth = club_depth_minutes(
        player,
        elements=elements,
        team_short=destination_team_short or player.get("team_short") or el.get("team_short"),
        team_id=destination_team_id
        if destination_team_id is not None
        else (player.get("team_id") or el.get("team") or el.get("team_id")),
        element_type=player.get("element_type") or el.get("element_type"),
    )
    if depth is not None and (
        recent_transfer
        or destination_team_short
        or float(SIGNAL_INGEST.get("prefer_depth_over_fpl_implied", True))
    ):
        mins, role, conf, why = depth
        src = "depth_model"
        evidence = why
    else:
        mins, role, conf = implied_minutes_point(
            el.get("ep_next"),
            el.get("chance_of_playing_next_round"),
            el.get("status"),
            el.get("minutes"),
            starts=el.get("starts"),
            ignore_hist_minutes=recent_transfer,
        )
        src = "fpl_implied"
        evidence = "FPL ep/status/minutes proxy"
        if recent_transfer:
            evidence += " (hist minutes ignored: recent transfer)"

    team_short = (
        destination_team_short
        or player.get("team_short")
        or el.get("team_short")
    )
    return {
        "player_id": int(player["id"]),
        "web_name": player.get("web_name"),
        "first_name": player.get("first_name") or el.get("first_name"),
        "second_name": player.get("second_name") or el.get("second_name"),
        "team_short": team_short,
        "role": role,
        "predicted_minutes": mins,
        "confidence": conf,
        "gameweek": gameweek,
        "evidence": evidence,
        "note": f"[{src}] {role} (~{mins}m): {player.get('web_name')}",
        "source": src,
    }


def _row_rank(row: dict[str, Any]) -> tuple:
    """Higher is better: has minutes, news/transfer over proxy, higher confidence."""
    src = str(row.get("source") or "")
    has_mins = 1 if row.get("predicted_minutes") is not None else 0
    # Prefer explicit news / transfer / depth over raw FPL volume proxy.
    if src in {"fpl_implied"}:
        quality = 0
    elif src in {"depth_model", "heuristic_mention"}:
        quality = 1
    elif src in {"transfer_lineup", "heuristic_lineup", "ollama_lineup"}:
        quality = 2
    else:
        quality = 1
    conf = float(row.get("confidence") or 0)
    # Prefer lower minutes when both are weak proxies? No — trust quality then conf.
    return (has_mins, quality, conf)


def ensure_owned_minutes_coverage(
    owned_players: list[dict[str, Any]],
    lineups: list[dict[str, Any]],
    *,
    elements_by_id: dict[int, dict[str, Any]] | None = None,
    elements: list[dict[str, Any]] | None = None,
    gameweek: int | None = None,
    transfer_destinations: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """Guarantee every owned player has a minutes row; news beats FPL implied."""
    if not owned_players:
        return [r for r in (lineups or []) if isinstance(r, dict)]

    elements_map = elements_by_id or {}
    all_elements = elements or list(elements_map.values())
    dest_map = transfer_destinations or {}
    owned_ids = {int(p["id"]) for p in owned_players}
    best_owned: dict[int, dict[str, Any]] = {}
    non_owned: list[dict[str, Any]] = []

    for row in lineups or []:
        if not isinstance(row, dict):
            continue
        pid = row.get("player_id")
        if pid is None:
            non_owned.append(row)
            continue
        pid_i = int(pid)
        if pid_i not in owned_ids:
            non_owned.append(row)
            continue
        prev = best_owned.get(pid_i)
        if prev is None or _row_rank(row) >= _row_rank(prev):
            best_owned[pid_i] = row

    owned_rows: list[dict[str, Any]] = []
    for p in owned_players:
        pid = int(p["id"])
        hit = best_owned.get(pid)
        if hit is not None and hit.get("predicted_minutes") is not None:
            owned_rows.append(hit)
            continue
        el = elements_map.get(pid) or p
        dest_short = dest_map.get(pid)
        dest_id = None
        if dest_short and all_elements:
            # Resolve team id from any peer with that short if present.
            for peer in all_elements:
                if str(peer.get("team_short") or "").upper() == dest_short:
                    dest_id = peer.get("team") or peer.get("team_id")
                    break
        owned_rows.append(
            fpl_fallback_row(
                p,
                el,
                gameweek=gameweek,
                elements=all_elements,
                destination_team_short=dest_short,
                destination_team_id=int(dest_id) if dest_id is not None else None,
                recent_transfer=bool(dest_short),
            )
        )

    return non_owned + owned_rows


def minutes_to_mult(predicted_minutes: int | float, confidence: float) -> float:
    """Map expected minutes → EV minutes_mult (stronger than old soft conf blend)."""
    mins_floor = float(SIGNAL_INGEST.get("lineup_minutes_floor") or 0.35)
    mins_ceil = float(SIGNAL_INGEST.get("lineup_minutes_ceil") or 1.15)
    # Weight how fully we trust the minutes point estimate.
    weight = float(SIGNAL_INGEST.get("lineup_minutes_weight") or 0.85)
    conf = max(0.0, min(1.0, float(confidence or 0.5)))
    try:
        mins = max(0, min(90, int(predicted_minutes)))
    except (TypeError, ValueError):
        return 1.0
    play = max(mins_floor, min(mins_ceil, mins / 90.0))
    # Blend toward play_prob with conf*weight (high-conf bench really hurts EV).
    return 1.0 + (play - 1.0) * conf * weight
