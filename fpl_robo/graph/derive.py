"""Second-wave KG feature derivation: set-pieces, rest, rivals, defensive rates."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fpl_robo.config import KG_ENRICH
from fpl_robo.graph.rates import rate_sample_ok, reliable_cs_per_90
from fpl_robo.graph.schema import EdgeType, GraphEdge, NodeId, NodeType
from fpl_robo.graph.store import KnowledgeGraph


def _parse_ko(raw: Any) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def team_last_and_next_kickoffs(
    kg: KnowledgeGraph,
    *,
    decision_gw: int,
) -> dict[int, dict[str, Any]]:
    """team_id -> {last_ko, next_ko, rest_days} around the decision GW."""
    by_team: dict[int, list[tuple[datetime, int]]] = {}
    for nid, attrs in kg.nodes_of_type(NodeType.FIXTURE):
        event = attrs.get("event")
        if event is None:
            continue
        ko = _parse_ko(attrs.get("kickoff_time"))
        if ko is None:
            continue
        for key in ("team_h", "team_a"):
            tid = attrs.get(key)
            if tid is None:
                continue
            by_team.setdefault(int(tid), []).append((ko, int(event)))

    out: dict[int, dict[str, Any]] = {}
    for tid, rows in by_team.items():
        rows = sorted(rows, key=lambda x: x[0])
        next_ko = None
        last_ko = None
        for ko, event in rows:
            if event < int(decision_gw):
                last_ko = ko
            elif event == int(decision_gw) and next_ko is None:
                next_ko = ko
            elif event > int(decision_gw) and next_ko is None:
                next_ko = ko
        rest_days = None
        if last_ko is not None and next_ko is not None:
            rest_days = max(0.0, (next_ko - last_ko).total_seconds() / 86400.0)
        out[tid] = {
            "last_kickoff": last_ko.isoformat() if last_ko else None,
            "next_kickoff": next_ko.isoformat() if next_ko else None,
            "rest_days": round(rest_days, 2) if rest_days is not None else None,
        }
    return out


def apply_rest_and_rate_features(
    kg: KnowledgeGraph,
    *,
    decision_gw: int,
    cfg: dict[str, Any] | None = None,
) -> int:
    """Stamp rest_days, xGI90, starts rate, set-piece scores onto players."""
    cfg = {**KG_ENRICH, **(cfg or {})}
    rest_map = team_last_and_next_kickoffs(kg, decision_gw=decision_gw)
    updated = 0
    for nid, attrs in kg.nodes_of_type(NodeType.PLAYER):
        tid = attrs.get("team_id")
        rest = rest_map.get(int(tid)) if tid is not None else None
        if rest:
            attrs["rest_days"] = rest.get("rest_days")
            attrs["next_kickoff"] = rest.get("next_kickoff")
            attrs["last_kickoff"] = rest.get("last_kickoff")
        else:
            attrs["rest_days"] = None

        xgi90 = attrs.get("expected_goal_involvements_per_90")
        if xgi90 is None:
            xg90 = float(attrs.get("expected_goals_per_90") or 0.0)
            xa90 = float(attrs.get("expected_assists_per_90") or 0.0)
            xgi90 = xg90 + xa90 if (xg90 or xa90) else None
        # Keep the raw FPL rate for diagnostics; only publish a trusted xgi90
        # once the minutes sample is large enough (blocks injury-cameo bombs).
        attrs["xgi90_raw"] = round(float(xgi90), 4) if xgi90 is not None else None
        if rate_sample_ok(attrs, cfg=cfg) and xgi90 is not None:
            hard_cap = float(cfg.get("xgi90_hard_cap") or 1.25)
            attrs["xgi90"] = round(min(float(xgi90), hard_cap), 4)
        else:
            attrs["xgi90"] = None

        starts = float(attrs.get("starts") or 0.0)
        minutes = float(attrs.get("minutes") or 0.0)
        starts_p90 = attrs.get("starts_per_90")
        if starts_p90 is None and minutes > 0:
            starts_p90 = starts / max(1.0, minutes / 90.0)
        attrs["starts_rate"] = (
            round(float(starts_p90), 4) if starts_p90 is not None else None
        )

        pen = attrs.get("penalties_order")
        corner = attrs.get("corners_order")
        set_piece = 0.0
        if pen is not None:
            try:
                order = int(pen)
                if order == 1:
                    set_piece += float(cfg.get("set_piece_pen_boost") or 0.08)
                elif order == 2:
                    set_piece += 0.5 * float(cfg.get("set_piece_pen_boost") or 0.08)
            except (TypeError, ValueError):
                pass
        if corner is not None:
            try:
                order = int(corner)
                if order == 1:
                    set_piece += float(cfg.get("set_piece_corner_boost") or 0.04)
                elif order == 2:
                    set_piece += 0.5 * float(cfg.get("set_piece_corner_boost") or 0.04)
            except (TypeError, ValueError):
                pass
        attrs["set_piece_factor"] = round(1.0 + set_piece, 4)
        attrs["set_piece_score"] = round(set_piece, 4)

        cs90 = attrs.get("clean_sheets_per_90")
        if rate_sample_ok(attrs, cfg=cfg) and cs90 is not None:
            attrs["cs_per_90"] = round(min(float(cs90), 1.0), 4)
        else:
            attrs["cs_per_90"] = None
        xgc90 = attrs.get("expected_goals_conceded_per_90")
        attrs["xgc_per_90"] = round(float(xgc90), 4) if xgc90 is not None else None
        updated += 1
    return updated


def apply_focal_threat_and_rivals(
    kg: KnowledgeGraph,
    *,
    cfg: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Club×position threat share + optional RIVALS edges for minutes competition."""
    cfg = {**KG_ENRICH, **(cfg or {})}
    buckets: dict[tuple[int, str], list[tuple[Any, dict[str, Any]]]] = {}
    for nid, attrs in kg.nodes_of_type(NodeType.PLAYER):
        tid = attrs.get("team_id")
        pos = (attrs.get("position") or "UNK").upper()
        if tid is None:
            continue
        buckets.setdefault((int(tid), pos), []).append((nid, attrs))

    rivals = 0
    for (tid, pos), members in buckets.items():
        scored = []
        for nid, attrs in members:
            threat = float(
                attrs.get("threat")
                if attrs.get("threat") is not None
                else (attrs.get("ict_index") or 0.0)
            )
            scored.append((threat, nid, attrs))
        total = sum(t for t, _, _ in scored) or 1.0
        scored.sort(key=lambda x: x[0], reverse=True)
        for rank, (threat, nid, attrs) in enumerate(scored, start=1):
            share = threat / total
            attrs["club_pos_threat_share"] = round(share, 4)
            attrs["club_pos_threat_rank"] = rank
            attrs["is_focal_attacker"] = bool(
                pos in {"MID", "FWD"} and rank == 1 and share >= 0.22
            )

        if bool(cfg.get("write_rival_edges", True)) and len(scored) >= 2:
            # Link top-2 by minutes/threat as rivals (idempotent via unique keys).
            top = scored[:2]
            a_nid, b_nid = top[0][1], top[1][1]
            for src, dst in ((a_nid, b_nid), (b_nid, a_nid)):
                kg.add_edge(
                    GraphEdge(
                        EdgeType.RIVALS,
                        NodeId(NodeType.PLAYER, int(src.key)),
                        NodeId(NodeType.PLAYER, int(dst.key)),
                        {
                            "team_id": tid,
                            "position": pos,
                            "enriched_rivals": True,
                        },
                    )
                )
                rivals += 1
    return {"players": sum(len(v) for v in buckets.values()), "rival_edges": rivals}


def write_set_piece_edges(kg: KnowledgeGraph, *, cfg: dict[str, Any] | None = None) -> int:
    """PLAYER → TEAM SET_PIECE edges for primary/secondary takers."""
    cfg = {**KG_ENRICH, **(cfg or {})}
    if not bool(cfg.get("write_set_piece_edges", True)):
        return 0
    # Drop prior enriched set-piece edges.
    to_remove = []
    for u, v, key, data in kg.g.edges(keys=True, data=True):
        if data.get("edge_type") != EdgeType.SET_PIECE.value:
            continue
        if data.get("enriched_set_piece"):
            to_remove.append((u, v, key))
    for u, v, key in to_remove:
        kg.g.remove_edge(u, v, key)

    written = 0
    for nid, attrs in kg.nodes_of_type(NodeType.PLAYER):
        tid = attrs.get("team_id")
        if tid is None:
            continue
        kinds = []
        pen = attrs.get("penalties_order")
        corner = attrs.get("corners_order")
        if pen is not None:
            try:
                kinds.append(("penalty", int(pen)))
            except (TypeError, ValueError):
                pass
        if corner is not None:
            try:
                kinds.append(("corner", int(corner)))
            except (TypeError, ValueError):
                pass
        for kind, order in kinds:
            if order not in {1, 2}:
                continue
            kg.add_edge(
                GraphEdge(
                    EdgeType.SET_PIECE,
                    NodeId(NodeType.PLAYER, int(nid.key)),
                    NodeId(NodeType.TEAM, int(tid)),
                    {
                        "kind": kind,
                        "order": order,
                        "enriched_set_piece": True,
                    },
                )
            )
            written += 1
    return written


def fold_signals_onto_players(kg: KnowledgeGraph) -> int:
    """Copy HAS_SIGNAL summaries onto player attrs (for KG queries / future EV).

    Live EV still applies signal multipliers post-predict to avoid double-count.
    """
    # Reset
    for _, attrs in kg.nodes_of_type(NodeType.PLAYER):
        attrs.pop("lineup_role", None)
        attrs.pop("lineup_confidence", None)
        attrs.pop("signal_kinds", None)

    updated = 0
    for u, v, _, data in kg.g.edges(keys=True, data=True):
        if data.get("edge_type") != EdgeType.HAS_SIGNAL.value:
            continue
        if u[0] != NodeType.PLAYER.value:
            continue
        player = kg.g.nodes[u]
        sig = kg.g.nodes.get(v) or {}
        kind = sig.get("kind") or data.get("kind")
        kinds = list(player.get("signal_kinds") or [])
        if kind and kind not in kinds:
            kinds.append(kind)
        player["signal_kinds"] = kinds
        if kind == "predicted_lineup":
            role = (sig.get("role") or "").lower()
            conf = sig.get("confidence")
            prev = player.get("lineup_confidence")
            if prev is None or (conf is not None and float(conf) >= float(prev)):
                player["lineup_role"] = role
                player["lineup_confidence"] = conf
        updated += 1
    return updated


def cs_ev_boost(player: dict[str, Any]) -> float:
    """Additive clean-sheet EV for GK/DEF from rate × matchup."""
    if not bool(KG_ENRICH.get("enabled", True)):
        return 0.0
    pos = (player.get("position") or "").upper()
    if pos not in {"GK", "DEF"}:
        return 0.0
    scale = float(KG_ENRICH.get("cs_boost_scale") or 0.0)
    cap = float(KG_ENRICH.get("cs_boost_cap") or 0.0)
    if scale <= 0 or cap <= 0:
        return 0.0
    cs = reliable_cs_per_90(player)
    if cs is None:
        return 0.0
    matchup = float(player.get("next_matchup_factor") or 1.0)
    # Lower xGC/90 is good for CS; soft inverse if present.
    xgc = player.get("xgc_per_90")
    xgc_term = 1.0
    if xgc is not None and float(xgc) > 0:
        xgc_term = max(0.7, min(1.15, 1.2 - float(xgc)))
    raw = scale * float(cs) * matchup * xgc_term
    return max(0.0, min(cap, raw))


def saves_ev_boost(player: dict[str, Any]) -> float:
    """Additive saves EV for GK from official saves_per_90 (thin-sample guarded)."""
    if not bool(KG_ENRICH.get("enabled", True)):
        return 0.0
    pos = (player.get("position") or "").upper()
    if pos != "GK":
        return 0.0
    scale = float(KG_ENRICH.get("saves_boost_scale") or 0.0)
    cap = float(KG_ENRICH.get("saves_boost_cap") or 0.0)
    if scale <= 0 or cap <= 0:
        return 0.0
    saves90 = player.get("saves_per_90")
    if saves90 is None:
        return 0.0
    try:
        rate = float(saves90)
    except (TypeError, ValueError):
        return 0.0
    if rate <= 0:
        return 0.0
    min_mins = float(KG_ENRICH.get("rate_feature_min_minutes") or 180.0)
    minutes = float(player.get("minutes") or player.get("prior_minutes") or 0.0)
    if minutes < min_mins:
        return 0.0
    # ~3 saves ≈ 1 FPL point; soft scale so this stays a crumb vs CS/xGI.
    raw = scale * (rate / 3.0)
    return max(0.0, min(cap, raw))
