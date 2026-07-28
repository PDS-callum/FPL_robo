"""Enrich the FPL knowledge graph with matchup, fixture-load, and derived player features.

These attributes are written onto TEAM FACES edges and PLAYER nodes so live EV models
can consume richer context without re-walking the graph each time.
"""

from __future__ import annotations

from typing import Any

from fpl_robo.config import HORIZON_EV, KG_ENRICH
from fpl_robo.graph.schema import EdgeType, GraphEdge, NodeId, NodeType
from fpl_robo.graph.store import KnowledgeGraph


def enrich_knowledge_graph(
    kg: KnowledgeGraph,
    *,
    gameweek: int | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Mutate `kg` in place. Returns counts of enriched edges / players."""
    from fpl_robo.graph.derive import (
        apply_focal_threat_and_rivals,
        apply_rest_and_rate_features,
        fold_signals_onto_players,
        write_set_piece_edges,
    )

    cfg = {**KG_ENRICH, **(config or {})}
    if not bool(cfg.get("enabled", True)):
        return {"edges": 0, "players": 0, "teams": 0, "player_faces": 0}

    gw = int(gameweek or kg.current_gameweek() or 1)
    teams = _team_index(kg)
    _enrich_faces_edges(kg, teams)
    team_gw = _team_gameweek_bundles(kg, teams, start_gw=gw, horizon=int(cfg.get("horizon") or 3))
    n_players = _enrich_players(kg, teams, team_gw, decision_gw=gw, cfg=cfg)
    apply_rest_and_rate_features(kg, decision_gw=gw, cfg=cfg)
    # Recompute role stability now that starts_rate / rest exist.
    for _, attrs in kg.nodes_of_type(NodeType.PLAYER):
        attrs["role_stability"] = round(_role_stability(attrs, cfg), 4)
    focal = apply_focal_threat_and_rivals(kg, cfg=cfg)
    n_set = write_set_piece_edges(kg, cfg=cfg)
    n_sig = fold_signals_onto_players(kg)
    n_teams = _enrich_team_aggregates(kg, teams, team_gw, decision_gw=gw)
    _attach_opponent_pressure(kg, teams, team_gw, decision_gw=gw, cfg=cfg)
    n_faces = 0
    if bool(cfg.get("write_player_faces", True)):
        n_faces = _write_player_faces(kg, team_gw, decision_gw=gw)
    return {
        "edges": kg.g.number_of_edges(),
        "players": n_players,
        "teams": n_teams,
        "player_faces": n_faces,
        "set_piece_edges": n_set,
        "rival_edges": int(focal.get("rival_edges") or 0),
        "signals_folded": n_sig,
        "gameweek": gw,
    }


def _team_index(kg: KnowledgeGraph) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for nid, attrs in kg.nodes_of_type(NodeType.TEAM):
        out[int(nid.key)] = dict(attrs)
    return out


def _norm_strength(value: Any, *, mid: float = 1150.0, span: float = 250.0) -> float:
    """Map FPL strength (~1000–1400) to roughly 0..1 centred on mid."""
    if value is None:
        return 0.5
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, 0.5 + (v - mid) / (2.0 * span)))


def _matchup_factor(
    *,
    position: str | None,
    home: bool,
    own: dict[str, Any],
    opp: dict[str, Any],
    weight: float,
) -> float:
    """Position-aware attack/defence matchup multiplier (≈0.85–1.15)."""
    if home:
        own_atk = own.get("strength_attack_home")
        own_def = own.get("strength_defence_home")
        opp_atk = opp.get("strength_attack_away")
        opp_def = opp.get("strength_defence_away")
    else:
        own_atk = own.get("strength_attack_away")
        own_def = own.get("strength_defence_away")
        opp_atk = opp.get("strength_attack_home")
        opp_def = opp.get("strength_defence_home")

    # Preseason / unset: FPL often ships 0 strengths — stay neutral and lean on FDR.
    vals = [own_atk, own_def, opp_atk, opp_def]
    if all(v is None or float(v or 0) <= 0 for v in vals):
        return 1.0

    own_atk_n = _norm_strength(own_atk)
    own_def_n = _norm_strength(own_def)
    opp_atk_n = _norm_strength(opp_atk)
    opp_def_n = _norm_strength(opp_def)

    pos = (position or "").upper()
    w = max(0.0, float(weight))
    if pos in {"GK", "DEF"}:
        edge = 0.55 * (own_def_n - 0.5) + 0.45 * (0.5 - opp_atk_n)
    elif pos == "MID":
        edge = 0.45 * (own_atk_n - 0.5) + 0.35 * (0.5 - opp_def_n) + 0.20 * (own_def_n - 0.5)
    else:  # FWD / unknown
        edge = 0.55 * (own_atk_n - 0.5) + 0.45 * (0.5 - opp_def_n)
    return max(0.80, min(1.20, 1.0 + w * edge))


def _enrich_faces_edges(kg: KnowledgeGraph, teams: dict[int, dict[str, Any]]) -> None:
    for u, v, key, data in list(kg.g.edges(keys=True, data=True)):
        if data.get("edge_type") != EdgeType.FACES.value:
            continue
        if u[0] != NodeType.TEAM.value or v[0] != NodeType.TEAM.value:
            continue
        own_id = int(u[1])
        opp_id = int(v[1])
        own = teams.get(own_id) or {}
        opp = teams.get(opp_id) or {}
        home = bool(data.get("home"))
        if home:
            own_atk = own.get("strength_attack_home")
            own_def = own.get("strength_defence_home")
            opp_atk = opp.get("strength_attack_away")
            opp_def = opp.get("strength_defence_away")
        else:
            own_atk = own.get("strength_attack_away")
            own_def = own.get("strength_defence_away")
            opp_atk = opp.get("strength_attack_home")
            opp_def = opp.get("strength_defence_home")
        data["own_attack"] = own_atk
        data["own_defence"] = own_def
        data["opp_attack"] = opp_atk
        data["opp_defence"] = opp_def
        data["opp_strength"] = opp.get("strength")
        data["opp_short"] = opp.get("short_name")
        # Generic (MID-like) matchup stored on the edge; players refine by position later.
        data["matchup_factor"] = round(
            _matchup_factor(position="MID", home=home, own=own, opp=opp, weight=1.0),
            4,
        )


def _team_gameweek_bundles(
    kg: KnowledgeGraph,
    teams: dict[int, dict[str, Any]],
    *,
    start_gw: int,
    horizon: int,
) -> dict[int, dict[int, dict[str, Any]]]:
    """team_id -> gw -> fixture bundle for that side."""
    end_gw = int(start_gw) + max(1, int(horizon)) - 1
    out: dict[int, dict[int, dict[str, Any]]] = {}
    for u, v, _, data in kg.g.edges(keys=True, data=True):
        if data.get("edge_type") != EdgeType.FACES.value:
            continue
        gw = data.get("gameweek")
        if gw is None:
            continue
        gw = int(gw)
        if gw < int(start_gw) or gw > end_gw:
            continue
        if u[0] != NodeType.TEAM.value:
            continue
        team_id = int(u[1])
        opp_id = int(v[1]) if v[0] == NodeType.TEAM.value else None
        home = bool(data.get("home"))
        slot = out.setdefault(team_id, {}).setdefault(
            gw,
            {
                "n_fixtures": 0,
                "best_fdr": None,
                "fdrs": [],
                "homes": [],
                "opp_ids": [],
                "matchups_mid": [],
                "fixture_ids": [],
            },
        )
        slot["n_fixtures"] = int(slot["n_fixtures"]) + 1
        diff = data.get("difficulty")
        if diff is not None:
            d = float(diff)
            slot["fdrs"].append(d)
            prev = slot["best_fdr"]
            if prev is None or d < float(prev):
                slot["best_fdr"] = d
        slot["homes"].append(home)
        fid = data.get("fixture_id")
        if fid is not None:
            slot["fixture_ids"].append(int(fid))
        if opp_id is not None:
            slot["opp_ids"].append(opp_id)
            own = teams.get(team_id) or {}
            opp = teams.get(opp_id) or {}
            slot["matchups_mid"].append(
                _matchup_factor(position="MID", home=home, own=own, opp=opp, weight=1.0)
            )
    return out


def _bundle_scale(slot: dict[str, Any] | None) -> float:
    """Blank/DGW scale only — FDR lives in next_context_factor to avoid double-count."""
    from fpl_robo.predict.horizon import fixture_scale

    if not slot:
        return float(HORIZON_EV.get("blank_mult") or 0.20)
    n = int(slot.get("n_fixtures") or 0)
    fdr = slot.get("best_fdr")
    return float(
        fixture_scale(
            n_fixtures=n,
            fdr=float(fdr) if fdr is not None else None,
            fdr_weight=0.0,
        )
    )


def _slot_context(
    *,
    pos: str | None,
    own: dict[str, Any],
    teams: dict[int, dict[str, Any]],
    slot: dict[str, Any] | None,
    matchup_w: float,
    fdr_w: float,
) -> tuple[float, float, float]:
    """Return (combined_context, matchup, fdr_factor) for one team GW slot."""
    if not slot:
        blank = float(HORIZON_EV.get("blank_mult") or 0.20)
        return blank, 1.0, blank

    homes = slot.get("homes") or []
    opp_ids = slot.get("opp_ids") or []
    matchups: list[float] = []
    for i, opp_id in enumerate(opp_ids):
        home = bool(homes[i]) if i < len(homes) else True
        opp = teams.get(int(opp_id)) or {}
        matchups.append(
            _matchup_factor(
                position=str(pos) if pos else "MID",
                home=home,
                own=own,
                opp=opp,
                weight=1.0,
            )
        )
    matchup = sum(matchups) / len(matchups) if matchups else 1.0
    best_fdr = slot.get("best_fdr")
    fdr_factor = 1.0
    if best_fdr is not None:
        fdr_factor = 1.0 + 0.08 * (3.0 - float(best_fdr))
    combined = matchup_w * matchup + fdr_w * fdr_factor
    return combined, matchup, fdr_factor


def _market_momentum(attrs: dict[str, Any]) -> float:
    """Signed [-1, 1] from net transfers relative to ownership mass."""
    tin = float(attrs.get("transfers_in_event") or 0)
    tout = float(attrs.get("transfers_out_event") or 0)
    net = tin - tout
    own = float(attrs.get("selected_by_percent") or 1.0)
    # Soft scale: ~50k net on a 10% owned player ≈ moderate signal.
    denom = max(5_000.0, own * 8_000.0)
    return max(-1.0, min(1.0, net / denom))


def _role_stability(attrs: dict[str, Any], cfg: dict[str, Any]) -> float:
    """How locked-in the minutes role looks (rolling, starts rate, or season proxy)."""
    floor = float(cfg.get("role_stability_floor") or 0.92)
    if attrs.get("history_attached") and attrs.get("rolling_minutes") is not None:
        rm = float(attrs["rolling_minutes"])
        return max(floor, min(1.0, 0.85 + 0.15 * (rm / 70.0)))
    starts_rate = attrs.get("starts_rate")
    if starts_rate is None:
        starts_rate = attrs.get("starts_per_90")
    if starts_rate is not None:
        # ~1.0 starts per 90 ≈ nailed; 0.5 ≈ rotation.
        return max(floor, min(1.0, 0.88 + 0.12 * float(starts_rate)))
    minutes = float(attrs.get("minutes") or 0)
    if minutes <= 0:
        return floor
    if minutes < 180:
        return max(floor, 0.94)
    return 1.0


def _enrich_players(
    kg: KnowledgeGraph,
    teams: dict[int, dict[str, Any]],
    team_gw: dict[int, dict[int, dict[str, Any]]],
    *,
    decision_gw: int,
    cfg: dict[str, Any],
) -> int:
    matchup_w = float(cfg.get("matchup_weight") or 0.55)
    fdr_w = float(cfg.get("fdr_weight") or 0.45)
    tw = matchup_w + fdr_w
    if tw > 0:
        matchup_w, fdr_w = matchup_w / tw, fdr_w / tw
    look_blend = max(0.0, min(0.5, float(cfg.get("lookahead_blend") or 0.0)))
    horizon = int(cfg.get("horizon") or 3)

    # Team ICT totals for share features.
    team_ict: dict[int, float] = {}
    for nid, attrs in kg.nodes_of_type(NodeType.PLAYER):
        tid = attrs.get("team_id")
        if tid is None:
            continue
        ict = attrs.get("rolling_ict")
        if ict is None:
            ict = attrs.get("ict_index") or 0.0
        team_ict[int(tid)] = team_ict.get(int(tid), 0.0) + float(ict or 0.0)

    updated = 0
    for nid, attrs in kg.nodes_of_type(NodeType.PLAYER):
        team_id = attrs.get("team_id")
        if team_id is None:
            continue
        team_id = int(team_id)
        pos = attrs.get("position")
        own = teams.get(team_id) or {}
        bundles = team_gw.get(team_id) or {}
        slot = bundles.get(int(decision_gw))

        n_fix = int((slot or {}).get("n_fixtures") or 0)
        best_fdr = (slot or {}).get("best_fdr")
        homes = (slot or {}).get("homes") or []
        opp_ids = (slot or {}).get("opp_ids") or []

        next_ctx, matchup, fdr_factor = _slot_context(
            pos=str(pos) if pos else None,
            own=own,
            teams=teams,
            slot=slot,
            matchup_w=matchup_w,
            fdr_w=fdr_w,
        )
        fix_scale = _bundle_scale(slot)

        # Horizon lookahead: mean context over next GWs (blanks pull down).
        look_vals: list[float] = []
        upcoming = 0
        for offset in range(horizon):
            s = bundles.get(int(decision_gw) + offset)
            upcoming += int((s or {}).get("n_fixtures") or 0)
            c, _, _ = _slot_context(
                pos=str(pos) if pos else None,
                own=own,
                teams=teams,
                slot=s,
                matchup_w=matchup_w,
                fdr_w=fdr_w,
            )
            # Count-aware without FDR double-count inside scale.
            look_vals.append(c * _bundle_scale(s))
        congestion = max(0, upcoming - horizon)
        look_mean = sum(look_vals) / len(look_vals) if look_vals else next_ctx
        blended_ctx = (1.0 - look_blend) * next_ctx + look_blend * look_mean

        # Prefer rolling xGI when history is attached; season totals only as soft proxy.
        if attrs.get("history_attached") and attrs.get("rolling_xgi") is not None:
            xgi = float(attrs["rolling_xgi"])
            xgi_source = "rolling"
        else:
            xg = float(attrs.get("expected_goals") or 0.0)
            xa = float(attrs.get("expected_assists") or 0.0)
            # Preseason: bootstrap xG is prior-season cumulative — keep as metadata only.
            xgi = xg + xa
            if attrs.get("expected_goal_involvements") is not None:
                xgi = max(xgi, float(attrs["expected_goal_involvements"]))
            xgi_source = "season"
        attrs["xgi"] = round(xgi, 3)
        attrs["xgi_source"] = xgi_source
        # For EV boost, expose a per-GW-ish figure without prior-season blow-ups.
        if xgi_source == "rolling":
            attrs["xgi_per_gw"] = round(xgi, 3)
        else:
            attrs["xgi_per_gw"] = round(xgi / 38.0, 4)

        news = str(attrs.get("news") or "").strip()
        chance = attrs.get("chance_of_playing_next_round")
        news_risk = 0.0
        status = (attrs.get("status") or "a").lower()
        if news and chance is None and status == "a":
            news_risk = float(cfg.get("news_risk") or 0.15)
        elif news and status == "d":
            news_risk = float(cfg.get("news_risk_doubt") or 0.05)

        mom = _market_momentum(attrs)
        price_delta = int(attrs.get("cost_change_event") or 0)
        # ±1 = 0.1m this GW; cap soft signal.
        price_mom = max(-1.0, min(1.0, price_delta / 2.0))

        ict = attrs.get("rolling_ict")
        if ict is None:
            ict = attrs.get("ict_index") or 0.0
        ict_tot = team_ict.get(team_id) or 0.0
        ict_share = (float(ict) / ict_tot) if ict_tot > 0 else 0.0

        price = float(attrs.get("price") or 0.0)
        form_val = attrs.get("rolling_points")
        if form_val is None:
            form_val = attrs.get("form") or 0.0
        ppm = (float(form_val) / price) if price > 0 else 0.0

        attrs["next_n_fixtures"] = n_fix
        attrs["next_fdr"] = float(best_fdr) if best_fdr is not None else None
        attrs["next_home"] = bool(homes[0]) if homes else None
        attrs["next_opp_ids"] = list(opp_ids)
        attrs["next_matchup_factor"] = round(matchup, 4)
        attrs["next_fdr_factor"] = round(fdr_factor, 4)
        attrs["next_context_factor"] = round(blended_ctx, 4)
        attrs["next_gw_context"] = round(next_ctx, 4)
        attrs["lookahead_context"] = round(look_mean, 4)
        attrs["next_fixture_scale"] = round(fix_scale, 4)
        attrs["fixture_congestion"] = int(congestion)
        attrs["news_risk"] = round(news_risk, 3)
        attrs["market_momentum"] = round(mom, 4)
        attrs["price_momentum"] = round(price_mom, 4)
        attrs["ict_share"] = round(ict_share, 4)
        attrs["points_per_million"] = round(ppm, 4)
        attrs["role_stability"] = round(_role_stability(attrs, cfg), 4)
        attrs["kg_enriched"] = True
        updated += 1
    return updated


def _enrich_team_aggregates(
    kg: KnowledgeGraph,
    teams: dict[int, dict[str, Any]],
    team_gw: dict[int, dict[int, dict[str, Any]]],
    *,
    decision_gw: int,
) -> int:
    """Attach squad-level form/xGI and next-GW load onto TEAM nodes."""
    by_team: dict[int, list[dict[str, Any]]] = {}
    for nid, attrs in kg.nodes_of_type(NodeType.PLAYER):
        tid = attrs.get("team_id")
        if tid is None:
            continue
        by_team.setdefault(int(tid), []).append(attrs)

    updated = 0
    for tid, players in by_team.items():
        forms = []
        for p in players:
            if p.get("rolling_points") is not None:
                forms.append(float(p["rolling_points"]))
            elif p.get("form") is not None:
                forms.append(float(p["form"]))
        xgis = [float(p.get("xgi") or 0.0) for p in players]
        icts = [
            float(p.get("rolling_ict") if p.get("rolling_ict") is not None else (p.get("ict_index") or 0.0))
            for p in players
        ]
        slot = (team_gw.get(tid) or {}).get(int(decision_gw))
        node = kg.g.nodes[(NodeType.TEAM.value, str(tid))]
        node["squad_form"] = round(sum(forms) / len(forms), 3) if forms else None
        node["squad_xgi"] = round(sum(sorted(xgis, reverse=True)[:5]), 3)
        node["squad_ict"] = round(sum(sorted(icts, reverse=True)[:5]), 3)
        node["next_n_fixtures"] = int((slot or {}).get("n_fixtures") or 0)
        node["next_fdr"] = (slot or {}).get("best_fdr")
        node["kg_enriched"] = True
        teams[tid] = dict(node)
        updated += 1
    return updated


def _attach_opponent_pressure(
    kg: KnowledgeGraph,
    teams: dict[int, dict[str, Any]],
    team_gw: dict[int, dict[int, dict[str, Any]]],
    *,
    decision_gw: int,
    cfg: dict[str, Any],
) -> None:
    """After team aggregates exist, stamp opponent form onto players."""
    for nid, attrs in kg.nodes_of_type(NodeType.PLAYER):
        tid = attrs.get("team_id")
        if tid is None:
            continue
        slot = (team_gw.get(int(tid)) or {}).get(int(decision_gw))
        opp_ids = (slot or {}).get("opp_ids") or []
        if not opp_ids:
            attrs["opp_squad_form"] = None
            attrs["opp_pressure"] = 1.0
            continue
        forms = []
        for oid in opp_ids:
            sf = (teams.get(int(oid)) or {}).get("squad_form")
            if sf is not None:
                forms.append(float(sf))
        if not forms:
            attrs["opp_squad_form"] = None
            attrs["opp_pressure"] = 1.0
            continue
        mean_form = sum(forms) / len(forms)
        attrs["opp_squad_form"] = round(mean_form, 3)
        # League-average squad form ~3–4; stronger opp attack → slight haircut for outfield.
        w = float(cfg.get("opp_form_weight") or 0.06)
        # Pressure >1 means tougher opponent (lower EV).
        pressure = 1.0 + w * ((mean_form - 3.5) / 3.5)
        attrs["opp_pressure"] = round(max(0.90, min(1.10, pressure)), 4)


def _write_player_faces(
    kg: KnowledgeGraph,
    team_gw: dict[int, dict[int, dict[str, Any]]],
    *,
    decision_gw: int,
) -> int:
    """PLAYER → TEAM FACES edges for the decision GW (queryable matchup graph)."""
    # Drop prior player-faces for this gw to keep enrich idempotent.
    to_remove = []
    for u, v, key, data in kg.g.edges(keys=True, data=True):
        if data.get("edge_type") != EdgeType.FACES.value:
            continue
        if u[0] != NodeType.PLAYER.value:
            continue
        if data.get("gameweek") == int(decision_gw) and data.get("enriched_faces"):
            to_remove.append((u, v, key))
    for u, v, key in to_remove:
        kg.g.remove_edge(u, v, key)

    written = 0
    for nid, attrs in kg.nodes_of_type(NodeType.PLAYER):
        tid = attrs.get("team_id")
        if tid is None:
            continue
        slot = (team_gw.get(int(tid)) or {}).get(int(decision_gw))
        if not slot:
            continue
        opp_ids = slot.get("opp_ids") or []
        homes = slot.get("homes") or []
        fids = slot.get("fixture_ids") or []
        fdrs = slot.get("fdrs") or []
        for i, oid in enumerate(opp_ids):
            home = bool(homes[i]) if i < len(homes) else True
            fid = fids[i] if i < len(fids) else None
            fdr = fdrs[i] if i < len(fdrs) else slot.get("best_fdr")
            kg.add_edge(
                GraphEdge(
                    EdgeType.FACES,
                    NodeId(NodeType.PLAYER, int(nid.key)),
                    NodeId(NodeType.TEAM, int(oid)),
                    {
                        "gameweek": int(decision_gw),
                        "home": home,
                        "fixture_id": fid,
                        "difficulty": fdr,
                        "matchup_factor": attrs.get("next_matchup_factor"),
                        "enriched_faces": True,
                    },
                )
            )
            written += 1
    return written


def context_multiplier(player: dict[str, Any], *, difficulty_weight: float = 0.08) -> tuple[float, dict[str, Any]]:
    """EV multiplier from enriched player attrs (fallback to classic FDR)."""
    meta: dict[str, Any] = {}
    if player.get("kg_enriched"):
        ctx = float(player.get("next_context_factor") or 1.0)
        scale = float(player.get("next_fixture_scale") or 1.0)
        mult = ctx * scale
        congestion = int(player.get("fixture_congestion") or 0)
        if congestion > 0:
            pen = float(KG_ENRICH.get("congestion_penalty") or 0.04) * congestion
            mult *= max(0.85, 1.0 - pen)
        news_risk = float(player.get("news_risk") or 0.0)
        if news_risk > 0:
            mult *= max(0.5, 1.0 - news_risk)

        mom_cap = float(KG_ENRICH.get("market_momentum_cap") or 0.0)
        if mom_cap > 0 and player.get("market_momentum") is not None:
            mom = max(-1.0, min(1.0, float(player["market_momentum"])))
            mult *= 1.0 + mom_cap * mom

        price_cap = float(KG_ENRICH.get("price_momentum_cap") or 0.0)
        if price_cap > 0 and player.get("price_momentum") is not None:
            pm = max(-1.0, min(1.0, float(player["price_momentum"])))
            mult *= 1.0 + price_cap * pm

        if player.get("opp_pressure") is not None:
            # opp_pressure >1 → tougher side → divide (haircut).
            mult /= max(0.90, min(1.10, float(player["opp_pressure"])))

        if player.get("role_stability") is not None:
            mult *= float(player["role_stability"])

        if player.get("set_piece_factor") is not None:
            mult *= float(player["set_piece_factor"])

        # Short turnaround haircut (midweek + weekend).
        rest = player.get("rest_days")
        short = float(KG_ENRICH.get("rest_days_short") or 3.0)
        if rest is not None and float(rest) < short and float(rest) > 0:
            pen = float(KG_ENRICH.get("rest_days_penalty") or 0.05)
            mult *= max(0.90, 1.0 - pen * (short - float(rest)) / short)

        value_cap = float(KG_ENRICH.get("value_form_cap") or 0.0)
        if value_cap > 0 and player.get("value_form") is not None:
            # value_form is typically ~0–1-ish points per million proxy.
            vf = max(0.0, min(1.5, float(player["value_form"])))
            mult *= 1.0 + value_cap * (vf / 1.5)

        # Mild lift for high ICT share / focal club attackers.
        ict_share = float(player.get("ict_share") or 0.0)
        pos = (player.get("position") or "").upper()
        if pos in {"MID", "FWD"} and ict_share >= 0.18:
            mult *= 1.0 + min(0.04, (ict_share - 0.18) * 0.25)
        if player.get("is_focal_attacker"):
            mult *= 1.0 + float(KG_ENRICH.get("threat_focal_boost") or 0.0)

        meta.update(
            {
                "context": round(ctx, 4),
                "fixture_scale": round(scale, 4),
                "congestion": congestion,
                "news_risk": news_risk,
                "market_momentum": player.get("market_momentum"),
                "price_momentum": player.get("price_momentum"),
                "opp_pressure": player.get("opp_pressure"),
                "role_stability": player.get("role_stability"),
                "set_piece_factor": player.get("set_piece_factor"),
                "rest_days": player.get("rest_days"),
                "is_focal_attacker": player.get("is_focal_attacker"),
            }
        )
        return max(0.30, min(1.75, mult)), meta

    # Legacy FDR-only path.
    fdr = player.get("next_fdr")
    if fdr is None:
        return 1.0, meta
    adj = 1.0 + float(difficulty_weight) * (3.0 - float(fdr))
    meta["fdr_adj"] = round(adj, 4)
    return adj, meta


def base_ev_with_form_blend(player: dict[str, Any], *, ep_next: float | None, form: float | None) -> float:
    """Blend official EP with form / rolling points / xGI90 when enrichment is on."""
    from fpl_robo.graph.rates import reliable_xgi90

    blend = float(KG_ENRICH.get("form_blend") or 0.0)
    xgi_blend = float(KG_ENRICH.get("xgi90_blend") or 0.0)
    if ep_next is None:
        if form is not None:
            return float(form)
        return 0.0
    base = float(ep_next)
    if blend > 0:
        if player.get("history_attached") and player.get("rolling_points") is not None:
            form_term = float(player["rolling_points"])
            base = (1.0 - blend) * base + blend * form_term
        elif form is not None and float(form) > 0:
            base = (1.0 - blend) * base + blend * float(form)

    # Soft pull toward underlying involvement rate (helps when EP is sticky).
    # Only when minutes sample is trustworthy — never on injury cameos.
    if xgi_blend > 0:
        xgi90 = reliable_xgi90(player)
        if xgi90 is not None and float(xgi90) > 0:
            pos = (player.get("position") or "").upper()
            pts_per = 4.2 if pos in {"MID", "FWD"} else 3.2
            involvement_cap = float(KG_ENRICH.get("xgi_involvement_ev_cap") or 6.0)
            involvement_ev = min(involvement_cap, float(xgi90) * pts_per)
            base = (1.0 - xgi_blend) * base + xgi_blend * involvement_ev
    return base


def xgi_ev_boost(player: dict[str, Any]) -> float:
    """Small additive EV from season or rolling xGI involvement."""
    from fpl_robo.graph.rates import rate_sample_ok, reliable_xgi90

    if not bool(KG_ENRICH.get("enabled", True)):
        return 0.0
    scale = float(KG_ENRICH.get("xgi_boost_scale") or 0.0)
    cap = float(KG_ENRICH.get("xgi_boost_cap") or 0.0)
    if scale <= 0 or cap <= 0:
        return 0.0

    # Prefer gated xGI/90; fall back to rolling/season totals only with minutes.
    xgi90 = reliable_xgi90(player)
    if xgi90 is not None:
        raw = float(xgi90)
    elif rate_sample_ok(player) and player.get("rolling_xgi") is not None and player.get("history_attached"):
        raw = float(player["rolling_xgi"])
    elif rate_sample_ok(player) and player.get("xgi_per_gw") is not None:
        raw = float(player["xgi_per_gw"])
    elif rate_sample_ok(player):
        raw = float(player.get("xgi") or 0.0) / 38.0
    else:
        return 0.0

    share = float(player.get("ict_share") or 0.0)
    amp = 1.0 + min(0.25, share)
    if player.get("set_piece_score"):
        amp += min(0.15, float(player["set_piece_score"]) * 2.0)
    return max(0.0, min(cap, scale * raw * amp))
