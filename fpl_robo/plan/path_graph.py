"""3-week transfer path planning via beam search on a planning graph.

Nodes are squad states at a gameweek offset; edges are HOLD or TRANSFER actions.
Edge weights are *week-specific* FDR-scaled expected XI points minus hit cost,
so buying into a hard entry week is penalised relative to waiting for a better
fixture (the Gross-into-Arsenal case).

The live KnowledgeGraph supplies fixtures/FDR when available; historical
backtests pass `team_fixtures` built from vaastav FDR maps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import networkx as nx
import pandas as pd

from fpl_robo.config import PATH_PLAN
from fpl_robo.graph.schema import MAX_PER_CLUB, STARTING_BOUNDS, STARTING_SIZE
from fpl_robo.predict.horizon import fixture_scale


@dataclass(frozen=True)
class PathAction:
    kind: str  # hold | transfer
    element_out: int | None = None
    element_in: int | None = None
    out_name: str = ""
    in_name: str = ""
    hit_cost: float = 0.0
    week_ev: float = 0.0
    gameweek: int = 0


@dataclass
class PathPlanResult:
    status: str
    horizon: int
    start_gw: int
    path_score: float
    week0_action: PathAction
    actions: list[PathAction] = field(default_factory=list)
    path_nodes: list[str] = field(default_factory=list)
    graph_summary: dict[str, Any] = field(default_factory=dict)
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _cfg(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    return {**PATH_PLAN, **(overrides or {})}


def _fdr_scale(
    team_id: int,
    gw: int,
    team_fixtures: dict[int, dict[int, dict[str, float]]],
    *,
    fdr_weight: float,
    blank_mult: float,
    dgw_mult: float,
) -> float:
    slot = (team_fixtures.get(int(team_id)) or {}).get(int(gw))
    if slot is None:
        return 1.0
    n = int(float(slot.get("n_fixtures") or 0))
    fdr = slot.get("best_fdr")
    fdr = float(fdr) if fdr is not None else None
    return float(
        fixture_scale(
            n_fixtures=n,
            fdr=fdr,
            blank_mult=blank_mult,
            dgw_mult=dgw_mult,
            fdr_weight=fdr_weight,
        )
    )


def week_player_ev(
    row: pd.Series | dict[str, Any],
    *,
    gw: int,
    team_fixtures: dict[int, dict[int, dict[str, float]]],
    cfg: dict[str, Any],
) -> float:
    """Base EV × this-GW FDR scale (entry-week sensitive)."""
    get = row.get if isinstance(row, dict) else row.get
    base = float(get("ev") or get("ep_next") or get("xP") or 0.0)
    tid = int(get("team_id") or 0)
    scale = _fdr_scale(
        tid,
        gw,
        team_fixtures,
        fdr_weight=float(cfg["fdr_weight"]),
        blank_mult=float(cfg["blank_mult"]),
        dgw_mult=float(cfg["dgw_mult"]),
    )
    return max(0.0, base * scale)


def _xi_ev(
    squad: set[int],
    by_id: dict[int, dict[str, Any]],
    ev_map: dict[int, float],
) -> tuple[float, int | None]:
    """Formation-aware XI EV + captain (best EV doubled once)."""
    by_pos: dict[str, list[int]] = {"GK": [], "DEF": [], "MID": [], "FWD": []}
    for pid in squad:
        row = by_id.get(pid)
        if row is None:
            continue
        pos = str(row.get("position") or "MID").upper()
        by_pos.setdefault(pos, []).append(pid)
    for pos in by_pos:
        by_pos[pos].sort(key=lambda i: ev_map.get(i, 0.0), reverse=True)

    gk = by_pos.get("GK") or []
    if not gk:
        ordered = sorted(squad, key=lambda i: ev_map.get(i, 0.0), reverse=True)
        start = ordered[:STARTING_SIZE]
        if not start:
            return 0.0, None
        cap = max(start, key=lambda i: ev_map.get(i, 0.0))
        return sum(ev_map.get(i, 0.0) for i in start) + ev_map.get(cap, 0.0), cap

    best = -1e18
    best_start: list[int] = []
    defs, mids, fwds = by_pos.get("DEF") or [], by_pos.get("MID") or [], by_pos.get("FWD") or []
    for n_def in range(STARTING_BOUNDS["DEF"][0], STARTING_BOUNDS["DEF"][1] + 1):
        for n_mid in range(STARTING_BOUNDS["MID"][0], STARTING_BOUNDS["MID"][1] + 1):
            n_fwd = STARTING_SIZE - 1 - n_def - n_mid
            if n_fwd < STARTING_BOUNDS["FWD"][0] or n_fwd > STARTING_BOUNDS["FWD"][1]:
                continue
            if n_def > len(defs) or n_mid > len(mids) or n_fwd > len(fwds):
                continue
            start = [gk[0]] + defs[:n_def] + mids[:n_mid] + fwds[:n_fwd]
            score = sum(ev_map.get(i, 0.0) for i in start)
            if score > best:
                best = score
                best_start = start
    if not best_start:
        return 0.0, None
    cap = max(best_start, key=lambda i: ev_map.get(i, 0.0))
    return best + ev_map.get(cap, 0.0), cap


def _club_ok(squad: set[int], by_id: dict[int, dict[str, Any]], new_id: int) -> bool:
    teams: dict[int, int] = {}
    for pid in squad:
        row = by_id.get(pid)
        if row is None:
            continue
        tid = int(row.get("team_id") or 0)
        teams[tid] = teams.get(tid, 0) + 1
    nrow = by_id.get(new_id)
    if nrow is None:
        return False
    tid = int(nrow.get("team_id") or 0)
    return teams.get(tid, 0) < MAX_PER_CLUB


def _affordable(
    *,
    out_id: int,
    in_id: int,
    bank: float,
    by_id: dict[int, dict[str, Any]],
    selling: dict[int, float],
) -> bool:
    sell = float(selling.get(out_id, by_id.get(out_id, {}).get("price") or 4.5))
    buy = float(by_id.get(in_id, {}).get("price") or 99.0)
    return sell + float(bank) + 1e-9 >= buy


def _candidates_for_week(
    squad: set[int],
    *,
    gw: int,
    by_id: dict[int, dict[str, Any]],
    team_fixtures: dict[int, dict[int, dict[str, float]]],
    bank: float,
    selling: dict[int, float],
    cfg: dict[str, Any],
) -> list[tuple[int, int, float, float]]:
    """Return (out, in, out_ev, in_ev) transfer candidates for this GW."""
    ev_map = {
        pid: week_player_ev(row, gw=gw, team_fixtures=team_fixtures, cfg=cfg)
        for pid, row in by_id.items()
    }
    owned = [pid for pid in squad if pid in by_id]
    if not owned:
        return []
    owned.sort(key=lambda i: ev_map.get(i, 0.0))
    outs = owned[: int(cfg["candidate_outs"])]

    pool: list[tuple[int, int, float, float]] = []
    min_pp = float(cfg.get("min_play_prob") or 0.0)
    max_price = float(cfg.get("max_price") or 14.0)
    n_ins = int(cfg["candidate_ins_per_out"])

    for out_id in outs:
        out_row = by_id[out_id]
        pos = str(out_row.get("position") or "").upper()
        out_ev = ev_map.get(out_id, 0.0)
        # Same-position upgrades not owned.
        ins = []
        for pid, row in by_id.items():
            if pid in squad:
                continue
            if str(row.get("position") or "").upper() != pos:
                continue
            if float(row.get("price") or 99) > max_price:
                continue
            pp = row.get("play_prob")
            if pp is None:
                chance = row.get("chance_of_playing_next_round")
                pp = 1.0 if chance is None else float(chance) / 100.0
            if float(pp) + 1e-9 < min_pp:
                continue
            if not _club_ok(squad - {out_id}, by_id, pid):
                continue
            if not _affordable(
                out_id=out_id, in_id=pid, bank=bank, by_id=by_id, selling=selling
            ):
                continue
            ins.append((pid, ev_map.get(pid, 0.0)))
        ins.sort(key=lambda x: x[1], reverse=True)
        for in_id, in_ev in ins[:n_ins]:
            if in_ev + 1e-9 <= out_ev:
                continue
            pool.append((out_id, in_id, out_ev, in_ev))
    pool.sort(key=lambda t: t[3] - t[2], reverse=True)
    return pool[: max(12, n_ins * int(cfg["candidate_outs"]))]


def plan_transfer_path(
    *,
    start_gw: int,
    current_ids: list[int],
    predictions_by_gw: dict[int, pd.DataFrame],
    team_fixtures: dict[int, dict[int, dict[str, float]]],
    bank: float = 0.0,
    free_transfers: int = 1,
    selling_prices: dict[int, float] | None = None,
    config: dict[str, Any] | None = None,
) -> PathPlanResult:
    """Beam-search a horizon-length transfer path; return best week-0 action.

    `predictions_by_gw` must include frames for start_gw .. start_gw+horizon-1
    with at least id, web_name, position, team_id, price, ev.
    """
    cfg = _cfg(config)
    horizon = int(cfg["horizon"])
    beam_w = int(cfg["beam_width"])
    max_ft = int(cfg["max_ft"])
    hit_pts = float(cfg["hit_cost"])
    max_hits = int(cfg["max_hits_per_week"])
    max_xfers = int(cfg["max_transfers_per_week"])

    gws = [int(start_gw) + i for i in range(horizon)]
    for gw in gws:
        if gw not in predictions_by_gw or predictions_by_gw[gw].empty:
            return PathPlanResult(
                status="MissingPredictions",
                horizon=horizon,
                start_gw=int(start_gw),
                path_score=-1e18,
                week0_action=PathAction(kind="hold", gameweek=int(start_gw)),
                explanation=f"Missing predictions for GW{gw}",
            )

    # Merge player metadata (latest known row wins).
    by_id: dict[int, dict[str, Any]] = {}
    for gw in gws:
        for _, row in predictions_by_gw[gw].iterrows():
            by_id[int(row["id"])] = row.to_dict()

    selling = dict(selling_prices or {})
    for pid in current_ids:
        if pid not in selling and pid in by_id:
            selling[pid] = float(by_id[pid].get("price") or 4.5)

    graph = nx.DiGraph()
    # Beam entries: (path_score_so_far, gw_idx, squad, ft, bank, node_id, actions)
    start_squad = frozenset(int(i) for i in current_ids)
    start_node = f"gw{start_gw}|ft{int(free_transfers)}|{hash(start_squad) & 0xFFFFFFFF:x}"
    graph.add_node(
        start_node,
        gameweek=int(start_gw),
        ft=int(free_transfers),
        squad=list(start_squad),
        bank=float(bank),
    )

    beam: list[tuple[float, int, frozenset[int], int, float, str, list[PathAction]]] = [
        (0.0, 0, start_squad, int(free_transfers), float(bank), start_node, [])
    ]

    for step in range(horizon):
        gw = gws[step]
        preds = predictions_by_gw[gw]
        # Refresh prices / meta for this GW.
        for _, row in preds.iterrows():
            by_id[int(row["id"])] = row.to_dict()
        ev_map = {
            int(r["id"]): week_player_ev(
                r, gw=gw, team_fixtures=team_fixtures, cfg=cfg
            )
            for _, r in preds.iterrows()
        }
        # Ensure owned players missing from this GW frame still score something.
        for pid in set().union(*(set(b[2]) for b in beam)):
            if pid not in ev_map and pid in by_id:
                ev_map[pid] = week_player_ev(
                    by_id[pid], gw=gw, team_fixtures=team_fixtures, cfg=cfg
                )

        nxt: list[tuple[float, int, frozenset[int], int, float, str, list[PathAction]]] = []
        for score, _, squad, ft, cash, node, actions in beam:
            sell_map = {
                pid: float(selling.get(pid, by_id.get(pid, {}).get("price") or 4.5))
                for pid in squad
            }
            # HOLD always.
            hold_ev, _ = _xi_ev(set(squad), by_id, ev_map)
            hold_ft = min(max_ft, ft + 1)
            hold_act = PathAction(
                kind="hold",
                hit_cost=0.0,
                week_ev=hold_ev,
                gameweek=gw,
            )
            hold_node = f"gw{gw+1}|ft{hold_ft}|{hash(squad) & 0xFFFFFFFF:x}|h{len(actions)}"
            graph.add_node(
                hold_node,
                gameweek=gw + 1,
                ft=hold_ft,
                squad=list(squad),
                bank=cash,
            )
            graph.add_edge(
                node,
                hold_node,
                action="hold",
                week_ev=hold_ev,
                hit_cost=0.0,
                gameweek=gw,
            )
            nxt.append(
                (
                    score + hold_ev,
                    step + 1,
                    squad,
                    hold_ft,
                    cash,
                    hold_node,
                    actions + [hold_act],
                )
            )

            if max_xfers <= 0:
                continue
            cands = _candidates_for_week(
                set(squad),
                gw=gw,
                by_id=by_id,
                team_fixtures=team_fixtures,
                bank=cash,
                selling=sell_map,
                cfg=cfg,
            )
            for out_id, in_id, _oev, _iev in cands:
                hits_needed = max(0, 1 - ft)
                if hits_needed > max_hits:
                    continue
                hit_cost = hits_needed * hit_pts
                new_squad = set(squad)
                new_squad.discard(out_id)
                new_squad.add(in_id)
                if len(new_squad) != 15:
                    continue
                sell = sell_map.get(out_id, float(by_id.get(out_id, {}).get("price") or 4.5))
                buy = float(by_id.get(in_id, {}).get("price") or 99.0)
                new_bank = round(cash + sell - buy, 1)
                if new_bank < -1e-9:
                    continue
                new_ft = min(max_ft, max(0, ft - 1) + 1)  # spend FT, then bank +1
                if hits_needed:
                    new_ft = min(max_ft, 1)  # after a hit week, typically back to 1
                xfer_ev, _ = _xi_ev(new_squad, by_id, ev_map)
                net = xfer_ev - hit_cost
                act = PathAction(
                    kind="transfer",
                    element_out=out_id,
                    element_in=in_id,
                    out_name=str(by_id.get(out_id, {}).get("web_name") or out_id),
                    in_name=str(by_id.get(in_id, {}).get("web_name") or in_id),
                    hit_cost=hit_cost,
                    week_ev=xfer_ev,
                    gameweek=gw,
                )
                new_fs = frozenset(new_squad)
                xfer_node = (
                    f"gw{gw+1}|ft{new_ft}|{hash(new_fs) & 0xFFFFFFFF:x}|t{out_id}-{in_id}"
                )
                graph.add_node(
                    xfer_node,
                    gameweek=gw + 1,
                    ft=new_ft,
                    squad=list(new_fs),
                    bank=new_bank,
                )
                graph.add_edge(
                    node,
                    xfer_node,
                    action="transfer",
                    element_out=out_id,
                    element_in=in_id,
                    week_ev=xfer_ev,
                    hit_cost=hit_cost,
                    gameweek=gw,
                )
                nxt.append(
                    (
                        score + net,
                        step + 1,
                        new_fs,
                        new_ft,
                        new_bank,
                        xfer_node,
                        actions + [act],
                    )
                )

        nxt.sort(key=lambda t: t[0], reverse=True)
        beam = nxt[:beam_w]
        if not beam:
            break

    if not beam:
        return PathPlanResult(
            status="Empty",
            horizon=horizon,
            start_gw=int(start_gw),
            path_score=-1e18,
            week0_action=PathAction(kind="hold", gameweek=int(start_gw)),
            explanation="No feasible paths",
        )

    # Terminal FT bank bonus.
    ft_bonus = float(cfg.get("ft_bank_bonus") or 0.0)
    ranked = []
    for score, _, squad, ft, cash, node, actions in beam:
        bonus = ft_bonus * max(0, ft - 1)
        ranked.append((score + bonus, squad, ft, cash, node, actions))
    ranked.sort(key=lambda t: t[0], reverse=True)
    best_score, _, best_ft, _, best_node, best_actions = ranked[0]
    week0 = best_actions[0] if best_actions else PathAction(kind="hold", gameweek=int(start_gw))

    # Trace node path for explainability.
    path_nodes = [start_node]
    cur = start_node
    for act in best_actions:
        # pick successor matching action
        for _, v, data in graph.out_edges(cur, data=True):
            if act.kind == "hold" and data.get("action") == "hold":
                path_nodes.append(v)
                cur = v
                break
            if (
                act.kind == "transfer"
                and data.get("action") == "transfer"
                and int(data.get("element_out") or -1) == int(act.element_out or -2)
                and int(data.get("element_in") or -1) == int(act.element_in or -2)
            ):
                path_nodes.append(v)
                cur = v
                break

    parts = []
    for act in best_actions:
        if act.kind == "hold":
            parts.append(f"GW{act.gameweek}: HOLD (EV {act.week_ev:.1f})")
        else:
            hit = f", hit -{act.hit_cost:.0f}" if act.hit_cost else ""
            parts.append(
                f"GW{act.gameweek}: {act.out_name}→{act.in_name} "
                f"(EV {act.week_ev:.1f}{hit})"
            )
    expl = " → ".join(parts) + f" | path_score={best_score:.2f} end_ft={best_ft}"

    return PathPlanResult(
        status="Optimal",
        horizon=horizon,
        start_gw=int(start_gw),
        path_score=float(best_score),
        week0_action=week0,
        actions=best_actions,
        path_nodes=path_nodes,
        graph_summary={
            "n_nodes": int(graph.number_of_nodes()),
            "n_edges": int(graph.number_of_edges()),
            "beam_width": beam_w,
            "terminal_ft": best_ft,
        },
        explanation=expl,
    )
