"""Transfer MIP: evolve current squad under FT / hit constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import pulp

from fpl_robo.config import BENCH_COVER, PRICE_AWARE
from fpl_robo.graph.schema import (
    MAX_PER_CLUB,
    SQUAD_COUNTS,
    SQUAD_SIZE,
    STARTING_BOUNDS,
    STARTING_SIZE,
)
from fpl_robo.optimize.captaincy import pick_vice_captain_id
from fpl_robo.optimize.single_gw import SquadSolution


@dataclass
class TransferPlan:
    status: str
    objective: float
    budget_left: float
    free_transfers: int
    transfers_made: int
    hits: int
    hit_cost: int
    transfers: list[dict[str, int]]
    solution: SquadSolution

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "objective": self.objective,
            "budget_left": self.budget_left,
            "free_transfers": self.free_transfers,
            "transfers_made": self.transfers_made,
            "hits": self.hits,
            "hit_cost": self.hit_cost,
            "transfers": self.transfers,
            "solution": self.solution.to_dict(),
        }


def optimise_transfers(
    players: pd.DataFrame,
    *,
    current_ids: list[int],
    bank: float,
    selling_prices: dict[int, float] | None = None,
    free_transfers: int = 1,
    max_hits: int = 1,
    hit_cost: int = 4,
    min_play_chance: int | None = 50,
    exclude_unavailable: bool = True,
    require_cover_eligible: bool = True,
    bench_ev_weight: float = 0.2,
    min_buy_ev: float | None = None,
    enforce_buy_eligible: bool = False,
    bank_weight: float | None = None,
    rising_sell_penalty: float | None = None,
    falling_buy_bonus: float | None = None,
    use_tiebreak: bool | None = None,
    tiebreak_config: dict[str, Any] | None = None,
) -> TransferPlan:
    """Optimise transfers from an owned squad.

    Objective: XI EV + captain EV + bench weight - hit_cost * hits
    (+ soft bank / price-awareness terms)
    (+ optional quality tie-break: lexicographic EV-first, or soft blend).
    """
    from fpl_robo.config import OPTIMISER_TIEBREAK
    from fpl_robo.optimize.tiebreak import (
        attach_quality_scores,
        solve_lexicographic,
        tiebreak_enabled,
        tiebreak_mode,
    )
    from fpl_robo.optimize.kg_features import prediction_row_extras

    bank_w = float(
        PRICE_AWARE["bank_weight"] if bank_weight is None else bank_weight
    )
    rise_pen = float(
        PRICE_AWARE["rising_sell_penalty"]
        if rising_sell_penalty is None
        else rising_sell_penalty
    )
    fall_bonus = float(
        PRICE_AWARE["falling_buy_bonus"] if falling_buy_bonus is None else falling_buy_bonus
    )
    tb_cfg = dict(OPTIMISER_TIEBREAK)
    if tiebreak_config:
        tb_cfg.update(tiebreak_config)
    if use_tiebreak is None:
        use_tiebreak = tiebreak_enabled(tb_cfg)
    else:
        tb_cfg["enabled"] = bool(use_tiebreak)
    empty = TransferPlan(
        status="Empty player pool",
        objective=0.0,
        budget_left=bank,
        free_transfers=free_transfers,
        transfers_made=0,
        hits=0,
        hit_cost=0,
        transfers=[],
        solution=SquadSolution(
            status="Empty player pool",
            objective=0.0,
            budget_used=0.0,
            gameweek=None,
            squad=[],
            starting_xi=[],
            bench=[],
            captain_id=None,
            captain_name=None,
            vice_captain_id=None,
            vice_captain_name=None,
        ),
    )
    df = players.copy()
    if df.empty:
        return empty

    if use_tiebreak:
        df = attach_quality_scores(df, config=tb_cfg)
    elif "quality" not in df.columns:
        df["quality"] = 0.0

    gameweek = int(df["gameweek"].dropna().iloc[0]) if "gameweek" in df else None
    raw_by_id = {int(r["id"]): r for _, r in df.iterrows()}

    if exclude_unavailable and "status" in df.columns:
        df = df[~df["status"].isin(["u", "s"])].copy()
    if min_play_chance is not None and "chance_of_playing_next_round" in df.columns:
        chance = df["chance_of_playing_next_round"]
        df = df[(chance.isna()) | (chance >= min_play_chance)].copy()
    if "cover_eligible" not in df.columns:
        df["cover_eligible"] = True
    owned_set = set(int(x) for x in current_ids)
    if require_cover_eligible:
        # Non-owned must be real minutes cover. Owned non-cover stay in the pool
        # for legality but are crushed / force-sold below when FTs allow.
        df = df[df["cover_eligible"].astype(bool) | df["id"].isin(owned_set)].copy()

    # Hard buy gates for non-owned players.
    if "buy_eligible" not in df.columns:
        df["buy_eligible"] = True
    if enforce_buy_eligible:
        df = df[df["buy_eligible"].astype(bool) | df["id"].isin(owned_set)].copy()
    if min_buy_ev is not None:
        df = df[(df["ev"].astype(float) >= float(min_buy_ev)) | df["id"].isin(owned_set)].copy()

    # Keep owned players even with non-positive EV so squad remains legal.
    df = df[(df["ev"] > 0) | df["id"].isin(owned_set)].copy().reset_index(drop=True)
    if "id" in df.columns:
        df["ev"] = df.apply(
            lambda r: max(float(r["ev"]), 0.01) if int(r["id"]) in owned_set else float(r["ev"]),
            axis=1,
        )

    # Ensure all current owned players exist in the pool.
    have = set(int(x) for x in df["id"].tolist()) if len(df) else set()
    extras = []
    for pid in current_ids:
        if int(pid) in have:
            continue
        src = raw_by_id.get(int(pid), {})
        extras.append(
            {
                "id": int(pid),
                "web_name": src.get("web_name", str(pid)),
                "position": src.get("position", "MID"),
                "team_id": int(src.get("team_id") or 1),
                "price": float(
                    src.get("price")
                    or (selling_prices or {}).get(int(pid), 4.5)
                ),
                "ev": max(float(src.get("ev") or 0.01), 0.01),
                "quality": float(src.get("quality") or 0.0),
                "status": "a",
                "chance_of_playing_next_round": 100,
                "cover_eligible": True,
                "gameweek": gameweek,
            }
        )
        have.add(int(pid))
    if extras:
        df = pd.concat([df, pd.DataFrame(extras)], ignore_index=True)
    if "quality" not in df.columns:
        df["quality"] = 0.0
    df["quality"] = pd.to_numeric(df["quality"], errors="coerce").fillna(0.0)

    if len(df) < SQUAD_SIZE:
        empty.status = "Not enough eligible players"
        empty.solution.status = empty.status
        return empty

    ids = [int(x) for x in df["id"].tolist()]
    owned = [int(x) for x in current_ids]
    owned_set = set(owned)
    if "cost_change_event" not in df.columns:
        df["cost_change_event"] = 0
    cost_change = {
        int(r["id"]): float(r.get("cost_change_event") or 0)
        for _, r in df.iterrows()
    }
    ev = dict(zip(df["id"], df["ev"].astype(float)))
    cover_map = {
        int(r["id"]): bool(r["cover_eligible"]) if "cover_eligible" in df.columns else True
        for _, r in df.iterrows()
    }
    if require_cover_eligible:
        # Dead enablers must not look attractive to keep on the bench.
        for pid in list(ev):
            if not cover_map.get(int(pid), True):
                ev[pid] = min(float(ev[pid]), 0.05)
    quality = dict(zip(df["id"], df["quality"].astype(float)))
    buy_price = dict(zip(df["id"], df["price"].astype(float)))
    pos = dict(zip(df["id"], df["position"]))
    team = dict(zip(df["id"], df["team_id"]))
    name = dict(zip(df["id"], df["web_name"]))
    if "ep_next" in df.columns:
        ep_next = {
            int(r["id"]): float(r["ep_next"] or 0.0) for _, r in df.iterrows()
        }
    else:
        ep_next = {int(i): float(ev.get(i) or 0.0) for i in ids}
    meta_by_id = {
        int(r["id"]): prediction_row_extras(r)
        for _, r in df.iterrows()
    }
    sell = {
        int(i): float((selling_prices or {}).get(int(i), buy_price.get(int(i), 4.5)))
        for i in owned
    }

    max_transfers = int(free_transfers) + int(max_hits)
    prob = pulp.LpProblem("fpl_transfers", pulp.LpMaximize)
    in_squad = pulp.LpVariable.dicts("squad", ids, cat="Binary")
    start = pulp.LpVariable.dicts("start", ids, cat="Binary")
    captain = pulp.LpVariable.dicts("captain", ids, cat="Binary")
    # outs for owned players
    sold = pulp.LpVariable.dicts("sold", owned, cat="Binary")
    bought = pulp.LpVariable.dicts(
        "bought", [i for i in ids if i not in owned_set], cat="Binary"
    )
    hits = pulp.LpVariable("hits", lowBound=0, upBound=max_hits, cat="Integer")

    # With enough free transfers, dump owned non-cover (bench fodder) immediately.
    if require_cover_eligible:
        must_sell = [i for i in owned if not cover_map.get(int(i), True)]
        if must_sell and int(free_transfers) >= len(must_sell):
            for i in must_sell:
                prob += sold[i] == 1

    weight = max(0.0, float(bench_ev_weight))
    gk_weight = max(
        0.0, float(BENCH_COVER.get("bench_gk_ev_weight", weight))
    )
    bench_w = {
        i: (gk_weight if str(pos[i]).upper() in {"GK", "GKP"} else weight) for i in ids
    }
    budget_left_expr = (
        float(bank)
        + pulp.lpSum(sell[i] * sold[i] for i in owned)
        - pulp.lpSum(buy_price[i] * bought[i] for i in bought)
    )
    # Price awareness on transfer decisions only (not XI EV).
    rising_keep_bonus = pulp.lpSum(
        rise_pen * in_squad[i] for i in owned if cost_change.get(i, 0.0) > 0
    )
    falling_buy_term = pulp.lpSum(
        fall_bonus * bought[i]
        for i in bought
        if cost_change.get(i, 0.0) < 0
    )
    n_out = pulp.lpSum(sold[i] for i in owned)
    n_in = pulp.lpSum(bought[i] for i in bought)
    q_w = float(tb_cfg.get("quality_weight") or 0.0) if use_tiebreak else 0.0
    t_pen = float(tb_cfg.get("transfer_penalty") or 0.0) if use_tiebreak else 0.0
    mode = tiebreak_mode(tb_cfg) if use_tiebreak else "soft"

    # Unlimited FT (preseason / WC): prefer stability over reshuffling for tiny EV.
    unlimitedish = int(free_transfers) >= 10
    xfer_pen = 0.0
    keep_q_bonus = 0.0
    if unlimitedish:
        from fpl_robo.config import ATTACK_FLOOR, SEASON_RUNNER

        xfer_pen = float(SEASON_RUNNER.get("unlimited_transfer_ev_penalty") or 0.0)
        keep_q_bonus = float(SEASON_RUNNER.get("unlimited_keep_quality_fwd_bonus") or 0.0)
        min_ep = float(ATTACK_FLOOR.get("min_ep_next") or 2.1)
        quality_owned = [
            i
            for i in owned
            if str(pos.get(i) or "").upper() == "FWD"
            and float(ep_next.get(i) or 0.0) >= min_ep
        ]
        keep_quality = (
            pulp.lpSum(in_squad[i] for i in quality_owned) if quality_owned else 0
        )
    else:
        keep_quality = 0

    primary = (
        pulp.lpSum(ev[i] * start[i] for i in ids)
        + pulp.lpSum(ev[i] * captain[i] for i in ids)
        + pulp.lpSum(bench_w[i] * ev[i] * (in_squad[i] - start[i]) for i in ids)
        - hit_cost * hits
        + bank_w * budget_left_expr
        + rising_keep_bonus
        + falling_buy_term
        - xfer_pen * n_out
        + keep_q_bonus * keep_quality
    )
    secondary = None
    if use_tiebreak and (q_w or t_pen):
        quality_term = (
            q_w
            * (
                pulp.lpSum(quality[i] * start[i] for i in ids)
                + pulp.lpSum(quality[i] * captain[i] for i in ids)
                + pulp.lpSum(
                    bench_w[i] * quality[i] * (in_squad[i] - start[i]) for i in ids
                )
            )
            if q_w
            else 0
        )
        transfer_pen_term = t_pen * n_out if t_pen else 0
        secondary = quality_term - transfer_pen_term

    # Ownership / transfer accounting constraints continue below; objective set at solve.

    prob += pulp.lpSum(in_squad[i] for i in ids) == SQUAD_SIZE
    for i in ids:
        prob += start[i] <= in_squad[i]
        prob += captain[i] <= start[i]
    prob += pulp.lpSum(start[i] for i in ids) == STARTING_SIZE
    prob += pulp.lpSum(captain[i] for i in ids) == 1

    for position, count in SQUAD_COUNTS.items():
        pos_players = [i for i in ids if pos[i] == position]
        prob += pulp.lpSum(in_squad[i] for i in pos_players) == count
    for position, (lo, hi) in STARTING_BOUNDS.items():
        pos_players = [i for i in ids if pos[i] == position]
        if not pos_players:
            continue
        prob += pulp.lpSum(start[i] for i in pos_players) >= lo
        prob += pulp.lpSum(start[i] for i in pos_players) <= hi
    for club in set(int(t) for t in team.values()):
        club_players = [i for i in ids if int(team[i]) == club]
        prob += pulp.lpSum(in_squad[i] for i in club_players) <= MAX_PER_CLUB

    from fpl_robo.optimize.attack_floor import apply_attack_floor_constraint

    apply_attack_floor_constraint(
        prob, in_squad=in_squad, ids=ids, pos=pos, ep_next=ep_next
    )

    # Ownership / transfer accounting
    for i in owned:
        # sold == 1 - in_squad
        prob += sold[i] == 1 - in_squad[i]
    for i in ids:
        if i in owned_set:
            continue
        prob += bought[i] == in_squad[i]

    prob += n_out == n_in
    prob += n_out <= max_transfers
    # hits >= transfers - FT
    prob += hits >= n_out - int(free_transfers)

    # Budget: bank + money from sales >= money spent on purchases
    prob += (
        float(bank)
        + pulp.lpSum(sell[i] * sold[i] for i in owned)
        - pulp.lpSum(buy_price[i] * bought[i] for i in bought)
        >= 0
    )

    status_code = (
        solve_lexicographic(
            prob,
            primary,
            secondary,
            tolerance=float(tb_cfg.get("ev_tolerance") or 1e-3),
        )
        if use_tiebreak and mode == "lexicographic" and secondary is not None
        else _solve_soft(prob, primary, secondary)
    )
    status = pulp.LpStatus[status_code]
    if status != "Optimal":
        empty.status = status
        empty.solution.status = status
        empty.solution.gameweek = gameweek
        return empty

    selected = [i for i in ids if in_squad[i].value() > 0.5]
    starters = [i for i in ids if start[i].value() > 0.5]
    cap_ids = [i for i in ids if captain[i].value() > 0.5]
    captain_id = cap_ids[0] if cap_ids else None
    outs = [i for i in owned if sold[i].value() > 0.5]
    inns = [i for i in bought if bought[i].value() > 0.5]
    # Pair same-position where possible
    transfers = _pair_transfers(outs, inns, pos)
    hits_v = int(round(hits.value() or 0))
    transfers_made = len(transfers)
    budget_left = round(
        float(bank)
        + sum(sell[i] for i in outs)
        - sum(buy_price[i] for i in inns),
        1,
    )
    objective = (
        float(pulp.value(primary) or 0.0)
        if use_tiebreak and mode == "lexicographic"
        else float(pulp.value(prob.objective) or 0.0)
    )

    def row(i: int, *, is_start: bool, is_captain: bool) -> dict[str, Any]:
        base = {
            "id": int(i),
            "web_name": name.get(i),
            "position": pos[i],
            "team_id": int(team[i]),
            "price": float(buy_price[i]),
            "ev": float(ev[i]),
            "start": is_start,
            "captain": is_captain,
            "vice_captain": False,
        }
        base.update(meta_by_id.get(int(i)) or {})
        return base

    squad_rows = [
        row(i, is_start=(i in starters), is_captain=(i == captain_id)) for i in selected
    ]
    squad_rows.sort(key=lambda r: (r["position"], -r["ev"]))
    vice_id = pick_vice_captain_id(squad_rows, captain_id)
    for r in squad_rows:
        r["vice_captain"] = vice_id is not None and int(r["id"]) == int(vice_id)
    solution = SquadSolution(
        status=status,
        objective=round(objective, 3),
        budget_used=round(sum(buy_price[i] for i in selected), 1),
        gameweek=gameweek,
        squad=squad_rows,
        starting_xi=[r for r in squad_rows if r["start"]],
        bench=[r for r in squad_rows if not r["start"]],
        captain_id=captain_id,
        captain_name=name.get(captain_id) if captain_id else None,
        vice_captain_id=vice_id,
        vice_captain_name=name.get(vice_id) if vice_id else None,
    )
    return TransferPlan(
        status=status,
        objective=solution.objective,
        budget_left=budget_left,
        free_transfers=int(free_transfers),
        transfers_made=transfers_made,
        hits=hits_v,
        hit_cost=hits_v * hit_cost,
        transfers=transfers,
        solution=solution,
    )


def _solve_soft(prob: pulp.LpProblem, primary: Any, secondary: Any | None) -> int:
    obj = primary + secondary if secondary is not None else primary
    prob.setObjective(obj)
    return prob.solve(pulp.PULP_CBC_CMD(msg=False))


def _pair_transfers(
    outs: list[int], inns: list[int], pos: dict[Any, Any]
) -> list[dict[str, int]]:
    outs_by: dict[str, list[int]] = {}
    inns_by: dict[str, list[int]] = {}
    for i in outs:
        outs_by.setdefault(str(pos.get(i, "MID")), []).append(int(i))
    for i in inns:
        inns_by.setdefault(str(pos.get(i, "MID")), []).append(int(i))
    transfers: list[dict[str, int]] = []
    leftover_out: list[int] = []
    leftover_in: list[int] = []
    for p in sorted(set(outs_by) | set(inns_by)):
        o = outs_by.get(p, [])
        n = inns_by.get(p, [])
        while o and n:
            transfers.append({"element_out": o.pop(0), "element_in": n.pop(0)})
        leftover_out.extend(o)
        leftover_in.extend(n)
    for element_out, element_in in zip(leftover_out, leftover_in):
        transfers.append({"element_out": int(element_out), "element_in": int(element_in)})
    return transfers
