"""Simulated annealing over valid FPL squads (constraint repair, non-MIP)."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from fpl_robo.config import VECTOR_TEAM
from fpl_robo.graph.schema import MAX_PER_CLUB, SQUAD_COUNTS, SQUAD_SIZE
from fpl_robo.optimize.single_gw import SquadSolution
from fpl_robo.optimize.transfers import TransferPlan
from fpl_robo.predict.vector_team_score import FastTeamScorer, player_records, score_team


@dataclass
class _Pool:
    by_id: dict[int, pd.Series]
    by_pos: dict[str, list[int]]
    ids: list[int]


def _pool_from_preds(preds: pd.DataFrame) -> _Pool:
    by_id = {int(r["id"]): r for _, r in preds.iterrows()}
    by_pos: dict[str, list[int]] = {"GK": [], "DEF": [], "MID": [], "FWD": []}
    for pid, r in by_id.items():
        pos = str(r.get("position") or "MID").upper()
        by_pos.setdefault(pos, []).append(pid)
    for pos in by_pos:
        by_pos[pos].sort(
            key=lambda i: float(by_id[i].get("base_points") or by_id[i].get("ev") or 0.0),
            reverse=True,
        )
    return _Pool(by_id=by_id, by_pos=by_pos, ids=list(by_id.keys()))


def squad_is_valid(
    squad: list[int],
    pool: _Pool,
    *,
    budget: float = 100.0,
) -> bool:
    if len(squad) != SQUAD_SIZE or len(set(squad)) != SQUAD_SIZE:
        return False
    if any(i not in pool.by_id for i in squad):
        return False
    counts: Counter[str] = Counter()
    clubs: Counter[int] = Counter()
    cost = 0.0
    for pid in squad:
        r = pool.by_id[pid]
        counts[str(r.get("position") or "").upper()] += 1
        clubs[int(r.get("team_id") or 0)] += 1
        cost += float(r.get("price") or 4.5)
    if any(counts.get(pos, 0) != need for pos, need in SQUAD_COUNTS.items()):
        return False
    if any(n > MAX_PER_CLUB for n in clubs.values()):
        return False
    if cost > budget + 1e-6:
        return False
    return True


def squad_cost(squad: list[int], pool: _Pool) -> float:
    return sum(float(pool.by_id[i].get("price") or 4.5) for i in squad)


def _random_valid_squad(pool: _Pool, rng: random.Random, budget: float) -> list[int] | None:
    tries = int(VECTOR_TEAM["max_random_init_tries"])
    for _ in range(tries):
        squad: list[int] = []
        clubs: Counter[int] = Counter()
        ok = True
        for pos, need in SQUAD_COUNTS.items():
            cand = list(pool.by_pos.get(pos) or [])
            rng.shuffle(cand)
            picked = 0
            for pid in cand:
                tid = int(pool.by_id[pid].get("team_id") or 0)
                if clubs[tid] >= MAX_PER_CLUB:
                    continue
                squad.append(pid)
                clubs[tid] += 1
                picked += 1
                if picked >= need:
                    break
            if picked < need:
                ok = False
                break
        if not ok:
            continue
        if squad_cost(squad, pool) <= budget + 1e-6 and squad_is_valid(squad, pool, budget=budget):
            return squad
    return _greedy_squad(pool, budget)


def _greedy_squad(pool: _Pool, budget: float) -> list[int] | None:
    """Greedy by base_points with club/budget repair (strong SA warm start)."""
    squad: list[int] = []
    clubs: Counter[int] = Counter()
    cost = 0.0
    for pos, need in SQUAD_COUNTS.items():
        cand = list(pool.by_pos.get(pos) or [])
        picked = 0
        for pid in cand:
            tid = int(pool.by_id[pid].get("team_id") or 0)
            price = float(pool.by_id[pid].get("price") or 4.5)
            if clubs[tid] >= MAX_PER_CLUB:
                continue
            if cost + price > budget + 1e-6:
                continue
            squad.append(pid)
            clubs[tid] += 1
            cost += price
            picked += 1
            if picked >= need:
                break
        if picked < need:
            return None
    return squad if squad_is_valid(squad, pool, budget=budget) else None


def _swap_neighbour(
    squad: list[int],
    pool: _Pool,
    rng: random.Random,
    *,
    budget: float,
) -> list[int] | None:
    """Replace one squad player with an outsider of the same position."""
    out = list(squad)
    idx = rng.randrange(len(out))
    leave = out[idx]
    pos = str(pool.by_id[leave].get("position") or "").upper()
    owned = set(out)
    cand = [i for i in (pool.by_pos.get(pos) or []) if i not in owned]
    if not cand:
        return None
    cand.sort(key=lambda i: float(pool.by_id[i].get("base_points") or 0.0), reverse=True)
    top = cand[: max(12, len(cand) // 3)]
    enter = rng.choice(top if top else cand)
    out[idx] = enter
    if squad_is_valid(out, pool, budget=budget):
        return out
    return None


def _objective(squad: list[int], scorer: FastTeamScorer, hit_penalty: float = 0.0) -> float:
    return scorer.score(squad).total - float(hit_penalty)


def _hillclimb_swaps(
    squad: list[int],
    pool: _Pool,
    *,
    budget: float,
    obj_fn: Callable[[list[int]], float],
    top_k: int = 25,
    passes: int | None = None,
) -> list[int]:
    """Deterministic single-swap hill climb to a local optimum."""
    n_pass = int(passes if passes is not None else VECTOR_TEAM.get("hillclimb_passes") or 2)
    best = list(squad)
    best_obj = float(obj_fn(best))
    for _ in range(max(1, n_pass)):
        improved = False
        for idx, leave in enumerate(list(best)):
            pos = str(pool.by_id[leave].get("position") or "").upper()
            owned = set(best)
            cand = [i for i in (pool.by_pos.get(pos) or []) if i not in owned][:top_k]
            local_best = None
            local_obj = best_obj
            for enter in cand:
                trial = list(best)
                trial[idx] = enter
                if not squad_is_valid(trial, pool, budget=budget):
                    continue
                o = float(obj_fn(trial))
                if o > local_obj + 1e-9:
                    local_best, local_obj = trial, o
            if local_best is not None:
                best, best_obj = local_best, local_obj
                improved = True
        if not improved:
            break
    return best


def _sa_loop(
    start: list[int],
    pool: _Pool,
    *,
    budget: float,
    obj_fn: Callable[[list[int]], float],
    iters: int,
    rng: random.Random,
    feasible_fn: Callable[[list[int]], bool] | None = None,
    max_changes_from: list[int] | None = None,
    max_changes: int | None = None,
) -> tuple[list[int], float]:
    current = list(start)
    cur_obj = float(obj_fn(current))
    best = list(current)
    best_obj = cur_obj
    t0 = float(VECTOR_TEAM["sa_t0"])
    t_min = float(VECTOR_TEAM["sa_t_min"])
    for k in range(iters):
        frac = k / max(1, iters - 1)
        temp = max(t_min, t0 * (1.0 - frac))
        base = best if rng.random() < 0.15 else current
        neigh = _swap_neighbour(base, pool, rng, budget=budget)
        if neigh is None:
            continue
        if feasible_fn is not None and not feasible_fn(neigh):
            continue
        if max_changes_from is not None and max_changes is not None:
            if len(set(max_changes_from) - set(neigh)) > max_changes:
                continue
        o = float(obj_fn(neigh))
        delta = o - cur_obj
        if delta >= 0 or rng.random() < math.exp(delta / max(temp, 1e-9)):
            current = neigh
            cur_obj = o
            if o > best_obj:
                best = list(current)
                best_obj = o
    return best, best_obj


def _to_solution(squad: list[int], preds: pd.DataFrame, scorer: FastTeamScorer | None = None) -> SquadSolution:
    from fpl_robo.optimize.captaincy import pick_vice_captain_id

    pool = _pool_from_preds(preds)
    breakdown = (scorer or FastTeamScorer(preds)).score(squad)
    start_set = {int(i) for i in breakdown.starting_ids}
    cap_id = int(breakdown.captain_id) if breakdown.captain_id is not None else None
    squad_recs = player_records(squad, preds)
    for r in squad_recs:
        rid = int(r["id"])
        r["start"] = rid in start_set
        r["captain"] = cap_id is not None and rid == cap_id
        r["vice_captain"] = False
    vice_id = pick_vice_captain_id(squad_recs, cap_id)
    for r in squad_recs:
        if vice_id is not None and int(r["id"]) == int(vice_id):
            r["vice_captain"] = True
    starting = [r for r in squad_recs if r["start"]]
    bench = [r for r in squad_recs if not r["start"]]
    cap_name = None
    if cap_id is not None:
        for p in starting:
            if int(p["id"]) == cap_id:
                cap_name = p["web_name"]
                break
    used = squad_cost(squad, pool)
    vice_name = None
    if vice_id is not None:
        for p in squad_recs:
            if int(p["id"]) == int(vice_id):
                vice_name = p["web_name"]
                break
    return SquadSolution(
        status="Optimal",
        objective=float(breakdown.total),
        budget_used=float(used),
        gameweek=int(preds["gameweek"].iloc[0]) if "gameweek" in preds.columns and len(preds) else None,
        squad=squad_recs,
        starting_xi=starting,
        bench=bench,
        captain_id=cap_id,
        captain_name=cap_name,
        vice_captain_id=vice_id,
        vice_captain_name=vice_name,
    )


def anneal_squad(
    preds: pd.DataFrame,
    *,
    budget: float = 100.0,
    init_squad: list[int] | None = None,
    n_iters: int | None = None,
    seed: int | None = None,
    hit_penalty: float = 0.0,
) -> SquadSolution:
    """SA + hill-climb search for an initial 15-man squad."""
    pool = _pool_from_preds(preds)
    scorer = FastTeamScorer(preds)
    rng = random.Random(VECTOR_TEAM["sa_seed"] if seed is None else seed)
    iters = int(n_iters if n_iters is not None else VECTOR_TEAM["sa_iters_initial"])
    restarts = max(1, int(VECTOR_TEAM.get("sa_restarts") or 1))

    def obj_fn(squad: list[int]) -> float:
        return _objective(squad, scorer, hit_penalty)

    seeds: list[list[int]] = []
    if init_squad is not None and squad_is_valid(init_squad, pool, budget=budget):
        seeds.append(list(init_squad))
    greedy = _greedy_squad(pool, budget)
    if greedy is not None:
        seeds.append(greedy)
    for _ in range(restarts):
        rnd = _random_valid_squad(pool, rng, budget)
        if rnd is not None:
            seeds.append(rnd)
    if not seeds:
        return SquadSolution(
            status="Infeasible",
            objective=0.0,
            budget_used=0.0,
            gameweek=None,
            squad=[],
            starting_xi=[],
            bench=[],
            captain_id=None,
            captain_name=None,
        )

    best: list[int] | None = None
    best_obj = -1e18
    per = max(250, iters // max(1, len(seeds)))
    for start in seeds:
        cand, _ = _sa_loop(start, pool, budget=budget, obj_fn=obj_fn, iters=per, rng=rng)
        cand = _hillclimb_swaps(cand, pool, budget=budget, obj_fn=obj_fn)
        obj = float(obj_fn(cand))
        if obj > best_obj:
            best, best_obj = cand, obj

    assert best is not None
    return _to_solution(best, preds, scorer)


def anneal_transfers(
    preds: pd.DataFrame,
    *,
    current_ids: list[int],
    bank: float,
    selling_prices: dict[int, float] | None = None,
    free_transfers: int = 1,
    max_hits: int = 1,
    n_iters: int | None = None,
    seed: int | None = None,
) -> TransferPlan:
    """SA + hill-climb over squads reachable with ≤ FT+max_hits swaps."""
    pool = _pool_from_preds(preds)
    scorer = FastTeamScorer(preds)
    rng = random.Random(int(VECTOR_TEAM["sa_seed"]) + 17 if seed is None else seed)
    iters = int(n_iters if n_iters is not None else VECTOR_TEAM["sa_iters_transfer"])
    hit_cost = int(VECTOR_TEAM["hit_cost"])
    sell = selling_prices or {}
    owned0 = [int(i) for i in current_ids if int(i) in pool.by_id]
    if len(owned0) != SQUAD_SIZE:
        sol = anneal_squad(preds, budget=100.0, seed=seed)
        return TransferPlan(
            status=sol.status,
            objective=sol.objective,
            transfers=[],
            transfers_made=0,
            hits=0,
            hit_cost=0,
            budget_left=round(100.0 - sol.budget_used, 1),
            free_transfers=free_transfers,
            solution=sol,
        )

    capital = float(bank) + sum(
        float(sell.get(i, pool.by_id[i].get("price") or 4.5)) for i in owned0
    )

    def hits_for(squad: list[int]) -> int:
        return max(0, len(set(owned0) - set(squad)) - int(free_transfers))

    def feasible(squad: list[int]) -> bool:
        if not squad_is_valid(squad, pool, budget=1e9):
            return False
        if squad_cost(squad, pool) > capital + 1e-6:
            return False
        if hits_for(squad) > int(max_hits):
            return False
        return True

    def obj_fn(squad: list[int]) -> float:
        return scorer.score(squad).total - hit_cost * hits_for(squad)

    current = list(owned0)
    max_changes = int(free_transfers) + int(max_hits)

    seeds = [current]
    for _ in range(max(1, int(VECTOR_TEAM.get("sa_restarts") or 1))):
        trial = _swap_neighbour(current, pool, rng, budget=1e9)
        if (
            trial is not None
            and feasible(trial)
            and len(set(owned0) - set(trial)) <= max_changes
        ):
            seeds.append(trial)

    best = list(current)
    best_obj = float(obj_fn(best))
    per = max(250, iters // max(1, len(seeds)))

    def hc_obj(s: list[int]) -> float:
        if not feasible(s) or len(set(owned0) - set(s)) > max_changes:
            return -1e18
        return obj_fn(s)

    for start in seeds:
        cand, _ = _sa_loop(
            start,
            pool,
            budget=1e9,
            obj_fn=obj_fn,
            iters=per,
            rng=rng,
            feasible_fn=feasible,
            max_changes_from=owned0,
            max_changes=max_changes,
        )
        cand = _hillclimb_swaps(cand, pool, budget=1e9, obj_fn=hc_obj, top_k=15)
        if not feasible(cand):
            continue
        obj = float(obj_fn(cand))
        if obj > best_obj:
            best, best_obj = cand, obj

    h = hits_for(best)
    outs = sorted(set(owned0) - set(best))
    ins = sorted(set(best) - set(owned0))
    transfers: list[dict[str, int]] = []
    outs_left = list(outs)
    for enter in ins:
        epos = str(pool.by_id[enter].get("position") or "")
        match = None
        for leave in outs_left:
            if str(pool.by_id[leave].get("position") or "") == epos:
                match = leave
                break
        if match is None and outs_left:
            match = outs_left[0]
        if match is None:
            break
        outs_left.remove(match)
        transfers.append({"element_out": int(match), "element_in": int(enter)})

    sol = _to_solution(best, preds, scorer)
    budget_left = round(capital - squad_cost(best, pool), 1)
    return TransferPlan(
        status="Optimal",
        objective=float(best_obj),
        transfers=transfers,
        transfers_made=len(transfers),
        hits=int(h),
        hit_cost=int(h) * hit_cost,
        budget_left=budget_left,
        free_transfers=max(0, int(free_transfers) - len(transfers)),
        solution=sol,
    )


def dry_run_plan(
    preds: pd.DataFrame,
    *,
    current_ids: list[int] | None = None,
    bank: float = 0.0,
    free_transfers: int = 1,
    max_hits: int = 1,
    selling_prices: dict[int, float] | None = None,
) -> dict[str, Any]:
    """Convenience wrapper for live dry-run summaries."""
    if not current_ids:
        sol = anneal_squad(preds, budget=100.0)
        return {
            "mode": "vector_team_initial",
            "status": sol.status,
            "objective": sol.objective,
            "squad": sol.squad,
            "starting_xi": sol.starting_xi,
            "bench": sol.bench,
            "captain_id": sol.captain_id,
            "captain_name": sol.captain_name,
            "transfers": [],
            "hits": 0,
            "hit_cost": 0,
            "breakdown": score_team([int(p["id"]) for p in sol.squad], preds).__dict__
            if sol.squad
            else None,
        }
    plan = anneal_transfers(
        preds,
        current_ids=current_ids,
        bank=bank,
        selling_prices=selling_prices,
        free_transfers=free_transfers,
        max_hits=max_hits,
    )
    sol = plan.solution
    return {
        "mode": "vector_team_transfers",
        "status": plan.status,
        "objective": plan.objective,
        "squad": sol.squad,
        "starting_xi": sol.starting_xi,
        "bench": sol.bench,
        "captain_id": sol.captain_id,
        "captain_name": sol.captain_name,
        "transfers": plan.transfers,
        "transfers_made": plan.transfers_made,
        "hits": plan.hits,
        "hit_cost": plan.hit_cost,
        "budget_left": plan.budget_left,
        "breakdown": score_team([int(p["id"]) for p in sol.squad], preds).__dict__,
    }
