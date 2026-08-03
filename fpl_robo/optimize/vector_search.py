"""Simulated annealing over valid FPL squads (constraint repair, non-MIP)."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from fpl_robo.config import ATTACK_FLOOR, BUY_GATES, PRICE_AWARE, VECTOR_TEAM
from fpl_robo.graph.schema import MAX_PER_CLUB, SQUAD_COUNTS, SQUAD_SIZE
from fpl_robo.optimize.attack_quality import is_quality_fwd
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


def _fnum(val: Any, default: float = 0.0) -> float:
    try:
        x = float(val)
    except (TypeError, ValueError):
        return default
    if x != x:
        return default
    return x


def filter_vector_transfer_pool(
    preds: pd.DataFrame,
    *,
    current_ids: list[int],
    require_cover_eligible: bool = True,
    enforce_buy_eligible: bool = False,
    min_buy_ev: float | None = None,
    exclude_unavailable: bool = True,
    min_play_chance: float | None = None,
    max_news_risk: float | None = None,
    min_signal_news_mult: float | None = None,
) -> pd.DataFrame:
    """Mirror MIP buy/cover pool filters; always keep owned players."""
    df = preds.copy()
    if df.empty:
        return df
    owned_set = {int(x) for x in current_ids}
    raw_by_id = {int(r["id"]): r for _, r in df.iterrows()}
    if min_play_chance is None:
        min_play_chance = float(
            VECTOR_TEAM.get("min_play_chance")
            if VECTOR_TEAM.get("min_play_chance") is not None
            else BUY_GATES.get("min_buy_chance") or 50
        )
    if max_news_risk is None:
        max_news_risk = float(BUY_GATES.get("max_news_risk") or 0.08)
    if min_signal_news_mult is None:
        min_signal_news_mult = float(BUY_GATES.get("min_signal_news_mult") or 0.70)

    if exclude_unavailable and "status" in df.columns:
        df = df[~df["status"].isin(["u", "s"]) | df["id"].isin(owned_set)].copy()
    if min_play_chance is not None and "chance_of_playing_next_round" in df.columns:
        chance = df["chance_of_playing_next_round"]
        keep = chance.isna() | (chance >= float(min_play_chance)) | df["id"].isin(owned_set)
        df = df[keep].copy()

    if "cover_eligible" not in df.columns:
        df["cover_eligible"] = True
    if require_cover_eligible:
        df = df[df["cover_eligible"].astype(bool) | df["id"].isin(owned_set)].copy()

    if "buy_eligible" not in df.columns:
        df["buy_eligible"] = True
    if enforce_buy_eligible:
        df = df[df["buy_eligible"].astype(bool) | df["id"].isin(owned_set)].copy()
    if min_buy_ev is not None:
        ev_col = "base_points" if "base_points" in df.columns else "ev"
        df = df[
            (df[ev_col].astype(float) >= float(min_buy_ev)) | df["id"].isin(owned_set)
        ].copy()
    if max_news_risk is not None and "news_risk" in df.columns:
        risk = df["news_risk"].fillna(0.0).astype(float)
        df = df[(risk <= float(max_news_risk)) | df["id"].isin(owned_set)].copy()
    if min_signal_news_mult is not None and "signal_news_mult" in df.columns:
        snm = df["signal_news_mult"].fillna(1.0).astype(float)
        df = df[(snm >= float(min_signal_news_mult)) | df["id"].isin(owned_set)].copy()

    # Re-inject owned rows dropped by filters (legality).
    have = {int(x) for x in df["id"].tolist()} if len(df) else set()
    extras = []
    for pid in current_ids:
        if int(pid) in have:
            continue
        src = raw_by_id.get(int(pid), {})
        if not len(src):
            continue
        row = dict(src)
        row["id"] = int(pid)
        row["cover_eligible"] = True
        row["buy_eligible"] = True
        extras.append(row)
        have.add(int(pid))
    if extras:
        df = pd.concat([df, pd.DataFrame(extras)], ignore_index=True)

    # Crush owned non-cover base so SA prefers to sell dead bench fodder.
    if require_cover_eligible and "cover_eligible" in df.columns:
        ev_col = "base_points" if "base_points" in df.columns else "ev"
        for idx, r in df.iterrows():
            if int(r["id"]) in owned_set and not bool(r.get("cover_eligible", True)):
                df.at[idx, ev_col] = min(_fnum(r.get(ev_col), 0.0), 0.05)
                if "ev" in df.columns:
                    df.at[idx, "ev"] = df.at[idx, ev_col]
    return df.reset_index(drop=True)


def _raw_xi_ev(squad: list[int], scorer: FastTeamScorer) -> float:
    bd = scorer.score(squad, geometry=False)
    return float(sum(scorer.snaps[i].base for i in bd.starting_ids))


def _transfer_sanity_ok(
    owned: list[int],
    cand: list[int],
    scorer: FastTeamScorer,
    *,
    require_buy_starts: bool,
    min_transfer_net_ev: float,
    min_xi_ev_delta: float,
) -> bool:
    """Reject geometry-only thrash: bench buys / raw-EV regressions."""
    outs = set(owned) - set(cand)
    inns = set(cand) - set(owned)
    if not inns:
        return True
    bd = scorer.score(cand, geometry=False)
    start_set = set(bd.starting_ids)
    if require_buy_starts and any(i not in start_set for i in inns):
        return False
    net = 0.0
    # Pair same-position where possible for net EV.
    outs_left = list(outs)
    for enter in inns:
        epos = str(scorer.snaps[enter].position if enter in scorer.snaps else "")
        match = None
        for leave in outs_left:
            lpos = str(scorer.snaps[leave].position if leave in scorer.snaps else "")
            if lpos == epos:
                match = leave
                break
        if match is None and outs_left:
            match = outs_left[0]
        if match is None:
            break
        outs_left.remove(match)
        net += _fnum(scorer.snaps[enter].base) - _fnum(scorer.snaps[match].base)
    if net + 1e-9 < float(min_transfer_net_ev):
        return False
    xi_delta = _raw_xi_ev(cand, scorer) - _raw_xi_ev(owned, scorer)
    if xi_delta + 1e-9 < float(min_xi_ev_delta):
        return False
    return True


def _attack_floor_ok(squad: list[int], pool: _Pool, *, current: list[int] | None = None) -> bool:
    """Hard ATTACK_FLOOR for SA: require quality FWDs when the pool allows."""
    if not bool(VECTOR_TEAM.get("enforce_attack_floor", True)):
        return True
    if not bool(ATTACK_FLOOR.get("enabled", True)):
        return True
    need = int(ATTACK_FLOOR.get("min_quality_fwds") or 0)
    if need <= 0:
        return True

    def _ep(pid: int) -> float:
        r = pool.by_id.get(pid)
        if r is None:
            return 0.0
        return _fnum(r.get("ep_next") or r.get("base_points") or r.get("ev"))

    def _quality_count(ids: list[int]) -> int:
        n = 0
        for pid in ids:
            r = pool.by_id.get(pid)
            if r is None:
                continue
            if str(r.get("position") or "").upper() != "FWD":
                continue
            form_v = None
            try:
                if r.get("form") is not None:
                    form_v = float(r.get("form"))
            except (TypeError, ValueError):
                form_v = None
            roll = None
            try:
                if r.get("rolling_minutes") is not None:
                    roll = float(r.get("rolling_minutes"))
            except (TypeError, ValueError):
                roll = None
            if is_quality_fwd(
                ep_next=_ep(pid),
                status=str(r.get("status") or "a"),
                is_focal=bool(r.get("is_focal_attacker")),
                form=form_v,
                rolling_minutes=roll,
            ):
                n += 1
        return n

    # Eligible quality pool across all candidates (for thin-pool softening).
    pool_quality = _quality_count(list(pool.by_id.keys()))
    req = need if pool_quality >= need else (
        pool_quality if bool(ATTACK_FLOOR.get("soft_when_pool_thin", True)) else need
    )
    if req <= 0:
        return True
    squad_q = _quality_count(squad)
    if current is not None:
        # Never allow SA to drop below the live squad's quality count.
        req = max(req, min(_quality_count(current), need))
    return squad_q >= req


def _pair_transfers(
    owned: list[int], squad: list[int], pool: _Pool
) -> list[dict[str, int]]:
    outs = sorted(set(owned) - set(squad))
    inns = sorted(set(squad) - set(owned))
    transfers: list[dict[str, int]] = []
    outs_left = list(outs)
    for enter in inns:
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
    return transfers


def _bench_upgrade_pass(
    owned: list[int],
    squad: list[int],
    *,
    pool: _Pool,
    scorer: FastTeamScorer,
    free_left: int,
    max_hits: int,
    hits_for,
    feasible,
    require_starts: bool,
    min_net: float,
    min_xi: float,
) -> list[int]:
    """Sell bench players for higher-EV same-pos options that would start."""
    best = list(squad)
    already = len(set(owned) - set(best))
    slots = int(free_left) + int(max_hits) - max(already, hits_for(best))
    if slots <= 0:
        return best

    improved = True
    while improved and slots > 0:
        improved = False
        bd = scorer.score(best, geometry=False)
        for leave in list(bd.bench_ids):
            if leave not in pool.by_id:
                continue
            lpos = str(pool.by_id[leave].get("position") or "")
            leave_ev = _fnum(pool.by_id[leave].get("base_points") or pool.by_id[leave].get("ev"))
            cands = [
                pid
                for pid in (pool.by_pos.get(lpos) or [])
                if pid not in best and pid in pool.by_id
            ]
            cands.sort(
                key=lambda i: _fnum(pool.by_id[i].get("base_points") or pool.by_id[i].get("ev")),
                reverse=True,
            )
            for enter in cands[:12]:
                enter_ev = _fnum(
                    pool.by_id[enter].get("base_points") or pool.by_id[enter].get("ev")
                )
                if enter_ev <= leave_ev + 1e-9:
                    continue
                trial = [enter if x == leave else x for x in best]
                if not feasible(trial):
                    continue
                if not _attack_floor_ok(trial, pool, current=owned):
                    continue
                if not _transfer_sanity_ok(
                    owned,
                    trial,
                    scorer,
                    require_buy_starts=require_starts,
                    min_transfer_net_ev=min_net,
                    min_xi_ev_delta=min_xi,
                ):
                    continue
                if enter not in set(scorer.score(trial, geometry=False).starting_ids):
                    continue
                best = trial
                slots -= 1
                improved = True
                break
            if improved:
                break
    return best


def _consensus_squad(
    candidates: list[tuple[list[int], float]],
    *,
    owned: list[int],
    min_agree: int,
    owned_obj: float,
) -> tuple[list[int], float, dict[str, Any]]:
    """Pick majority-agree squad; else hold current."""
    if not candidates:
        return list(owned), float(owned_obj), {"votes": 0, "mode": "hold"}
    buckets: dict[frozenset[int], list[tuple[list[int], float]]] = {}
    for squad, obj in candidates:
        buckets.setdefault(frozenset(squad), []).append((squad, obj))
    ranked = sorted(
        buckets.items(),
        key=lambda kv: (-len(kv[1]), -max(o for _, o in kv[1])),
    )
    _best_key, best_group = ranked[0]
    votes = len(best_group)
    meta: dict[str, Any] = {
        "votes": votes,
        "candidates": len(candidates),
        "unique_squads": len(buckets),
        "min_agree": int(min_agree),
    }
    if votes >= int(min_agree) or len(candidates) == 1:
        squad, obj = max(best_group, key=lambda x: x[1])
        meta["mode"] = "consensus" if votes >= int(min_agree) else "single"
        return list(squad), float(obj), meta
    meta["mode"] = "hold_no_consensus"
    return list(owned), float(owned_obj), meta


def _transfer_ev_penalty(
    owned: list[int],
    cand: list[int],
    scorer: FastTeamScorer,
    *,
    weight: float,
) -> float:
    if weight <= 0 or set(cand) == set(owned):
        return 0.0
    pen = 0.0
    outs = set(owned) - set(cand)
    inns = set(cand) - set(owned)
    bd = scorer.score(cand, geometry=False)
    start_set = set(bd.starting_ids)
    for enter in inns:
        if enter not in start_set:
            pen += 4.0  # strong: never buy straight to bench
    outs_left = list(outs)
    for enter in inns:
        epos = str(scorer.snaps[enter].position if enter in scorer.snaps else "")
        match = None
        for leave in outs_left:
            lpos = str(scorer.snaps[leave].position if leave in scorer.snaps else "")
            if lpos == epos:
                match = leave
                break
        if match is None and outs_left:
            match = outs_left[0]
        if match is None:
            break
        outs_left.remove(match)
        delta = _fnum(scorer.snaps[enter].base) - _fnum(scorer.snaps[match].base)
        if delta < 0:
            pen += -delta
    xi_drop = _raw_xi_ev(owned, scorer) - _raw_xi_ev(cand, scorer)
    if xi_drop > 0:
        pen += xi_drop
    return float(weight) * pen


def _price_aware_term(
    owned: list[int],
    squad: list[int],
    pool: _Pool,
    *,
    capital: float,
) -> float:
    """Soft bank / riser-keep / faller-buy terms (MIP PRICE_AWARE parity)."""
    if not bool(VECTOR_TEAM.get("price_aware", True)):
        return 0.0
    bank_w = float(PRICE_AWARE.get("bank_weight") or 0.0)
    rise = float(PRICE_AWARE.get("rising_sell_penalty") or 0.0)
    fall = float(PRICE_AWARE.get("falling_buy_bonus") or 0.0)
    bank_left = float(capital) - squad_cost(squad, pool)
    term = bank_w * max(0.0, bank_left)
    owned_set = set(owned)
    squad_set = set(squad)
    for pid in squad_set & owned_set:
        if _fnum(pool.by_id[pid].get("cost_change_event")) > 0:
            term += rise
    for pid in squad_set - owned_set:
        if _fnum(pool.by_id[pid].get("cost_change_event")) < 0:
            term += fall
    return term


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


def _sa_burst_active(step: int, *, every: int, length: int) -> tuple[bool, float]:
    """Return (in_burst, spike_frac) for MD-style periodic reheating."""
    if every <= 0 or length <= 0:
        return False, 0.0
    phase = int(step) % int(every)
    if phase >= int(length):
        return False, 0.0
    # Linear cool within the burst window (1 → 0).
    spike = 1.0 - (phase / max(1, int(length)))
    return True, float(spike)


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
    stats: dict[str, Any] | None = None,
) -> tuple[list[int], float]:
    current = list(start)
    cur_obj = float(obj_fn(current))
    best = list(current)
    best_obj = cur_obj
    t0 = float(VECTOR_TEAM["sa_t0"])
    t_min = float(VECTOR_TEAM["sa_t_min"])
    bursts = bool(VECTOR_TEAM.get("sa_energy_bursts", False))
    burst_every = int(VECTOR_TEAM.get("sa_burst_every") or 300)
    burst_len = int(VECTOR_TEAM.get("sa_burst_length") or 40)
    burst_t0 = float(VECTOR_TEAM.get("sa_burst_t0") or max(8.0, t0 * 2.5))
    kick_swaps = max(1, int(VECTOR_TEAM.get("sa_burst_kick_swaps") or 2))
    kick_prob = float(VECTOR_TEAM.get("sa_burst_kick_prob") or 0.35)
    n_burst_steps = 0
    n_kicks = 0
    n_burst_accepts = 0
    n_escape_improves = 0  # best improved while in a burst window

    def _ok(squad: list[int]) -> bool:
        if feasible_fn is not None and not feasible_fn(squad):
            return False
        if max_changes_from is not None and max_changes is not None:
            if len(set(max_changes_from) - set(squad)) > max_changes:
                return False
        return True

    for k in range(iters):
        frac = k / max(1, iters - 1)
        temp = max(t_min, t0 * (1.0 - frac))
        in_burst = False
        if bursts:
            in_burst, spike = _sa_burst_active(k, every=burst_every, length=burst_len)
            if in_burst:
                n_burst_steps += 1
                temp = max(temp, burst_t0 * spike)
        base = best if rng.random() < 0.15 else current
        neigh = _swap_neighbour(base, pool, rng, budget=budget)
        if neigh is None or not _ok(neigh):
            continue
        # Kinetic kick: chain extra swaps while hot to jump farther than 1-edit.
        if in_burst and kick_swaps > 1 and rng.random() < kick_prob:
            trial = neigh
            chained = False
            for _ in range(kick_swaps - 1):
                nxt = _swap_neighbour(trial, pool, rng, budget=budget)
                if nxt is None or not _ok(nxt):
                    break
                trial = nxt
                chained = True
            if chained:
                neigh = trial
                n_kicks += 1
        o = float(obj_fn(neigh))
        delta = o - cur_obj
        if delta >= 0 or rng.random() < math.exp(delta / max(temp, 1e-9)):
            current = neigh
            cur_obj = o
            if in_burst:
                n_burst_accepts += 1
            if o > best_obj:
                best = list(current)
                best_obj = o
                if in_burst:
                    n_escape_improves += 1
    if stats is not None:
        stats.update(
            {
                "energy_bursts": bursts,
                "burst_steps": n_burst_steps,
                "burst_kicks": n_kicks,
                "burst_accepts": n_burst_accepts,
                "burst_best_improves": n_escape_improves,
                "best_obj": float(best_obj),
            }
        )
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
    require_cover_eligible: bool | None = None,
    enforce_buy_eligible: bool | None = None,
    min_buy_ev: float | None = None,
    exclude_unavailable: bool | None = None,
    min_play_chance: float | None = None,
) -> TransferPlan:
    """SA + hill-climb over squads reachable with ≤ FT+max_hits swaps."""
    cover = (
        bool(VECTOR_TEAM.get("require_cover_eligible", True))
        if require_cover_eligible is None
        else bool(require_cover_eligible)
    )
    excl = (
        bool(VECTOR_TEAM.get("exclude_unavailable", True))
        if exclude_unavailable is None
        else bool(exclude_unavailable)
    )
    buy_gate = (
        bool(VECTOR_TEAM.get("enforce_buy_eligible", True))
        if enforce_buy_eligible is None
        else bool(enforce_buy_eligible)
    )
    filtered = filter_vector_transfer_pool(
        preds,
        current_ids=current_ids,
        require_cover_eligible=cover,
        enforce_buy_eligible=buy_gate,
        min_buy_ev=min_buy_ev,
        exclude_unavailable=excl,
        min_play_chance=min_play_chance,
    )
    pool = _pool_from_preds(filtered)
    scorer = FastTeamScorer(filtered)
    iters = int(n_iters if n_iters is not None else VECTOR_TEAM["sa_iters_transfer"])
    hit_cost = int(VECTOR_TEAM["hit_cost"])
    require_starts = bool(VECTOR_TEAM.get("require_buy_starts", True))
    min_net = float(VECTOR_TEAM.get("min_transfer_net_ev") or 0.0)
    min_xi = float(VECTOR_TEAM.get("min_xi_ev_delta") or 0.0)
    ev_pen_w = float(VECTOR_TEAM.get("transfer_ev_penalty") or 0.0)
    sell = selling_prices or {}
    owned0 = [int(i) for i in current_ids if int(i) in pool.by_id]
    if len(owned0) != SQUAD_SIZE:
        sol = anneal_squad(filtered if len(filtered) >= SQUAD_SIZE else preds, budget=100.0, seed=seed)
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
        if not _attack_floor_ok(squad, pool, current=owned0):
            return False
        return True

    def obj_fn(squad: list[int]) -> float:
        base = scorer.score(squad).total - hit_cost * hits_for(squad)
        base = base - _transfer_ev_penalty(
            owned0, squad, scorer, weight=ev_pen_w
        )
        return base + _price_aware_term(owned0, squad, pool, capital=capital)

    current = list(owned0)
    max_changes = int(free_transfers) + int(max_hits)
    n_seeds = max(1, int(VECTOR_TEAM.get("sa_consensus_seeds") or VECTOR_TEAM.get("sa_restarts") or 1))
    min_agree = int(VECTOR_TEAM.get("sa_consensus_min_agree") or max(1, (n_seeds + 1) // 2))
    base_seed = int(VECTOR_TEAM["sa_seed"]) + 17 if seed is None else int(seed)
    per = max(250, iters // max(1, n_seeds))

    def hc_obj(s: list[int]) -> float:
        if not feasible(s) or len(set(owned0) - set(s)) > max_changes:
            return -1e18
        return obj_fn(s)

    candidates: list[tuple[list[int], float]] = []
    # Always include the hold squad as a candidate.
    hold_obj = float(obj_fn(current))
    candidates.append((list(current), hold_obj))
    seed_stats: list[dict[str, Any]] = []

    for k in range(n_seeds):
        rng_k = random.Random(base_seed + 97 * k)
        start = list(current)
        # Diversify with a few random legal neighbours.
        for _ in range(2):
            trial = _swap_neighbour(start, pool, rng_k, budget=1e9)
            if (
                trial is not None
                and feasible(trial)
                and len(set(owned0) - set(trial)) <= max_changes
            ):
                start = trial
                break
        loop_stats: dict[str, Any] = {}
        cand, _ = _sa_loop(
            start,
            pool,
            budget=1e9,
            obj_fn=obj_fn,
            iters=per,
            rng=rng_k,
            feasible_fn=feasible,
            max_changes_from=owned0,
            max_changes=max_changes,
            stats=loop_stats,
        )
        cand = _hillclimb_swaps(cand, pool, budget=1e9, obj_fn=hc_obj, top_k=15)
        raw_obj = float(obj_fn(cand))
        seed_row = {
            "seed_offset": k,
            "raw_obj": raw_obj,
            "feasible": feasible(cand),
            "transfers": len(set(owned0) - set(cand)),
            "sa": loop_stats,
        }
        if not feasible(cand):
            seed_row["accepted"] = False
            seed_row["reject"] = "infeasible"
            seed_stats.append(seed_row)
            continue
        if not _transfer_sanity_ok(
            owned0,
            cand,
            scorer,
            require_buy_starts=require_starts,
            min_transfer_net_ev=min_net,
            min_xi_ev_delta=min_xi,
        ):
            seed_row["accepted"] = False
            seed_row["reject"] = "transfer_sanity"
            seed_stats.append(seed_row)
            continue
        seed_row["accepted"] = True
        seed_stats.append(seed_row)
        candidates.append((list(cand), raw_obj))

    best, best_obj, consensus_meta = _consensus_squad(
        candidates,
        owned=owned0,
        min_agree=min_agree,
        owned_obj=hold_obj,
    )
    consensus_meta = {**consensus_meta, "seed_stats": seed_stats}
    if best_obj <= -1e17:
        best, best_obj = list(owned0), hold_obj

    # Bench-upgrade pass: sell bench fodder for higher-EV starters within FT budget.
    if bool(VECTOR_TEAM.get("bench_upgrade_pass", True)):
        upgraded = _bench_upgrade_pass(
            owned0,
            best,
            pool=pool,
            scorer=scorer,
            free_left=int(free_transfers),
            max_hits=int(max_hits),
            hits_for=hits_for,
            feasible=feasible,
            require_starts=require_starts,
            min_net=min_net,
            min_xi=min_xi,
        )
        if feasible(upgraded) and _transfer_sanity_ok(
            owned0,
            upgraded,
            scorer,
            require_buy_starts=require_starts,
            min_transfer_net_ev=min_net,
            min_xi_ev_delta=min_xi,
        ):
            up_obj = float(obj_fn(upgraded))
            if up_obj >= best_obj - 1e-9:
                best, best_obj = upgraded, up_obj
                consensus_meta = {**consensus_meta, "bench_upgrade": True}

    # Final hard sanity: fall back to hold if best still fails.
    if not _transfer_sanity_ok(
        owned0,
        best,
        scorer,
        require_buy_starts=require_starts,
        min_transfer_net_ev=min_net,
        min_xi_ev_delta=min_xi,
    ) or not feasible(best):
        best = list(owned0)
        best_obj = hold_obj
        consensus_meta = {**consensus_meta, "gate_reject": "sanity_or_attack_floor"}

    h = hits_for(best)
    transfers = _pair_transfers(owned0, best, pool)
    sol = _to_solution(best, filtered, scorer)
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
        meta={
            "consensus": consensus_meta,
            "attack_floor_ok": _attack_floor_ok(best, pool, current=owned0),
        },
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
