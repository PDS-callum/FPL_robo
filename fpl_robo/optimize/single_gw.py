"""Single-gameweek mixed-integer squad optimisation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import pulp

from fpl_robo.optimize.captaincy import pick_vice_captain_id
from fpl_robo.config import BENCH_COVER
from fpl_robo.graph.schema import (
    MAX_PER_CLUB,
    SQUAD_COUNTS,
    SQUAD_SIZE,
    STARTING_BOUNDS,
    STARTING_SIZE,
)


@dataclass
class SquadSolution:
    status: str
    objective: float
    budget_used: float
    gameweek: int | None
    squad: list[dict[str, Any]]
    starting_xi: list[dict[str, Any]]
    bench: list[dict[str, Any]]
    captain_id: int | None
    captain_name: str | None
    vice_captain_id: int | None = None
    vice_captain_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "objective": self.objective,
            "budget_used": self.budget_used,
            "gameweek": self.gameweek,
            "captain_id": self.captain_id,
            "captain_name": self.captain_name,
            "vice_captain_id": self.vice_captain_id,
            "vice_captain_name": self.vice_captain_name,
            "squad": self.squad,
            "starting_xi": self.starting_xi,
            "bench": self.bench,
        }


def optimise_single_gw(
    players: pd.DataFrame,
    budget: float = 100.0,
    min_play_chance: int | None = 50,
    exclude_unavailable: bool = True,
    require_cover_eligible: bool = True,
    bench_ev_weight: float = 0.2,
    use_tiebreak: bool | None = None,
    tiebreak_config: dict[str, Any] | None = None,
) -> SquadSolution:
    """Pick a legal 15-man squad, starting XI, and captain for one GW.

    Objective: maximise starting-XI EV + captain EV (captain doubles),
    plus a smaller weight on bench EV so substitutes are usable cover.
    Optional quality tie-break: lexicographic (EV first) or soft blend.
    """
    from fpl_robo.config import OPTIMISER_TIEBREAK
    from fpl_robo.optimize.tiebreak import (
        attach_quality_scores,
        solve_lexicographic,
        tiebreak_enabled,
        tiebreak_mode,
    )
    from fpl_robo.optimize.kg_features import prediction_row_extras

    tb_cfg = dict(OPTIMISER_TIEBREAK)
    if tiebreak_config:
        tb_cfg.update(tiebreak_config)
    if use_tiebreak is None:
        use_tiebreak = tiebreak_enabled(tb_cfg)
    else:
        tb_cfg["enabled"] = bool(use_tiebreak)

    df = players.copy()
    if df.empty:
        return _empty_solution("Empty player pool")

    if use_tiebreak:
        df = attach_quality_scores(df, config=tb_cfg)
    elif "quality" not in df.columns:
        df["quality"] = 0.0

    gameweek = int(df["gameweek"].dropna().iloc[0]) if "gameweek" in df else None

    if exclude_unavailable and "status" in df.columns:
        df = df[~df["status"].isin(["u", "s"])].copy()

    if min_play_chance is not None and "chance_of_playing_next_round" in df.columns:
        chance = df["chance_of_playing_next_round"]
        df = df[(chance.isna()) | (chance >= min_play_chance)].copy()

    if "cover_eligible" not in df.columns:
        df["cover_eligible"] = True
    if require_cover_eligible:
        # Entire squad must be potential minutes cover — dead fodder is useless.
        df = df[df["cover_eligible"].astype(bool)].copy()

    df = df[df["ev"] > 0].copy()
    df = df.reset_index(drop=True)
    if len(df) < SQUAD_SIZE:
        return _empty_solution("Not enough eligible players after filters")

    ids = df["id"].tolist()
    ev = dict(zip(df["id"], df["ev"]))
    quality = dict(zip(df["id"], df["quality"].astype(float)))
    price = dict(zip(df["id"], df["price"]))
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
    prob = pulp.LpProblem("fpl_single_gw", pulp.LpMaximize)
    in_squad = pulp.LpVariable.dicts("squad", ids, cat="Binary")
    start = pulp.LpVariable.dicts("start", ids, cat="Binary")
    captain = pulp.LpVariable.dicts("captain", ids, cat="Binary")

    # Objective: XI EV + extra captain EV + soft bench-cover EV (+ optional quality)
    weight = max(0.0, float(bench_ev_weight))
    gk_weight = max(0.0, float(BENCH_COVER.get("bench_gk_ev_weight", weight)))
    bench_w = {
        i: (gk_weight if str(pos[i]).upper() in {"GK", "GKP"} else weight) for i in ids
    }
    q_w = float(tb_cfg.get("quality_weight") or 0.0) if use_tiebreak else 0.0
    mode = tiebreak_mode(tb_cfg) if use_tiebreak else "soft"
    primary = (
        pulp.lpSum(ev[i] * start[i] for i in ids)
        + pulp.lpSum(ev[i] * captain[i] for i in ids)
        + pulp.lpSum(bench_w[i] * ev[i] * (in_squad[i] - start[i]) for i in ids)
    )
    secondary = None
    if q_w:
        secondary = q_w * (
            pulp.lpSum(quality[i] * start[i] for i in ids)
            + pulp.lpSum(quality[i] * captain[i] for i in ids)
            + pulp.lpSum(
                bench_w[i] * quality[i] * (in_squad[i] - start[i]) for i in ids
            )
        )

    # Squad size / budget / club limit
    prob += pulp.lpSum(in_squad[i] for i in ids) == SQUAD_SIZE
    prob += pulp.lpSum(price[i] * in_squad[i] for i in ids) <= budget

    for club in set(team.values()):
        club_players = [i for i in ids if team[i] == club]
        prob += pulp.lpSum(in_squad[i] for i in club_players) <= MAX_PER_CLUB

    for position, count in SQUAD_COUNTS.items():
        pos_players = [i for i in ids if pos[i] == position]
        prob += pulp.lpSum(in_squad[i] for i in pos_players) == count

    # Starting XI subset of squad
    for i in ids:
        prob += start[i] <= in_squad[i]
        prob += captain[i] <= start[i]

    prob += pulp.lpSum(start[i] for i in ids) == STARTING_SIZE
    prob += pulp.lpSum(captain[i] for i in ids) == 1

    for position, (lo, hi) in STARTING_BOUNDS.items():
        pos_players = [i for i in ids if pos[i] == position]
        if not pos_players:
            continue
        prob += pulp.lpSum(start[i] for i in pos_players) >= lo
        prob += pulp.lpSum(start[i] for i in pos_players) <= hi

    from fpl_robo.optimize.attack_floor import apply_attack_floor_constraint

    apply_attack_floor_constraint(
        prob, in_squad=in_squad, ids=ids, pos=pos, ep_next=ep_next
    )

    if use_tiebreak and mode == "lexicographic" and secondary is not None:
        status_code = solve_lexicographic(
            prob,
            primary,
            secondary,
            tolerance=float(tb_cfg.get("ev_tolerance") or 1e-3),
        )
        objective = float(pulp.value(primary) or 0.0)
    else:
        obj = primary + secondary if secondary is not None else primary
        prob.setObjective(obj)
        status_code = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        objective = float(pulp.value(prob.objective) or 0.0)

    status = pulp.LpStatus[status_code]
    if status != "Optimal":
        return _empty_solution(status, gameweek=gameweek)

    selected = [i for i in ids if in_squad[i].value() > 0.5]
    starters = [i for i in ids if start[i].value() > 0.5]
    cap_ids = [i for i in ids if captain[i].value() > 0.5]
    captain_id = cap_ids[0] if cap_ids else None

    def row(i: int, *, is_start: bool, is_captain: bool, is_vice: bool = False) -> dict[str, Any]:
        base = {
            "id": i,
            "web_name": name[i],
            "position": pos[i],
            "team_id": int(team[i]),
            "price": float(price[i]),
            "ev": float(ev[i]),
            "start": is_start,
            "captain": is_captain,
            "vice_captain": is_vice,
        }
        base.update(meta_by_id.get(int(i)) or {})
        return base

    squad_rows = [
        row(i, is_start=(i in starters), is_captain=(i == captain_id))
        for i in selected
    ]
    squad_rows.sort(key=lambda r: (r["position"], -r["ev"]))
    vice_id = pick_vice_captain_id(squad_rows, captain_id)
    for r in squad_rows:
        r["vice_captain"] = vice_id is not None and int(r["id"]) == int(vice_id)
    xi_rows = [r for r in squad_rows if r["start"]]
    bench_rows = [r for r in squad_rows if not r["start"]]
    budget_used = round(sum(price[i] for i in selected), 1)

    return SquadSolution(
        status=status,
        objective=round(objective, 3),
        budget_used=budget_used,
        gameweek=gameweek,
        squad=squad_rows,
        starting_xi=xi_rows,
        bench=bench_rows,
        captain_id=captain_id,
        captain_name=name.get(captain_id) if captain_id else None,
        vice_captain_id=vice_id,
        vice_captain_name=name.get(vice_id) if vice_id else None,
    )


def _empty_solution(status: str, gameweek: int | None = None) -> SquadSolution:
    return SquadSolution(
        status=status,
        objective=0.0,
        budget_used=0.0,
        gameweek=gameweek,
        squad=[],
        starting_xi=[],
        bench=[],
        captain_id=None,
        captain_name=None,
        vice_captain_id=None,
        vice_captain_name=None,
    )
