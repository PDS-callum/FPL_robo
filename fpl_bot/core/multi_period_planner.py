"""
Multi-Period MIP Optimizer for FPL Bot

Uses Mixed Integer Programming to optimize:
- Team selection across 7 gameweeks
- Transfer decisions and banking (up to 5 FTs)
- Chip usage timing based on actual planned teams
- Budget management

This is the default planning method.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

try:
    from pulp import *
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

# Import refactored MIP optimizer
try:
    from .multi_period_planner_refactored import MultiPeriodPlannerRefactored
    REFACTORED_AVAILABLE = True
except ImportError:
    REFACTORED_AVAILABLE = False
    print("  WARNING: Refactored MIP not available")

# Import V3 scenario-based optimizer  
# DISABLED for now - needs more testing
V3_AVAILABLE = False
# try:
#     from .multi_period_planner_v3 import MultiPeriodPlannerV3
#     V3_AVAILABLE = True
# except ImportError:
#     V3_AVAILABLE = False
#     print("  WARNING: V3 optimizer not available")


class MultiPeriodPlanner:
    """7-gameweek MIP optimizer with chip planning"""
    
    def __init__(self, data_collector, predictor, chip_manager):
        self.data_collector = data_collector
        self.predictor = predictor
        self.chip_manager = chip_manager
        
        # Initialize V3 scenario-based planner (preferred)
        if V3_AVAILABLE:
            self.v3_planner = MultiPeriodPlannerV3(
                data_collector, predictor, chip_manager
            )
            print("  Using V3 scenario-based optimizer")
            self.refactored_planner = None
        elif REFACTORED_AVAILABLE:
            self.refactored_planner = MultiPeriodPlannerRefactored(
                data_collector, predictor, chip_manager
            )
            self.v3_planner = None
            print("  Using refactored MIP optimizer")
        else:
            self.refactored_planner = None
            self.v3_planner = None
        
        # Constants
        self.horizon = 7  # Plan 7 gameweeks ahead
        self.transfer_cost = 4
        self.max_free_transfers = 5  # Maximum free transfers you can bank
        # (Optional) Wildcard encouragement can be handled via objective tie-breakers.
        
        # Squad constraints
        self.squad_size = 15
        self.position_min = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        self.position_max = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        
    def plan_gameweeks(self,
                        current_team: List[int],
                        current_gw: int,
                        budget: float,
                        free_transfers: int,
                        predictions_df: pd.DataFrame,
                        fixtures_df: pd.DataFrame,
                        teams_data: List[Dict],
                        chip_status: Dict) -> Dict:
        """
        Create comprehensive multi-gameweek plan including:
        - Weekly team composition
        - Transfer decisions
        - Chip usage timing
        
        Workflow:
        1. Use ML model to predict player scores for each week with matchup analysis
        2. Identify favorable matchup windows for strong teams
        3. Use MIP to optimize team towards predicted optimal team each week
        4. Optimize chip usage within the MIP to maximize total points
        """
        # Dynamically determine planning horizon per business rules:
        # - If current_gw <= 19: plan from current_gw up to GW19 OR next 7 GWs, whichever is larger
        # - If current_gw > 19: plan until end of season (GW38)
        season_end_gw = 38
        if current_gw is not None:
            if current_gw <= 19:
                to_gw19 = 19 - current_gw + 1
                self.horizon = max(7, to_gw19)
            else:
                self.horizon = max(1, season_end_gw - current_gw + 1)
        # If current_gw is None, retain existing default horizon

        print(f"\nOptimizing next {self.horizon} gameweeks (GW{current_gw}-{current_gw + self.horizon - 1})...")
        
        # Step 1: Use ML model to predict multi-gameweek performance with matchup analysis
        print("  Step 1: Running ML predictions and identifying favorable matchup windows...")
        ml_predictions = self.predictor.predict_multi_gameweek(
            predictions_df, fixtures_df, teams_data, self.horizon
        )
        player_projections = ml_predictions['player_predictions']
        favorable_matchups = ml_predictions['favorable_matchups']
        
        
        # Print identified matchup windows
        if favorable_matchups:
            print(f"  Found {len(favorable_matchups)} favorable matchup windows:")
            for i, window in enumerate(favorable_matchups[:3], 1):
                print(f"    {i}. {window['team_name']} (GW{window['start_gw']}-{window['end_gw']}): "
                      f"{window['length']} easy fixtures, avg difficulty {window['avg_difficulty']:.2f}")
        
        # Step 2: Run MIP optimization using ML predictions
        print(f"  Step 2: Running MIP optimization...")
        if not PULP_AVAILABLE:
            raise RuntimeError("MIP optimization requires PuLP. Install with: pip install pulp")
        
        team_evolution = self._mip_optimize_with_ml_predictions(
            current_team,
            current_gw,
            budget,
            free_transfers,
            player_projections,
            predictions_df,
            chip_status,
            favorable_matchups,
            teams_data,
            fixtures_df
        )
        
        # Step 3: Extract chip plan from team evolution (MIP already optimized chip usage)
        chip_plan = self._extract_chip_plan_from_evolution(
            team_evolution,
            player_projections,
            chip_status,
            current_gw
        )
        
        # Step 4: Generate recommendations
        recommendations = self._generate_multi_period_recommendations(
            team_evolution,
            chip_plan,
            favorable_matchups,
            current_gw
        )
        
        return {
            'horizon': self.horizon,
            'start_gw': current_gw,
            'end_gw': current_gw + self.horizon - 1,
            'team_evolution': team_evolution,
            'chip_plan': chip_plan,
            'fixture_runs': favorable_matchups,  # Use favorable matchups instead
            'recommendations': recommendations,
            'player_projections': player_projections
        }
    
    def _project_multi_period_points(self,
                                     predictions_df: pd.DataFrame,
                                     fixtures_df: pd.DataFrame,
                                     teams_data: List[Dict],
                                     start_gw: int) -> Dict:
        """Project each player's points for next N GWs with fixture adjustments"""
        
        teams_dict = {t['id']: t for t in teams_data}
        projections = {}
        
        for _, player in predictions_df.iterrows():
            player_id = player['player_id']
            team_id = player.get('team')
            position = player.get('position_name', 'MID')
            base_points = player['predicted_points']
            
            gw_points = {}
            
            for offset in range(self.horizon):
                gw = start_gw + offset
                
                # Find fixture
                fixture_row = fixtures_df[
                    ((fixtures_df['team_h'] == team_id) | (fixtures_df['team_a'] == team_id)) &
                    (fixtures_df['event'] == gw)
                ]
                
                if fixture_row.empty:
                    gw_points[gw] = 0  # Blank gameweek
                    continue
                
                fixture = fixture_row.iloc[0]
                is_home = fixture['team_h'] == team_id
                opponent_id = fixture['team_a'] if is_home else fixture['team_h']
                opponent = teams_dict.get(opponent_id, {})
                
                # Calculate fixture difficulty
                difficulty = self.chip_manager._calculate_fixture_difficulty_simple(
                    is_home, opponent, teams_dict.get(team_id, {})
                )
                
                # Position-specific fixture adjustment
                if position in ['DEF', 'GK']:
                    adj = (3.0 - difficulty) * 0.5  # ±1.0 pts
                elif position == 'MID':
                    adj = (3.0 - difficulty) * 0.35  # ±0.7 pts
                else:  # FWD
                    adj = (3.0 - difficulty) * 0.25  # ±0.5 pts
                
                gw_points[gw] = max(0, round(base_points + adj, 2))
            
            projections[player_id] = {
                'player_id': player_id,
                'web_name': player['web_name'],
                'team': player.get('team_name', 'Unknown'),
                'position': position,
                'cost': player['cost'],
                'base_points': base_points,
                'gw_points': gw_points,
                'total_horizon': sum(gw_points.values())
            }
        
        return projections
    
    # REMOVED: Heuristic planner (_plan_team_evolution) - only MIP optimization is used now
    
    def _mip_optimize_with_ml_predictions(self,
                                          current_team: List[int],
                                          start_gw: int,
                                          bank: float,
                                          free_transfers_start: int,
                                          player_projections: Dict,
                                          predictions_df: pd.DataFrame,
                                          chip_status: Dict,
                                          favorable_matchups: List[Dict],
                                          teams_data: List[Dict],
                                          fixtures_df: pd.DataFrame = None) -> Dict:
        """
        MIP optimization using ML predictions to optimize towards best team each GW
        
        Key improvements:
        1. Uses ML predictions which already factor in matchup windows
        2. Optimizes towards the best predicted team for each gameweek
        3. Integrates chip decisions to maximize total points
        4. Properly models transfer costs (4 points per extra transfer)
        """
        # Use V3 scenario-based optimizer if available (preferred)
        if self.v3_planner:
            print("  Using V3 SCENARIO-BASED optimizer...")
            result = self.v3_planner.optimize(
                current_team=current_team,
                current_gw=start_gw,
                budget=bank,
                free_transfers=free_transfers_start,
                player_projections=player_projections,
                predictions_df=predictions_df,
                chip_status=chip_status
            )
            return result['team_evolution']
        elif self.refactored_planner:
            print("  Using REFACTORED MIP optimizer...")
            result = self.refactored_planner.optimize_multi_period(
                current_team=current_team,
                current_gw=start_gw,
                budget=bank,
                free_transfers=free_transfers_start,
                player_projections=player_projections,
                predictions_df=predictions_df,
                chip_status=chip_status,
                fixtures_df=fixtures_df,
                teams_data=teams_data,
                favorable_matchups=favorable_matchups
            )
            return result['team_evolution']
        else:
            # Fallback to old optimizer
            print("  WARNING: Using LEGACY MIP optimizer (may be infeasible)...")
            return self._mip_optimize_team_evolution(
                current_team, start_gw, bank, free_transfers_start,
                player_projections, predictions_df, chip_status,
                favorable_matchups, teams_data, fixtures_df
            )
    
    def _mip_optimize_team_evolution(self,
                                     current_team: List[int],
                                     start_gw: int,
                                     bank: float,
                                     free_transfers_start: int,
                                     player_projections: Dict,
                                     predictions_df: pd.DataFrame,
                                     chip_status: Dict,
                                     fixture_runs: List[Dict],
                                     teams_data: List[Dict],
                                     fixtures_df: pd.DataFrame = None) -> Dict:
        """
        Multi-period MIP to jointly optimize squad evolution and starting XI
        across the planning horizon with transfer hits.

        Key features:
        - Free transfer banking modeled explicitly (up to 5)
        - Transfer hits cost 4 points per extra transfer beyond available FTs
        - Wildcard and Free Hit chips supported (no hit cost on chip weeks)
        - Squad cost at each GW cannot exceed initial squad cost + bank
        - Candidate pool trimmed for tractability
        """
        # Build candidate set: current team + top candidates by total horizon points
        # and top per-GW candidates; enforce club max later
        pos_map = predictions_df.set_index('player_id')['position_name'].to_dict()
        cost_map = predictions_df.set_index('player_id')['cost'].to_dict()
        team_map = predictions_df.set_index('player_id')['team_name'].to_dict()
        team_id_map = predictions_df.set_index('player_id')['team'].to_dict()

        # Helper function to get predicted points for a player in a gameweek
        def get_gw_points(pid, gw):
            """Get predicted points for a player in a gameweek"""
            if pid not in player_projections:
                return 0
            proj = player_projections[pid]
            # Try new format first, then old format
            if 'gameweek_predictions' in proj:
                return proj['gameweek_predictions'].get(gw, 0)
            elif 'gw_points' in proj:
                return proj['gw_points'].get(gw, 0)
            return 0

        # Ensure only players with projections are considered
        def has_proj(pid):
            return pid in player_projections

        # Filter current team - keep all players that have any projection
        original_team = current_team.copy()
        current_team = [pid for pid in current_team if has_proj(pid)]
        
        if len(current_team) == 0:
            print(f"  WARNING: Current team was filtered to 0 players - restoring original")
            current_team = original_team

        # Rank candidates by total horizon points
        cand_df = []
        for pid, proj in player_projections.items():
            if pid not in pos_map:
                continue
            # Calculate total using new or old format
            if 'total_horizon_points' in proj:
                total = proj['total_horizon_points']
            elif 'total_horizon' in proj:
                total = proj['total_horizon']
            else:
                # Calculate from gameweek predictions
                if 'gameweek_predictions' in proj:
                    total = sum(proj['gameweek_predictions'].values())
                elif 'gw_points' in proj:
                    total = sum(proj['gw_points'].values())
                else:
                    total = 0
            
            cand_df.append({
                'player_id': pid,
                'position': pos_map[pid],
                'total': total
            })
        cand_df = pd.DataFrame(cand_df)
        if cand_df.empty:
            return self._plan_team_evolution(current_team, start_gw, bank, free_transfers_start, player_projections, predictions_df)

        # Per-position caps for candidate pool by horizon total
        # IMPROVEMENT: Expanded pool for better coverage and differential options
        caps_total = {'GK': 10, 'DEF': 25, 'MID': 40, 'FWD': 20}
        selected_ids = set(current_team)
        for pos, cap in caps_total.items():
            pos_ids = cand_df[cand_df['position'] == pos].sort_values('total', ascending=False)['player_id'].tolist()
            selected_ids.update(pos_ids[:cap])

        # Also include per-GW top candidates (captures one-off great matchups)
        T = [start_gw + t for t in range(self.horizon)]
        caps_weekly = {'GK': 8, 'DEF': 20, 'MID': 25, 'FWD': 15}
        for t in T:
            # Build list of (pid, pos, pts) for this GW
            per_gw = []
            for pid, proj in player_projections.items():
                if pid not in pos_map:
                    continue
                pts = get_gw_points(pid, t)
                if pts is None:
                    pts = 0
                per_gw.append((pid, pos_map[pid], pts))
            if per_gw:
                per_gw_df = pd.DataFrame(per_gw, columns=['player_id', 'position', 'gw_pts'])
                for pos, cap in caps_weekly.items():
                    pos_ids = per_gw_df[per_gw_df['position'] == pos].sort_values('gw_pts', ascending=False)['player_id'].tolist()
                    selected_ids.update(pos_ids[:cap])

        candidates = [pid for pid in selected_ids if has_proj(pid)]
        
        # Precompute initial squad cost
        initial_cost = sum(cost_map.get(pid, 0) for pid in current_team)
        cost_cap = initial_cost + bank

        # Time indices
        # T already defined above

        # Build model
        model = LpProblem("FPL_MultiPeriod_Optimization", LpMaximize)

        # Decision variables
        y = LpVariable.dicts('in_squad', [(p, t) for p in candidates for t in T], 0, 1, cat=LpBinary)
        x = LpVariable.dicts('in_xi', [(p, t) for p in candidates for t in T], 0, 1, cat=LpBinary)
        cpt = LpVariable.dicts('captain', [(p, t) for p in candidates for t in T], 0, 1, cat=LpBinary)
        vcpt = LpVariable.dicts('vice_captain', [(p, t) for p in candidates for t in T], 0, 1, cat=LpBinary)
        w_in = LpVariable.dicts('transfer_in', [(p, t) for p in candidates for t in T[1:]], 0, 1, cat=LpBinary)
        w_out = LpVariable.dicts('transfer_out', [(p, t) for p in candidates for t in T[1:]], 0, 1, cat=LpBinary)
        transfers = LpVariable.dicts('transfers', [t for t in T[1:]], lowBound=0, cat=LpInteger)
        extra_transfers = LpVariable.dicts('extra_transfers', [t for t in T[1:]], lowBound=0, cat=LpInteger)
        # Free transfer banking (0..5) and usage
        ft_start = LpVariable.dicts('ft_start', [t for t in T], lowBound=0, upBound=5, cat=LpInteger)
        free_used = LpVariable.dicts('free_used', [t for t in T[1:]], lowBound=0, upBound=5, cat=LpInteger)
        spill = LpVariable.dicts('ft_spill', [t for t in T[1:]], lowBound=0, cat=LpContinuous)
        # Wildcard decision (at most one in horizon if available)
        wc = LpVariable.dicts('wildcard', [t for t in T], 0, 1, cat=LpBinary)
        # Free Hit decision (at most one in horizon if available)
        fh = LpVariable.dicts('free_hit', [t for t in T], 0, 1, cat=LpBinary)
        # Triple Captain & Bench Boost decisions (at most one each across horizon)
        tc = LpVariable.dicts('triple_captain', [t for t in T], 0, 1, cat=LpBinary)
        bb = LpVariable.dicts('bench_boost', [t for t in T], 0, 1, cat=LpBinary)
        # Auxiliary variables to linearize TC/BB extra points
        z_tc = LpVariable.dicts('tc_extra_points', [t for t in T], lowBound=0, cat=LpContinuous)
        z_bb = LpVariable.dicts('bb_extra_points', [t for t in T], lowBound=0, cat=LpContinuous)
        # Budget tracking (disabled for now - was causing infeasibility)
        # budget_remaining = LpVariable.dicts('budget_remaining', [t for t in T], lowBound=0, cat=LpContinuous)
        # excess_budget = LpVariable.dicts('excess_budget', [t for t in T], lowBound=0, cat=LpContinuous)

        wildcard_available = 1 if chip_status.get('wildcard', {}).get('available', False) else 0
        if wildcard_available == 0:
            for t in T:
                model += wc[t] == 0
        else:
            model += lpSum(wc[t] for t in T) <= 1

        free_hit_available = 1 if chip_status.get('free_hit', {}).get('available', False) else 0
        if free_hit_available == 0:
            for t in T:
                model += fh[t] == 0
        else:
            model += lpSum(fh[t] for t in T) <= 1

        # Triple Captain availability
        tc_available = 1 if chip_status.get('triple_captain', {}).get('available', False) else 0
        if tc_available == 0:
            for t in T:
                model += tc[t] == 0
        else:
            model += lpSum(tc[t] for t in T) <= 1

        # Bench Boost availability
        bb_available = 1 if chip_status.get('bench_boost', {}).get('available', False) else 0
        if bb_available == 0:
            for t in T:
                model += bb[t] == 0
        else:
            model += lpSum(bb[t] for t in T) <= 1

        # Objective: XI points + captain bonus + vice-captain injury insurance + TC/BB + run bonus - hits
        xi_points = []
        cap_points = []
        vice_points = []  # New: vice-captain expected value from injury risk
        run_bonus_terms = []
        # Precompute safe big-Ms for TC and BB linking
        max_cap_pts_by_t = {}
        max_bench_pts_by_t = {}
        for t in T:
            pts_list = [get_gw_points(p, t) for p in candidates]
            max_cap_pts_by_t[t] = max(pts_list) if pts_list else 0
            max_bench_pts_by_t[t] = sum(pts_list) if pts_list else 0
        # Build run bonus map: (team_id, gw) -> bonus
        run_bonus_map = {}
        for run in (fixture_runs or []):
            team_id = run.get('team_id')
            quality = float(run.get('quality_score', 0))
            for fx in run.get('fixtures', []):
                gw = int(fx.get('gw', 0))
                # scale bonus moderately to guide, not dominate
                run_bonus_map[(team_id, gw)] = run_bonus_map.get((team_id, gw), 0.0) + min(1.0, max(0.0, quality / 50.0))

        # Time-based discounting: Value near-term points more than distant future
        # This prevents churning by making distant swaps less valuable
        # Discount factor: 0.95 per week (5% decay) - prioritizes near-term while considering horizon
        for idx_t, t in enumerate(T):
            weeks_ahead = idx_t
            discount_factor = 0.95 ** weeks_ahead  # 1.0 for current week, 0.95 for next week, 0.90 for week 2, etc.
            
            # Sum expressions per t
            cap_sum_expr = []
            bench_sum_expr = []
            for p in candidates:
                pts = get_gw_points(p, t)
                # Apply time discount to all point values
                discounted_pts = pts * discount_factor
                xi_points.append(x[(p, t)] * discounted_pts)
                cap_points.append(cpt[(p, t)] * discounted_pts)
                
                # Vice-captain insurance: contributes points if captain injured
                # Expected value = injury_prob(captain) * vice_points  
                # Keep this VERY small to avoid overwhelming the objective or causing infeasibility
                injury_risk = player_projections.get(p, {}).get('injury_prob', 0.05)
                vice_insurance_value = injury_risk * discounted_pts * 0.05  # 5% weight (very small)
                vice_points.append(vcpt[(p, t)] * vice_insurance_value)
                
                bench_sum_expr.append((y[(p, t)] - x[(p, t)]) * discounted_pts)
                # run bonus
                pid_team_id = team_id_map.get(p)
                if pid_team_id is not None:
                    bonus = run_bonus_map.get((pid_team_id, t), 0.0)
                    if bonus > 0:
                        run_bonus_terms.append(y[(p, t)] * bonus * discount_factor)
            # Linearize TC extra: z_tc[t] == cap_sum if tc[t]==1 else 0
            cap_sum = lpSum(cap_sum_expr)
            M_tc = max_cap_pts_by_t.get(t, 0)
            model += z_tc[t] <= cap_sum
            model += z_tc[t] <= M_tc * tc[t]
            model += z_tc[t] >= cap_sum - (1 - tc[t]) * M_tc
            model += z_tc[t] >= 0
            # Linearize BB extra: z_bb[t] == bench_sum if bb[t]==1 else 0
            bench_sum = lpSum(bench_sum_expr)
            M_bb = max_bench_pts_by_t.get(t, 0)
            model += z_bb[t] <= bench_sum
            model += z_bb[t] <= M_bb * bb[t]
            model += z_bb[t] >= bench_sum - (1 - bb[t]) * M_bb
            model += z_bb[t] >= 0
        # Hit penalty: -4 pts per extra transfer beyond free transfers
        # Time discounting handles churning prevention, so just use standard FPL penalty
        hit_penalty = []
        for t in T[1:]:
            # Standard -4 pts per hit (no additional stability penalty needed)
            hit_penalty.append(4.0 * extra_transfers[t])
        
        # Swap penalty: Penalize buying back players that were recently sold
        # For each player, if they're transferred out at week t1 and back in at week t2 (within 4 weeks),
        # add a penalty proportional to how soon they're re-bought
        swap_penalty_terms = []
        for idx1 in range(len(T) - 1):
            t1 = T[idx1 + 1]  # Week when transfer out happens (T[1:])
            for idx2 in range(idx1 + 1, min(idx1 + 5, len(T))):  # Check next 1-4 weeks
                t2 = T[idx2 + 1] if idx2 < len(T) - 1 else None
                if t2 is None or t2 not in T[1:]:
                    continue
                weeks_gap = idx2 - idx1
                # Penalty decreases with gap: 3 pts if immediate, 2 pts if 1 week gap, 1 pt if 2+ weeks
                penalty = max(1.0, 4.0 - weeks_gap)
                for p in candidates:
                    # Create auxiliary variable for w_out[t1] * w_in[t2] interaction
                    z_swap = LpVariable(f'swap_{p}_{t1}_{t2}', 0, 1, cat=LpBinary)
                    model += z_swap <= w_out[(p, t1)]
                    model += z_swap <= w_in[(p, t2)]
                    model += z_swap >= w_out[(p, t1)] + w_in[(p, t2)] - 1
                    swap_penalty_terms.append(penalty * z_swap)
        
        # Use-it-or-lose-it tie-break: encourage using Wildcard before GW19 (small positive bonus)
        wc_bonus_points = 0
        try:
            pre_christmas_weeks = [t for t in T if t <= 19]
            if pre_christmas_weeks:
                wc_bonus_points = lpSum(0.5 * wc[t] for t in pre_christmas_weeks)
        except Exception:
            wc_bonus_points = 0
        
        # Build team strength map and fixture map (used by both TC and FH optimizations)
        team_strength_map = {}
        team_name_to_id = {}
        if teams_data:
            for team in teams_data:
                team_id = team.get('id')
                team_name = team.get('name', '')
                # Use overall team strength (higher = better attack/defense)
                strength_overall = team.get('strength', 3)
                strength_attack = team.get('strength_attack_home', 1000) + team.get('strength_attack_away', 1000)
                strength_defense = team.get('strength_defence_home', 1000) + team.get('strength_defence_away', 1000)
                
                team_strength_map[team_id] = {
                    'overall': strength_overall,
                    'attack': strength_attack,
                    'defense': strength_defense
                }
                team_name_to_id[team_name] = team_id
        
        # Build fixture map: team_id -> {gw: {opponent_id, is_home, difficulty}}
        fixture_map = {}
        if fixtures_df is not None and not fixtures_df.empty:
            for _, fixture in fixtures_df.iterrows():
                gw = fixture.get('event')
                if gw is None or gw not in T:
                    continue
                
                team_h = fixture.get('team_h')
                team_a = fixture.get('team_a')
                difficulty_h = fixture.get('team_h_difficulty', 3)
                difficulty_a = fixture.get('team_a_difficulty', 3)
                
                # Home team fixture
                if team_h not in fixture_map:
                    fixture_map[team_h] = {}
                fixture_map[team_h][gw] = {
                    'opponent_id': team_a,
                    'is_home': True,
                    'difficulty': difficulty_h
                }
                
                # Away team fixture
                if team_a not in fixture_map:
                    fixture_map[team_a] = {}
                fixture_map[team_a][gw] = {
                    'opponent_id': team_h,
                    'is_home': False,
                    'difficulty': difficulty_a
                }
        
        # TC Fixture Quality Bonus: Reward TC on weeks with favorable matchups
        # Calculate fixture quality bonus for each week based on captain strength vs opposition
        tc_bonus_points = 0
        if tc_available > 0:
            
            # Create auxiliary variables for TC-captain interaction: z_tc_cpt[(p,t)] = tc[t] * cpt[(p,t)]
            z_tc_cpt = LpVariable.dicts('tc_captain_interaction', [(p, t) for p in candidates for t in T], 0, 1, cat=LpBinary)
            
            # Linearization constraints and bonus calculation
            tc_matchup_bonus_terms = []
            for t in T:
                for p in candidates:
                    # Linearize z_tc_cpt[(p,t)] = tc[t] * cpt[(p,t)]
                    model += z_tc_cpt[(p, t)] <= tc[t]
                    model += z_tc_cpt[(p, t)] <= cpt[(p, t)]
                    model += z_tc_cpt[(p, t)] >= tc[t] + cpt[(p, t)] - 1
                    
                    # Calculate matchup quality for this player-week
                    pid_team_id = team_id_map.get(p)
                    if pid_team_id is None:
                        continue
                    
                    pts = get_gw_points(p, t)
                    if pts <= 5.0:  # Only consider decent captain options
                        continue
                    
                    # Get fixture info from fixture_map
                    fixture_info = fixture_map.get(pid_team_id, {}).get(t, {})
                    opponent_id = fixture_info.get('opponent_id')
                    is_home = fixture_info.get('is_home', False)
                    fixture_difficulty = fixture_info.get('difficulty', 3)
                    
                    # Calculate matchup quality
                    player_team_str = team_strength_map.get(pid_team_id, {})
                    opponent_str = team_strength_map.get(opponent_id, {}) if opponent_id else {}
                    
                    # Strength differential - higher is better matchup for TC
                    player_attack = player_team_str.get('attack', 2000)
                    opponent_defense = opponent_str.get('defense', 2000)
                    
                    # Calculate matchup advantage (higher = easier fixture)
                    matchup_advantage = player_attack - opponent_defense
                    
                    # Base bonus based on fixture difficulty (1=very hard, 5=very easy)
                    difficulty_bonus = 0
                    if fixture_difficulty == 1:  # Very hard
                        difficulty_bonus = 0
                    elif fixture_difficulty == 2:  # Hard
                        difficulty_bonus = 0.5
                    elif fixture_difficulty == 3:  # Medium
                        difficulty_bonus = 1.0
                    elif fixture_difficulty == 4:  # Easy
                        difficulty_bonus = 2.0
                    elif fixture_difficulty == 5:  # Very easy
                        difficulty_bonus = 3.0
                    
                    # Home advantage bonus
                    if is_home:
                        difficulty_bonus += 0.5
                    
                    # Extra bonus for strong team vs weak team (strength differential)
                    if matchup_advantage > 200:  # Strong attack vs weak defense
                        difficulty_bonus += 1.5
                    elif matchup_advantage > 100:
                        difficulty_bonus += 1.0
                    
                    # Scale by predicted points (higher points = more valuable TC target)
                    if pts >= 8.0:  # High predicted points
                        difficulty_bonus *= 1.5
                    elif pts >= 7.0:  # Good predicted points
                        difficulty_bonus *= 1.2
                    elif pts >= 6.0:  # Decent predicted points
                        difficulty_bonus *= 1.0
                    else:
                        difficulty_bonus *= 0.7  # Moderate points
                    
                    # Add bonus when TC is used on this captain
                    if difficulty_bonus > 0:
                        tc_matchup_bonus_terms.append(z_tc_cpt[(p, t)] * difficulty_bonus)
            
            # Combine base TC usage bonus + matchup quality bonus
            base_tc_bonus = lpSum(0.3 * tc[t] for t in T)
            tc_matchup_bonus = lpSum(tc_matchup_bonus_terms) if tc_matchup_bonus_terms else 0
            tc_bonus_points = base_tc_bonus + tc_matchup_bonus
        
        # Free Hit Fixture Quality Bonus: Reward FH on weeks with many favorable fixtures
        # FH should be used when you can field a completely different team with much better fixtures
        fh_bonus_points = 0
        if free_hit_available > 0:
            fh_fixture_bonus_terms = []
            
            for t in T:
                # Calculate aggregate fixture quality for this gameweek
                week_fixture_quality = 0
                favorable_players = 0  # Count of players with good fixtures
                
                for p in candidates:
                    pid_team_id = team_id_map.get(p)
                    if pid_team_id is None:
                        continue
                    
                    pts = get_gw_points(p, t)
                    if pts <= 4.0:  # Only consider viable options
                        continue
                    
                    # Get fixture info from fixture_map
                    fixture_info = fixture_map.get(pid_team_id, {}).get(t, {})
                    is_home = fixture_info.get('is_home', False)
                    fixture_difficulty = fixture_info.get('difficulty', 3)
                    opponent_id = fixture_info.get('opponent_id')
                    
                    # Calculate individual player fixture quality
                    player_fixture_score = 0
                    
                    # Base on fixture difficulty (4-5 are favorable)
                    if fixture_difficulty == 5:  # Very easy
                        player_fixture_score = 3.0
                    elif fixture_difficulty == 4:  # Easy
                        player_fixture_score = 2.0
                    elif fixture_difficulty == 3:  # Medium
                        player_fixture_score = 0.5
                    else:  # Hard fixtures (1-2)
                        player_fixture_score = 0
                    
                    # Home advantage
                    if is_home:
                        player_fixture_score += 0.3
                    
                    # Strength differential bonus
                    if opponent_id:
                        player_team_str = team_strength_map.get(pid_team_id, {})
                        opponent_str = team_strength_map.get(opponent_id, {})
                        player_attack = player_team_str.get('attack', 2000)
                        opponent_defense = opponent_str.get('defense', 2000)
                        matchup_advantage = player_attack - opponent_defense
                        
                        if matchup_advantage > 200:
                            player_fixture_score += 0.5
                        elif matchup_advantage > 100:
                            player_fixture_score += 0.3
                    
                    # Weight by predicted points (better players matter more)
                    if pts >= 7.0:
                        player_fixture_score *= 1.3
                    elif pts >= 6.0:
                        player_fixture_score *= 1.1
                    
                    # Count favorable fixtures
                    if player_fixture_score >= 2.0:  # Good fixture
                        favorable_players += 1
                        week_fixture_quality += player_fixture_score
                
                # Free Hit is valuable when MANY top players have good fixtures
                # This represents a "fixture swing" opportunity
                if favorable_players >= 15:  # Many good fixtures available
                    # Strong week for FH - can build completely different optimal team
                    fh_week_bonus = min(10.0, week_fixture_quality * 0.3)
                elif favorable_players >= 10:
                    # Decent week for FH
                    fh_week_bonus = min(6.0, week_fixture_quality * 0.2)
                elif favorable_players >= 7:
                    # Moderate week for FH
                    fh_week_bonus = min(3.0, week_fixture_quality * 0.1)
                else:
                    # Not enough good fixtures to warrant FH
                    fh_week_bonus = 0
                
                # Add bonus when FH is used on this week
                if fh_week_bonus > 0:
                    fh_fixture_bonus_terms.append(fh[t] * fh_week_bonus)
            
            # Only add base bonus if there's potential value (don't force FH usage)
            if fh_fixture_bonus_terms:
                fh_fixture_bonus = lpSum(fh_fixture_bonus_terms)
                fh_bonus_points = fh_fixture_bonus
            else:
                # No good weeks found - don't incentivize using FH
                fh_bonus_points = 0
        
        # Budget opportunity cost: DISABLED (was causing infeasibility)
        # budget_waste_penalty = lpSum(excess_budget[t] * 0.01 for t in T)
        budget_waste_penalty = 0
        
        # Differential/Ownership logic: DISABLED to avoid numerical issues
        differential_bonus_terms = []
        # differential_mode = 'balanced'
        # if differential_mode == 'balanced':
        #     for t in T:
        #         for p in candidates:
        #             ownership = player_projections.get(p, {}).get('ownership_pct', 50)
        #             if ownership < 30:
        #                 diff_bonus = (30 - ownership) / 100 * 0.02
        #                 differential_bonus_terms.append(y[(p, t)] * diff_bonus)
        
        model.setObjective(
            lpSum(xi_points)  # Discounted playing XI points
            + lpSum(cap_points)  # Discounted normal captain doubling
            + lpSum(vice_points)  # Vice-captain injury insurance value
            + lpSum(z_bb[t] for t in T)  # Bench boost extra points
            + lpSum(z_tc[t] for t in T)  # Extra captain points from TC
            + (lpSum(run_bonus_terms) if run_bonus_terms else 0)  # Favorable fixture run bonuses
            - lpSum(hit_penalty)  # -4.0 pts per hit
            - (lpSum(swap_penalty_terms) if swap_penalty_terms else 0)  # Penalty for re-buying recently sold players
            - budget_waste_penalty  # Penalize excess unused budget (opportunity cost)
            + (lpSum(differential_bonus_terms) if differential_bonus_terms else 0)  # Differential/ownership bonus
            + wc_bonus_points
            + tc_bonus_points  # TC matchup quality bonus
            + fh_bonus_points  # FH fixture swing opportunity bonus
        )

        # Constraints per week
        for t in T:
            # Squad size and composition
            model += lpSum(y[(p, t)] for p in candidates) == self.squad_size
            model += lpSum(y[(p, t)] for p in candidates if pos_map.get(p) == 'GK') == self.position_max['GK']
            model += lpSum(y[(p, t)] for p in candidates if pos_map.get(p) == 'DEF') == self.position_max['DEF']
            model += lpSum(y[(p, t)] for p in candidates if pos_map.get(p) == 'MID') == self.position_max['MID']
            model += lpSum(y[(p, t)] for p in candidates if pos_map.get(p) == 'FWD') == self.position_max['FWD']

            # Starting XI subset and formation
            model += lpSum(x[(p, t)] for p in candidates) == 11
            # XI positional constraints
            model += lpSum(x[(p, t)] for p in candidates if pos_map.get(p) == 'GK') == 1
            model += lpSum(x[(p, t)] for p in candidates if pos_map.get(p) == 'DEF') >= 3
            model += lpSum(x[(p, t)] for p in candidates if pos_map.get(p) == 'MID') >= 2
            model += lpSum(x[(p, t)] for p in candidates if pos_map.get(p) == 'FWD') >= 1
            # XI must be subset of squad, except on Free Hit week
            for p in candidates:
                model += x[(p, t)] <= y[(p, t)] + fh[t]
                model += cpt[(p, t)] <= x[(p, t)]
                model += vcpt[(p, t)] <= x[(p, t)]  # Vice must be in XI
            # Exactly one captain
            model += lpSum(cpt[(p, t)] for p in candidates) == 1
            # Exactly one vice-captain
            model += lpSum(vcpt[(p, t)] for p in candidates) == 1
            # Vice-captain cannot be captain (mutually exclusive)
            for p in candidates:
                model += cpt[(p, t)] + vcpt[(p, t)] <= 1

            # Budget cap approximation on squad
            squad_cost = lpSum(cost_map.get(p, 0) * y[(p, t)] for p in candidates)
            model += squad_cost <= cost_cap

            # Club constraint: max 3 from any club
            # Build clubs from candidate mapping; fallback to projections if needed
            clubs = set()
            for p in candidates:
                club = team_map.get(p, player_projections.get(p, {}).get('team'))
                if club is not None:
                    clubs.add(club)
            for club in clubs:
                model += lpSum(y[(p, t)] for p in candidates if (team_map.get(p, player_projections.get(p, {}).get('team')) == club)) <= 3

            # Free Hit: impose realistic XI constraints when fh[t] == 1 and ensure chip exclusivity
            # - XI club cap (<=3 per club)
            # - XI budget cap (approximate, using overall cost cap)
            # These constraints are relaxed when fh[t] == 0 via big-M.
            M_relax = 300  # sufficiently large to relax constraints when not Free Hit
            for club in clubs:
                model += lpSum(x[(p, t)] for p in candidates if (team_map.get(p, player_projections.get(p, {}).get('team')) == club)) <= 3 + (1 - fh[t]) * M_relax
            model += lpSum(cost_map.get(p, 0) * x[(p, t)] for p in candidates) <= cost_cap + (1 - fh[t]) * M_relax
            # Only one chip can be used per week
            model += wc[t] + fh[t] + tc[t] + bb[t] <= 1

        # Initial squad fixed to current team
        t0 = T[0]
        
        # Safety check: if current team is empty, we cannot constrain initial squad
        if len(current_team) == 0:
            print(f"  ERROR: Current team is empty! Cannot optimize without a starting squad.")
            print(f"  This is likely an issue with the manager analyzer not fetching team data.")
            # Return a fallback empty evolution
            return {t: {'gw': t, 'team': [], 'transfers': [], 'num_transfers': 0, 
                       'transfer_cost': 0, 'expected_points': 0} for t in T}
        
        for p in candidates:
            if p in current_team:
                model += y[(p, t0)] == 1
            else:
                model += y[(p, t0)] == 0
        # Allow wildcard and free hit in the first planned GW (we already select next unplayed GW)

        # Initialize starting free transfers before weekly transition loop
        model += ft_start[T[0]] == min(5, int(free_transfers_start or 1))

        # Transfer dynamics and hit cost (from t-1 to t)
        M = 30  # big-M for wildcard/Free Hit related constraints
        for idx in range(1, len(T)):
            t_prev = T[idx - 1]
            t = T[idx]
            # Cannot use wildcard and free hit in the same week
            model += wc[t] + fh[t] <= 1
            # Transfer-in and transfer-out indicator linearization
            for p in candidates:
                # w_in = 1 if player added this week
                model += w_in[(p, t)] >= y[(p, t)] - y[(p, t_prev)]
                model += w_in[(p, t)] <= y[(p, t)]
                model += w_in[(p, t)] <= 1 - y[(p, t_prev)]
                # w_out = 1 if player removed this week
                model += w_out[(p, t)] >= y[(p, t_prev)] - y[(p, t)]
                model += w_out[(p, t)] <= y[(p, t_prev)]
                model += w_out[(p, t)] <= 1 - y[(p, t)]
            model += transfers[t] == lpSum(w_in[(p, t)] for p in candidates)

            # Free transfer usage and hit cost modeling with banking (up to 5)
            # Free transfers used cannot exceed transfers or available FTs at start of the week
            model += free_used[t] <= transfers[t]
            model += free_used[t] <= ft_start[t_prev]
            # Extra transfers beyond free used incur -4 unless chip used
            model += extra_transfers[t] >= transfers[t] - free_used[t] - M * wc[t] - M * fh[t]
            model += extra_transfers[t] >= 0
            # No hit cost if wildcard or free hit
            model += extra_transfers[t] <= (1 - wc[t]) * M
            model += extra_transfers[t] <= (1 - fh[t]) * M
            
            # IMPROVEMENT: Enforce minimum changes when wildcard is used (makes it meaningful)
            # If wc[t] == 1, ensure at least 8 transfers are made
            model += transfers[t] >= 8 * wc[t]
            
            # Simplified FT banking (without complex big-M constraints that cause infeasibility)
            # Basic rule: ft_start[t] = ft_start[t_prev] - free_used[t] + 1, capped at 5
            # If wildcard used in t_prev, reset to 1
            # Approximation: Just enforce reasonable bounds and let objective optimize
            model += ft_start[t] <= 5  # Cap at 5
            model += ft_start[t] >= 1  # At least 1 FT per week
            # Link to previous week (relaxed constraint)
            model += ft_start[t] <= ft_start[t_prev] + 1  # Can gain at most 1 FT
            model += ft_start[t] >= ft_start[t_prev] - free_used[t] - 1  # Can lose FTs used (with buffer)

            # Free Hit squad reversion: keep permanent squad unchanged on FH week
            # When fh[t] == 1, enforce y[:, t] == y[:, t_prev] (no permanent transfers)
            for p in candidates:
                model += y[(p, t)] - y[(p, t_prev)] <= (1 - fh[t])
                model += y[(p, t_prev)] - y[(p, t)] <= (1 - fh[t])

        # Solve with timeout
        print(f"  Solving MIP with {len(candidates)} candidates over {len(T)} gameweeks...")
        solver = PULP_CBC_CMD(msg=False, timeLimit=30)  # 30 second timeout
        model.solve(solver)
        
        if model.status != 1:  # Not optimal
            print(f"  WARNING: MIP solver status: {LpStatus[model.status]}")
            # Try to diagnose infeasibility
            if model.status == -1 or LpStatus[model.status] == 'Infeasible':
                print(f"  ERROR: MIP is INFEASIBLE - constraints cannot be satisfied!")
                print(f"  Possible causes:")
                print(f"    - Budget constraints too tight")
                print(f"    - Captain/Vice-captain constraints conflicting")
                print(f"    - Free transfer banking constraints invalid")
                print(f"  Continuing with best-effort solution...")

        # Build evolution result
        evolution = {}
        # Extract y to build teams and transfers
        for idx, t in enumerate(T):
            team_players = [p for p in candidates if value(y[(p, t)]) >= 0.5]
            xi_players = [p for p in candidates if value(x[(p, t)]) >= 0.5]
            captain_id = None
            for p in candidates:
                if value(cpt[(p, t)]) >= 0.5:
                    captain_id = p
                    break
            
            # Safety check: Must have a captain
            if not captain_id:
                print(f"  ERROR: GW{t} - No captain selected by MIP! Selecting best from XI...")
                if xi_players:
                    xi_with_points = [(p, get_gw_points(p, t)) for p in xi_players]
                    xi_with_points.sort(key=lambda x: x[1], reverse=True)
                    captain_id = xi_with_points[0][0]
                    print(f"  Fixed: Captain set to {player_projections.get(captain_id, {}).get('web_name', 'Unknown')} ({xi_with_points[0][1]:.1f} pts)")
            
            # Safety check: Captain must be in XI
            elif captain_id and captain_id not in xi_players:
                print(f"  ERROR: GW{t} - Captain {player_projections.get(captain_id, {}).get('web_name', captain_id)} NOT in XI!")
                print(f"  XI players: {[player_projections.get(p, {}).get('web_name', p) for p in xi_players[:5]]}...")
                # Fix: Select best player from actual XI
                xi_with_points = [(p, get_gw_points(p, t)) for p in xi_players]
                xi_with_points.sort(key=lambda x: x[1], reverse=True)
                if xi_with_points:
                    captain_id = xi_with_points[0][0]
                    print(f"  Fixed: Captain changed to {player_projections.get(captain_id, {}).get('web_name', 'Unknown')}")
            
            # Find vice captain (from MIP decision, not heuristic)
            vice_captain_id = None
            for p in candidates:
                if value(vcpt[(p, t)]) >= 0.5:
                    vice_captain_id = p
                    break
            
            # Safety check: Must have a vice-captain
            if not vice_captain_id:
                print(f"  ERROR: GW{t} - No vice-captain selected by MIP! Selecting second-best from XI...")
                if xi_players and captain_id:
                    xi_with_points = [(p, get_gw_points(p, t)) for p in xi_players if p != captain_id]
                    xi_with_points.sort(key=lambda x: x[1], reverse=True)
                    if xi_with_points:
                        vice_captain_id = xi_with_points[0][0]
                        print(f"  Fixed: Vice-captain set to {player_projections.get(vice_captain_id, {}).get('web_name', 'Unknown')} ({xi_with_points[0][1]:.1f} pts)")
            
            # Safety check: Vice-captain must be in XI and different from captain
            elif vice_captain_id and vice_captain_id not in xi_players:
                print(f"  ERROR: GW{t} - Vice-captain {player_projections.get(vice_captain_id, {}).get('web_name', vice_captain_id)} NOT in XI!")
                # Fix: Select second-best from XI
                xi_with_points = [(p, get_gw_points(p, t)) for p in xi_players if p != captain_id]
                xi_with_points.sort(key=lambda x: x[1], reverse=True)
                if xi_with_points:
                    vice_captain_id = xi_with_points[0][0]
                    print(f"  Fixed: Vice-captain changed to {player_projections.get(vice_captain_id, {}).get('web_name', 'Unknown')}")
            
            # Safety check: Captain and vice-captain must be different
            if captain_id and vice_captain_id and captain_id == vice_captain_id:
                print(f"  WARNING: GW{t} - Same player selected as C and VC! Fixing...")
                # Find alternative vice-captain (second-best in XI)
                xi_with_points = [(p, get_gw_points(p, t)) for p in xi_players if p != captain_id]
                xi_with_points.sort(key=lambda x: x[1], reverse=True)
                if xi_with_points:
                    vice_captain_id = xi_with_points[0][0]
                    print(f"  Fixed: Vice-captain changed to {player_projections.get(vice_captain_id, {}).get('web_name', 'Unknown')}")
            
            transfers_list = []
            if idx > 0:
                t_prev = T[idx - 1]
                ins = [p for p in candidates if value(y[(p, t_prev)]) < 0.5 and value(y[(p, t)]) >= 0.5]
                outs = [p for p in candidates if value(y[(p, t_prev)]) >= 0.5 and value(y[(p, t)]) < 0.5]
                
                # Group ins and outs by position for proper pairing
                ins_by_pos = {'GK': [], 'DEF': [], 'MID': [], 'FWD': []}
                outs_by_pos = {'GK': [], 'DEF': [], 'MID': [], 'FWD': []}
                
                for pid_in in ins:
                    pos = pos_map.get(pid_in, 'MID')
                    ins_by_pos[pos].append(pid_in)
                
                for pid_out in outs:
                    pos = pos_map.get(pid_out, 'MID')
                    outs_by_pos[pos].append(pid_out)
                
                # Pair by position
                for pos in ['GK', 'DEF', 'MID', 'FWD']:
                    for i in range(min(len(ins_by_pos[pos]), len(outs_by_pos[pos]))):
                        pid_in = ins_by_pos[pos][i]
                        pid_out = outs_by_pos[pos][i]
                        gain = get_gw_points(pid_in, t) - get_gw_points(pid_out, t)
                        
                        in_name = player_projections.get(pid_in, {}).get('web_name', str(pid_in))
                        out_name = player_projections.get(pid_out, {}).get('web_name', str(pid_out))
                        
                        transfers_list.append({
                            'out': pid_out,
                            'in': pid_in,
                            'out_name': out_name,
                            'in_name': in_name,
                            'gain': round(gain, 2),
                            'cost_change': cost_map.get(pid_in, 0) - cost_map.get(pid_out, 0)
                        })

            expected_pts = sum(
                get_gw_points(pid, t)
                for pid in xi_players
                if pid in player_projections
            )
            
            # For Free Hit weeks, identify temporary players (in XI but not in squad)
            fh_active = value(fh[t]) >= 0.5 if t in fh else False
            fh_players_in = []
            if fh_active and idx > 0:
                # Players in XI but not in permanent squad this week
                for pid in xi_players:
                    if pid not in team_players:
                        player_name = player_projections.get(pid, {}).get('web_name', str(pid))
                        fh_players_in.append({
                            'player_id': pid,
                            'name': player_name,
                            'predicted_points': get_gw_points(pid, t)
                        })

            evolution[t] = {
                'gw': t,
                'team': team_players,
                'starting_xi': xi_players,
                'captain_id': captain_id,
                'captain_name': player_projections.get(captain_id, {}).get('web_name') if captain_id else None,
                'vice_captain_id': vice_captain_id,
                'vice_captain_name': player_projections.get(vice_captain_id, {}).get('web_name') if vice_captain_id else None,
                'transfers': transfers_list,
                'num_transfers': len(transfers_list),
                'transfer_cost': 0 if (value(wc[t]) >= 0.5 or value(fh[t]) >= 0.5) else int(max(0, value(extra_transfers[t]) if t in extra_transfers else 0)) * self.transfer_cost,
                'free_transfers_used': 0 if (value(wc[t]) >= 0.5 or value(fh[t]) >= 0.5) else int(value(free_used[t]) if t in free_used else 0),
                'free_transfers_available': int(value(ft_start[t])) if t in ft_start and value(ft_start[t]) is not None else 0,
                'budget_remaining': max(0.0, cost_cap - sum(cost_map.get(pid, 0) for pid in team_players)),
                'expected_points': round(expected_pts, 1),
                'chip': 'wildcard' if value(wc[t]) >= 0.5 else ('free_hit' if value(fh[t]) >= 0.5 else ('triple_captain' if value(tc[t]) >= 0.5 else ('bench_boost' if value(bb[t]) >= 0.5 else None))),
                'fh_players_in': fh_players_in if fh_active else []
            }

        return evolution
    
    def _identify_valuable_multi_transfer_weeks(self,
                                                current_team: List[int],
                                                start_gw: int,
                                                budget: float,
                                                player_projections: Dict,
                                                predictions_df: pd.DataFrame) -> Dict:
        """
        Look ahead to identify weeks where having multiple FTs would be highly valuable
        
        Returns dict of {gw: {'expected_gain': float, 'num_transfers': int}}
        """
        valuable_weeks = {}
        
        # Check each future gameweek for multi-transfer opportunities
        for offset in range(1, min(self.horizon, 4)):  # Look 3 weeks ahead
            gw = start_gw + offset
            
            # Simulate having 2-3 FTs and see what gain we could get
            multi_transfer_rec = self._optimize_single_gw_transfers(
                current_team, gw, ft_available=3, budget=budget,
                player_projections=player_projections, predictions_df=predictions_df
            )
            
            num_transfers = len(multi_transfer_rec.get('transfers', []))
            gain = multi_transfer_rec.get('gain', 0)
            
            # A week is valuable if 2+ transfers give significant gain
            if num_transfers >= 2 and gain >= 2.5:
                valuable_weeks[gw] = {
                    'expected_gain': gain,
                    'num_transfers': num_transfers
                }
        
        return valuable_weeks
    
    def _optimize_single_gw_transfers(self,
                                     current_team: List[int],
                                     gw: int,
                                     ft_available: int,
                                     budget: float,
                                     player_projections: Dict,
                                     predictions_df: pd.DataFrame) -> Dict:
        """Optimize transfers for a single gameweek - can find multiple transfers"""
        
        current_team_df = predictions_df[predictions_df['player_id'].isin(current_team)]
        available_df = predictions_df[~predictions_df['player_id'].isin(current_team)]
        
        # Find all beneficial transfers
        all_transfers = []
        
        for _, player_out in current_team_df.iterrows():
            out_id = player_out['player_id']
            out_points = player_projections.get(out_id, {}).get('gw_points', {}).get(gw, 0)
            out_cost = player_out['cost']
            position = player_out['position_name']
            
            # Find same position replacements
            same_position = available_df[
                (available_df['position_name'] == position) &
                (available_df['cost'] <= out_cost + budget + 5)  # Allow some budget flexibility
            ]
            
            for _, player_in in same_position.iterrows():
                in_id = player_in['player_id']
                in_points = player_projections.get(in_id, {}).get('gw_points', {}).get(gw, 0)
                in_cost = player_in['cost']
                
                gain = in_points - out_points
                cost_change = in_cost - out_cost
                
                if gain > 0 and cost_change <= budget:
                    # BONUS: Check if this transfer sets up a good chip opportunity
                    # Premium players with good fixture runs get a boost
                    fixture_run_bonus = 0
                    if in_cost >= 8.5:  # Premium player
                        # Check next 3 GWs for good fixtures
                        future_gws = [gw + offset for offset in range(1, 4)]
                        future_points = [player_projections.get(in_id, {}).get('gw_points', {}).get(future_gw, 0) 
                                       for future_gw in future_gws]
                        avg_future_points = sum(future_points) / len(future_points) if future_points else 0
                        
                        # If premium player has strong fixtures ahead, boost transfer value
                        if avg_future_points >= 6.0:
                            fixture_run_bonus = 1.0  # +1 pt for good fixture run
                    
                    all_transfers.append({
                        'out': out_id,
                        'in': in_id,
                        'out_name': player_out['web_name'],
                        'in_name': player_in['web_name'],
                        'gain': gain + fixture_run_bonus,
                        'immediate_gain': gain,  # Store original gain
                        'fixture_run_bonus': fixture_run_bonus,
                        'cost_change': cost_change,
                        'position': position,
                        'in_cost': in_cost
                    })
        
        if not all_transfers:
            return {'transfers': [], 'ft_used': 0, 'cost': 0, 'gain': 0}
        
        # Sort by gain
        all_transfers.sort(key=lambda x: x['gain'], reverse=True)
        
        # Greedy selection: pick top transfers up to FT limit (max 3 for practicality)
        max_transfers = min(ft_available, 3)
        selected_transfers = []
        total_gain = 0
        used_out = set()
        used_in = set()
        budget_used = 0
        
        for transfer in all_transfers:
            # Check if we can add this transfer
            if len(selected_transfers) >= max_transfers:
                break
            
            # Check not already using this player
            if transfer['out'] in used_out or transfer['in'] in used_in:
                continue
            
            # Check budget
            if budget_used + transfer['cost_change'] > budget:
                continue
            
            # Add transfer
            selected_transfers.append(transfer)
            total_gain += transfer['gain']
            used_out.add(transfer['out'])
            used_in.add(transfer['in'])
            budget_used += transfer['cost_change']
        
        # Calculate cost
        if len(selected_transfers) > 0:
            ft_used = min(len(selected_transfers), ft_available)
            hits = max(0, len(selected_transfers) - ft_available)
            cost = hits * self.transfer_cost
            
            return {
                'transfers': selected_transfers,
                'ft_used': ft_used,
                'cost': cost,
                'gain': total_gain
            }
        else:
            return {
                'transfers': [],
                'ft_used': 0,
                'cost': 0,
                'gain': 0
            }
    
    # REMOVED: Heuristic wrapper (_plan_team_evolution_with_ml) - only MIP optimization is used now
    
    def _extract_chip_plan_from_evolution(self,
                                          team_evolution: Dict,
                                          player_projections: Dict,
                                          chip_status: Dict,
                                          start_gw: int) -> Dict:
        """
        Extract chip usage plan from team evolution
        
        MIP already optimized chip usage, so we extract what it decided
        """
        chip_plan = {}
        
        # Initialize chip plan for each chip type
        for chip_name in ['wildcard', 'free_hit', 'triple_captain', 'bench_boost']:
            chip_plan[chip_name] = {
                'chip': chip_name,
                'best_gw': None,
                'expected_benefit': 0,
                'recommended': False,
                'details': {}
            }
        
        # Extract chip usage from team evolution
        # Track which chips have been used to prevent duplicates
        chips_used = set()
        
        for gw, week_plan in team_evolution.items():
            chip_used = week_plan.get('chip')
            
            if chip_used:
                # Check if this chip has already been used (bug fix for infeasible MIP solutions)
                if chip_used in chips_used:
                    print(f"  WARNING: Chip '{chip_used}' appears multiple times (GW{gw}). Skipping duplicate - keeping first occurrence.")
                    # Remove duplicate chip from this week's plan
                    week_plan['chip'] = None
                    continue
                
                # Mark this chip as used
                chips_used.add(chip_used)
                
                # Calculate benefit based on chip type
                benefit = 0
                details = {}
                
                if chip_used == 'triple_captain':
                    captain_id = week_plan.get('captain_id')
                    if captain_id and captain_id in player_projections:
                        captain_proj = player_projections[captain_id]
                        captain_pts = captain_proj.get('gameweek_predictions', {}).get(gw, 0)
                        benefit = captain_pts * 2  # TC doubles captain points
                        details = {
                            'captain_id': captain_id,
                            'captain_name': captain_proj.get('web_name', 'Unknown'),
                            'captain_points': captain_pts
                        }
                
                elif chip_used == 'bench_boost':
                    team_players = week_plan.get('team', [])
                    starting_xi = week_plan.get('starting_xi', [])
                    bench = [p for p in team_players if p not in starting_xi]
                    
                    bench_pts = sum(
                        player_projections.get(p, {}).get('gameweek_predictions', {}).get(gw, 0)
                        for p in bench
                    )
                    benefit = bench_pts
                    details = {'bench_points': bench_pts}
                
                elif chip_used in ['wildcard', 'free_hit']:
                    # Benefit is already accounted for in transfer savings
                    benefit = week_plan.get('expected_points', 0)
                    details = {'expected_points': benefit}
                
                chip_plan[chip_used] = {
                    'chip': chip_used,
                    'best_gw': gw,
                    'expected_benefit': round(benefit, 1),
                    'recommended': True,
                    'details': details
                }
        
        return chip_plan
    
    def _optimize_chips_with_team_evolution(self,
                                           team_evolution: Dict,
                                           player_projections: Dict,
                                           chip_status: Dict,
                                           start_gw: int) -> Dict:
        """
        Optimize chip timing considering actual planned team for each GW
        
        This is the KEY improvement - chips are optimized based on the
        team you'll ACTUALLY have in each gameweek, not current team
        
        NEW: Explicitly tracks new transfers and prioritizes them in chip planning
        """
        
        chip_plans = {}
        used_gws = set()  # Track which GWs have chips (max 1 per GW)
        
        # Track players brought in across all gameweeks (for TC prioritization)
        new_player_ids = set()
        for gw, week_plan in team_evolution.items():
            transfers = week_plan.get('transfers', [])
            for transfer in transfers:
                new_player_ids.add(transfer.get('in'))
        
        # Priority order for chip planning
        chip_priority = ['triple_captain', 'bench_boost', 'wildcard', 'free_hit']
        
        for chip_name in chip_priority:
            if chip_name not in chip_status or not chip_status[chip_name].get('available', False):
                continue
            
            best_gw = None
            best_benefit = 0
            best_details = {}
            
            # Evaluate each GW with the ACTUAL team planned for that week
            for gw, week_plan in team_evolution.items():
                if gw in used_gws:
                    continue  # Already using another chip this week
                
                team_for_gw = week_plan['team']
                
                if chip_name == 'triple_captain':
                    # Find best captain for this specific GW's team composition
                    # This properly considers ALL players including new signings
                    captain_analysis = self._analyze_tc_for_gw(
                        team_for_gw, gw, player_projections, new_player_ids
                    )
                    benefit = captain_analysis['benefit']
                    
                    # Track if TC would be on a new signing (for reporting purposes)
                    if captain_analysis.get('captain_id') in new_player_ids:
                        captain_analysis['is_new_signing'] = True
                    
                    if benefit > best_benefit:
                        best_benefit = benefit
                        best_gw = gw
                        best_details = captain_analysis
                
                elif chip_name == 'bench_boost':
                    # Calculate bench strength for this GW
                    bb_analysis = self._analyze_bb_for_gw(
                        team_for_gw, gw, player_projections
                    )
                    benefit = bb_analysis['benefit']
                    
                    if benefit > best_benefit:
                        best_benefit = benefit
                        best_gw = gw
                        best_details = bb_analysis
                elif chip_name == 'wildcard':
                    # If MIP selected a wildcard week, surface it
                    if not chip_status.get('wildcard', {}).get('available', False):
                        continue
                    if week_plan.get('chip') == 'wildcard':
                        prev_plan = team_evolution.get(gw - 1, {})
                        prev_pts = prev_plan.get('expected_points', 0)
                        curr_pts = week_plan.get('expected_points', 0)
                        benefit = max(0, curr_pts - prev_pts)
                        if benefit >= best_benefit:
                            best_benefit = benefit
                            best_gw = gw
                            best_details = {
                                'expected_points_after': curr_pts,
                                'expected_points_before': prev_pts,
                                'delta': round(benefit, 1)
                            }
                elif chip_name == 'free_hit':
                    if not chip_status.get('free_hit', {}).get('available', False):
                        continue
                    if week_plan.get('chip') == 'free_hit':
                        prev_plan = team_evolution.get(gw - 1, {})
                        prev_pts = prev_plan.get('expected_points', 0)
                        curr_pts = week_plan.get('expected_points', 0)
                        benefit = max(0, curr_pts - prev_pts)
                        if benefit >= best_benefit:
                            best_benefit = benefit
                            best_gw = gw
                            best_details = {
                                'expected_points_after': curr_pts,
                                'expected_points_before': prev_pts,
                                'delta': round(benefit, 1)
                            }
            
            # Store chip plan
            if best_gw:
                chip_plans[chip_name] = {
                    'chip': chip_name,
                    'best_gw': best_gw,
                    'expected_benefit': round(best_benefit, 1),
                    'recommended': best_benefit >= 8.0,  # Threshold for recommendation
                    'details': best_details
                }
                
                # Mark this GW as used
                if chip_plans[chip_name]['recommended']:
                    used_gws.add(best_gw)
            else:
                chip_plans[chip_name] = {
                    'chip': chip_name,
                    'best_gw': None,
                    'expected_benefit': 0,
                    'recommended': False,
                    'details': {}
                }
        
        return chip_plans
    
    def _analyze_tc_for_gw(self,
                          team: List[int],
                          gw: int,
                          player_projections: Dict,
                          new_player_ids: set = None) -> Dict:
        """
        Analyze Triple Captain potential for specific GW with specific team
        
        Evaluates ALL players in the team fairly (including new signings) to find
        the best TC option based on predicted points.
        
        Args:
            team: List of player IDs in the team for this GW
            gw: Gameweek number
            player_projections: Point projections for all players
            new_player_ids: Set of player IDs that were transferred in (for tracking only)
        """
        
        # Find best captain based purely on predicted points for this GW
        best_captain_id = None
        best_captain_name = None
        best_captain_pts = 0
        captain_options = []
        
        new_player_ids = new_player_ids or set()
        
        for pid in team:
            if pid not in player_projections:
                continue
            
            proj = player_projections[pid]
            pts = proj['gw_points'].get(gw, 0)
            cost = proj.get('cost', 0)
            
            is_new = pid in new_player_ids
            is_premium = cost >= 8.5
            
            captain_options.append({
                'player_id': pid,
                'web_name': proj['web_name'],
                'position': proj['position'],
                'predicted_points': pts,
                'is_new_signing': is_new,
                'is_premium': is_premium,
                'cost': cost
            })
            
            # Select best captain purely on predicted points
            if pts > best_captain_pts:
                best_captain_pts = pts
                best_captain_name = proj['web_name']
                best_captain_id = pid
        
        # Sort by predicted points (fair comparison)
        captain_options.sort(key=lambda x: x['predicted_points'], reverse=True)
        
        return {
            'benefit': best_captain_pts * 2,  # TC gives 2x captain points
            'captain_id': best_captain_id,
            'captain_name': best_captain_name,
            'captain_points': best_captain_pts,
            'all_options': captain_options[:5],  # Top 5 options
            'is_new_signing': best_captain_id in new_player_ids
        }
    
    def _analyze_bb_for_gw(self,
                          team: List[int],
                          gw: int,
                          player_projections: Dict) -> Dict:
        """Analyze Bench Boost potential for specific GW with specific team"""
        
        # Sort team by predicted points for this GW
        team_with_points = []
        for pid in team:
            if pid in player_projections:
                proj = player_projections[pid]
                pts = proj['gw_points'].get(gw, 0)
                team_with_points.append({
                    'player_id': pid,
                    'web_name': proj['web_name'],
                    'position': proj['position'],
                    'predicted_points': pts
                })
        
        # Sort by points (highest first)
        team_with_points.sort(key=lambda x: x['predicted_points'], reverse=True)
        
        # Starting XI = top 11, Bench = positions 12-15
        bench = team_with_points[11:15] if len(team_with_points) >= 15 else []
        
        bench_total = sum(p['predicted_points'] for p in bench)
        
        return {
            'benefit': bench_total,
            'bench_points': bench_total,
            'bench_players': bench,
            'starting_xi': team_with_points[:11]
        }
    
    def _identify_quality_fixture_runs(self,
                                       fixtures_df: pd.DataFrame,
                                       teams_data: List[Dict],
                                       start_gw: int) -> List[Dict]:
        """Identify fixture runs for QUALITY teams only"""
        
        if fixtures_df.empty:
            return []
        
        runs = []
        teams_dict = {t['id']: t for t in teams_data}
        end_gw = start_gw + self.horizon - 1
        
        # Only analyze top 10 teams
        top_teams = [t for t in teams_data if t.get('position', 20) <= 10]
        
        for team in top_teams:
            team_id = team['id']
            
            # Get fixtures for this team
            team_fixtures = fixtures_df[
                ((fixtures_df['team_h'] == team_id) | (fixtures_df['team_a'] == team_id)) &
                (fixtures_df['event'] >= start_gw) &
                (fixtures_df['event'] <= end_gw)
            ].sort_values('event')
            
            if len(team_fixtures) < 3:
                continue
            
            # Calculate difficulties
            fixture_sequence = []
            for _, fixture in team_fixtures.iterrows():
                is_home = fixture['team_h'] == team_id
                opponent_id = fixture['team_a'] if is_home else fixture['team_h']
                opponent = teams_dict.get(opponent_id, {})
                
                difficulty = self.chip_manager._calculate_fixture_difficulty_simple(
                    is_home, opponent, team
                )
                
                fixture_sequence.append({
                    'gw': fixture['event'],
                    'opponent': opponent.get('name', 'Unknown'),
                    'is_home': is_home,
                    'difficulty': difficulty
                })
            
            # Find runs of 3+ favorable fixtures
            current_run = []
            for fixture in fixture_sequence:
                if fixture['difficulty'] <= 2.8:  # Easy/Very Easy threshold
                    current_run.append(fixture)
                else:
                    if len(current_run) >= 3:
                        self._add_fixture_run(runs, team, current_run, teams_dict)
                    current_run = []
            
            # Final run
            if len(current_run) >= 3:
                self._add_fixture_run(runs, team, current_run, teams_dict)
        
        # Sort by quality score
        runs.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
        
        return runs
    
    def _add_fixture_run(self, runs: List, team: Dict, fixtures: List, teams_dict: Dict):
        """Add a fixture run to the list with quality scoring"""
        
        team_position = team.get('position', 20)
        avg_diff = np.mean([f['difficulty'] for f in fixtures])
        
        # Quality score: team strength + fixture ease
        position_bonus = max(0, 15 - team_position)
        fixture_bonus = (4.0 - avg_diff) * len(fixtures)
        quality_score = position_bonus + fixture_bonus
        
        # Classification
        if team_position <= 6:
            recommendation = "PREMIUM TARGET"
        elif team_position <= 10:
            recommendation = "VALUE OPTION"
        else:
            recommendation = "AVOID (low quality)"
        
        runs.append({
            'team_id': team['id'],
            'team_name': team['name'],
            'team_position': team_position,
            'start_gw': fixtures[0]['gw'],
            'end_gw': fixtures[-1]['gw'],
            'length': len(fixtures),
            'avg_difficulty': avg_diff,
            'quality_score': quality_score,
            'recommendation': recommendation,
            'fixtures': fixtures
        })
    
    def _generate_multi_period_recommendations(self,
                                              team_evolution: Dict,
                                              chip_plan: Dict,
                                              fixture_runs: List[Dict],
                                              start_gw: int) -> Dict:
        """Generate comprehensive multi-period recommendations"""
        
        recommendations = {
            'immediate_actions': [],
            'future_plans': [],
            'chip_recommendations': [],
            'fixture_opportunities': []
        }
        
        # Immediate transfer action (GW1)
        gw1_plan = team_evolution.get(start_gw, {})
        if gw1_plan.get('num_transfers', 0) > 0:
            transfers = gw1_plan['transfers']
            for t in transfers:
                recommendations['immediate_actions'].append(
                    f"Transfer: {t.get('out_name', 'Unknown')} -> {t.get('in_name', 'Unknown')} "
                    f"(+{t.get('gain', 0):.1f} pts)"
                )
        else:
            recommendations['immediate_actions'].append("Hold transfers this week (bank FT)")
        
        # Future transfer plans (GW2-5)
        for gw in range(start_gw + 1, start_gw + self.horizon):
            if gw in team_evolution:
                plan = team_evolution[gw]
                if plan.get('num_transfers', 0) > 0:
                    recommendations['future_plans'].append(
                        f"GW{gw}: Plan {plan['num_transfers']} transfer(s)"
                    )
        
        # Chip recommendations
        for chip_name, chip_info in chip_plan.items():
            if chip_info['recommended']:
                details = chip_info.get('details', {})
                
                if chip_name == 'triple_captain':
                    captain = details.get('captain_name', 'Unknown')
                    pts = details.get('captain_points', 0)
                    recommendations['chip_recommendations'].append(
                        f"Use Triple Captain in GW{chip_info['best_gw']} on {captain} (~{pts:.1f} pts, 2x = {chip_info['expected_benefit']:.1f} pts total)"
                    )
                elif chip_name == 'bench_boost':
                    bench_pts = details.get('bench_points', 0)
                    recommendations['chip_recommendations'].append(
                        f"Use Bench Boost in GW{chip_info['best_gw']} (~{bench_pts:.1f} pts from bench)"
                    )
        
        # Fixture run opportunities (top 6 teams only)
        premium_runs = [r for r in fixture_runs if r['team_position'] <= 6]
        for run in premium_runs[:2]:  # Top 2 only
            recommendations['fixture_opportunities'].append(
                f"{run['team_name']} has {run['length']} easy fixtures (GW{run['start_gw']}-{run['end_gw']}) - "
                f"Target their premium players"
            )
        
        return recommendations

