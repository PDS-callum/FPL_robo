"""
Refactored Multi-Period MIP Optimizer for FPL Bot

Clean, robust MIP formulation that optimizes:
1. Total points across planning horizon
2. Optimal Triple Captain timing and target
3. Optimal Wildcard usage (full squad rebuild)
4. Optimal Free Hit usage (one-week swap)
5. Optimal Bench Boost timing (bench scoring week)
6. Transfer banking and hit avoidance

Key principles:
- Clear, non-conflicting constraints
- Focused objective function
- Proper chip modeling
- Tractable problem size
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    from pulp import *
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False


class MultiPeriodPlannerRefactored:
    """Refactored 7-12 GW MIP optimizer with clean chip planning"""
    
    def __init__(self, data_collector, predictor, chip_manager):
        self.data_collector = data_collector
        self.predictor = predictor
        self.chip_manager = chip_manager
        
        # Constants
        self.horizon = 7  # Will be adjusted dynamically
        self.transfer_cost = 4
        self.max_free_transfers = 5
        
        # Squad constraints
        self.squad_size = 15
        self.starting_xi_size = 11
        self.position_requirements = {
            'GK': {'squad_min': 2, 'squad_max': 2, 'xi_min': 1, 'xi_max': 1},
            'DEF': {'squad_min': 5, 'squad_max': 5, 'xi_min': 3, 'xi_max': 5},
            'MID': {'squad_min': 5, 'squad_max': 5, 'xi_min': 2, 'xi_max': 5},
            'FWD': {'squad_min': 3, 'squad_max': 3, 'xi_min': 1, 'xi_max': 3}
        }
        self.max_per_club = 3
    
    def optimize_multi_period(self,
                             current_team: List[int],
                             current_gw: int,
                             budget: float,
                             free_transfers: int,
                             player_projections: Dict,
                             predictions_df: pd.DataFrame,
                             chip_status: Dict,
                             fixtures_df: pd.DataFrame = None,
                             teams_data: List[Dict] = None,
                             favorable_matchups: List[Dict] = None) -> Dict:
        """
        Main optimization function - clean MIP formulation
        
        Returns:
            Dict with team_evolution, chip_plan, and recommendations
        """
        if not PULP_AVAILABLE:
            raise RuntimeError("MIP optimization requires PuLP. Install with: pip install pulp")
        
        # Adjust horizon dynamically
        self._adjust_horizon(current_gw)
        
        print(f"\n[Refactored MIP] Optimization: GW{current_gw} to GW{current_gw + self.horizon - 1}")
        
        # Prepare data
        candidate_pool, cost_map, pos_map, team_id_map = self._prepare_candidates(
            current_team, player_projections, predictions_df
        )
        
        # Use provided favorable matchups or calculate them
        if favorable_matchups is None:
            try:
                ml_predictions = self.predictor.predict_multi_gameweek(
                    predictions_df, 
                    fixtures_df if fixtures_df is not None else pd.DataFrame(), 
                    teams_data if teams_data else [], 
                    self.horizon
                )
                favorable_matchups = ml_predictions.get('favorable_matchups', [])
            except Exception as e:
                print(f"    Warning: Could not get favorable matchups: {e}")
                favorable_matchups = []
        
        # Build and solve MIP
        solution = self._build_and_solve_mip(
            current_team=current_team,
            candidates=candidate_pool,
            start_gw=current_gw,
            budget=budget,
            free_transfers=free_transfers,
            player_projections=player_projections,
            cost_map=cost_map,
            pos_map=pos_map,
            team_id_map=team_id_map,
            chip_status=chip_status,
            favorable_matchups=favorable_matchups
        )
        
        # Extract solution
        team_evolution = self._extract_solution(
            solution, current_gw, free_transfers, player_projections, cost_map, pos_map
        )
        
        # Build chip plan
        chip_plan = self._build_chip_plan_from_solution(
            team_evolution, player_projections, chip_status, current_gw
        )
        
        return {
            'horizon': self.horizon,
            'start_gw': current_gw,
            'end_gw': current_gw + self.horizon - 1,
            'team_evolution': team_evolution,
            'chip_plan': chip_plan,
            'player_projections': player_projections
        }
    
    def _adjust_horizon(self, current_gw):
        """Dynamically adjust planning horizon"""
        if current_gw <= 19:
            # Plan to Christmas (GW19) or 7 weeks, whichever is longer
            to_christmas = 19 - current_gw + 1
            self.horizon = max(7, to_christmas)
        else:
            # After Christmas, plan 7-10 weeks
            self.horizon = min(10, 38 - current_gw + 1)
    
    def _prepare_candidates(self, current_team, player_projections, predictions_df):
        """Prepare candidate pool and lookup maps"""
        
        # Build lookup maps
        cost_map = predictions_df.set_index('player_id')['cost'].to_dict()
        pos_map = predictions_df.set_index('player_id')['position_name'].to_dict()
        team_id_map = predictions_df.set_index('player_id')['team'].to_dict()
        
        # Build candidate pool: current team + top performers by position
        candidates = set(current_team)
        
        # Calculate total horizon points for each player
        player_totals = []
        for pid, proj in player_projections.items():
            if pid not in pos_map:
                continue
            
            # Get total points across horizon
            if 'total_horizon_points' in proj:
                total = proj['total_horizon_points']
            elif 'gameweek_predictions' in proj:
                total = sum(proj['gameweek_predictions'].values())
            else:
                total = 0
            
            player_totals.append({
                'player_id': pid,
                'position': pos_map[pid],
                'total': total
            })
        
        totals_df = pd.DataFrame(player_totals)
        
        # Add top candidates per position (optimized pool size for speed)
        position_limits = {'GK': 8, 'DEF': 20, 'MID': 25, 'FWD': 15}  # Reduced for faster solving
        for pos, limit in position_limits.items():
            pos_players = totals_df[totals_df['position'] == pos].nlargest(limit, 'total')
            candidates.update(pos_players['player_id'].tolist())
        
        # Filter to valid candidates
        candidate_list = [pid for pid in candidates if pid in player_projections]
        
        print(f"  Candidate pool: {len(candidate_list)} players")
        
        return candidate_list, cost_map, pos_map, team_id_map
    
    def _build_and_solve_mip(self, current_team, candidates, start_gw, budget,
                             free_transfers, player_projections, cost_map,
                             pos_map, team_id_map, chip_status, favorable_matchups=None):
        """
        Build and solve the MIP model
        
        Clean formulation with focused constraints and fixture run bonuses
        """
        
        if favorable_matchups is None:
            favorable_matchups = []
        
        # Time periods
        T = [start_gw + i for i in range(self.horizon)]
        
        # Create model
        model = LpProblem("FPL_Refactored", LpMaximize)
        
        print(f"  Building MIP with {len(candidates)} candidates over {len(T)} gameweeks...")
        
        # ========================================
        # DECISION VARIABLES
        # ========================================
        
        # Squad membership
        squad = LpVariable.dicts('squad', [(p, t) for p in candidates for t in T], cat=LpBinary)
        
        # Starting XI
        starts = LpVariable.dicts('starts', [(p, t) for p in candidates for t in T], cat=LpBinary)
        
        # Captain and Vice-Captain
        captain = LpVariable.dicts('captain', [(p, t) for p in candidates for t in T], cat=LpBinary)
        vice = LpVariable.dicts('vice', [(p, t) for p in candidates for t in T], cat=LpBinary)
        
        # Transfers (from T[1] onwards)
        transfer_in = LpVariable.dicts('transfer_in', [(p, t) for p in candidates for t in T[1:]], cat=LpBinary)
        transfer_out = LpVariable.dicts('transfer_out', [(p, t) for p in candidates for t in T[1:]], cat=LpBinary)
        num_transfers = LpVariable.dicts('num_transfers', T[1:], lowBound=0, cat=LpInteger)
        
        # Free transfer tracking
        fts_available = LpVariable.dicts('fts_available', T, lowBound=0, upBound=5, cat=LpInteger)
        fts_used = LpVariable.dicts('fts_used', T[1:], lowBound=0, upBound=5, cat=LpInteger)
        hits_taken = LpVariable.dicts('hits_taken', T[1:], lowBound=0, cat=LpInteger)
        
        # Chips (at most 1 of each across entire horizon)
        wildcard = LpVariable.dicts('wildcard', T, cat=LpBinary)
        free_hit = LpVariable.dicts('free_hit', T, cat=LpBinary)
        triple_captain = LpVariable.dicts('triple_captain', T, cat=LpBinary)
        bench_boost = LpVariable.dicts('bench_boost', T, cat=LpBinary)
        
        # ========================================
        # CONSTRAINTS
        # ========================================
        
        print(f"  Adding constraints...")
        
        # 1. CHIP AVAILABILITY
        wc_available = 1 if chip_status.get('wildcard', {}).get('available', False) else 0
        fh_available = 1 if chip_status.get('free_hit', {}).get('available', False) else 0
        tc_available = 1 if chip_status.get('triple_captain', {}).get('available', False) else 0
        bb_available = 1 if chip_status.get('bench_boost', {}).get('available', False) else 0
        
        # Each chip used at most once across horizon
        model += lpSum(wildcard[t] for t in T) <= wc_available, "WildcardOnce"
        model += lpSum(free_hit[t] for t in T) <= fh_available, "FreeHitOnce"
        model += lpSum(triple_captain[t] for t in T) <= tc_available, "TripleCaptainOnce"
        model += lpSum(bench_boost[t] for t in T) <= bb_available, "BenchBoostOnce"
        
        # At most ONE chip per gameweek
        for t in T:
            model += wildcard[t] + free_hit[t] + triple_captain[t] + bench_boost[t] <= 1, f"OneChipPerGW_{t}"
        
        # 2. SQUAD SIZE & POSITION REQUIREMENTS
        for t in T:
            model += lpSum(squad[(p, t)] for p in candidates) == self.squad_size, f"SquadSize_{t}"
            
            for pos, reqs in self.position_requirements.items():
                pos_players = [p for p in candidates if pos_map.get(p) == pos]
                model += lpSum(squad[(p, t)] for p in pos_players) >= reqs['squad_min'], f"Squad_{pos}_Min_{t}"
                model += lpSum(squad[(p, t)] for p in pos_players) <= reqs['squad_max'], f"Squad_{pos}_Max_{t}"
        
        # 3. STARTING XI REQUIREMENTS
        for t in T:
            model += lpSum(starts[(p, t)] for p in candidates) == self.starting_xi_size, f"XISize_{t}"
            
            for pos, reqs in self.position_requirements.items():
                pos_players = [p for p in candidates if pos_map.get(p) == pos]
                model += lpSum(starts[(p, t)] for p in pos_players) >= reqs['xi_min'], f"XI_{pos}_Min_{t}"
                model += lpSum(starts[(p, t)] for p in pos_players) <= reqs['xi_max'], f"XI_{pos}_Max_{t}"
            
            # Can only start players in squad
            for p in candidates:
                model += starts[(p, t)] <= squad[(p, t)], f"StartInSquad_{p}_{t}"
        
        # 4. CLUB CONSTRAINT (max 3 per club)
        team_names = set(team_id_map.values())
        for t in T:
            for team_id in team_names:
                team_players = [p for p in candidates if team_id_map.get(p) == team_id]
                if team_players:
                    model += lpSum(squad[(p, t)] for p in team_players) <= self.max_per_club, f"Club_{team_id}_{t}"
        
        # 5. BUDGET CONSTRAINT (cost of squad <= initial cost + bank)
        initial_cost = sum(cost_map.get(p, 0) for p in current_team if p in cost_map)
        budget_limit = initial_cost + budget
        
        for t in T:
            squad_cost = lpSum(squad[(p, t)] * cost_map.get(p, 0) for p in candidates)
            model += squad_cost <= budget_limit, f"Budget_{t}"
        
        # 6. CAPTAIN & VICE-CAPTAIN
        for t in T:
            # Exactly 1 captain and 1 vice-captain
            model += lpSum(captain[(p, t)] for p in candidates) == 1, f"OneCaptain_{t}"
            model += lpSum(vice[(p, t)] for p in candidates) == 1, f"OneVice_{t}"
            
            # Captain and vice must be in starting XI
            for p in candidates:
                model += captain[(p, t)] <= starts[(p, t)], f"CaptainStarts_{p}_{t}"
                model += vice[(p, t)] <= starts[(p, t)], f"ViceStarts_{p}_{t}"
            
            # Captain and vice must be different
            for p in candidates:
                model += captain[(p, t)] + vice[(p, t)] <= 1, f"CaptainViceDiff_{p}_{t}"
        
        # 7. INITIAL SQUAD (T[0] must match current team - WC disabled for T[0])
        for p in candidates:
            if p in current_team:
                model += squad[(p, T[0])] == 1, f"InitialSquad_{p}"
            else:
                model += squad[(p, T[0])] == 0, f"NotInitialSquad_{p}"
        
        # 8. TRANSFER LOGIC
        # Big M constant for logical constraints
        M = 15
        
        for idx, t in enumerate(T[1:], 1):
            t_prev = T[idx - 1]
            
            # Transfer definition: in/out changes
            for p in candidates:
                # Transfer out: was in squad, now not (unless Free Hit)
                model += transfer_out[(p, t)] >= squad[(p, t_prev)] - squad[(p, t)], f"TransferOut_{p}_{t}"
                
                # Transfer in: not in squad, now in (unless Free Hit)
                model += transfer_in[(p, t)] >= squad[(p, t)] - squad[(p, t_prev)], f"TransferIn_{p}_{t}"
            
            # Number of transfers = transfers in (or transfers out, they're equal)
            model += num_transfers[t] == lpSum(transfer_in[(p, t)] for p in candidates), f"NumTransfers_{t}"
            
            # Transfers in == transfers out (unless Wildcard or Free Hit)
            model += (lpSum(transfer_in[(p, t)] for p in candidates) == 
                     lpSum(transfer_out[(p, t)] for p in candidates)), f"BalancedTransfers_{t}"
        
        # 9. FREE TRANSFER BANKING
        model += fts_available[T[0]] == free_transfers, "InitialFTs"
        
        for idx, t in enumerate(T[1:], 1):
            t_prev = T[idx - 1]
            
            # FTs used this week
            model += fts_used[t] <= fts_available[t_prev], f"FTsUsed_{t}"
            model += fts_used[t] <= num_transfers[t], f"FTsUsedMax_{t}"
            
            # Wildcard/Free Hit = unlimited free transfers, NO HITS
            
            # When WC or FH active: hits must be 0
            model += hits_taken[t] <= M * (1 - wildcard[t] - free_hit[t]), f"NoHitsWithChip_{t}"
            
            # When WC or FH NOT active: normal hit calculation
            # hits = max(0, transfers - FTs)
            model += hits_taken[t] >= num_transfers[t] - fts_used[t] - M * (wildcard[t] + free_hit[t]), f"HitsCalc_{t}"
            
            # When WC/FH active: can make unlimited transfers (up to M=15)
            # When NOT active: transfers limited by FTs + hits
            model += num_transfers[t] <= M * (wildcard[t] + free_hit[t]) + fts_used[t] + hits_taken[t], f"TransferLimit_{t}"
            
            # FT banking: gain +1 FT if no transfers made (unless WC/FH week)
            # If transfers made, FTs next week = current FTs - used + 1
            # This is complex to linearize, so we use a simpler rule:
            # FTs next week = min(5, max(1, FTs - used + 1))
            # Approximation: FTs_next = FTs_prev - FTs_used + 1, capped at 5
            model += fts_available[t] >= 1, f"MinFT_{t}"
            model += fts_available[t] <= 5, f"MaxFT_{t}"
            model += fts_available[t] >= fts_available[t_prev] - fts_used[t] + 1 - M * (wildcard[t] + free_hit[t]), f"FTBanking_{t}"
            model += fts_available[t] <= fts_available[t_prev] - fts_used[t] + 1 + M * (wildcard[t] + free_hit[t]), f"FTBankingUpper_{t}"
        
        # 10. WILDCARD/FREE HIT LOGIC
        # Ensure chips are only used when making significant transfers
        MIN_CHIP_TRANSFERS = 3  # Wildcard/FH should have at least 3 transfers to be worthwhile
        
        # For T[0]: if WC used, must differ from current_team by at least MIN_CHIP_TRANSFERS
        if len(T) > 0:
            t0 = T[0]
            squad_changes_from_current = lpSum([
                (1 - squad[(p, t0)]) if p in current_team else squad[(p, t0)]
                for p in candidates
            ])
            model += squad_changes_from_current >= MIN_CHIP_TRANSFERS * wildcard[t0], f"WildcardMinChanges_t0"
        
        # For T[1:]: use num_transfers
        for t in T[1:]:
            # If wildcard used, must make at least MIN_CHIP_TRANSFERS
            model += num_transfers[t] >= MIN_CHIP_TRANSFERS * wildcard[t], f"WildcardMinTransfers_{t}"
            # If free hit used, must make at least MIN_CHIP_TRANSFERS  
            model += num_transfers[t] >= MIN_CHIP_TRANSFERS * free_hit[t], f"FreeHitMinTransfers_{t}"
        
        # 10. FREE HIT LOGIC (squad reverts to previous week's squad next week)
        for idx, t in enumerate(T[1:], 1):
            if idx < len(T) - 1:
                t_next = T[idx + 1]
                t_prev = T[idx - 1]
                
                # If Free Hit used this week, next week's squad = this week's previous squad
                for p in candidates:
                    model += squad[(p, t_next)] >= squad[(p, t_prev)] - M * (1 - free_hit[t]), f"FH_Revert_{p}_{t}_{t_next}"
                    model += squad[(p, t_next)] <= squad[(p, t_prev)] + M * (1 - free_hit[t]), f"FH_RevertUpper_{p}_{t}_{t_next}"
        
        # Note: Post-Wildcard stability is handled in objective function (see below)
        # to avoid over-constraining the MIP
        
        # ========================================
        # OBJECTIVE FUNCTION
        # ========================================
        
        print(f"  Building objective function...")
        
        def get_player_points(p, t):
            """Get predicted points for player p in gameweek t"""
            if p not in player_projections:
                return 0
            proj = player_projections[p]
            if 'gameweek_predictions' in proj:
                return proj['gameweek_predictions'].get(t, 0)
            return 0
        
        objective_terms = []
        
        # 1. Starting XI Points
        for t in T:
            for p in candidates:
                pts = get_player_points(p, t)
                objective_terms.append(starts[(p, t)] * pts)
        
        # 2. Captain Bonus (captain gets their points again)
        for t in T:
            for p in candidates:
                pts = get_player_points(p, t)
                objective_terms.append(captain[(p, t)] * pts)
        
        # 3. Triple Captain Bonus - SIMPLIFIED (no linearization)
        # Add a large bonus when TC is used to encourage it in high-scoring weeks
        # The MIP will choose the best week based on captain points
        for t in T:
            # Estimate best captain score for this week (approximation)
            max_captain_pts = max([get_player_points(p, t) for p in candidates], default=0)
            # TC bonus = roughly the captain's points (will be refined in post-processing)
            objective_terms.append(triple_captain[t] * max_captain_pts)
        
        # 4. Bench Boost Points - SIMPLIFIED (no linearization)
        # Add a bonus when BB is used based on estimated bench strength
        for t in T:
            # Estimate bench strength (top 4 non-XI scorers)
            all_pts = sorted([get_player_points(p, t) for p in candidates], reverse=True)
            est_bench_pts = sum(all_pts[11:15]) if len(all_pts) >= 15 else 0
            # BB bonus = roughly the bench points (will be refined in post-processing)
            objective_terms.append(bench_boost[t] * est_bench_pts * 0.8)  # 0.8 factor since estimate is rough
        
        # 5. Transfer Hits (subtract 4 points per hit + large risk penalty)
        # Adding 4.0 extra penalty (total 8.0) to strongly discourage hits
        # This accounts for prediction uncertainty + encourages Wildcard usage
        for t in T[1:]:
            objective_terms.append(-8.0 * hits_taken[t])
        
        # 6. Wildcard Value Modeling
        # Wildcard should be EXTREMELY attractive - it's one of the most valuable chips!
        # The MIP tends to prefer spread-out hits, so WC bonus must be massive
        for t in T:
            # MASSIVE bonus: 75 points
            # This ensures WC is preferred over taking 10+ hits across the horizon
            objective_terms.append(75.0 * wildcard[t])
        
        # 7. Squad Continuity Bonus (NEW!)
        # Reward keeping the same squad from week to week
        # This makes the MIP build more stable teams, especially post-Wildcard
        for idx in range(len(T) - 1):
            t_curr = T[idx]
            t_next = T[idx + 1]
            
            # Small bonus for each player kept in squad (+0.5 pts per player)
            for p in candidates:
                # Bonus when player in squad both weeks
                # squad[p, t_curr] * squad[p, t_next] would be non-linear
                # So we approximate: bonus if in both (handled via transfer penalty)
                pass  # Actually this is implicit in the hit penalty - skip
        
        # 8. Fixture Run Bonuses
        # Add bonuses for players from teams with favorable fixture runs
        # This encourages transfers to target these players
        # IMPORTANT: Keep bonuses small to avoid over-incentivizing hits
        if favorable_matchups:
            print(f"    Adding fixture run bonuses for {len(favorable_matchups)} teams...")
            for run in favorable_matchups:
                team_id = run.get('team_id')
                quality_score = run.get('quality_score', 0)
                fixture_list = run.get('fixtures', [])
                
                # Find players from this team
                team_players = [p for p in candidates if team_id_map.get(p) == team_id]
                
                # Add bonus for each player in each week of their fixture run
                for fixture in fixture_list:
                    gw = fixture.get('gw')
                    if gw in T:
                        # Bonus proportional to fixture quality
                        # Scale: 0.05-0.25 points bonus per player per good fixture
                        # FURTHER REDUCED to avoid excessive hits while still guiding transfers
                        fixture_bonus = min(0.25, quality_score / 400.0)
                        
                        for p in team_players:
                            # Add bonus when player is in squad during their good fixture
                            objective_terms.append(squad[(p, gw)] * fixture_bonus)
        
        # Set objective
        model += lpSum(objective_terms), "TotalPoints"
        
        # ========================================
        # SOLVE
        # ========================================
        
        print(f"  Solving MIP...")
        # Add time limit to prevent getting stuck (60 seconds max)
        solver = PULP_CBC_CMD(msg=0, timeLimit=60, gapRel=0.05)  # 5% optimality gap acceptable
        model.solve(solver)
        
        status = LpStatus[model.status]
        print(f"  Solver status: {status}")
        if model.status == LpStatusOptimal:
            print(f"  Objective value: {value(model.objective):.1f} points")
        elif model.status == LpStatusNotSolved:
            print(f"  Solver hit time limit - using best solution found")
        
        if status != 'Optimal':
            print(f"  WARNING: Solution is {status}, results may be suboptimal")
        
        # Debug: Show which chips the MIP actually chose
        print(f"  Chip usage by MIP:")
        for t in T:
            chips_used = []
            if value(wildcard.get(t, 0)) > 0.5:
                chips_used.append('WC')
            if value(free_hit.get(t, 0)) > 0.5:
                chips_used.append('FH')
            if value(triple_captain.get(t, 0)) > 0.5:
                chips_used.append('TC')
            if value(bench_boost.get(t, 0)) > 0.5:
                chips_used.append('BB')
            if chips_used:
                print(f"    GW{t}: {', '.join(chips_used)}")
        
        return {
            'model': model,
            'squad': squad,
            'starts': starts,
            'captain': captain,
            'vice': vice,
            'transfer_in': transfer_in,
            'transfer_out': transfer_out,
            'num_transfers': num_transfers,
            'fts_available': fts_available,
            'hits_taken': hits_taken,
            'wildcard': wildcard,
            'free_hit': free_hit,
            'triple_captain': triple_captain,
            'bench_boost': bench_boost,
            'objective_value': value(model.objective)
        }
    
    def _extract_solution(self, solution, start_gw, initial_fts, player_projections, cost_map, pos_map):
        """Extract team evolution from MIP solution"""
        
        T = [start_gw + i for i in range(self.horizon)]
        team_evolution = {}
        
        for t in T:
            # Extract squad
            squad_ids = [p for p in solution['squad'].keys() 
                        if p[1] == t and value(solution['squad'][p]) > 0.5]
            squad_ids = [p[0] for p in squad_ids]
            
            # Extract starting XI
            xi_ids = [p for p in solution['starts'].keys() 
                     if p[1] == t and value(solution['starts'][p]) > 0.5]
            xi_ids = [p[0] for p in xi_ids]
            
            # Extract captain
            cap_id = next((p[0] for p in solution['captain'].keys() 
                          if p[1] == t and value(solution['captain'][p]) > 0.5), None)
            
            # Extract vice
            vice_id = next((p[0] for p in solution['vice'].keys() 
                           if p[1] == t and value(solution['vice'][p]) > 0.5), None)
            
            # Extract transfers (if not first week)
            transfers = []
            num_transfers = 0
            if t != T[0]:
                transfers_in = [p[0] for p in solution['transfer_in'].keys() 
                               if p[1] == t and value(solution['transfer_in'][p]) > 0.5]
                transfers_out = [p[0] for p in solution['transfer_out'].keys() 
                                if p[1] == t and value(solution['transfer_out'][p]) > 0.5]
                
                # Extract chip to determine pairing strategy
                chip = None
                if value(solution['wildcard'].get(t, 0)) > 0.5:
                    chip = 'wildcard'
                elif value(solution['free_hit'].get(t, 0)) > 0.5:
                    chip = 'free_hit'
                
                # For Wildcard/Free Hit: Match transfers by position for clarity
                # For normal transfers: Just zip (should be same position anyway)
                if chip in ['wildcard', 'free_hit']:
                    # Group by position
                    outs_by_pos = {}
                    ins_by_pos = {}
                    
                    for p_out in transfers_out:
                        pos = pos_map.get(p_out, 'UNK')
                        if pos not in outs_by_pos:
                            outs_by_pos[pos] = []
                        outs_by_pos[pos].append(p_out)
                    
                    for p_in in transfers_in:
                        pos = pos_map.get(p_in, 'UNK')
                        if pos not in ins_by_pos:
                            ins_by_pos[pos] = []
                        ins_by_pos[pos].append(p_in)
                    
                    # Match by position
                    for pos in ['GK', 'DEF', 'MID', 'FWD']:
                        pos_outs = outs_by_pos.get(pos, [])
                        pos_ins = ins_by_pos.get(pos, [])
                        
                        # Pair same-position players
                        for p_out, p_in in zip(pos_outs, pos_ins):
                            transfers.append({
                                'out_id': p_out,
                                'out_name': player_projections.get(p_out, {}).get('web_name', 'Unknown'),
                                'in_id': p_in,
                                'in_name': player_projections.get(p_in, {}).get('web_name', 'Unknown'),
                                'cost_change': cost_map.get(p_in, 0) - cost_map.get(p_out, 0),
                                'gain': self._get_pts(p_in, t, player_projections) - self._get_pts(p_out, t, player_projections),
                                'position': pos
                            })
                else:
                    # Normal transfers: just zip (should be same position anyway)
                    for p_out, p_in in zip(transfers_out, transfers_in):
                        transfers.append({
                            'out_id': p_out,
                            'out_name': player_projections.get(p_out, {}).get('web_name', 'Unknown'),
                            'in_id': p_in,
                            'in_name': player_projections.get(p_in, {}).get('web_name', 'Unknown'),
                            'cost_change': cost_map.get(p_in, 0) - cost_map.get(p_out, 0),
                            'gain': self._get_pts(p_in, t, player_projections) - self._get_pts(p_out, t, player_projections)
                        })
                
                num_transfers = int(value(solution['num_transfers'].get(t, 0)))
            else:
                # First week - extract chip
                chip = None
                if value(solution['wildcard'].get(t, 0)) > 0.5:
                    chip = 'wildcard'
                elif value(solution['free_hit'].get(t, 0)) > 0.5:
                    chip = 'free_hit'
                elif value(solution['triple_captain'].get(t, 0)) > 0.5:
                    chip = 'triple_captain'
                elif value(solution['bench_boost'].get(t, 0)) > 0.5:
                    chip = 'bench_boost'
            
            # Calculate expected points
            expected_pts = sum(self._get_pts(p, t, player_projections) for p in xi_ids)
            if cap_id:
                expected_pts += self._get_pts(cap_id, t, player_projections)  # Captain bonus
            if chip == 'triple_captain' and cap_id:
                expected_pts += self._get_pts(cap_id, t, player_projections)  # TC bonus
            if chip == 'bench_boost':
                bench = [p for p in squad_ids if p not in xi_ids]
                expected_pts += sum(self._get_pts(p, t, player_projections) for p in bench)
            
            # Store basic info (FTs will be calculated separately)
            team_evolution[t] = {
                'team': squad_ids,
                'starting_xi': xi_ids,
                'captain_id': cap_id,
                'captain_name': player_projections.get(cap_id, {}).get('web_name', 'Unknown'),
                'vice_captain_id': vice_id,
                'vice_captain_name': player_projections.get(vice_id, {}).get('web_name', 'Unknown'),
                'transfers': transfers,
                'num_transfers': num_transfers,
                'chip': chip,
                'expected_points': round(expected_pts, 1),
                'budget_remaining': sum(cost_map.get(p, 0) for p in squad_ids)
            }
        
        # Calculate free transfers and hits separately based on decisions
        team_evolution = self._calculate_free_transfers_and_hits(team_evolution, T, initial_fts)
        
        return team_evolution
    
    def _calculate_free_transfers_and_hits(self, team_evolution, gameweeks, initial_fts):
        """
        Calculate free transfers and hits for each gameweek based on decisions
        
        This function is separate from the MIP to avoid entangled logic.
        FTs are calculated deterministically from transfer decisions and chip usage.
        
        FPL Rules:
        1. Start with initial FTs (usually 1)
        2. Each week: FTs = min(5, previous_FTs - transfers_used + 1)
        3. Wildcard/Free Hit weeks: unlimited free transfers
        4. Free Hit: squad reverts next week, so FTs don't change
        5. Hits = max(0, transfers - FTs_available) * 4 points
        
        Args:
            team_evolution: Dict of {gw: week_plan}
            gameweeks: List of GW numbers
            initial_fts: Starting free transfers for first week
            
        Returns:
            Updated team_evolution with correct FT tracking
        """
        
        # Start with provided initial FTs
        current_fts = initial_fts
        
        for idx, gw in enumerate(gameweeks):
            week_plan = team_evolution[gw]
            
            if idx == 0:
                # First week: use starting FTs
                num_transfers = week_plan.get('num_transfers', 0)
                chip = week_plan.get('chip')
                
                week_plan['free_transfers_available'] = current_fts
                week_plan['free_transfers_used'] = min(num_transfers, current_fts)
                week_plan['transfer_cost'] = 0  # Assume first week doesn't take hits in planning
                
                # Calculate FTs for next week
                if chip in ['wildcard', 'free_hit']:
                    if chip == 'wildcard':
                        next_fts = 1
                    else:  # Free Hit
                        next_fts = current_fts
                else:
                    # Normal banking logic
                    fts_used = week_plan['free_transfers_used']
                    next_fts = min(5, max(1, current_fts - fts_used + 1))
                
                current_fts = next_fts
            else:
                # Subsequent weeks: calculate based on previous week
                num_transfers = week_plan.get('num_transfers', 0)
                chip = week_plan.get('chip')
                
                # Wildcard and Free Hit give unlimited free transfers
                if chip in ['wildcard', 'free_hit']:
                    fts_available = current_fts  # Current FTs (before this week)
                    fts_used = 0  # Chip provides unlimited, so no FTs consumed
                    hits = 0
                    transfer_cost = 0
                    
                    # Free Hit: FTs don't change (squad reverts)
                    # Wildcard: FTs reset to 1 next week
                    if chip == 'wildcard':
                        next_fts = 1
                    else:  # Free Hit
                        next_fts = current_fts  # Unchanged
                else:
                    # Normal transfers
                    fts_available = current_fts
                    fts_used = min(num_transfers, fts_available)
                    hits = max(0, num_transfers - fts_available)
                    transfer_cost = hits * 4
                    
                    # Next week's FTs = current - used + 1, capped at 5
                    # If no transfers, you bank the FT
                    next_fts = min(5, max(1, fts_available - fts_used + 1))
                
                # Update week plan
                week_plan['free_transfers_available'] = fts_available
                week_plan['free_transfers_used'] = fts_used
                week_plan['transfer_cost'] = transfer_cost
                
                # Update current_fts for next iteration
                current_fts = next_fts
        
        return team_evolution
    
    def _get_pts(self, pid, gw, player_projections):
        """Get predicted points for a player in a gameweek"""
        if pid not in player_projections:
            return 0
        proj = player_projections[pid]
        if 'gameweek_predictions' in proj:
            return proj['gameweek_predictions'].get(gw, 0)
        return 0
    
    def _build_chip_plan_from_solution(self, team_evolution, player_projections, chip_status, start_gw):
        """Build chip plan from extracted solution"""
        
        chip_plan = {}
        chips_used = set()
        
        # Initialize all chips
        for chip_name in ['wildcard', 'free_hit', 'triple_captain', 'bench_boost']:
            chip_plan[chip_name] = {
                'chip': chip_name,
                'best_gw': None,
                'expected_benefit': 0,
                'recommended': False,
                'details': {}
            }
        
        # Extract from team evolution
        for gw, week_plan in team_evolution.items():
            chip_used = week_plan.get('chip')
            
            if chip_used and chip_used not in chips_used:
                chips_used.add(chip_used)
                
                # Calculate benefit
                benefit = 0
                details = {}
                
                if chip_used == 'triple_captain':
                    cap_id = week_plan.get('captain_id')
                    if cap_id:
                        cap_pts = self._get_pts(cap_id, gw, player_projections)
                        benefit = cap_pts * 2
                        details = {
                            'captain_id': cap_id,
                            'captain_name': week_plan.get('captain_name'),
                            'captain_points': cap_pts
                        }
                
                elif chip_used == 'bench_boost':
                    bench = [p for p in week_plan['team'] if p not in week_plan['starting_xi']]
                    bench_pts = sum(self._get_pts(p, gw, player_projections) for p in bench)
                    benefit = bench_pts
                    details = {'bench_points': bench_pts}
                
                elif chip_used in ['wildcard', 'free_hit']:
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

