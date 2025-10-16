"""
Clean Multi-Period Optimizer v3 - Scenario-Based Chip Optimization

Architecture:
1. Core MIP: Optimizes team selection and transfers (given chip configuration)
2. Scenario Generator: Creates different chip timing combinations
3. Scenario Evaluator: Runs MIP for each scenario and picks best total score

This is faster and more robust than trying to optimize chips within the MIP!
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from itertools import combinations

try:
    from pulp import *
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False


class MultiPeriodPlannerV3:
    """Scenario-based optimizer: Test different chip combinations"""
    
    def __init__(self, data_collector, predictor, chip_manager):
        self.data_collector = data_collector
        self.predictor = predictor
        self.chip_manager = chip_manager
        
        self.horizon = 7
        self.transfer_cost = 4
        self.max_free_transfers = 5
        self.squad_size = 15
        self.starting_xi_size = 11
        
        self.position_requirements = {
            'GK': {'squad_min': 2, 'squad_max': 2, 'xi_min': 1, 'xi_max': 1},
            'DEF': {'squad_min': 5, 'squad_max': 5, 'xi_min': 3, 'xi_max': 5},
            'MID': {'squad_min': 5, 'squad_max': 5, 'xi_min': 2, 'xi_max': 5},
            'FWD': {'squad_min': 3, 'squad_max': 3, 'xi_min': 1, 'xi_max': 3}
        }
        self.max_per_club = 3
    
    def optimize(self, current_team, current_gw, budget, free_transfers,
                 player_projections, predictions_df, chip_status):
        """
        Main optimization using scenario-based approach
        
        1. Generate chip timing scenarios
        2. Run MIP for each scenario
        3. Pick best total score
        """
        if not PULP_AVAILABLE:
            raise RuntimeError("Requires PuLP: pip install pulp")
        
        self._adjust_horizon(current_gw)
        
        print(f"\n[V3 Optimizer] Scenario-based optimization: GW{current_gw}-{current_gw + self.horizon - 1}")
        
        # Generate scenarios to test
        scenarios = self._generate_chip_scenarios(current_gw, chip_status)
        print(f"  Testing {len(scenarios)} chip scenarios...")
        
        # Evaluate each scenario
        best_scenario = None
        best_score = -float('inf')
        
        for idx, scenario in enumerate(scenarios):
            print(f"    Scenario {idx+1}/{len(scenarios)}: {scenario['name']}")
            
            result = self._run_mip_with_chips(
                current_team, current_gw, budget, free_transfers,
                player_projections, predictions_df, scenario
            )
            
            if result and result['total_score'] > best_score:
                best_score = result['total_score']
                best_scenario = {
                    'scenario': scenario,
                    'result': result
                }
                print(f"      New best! Score: {best_score:.1f}")
        
        if not best_scenario:
            raise RuntimeError("No valid scenarios found")
        
        print(f"  Best scenario: {best_scenario['scenario']['name']} (Score: {best_score:.1f})")
        
        # Extract results from best scenario
        team_evolution = best_scenario['result']['team_evolution']
        chip_plan = self._build_chip_plan_from_scenario(
            best_scenario['scenario'], team_evolution, player_projections
        )
        
        return {
            'horizon': self.horizon,
            'start_gw': current_gw,
            'end_gw': current_gw + self.horizon - 1,
            'team_evolution': team_evolution,
            'chip_plan': chip_plan,
            'player_projections': player_projections,
            'best_score': best_score
        }
    
    def _adjust_horizon(self, current_gw):
        """Adjust planning horizon"""
        if current_gw <= 19:
            self.horizon = max(7, 19 - current_gw + 1)
        else:
            self.horizon = min(10, 38 - current_gw + 1)
    
    def _generate_chip_scenarios(self, start_gw, chip_status):
        """
        Generate chip timing scenarios to test
        
        Returns list of scenarios, each specifying when to use which chips
        """
        scenarios = []
        
        gameweeks = [start_gw + i for i in range(1, min(self.horizon, 8))]  # Skip T[0], test next 7 weeks
        
        # Available chips
        chips_available = []
        if chip_status.get('wildcard', {}).get('available', False):
            chips_available.append('wildcard')
        if chip_status.get('free_hit', {}).get('available', False):
            chips_available.append('free_hit')
        if chip_status.get('triple_captain', {}).get('available', False):
            chips_available.append('triple_captain')
        if chip_status.get('bench_boost', {}).get('available', False):
            chips_available.append('bench_boost')
        
        # Scenario 1: No chips
        scenarios.append({
            'name': 'No chips',
            'chips': {}
        })
        
        # Scenario 2-N: Each chip individually in different weeks
        for chip in chips_available:
            for gw in gameweeks[:6]:  # Test in first 6 weeks
                scenarios.append({
                    'name': f'{chip.upper()} in GW{gw}',
                    'chips': {gw: chip}
                })
        
        # Scenario N+1: Popular combinations (limit to avoid explosion)
        # WC + TC combination
        if 'wildcard' in chips_available and 'triple_captain' in chips_available:
            for wc_gw in gameweeks[:4]:
                for tc_gw in gameweeks:
                    if tc_gw != wc_gw and tc_gw > wc_gw:  # TC after WC
                        scenarios.append({
                            'name': f'WC-GW{wc_gw} + TC-GW{tc_gw}',
                            'chips': {wc_gw: 'wildcard', tc_gw: 'triple_captain'}
                        })
                        if len(scenarios) > 50:  # Limit scenarios
                            break
                if len(scenarios) > 50:
                    break
        
        # WC + BB combination
        if 'wildcard' in chips_available and 'bench_boost' in chips_available:
            for wc_gw in gameweeks[:3]:
                for bb_gw in gameweeks:
                    if bb_gw != wc_gw and bb_gw > wc_gw + 1:  # BB a bit after WC
                        scenarios.append({
                            'name': f'WC-GW{wc_gw} + BB-GW{bb_gw}',
                            'chips': {wc_gw: 'wildcard', bb_gw: 'bench_boost'}
                        })
                        if len(scenarios) > 60:
                            break
                if len(scenarios) > 60:
                    break
        
        # Limit total scenarios for performance
        return scenarios[:60]
    
    def _run_mip_with_chips(self, current_team, start_gw, budget, free_transfers,
                            player_projections, predictions_df, scenario):
        """
        Run MIP with a specific chip configuration
        
        Args:
            scenario: Dict with 'chips' = {gw: chip_name}
        
        Returns:
            Dict with team_evolution and total_score
        """
        
        # Prepare data
        cost_map = predictions_df.set_index('player_id')['cost'].to_dict()
        pos_map = predictions_df.set_index('player_id')['position_name'].to_dict()
        team_id_map = predictions_df.set_index('player_id')['team'].to_dict()
        
        candidates = self._build_candidate_pool(current_team, player_projections, predictions_df)
        
        T = [start_gw + i for i in range(self.horizon)]
        chip_config = scenario['chips']  # {gw: chip_name}
        
        def get_pts(p, t):
            if p not in player_projections:
                return 0
            return player_projections[p].get('gameweek_predictions', {}).get(t, 0)
        
        # Build MIP
        model = LpProblem(f"FPL_{scenario['name']}", LpMaximize)
        
        # Variables
        squad = LpVariable.dicts('squad', [(p, t) for p in candidates for t in T], cat=LpBinary)
        starts = LpVariable.dicts('starts', [(p, t) for p in candidates for t in T], cat=LpBinary)
        captain = LpVariable.dicts('captain', [(p, t) for p in candidates for t in T], cat=LpBinary)
        vice = LpVariable.dicts('vice', [(p, t) for p in candidates for t in T], cat=LpBinary)
        transfer_in = LpVariable.dicts('transfer_in', [(p, t) for p in candidates for t in T[1:]], cat=LpBinary)
        transfer_out = LpVariable.dicts('transfer_out', [(p, t) for p in candidates for t in T[1:]], cat=LpBinary)
        num_transfers = LpVariable.dicts('num_transfers', T[1:], lowBound=0, cat=LpInteger)
        
        # Constraints
        # Squad size
        for t in T:
            model += lpSum(squad[(p, t)] for p in candidates) == self.squad_size
            for pos, reqs in self.position_requirements.items():
                pos_players = [p for p in candidates if pos_map.get(p) == pos]
                model += lpSum(squad[(p, t)] for p in pos_players) >= reqs['squad_min']
                model += lpSum(squad[(p, t)] for p in pos_players) <= reqs['squad_max']
        
        # XI
        for t in T:
            model += lpSum(starts[(p, t)] for p in candidates) == self.starting_xi_size
            for pos, reqs in self.position_requirements.items():
                pos_players = [p for p in candidates if pos_map.get(p) == pos]
                model += lpSum(starts[(p, t)] for p in pos_players) >= reqs['xi_min']
                model += lpSum(starts[(p, t)] for p in pos_players) <= reqs['xi_max']
            for p in candidates:
                model += starts[(p, t)] <= squad[(p, t)]
        
        # Club
        team_ids = set(team_id_map.values())
        for t in T:
            for tid in team_ids:
                team_players = [p for p in candidates if team_id_map.get(p) == tid]
                if team_players:
                    model += lpSum(squad[(p, t)] for p in team_players) <= self.max_per_club
        
        # Budget
        initial_cost = sum(cost_map.get(p, 0) for p in current_team if p in cost_map)
        budget_limit = initial_cost + budget
        for t in T:
            model += lpSum(squad[(p, t)] * cost_map.get(p, 0) for p in candidates) <= budget_limit
        
        # Captain/Vice
        for t in T:
            model += lpSum(captain[(p, t)] for p in candidates) == 1
            model += lpSum(vice[(p, t)] for p in candidates) == 1
            for p in candidates:
                model += captain[(p, t)] <= starts[(p, t)]
                model += vice[(p, t)] <= starts[(p, t)]
                model += captain[(p, t)] + vice[(p, t)] <= 1
        
        # Initial squad
        for p in candidates:
            if p in current_team:
                model += squad[(p, T[0])] == 1
            else:
                model += squad[(p, T[0])] == 0
        
        # Transfers and Squad Continuity
        for idx, t in enumerate(T[1:], 1):
            t_prev = T[idx - 1]
            chip_this_week = chip_config.get(t)
            
            if chip_this_week == 'wildcard':
                # WILDCARD: Squad can be ANYTHING (unlimited transfers)
                # No constraints from previous week - optimizer picks best 15 players
                # Just count the changes for reporting
                for p in candidates:
                    model += transfer_out[(p, t)] >= squad[(p, t_prev)] - squad[(p, t)]
                    model += transfer_in[(p, t)] >= squad[(p, t)] - squad[(p, t_prev)]
                model += num_transfers[t] == lpSum(transfer_in[(p, t)] for p in candidates)
            
            elif chip_this_week == 'free_hit':
                # FREE HIT: Squad can be ANYTHING this week, reverts next week
                # No constraints from previous week
                # Count changes
                for p in candidates:
                    model += transfer_out[(p, t)] >= squad[(p, t_prev)] - squad[(p, t)]
                    model += transfer_in[(p, t)] >= squad[(p, t)] - squad[(p, t_prev)]
                model += num_transfers[t] == lpSum(transfer_in[(p, t)] for p in candidates)
                
                # REVERT: Next week's squad = previous week's squad (before FH)
                if idx < len(T) - 1:
                    t_next = T[idx + 1]
                    for p in candidates:
                        model += squad[(p, t_next)] == squad[(p, t_prev)]
            
            else:
                # NORMAL WEEK: Squad changes via standard transfers
                # Squad continuity: changes must be balanced (transfers in = transfers out)
                for p in candidates:
                    model += transfer_out[(p, t)] >= squad[(p, t_prev)] - squad[(p, t)]
                    model += transfer_in[(p, t)] >= squad[(p, t)] - squad[(p, t_prev)]
                
                model += num_transfers[t] == lpSum(transfer_in[(p, t)] for p in candidates)
                model += lpSum(transfer_in[(p, t)] for p in candidates) == lpSum(transfer_out[(p, t)] for p in candidates)
        
        # Objective: Total points across horizon with chip bonuses
        objective = []
        
        # Track FTs for hit calculation
        current_fts = free_transfers
        
        for idx, t in enumerate(T):
            # XI points
            for p in candidates:
                pts = get_pts(p, t)
                objective.append(starts[(p, t)] * pts)
                objective.append(captain[(p, t)] * pts)  # Captain bonus
            
            # Chip bonuses (pre-calculated, not optimized in MIP)
            chip_this_week = chip_config.get(t)
            
            if chip_this_week == 'triple_captain':
                # Add extra captain bonus (approximation)
                max_cap_pts = max([get_pts(p, t) for p in candidates], default=0)
                objective.append(max_cap_pts)  # TC doubles captain
            
            elif chip_this_week == 'bench_boost':
                # Add bench points (approximation)
                all_pts = sorted([get_pts(p, t) for p in candidates], reverse=True)
                bench_est = sum(all_pts[11:15]) if len(all_pts) >= 15 else 0
                objective.append(bench_est * 0.7)  # Conservative estimate
            
            # Transfer costs (calculated based on chip config)
            if idx > 0:
                if chip_this_week in ['wildcard', 'free_hit']:
                    # No hits with these chips
                    pass
                else:
                    # Penalty for transfers beyond FTs
                    # Approximate: 4 pts per transfer assuming limited FTs
                    # This encourages chip use when many transfers needed
                    objective.append(-4 * num_transfers[t])
        
        model += lpSum(objective)
        
        # Solve with tight time limit
        try:
            solver = PULP_CBC_CMD(msg=0, timeLimit=20, gapRel=0.1)
            model.solve(solver)
            
            if model.status not in [LpStatusOptimal, LpStatusNotSolved]:
                print(f"      FAILED: Status = {LpStatus[model.status]}")
                return None
        except Exception as e:
            print(f"      EXCEPTION: {e}")
            return None
        
        # Extract solution
        team_evolution = self._extract_solution(
            model, squad, starts, captain, vice, transfer_in, transfer_out,
            num_transfers, T, candidates, player_projections, chip_config
        )
        
        # Calculate actual total score with proper FT tracking
        total_score = self._calculate_scenario_score(
            team_evolution, start_gw, free_transfers, player_projections
        )
        
        return {
            'team_evolution': team_evolution,
            'total_score': total_score
        }
    
    def _generate_chip_scenarios(self, start_gw, chip_status):
        """Generate chip timing scenarios to test"""
        
        scenarios = []
        gameweeks = [start_gw + i for i in range(1, min(self.horizon, 8))]
        
        # Check available chips
        has_wc = chip_status.get('wildcard', {}).get('available', False)
        has_fh = chip_status.get('free_hit', {}).get('available', False)
        has_tc = chip_status.get('triple_captain', {}).get('available', False)
        has_bb = chip_status.get('bench_boost', {}).get('available', False)
        
        # Base scenario: No chips
        scenarios.append({'name': 'No chips', 'chips': {}})
        
        # Single chip scenarios
        if has_wc:
            for gw in gameweeks[:5]:
                scenarios.append({'name': f'WC-GW{gw}', 'chips': {gw: 'wildcard'}})
        
        if has_fh:
            for gw in gameweeks[:5]:
                scenarios.append({'name': f'FH-GW{gw}', 'chips': {gw: 'free_hit'}})
        
        if has_tc:
            for gw in gameweeks[:6]:
                scenarios.append({'name': f'TC-GW{gw}', 'chips': {gw: 'triple_captain'}})
        
        if has_bb:
            for gw in gameweeks[:6]:
                scenarios.append({'name': f'BB-GW{gw}', 'chips': {gw: 'bench_boost'}})
        
        # Skip two-chip combinations for now (too many scenarios)
        # Will add back once single-chip scenarios are working
        
        # Limit scenarios for speed
        # Start with fewer scenarios while debugging
        return scenarios[:15]  # Test max 15 scenarios for now
    
    def _build_candidate_pool(self, current_team, player_projections, predictions_df):
        """Build candidate pool"""
        pos_map = predictions_df.set_index('player_id')['position_name'].to_dict()
        
        candidates = set(current_team)
        
        player_totals = []
        for pid, proj in player_projections.items():
            if pid not in pos_map:
                continue
            total = sum(proj.get('gameweek_predictions', {}).values())
            player_totals.append({
                'player_id': pid,
                'position': pos_map[pid],
                'total': total
            })
        
        df = pd.DataFrame(player_totals)
        limits = {'GK': 6, 'DEF': 15, 'MID': 20, 'FWD': 12}
        for pos, limit in limits.items():
            top = df[df['position'] == pos].nlargest(limit, 'total')
            candidates.update(top['player_id'].tolist())
        
        return [p for p in candidates if p in player_projections]
    
    def _extract_solution(self, model, squad, starts, captain, vice, transfer_in,
                          transfer_out, num_transfers, T, candidates,
                          player_projections, chip_config):
        """Extract team evolution from MIP solution"""
        
        team_evolution = {}
        
        for t in T:
            squad_ids = [p[0] for p in squad.keys() if p[1] == t and value(squad[p]) > 0.5]
            xi_ids = [p[0] for p in starts.keys() if p[1] == t and value(starts[p]) > 0.5]
            cap_id = next((p[0] for p in captain.keys() if p[1] == t and value(captain[p]) > 0.5), None)
            vice_id = next((p[0] for p in vice.keys() if p[1] == t and value(vice[p]) > 0.5), None)
            
            transfers = []
            if t != T[0]:
                ins = [p[0] for p in transfer_in.keys() if p[1] == t and value(transfer_in[p]) > 0.5]
                outs = [p[0] for p in transfer_out.keys() if p[1] == t and value(transfer_out[p]) > 0.5]
                for p_out, p_in in zip(outs, ins):
                    transfers.append({
                        'out_id': p_out,
                        'out_name': player_projections.get(p_out, {}).get('web_name', 'Unknown'),
                        'in_id': p_in,
                        'in_name': player_projections.get(p_in, {}).get('web_name', 'Unknown'),
                        'gain': self._get_pts(p_in, t, player_projections) - self._get_pts(p_out, t, player_projections)
                    })
            
            expected_pts = sum(self._get_pts(p, t, player_projections) for p in xi_ids)
            if cap_id:
                expected_pts += self._get_pts(cap_id, t, player_projections)
            
            team_evolution[t] = {
                'team': squad_ids,
                'starting_xi': xi_ids,
                'captain_id': cap_id,
                'captain_name': player_projections.get(cap_id, {}).get('web_name', 'Unknown'),
                'vice_captain_id': vice_id,
                'vice_captain_name': player_projections.get(vice_id, {}).get('web_name', 'Unknown'),
                'transfers': transfers,
                'num_transfers': len(transfers),
                'expected_points': round(expected_pts, 1),
                'chip': chip_config.get(t)  # Apply chip from scenario
            }
        
        return team_evolution
    
    def _calculate_scenario_score(self, team_evolution, start_gw, initial_fts, player_projections):
        """Calculate total score for scenario with proper FT/hit tracking"""
        
        gameweeks = [start_gw + i for i in range(self.horizon)]
        team_evolution = self._calculate_free_transfers(team_evolution, gameweeks, initial_fts)
        
        total_score = 0
        
        for gw, week in team_evolution.items():
            # Base points
            total_score += week['expected_points']
            
            # Chip bonuses
            chip = week.get('chip')
            if chip == 'triple_captain':
                cap_id = week.get('captain_id')
                if cap_id:
                    total_score += self._get_pts(cap_id, gw, player_projections)
            elif chip == 'bench_boost':
                bench = [p for p in week['team'] if p not in week['starting_xi']]
                total_score += sum(self._get_pts(p, gw, player_projections) for p in bench)
            
            # Transfer costs
            total_score -= week.get('transfer_cost', 0)
        
        return total_score
    
    def _calculate_free_transfers(self, team_evolution, gameweeks, initial_fts):
        """Calculate FTs deterministically"""
        
        current_fts = initial_fts
        
        for idx, gw in enumerate(gameweeks):
            week = team_evolution[gw]
            num_trans = week.get('num_transfers', 0)
            chip = week.get('chip')
            
            week['free_transfers_available'] = current_fts
            
            if chip in ['wildcard', 'free_hit']:
                week['free_transfers_used'] = 0
                week['transfer_cost'] = 0
                next_fts = 1 if chip == 'wildcard' else current_fts
            else:
                week['free_transfers_used'] = min(num_trans, current_fts)
                hits = max(0, num_trans - current_fts)
                week['transfer_cost'] = hits * 4
                next_fts = min(5, max(1, current_fts - week['free_transfers_used'] + 1))
            
            current_fts = next_fts
        
        return team_evolution
    
    def _get_pts(self, pid, gw, player_projections):
        """Get predicted points"""
        if pid not in player_projections:
            return 0
        return player_projections[pid].get('gameweek_predictions', {}).get(gw, 0)
    
    def _build_chip_plan_from_scenario(self, scenario, team_evolution, player_projections):
        """Build chip plan from best scenario"""
        
        chip_plan = {}
        
        for chip_name in ['wildcard', 'free_hit', 'triple_captain', 'bench_boost']:
            chip_plan[chip_name] = {
                'chip': chip_name,
                'best_gw': None,
                'expected_benefit': 0,
                'recommended': False,
                'details': {}
            }
        
        for gw, chip_name in scenario['chips'].items():
            week = team_evolution.get(gw)
            if not week:
                continue
            
            benefit = 0
            details = {}
            
            if chip_name == 'triple_captain':
                cap_id = week.get('captain_id')
                if cap_id:
                    cap_pts = self._get_pts(cap_id, gw, player_projections)
                    benefit = cap_pts * 2
                    details = {
                        'captain_id': cap_id,
                        'captain_name': week.get('captain_name'),
                        'captain_points': cap_pts
                    }
            
            elif chip_name == 'bench_boost':
                bench = [p for p in week['team'] if p not in week['starting_xi']]
                bench_pts = sum(self._get_pts(p, gw, player_projections) for p in bench)
                benefit = bench_pts
                details = {'bench_points': bench_pts}
            
            elif chip_name == 'wildcard':
                # Benefit = hits saved
                benefit = max(0, week.get('num_transfers', 0) - 1) * 4
                details = {'hits_saved': benefit}
            
            elif chip_name == 'free_hit':
                benefit = week.get('num_transfers', 0) * 2  # Rough estimate
                details = {'num_swaps': week.get('num_transfers', 0)}
            
            chip_plan[chip_name] = {
                'chip': chip_name,
                'best_gw': gw,
                'expected_benefit': round(benefit, 1),
                'recommended': True,
                'details': details
            }
        
        return chip_plan
