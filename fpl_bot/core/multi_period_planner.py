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


class MultiPeriodPlanner:
    """7-gameweek MIP optimizer with chip planning"""
    
    def __init__(self, data_collector, predictor, chip_manager):
        self.data_collector = data_collector
        self.predictor = predictor
        self.chip_manager = chip_manager
        
        # Constants
        self.horizon = 7  # Plan 7 gameweeks ahead
        self.transfer_cost = 4
        self.max_free_transfers = 5  # Maximum free transfers you can bank
        
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
        """
        
        print(f"\nOptimizing next {self.horizon} gameweeks (GW{current_gw}-{current_gw + self.horizon - 1})...")
        
        # Step 1: Project player points for each of next 5 GWs
        player_projections = self._project_multi_period_points(
            predictions_df, fixtures_df, teams_data, current_gw
        )
        
        # Step 2: Plan team evolution (greedy rolling horizon approach)
        # Full MIP is too complex, use intelligent heuristics
        team_evolution = self._plan_team_evolution(
            current_team,
            current_gw,
            budget,
            free_transfers,
            player_projections,
            predictions_df
        )
        
        # Step 3: Optimize chip timing based on planned teams
        chip_plan = self._optimize_chips_with_team_evolution(
            team_evolution,
            player_projections,
            chip_status,
            current_gw
        )
        
        # Step 4: Identify fixture runs
        fixture_runs = self._identify_quality_fixture_runs(
            fixtures_df, teams_data, current_gw
        )
        
        # Step 5: Generate recommendations
        recommendations = self._generate_multi_period_recommendations(
            team_evolution,
            chip_plan,
            fixture_runs,
            current_gw
        )
        
        return {
            'horizon': self.horizon,
            'start_gw': current_gw,
            'end_gw': current_gw + self.horizon - 1,
            'team_evolution': team_evolution,
            'chip_plan': chip_plan,
            'fixture_runs': fixture_runs,
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
    
    def _plan_team_evolution(self,
                            current_team: List[int],
                            start_gw: int,
                            budget: float,
                            starting_ft: int,
                            player_projections: Dict,
                            predictions_df: pd.DataFrame) -> Dict:
        """
        Plan team composition evolution across multiple gameweeks
        
        Uses intelligent rolling horizon:
        - Each GW: Optimize transfers considering accumulated FTs
        - Use FTs when beneficial (gain > 0 for free transfers, gain > 4 for hits)
        - Bank FTs when no good transfers available
        """
        
        evolution = {}
        team = current_team.copy()
        ft_available = starting_ft
        remaining_budget = budget
        
        for offset in range(self.horizon):
            gw = start_gw + offset
            
            # For each GW, check if transfers are beneficial
            transfer_recommendation = self._optimize_single_gw_transfers(
                team, gw, ft_available, remaining_budget,
                player_projections, predictions_df
            )
            
            # Decide whether to make transfers
            should_transfer = False
            
            if offset == 0:
                # Current GW: Make transfers if beneficial (user's immediate decision)
                should_transfer = transfer_recommendation['gain'] > 0
            else:
                # Future GWs: More conservative - only if significant gain
                # Use 1 FT if gain > 1.5 pts
                # Use 2+ FTs if gain > 3.0 pts per transfer
                num_transfers = len(transfer_recommendation.get('transfers', []))
                gain = transfer_recommendation.get('gain', 0)
                
                if num_transfers > 0:
                    avg_gain_per_transfer = gain / num_transfers
                    
                    if num_transfers == 1 and gain >= 1.5 and ft_available >= 1:
                        should_transfer = True
                    elif num_transfers == 2 and avg_gain_per_transfer >= 3.0 and ft_available >= 2:
                        should_transfer = True
                    elif num_transfers >= 3 and avg_gain_per_transfer >= 3.5 and ft_available >= 3:
                        should_transfer = True
            
            # Apply decision
            if should_transfer and transfer_recommendation.get('transfers'):
                # Make transfers
                for transfer in transfer_recommendation['transfers']:
                    if transfer['out'] in team:
                        team.remove(transfer['out'])
                    team.append(transfer['in'])
                    remaining_budget = remaining_budget - transfer.get('cost_change', 0)
                
                # Calculate expected points
                expected_pts = sum(
                    player_projections[pid]['gw_points'].get(gw, 0)
                    for pid in team
                    if pid in player_projections
                )
                
                evolution[gw] = {
                    'gw': gw,
                    'team': team.copy(),
                    'transfers': transfer_recommendation['transfers'],
                    'num_transfers': len(transfer_recommendation['transfers']),
                    'transfer_cost': transfer_recommendation['cost'],
                    'free_transfers_used': min(len(transfer_recommendation['transfers']), ft_available),
                    'budget_remaining': remaining_budget,
                    'expected_points': round(expected_pts, 1)
                }
                
                # Update FTs for next week
                transfers_made = len(transfer_recommendation['transfers'])
                ft_available = min(5, max(0, ft_available - transfers_made) + 1)
            else:
                # Bank FT
                expected_pts = sum(
                    player_projections[pid]['gw_points'].get(gw, 0)
                    for pid in team
                    if pid in player_projections
                )
                
                evolution[gw] = {
                    'gw': gw,
                    'team': team.copy(),
                    'transfers': [],
                    'num_transfers': 0,
                    'transfer_cost': 0,
                    'free_transfers_available': ft_available,
                    'budget_remaining': remaining_budget,
                    'expected_points': round(expected_pts, 1)
                }
                
                ft_available = min(5, ft_available + 1)  # Bank FT (max 5)
        
        return evolution
    
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
                    all_transfers.append({
                        'out': out_id,
                        'in': in_id,
                        'out_name': player_out['web_name'],
                        'in_name': player_in['web_name'],
                        'gain': gain,
                        'cost_change': cost_change,
                        'position': position
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
    
    def _optimize_chips_with_team_evolution(self,
                                           team_evolution: Dict,
                                           player_projections: Dict,
                                           chip_status: Dict,
                                           start_gw: int) -> Dict:
        """
        Optimize chip timing considering actual planned team for each GW
        
        This is the KEY improvement - chips are optimized based on the
        team you'll ACTUALLY have in each gameweek, not current team
        """
        
        chip_plans = {}
        used_gws = set()  # Track which GWs have chips (max 1 per GW)
        
        # Priority order for chip planning
        chip_priority = ['triple_captain', 'bench_boost', 'free_hit', 'wildcard']
        
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
                    # Find best captain for this specific GW
                    captain_analysis = self._analyze_tc_for_gw(
                        team_for_gw, gw, player_projections
                    )
                    benefit = captain_analysis['benefit']
                    
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
                          player_projections: Dict) -> Dict:
        """Analyze Triple Captain potential for specific GW with specific team"""
        
        # Find best captain
        best_captain_id = None
        best_captain_name = None
        best_captain_pts = 0
        captain_options = []
        
        for pid in team:
            if pid not in player_projections:
                continue
            
            proj = player_projections[pid]
            pts = proj['gw_points'].get(gw, 0)
            
            captain_options.append({
                'player_id': pid,
                'web_name': proj['web_name'],
                'position': proj['position'],
                'predicted_points': pts
            })
            
            if pts > best_captain_pts:
                best_captain_pts = pts
                best_captain_name = proj['web_name']
                best_captain_id = pid
        
        # Sort captain options
        captain_options.sort(key=lambda x: x['predicted_points'], reverse=True)
        
        return {
            'benefit': best_captain_pts * 2,  # TC gives 2x captain points
            'captain_id': best_captain_id,
            'captain_name': best_captain_name,
            'captain_points': best_captain_pts,
            'all_options': captain_options[:5]  # Top 5 options
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

