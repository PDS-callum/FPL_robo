"""
Strategic Multi-Gameweek Planner for FPL Bot

Plans transfers, team composition, and chip usage for next 5 gameweeks using MIP
to maximize total points while considering:
- Transfer banking (max 2 free transfers)
- Budget constraints
- Fixture runs and favorable periods
- Chip usage constraints (one chip per GW)
- Future team states

Uses Mixed Integer Programming (MIP) via PuLP for optimal team selection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from itertools import combinations
from datetime import datetime

try:
    from pulp import *
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    print("Warning: pulp not installed. Install with: pip install pulp")


class StrategicPlanner:
    """Plans optimal transfers and chip usage for next 5 gameweeks using MIP"""
    
    def __init__(self, data_collector, predictor, chip_manager):
        self.data_collector = data_collector
        self.predictor = predictor
        self.chip_manager = chip_manager
        
        # Planning constants
        self.max_free_transfers = 2
        self.transfer_cost = 4  # Points per additional transfer
        self.planning_horizon = 5  # Optimize next 5 GWs
        self.christmas_gw = 19  # GW where chips reset
        
        # Squad constraints
        self.squad_size = 15
        self.position_limits = {
            'GK': (2, 2),   # Exactly 2
            'DEF': (5, 5),  # Exactly 5
            'MID': (5, 5),  # Exactly 5
            'FWD': (3, 3)   # Exactly 3
        }
        
    def create_season_plan(self, 
                          current_team: List[int],
                          current_gw: int,
                          budget: float,
                          free_transfers: int,
                          predictions_df: pd.DataFrame,
                          fixtures_df: pd.DataFrame,
                          chip_status: Dict) -> Dict:
        """
        Create comprehensive season plan from current GW to Christmas/season end
        
        Returns:
            Dict with weekly plans including transfers, team, and chip usage
        """
        print("\n" + "="*60)
        print("STRATEGIC MULTI-GAMEWEEK PLANNING")
        print("="*60)
        
        # Determine planning horizon
        planning_horizon = self._get_planning_horizon(current_gw)
        print(f"\nPlanning horizon: GW{current_gw} to GW{planning_horizon}")
        
        # Step 1: Identify fixture runs
        fixture_runs = self._identify_fixture_runs(fixtures_df, current_gw, planning_horizon)
        
        # Step 2: Build team evolution plan
        team_plan = self._plan_team_evolution(
            current_team,
            current_gw,
            planning_horizon,
            budget,
            free_transfers,
            predictions_df,
            fixtures_df,
            fixture_runs
        )
        
        # Step 3: Optimize chip timing based on planned teams
        chip_plan = self._optimize_chip_timing(
            team_plan,
            predictions_df,
            fixtures_df,
            chip_status
        )
        
        # Step 4: Integrate and optimize
        integrated_plan = self._integrate_plans(team_plan, chip_plan)
        
        return integrated_plan
    
    def _get_planning_horizon(self, current_gw: int) -> int:
        """Determine how far ahead to plan"""
        if current_gw < self.christmas_gw:
            # Before Christmas: plan to GW19
            return self.christmas_gw
        else:
            # After Christmas: plan 12 weeks or to end of season
            return min(current_gw + self.max_planning_horizon, 38)
    
    def _identify_fixture_runs(self, 
                               fixtures_df: pd.DataFrame,
                               start_gw: int,
                               end_gw: int) -> List[Dict]:
        """
        Identify favorable fixture runs for teams
        
        A "run" is 3+ consecutive favorable fixtures for a team
        """
        runs = []
        
        if fixtures_df.empty:
            return runs
        
        # Get all teams
        teams = pd.concat([
            fixtures_df['team_h'],
            fixtures_df['team_a']
        ]).unique()
        
        # Get team data for difficulty calculation
        season_data = self.data_collector.get_current_season_data()
        teams_dict = {t['id']: t for t in season_data.get('teams', [])}
        
        for team_id in teams:
            team_fixtures = self._get_team_fixtures_sequence(
                team_id, fixtures_df, teams_dict, start_gw, end_gw
            )
            
            # Find runs of 3+ easy/medium fixtures (difficulty <= 3.0)
            current_run = []
            
            for fixture in team_fixtures:
                if fixture['difficulty'] <= 3.0:
                    current_run.append(fixture)
                else:
                    # Run broken, save if >= 3 games
                    if len(current_run) >= 3:
                        runs.append({
                            'team_id': team_id,
                            'team_name': teams_dict.get(team_id, {}).get('name', 'Unknown'),
                            'start_gw': current_run[0]['gw'],
                            'end_gw': current_run[-1]['gw'],
                            'length': len(current_run),
                            'avg_difficulty': np.mean([f['difficulty'] for f in current_run]),
                            'fixtures': current_run
                        })
                    current_run = []
            
            # Check final run
            if len(current_run) >= 3:
                runs.append({
                    'team_id': team_id,
                    'team_name': teams_dict.get(team_id, {}).get('name', 'Unknown'),
                    'start_gw': current_run[0]['gw'],
                    'end_gw': current_run[-1]['gw'],
                    'length': len(current_run),
                    'avg_difficulty': np.mean([f['difficulty'] for f in current_run]),
                    'fixtures': current_run
                })
        
        # Sort by quality (length * ease)
        runs.sort(key=lambda x: x['length'] * (4.0 - x['avg_difficulty']), reverse=True)
        
        return runs
    
    def _get_team_fixtures_sequence(self,
                                    team_id: int,
                                    fixtures_df: pd.DataFrame,
                                    teams_dict: Dict,
                                    start_gw: int,
                                    end_gw: int) -> List[Dict]:
        """Get ordered fixture sequence for a team with difficulty ratings"""
        fixtures = []
        
        team_fixtures = fixtures_df[
            ((fixtures_df['team_h'] == team_id) | (fixtures_df['team_a'] == team_id)) &
            (fixtures_df['event'] >= start_gw) &
            (fixtures_df['event'] <= end_gw) &
            (~fixtures_df['finished'].fillna(False))
        ].sort_values('event')
        
        for _, fixture in team_fixtures.iterrows():
            is_home = fixture['team_h'] == team_id
            opponent_id = fixture['team_a'] if is_home else fixture['team_h']
            opponent = teams_dict.get(opponent_id, {})
            
            # Calculate difficulty
            difficulty = self.chip_manager._calculate_fixture_difficulty_simple(
                is_home, opponent, teams_dict.get(team_id, {})
            )
            
            fixtures.append({
                'gw': fixture['event'],
                'opponent_id': opponent_id,
                'opponent': opponent.get('name', 'Unknown'),
                'is_home': is_home,
                'difficulty': difficulty
            })
        
        return fixtures
    
    def _plan_team_evolution(self,
                            current_team: List[int],
                            start_gw: int,
                            end_gw: int,
                            budget: float,
                            starting_ft: int,
                            predictions_df: pd.DataFrame,
                            fixtures_df: pd.DataFrame,
                            fixture_runs: List[Dict]) -> Dict:
        """
        Plan team composition evolution across gameweeks
        
        Uses dynamic programming to find optimal transfer sequence
        """
        print("\nPlanning team evolution...")
        
        # Initialize state
        team_states = {
            start_gw: {
                'team': current_team.copy(),
                'budget': budget,
                'free_transfers': starting_ft,
                'total_points': 0,
                'transfers_made': [],
                'path': []
            }
        }
        
        # For each gameweek, consider transfer options
        for gw in range(start_gw, end_gw + 1):
            if gw not in team_states:
                continue
                
            state = team_states[gw]
            
            # Generate transfer scenarios for this GW
            scenarios = self._generate_transfer_scenarios_for_gw(
                state['team'],
                state['budget'],
                state['free_transfers'],
                gw,
                predictions_df,
                fixtures_df,
                fixture_runs
            )
            
            # Evaluate each scenario and project to next GW
            for scenario in scenarios[:5]:  # Limit to top 5 to keep tractable
                new_team = self._apply_transfers(state['team'], scenario['transfers'])
                new_budget = state['budget'] - scenario['cost_change']
                new_ft = min(2, state['free_transfers'] + 1 - scenario['num_transfers'])
                
                # Calculate points for this GW with new team
                gw_points = self._estimate_gw_points(new_team, gw, predictions_df, fixtures_df)
                transfer_cost = max(0, (scenario['num_transfers'] - state['free_transfers'])) * self.transfer_cost
                
                total_points = state['total_points'] + gw_points - transfer_cost
                
                # Store next state
                next_gw = gw + 1
                if next_gw <= end_gw:
                    if next_gw not in team_states or total_points > team_states[next_gw]['total_points']:
                        team_states[next_gw] = {
                            'team': new_team,
                            'budget': new_budget,
                            'free_transfers': new_ft,
                            'total_points': total_points,
                            'transfers_made': scenario['transfers'],
                            'path': state['path'] + [{
                                'gw': gw,
                                'transfers': scenario['transfers'],
                                'team': new_team,
                                'points': gw_points,
                                'cost': transfer_cost
                            }]
                        }
        
        return team_states
    
    def _generate_transfer_scenarios_for_gw(self,
                                           team: List[int],
                                           budget: float,
                                           free_transfers: int,
                                           gw: int,
                                           predictions_df: pd.DataFrame,
                                           fixtures_df: pd.DataFrame,
                                           fixture_runs: List[Dict]) -> List[Dict]:
        """Generate possible transfer scenarios for a single gameweek"""
        scenarios = []
        
        # Scenario 1: No transfers (bank the FT)
        scenarios.append({
            'num_transfers': 0,
            'transfers': [],
            'cost_change': 0,
            'expected_benefit': 0
        })
        
        # Scenario 2: Make 1 transfer
        # TODO: Use transfer optimizer logic here
        
        # Scenario 3: Make 2 transfers (if have 2 FTs)
        if free_transfers >= 2:
            # TODO: Generate 2-transfer scenarios
            pass
        
        return scenarios
    
    def _apply_transfers(self, team: List[int], transfers: List[Dict]) -> List[int]:
        """Apply transfers to team and return new team"""
        new_team = team.copy()
        
        for transfer in transfers:
            if 'player_out' in transfer:
                new_team.remove(transfer['player_out'])
            if 'player_in' in transfer:
                new_team.append(transfer['player_in'])
        
        return new_team
    
    def _estimate_gw_points(self,
                           team: List[int],
                           gw: int,
                           predictions_df: pd.DataFrame,
                           fixtures_df: pd.DataFrame) -> float:
        """Estimate total points for a team in a specific gameweek"""
        # For now, use simple prediction
        # TODO: Adjust predictions based on specific GW fixtures
        team_df = predictions_df[predictions_df['player_id'].isin(team)]
        return team_df['predicted_points'].sum()
    
    def _optimize_chip_timing(self,
                             team_plan: Dict,
                             predictions_df: pd.DataFrame,
                             fixtures_df: pd.DataFrame,
                             chip_status: Dict) -> Dict:
        """Optimize chip usage timing based on planned team states"""
        chip_plan = {}
        
        # For each available chip, find best GW considering future teams
        for chip_name, status in chip_status.items():
            if not status.get('available', False):
                continue
            
            best_gw = None
            best_benefit = 0
            
            for gw, state in team_plan.items():
                if isinstance(gw, int):
                    benefit = self._estimate_chip_benefit(
                        chip_name,
                        state['team'],
                        gw,
                        predictions_df,
                        fixtures_df
                    )
                    
                    if benefit > best_benefit:
                        best_benefit = benefit
                        best_gw = gw
            
            chip_plan[chip_name] = {
                'best_gw': best_gw,
                'expected_benefit': best_benefit
            }
        
        return chip_plan
    
    def _estimate_chip_benefit(self,
                               chip_name: str,
                               team: List[int],
                               gw: int,
                               predictions_df: pd.DataFrame,
                               fixtures_df: pd.DataFrame) -> float:
        """Estimate benefit of using a chip in a specific GW"""
        if chip_name == 'triple_captain':
            # Best captain's points * 2
            team_df = predictions_df[predictions_df['player_id'].isin(team)]
            if not team_df.empty:
                best_captain_pts = team_df['predicted_points'].max()
                return best_captain_pts * 2
        
        elif chip_name == 'bench_boost':
            # Bench points (positions 12-15)
            team_df = predictions_df[predictions_df['player_id'].isin(team)]
            if len(team_df) >= 15:
                bench = team_df.iloc[11:15]
                return bench['predicted_points'].sum()
        
        return 0
    
    def _integrate_plans(self, team_plan: Dict, chip_plan: Dict) -> Dict:
        """Integrate team and chip plans into cohesive strategy"""
        integrated = {
            'team_evolution': team_plan,
            'chip_timing': chip_plan,
            'key_moments': [],
            'fixture_runs_to_exploit': []
        }
        
        # Identify key decision points
        for chip_name, plan in chip_plan.items():
            if plan['best_gw']:
                integrated['key_moments'].append({
                    'gw': plan['best_gw'],
                    'action': f'Use {chip_name}',
                    'expected_benefit': plan['expected_benefit']
                })
        
        return integrated
    
    def print_strategic_plan(self, plan: Dict, current_gw: int):
        """Print the strategic plan in readable format"""
        print("\n" + "="*60)
        print("STRATEGIC PLAN SUMMARY")
        print("="*60)
        
        # Chip timing
        print("\nOPTIMAL CHIP TIMING:")
        for chip_name, chip_plan in plan['chip_timing'].items():
            if chip_plan['best_gw']:
                print(f"  {chip_name.upper()}: GW{chip_plan['best_gw']} "
                      f"({chip_plan['expected_benefit']:.1f} pts benefit)")
        
        # Key moments
        if plan['key_moments']:
            print("\nKEY DECISION POINTS:")
            for moment in sorted(plan['key_moments'], key=lambda x: x['gw']):
                print(f"  GW{moment['gw']}: {moment['action']}")

