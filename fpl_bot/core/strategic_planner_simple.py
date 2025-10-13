"""
Simplified Strategic 5-Gameweek Planner for FPL Bot

Practical implementation that plans next 5 gameweeks:
- Identifies fixture runs
- Plans transfers for next 2-3 GWs
- Optimizes chip timing
- Provides actionable recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


class SimplifiedStrategicPlanner:
    """Practical 5-gameweek strategic planner"""
    
    def __init__(self, data_collector, predictor, chip_manager):
        self.data_collector = data_collector
        self.predictor = predictor
        self.chip_manager = chip_manager
        
        # Constants
        self.planning_horizon = 5
        self.transfer_cost = 4
        
    def create_5gw_plan(self,
                        current_team: List[int],
                        current_gw: int,
                        budget: float,
                        free_transfers: int,
                        predictions_df: pd.DataFrame,
                        fixtures_df: pd.DataFrame,
                        teams_data: List[Dict],
                        chip_status: Dict) -> Dict:
        """Create 5-gameweek strategic plan"""
        
        print("\n" + "="*70)
        print("5-GAMEWEEK STRATEGIC PLAN")
        print("="*70)
        print(f"Planning GW{current_gw} to GW{current_gw + self.planning_horizon - 1}")
        
        # Step 1: Identify fixture runs
        fixture_runs = self._identify_fixture_runs(
            fixtures_df, teams_data, current_gw
        )
        
        # Step 2: Calculate player points for each of next 5 GWs
        player_projections = self._project_player_points_5gw(
            predictions_df, fixtures_df, teams_data, current_gw
        )
        
        # Step 3: Find best teams for each GW
        optimal_teams = self._find_optimal_teams_per_gw(
            player_projections, current_team, budget
        )
        
        # Step 4: Plan transfers for next 2-3 GWs
        transfer_plan = self._plan_transfers_rolling(
            current_team, optimal_teams, free_transfers, budget, predictions_df
        )
        
        # Step 5: Optimize chip timing
        chip_plan = self._optimize_chip_timing_5gw(
            transfer_plan, player_projections, chip_status, current_gw, current_team
        )
        
        # Step 6: Compile strategic insights
        strategic_plan = {
            'horizon': {
                'start_gw': current_gw,
                'end_gw': current_gw + self.planning_horizon - 1,
                'num_weeks': self.planning_horizon
            },
            'fixture_runs': fixture_runs[:5],  # Top 5 runs
            'player_projections': player_projections,
            'optimal_teams': optimal_teams,
            'transfer_plan': transfer_plan,
            'chip_plan': chip_plan,
            'key_insights': self._generate_key_insights(
                fixture_runs, transfer_plan, chip_plan, current_gw
            )
        }
        
        return strategic_plan
    
    def _identify_fixture_runs(self,
                               fixtures_df: pd.DataFrame,
                               teams_data: List[Dict],
                               start_gw: int) -> List[Dict]:
        """Identify 3+ consecutive favorable fixtures for teams"""
        
        if fixtures_df.empty:
            return []
        
        runs = []
        teams_dict = {t['id']: t for t in teams_data}
        end_gw = start_gw + self.planning_horizon - 1
        
        for team_id, team in teams_dict.items():
            # Get team's fixtures in order
            team_fixtures = fixtures_df[
                ((fixtures_df['team_h'] == team_id) | (fixtures_df['team_a'] == team_id)) &
                (fixtures_df['event'] >= start_gw) &
                (fixtures_df['event'] <= end_gw) &
                (~fixtures_df.get('finished', pd.Series([False]*len(fixtures_df))))
            ].sort_values('event')
            
            if len(team_fixtures) < 3:
                continue
            
            # Calculate difficulty for each fixture
            difficulties = []
            for _, fixture in team_fixtures.iterrows():
                is_home = fixture['team_h'] == team_id
                opponent_id = fixture['team_a'] if is_home else fixture['team_h']
                opponent = teams_dict.get(opponent_id, {})
                
                # Simple difficulty calculation
                difficulty = self.chip_manager._calculate_fixture_difficulty_simple(
                    is_home, opponent, team
                )
                
                difficulties.append({
                    'gw': fixture['event'],
                    'opponent': opponent.get('name', 'Unknown'),
                    'is_home': is_home,
                    'difficulty': difficulty
                })
            
            # Find runs of 3+ favorable fixtures (diff <= 2.8)
            current_run = []
            for diff_info in difficulties:
                if diff_info['difficulty'] <= 2.8:
                    current_run.append(diff_info)
                else:
                    if len(current_run) >= 3:
                        runs.append({
                            'team_id': team_id,
                            'team_name': team.get('name', 'Unknown'),
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
                    'team_name': team.get('name', 'Unknown'),
                    'start_gw': current_run[0]['gw'],
                    'end_gw': current_run[-1]['gw'],
                    'length': len(current_run),
                    'avg_difficulty': np.mean([f['difficulty'] for f in current_run]),
                    'fixtures': current_run
                })
        
        # Add team position and quality score to all runs
        for run in runs:
            team_position = teams_dict.get(run['team_id'], {}).get('position', 20)
            run['team_position'] = team_position
            
            # Score = (better position) + (easier fixtures) + (longer run)
            # Elite teams (1-6): base score 10-15
            # Good teams (7-10): base score 6-9
            # Lower teams: score < 6
            position_bonus = max(0, 15 - team_position)
            fixture_bonus = (4.0 - run['avg_difficulty']) * run['length']
            
            run['quality_score'] = position_bonus + fixture_bonus
            
            # Classify run quality
            if team_position <= 6:
                run['recommendation'] = "PREMIUM TARGET"
            elif team_position <= 10:
                run['recommendation'] = "VALUE TARGET"
            else:
                run['recommendation'] = "AVOID (low quality team)"
        
        # Sort by quality score
        runs.sort(key=lambda x: x['quality_score'], reverse=True)
        
        return runs
    
    def _project_player_points_5gw(self,
                                   predictions_df: pd.DataFrame,
                                   fixtures_df: pd.DataFrame,
                                   teams_data: List[Dict],
                                   start_gw: int) -> Dict:
        """Project each player's points for next 5 gameweeks"""
        
        teams_dict = {t['id']: t for t in teams_data}
        player_projections = {}
        
        for _, player in predictions_df.iterrows():
            player_id = player['player_id']
            team_id = player.get('team')
            base_points = player['predicted_points']
            position = player.get('position_name', 'MID')
            
            gw_points = {}
            
            for gw_offset in range(self.planning_horizon):
                gw = start_gw + gw_offset
                
                # Find fixture for this GW
                fixture = fixtures_df[
                    ((fixtures_df['team_h'] == team_id) | (fixtures_df['team_a'] == team_id)) &
                    (fixtures_df['event'] == gw)
                ]
                
                if fixture.empty:
                    # No fixture this GW (blank)
                    gw_points[gw] = 0
                    continue
                
                fixture = fixture.iloc[0]
                is_home = fixture['team_h'] == team_id
                opponent_id = fixture['team_a'] if is_home else fixture['team_h']
                opponent = teams_dict.get(opponent_id, {})
                
                # Calculate difficulty
                difficulty = self.chip_manager._calculate_fixture_difficulty_simple(
                    is_home, opponent, teams_dict.get(team_id, {})
                )
                
                # Adjust points based on fixture
                if position in ['DEF', 'GK']:
                    fixture_adj = (3.0 - difficulty) * 0.6  # Defenders benefit more
                elif position == 'MID':
                    fixture_adj = (3.0 - difficulty) * 0.4
                else:  # FWD
                    fixture_adj = (3.0 - difficulty) * 0.3
                
                adjusted_points = max(0, base_points + fixture_adj)
                gw_points[gw] = round(adjusted_points, 2)
            
            player_projections[player_id] = {
                'web_name': player['web_name'],
                'team_name': player.get('team_name', 'Unknown'),
                'position_name': position,
                'cost': player['cost'],
                'base_points': base_points,
                'gw_points': gw_points,
                'total_5gw': sum(gw_points.values())
            }
        
        return player_projections
    
    def _find_optimal_teams_per_gw(self,
                                   player_projections: Dict,
                                   current_team: List[int],
                                   budget: float) -> Dict:
        """Find theoretically optimal team for each GW (ignoring transfer constraints)"""
        
        optimal_teams = {}
        
        for gw_offset in range(self.planning_horizon):
            gw = list(player_projections.values())[0]['gw_points'].keys()
            gw_list = sorted(gw)
            if gw_offset >= len(gw_list):
                continue
            target_gw = gw_list[gw_offset]
            
            # Simple greedy selection by points for this GW
            players_sorted = sorted(
                player_projections.items(),
                key=lambda x: x[1]['gw_points'].get(target_gw, 0),
                reverse=True
            )
            
            # Top 15 affordable players (simplified)
            team = []
            total_cost = 0
            for player_id, player_data in players_sorted:
                if len(team) < 15 and total_cost + player_data['cost'] <= budget + 10:
                    team.append(player_id)
                    total_cost += player_data['cost']
            
            optimal_teams[target_gw] = team[:15]
        
        return optimal_teams
    
    def _plan_transfers_rolling(self,
                               current_team: List[int],
                               optimal_teams: Dict,
                               free_transfers: int,
                               budget: float,
                               predictions_df: pd.DataFrame) -> List[Dict]:
        """Plan transfers for next 2-3 GWs using rolling horizon"""
        
        transfer_plan = []
        team = current_team.copy()
        ft_available = free_transfers
        
        # Plan for next 2-3 GWs
        for i, (gw, optimal_team) in enumerate(list(optimal_teams.items())[:3]):
            # Find best transfers to move toward optimal team
            needed_players = [p for p in optimal_team if p not in team]
            excess_players = [p for p in team if p not in optimal_team]
            
            if not needed_players or i >= 2:  # Only plan 2 GWs ahead
                transfer_plan.append({
                    'gw': gw,
                    'action': 'HOLD',
                    'transfers': [],
                    'free_transfers_used': 0,
                    'cost': 0
                })
                ft_available = min(2, ft_available + 1)
                continue
            
            # Make 1-2 transfers
            num_transfers = min(len(needed_players), ft_available, 2)
            transfers = []
            
            for j in range(num_transfers):
                if j < len(excess_players) and j < len(needed_players):
                    transfers.append({
                        'out': excess_players[j],
                        'in': needed_players[j]
                    })
                    team.remove(excess_players[j])
                    team.append(needed_players[j])
            
            hit_cost = max(0, num_transfers - ft_available) * self.transfer_cost
            
            transfer_plan.append({
                'gw': gw,
                'action': 'TRANSFER',
                'transfers': transfers,
                'num_transfers': len(transfers),
                'free_transfers_used': min(num_transfers, ft_available),
                'cost': hit_cost
            })
            
            ft_available = min(2, ft_available + 1 - num_transfers)
        
        return transfer_plan
    
    def _optimize_chip_timing_5gw(self,
                                  transfer_plan: List[Dict],
                                  player_projections: Dict,
                                  chip_status: Dict,
                                  start_gw: int,
                                  current_team: List[int]) -> Dict:
        """Find best GW for each chip within next 5 weeks"""
        
        chip_plan = {}
        
        for chip_name, status in chip_status.items():
            if not status.get('available', False):
                continue
            
            best_gw = None
            best_benefit = 0
            best_player = None
            best_player_id = None
            
            # Check each GW in plan
            for gw_offset in range(self.planning_horizon):
                gw = start_gw + gw_offset
                
                # Use current team for simplicity (could evolve based on transfer plan)
                team = current_team
                
                # Calculate chip benefit
                if chip_name == 'triple_captain':
                    # Find best captain for this GW
                    best_captain_pts = 0
                    best_captain_name = None
                    best_captain_id = None
                    
                    for pid in team:
                        if pid in player_projections:
                            pts = player_projections[pid]['gw_points'].get(gw, 0)
                            if pts > best_captain_pts:
                                best_captain_pts = pts
                                best_captain_name = player_projections[pid]['web_name']
                                best_captain_id = pid
                    
                    benefit = best_captain_pts * 2
                    
                    # Track best captain if this is the best GW so far
                    if benefit > best_benefit:
                        best_player = best_captain_name
                        best_player_id = best_captain_id
                
                elif chip_name == 'bench_boost':
                    # Bench points (simplified: last 4 players)
                    team_sorted = sorted(
                        [pid for pid in team if pid in player_projections],
                        key=lambda x: player_projections[x]['gw_points'].get(gw, 0),
                        reverse=True
                    )
                    bench = team_sorted[11:15] if len(team_sorted) >= 15 else []
                    benefit = sum(
                        player_projections[pid]['gw_points'].get(gw, 0)
                        for pid in bench
                    )
                
                else:
                    benefit = 0
                
                if benefit > best_benefit:
                    best_benefit = benefit
                    best_gw = gw
            
            chip_plan[chip_name] = {
                'best_gw': best_gw,
                'expected_benefit': round(best_benefit, 1),
                'recommended': best_benefit > 8.0,  # Only recommend if substantial
                'player': best_player,  # For TC, the captain choice
                'player_id': best_player_id
            }
        
        return chip_plan
    
    def _estimate_team_at_gw(self, transfer_plan: List[Dict], target_gw: int) -> List[int]:
        """Estimate what the team will be at a given GW based on transfer plan"""
        # For now, use the first plan's team (simplified)
        # In full implementation, would track team evolution through transfers
        for plan in transfer_plan:
            if plan.get('gw') == target_gw and 'team' in plan:
                return plan['team']
        
        # Fallback: return empty (will be handled by caller)
        return []
    
    def _generate_key_insights(self,
                              fixture_runs: List[Dict],
                              transfer_plan: List[Dict],
                              chip_plan: Dict,
                              current_gw: int) -> List[str]:
        """Generate actionable strategic insights"""
        
        insights = []
        
        # Fixture run insights - only recommend if team is elite (top 6)
        if fixture_runs:
            best_run = fixture_runs[0]
            
            # Get team position to determine if worth targeting
            season_data = self.data_collector.get_current_season_data()
            teams_dict = {t['id']: t for t in season_data.get('teams', [])} if season_data else {}
            team_position = teams_dict.get(best_run['team_id'], {}).get('position', 20)
            
            # Only strongly recommend fixture runs for elite teams (top 6)
            if team_position <= 6:
                insights.append(
                    f"[FIXTURE RUN] {best_run['team_name']} (Pos: {team_position}) has {best_run['length']} favorable fixtures "
                    f"(GW{best_run['start_gw']}-{best_run['end_gw']}). "
                    f"Strong opportunity to target their premium players."
                )
            elif team_position <= 10 and best_run['avg_difficulty'] < 2.0:
                # Mid-table teams only if fixtures are VERY easy
                insights.append(
                    f"[FIXTURE RUN] {best_run['team_name']} (Pos: {team_position}) has {best_run['length']} very easy fixtures "
                    f"(GW{best_run['start_gw']}-{best_run['end_gw']}). "
                    f"Consider their budget-friendly attackers (not premium players)."
                )
            # Otherwise don't recommend - quality teams are better even with harder fixtures
        
        # Transfer insights
        next_transfer = next((tp for tp in transfer_plan if tp.get('action') == 'TRANSFER'), None)
        if next_transfer:
            insights.append(
                f"[TRANSFERS] Plan {next_transfer['num_transfers']} transfer(s) for GW{next_transfer['gw']}"
            )
        
        # Chip insights
        for chip_name, plan in chip_plan.items():
            if plan['recommended'] and plan['best_gw']:
                # Add player info for Triple Captain
                player_info = ""
                if chip_name == 'triple_captain' and plan.get('player'):
                    player_info = f" on {plan['player']}"
                
                insights.append(
                    f"[CHIP] Best time for {chip_name.upper()}: GW{plan['best_gw']}{player_info} "
                    f"(~{plan['expected_benefit']:.0f} pts benefit)"
                )
        
        return insights
    
    def print_strategic_plan(self, plan: Dict):
        """Print the 5-GW strategic plan in a readable format"""
        
        print("\n" + "="*70)
        print("FIXTURE RUN OPPORTUNITIES")
        print("="*70)
        
        if plan['fixture_runs']:
            print("\nFixture Runs (3+ easy games, ranked by quality):")
            for i, run in enumerate(plan['fixture_runs'][:5], 1):
                fixtures_str = ", ".join([
                    f"vs {f['opponent']} ({'H' if f['is_home'] else 'A'})"
                    for f in run['fixtures']
                ])
                
                team_pos = run.get('team_position', '?')
                recommendation = run.get('recommendation', '')
                
                print(f"{i}. {run['team_name']} [Pos: {team_pos}] (GW{run['start_gw']}-{run['end_gw']})")
                print(f"   Fixtures: {fixtures_str}")
                print(f"   Difficulty: {run['avg_difficulty']:.2f}, Quality: {run.get('quality_score', 0):.1f} - {recommendation}")
        else:
            print("No significant fixture runs identified")
        
        print("\n" + "="*70)
        print("CHIP TIMING RECOMMENDATIONS")
        print("="*70)
        
        for chip_name, chip_info in plan['chip_plan'].items():
            if chip_info['best_gw']:
                status = "RECOMMENDED" if chip_info['recommended'] else "Possible"
                
                # Add player info for Triple Captain
                player_info = ""
                if chip_name == 'triple_captain' and chip_info.get('player'):
                    player_info = f" on {chip_info['player']}"
                
                print(f"{chip_name.upper()}: GW{chip_info['best_gw']}{player_info} - "
                      f"{chip_info['expected_benefit']:.1f} pts benefit [{status}]")
        
        print("\n" + "="*70)
        print("KEY STRATEGIC INSIGHTS")
        print("="*70)
        
        for insight in plan['key_insights']:
            print(f"  {insight}")
        
        print()

