"""
Transfer optimization module for FPL Bot

Optimizes transfer decisions considering:
- Free transfers (1 per week)
- Point costs for additional transfers (4 points each)
- Budget constraints
- Team composition requirements
- Predicted points improvement
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from itertools import combinations
import math


class TransferOptimizer:
    """Optimizes transfer decisions for maximum points gain"""
    
    def __init__(self, predictor):
        self.predictor = predictor
        self.free_transfer_cost = 0
        self.additional_transfer_cost = 4  # 4 points per additional transfer
        
    def optimize_transfers(self, 
                          current_team: List[int], 
                          predictions_df: pd.DataFrame,
                          budget: float,
                          free_transfers: int = 1,
                          max_additional_transfers: int = 2) -> Dict:
        """Optimize transfer decisions for maximum points gain"""
        
        print(f"Optimizing transfers with {free_transfers} free transfer(s) and £{budget:.1f}m budget...")
        
        # Get current team predictions
        current_team_predictions = predictions_df[predictions_df['player_id'].isin(current_team)]
        current_points = current_team_predictions['predicted_points'].sum()
        
        # Find potential transfer targets
        transfer_targets = self._find_transfer_targets(predictions_df, current_team, budget)
        
        # Generate transfer scenarios
        scenarios = self._generate_transfer_scenarios(
            current_team_predictions,
            transfer_targets,
            free_transfers,
            max_additional_transfers
        )
        
        # Evaluate scenarios
        evaluated_scenarios = self._evaluate_scenarios(scenarios)
        
        # Get best scenario
        best_scenario = self._get_best_scenario(evaluated_scenarios)
        
        # Create optimized team
        optimized_team = self._create_optimized_team(current_team, best_scenario, predictions_df)
        
        return {
            'current_points': current_points,
            'best_scenario': best_scenario,
            'all_scenarios': evaluated_scenarios,
            'transfer_targets': transfer_targets,
            'optimized_team': optimized_team,
            'transfer_made': best_scenario.get('num_transfers', 0) > 0
        }
    
    def _find_transfer_targets(self, predictions_df: pd.DataFrame, current_team: List[int], budget: float) -> pd.DataFrame:
        """Find potential transfer targets based on predictions and budget"""
        # Exclude current team players
        available_players = predictions_df[~predictions_df['player_id'].isin(current_team)]
        
        # Get current team costs to calculate realistic budget
        current_team_df = predictions_df[predictions_df['player_id'].isin(current_team)]
        current_team_costs = current_team_df['cost'].tolist()
        
        # Allow for players up to 50% more expensive than current team average
        max_affordable_cost = max(current_team_costs) * 1.5 if current_team_costs else budget * 2
        
        # Filter by budget constraints
        affordable_players = available_players[available_players['cost'] <= max_affordable_cost]
        
        # Sort by predicted value and points
        transfer_targets = affordable_players.sort_values(['predicted_points', 'value_prediction'], ascending=False)
        
        return transfer_targets
    
    def _generate_transfer_scenarios(self, 
                                   current_team: pd.DataFrame,
                                   transfer_targets: pd.DataFrame,
                                   free_transfers: int,
                                   max_additional_transfers: int) -> List[Dict]:
        """Generate different transfer scenarios"""
        scenarios = []
        
        # Get current team players sorted by predicted points (worst first)
        current_team_sorted = current_team.sort_values('predicted_points')
        
        # Generate scenarios with different numbers of transfers
        for num_transfers in range(free_transfers, min(len(current_team), max_additional_transfers + free_transfers + 1)):
            
            # Get worst performing players to potentially transfer out
            players_out = current_team_sorted.head(num_transfers)
            
            # Get best available players to potentially transfer in
            players_in = transfer_targets.head(num_transfers)
            
            # Skip if we don't have enough targets
            if len(players_in) < num_transfers:
                continue
            
            # Calculate transfer cost
            additional_transfers = max(0, num_transfers - free_transfers)
            transfer_cost = additional_transfers * self.additional_transfer_cost
            
            scenario = {
                'num_transfers': num_transfers,
                'players_out': players_out.to_dict('records'),
                'players_in': players_in.to_dict('records'),
                'transfer_cost': transfer_cost,
                'points_gained': 0,  # Will be calculated later
                'net_points_gained': 0  # Will be calculated later
            }
            
            scenarios.append(scenario)
        
        # Also add some position-specific transfer scenarios
        scenarios.extend(self._generate_position_specific_scenarios(current_team, transfer_targets, free_transfers))
        
        return scenarios
    
    def _generate_position_specific_scenarios(self, current_team: pd.DataFrame, transfer_targets: pd.DataFrame, free_transfers: int) -> List[Dict]:
        """Generate position-specific transfer scenarios"""
        scenarios = []
        
        # Find worst player in each position
        for position in ['GK', 'DEF', 'MID', 'FWD']:
            position_players = current_team[current_team['position_name'] == position]
            if len(position_players) == 0:
                continue
                
            worst_position_player = position_players.sort_values('predicted_points').head(1)
            
            # Find best replacement for this position
            position_targets = transfer_targets[transfer_targets['position_name'] == position]
            if len(position_targets) == 0:
                continue
                
            best_position_replacement = position_targets.head(1)
            
            # Check if this is a beneficial transfer
            current_points = worst_position_player['predicted_points'].iloc[0]
            new_points = best_position_replacement['predicted_points'].iloc[0]
            
            if new_points > current_points:
                scenario = {
                    'num_transfers': 1,
                    'players_out': worst_position_player.to_dict('records'),
                    'players_in': best_position_replacement.to_dict('records'),
                    'transfer_cost': 0,  # Free transfer (assuming we have at least 1)
                    'points_gained': 0,
                    'net_points_gained': 0
                }
                scenarios.append(scenario)
        
        return scenarios
    
    def _evaluate_scenarios(self, scenarios: List[Dict]) -> List[Dict]:
        """Evaluate transfer scenarios and calculate net points gained"""
        evaluated_scenarios = []
        
        for scenario in scenarios:
            # Calculate points lost from players out
            points_lost = sum(player['predicted_points'] for player in scenario['players_out'])
            
            # Calculate points gained from players in
            points_gained = sum(player['predicted_points'] for player in scenario['players_in'])
            
            # Calculate net points gained (minus transfer cost)
            net_points_gained = points_gained - points_lost - scenario['transfer_cost']
            
            # Update scenario
            scenario['points_lost'] = points_lost
            scenario['points_gained'] = points_gained
            scenario['net_points_gained'] = net_points_gained
            
            # Calculate cost analysis
            cost_out = sum(player['cost'] for player in scenario['players_out'])
            cost_in = sum(player['cost'] for player in scenario['players_in'])
            cost_difference = cost_in - cost_out
            
            scenario['cost_out'] = cost_out
            scenario['cost_in'] = cost_in
            scenario['cost_difference'] = cost_difference
            
            evaluated_scenarios.append(scenario)
        
        # Sort by net points gained
        evaluated_scenarios.sort(key=lambda x: x['net_points_gained'], reverse=True)
        
        return evaluated_scenarios
    
    def _get_best_scenario(self, evaluated_scenarios: List[Dict]) -> Optional[Dict]:
        """Get the best transfer scenario"""
        if not evaluated_scenarios:
            # If no scenarios, return "no transfers" scenario
            return {
                'num_transfers': 0,
                'players_out': [],
                'players_in': [],
                'transfer_cost': 0,
                'points_gained': 0,
                'points_lost': 0,
                'net_points_gained': 0,
                'recommendation': 'No beneficial transfers found'
            }
        
        # Filter scenarios that are actually beneficial (positive net points)
        beneficial_scenarios = [s for s in evaluated_scenarios if s.get('net_points_gained', 0) > 0]
        
        if beneficial_scenarios:
            return beneficial_scenarios[0]
        else:
            # If no beneficial scenarios, return "no transfers" scenario
            return {
                'num_transfers': 0,
                'players_out': [],
                'players_in': [],
                'transfer_cost': 0,
                'points_gained': 0,
                'points_lost': 0,
                'net_points_gained': 0,
                'recommendation': 'No beneficial transfers found'
            }
    
    def optimize_single_transfer(self, 
                               current_team: List[int],
                               predictions_df: pd.DataFrame,
                               budget: float) -> Dict:
        """Optimize for a single free transfer"""
        return self.optimize_transfers(current_team, predictions_df, budget, free_transfers=1, max_additional_transfers=0)
    
    def optimize_multiple_transfers(self, 
                                  current_team: List[int],
                                  predictions_df: pd.DataFrame,
                                  budget: float,
                                  max_transfers: int = 3) -> Dict:
        """Optimize for multiple transfers (including paid transfers)"""
        return self.optimize_transfers(current_team, predictions_df, budget, free_transfers=1, max_additional_transfers=max_transfers-1)
    
    def analyze_transfer_value(self, player_out_id: int, player_in_id: int, predictions_df: pd.DataFrame) -> Dict:
        """Analyze the value of a specific transfer"""
        player_out = predictions_df[predictions_df['player_id'] == player_out_id]
        player_in = predictions_df[predictions_df['player_id'] == player_in_id]
        
        if player_out.empty or player_in.empty:
            return {'error': 'One or both players not found'}
        
        player_out = player_out.iloc[0]
        player_in = player_in.iloc[0]
        
        points_gained = player_in['predicted_points'] - player_out['predicted_points']
        cost_difference = player_in['cost'] - player_out['cost']
        
        return {
            'player_out': {
                'name': player_out['web_name'],
                'predicted_points': player_out['predicted_points'],
                'cost': player_out['cost']
            },
            'player_in': {
                'name': player_in['web_name'],
                'predicted_points': player_in['predicted_points'],
                'cost': player_in['cost']
            },
            'points_gained': points_gained,
            'cost_difference': cost_difference,
            'value_rating': points_gained / max(abs(cost_difference), 0.1)  # Avoid division by zero
        }
    
    def get_transfer_recommendations(self, 
                                   current_team: List[int],
                                   predictions_df: pd.DataFrame,
                                   budget: float,
                                   position: str = None) -> List[Dict]:
        """Get transfer recommendations for specific positions"""
        current_team_df = predictions_df[predictions_df['player_id'].isin(current_team)]
        
        if position:
            current_team_df = current_team_df[current_team_df['position_name'] == position]
        
        # Sort current team by predicted points (worst first)
        worst_players = current_team_df.sort_values('predicted_points').head(3)
        
        recommendations = []
        for _, player in worst_players.iterrows():
            # Find better alternatives
            alternatives = predictions_df[
                (predictions_df['position_name'] == player['position_name']) &
                (predictions_df['predicted_points'] > player['predicted_points']) &
                (predictions_df['cost'] <= budget)
            ].sort_values('value_prediction', ascending=False).head(3)
            
            if not alternatives.empty:
                recommendations.append({
                    'consider_transferring_out': {
                        'name': player['web_name'],
                        'predicted_points': player['predicted_points'],
                        'cost': player['cost']
                    },
                    'alternatives': alternatives[['web_name', 'predicted_points', 'cost', 'value_prediction']].to_dict('records')
                })
        
        return recommendations
    
    def calculate_transfer_impact(self, 
                                current_team: List[int],
                                transfer_scenario: Dict,
                                predictions_df: pd.DataFrame) -> Dict:
        """Calculate the detailed impact of a transfer scenario"""
        current_team_df = predictions_df[predictions_df['player_id'].isin(current_team)]
        current_total_points = current_team_df['predicted_points'].sum()
        current_total_cost = current_team_df['cost'].sum()
        
        # Calculate new team after transfers
        new_team = current_team.copy()
        
        # Remove players out
        players_out_ids = [p['player_id'] for p in transfer_scenario['players_out']]
        new_team = [p for p in new_team if p not in players_out_ids]
        
        # Add players in
        players_in_ids = [p['player_id'] for p in transfer_scenario['players_in']]
        new_team.extend(players_in_ids)
        
        # Calculate new team stats
        new_team_df = predictions_df[predictions_df['player_id'].isin(new_team)]
        new_total_points = new_team_df['predicted_points'].sum()
        new_total_cost = new_team_df['cost'].sum()
        
        return {
            'current_team': {
                'total_points': current_total_points,
                'total_cost': current_total_cost,
                'player_count': len(current_team)
            },
            'new_team': {
                'total_points': new_total_points,
                'total_cost': new_total_cost,
                'player_count': len(new_team)
            },
            'impact': {
                'points_change': new_total_points - current_total_points,
                'cost_change': new_total_cost - current_total_cost,
                'transfer_cost': transfer_scenario['transfer_cost'],
                'net_points_change': (new_total_points - current_total_points) - transfer_scenario['transfer_cost']
            }
        }
    
    def _create_optimized_team(self, current_team: List[int], best_scenario: Dict, predictions_df: pd.DataFrame) -> Dict:
        """Create the optimized team after making transfers"""
        if not best_scenario or best_scenario.get('num_transfers', 0) == 0:
            # No transfers made, return current team
            current_team_df = predictions_df[predictions_df['player_id'].isin(current_team)]
            return {
                'transfers_made': 0,
                'team_players': current_team_df[['player_id', 'web_name', 'team_name', 'position_name', 'cost', 'predicted_points']].to_dict('records'),
                'total_cost': current_team_df['cost'].sum(),
                'total_predicted_points': current_team_df['predicted_points'].sum(),
                'formation': self._get_team_formation(current_team_df),
                'message': 'No beneficial transfers found - keeping current team'
            }
        
        # Make the transfers
        new_team = current_team.copy()
        
        # Remove players out
        players_out_ids = [p['player_id'] for p in best_scenario['players_out']]
        new_team = [p for p in new_team if p not in players_out_ids]
        
        # Add players in
        players_in_ids = [p['player_id'] for p in best_scenario['players_in']]
        new_team.extend(players_in_ids)
        
        # Get new team details
        new_team_df = predictions_df[predictions_df['player_id'].isin(new_team)]
        
        return {
            'transfers_made': best_scenario['num_transfers'],
            'players_out': best_scenario['players_out'],
            'players_in': best_scenario['players_in'],
            'transfer_cost': best_scenario['transfer_cost'],
            'net_points_gained': best_scenario['net_points_gained'],
            'team_players': new_team_df[['player_id', 'web_name', 'team_name', 'position_name', 'cost', 'predicted_points']].to_dict('records'),
            'total_cost': new_team_df['cost'].sum(),
            'total_predicted_points': new_team_df['predicted_points'].sum(),
            'formation': self._get_team_formation(new_team_df),
            'message': f"Made {best_scenario['num_transfers']} transfer(s) for {best_scenario['net_points_gained']:.1f} net points gain"
        }
    
    def _get_team_formation(self, team_df: pd.DataFrame) -> str:
        """Get team formation from player positions"""
        if team_df.empty:
            return "Unknown"
        
        # Count players by position
        position_counts = team_df['position_name'].value_counts()
        
        gk_count = position_counts.get('GK', 0)
        def_count = position_counts.get('DEF', 0)
        mid_count = position_counts.get('MID', 0)
        fwd_count = position_counts.get('FWD', 0)
        
        return f"{def_count}-{mid_count}-{fwd_count} (GK: {gk_count})"
