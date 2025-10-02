"""
Simple FPL prediction using current season averages
This provides more realistic predictions than the ML model
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .current_season_collector import FPLCurrentSeasonCollector


class SimpleFPLPredictor:
    """Simple predictor using current season averages"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.collector = FPLCurrentSeasonCollector(data_dir)
    
    def get_player_predictions(self, gameweek: int) -> pd.DataFrame:
        """
        Get player predictions based on current season averages
        
        Returns:
        --------
        pd.DataFrame with columns: ['id', 'predicted_points']
        """
        try:
            # Get current season data
            current_data = self.collector.collect_current_season_data()
            bootstrap = self.collector.get_bootstrap_static()
            players_df = pd.DataFrame(bootstrap['elements'])
            
            predictions = []
            
            for _, player in players_df.iterrows():
                player_id = player['id']
                player_name = f"{player['first_name']} {player['second_name']}"
                
                # Get player's gameweek data
                player_gw_data = []
                for gw_id, gw_data in current_data['gameweeks'].items():
                    if 'elements' in gw_data:
                        for element in gw_data['elements']:
                            if element['id'] == player_id:
                                stats = element['stats']
                                player_gw_data.append({
                                    'gameweek': int(gw_id),
                                    'points': stats['total_points'],
                                    'minutes': stats['minutes']
                                })
                                break
                
                if not player_gw_data:
                    # No data available, use form as fallback
                    avg_points = float(player.get('form', 0))
                else:
                    # Calculate average points per game (only games where they played)
                    played_games = [gw for gw in player_gw_data if gw['minutes'] > 0]
                    if played_games:
                        avg_points = np.mean([gw['points'] for gw in played_games])
                    else:
                        # No games played, use form
                        avg_points = float(player.get('form', 0))
                
                # Apply some basic adjustments
                # If player hasn't played much, reduce prediction
                total_minutes = sum(gw['minutes'] for gw in player_gw_data)
                if total_minutes < 90:  # Less than 1 full game
                    avg_points *= 0.5
                elif total_minutes < 180:  # Less than 2 full games
                    avg_points *= 0.7
                
                # Ensure minimum of 0 points
                avg_points = max(0, avg_points)
                
                predictions.append({
                    'id': player_id,
                    'predicted_points': avg_points,
                    'name': player_name,
                    'position': player.get('element_type', 0),
                    'team': player.get('team', 0),
                    'price': player.get('now_cost', 0) / 10,
                    'form': player.get('form', 0),
                    'total_points': player.get('total_points', 0),
                    'minutes': player.get('minutes', 0)
                })
            
            return pd.DataFrame(predictions)
            
        except Exception as e:
            print(f"Error in simple prediction: {e}")
            return pd.DataFrame()
    
    def predict_team_simple(self, gameweek: int, budget: float = 100.0) -> Optional[Dict[str, Any]]:
        """
        Predict team using simple averages instead of ML model
        
        Returns:
        --------
        Dict with team prediction results
        """
        try:
            from .team_optimizer import FPLTeamOptimizer
            
            # Get predictions
            predictions_df = self.get_player_predictions(gameweek)
            
            if predictions_df.empty:
                print("❌ No predictions available")
                return None
            
            print(f"✅ Generated {len(predictions_df)} player predictions using season averages")
            
            # Get player data for team optimization
            bootstrap = self.collector.get_bootstrap_static()
            players_df = pd.DataFrame(bootstrap['elements'])
            
            # Convert position codes to names
            position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
            players_df['position'] = players_df['element_type'].map(position_map)
            
            # Filter out players with no position
            players_df = players_df.dropna(subset=['position'])
            
            # Create predictions DataFrame with only id and predicted_points
            predictions_df = predictions_df[['id', 'predicted_points']].copy()
            
            # Use team optimizer
            optimizer = FPLTeamOptimizer(total_budget=budget, data_dir=self.data_dir)
            selected_team = optimizer.optimize_team(players_df, predictions_df, budget)
            
            if selected_team is None or selected_team.empty:
                print("❌ Team optimization failed")
                return None
            
            # Select playing XI and captain
            playing_xi, captain, vice_captain, formation = optimizer.select_playing_xi(selected_team)
            
            # Calculate total predicted points
            total_points = playing_xi['predicted_points'].sum()
            
            # Convert captain and vice-captain from Series to dict
            # The captain/vice-captain are pandas Series, we need to convert them properly
            captain_dict = {
                'id': captain['id'],
                'name': f"{captain.get('first_name', '')} {captain.get('second_name', '')}".strip(),
                'predicted_points': captain['predicted_points'],
                'position': captain['position'],
                'team': captain.get('team', 0),
                'price': captain.get('price', 0)
            }
            
            vice_captain_dict = {
                'id': vice_captain['id'],
                'name': f"{vice_captain.get('first_name', '')} {vice_captain.get('second_name', '')}".strip(),
                'predicted_points': vice_captain['predicted_points'],
                'position': vice_captain['position'],
                'team': vice_captain.get('team', 0),
                'price': vice_captain.get('price', 0)
            }
            
            # Convert playing_xi to records with proper names
            playing_xi_records = []
            for _, player in playing_xi.iterrows():
                player_dict = player.to_dict()
                player_dict['name'] = f"{player_dict.get('first_name', '')} {player_dict.get('second_name', '')}".strip()
                playing_xi_records.append(player_dict)
            
            # Convert bench to records with proper names
            bench_players = selected_team[~selected_team['id'].isin(playing_xi['id'])]
            bench_records = []
            for _, player in bench_players.iterrows():
                player_dict = player.to_dict()
                player_dict['name'] = f"{player_dict.get('first_name', '')} {player_dict.get('second_name', '')}".strip()
                bench_records.append(player_dict)
            
            # Create result
            result = {
                'gameweek': gameweek,
                'team': selected_team,
                'playing_xi': playing_xi_records,
                'bench': bench_records,
                'captain': captain_dict,
                'vice_captain': vice_captain_dict,
                'formation': formation,
                'total_cost': selected_team['price'].sum(),
                'total_predicted_points': total_points,
                'budget_remaining': budget - selected_team['price'].sum(),
                'chip_used': None,
                'prediction_method': 'season_averages'
            }
            
            return result
            
        except Exception as e:
            print(f"Error in simple team prediction: {e}")
            return None
