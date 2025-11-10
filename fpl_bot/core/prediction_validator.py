"""
Prediction Validation Framework

Validates player points predictions against actual results for backtesting
and accuracy measurement.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class PredictionValidator:
    """
    Validates predictions against actual results
    
    Calculates metrics:
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - Correlation coefficient
    - Position-specific accuracy
    """
    
    def __init__(self):
        self.validation_results = []
    
    def validate_predictions(
        self,
        predictions_df: pd.DataFrame,
        actual_results_df: pd.DataFrame,
        players_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Validate predictions against actual results
        
        Args:
            predictions_df: DataFrame with columns: player_id, gameweek, predicted_points
            actual_results_df: DataFrame with columns: player_id, gameweek, actual_points
            players_df: Optional DataFrame with player info (for position-specific analysis)
            
        Returns:
            Dict with validation metrics
        """
        # Merge predictions with actual results
        merged = predictions_df.merge(
            actual_results_df,
            on=['player_id', 'gameweek'],
            how='inner',
            suffixes=('_predicted', '_actual')
        )
        
        if len(merged) == 0:
            return {
                'error': 'No matching predictions and actual results found',
                'num_predictions': len(predictions_df),
                'num_actual': len(actual_results_df)
            }
        
        predicted = merged['predicted_points'].values
        actual = merged['actual_points'].values
        
        # Calculate metrics
        mae = np.mean(np.abs(predicted - actual))
        rmse = np.sqrt(np.mean((predicted - actual) ** 2))
        correlation = np.corrcoef(predicted, actual)[0, 1] if len(predicted) > 1 else 0.0
        
        # Calculate percentage within ranges
        within_1 = np.mean(np.abs(predicted - actual) <= 1.0) * 100
        within_2 = np.mean(np.abs(predicted - actual) <= 2.0) * 100
        within_3 = np.mean(np.abs(predicted - actual) <= 3.0) * 100
        
        results = {
            'num_predictions': len(merged),
            'mae': float(mae),
            'rmse': float(rmse),
            'correlation': float(correlation),
            'within_1_point': float(within_1),
            'within_2_points': float(within_2),
            'within_3_points': float(within_3),
            'mean_predicted': float(np.mean(predicted)),
            'mean_actual': float(np.mean(actual)),
            'std_predicted': float(np.std(predicted)),
            'std_actual': float(np.std(actual))
        }
        
        # Position-specific analysis if players_df provided
        if players_df is not None and 'position_name' in players_df.columns:
            results['by_position'] = self._analyze_by_position(merged, players_df)
        
        # Gameweek-specific analysis
        if 'gameweek' in merged.columns:
            results['by_gameweek'] = self._analyze_by_gameweek(merged)
        
        self.validation_results.append({
            'timestamp': datetime.now(),
            'results': results
        })
        
        return results
    
    def _analyze_by_position(
        self,
        merged_df: pd.DataFrame,
        players_df: pd.DataFrame
    ) -> Dict:
        """Analyze prediction accuracy by position"""
        # Merge with player positions
        merged_with_pos = merged_df.merge(
            players_df[['id', 'position_name']],
            left_on='player_id',
            right_on='id',
            how='left'
        )
        
        by_position = {}
        for position in ['GK', 'DEF', 'MID', 'FWD']:
            pos_data = merged_with_pos[merged_with_pos['position_name'] == position]
            if len(pos_data) == 0:
                continue
            
            predicted = pos_data['predicted_points'].values
            actual = pos_data['actual_points'].values
            
            by_position[position] = {
                'count': len(pos_data),
                'mae': float(np.mean(np.abs(predicted - actual))),
                'rmse': float(np.sqrt(np.mean((predicted - actual) ** 2))),
                'correlation': float(np.corrcoef(predicted, actual)[0, 1]) if len(predicted) > 1 else 0.0,
                'mean_predicted': float(np.mean(predicted)),
                'mean_actual': float(np.mean(actual))
            }
        
        return by_position
    
    def _analyze_by_gameweek(self, merged_df: pd.DataFrame) -> Dict:
        """Analyze prediction accuracy by gameweek"""
        by_gameweek = {}
        
        for gw in merged_df['gameweek'].unique():
            gw_data = merged_df[merged_df['gameweek'] == gw]
            
            predicted = gw_data['predicted_points'].values
            actual = gw_data['actual_points'].values
            
            by_gameweek[int(gw)] = {
                'count': len(gw_data),
                'mae': float(np.mean(np.abs(predicted - actual))),
                'rmse': float(np.sqrt(np.mean((predicted - actual) ** 2))),
                'correlation': float(np.corrcoef(predicted, actual)[0, 1]) if len(predicted) > 1 else 0.0
            }
        
        return by_gameweek
    
    def compare_models(
        self,
        predictions_old: pd.DataFrame,
        predictions_new: pd.DataFrame,
        actual_results: pd.DataFrame,
        players_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Compare two prediction models
        
        Args:
            predictions_old: Old model predictions
            predictions_new: New model predictions
            actual_results: Actual results
            players_df: Optional player info
            
        Returns:
            Dict with comparison metrics
        """
        results_old = self.validate_predictions(predictions_old, actual_results, players_df)
        results_new = self.validate_predictions(predictions_new, actual_results, players_df)
        
        comparison = {
            'old_model': results_old,
            'new_model': results_new,
            'improvement': {
                'mae': results_old['mae'] - results_new['mae'],
                'rmse': results_old['rmse'] - results_new['rmse'],
                'correlation': results_new['correlation'] - results_old['correlation'],
                'within_1_point': results_new['within_1_point'] - results_old['within_1_point'],
                'within_2_points': results_new['within_2_points'] - results_old['within_2_points'],
                'within_3_points': results_new['within_3_points'] - results_old['within_3_points']
            }
        }
        
        return comparison
    
    def get_validation_history(self) -> List[Dict]:
        """Get history of all validation runs"""
        return self.validation_results

