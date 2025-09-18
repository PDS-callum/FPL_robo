"""
Performance tracking and analytics for FPL Bot.
Tracks prediction accuracy, transfer success, and team performance over time.
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path


class FPLPerformanceTracker:
    """
    Tracks and analyzes FPL Bot performance metrics.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.performance_file = os.path.join(data_dir, "performance_metrics.json")
        self.metrics = self._load_metrics()
    
    def _load_metrics(self) -> Dict[str, Any]:
        """Load existing performance metrics from file."""
        try:
            if os.path.exists(self.performance_file):
                with open(self.performance_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️  Failed to load performance metrics: {e}")
        
        return {
            'predictions': [],
            'transfers': [],
            'team_performance': [],
            'accuracy_metrics': {},
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
    
    def _save_metrics(self):
        """Save performance metrics to file."""
        try:
            self.metrics['last_updated'] = datetime.now().isoformat()
            os.makedirs(os.path.dirname(self.performance_file), exist_ok=True)
            with open(self.performance_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            print(f"⚠️  Failed to save performance metrics: {e}")
    
    def record_prediction(self, gameweek: int, predicted_points: float, 
                         team_cost: float, captain: str, formation: str,
                         target: str = 'points_scored') -> None:
        """Record a prediction for tracking."""
        prediction_record = {
            'gameweek': gameweek,
            'predicted_points': predicted_points,
            'team_cost': team_cost,
            'captain': captain,
            'formation': formation,
            'target': target,
            'timestamp': datetime.now().isoformat(),
            'actual_points': None,  # Will be updated when actual results are available
            'accuracy': None
        }
        
        # Check if prediction already exists for this gameweek
        existing_idx = None
        for i, pred in enumerate(self.metrics['predictions']):
            if pred['gameweek'] == gameweek and pred['target'] == target:
                existing_idx = i
                break
        
        if existing_idx is not None:
            self.metrics['predictions'][existing_idx] = prediction_record
        else:
            self.metrics['predictions'].append(prediction_record)
        
        self._save_metrics()
    
    def record_transfer(self, gameweek: int, transfers: List[Dict[str, Any]], 
                       transfer_cost: int, points_gain: float) -> None:
        """Record transfer decisions for tracking."""
        transfer_record = {
            'gameweek': gameweek,
            'transfers': transfers,
            'transfer_cost': transfer_cost,
            'points_gain': points_gain,
            'timestamp': datetime.now().isoformat(),
            'success_rating': None  # Will be calculated based on actual performance
        }
        
        self.metrics['transfers'].append(transfer_record)
        self._save_metrics()
    
    def update_actual_results(self, gameweek: int, actual_points: float, 
                             player_scores: Dict[str, float]) -> None:
        """Update predictions with actual results when available."""
        # Update prediction accuracy
        for pred in self.metrics['predictions']:
            if pred['gameweek'] == gameweek and pred['actual_points'] is None:
                pred['actual_points'] = actual_points
                pred['accuracy'] = self._calculate_prediction_accuracy(
                    pred['predicted_points'], actual_points
                )
                break
        
        # Update transfer success ratings
        for transfer in self.metrics['transfers']:
            if transfer['gameweek'] == gameweek and transfer['success_rating'] is None:
                transfer['success_rating'] = self._calculate_transfer_success(
                    transfer, player_scores
                )
                break
        
        self._save_metrics()
    
    def _calculate_prediction_accuracy(self, predicted: float, actual: float) -> float:
        """Calculate prediction accuracy as percentage."""
        if actual == 0:
            return 100.0 if predicted == 0 else 0.0
        return max(0, 100 - abs(predicted - actual) / actual * 100)
    
    def _calculate_transfer_success(self, transfer_record: Dict[str, Any], 
                                   player_scores: Dict[str, float]) -> float:
        """Calculate transfer success rating based on actual player performance."""
        if not transfer_record['transfers']:
            return 0.0
        
        total_actual_gain = 0
        total_predicted_gain = 0
        
        for transfer in transfer_record['transfers']:
            out_player = transfer.get('transfer_out', {}).get('name', '')
            in_player = transfer.get('transfer_in', {}).get('name', '')
            
            out_actual = player_scores.get(out_player, 0)
            in_actual = player_scores.get(in_player, 0)
            
            actual_gain = in_actual - out_actual
            predicted_gain = transfer.get('points_gain', 0)
            
            total_actual_gain += actual_gain
            total_predicted_gain += predicted_gain
        
        if total_predicted_gain == 0:
            return 50.0  # Neutral rating if no predicted gain
        
        # Success rating: how close actual gain was to predicted gain
        accuracy = max(0, 100 - abs(total_actual_gain - total_predicted_gain) / abs(total_predicted_gain) * 100)
        return accuracy
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        predictions = [p for p in self.metrics['predictions'] if p['actual_points'] is not None]
        transfers = [t for t in self.metrics['transfers'] if t['success_rating'] is not None]
        
        if not predictions:
            return {'message': 'No completed predictions available for analysis'}
        
        # Calculate accuracy metrics
        accuracies = [p['accuracy'] for p in predictions if p['accuracy'] is not None]
        avg_accuracy = np.mean(accuracies) if accuracies else 0
        
        # Calculate prediction bias
        predicted_points = [p['predicted_points'] for p in predictions]
        actual_points = [p['actual_points'] for p in predictions]
        bias = np.mean(np.array(predicted_points) - np.array(actual_points))
        
        # Transfer success metrics
        transfer_ratings = [t['success_rating'] for t in transfers if t['success_rating'] is not None]
        avg_transfer_success = np.mean(transfer_ratings) if transfer_ratings else 0
        
        # Recent performance (last 5 gameweeks)
        recent_predictions = sorted(predictions, key=lambda x: x['gameweek'])[-5:]
        recent_accuracy = np.mean([p['accuracy'] for p in recent_predictions if p['accuracy'] is not None])
        
        return {
            'total_predictions': len(predictions),
            'total_transfers': len(transfers),
            'average_accuracy': round(avg_accuracy, 1),
            'recent_accuracy': round(recent_accuracy, 1) if recent_accuracy else 0,
            'prediction_bias': round(bias, 1),
            'average_transfer_success': round(avg_transfer_success, 1),
            'best_prediction': max(predictions, key=lambda x: x['accuracy']) if predictions else None,
            'worst_prediction': min(predictions, key=lambda x: x['accuracy']) if predictions else None,
            'most_successful_transfer': max(transfers, key=lambda x: x['success_rating']) if transfers else None,
            'gameweeks_analyzed': sorted(set(p['gameweek'] for p in predictions))
        }
    
    def get_weekly_performance(self, gameweek: int) -> Dict[str, Any]:
        """Get performance metrics for a specific gameweek."""
        prediction = next((p for p in self.metrics['predictions'] if p['gameweek'] == gameweek), None)
        transfers = [t for t in self.metrics['transfers'] if t['gameweek'] == gameweek]
        
        if not prediction:
            return {'message': f'No prediction data found for gameweek {gameweek}'}
        
        result = {
            'gameweek': gameweek,
            'prediction': prediction,
            'transfers': transfers,
            'has_actual_results': prediction['actual_points'] is not None
        }
        
        if prediction['actual_points'] is not None:
            result['accuracy'] = prediction['accuracy']
            result['points_difference'] = prediction['actual_points'] - prediction['predicted_points']
            result['performance_rating'] = self._get_performance_rating(prediction['accuracy'])
        
        return result
    
    def _get_performance_rating(self, accuracy: float) -> str:
        """Get performance rating based on accuracy."""
        if accuracy >= 90:
            return "Excellent"
        elif accuracy >= 80:
            return "Good"
        elif accuracy >= 70:
            return "Average"
        elif accuracy >= 60:
            return "Below Average"
        else:
            return "Poor"
    
    def generate_performance_trends(self) -> Dict[str, Any]:
        """Generate performance trends over time."""
        predictions = [p for p in self.metrics['predictions'] if p['actual_points'] is not None]
        
        if len(predictions) < 3:
            return {'message': 'Insufficient data for trend analysis'}
        
        # Sort by gameweek
        predictions.sort(key=lambda x: x['gameweek'])
        
        # Calculate rolling averages
        gameweeks = [p['gameweek'] for p in predictions]
        accuracies = [p['accuracy'] for p in predictions]
        predicted_points = [p['predicted_points'] for p in predictions]
        actual_points = [p['actual_points'] for p in predictions]
        
        # 3-gameweek rolling average
        rolling_accuracy = []
        rolling_bias = []
        
        for i in range(2, len(predictions)):
            recent_acc = np.mean(accuracies[i-2:i+1])
            recent_bias = np.mean(np.array(predicted_points[i-2:i+1]) - np.array(actual_points[i-2:i+1]))
            rolling_accuracy.append(recent_acc)
            rolling_bias.append(recent_bias)
        
        return {
            'gameweeks': gameweeks,
            'accuracies': accuracies,
            'rolling_accuracy': rolling_accuracy,
            'rolling_bias': rolling_bias,
            'trend_direction': self._calculate_trend_direction(accuracies),
            'consistency_score': self._calculate_consistency_score(accuracies)
        }
    
    def _calculate_trend_direction(self, accuracies: List[float]) -> str:
        """Calculate if performance is improving, declining, or stable."""
        if len(accuracies) < 3:
            return "Insufficient Data"
        
        recent_avg = np.mean(accuracies[-3:])
        early_avg = np.mean(accuracies[:3])
        
        if recent_avg > early_avg + 5:
            return "Improving"
        elif recent_avg < early_avg - 5:
            return "Declining"
        else:
            return "Stable"
    
    def _calculate_consistency_score(self, accuracies: List[float]) -> float:
        """Calculate consistency score (lower standard deviation = more consistent)."""
        if len(accuracies) < 2:
            return 0.0
        
        std_dev = np.std(accuracies)
        # Convert to 0-100 scale (lower std dev = higher consistency)
        max_expected_std = 20  # Assume max reasonable std dev is 20
        consistency = max(0, 100 - (std_dev / max_expected_std * 100))
        return round(consistency, 1)


