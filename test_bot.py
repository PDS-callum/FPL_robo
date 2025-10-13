#!/usr/bin/env python3
"""
Test script for FPL Bot

Tests the basic functionality of the FPL Bot without requiring a full analysis.
"""

import sys
import os

# Add the package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'fpl_bot'))

from fpl_bot.core.data_collector import DataCollector
from fpl_bot.core.predictor import Predictor
from fpl_bot.core.transfer_optimizer import TransferOptimizer
from fpl_bot.core.chip_manager import ChipManager


def test_data_collection():
    """Test data collection functionality"""
    print("Testing data collection...")
    
    collector = DataCollector()
    
    # Test current season data
    season_data = collector.get_current_season_data()
    if season_data:
        print(f"[OK] Successfully collected data for {len(season_data['elements'])} players")
        
        # Test DataFrame creation
        players_df = collector.create_players_dataframe(season_data)
        if players_df is not None:
            print(f"[OK] Successfully created DataFrame with {len(players_df)} players")
            print(f"  Columns: {list(players_df.columns)}")
            return players_df
        else:
            print("[FAIL] Failed to create DataFrame")
            return None
    else:
        print("[FAIL] Failed to collect season data")
        return None


def test_predictions(players_df):
    """Test prediction functionality"""
    print("\nTesting predictions...")
    
    if players_df is None:
        print("[FAIL] Cannot test predictions without players data")
        return None
    
    collector = DataCollector()
    predictor = Predictor(collector)
    
    # Get teams data for fixture difficulty calculations
    season_data = collector.get_current_season_data()
    teams_data = season_data.get('teams', []) if season_data else []
    
    # Test predictions
    predictions = predictor.predict_next_gameweek(players_df, None, teams_data)
    if predictions is not None and not predictions.empty:
        print(f"[OK] Successfully generated predictions for {len(predictions)} players")
        print(f"  Top 3 predictions:")
        for _, player in predictions.head(3).iterrows():
            print(f"    {player['web_name']}: {player['predicted_points']:.1f} points")
        return predictions
    else:
        print("[FAIL] Failed to generate predictions")
        return None


def test_transfer_optimization(predictions):
    """Test transfer optimization"""
    print("\nTesting transfer optimization...")
    
    if predictions is None:
        print("[FAIL] Cannot test transfer optimization without predictions")
        return
    
    collector = DataCollector()
    predictor = Predictor(collector)
    optimizer = TransferOptimizer(predictor)
    
    # Create a mock current team (first 15 players)
    current_team = predictions.head(15)['player_id'].tolist()
    
    # Test optimization
    result = optimizer.optimize_transfers(current_team, predictions, budget=5.0)
    
    if result and 'best_scenario' in result:
        best_scenario = result['best_scenario']
        print(f"[OK] Successfully optimized transfers")
        print(f"  Best scenario: {best_scenario['num_transfers']} transfers")
        print(f"  Net points gain: {best_scenario.get('net_points_gained', 0):.1f}")
    else:
        print("[FAIL] Failed to optimize transfers")


def test_chip_management():
    """Test chip management"""
    print("\nTesting chip management...")
    
    collector = DataCollector()
    chip_manager = ChipManager(collector)
    
    # Test chip status
    chips = chip_manager.get_chip_status(1791695)  # Test with a real manager ID
    
    if chips:
        print("[OK] Successfully retrieved chip status")
        for chip_name, status in chips.items():
            available = "[OK]" if status['available'] else "[USED]"
            print(f"  {chip_name}: {'Available' if status['available'] else 'Used'}")
    else:
        print("[FAIL] Failed to retrieve chip status")


def main():
    """Run all tests"""
    print("FPL Bot Test Suite")
    print("=" * 40)
    
    # Test data collection
    players_df = test_data_collection()
    
    # Test predictions
    predictions = test_predictions(players_df)
    
    # Test transfer optimization
    test_transfer_optimization(predictions)
    
    # Test chip management
    test_chip_management()
    
    print("\n" + "=" * 40)
    print("Test suite complete!")
    
    if players_df is not None and predictions is not None:
        print("[OK] All core functionality working")
        return 0
    else:
        print("[FAIL] Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
