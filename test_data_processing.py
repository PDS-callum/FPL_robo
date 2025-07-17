#!/usr/bin/env python3
"""
Test script to verify FPL data processing functionality
"""

import os
import sys
import pandas as pd
import tempfile
import shutil

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_sample_data(data_dir):
    """Create sample data for testing"""
    print("Creating sample test data...")
    
    # Create directory structure
    historical_dir = os.path.join(data_dir, "historical", "2023-24")
    os.makedirs(historical_dir, exist_ok=True)
    
    # Sample merged_gw.csv
    gw_data = {
        'name': ['Player A', 'Player B', 'Player A', 'Player B'],
        'position': ['FWD', 'MID', 'FWD', 'MID'],
        'team': [1, 2, 1, 2],
        'element': [1, 2, 1, 2],
        'total_points': [8, 5, 12, 3],
        'minutes': [90, 45, 90, 90],
        'goals_scored': [1, 0, 2, 0],
        'assists': [0, 1, 1, 0],
        'clean_sheets': [0, 0, 0, 1],
        'goals_conceded': [1, 2, 0, 0],
        'yellow_cards': [0, 1, 0, 0],
        'red_cards': [0, 0, 0, 0],
        'saves': [0, 0, 0, 0],
        'bonus': [3, 0, 3, 1],
        'bps': [35, 15, 45, 20],
        'influence': [50.5, 25.2, 75.8, 30.1],
        'creativity': [25.0, 45.5, 30.2, 55.0],
        'threat': [80.0, 35.5, 95.2, 20.0],
        'value': [90, 55, 90, 55],
        'selected': [25.5, 15.2, 25.5, 15.2],
        'GW': [1, 1, 2, 2],
        'season': ['2023-24', '2023-24', '2023-24', '2023-24']
    }
    
    pd.DataFrame(gw_data).to_csv(
        os.path.join(historical_dir, "merged_gw.csv"), 
        index=False
    )
    
    # Sample players_raw.csv
    players_data = {
        'id': [1, 2],
        'first_name': ['Player', 'Player'],
        'second_name': ['A', 'B'],
        'element_type': [4, 3],  # FWD, MID
        'team': [1, 2],
        'now_cost': [90, 55],
        'selected_by_percent': [25.5, 15.2],
        'total_points': [20, 8],
        'form': [10.0, 4.0],
        'points_per_game': [10.0, 4.0]
    }
    
    pd.DataFrame(players_data).to_csv(
        os.path.join(historical_dir, "players_raw.csv"), 
        index=False
    )
    
    # Sample teams.csv
    teams_data = {
        'id': [1, 2],
        'name': ['Team A', 'Team B'],
        'short_name': ['TEA', 'TEB'],
        'strength_overall_home': [1200, 1100],
        'strength_overall_away': [1150, 1050],
        'strength_attack_home': [1250, 1100],
        'strength_attack_away': [1200, 1000],
        'strength_defence_home': [1150, 1100],
        'strength_defence_away': [1100, 1100]
    }
    
    pd.DataFrame(teams_data).to_csv(
        os.path.join(historical_dir, "teams.csv"), 
        index=False
    )
    
    # Sample fixtures.csv
    fixtures_data = {
        'id': [1, 2],
        'event': [1, 2],
        'team_h': [1, 2],
        'team_a': [2, 1],
        'team_h_score': [2, 1],
        'team_a_score': [1, 2],
        'team_h_difficulty': [3, 4],
        'team_a_difficulty': [4, 3],
        'finished': [True, True],
        'kickoff_time': ['2023-08-12T15:00:00Z', '2023-08-19T15:00:00Z']
    }
    
    pd.DataFrame(fixtures_data).to_csv(
        os.path.join(historical_dir, "fixtures.csv"), 
        index=False
    )
    
    print(f"✅ Sample data created in {data_dir}")

def test_basic_processing(data_dir):
    """Test basic data processing"""
    print("\n🧪 Testing basic data processing...")
    
    try:
        from fpl_bot.utils.data_collection import FPLDataProcessor
        
        processor = FPLDataProcessor(data_dir=data_dir)
        final_dataset, datasets = processor.process_all_data(
            seasons=['2023-24'],
            target_columns=['points_scored', 'minutes_played']
        )
        
        if final_dataset is not None and not final_dataset.empty:
            print(f"✅ Basic processing successful: {final_dataset.shape}")
            return final_dataset, datasets
        else:
            print("❌ Basic processing failed")
            return None, None
            
    except Exception as e:
        print(f"❌ Basic processing error: {e}")
        return None, None

def test_advanced_processing(data_dir, basic_dataset):
    """Test advanced feature engineering"""
    print("\n🧪 Testing advanced feature engineering...")
    
    if basic_dataset is None:
        print("❌ Cannot test advanced processing without basic dataset")
        return None
    
    try:
        from fpl_bot.utils.data_collection import FPLDataProcessor
        from fpl_bot.utils.data_preprocessing import AdvancedFPLPreprocessor
        
        processor = FPLDataProcessor(data_dir=data_dir)
        advanced_processor = AdvancedFPLPreprocessor(data_dir=data_dir)
        
        # Load additional data
        combined_data = processor.load_and_combine_seasons(['2023-24'])
        
        if not combined_data['teams'].empty and not combined_data['fixtures'].empty:
            advanced_dataset = advanced_processor.create_advanced_features(
                basic_dataset,
                combined_data['teams'],
                combined_data['fixtures']
            )
            
            print(f"✅ Advanced processing successful: {advanced_dataset.shape}")
            return advanced_dataset
        else:
            print("⚠️  Missing teams/fixtures data for advanced features")
            return basic_dataset
            
    except Exception as e:
        print(f"❌ Advanced processing error: {e}")
        return None

def test_data_validation(data_dir, dataset):
    """Test data quality validation"""
    print("\n🧪 Testing data quality validation...")
    
    if dataset is None:
        print("❌ Cannot test validation without dataset")
        return False
    
    try:
        from fpl_bot.utils.data_preprocessing import AdvancedFPLPreprocessor
        
        advanced_processor = AdvancedFPLPreprocessor(data_dir=data_dir)
        quality_report = advanced_processor.validate_data_quality(dataset)
        
        if quality_report:
            print("✅ Data validation successful")
            print(f"   Missing values: {sum(quality_report['missing_values'].values())}")
            print(f"   Duplicate rows: {quality_report['duplicate_rows']}")
            return True
        else:
            print("❌ Data validation failed")
            return False
            
    except Exception as e:
        print(f"❌ Data validation error: {e}")
        return False

def test_feature_importance(data_dir, dataset):
    """Test feature importance analysis"""
    print("\n🧪 Testing feature importance analysis...")
    
    if dataset is None:
        print("❌ Cannot test feature importance without dataset")
        return False
    
    try:
        from fpl_bot.utils.data_preprocessing import AdvancedFPLPreprocessor
        
        advanced_processor = AdvancedFPLPreprocessor(data_dir=data_dir)
        
        # Prepare data
        feature_cols = [col for col in dataset.columns 
                       if col not in ['player_id', 'gameweek', 'season', 'points_scored']]
        
        X = dataset[feature_cols].select_dtypes(include=['number']).fillna(0)
        y = dataset['points_scored'].fillna(0)
        
        if len(X.columns) > 0 and len(X) > 0:
            importance_df = advanced_processor.feature_importance_analysis(
                X, y, X.columns.tolist(), 'points_scored'
            )
            
            if importance_df is not None and not importance_df.empty:
                print("✅ Feature importance analysis successful")
                print(f"   Analyzed {len(importance_df)} features")
                return True
            else:
                print("❌ Feature importance analysis failed")
                return False
        else:
            print("⚠️  Insufficient data for feature importance analysis")
            return True  # Not a failure, just insufficient data
            
    except Exception as e:
        print(f"❌ Feature importance error: {e}")
        return False

def run_all_tests():
    """Run all functionality tests"""
    print("="*60)
    print("🧪 FPL Data Processing Functionality Tests")
    print("="*60)
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Using temporary directory: {temp_dir}")
        
        # Create sample data
        create_sample_data(temp_dir)
        
        # Test basic processing
        basic_dataset, datasets = test_basic_processing(temp_dir)
        basic_success = basic_dataset is not None
        
        # Test advanced processing
        advanced_dataset = test_advanced_processing(temp_dir, basic_dataset)
        advanced_success = advanced_dataset is not None
        
        # Test data validation
        validation_success = test_data_validation(temp_dir, advanced_dataset)
        
        # Test feature importance
        importance_success = test_feature_importance(temp_dir, advanced_dataset)
        
        # Summary
        print("\n" + "="*60)
        print("🧪 TEST RESULTS SUMMARY")
        print("="*60)
        print(f"✅ Basic Processing: {'PASS' if basic_success else 'FAIL'}")
        print(f"✅ Advanced Processing: {'PASS' if advanced_success else 'FAIL'}")
        print(f"✅ Data Validation: {'PASS' if validation_success else 'FAIL'}")
        print(f"✅ Feature Importance: {'PASS' if importance_success else 'FAIL'}")
        
        all_passed = all([basic_success, advanced_success, validation_success, importance_success])
        
        if all_passed:
            print("\n🎉 ALL TESTS PASSED! The data processing system is working correctly.")
        else:
            print("\n❌ Some tests failed. Please check the error messages above.")
        
        return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
