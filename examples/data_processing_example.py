#!/usr/bin/env python3
"""
Example usage of FPL Data Processing
This script demonstrates how to process FPL data for machine learning.
"""

import os
import sys
import pandas as pd

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpl_bot.utils.data_collection import FPLDataProcessor
from fpl_bot.utils.data_preprocessing import AdvancedFPLPreprocessor

def example_basic_processing():
    """Example of basic data processing"""
    print("🏈 Running basic FPL data processing example...")
    
    # Initialize processor
    processor = FPLDataProcessor(data_dir="data")
    
    # Process data for all available seasons
    final_dataset, datasets = processor.process_all_data(
        target_columns=['points_scored', 'minutes_played', 'goals_scored']
    )
    
    if final_dataset is not None:
        print(f"✅ Successfully processed {len(final_dataset)} rows")
        print(f"📊 Dataset shape: {final_dataset.shape}")
        print(f"👥 Unique players: {final_dataset['player_id'].nunique()}")
        print(f"🗓️  Gameweeks: {final_dataset['gameweek'].min()} to {final_dataset['gameweek'].max()}")
        
        # Show sample of data
        print("\n📋 Sample of processed data:")
        print(final_dataset[['player_id', 'gameweek', 'position', 'points_scored', 
                           'avg_points_3gw', 'total_points_season']].head())
        
        return final_dataset, datasets
    else:
        print("❌ Basic processing failed")
        return None, None

def example_advanced_processing(basic_dataset):
    """Example of advanced feature engineering"""
    print("\n🔧 Running advanced feature engineering example...")
    
    if basic_dataset is None:
        print("❌ Cannot run advanced processing without basic dataset")
        return None
    
    # Initialize processors
    processor = FPLDataProcessor(data_dir="data")
    advanced_processor = AdvancedFPLPreprocessor(data_dir="data")
    
    # Load additional data needed for advanced features
    combined_data = processor.load_and_combine_seasons()
    
    if combined_data['teams'].empty or combined_data['fixtures'].empty:
        print("⚠️  Missing teams/fixtures data for advanced features")
        return basic_dataset
    
    # Create advanced features
    advanced_dataset = advanced_processor.create_advanced_features(
        basic_dataset, 
        combined_data['teams'], 
        combined_data['fixtures']
    )
    
    print(f"✅ Advanced features created")
    print(f"📊 Advanced dataset shape: {advanced_dataset.shape}")
    print(f"🔧 New features added: {advanced_dataset.shape[1] - basic_dataset.shape[1]}")
    
    # Show new feature examples
    new_features = [col for col in advanced_dataset.columns if col not in basic_dataset.columns]
    if new_features:
        print(f"\n🆕 Example new features: {new_features[:10]}")
    
    # Save advanced dataset
    output_path = advanced_processor.save_feature_engineered_data(advanced_dataset)
    print(f"💾 Advanced dataset saved to: {output_path}")
    
    return advanced_dataset

def example_feature_importance(advanced_dataset):
    """Example of feature importance analysis"""
    print("\n📈 Running feature importance analysis example...")
    
    if advanced_dataset is None:
        print("❌ Cannot analyze features without advanced dataset")
        return None
    
    advanced_processor = AdvancedFPLPreprocessor(data_dir="data")
    
    # Prepare data for analysis
    target = 'points_scored'
    feature_cols = [col for col in advanced_dataset.columns 
                   if col not in ['player_id', 'gameweek', 'season', 'points_scored', 
                                 'minutes_played', 'goals_scored', 'assists']]
    
    # Select only numeric columns and handle missing values
    X = advanced_dataset[feature_cols].select_dtypes(include=['number']).fillna(0)
    y = advanced_dataset[target].fillna(0)
    
    print(f"🎯 Analyzing {len(X.columns)} features for target: {target}")
    print(f"📊 Using {len(X)} samples")
    
    # Run feature importance analysis
    importance_df = advanced_processor.feature_importance_analysis(
        X, y, X.columns.tolist(), target
    )
    
    print(f"✅ Feature importance analysis complete")
    print(f"🏆 Top 5 most important features:")
    for i, (_, row) in enumerate(importance_df.head(5).iterrows(), 1):
        print(f"  {i}. {row['feature']}: {row['importance']:.4f}")
    
    return importance_df

def example_prediction_ready_data(advanced_dataset):
    """Example of preparing data for machine learning models"""
    print("\n🤖 Preparing data for machine learning example...")
    
    if advanced_dataset is None:
        print("❌ Cannot prepare ML data without processed dataset")
        return None
    
    # Split features and target
    target_col = 'points_scored'
    feature_cols = [col for col in advanced_dataset.columns 
                   if col not in ['player_id', 'gameweek', 'season', target_col]]
    
    X = advanced_dataset[feature_cols].select_dtypes(include=['number']).fillna(0)
    y = advanced_dataset[target_col].fillna(0)
    
    # Split by time (earlier gameweeks for training, later for testing)
    split_gw = advanced_dataset['gameweek'].quantile(0.8)
    
    train_mask = advanced_dataset['gameweek'] <= split_gw
    test_mask = advanced_dataset['gameweek'] > split_gw
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    print(f"📊 ML Data Preparation Summary:")
    print(f"  Features: {len(X.columns)}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Target variable: {target_col}")
    print(f"  Split point: Gameweek {split_gw:.0f}")
    
    # Show basic statistics
    print(f"\n📈 Target Variable Statistics:")
    print(f"  Training mean: {y_train.mean():.2f}")
    print(f"  Training std: {y_train.std():.2f}")
    print(f"  Test mean: {y_test.mean():.2f}")
    print(f"  Test std: {y_test.std():.2f}")
    
    return X_train, X_test, y_train, y_test

def main():
    """Run all examples"""
    print("="*60)
    print("FPL Data Processing Examples")
    print("="*60)
    
    # Check if data exists
    if not os.path.exists("data/historical"):
        print("❌ No historical data found in 'data/historical' directory")
        print("💡 Please run data collection first or check your data directory")
        return
    
    # Run examples
    try:
        # Basic processing
        basic_dataset, datasets = example_basic_processing()
        
        # Advanced processing
        advanced_dataset = example_advanced_processing(basic_dataset)
        
        # Feature importance
        importance_df = example_feature_importance(advanced_dataset)
        
        # ML preparation
        ml_data = example_prediction_ready_data(advanced_dataset)
        
        print("\n" + "="*60)
        print("🎉 All examples completed successfully!")
        print("="*60)
        print("\n📁 Check the following directories for outputs:")
        print("  - data/processed/ - Processed datasets")
        print("  - data/features/ - ML-ready features and targets")
        print("\n💡 Next steps:")
        print("  - Use the processed data to train ML models")
        print("  - Experiment with different feature combinations")
        print("  - Build prediction and optimization systems")
        
    except Exception as e:
        print(f"\n❌ Error during example execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
