#!/usr/bin/env python3
"""
FPL Data Processing Pipeline
This script processes Fantasy Premier League data for machine learning training.
"""

import sys
import os
import argparse
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpl_bot.utils.data_collection import FPLDataProcessor
from fpl_bot.utils.data_preprocessing import AdvancedFPLPreprocessor

def main():
    parser = argparse.ArgumentParser(description='Process FPL data for machine learning')
    parser.add_argument('--data-dir', default='data', help='Data directory path')
    parser.add_argument('--seasons', nargs='+', help='Specific seasons to process (e.g., 2023-24 2024-25)')
    parser.add_argument('--targets', nargs='+', default=['points_scored', 'minutes_played', 'goals_scored', 'assists'],
                       help='Target variables to prepare for ML')
    parser.add_argument('--advanced', action='store_true', help='Create advanced features')
    parser.add_argument('--validate', action='store_true', help='Run data quality validation')
    parser.add_argument('--feature-importance', action='store_true', help='Analyze feature importance')
    
    args = parser.parse_args()
    
    print("="*60)
    print("FPL Data Processing Pipeline")
    print("="*60)
    print(f"Start time: {datetime.now()}")
    print(f"Data directory: {args.data_dir}")
    print(f"Seasons: {args.seasons if args.seasons else 'All available'}")
    print(f"Target variables: {args.targets}")
    print(f"Advanced features: {args.advanced}")
    print("="*60)
    
    try:
        # Initialize processors
        processor = FPLDataProcessor(data_dir=args.data_dir)
        
        # Step 1: Process basic data
        print("\n🏈 Step 1: Processing basic FPL data...")
        final_dataset, datasets = processor.process_all_data(
            seasons=args.seasons, 
            target_columns=args.targets
        )
        
        if final_dataset is None:
            print("❌ Failed to process basic data. Check that data exists in the historical folder.")
            return 1
        
        print(f"✅ Basic processing complete. Dataset shape: {final_dataset.shape}")
        
        # Step 2: Advanced feature engineering (if requested)
        if args.advanced:
            print("\n🔧 Step 2: Creating advanced features...")
            advanced_processor = AdvancedFPLPreprocessor(data_dir=args.data_dir)
            
            # Load additional data needed for advanced features
            combined_data = processor.load_and_combine_seasons(args.seasons)
            
            if not combined_data['teams'].empty and not combined_data['fixtures'].empty:
                advanced_dataset = advanced_processor.create_advanced_features(
                    final_dataset, 
                    combined_data['teams'], 
                    combined_data['fixtures']
                )
                
                # Save advanced dataset
                advanced_path = advanced_processor.save_feature_engineered_data(
                    advanced_dataset, 
                    suffix="advanced"
                )
                
                print(f"✅ Advanced features created. Dataset shape: {advanced_dataset.shape}")
                print(f"📁 Saved to: {advanced_path}")
                
                # Use advanced dataset for further analysis
                analysis_dataset = advanced_dataset
            else:
                print("⚠️  Teams/fixtures data not available for advanced features. Using basic dataset.")
                analysis_dataset = final_dataset
        else:
            analysis_dataset = final_dataset
        
        # Step 3: Data quality validation (if requested)
        if args.validate:
            print("\n🔍 Step 3: Validating data quality...")
            if args.advanced:
                quality_report = advanced_processor.validate_data_quality(analysis_dataset)
            else:
                # Create a basic validation for non-advanced datasets
                print("Basic validation completed (advanced validation requires --advanced flag)")
                quality_report = {
                    'missing_values': analysis_dataset.isnull().sum().to_dict(),
                    'shape': analysis_dataset.shape,
                    'memory_usage': analysis_dataset.memory_usage().sum()
                }
            
            print("✅ Data quality validation complete")
            print(f"📊 Missing values found: {sum(quality_report['missing_values'].values())}")
        
        # Step 4: Feature importance analysis (if requested)
        if args.feature_importance and args.advanced:
            print("\n📈 Step 4: Analyzing feature importance...")
            
            for target in args.targets:
                if target in analysis_dataset.columns:
                    print(f"\nAnalyzing features for target: {target}")
                    
                    # Prepare features for analysis
                    feature_cols = [col for col in analysis_dataset.columns 
                                  if col not in ['player_id', 'gameweek', 'season'] + args.targets]
                    
                    X = analysis_dataset[feature_cols].select_dtypes(include=['number']).fillna(0)
                    y = analysis_dataset[target].fillna(0)
                    
                    # Run feature importance analysis
                    importance_df = advanced_processor.feature_importance_analysis(
                        X, y, X.columns.tolist(), target
                    )
                    
                    print(f"✅ Feature importance analysis complete for {target}")
        
        # Step 5: Generate summary report
        print("\n📋 Step 5: Generating summary report...")
        
        summary = processor.get_data_summary()
        if summary:
            print("\n" + "="*50)
            print("DATA PROCESSING SUMMARY")
            print("="*50)
            print(f"📊 Total rows: {summary['total_rows']:,}")
            print(f"👥 Unique players: {summary['unique_players']:,}")
            print(f"🗓️  Gameweeks covered: {summary['gameweeks_covered']}")
            print(f"🎯 Features count: {summary['features_count']}")
            print(f"⚽ Average points per gameweek: {summary['avg_points_per_gw']:.2f}")
            print("\n📍 Position distribution:")
            for pos, count in summary['positions_distribution'].items():
                pos_name = {1: 'Goalkeepers', 2: 'Defenders', 3: 'Midfielders', 4: 'Forwards'}.get(pos, f'Position {pos}')
                print(f"  {pos_name}: {count:,}")
        
        print("\n" + "="*60)
        print("🎉 DATA PROCESSING COMPLETE!")
        print("="*60)
        print(f"End time: {datetime.now()}")
        
        # Show where files are saved
        print(f"\n📁 Processed data location:")
        print(f"  - Main dataset: {os.path.join(args.data_dir, 'processed', 'fpl_ml_dataset.csv')}")
        print(f"  - Feature files: {os.path.join(args.data_dir, 'features')}/")
        if args.advanced:
            print(f"  - Advanced dataset: {os.path.join(args.data_dir, 'processed', 'fpl_dataset_advanced.csv')}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
