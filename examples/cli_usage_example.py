#!/usr/bin/env python3
"""
Example usage of the new FPL Bot CLI data processing functionality

This example demonstrates how to:
1. Collect historical FPL data from GitHub
2. Process the data into ML-ready format
3. Use the processed data for training

The data that gets collected includes:
- fixtures.csv: Match fixtures and results
- merged_gw.csv: Player performance data by gameweek
- players_raw.csv: Player information and season stats
- teams.csv: Team information and strength ratings
"""

import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpl_bot.utils.data_collection import FPLDataCollector, FPLDataProcessor

def demo_data_collection_and_processing():
    """Demonstrate the data collection and processing workflow"""
    
    print("="*60)
    print("FPL Bot Data Processing Demo")
    print("="*60)
    
    # Step 1: Initialize the data collector
    print("\n1. Initializing data collector...")
    collector = FPLDataCollector(data_dir="data")
    
    # Show available seasons
    available_seasons = collector.get_available_seasons()
    print(f"Available seasons: {available_seasons}")
    
    # Step 2: Collect data for a specific season (demo with 2023-24)
    demo_season = "2023-24"
    print(f"\n2. Collecting data for {demo_season}...")
    
    try:
        season_data = collector.collect_season_data(demo_season)
        print(f"✅ Data collection complete for {demo_season}")
        
        # Show what was collected
        for data_type, data in season_data.items():
            if data is not None:
                print(f"  - {data_type}: {data.shape[0]} records")
    except Exception as e:
        print(f"❌ Data collection failed: {e}")
        return
    
    # Step 3: Process the collected data
    print(f"\n3. Processing data for ML training...")
    processor = FPLDataProcessor(data_dir="data")
    
    try:
        # Process data for multiple targets
        target_variables = ['points_scored', 'minutes_played', 'goals_scored', 'assists']
        
        final_dataset, datasets = processor.process_all_data(
            seasons=[demo_season],
            target_columns=target_variables
        )
        
        if final_dataset is not None:
            print(f"✅ Data processing complete!")
            print(f"Final dataset shape: {final_dataset.shape}")
            print(f"Features created: {len(final_dataset.columns)}")
            
            # Show summary
            summary = processor.get_data_summary()
            if summary:
                print(f"\n📊 Summary:")
                print(f"  - Total rows: {summary['total_rows']:,}")
                print(f"  - Unique players: {summary['unique_players']:,}")
                print(f"  - Gameweeks covered: {summary['gameweeks_covered']}")
                print(f"  - Average points per gameweek: {summary['avg_points_per_gw']:.2f}")
            
            # Show where files are saved
            print(f"\n📁 Files saved to:")
            print(f"  - data/processed/fpl_ml_dataset.csv")
            print(f"  - data/features/features_*.csv")
            print(f"  - data/features/target_*.csv")
            
        else:
            print("❌ Data processing failed")
            
    except Exception as e:
        print(f"❌ Data processing failed: {e}")
        import traceback
        traceback.print_exc()

def show_cli_usage():
    """Show how to use the CLI commands"""
    print("\n" + "="*60)
    print("CLI Usage Examples")
    print("="*60)
    
    print("\n🔧 Collect data only:")
    print("python -m fpl_bot.main collect --seasons 2023-24")
    print("python -m fpl_bot.main collect --all")
    
    print("\n⚙️  Process data only (requires existing data):")
    print("python -m fpl_bot.main process --seasons 2023-24")
    print("python -m fpl_bot.main process --targets points_scored goals_scored")
    
    print("\n🚀 Collect and process in one command:")
    print("python -m fpl_bot.main collect-and-process --seasons 2023-24")
    print("python -m fpl_bot.main collect-and-process --all")
    
    print("\n📊 Process with custom targets:")
    print("python -m fpl_bot.main collect-and-process --seasons 2022-23 2023-24 --targets points_scored minutes_played")

def show_data_structure():
    """Show the structure of the processed data"""
    print("\n" + "="*60)
    print("Data Structure")
    print("="*60)
    
    print("\n📂 Raw Historical Data (data/historical/):")
    print("  └── 2023-24/")
    print("      ├── fixtures.csv      # Match fixtures and results")
    print("      ├── merged_gw.csv     # Player performance by gameweek")
    print("      ├── players_raw.csv   # Player information")
    print("      └── teams.csv         # Team information")
    
    print("\n📂 Processed Data (data/processed/):")
    print("  ├── fpl_ml_dataset.csv    # Main ML-ready dataset")
    print("  ├── label_encoders.pkl    # Categorical encoders")
    print("  └── scalers.pkl           # Feature scalers")
    
    print("\n📂 Feature Files (data/features/):")
    print("  ├── features_points_scored.csv  # Features for points prediction")
    print("  ├── target_points_scored.csv    # Points target values")
    print("  ├── features_goals_scored.csv   # Features for goals prediction")
    print("  ├── target_goals_scored.csv     # Goals target values")
    print("  └── feature_names_*.json        # Feature name mappings")

if __name__ == "__main__":
    try:
        print("This example demonstrates the FPL Bot data processing capabilities.")
        print("Choose an option:")
        print("1. Run data collection and processing demo")
        print("2. Show CLI usage examples")
        print("3. Show data structure information")
        print("4. All of the above")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            demo_data_collection_and_processing()
        elif choice == "2":
            show_cli_usage()
        elif choice == "3":
            show_data_structure()
        elif choice == "4":
            demo_data_collection_and_processing()
            show_cli_usage()
            show_data_structure()
        else:
            print("Invalid choice. Showing all information:")
            show_cli_usage()
            show_data_structure()
            
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nError running demo: {e}")
        import traceback
        traceback.print_exc()
