import os
import argparse
from .utils.data_collection import FPLDataCollector, FPLDataProcessor
from .utils.data_conversion import create_api_compatible_data
from .utils.constants import POSITION_NAMES, LIMITED_SEASONS, AVAILABLE_SEASONS
from .train_model import train_model, iterative_training_update
from .predict_team import predict_team_for_gameweek
from .iterative_season_manager import FPLIterativeSeasonManager, run_season_manager

def main():
    """Main entry point for FPL Bot"""
    parser = argparse.ArgumentParser(description='FPL Bot with CNN modeling')
    
    # Create subparsers for different actions
    subparsers = parser.add_subparsers(dest='action', help='Action to perform')
    
    # Collect multi data parser (replaces the old collect parser)
    collect_parser = subparsers.add_parser('collect', help='Collect FPL data from multiple seasons')
    collect_parser.add_argument('--seasons', type=str, nargs='+', help='Seasons to collect data for (e.g., 2022-23 2023-24)')
    collect_parser.add_argument('--all', action='store_true', help='Collect data for seasons 2019-20 to 2024-25')
    
    # Process data parser
    process_parser = subparsers.add_parser('process', help='Process collected historical data for ML training')
    process_parser.add_argument('--seasons', type=str, nargs='+', help='Specific seasons to process (e.g., 2022-23 2023-24)')
    process_parser.add_argument('--targets', type=str, nargs='+', default=['points_scored', 'minutes_played', 'goals_scored', 'assists'],
                               help='Target variables to prepare for ML')
    process_parser.add_argument('--data-dir', default='data', help='Data directory path')
    
    # Collect and process data parser
    collect_process_parser = subparsers.add_parser('collect-and-process', help='Collect historical data and then process it for ML training')
    collect_process_parser.add_argument('--seasons', type=str, nargs='+', help='Seasons to collect and process (e.g., 2022-23 2023-24)')
    collect_process_parser.add_argument('--all', action='store_true', help='Collect and process data for seasons 2019-20 to 2024-25')
    collect_process_parser.add_argument('--targets', type=str, nargs='+', default=['points_scored', 'minutes_played', 'goals_scored', 'assists'],
                                       help='Target variables to prepare for ML')
    collect_process_parser.add_argument('--data-dir', default='data', help='Data directory path')
    
    # Train model parser
    train_parser = subparsers.add_parser('train', help='Train FPL prediction model')
    train_parser.add_argument('--target', type=str, default='points_scored',
                             choices=['points_scored', 'goals_scored', 'assists', 'minutes_played'],
                             help='Target variable to predict')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    train_parser.add_argument('--no-current-season', action='store_true',
                             help='Exclude current season data from training')
    train_parser.add_argument('--seasons', nargs='+', help='Specific historical seasons to include')
    train_parser.add_argument('--data-dir', default='data', help='Data directory path')
    train_parser.add_argument('--iterative-update', action='store_true',
                             help='Perform iterative update instead of full training')
    train_parser.add_argument('--gameweek', type=int, help='Specific gameweek for iterative update')

    # Predict team parser
    predict_parser = subparsers.add_parser('predict', help='Predict optimal FPL team for a gameweek')
    predict_parser.add_argument('--gameweek', type=int, help='Gameweek to predict for (default: next gameweek)')
    predict_parser.add_argument('--budget', type=float, default=100.0, 
                               help='Available budget in millions (default: 100.0)')
    predict_parser.add_argument('--target', type=str, default='points_scored',
                               choices=['points_scored', 'goals_scored', 'assists', 'minutes_played'],
                               help='Target model to use for predictions')
    predict_parser.add_argument('--data-dir', default='data', help='Data directory path')
    predict_parser.add_argument('--no-save', action='store_true', help='Do not save prediction results')
    
    # Iterative season management parser
    season_parser = subparsers.add_parser('run-season', help='Run iterative season management')
    season_parser.add_argument('--target', type=str, default='points_scored',
                              choices=['points_scored', 'goals_scored', 'assists', 'minutes_played'],
                              help='Target variable for predictions')
    season_parser.add_argument('--budget', type=float, default=100.0,
                              help='Team budget in millions (default: 100.0)')
    season_parser.add_argument('--start-gameweek', type=int, default=1,
                              help='Gameweek to start from (default: 1)')
    season_parser.add_argument('--initial-epochs', type=int, default=100,
                              help='Epochs for initial training (default: 100)')
    season_parser.add_argument('--data-dir', default='data', help='Data directory path')
    
    # Resume season management parser
    resume_parser = subparsers.add_parser('resume-season', help='Resume existing season management')
    resume_parser.add_argument('--data-dir', default='data', help='Data directory path')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.action == 'collect':
        # Collect data using only historical data collector (GitHub source)
        print("Collecting FPL data from GitHub repository...")
        collector = FPLDataCollector()
        all_available_seasons = collector.get_available_seasons()
        
        # Define the range of seasons for --all option (2019-20 to 2024-25)
        limited_seasons = LIMITED_SEASONS
        
        # Determine which seasons to collect
        if args.all:
            seasons_to_collect = limited_seasons
            print(f"--all flag will collect seasons from 2019-20 to 2024-25 only")
        elif args.seasons:
            seasons_to_collect = args.seasons
            # Validate specified seasons
            for season in seasons_to_collect:
                if season not in all_available_seasons:
                    print(f"Warning: Season {season} is not in the list of available seasons: {', '.join(all_available_seasons)}")
                    print("Proceeding anyway, but data collection may fail.")
        else:
            # Default to just the latest season
            seasons_to_collect = [collector.get_latest_season()]
        
        print(f"Collecting data for seasons: {', '.join(seasons_to_collect)}")
        
        # Create raw directory expected by other components
        raw_dir = os.path.join("data", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        
        # Collect data for each season
        for season in seasons_to_collect:
            print(f"\nCollecting data for {season}...")
            collector.collect_season_data(season)
            
            # Convert to API compatible format
            print(f"Converting {season} data to API compatible format...")
            create_api_compatible_data(season)
            print("\nData collection complete!")
    
    elif args.action == 'process':
        # Process historical data for ML training
        print("Processing historical FPL data for ML training...")
        
        # Import the data processor from data_collection
        data_processor = FPLDataProcessor(data_dir=args.data_dir)
        
        # Process the data
        print(f"Processing data for targets: {args.targets}")
        if args.seasons:
            print(f"Processing specific seasons: {args.seasons}")
        else:
            print("Processing all available seasons")
        
        final_dataset, datasets = data_processor.process_all_data(
            seasons=args.seasons, 
            target_columns=args.targets
        )
        
        if final_dataset is not None:
            print("\n" + "="*60)
            print("DATA PROCESSING COMPLETE!")
            print("="*60)
            
            # Show summary
            summary = data_processor.get_data_summary()
            if summary:
                print(f"📊 Total rows: {summary['total_rows']:,}")
                print(f"👥 Unique players: {summary['unique_players']:,}")
                print(f"🗓️  Gameweeks covered: {summary['gameweeks_covered']}")
                print(f"🎯 Features count: {summary['features_count']}")
                print(f"⚽ Average points per gameweek: {summary['avg_points_per_gw']:.2f}")
                print("\n📍 Position distribution:")
                for pos, count in summary['positions_distribution'].items():
                    pos_name = POSITION_NAMES.get(pos, f'Position {pos}')
                    print(f"  {pos_name}: {count:,}")
            
            print(f"\n📁 Processed data saved to:")
            print(f"  - Main dataset: {os.path.join(args.data_dir, 'processed', 'fpl_ml_dataset.csv')}")
            print(f"  - Feature files: {os.path.join(args.data_dir, 'features')}/")
            print(f"  - Models directory: {os.path.join(args.data_dir, 'models')}/")
        else:
            print("❌ Data processing failed. Check that historical data exists in the data/historical folder.")
    
    elif args.action == 'collect-and-process':
        # Collect and then process historical data
        print("Collecting and processing FPL data...")
        
        # Step 1: Collect data
        print("\n" + "="*60)
        print("STEP 1: COLLECTING DATA")
        print("="*60)
        
        collector = FPLDataCollector()
        all_available_seasons = collector.get_available_seasons()
        
        # Define the range of seasons for --all option (2019-20 to 2024-25)
        limited_seasons = LIMITED_SEASONS
        
        # Determine which seasons to collect
        if args.all:
            seasons_to_collect = limited_seasons
            print(f"--all flag will collect seasons from 2019-20 to 2024-25 only")
        elif args.seasons:
            seasons_to_collect = args.seasons
            # Validate specified seasons
            for season in seasons_to_collect:
                if season not in all_available_seasons:
                    print(f"Warning: Season {season} is not in the list of available seasons: {', '.join(all_available_seasons)}")
                    print("Proceeding anyway, but data collection may fail.")
        else:
            # Default to just the latest season
            seasons_to_collect = [collector.get_latest_season()]
        
        print(f"Collecting data for seasons: {', '.join(seasons_to_collect)}")
        
        # Create raw directory expected by other components
        raw_dir = os.path.join(getattr(args, 'data_dir', 'data'), "raw")
        os.makedirs(raw_dir, exist_ok=True)
        
        # Collect data for each season
        for season in seasons_to_collect:
            print(f"\nCollecting data for {season}...")
            collector.collect_season_data(season)
            
            # Convert to API compatible format
            print(f"Converting {season} data to API compatible format...")
            create_api_compatible_data(season)
        
        print("Data collection complete!")
        
        # Step 2: Process data
        print("\n" + "="*60)
        print("STEP 2: PROCESSING DATA")
        print("="*60)
        
        data_processor = FPLDataProcessor(data_dir=getattr(args, 'data_dir', 'data'))
        
        # Process the data
        print(f"Processing data for targets: {args.targets}")
        print(f"Processing seasons: {seasons_to_collect}")
        
        final_dataset, datasets = data_processor.process_all_data(
            seasons=seasons_to_collect, 
            target_columns=args.targets
        )
        
        if final_dataset is not None:
            print("\n" + "="*60)
            print("COLLECT AND PROCESS COMPLETE!")
            print("="*60)
            
            # Show summary
            summary = data_processor.get_data_summary()
            if summary:
                print(f"📊 Total rows: {summary['total_rows']:,}")
                print(f"👥 Unique players: {summary['unique_players']:,}")
                print(f"🗓️  Gameweeks covered: {summary['gameweeks_covered']}")
                print(f"🎯 Features count: {summary['features_count']}")
                print(f"⚽ Average points per gameweek: {summary['avg_points_per_gw']:.2f}")
                print("\n📍 Position distribution:")
                for pos, count in summary['positions_distribution'].items():
                    pos_name = POSITION_NAMES.get(pos, f'Position {pos}')
                    print(f"  {pos_name}: {count:,}")
            
            data_dir = getattr(args, 'data_dir', 'data')
            print(f"\n📁 Processed data saved to:")
            print(f"  - Main dataset: {os.path.join(data_dir, 'processed', 'fpl_ml_dataset.csv')}")
            print(f"  - Feature files: {os.path.join(data_dir, 'features')}/")
            print(f"  - Models directory: {os.path.join(data_dir, 'models')}/")
            
            print(f"\n🚀 Ready for model training! You can now train models using the processed data.")
        else:
            print("❌ Data processing failed after collection.")
    
    elif args.action == 'train':
        # Train FPL prediction model
        print("Training FPL prediction model...")
        
        if args.iterative_update:
            # Perform iterative update
            model, info = iterative_training_update(
                gameweek=args.gameweek,
                target=args.target,
                data_dir=args.data_dir
            )
        else:
            # Perform full training
            model, info = train_model(
                target=args.target,
                epochs=args.epochs,
                batch_size=args.batch_size,
                include_current_season=not args.no_current_season,
                historical_seasons=args.seasons,
                data_dir=args.data_dir
            )
    
    elif args.action == 'predict':
        # Predict optimal FPL team
        print("Predicting optimal FPL team...")
        
        results = predict_team_for_gameweek(
            gameweek=args.gameweek,
            budget=args.budget,
            target=args.target,
            data_dir=args.data_dir,
            save_results=not args.no_save        )
        
        if results is None:
            print("❌ Prediction failed. Please ensure you have a trained model.")
    
    elif args.action == 'run-season':
        # Run iterative season management
        print("🏆 Starting FPL iterative season management...")
        
        success = run_season_manager(
            target=args.target,
            budget=args.budget,
            start_gameweek=args.start_gameweek,
            initial_training_epochs=args.initial_epochs,
            data_dir=args.data_dir
        )
        
        if success:
            print("✅ Season management completed successfully!")
        else:
            print("❌ Season management failed or was interrupted.")
    
    elif args.action == 'resume-season':
        # Resume existing season management
        print("🔄 Resuming FPL season management...")
        
        manager = FPLIterativeSeasonManager(data_dir=args.data_dir)
        manager.resume_season()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()