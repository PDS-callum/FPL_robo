import os
import argparse
import pandas as pd
import numpy as np
import json
from datetime import datetime
# Remove FPLDataCollector since we're only using GitHub data
from .utils.history_data_collector import FPLHistoricalDataCollector
from .utils.data_processing import FPLDataProcessor, MultiSeasonDataProcessor
from .train_model import train_model, train_model_with_history
from .predict_team import predict_team_for_gameweek
from .utils.data_conversion import create_api_compatible_data
from .utils.enhanced_data_preparation import FPLDataPreprocessor, create_directories

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
    process_parser = subparsers.add_parser('process', help='Process collected data')
    process_parser.add_argument('--lookback', type=int, default=3, help='Number of gameweeks to look back (default: 3)')
    process_parser.add_argument('--seasons', type=str, nargs='+', help='Seasons to process (e.g., 2022-23 2023-24)')
    process_parser.add_argument('--all', action='store_true', help='Process data for all collected seasons')
    
    # Train model parser
    train_parser = subparsers.add_parser('train', help='Train CNN model')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (default: 50)')
    train_parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    
    # Predict team parser
    predict_parser = subparsers.add_parser('predict', help='Predict optimal team')
    predict_parser.add_argument('--gameweek', type=int, help='Gameweek number to predict for')
    predict_parser.add_argument('--budget', type=float, default=100.0, help='Available budget (default: 100.0)')
    predict_parser.add_argument('--lookback', type=int, default=3, help='Number of gameweeks to look back (default: 3)')
    predict_parser.add_argument('--apply', action='store_true', help='Apply transfers to team state')
    predict_parser.add_argument('--use-history', action='store_true', help='Use model trained with historical data')
    predict_parser.add_argument('--cutoff-gw', type=int, help='Use data only up to this gameweek')
    predict_parser.add_argument('--next-season', action='store_true', 
                           help='Predict for first gameweek of next season using recent data')
    predict_parser.add_argument('--teams', type=str, nargs='+', 
                           help='List of teams in the next season (e.g., "Arsenal" "Man City" "Liverpool")')
    
    # Add new options for transfers
    transfers_parser = subparsers.add_parser('transfers', help='Suggest transfers for next gameweek')
    transfers_parser.add_argument('--gameweek', type=int, help='Gameweek number to predict for')
    transfers_parser.add_argument('--budget', type=float, default=100.0, help='Available budget (default: 100.0)')
    transfers_parser.add_argument('--lookback', type=int, default=3, help='Number of gameweeks to look back (default: 3)')
    transfers_parser.add_argument('--apply', action='store_true', help='Apply transfers to team state')
    
    # View team parser
    view_parser = subparsers.add_parser('view-team', help='View current team')
    
    # Add historical data parser
    history_parser = subparsers.add_parser('fetch-history', help='Fetch historical data')
    history_parser.add_argument('--seasons', type=str, nargs='+', help='Seasons to fetch (e.g., 2022-23)')
    history_parser.add_argument('--all', action='store_true', help='Fetch all available seasons')
    
    # Add train with history parser
    train_history_parser = subparsers.add_parser('train-with-history', help='Train model with historical data')
    train_history_parser.add_argument('--seasons', type=str, nargs='+', help='Seasons to use (e.g., 2022-23)')
    train_history_parser.add_argument('--all', action='store_true', help='Use all available seasons for training')
    train_history_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (default: 50)')
    train_history_parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.action == 'collect':
        # Collect data using only historical data collector (GitHub source)
        print("Collecting FPL data from GitHub repository...")
        collector = FPLHistoricalDataCollector()
        all_available_seasons = collector.get_available_seasons()
        
        # Define the range of seasons for --all option (2019-20 to 2024-25)
        limited_seasons = [s for s in all_available_seasons if s >= "2019-20" and s <= "2024-25"]
        
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
        # Process data using enhanced preprocessor for optimal model training
        print("Processing FPL data with enhanced preprocessing...")
        
        # Create all required directories
        create_directories()
        
        # Use the historical data collector to get available seasons
        collector = FPLHistoricalDataCollector()
        available_seasons = collector.get_available_seasons()
        
        # Determine which seasons to process
        if args.all:
            seasons_to_process = available_seasons
        elif args.seasons:
            seasons_to_process = args.seasons
            # Validate specified seasons
            for season in seasons_to_process:
                if season not in available_seasons:
                    print(f"Warning: Season {season} is not in the list of available seasons: {', '.join(available_seasons)}")
                    print("Proceeding anyway, but processing may fail.")
        else:
            # Default to just the latest season
            seasons_to_process = [collector.get_latest_season()]
        
        print(f"Processing data for seasons: {', '.join(seasons_to_process)}")
        
        # Initialize enhanced preprocessor
        preprocessor = FPLDataPreprocessor(
            data_dir="data",
            lookback=args.lookback,
            validation_split=0.2
        )
        
        # Load and process data from all selected seasons
        all_gameweek_data = []
        all_bootstrap_data = []
        
        for season in seasons_to_process:
            print(f"\nLoading data for season {season}...")
            season_data = collector.load_season_data(season)
            
            if not season_data:
                print(f"No data found for season {season}, skipping...")
                continue
            
            # Load bootstrap data for team strength features
            bootstrap_data = preprocessor.load_bootstrap_data(season)
            if bootstrap_data:
                all_bootstrap_data.append((season, bootstrap_data))
                print(f"Loaded bootstrap data for {season}")
            
            # Add gameweek data with season marker
            if "merged_gw" in season_data and not season_data["merged_gw"].empty:
                gw_data = season_data["merged_gw"].copy()
                gw_data['season'] = season
                
                # Standardize column names
                if 'name' not in gw_data.columns and 'player_name' in gw_data.columns:
                    gw_data['name'] = gw_data['player_name']
                if 'element' not in gw_data.columns and 'id' in gw_data.columns:
                    gw_data['element'] = gw_data['id']
                if 'round' not in gw_data.columns and 'GW' in gw_data.columns:
                    gw_data['round'] = gw_data['GW']
                
                all_gameweek_data.append(gw_data)
                print(f"Added {len(gw_data)} gameweek entries from {season}")
        
        if not all_gameweek_data:
            print("ERROR: No gameweek data found across seasons")
            return
        
        # Combine all gameweek data
        combined_gw = pd.concat(all_gameweek_data, ignore_index=True)
        print(f"Combined gameweek data: {len(combined_gw)} entries")
        
        # Process using enhanced preprocessor
        print("Running enhanced data preprocessing pipeline...")
        
        # Step 1: Calculate advanced rolling features
        enhanced_data = preprocessor.calculate_advanced_rolling_features(
            combined_gw, 
            group_col='element', 
            sort_col='round'
        )
        
        # Step 2: Add team strength features from bootstrap data
        if all_bootstrap_data:
            enhanced_data = preprocessor.add_team_strength_features(
                enhanced_data, 
                all_bootstrap_data
            )
        
        # Step 3: Add position-based features
        enhanced_data = preprocessor.add_position_features(enhanced_data, all_bootstrap_data)
        
        # Step 4: Calculate performance efficiency metrics
        enhanced_data = preprocessor.calculate_performance_efficiency(enhanced_data)
        
        # Step 5: Add consistency scoring
        enhanced_data = preprocessor.calculate_consistency_metrics(enhanced_data)
        
        # Step 6: Prepare model-ready sequences
        print("Generating model-ready training sequences...")
        
        X_train, X_val, y_train, y_val, feature_names, metadata = preprocessor.prepare_model_sequences(
            enhanced_data,
            target_col='total_points',
            player_col='element',
            gameweek_col='round',
            lookback=args.lookback,
            validation_split=0.2
        )
        
        print(f"Training sequences: {X_train.shape}")
        print(f"Validation sequences: {X_val.shape}")
        print(f"Feature count: {len(feature_names)}")
        
        # Step 7: Save processed data and metadata
        preprocessor.save_processed_data(
            X_train, X_val, y_train, y_val, 
            feature_names, metadata,
            filename_prefix="enhanced"
        )
        
        print(f"Enhanced data processing complete!")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Features per sequence: {len(feature_names)}")
        print(f"Lookback window: {args.lookback} gameweeks")
        
        # Save summary statistics
        summary = {
            'processing_date': datetime.now().isoformat(),
            'seasons_processed': seasons_to_process,
            'total_players': len(enhanced_data['element'].unique()),
            'total_gameweeks': len(enhanced_data),
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'feature_count': len(feature_names),
            'lookback_window': args.lookback,
            'feature_names': feature_names.tolist() if hasattr(feature_names, 'tolist') else list(feature_names)
        }
        
        with open(os.path.join("data", "processed", "processing_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("Processing summary saved to data/processed/processing_summary.json")
    
    elif args.action == 'train':
        # Train model
        print("Training CNN model...")
        train_model(epochs=args.epochs, batch_size=args.batch_size)
        
    elif args.action == 'predict':
        # Predict team
        print("Predicting optimal team...")
        
        # Check if we should use the historical model or standard model
        use_historical_model = args.use_history
        
        if use_historical_model:
            print("Using model trained with historical data")
            # Load the model trained with historical data
            from .models.cnn_model import FPLPredictionModel
            historical_model = FPLPredictionModel(lookback=args.lookback)
            try:
                historical_model.load(model_type="historical")
                print("Successfully loaded historical model")
            except Exception as e:
                print(f"Failed to load historical model: {e}")
                print("Falling back to standard model")
                use_historical_model = False
        
        predict_team_for_gameweek(
            gameweek=args.gameweek, 
            budget=args.budget, 
            lookback=args.lookback, 
            n_features=14,
            cutoff_gw=args.cutoff_gw, 
            apply_transfers=args.apply,
            use_historical_model=args.use_history, 
            next_season=args.next_season,
            next_season_teams=args.teams
        )
    
    elif args.action == 'transfers':
        # Suggest transfers
        print("Suggesting transfers for next gameweek...")
        predict_team_for_gameweek(args.gameweek, args.budget, args.lookback, n_features=14, apply_transfers=args.apply)
    
    elif args.action == 'view-team':
        # View current team
        from .utils.team_state_manager import TeamStateManager
        team_state = TeamStateManager().load_team_state()
        
        if team_state:
            gw = team_state.get("gameweek", 0)
            free_transfers = team_state.get("free_transfers", 1)
            team = team_state.get("team", {})
            
            print(f"Current team (Gameweek {gw}):")
            print(f"Free transfers available for next gameweek: {free_transfers}")
            
            if "squad" in team:
                print("\nSquad:")
                for position in ['GKP', 'DEF', 'MID', 'FWD']:
                    print(f"\n{position}:")
                    for player in team["squad"]:
                        if player["position"] == position:
                            captain_mark = "(C)" if player.get("is_captain") else "(V)" if player.get("is_vice_captain") else ""
                            print(f"  {player['name']} {captain_mark} - {player['team']} - £{player['cost']}m")
        else:
            print("No team state found. Run 'predict' command with --apply to create a team.")
    
    elif args.action == 'fetch-history':
        # Fetch historical data
        collector = FPLHistoricalDataCollector()
        
        if args.all:
            seasons = collector.get_available_seasons()
        elif args.seasons:
            seasons = args.seasons
        else:
            # Default to last completed season and current season
            seasons = collector.get_available_seasons()[-2:]
            
        print(f"Fetching historical data for seasons: {', '.join(seasons)}")
        collector.collect_all_seasons(seasons)
    
    elif args.action == 'train-with-history':
        # Train with historical data
        print("Training with historical data...")
        
        # Get available seasons
        collector = FPLHistoricalDataCollector()
        available_seasons = collector.get_available_seasons()
        
        # Determine which seasons to use for training
        if args.all:
            seasons = available_seasons
            print(f"Using all available seasons: {', '.join(seasons)}")
        elif args.seasons:
            seasons = args.seasons
            # Validate specified seasons
            for season in seasons:
                if season not in available_seasons:
                    print(f"Warning: Season {season} is not in the list of available seasons: {', '.join(available_seasons)}")
                    print("Proceeding anyway, but training may fail if data is missing.")
        else:
            # Default to using the two most recent seasons
            seasons = available_seasons[-2:] if len(available_seasons) >= 2 else available_seasons
            print(f"Using default seasons: {', '.join(seasons)}")
        
        train_model_with_history(seasons=seasons, epochs=args.epochs, batch_size=args.batch_size)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()