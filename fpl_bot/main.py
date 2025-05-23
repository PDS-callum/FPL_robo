import os
import argparse
import pandas as pd
import numpy as np
# Remove FPLDataCollector since we're only using GitHub data
from .utils.history_data_collector import FPLHistoricalDataCollector
from .utils.data_processing import FPLDataProcessor, MultiSeasonDataProcessor
from .train_model import train_model, train_model_with_history
from .predict_team import predict_team_for_gameweek
from .utils.data_conversion import create_api_compatible_data
from .utils.data_preparation import create_directories, calculate_rolling_features, normalize_features
from .utils.data_preparation import create_player_team_features, create_fixture_difficulty_features
from .utils.data_preparation import prepare_training_sequences, save_processed_data

def main():
    """Main entry point for FPL Bot"""
    parser = argparse.ArgumentParser(description='FPL Bot with CNN modeling')
    
    # Create subparsers for different actions
    subparsers = parser.add_subparsers(dest='action', help='Action to perform')
    
    # Collect data parser
    collect_parser = subparsers.add_parser('collect', help='Collect FPL data')
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
        # Process data with improved error handling and feature generation
        print("Processing FPL data...")
        
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
        
        # Process each season
        if len(seasons_to_process) > 1:
            print("Using multi-season data processor for enhanced feature generation...")
            
            # Load and combine data from all selected seasons
            all_players_data = []
            all_gw_data = []
            all_teams_data = []
            all_fixtures_data = []
            
            for season in seasons_to_process:
                print(f"\nProcessing season {season}...")
                season_data = collector.load_season_data(season)
                
                if not season_data:
                    print(f"No data found for season {season}, skipping...")
                    continue
                
                # Add season marker to all data
                if "players" in season_data and not season_data["players"].empty:
                    players = season_data["players"].copy()
                    players['season'] = season
                    all_players_data.append(players)
                    print(f"Added {len(players)} players from {season}")
                
                if "merged_gw" in season_data and not season_data["merged_gw"].empty:
                    gw_data = season_data["merged_gw"].copy()
                    gw_data['season'] = season
                    all_gw_data.append(gw_data)
                    print(f"Added {len(gw_data)} gameweek entries from {season}")
                
                if "teams" in season_data and not season_data["teams"].empty:
                    teams = season_data["teams"].copy()
                    teams['season'] = season
                    all_teams_data.append(teams)
                    print(f"Added {len(teams)} teams from {season}")
                
                if "fixtures" in season_data and not season_data["fixtures"].empty:
                    fixtures = season_data["fixtures"].copy()
                    fixtures['season'] = season
                    all_fixtures_data.append(fixtures)
                    print(f"Added {len(fixtures)} fixtures from {season}")
            
            # Combine all data
            if all_players_data:
                combined_players = pd.concat(all_players_data, ignore_index=True)
                print(f"Combined player data: {len(combined_players)} entries")
                save_processed_data(combined_players, "all_players")
            else:
                combined_players = pd.DataFrame()
                print("Warning: No player data found across seasons")
            
            if all_gw_data:
                combined_gw = pd.concat(all_gw_data, ignore_index=True)
                print(f"Combined gameweek data: {len(combined_gw)} entries")
                
                # Make sure column names are consistent
                if 'name' not in combined_gw.columns and 'player_name' in combined_gw.columns:
                    combined_gw['name'] = combined_gw['player_name']
                
                if 'element' not in combined_gw.columns and 'id' in combined_gw.columns:
                    combined_gw['element'] = combined_gw['id']
                
                if 'round' not in combined_gw.columns and 'GW' in combined_gw.columns:
                    combined_gw['round'] = combined_gw['GW']
                
                # Generate enhanced features for player performance
                value_cols = ['total_points', 'minutes', 'goals_scored', 'assists', 
                             'clean_sheets', 'goals_conceded', 'bonus', 'bps']
                
                print("Calculating rolling performance metrics...")
                enhanced_gw = calculate_rolling_features(
                    combined_gw, 
                    group_col='element',
                    sort_col='round',
                    value_cols=value_cols,
                    window_sizes=[3, 5, 10]
                )
                print(f"Enhanced gameweek data with {len(enhanced_gw.columns) - len(combined_gw.columns)} new features")
                
                # Normalize the features
                feature_cols = [col for col in enhanced_gw.columns 
                               if any(col.startswith(f"{val}_rolling") for val in value_cols)]
                
                print("Normalizing features...")
                normalized_gw, _ = normalize_features(enhanced_gw, feature_cols)
                
                # Save processed data
                save_processed_data(normalized_gw, "enhanced_gameweek_data")
                
                # Prepare training sequences
                print("Preparing training sequences...")
                train_features = feature_cols + ['minutes', 'was_home', 'team']
                
                X, y, player_ids = prepare_training_sequences(
                    normalized_gw,
                    player_col='element',
                    gameweek_col='round',
                    feature_cols=train_features,
                    target_col='total_points',
                    lookback=args.lookback
                )
                
                print(f"Created {len(X)} training sequences with {len(train_features)} features each")
                
                # Save the training data
                np.save(os.path.join("data", "processed", "X_train.npy"), X)
                np.save(os.path.join("data", "processed", "y_train.npy"), y)
                
                # Save player IDs mapping
                pd.DataFrame({'player_id': player_ids}).to_csv(
                    os.path.join("data", "processed", "player_ids.csv"),
                    index=False
                )
                
                print(f"Multi-season data processing complete! Created {len(X)} training samples.")
            
            else:
                print("Warning: No gameweek data found across seasons")
        
        else:
            # Single season processing
            print("Using single-season data processor...")
            processor = FPLDataProcessor(lookback=args.lookback)
            
            try:
                print("Processing player features...")
                players_df = processor.create_player_features()
                
                print("Processing fixtures data...")
                fixtures_df = processor.create_fixtures_features()
                
                print("Processing player history...")
                player_history_df = processor.create_player_history_features()
                
                print("Creating position encoding...")
                players_with_pos = processor.create_player_position_encoding()
                
                print("Preparing training data...")
                X, y, player_ids = processor.prepare_training_data(lookback=args.lookback)
                
                print(f"Single-season data processing complete! Created {len(X)} training samples.")
            
            except Exception as e:
                print(f"Error during single-season processing: {e}")
                print("Consider using multi-season processing with the --all flag for more robust data")
    
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