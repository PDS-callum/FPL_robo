import os
import argparse
from .fpl_data_collector import FPLDataCollector
from .utils.history_data_collector import FPLHistoricalDataCollector
from .utils.data_processing import FPLDataProcessor
from .train_model import train_model, train_model_with_history
from .predict_team import predict_team_for_gameweek

def main():
    """Main entry point for FPL Bot"""
    parser = argparse.ArgumentParser(description='FPL Bot with CNN modeling')
    
    # Create subparsers for different actions
    subparsers = parser.add_subparsers(dest='action', help='Action to perform')
    
    # Collect data parser
    collect_parser = subparsers.add_parser('collect', help='Collect FPL data')
    
    # Process data parser
    process_parser = subparsers.add_parser('process', help='Process collected data')
    process_parser.add_argument('--lookback', type=int, default=3, help='Number of gameweeks to look back (default: 3)')
    
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
    train_history_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (default: 50)')
    train_history_parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.action == 'collect':
        # Collect data
        print("Collecting FPL data...")
        collector = FPLDataCollector()
        collector.collect_all_data()
        
    elif args.action == 'process':
        # Process data
        print("Processing FPL data...")
        processor = FPLDataProcessor(lookback=args.lookback)
        processor.process_all_data()
        
    elif args.action == 'train':
        # Train model
        print("Training CNN model...")
        train_model(epochs=args.epochs, batch_size=args.batch_size)
        
    elif args.action == 'predict':
        # Predict team
        print("Predicting optimal team...")
        predict_team_for_gameweek(args.gameweek, args.budget, args.lookback, apply_transfers=args.apply)
    
    elif args.action == 'transfers':
        # Suggest transfers
        print("Suggesting transfers for next gameweek...")
        predict_team_for_gameweek(args.gameweek, args.budget, args.lookback, apply_transfers=args.apply)
    
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
        if args.seasons:
            seasons = args.seasons
        else:
            # Default to using the two most recent seasons
            collector = FPLHistoricalDataCollector()
            seasons = collector.get_available_seasons()[-2:]
            
        print(f"Training with historical data from seasons: {', '.join(seasons)}")
        train_model_with_history(seasons=seasons, epochs=args.epochs, batch_size=args.batch_size)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()