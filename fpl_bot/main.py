import os
import argparse
from fpl_bot.fpl_data_collector import FPLDataCollector
from fpl_bot.utils.data_processing import FPLDataProcessor
from fpl_bot.train_model import train_model
from fpl_bot.predict_team import predict_team_for_gameweek

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
    
    # Run all steps parser
    all_parser = subparsers.add_parser('all', help='Run all steps (collect, process, train, predict)')
    all_parser.add_argument('--gameweek', type=int, help='Gameweek number to predict for')
    all_parser.add_argument('--budget', type=float, default=100.0, help='Available budget (default: 100.0)')
    all_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (default: 50)')
    all_parser.add_argument('--lookback', type=int, default=3, help='Number of gameweeks to look back (default: 3)')
    
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
        predict_team_for_gameweek(args.gameweek, args.budget, args.lookback)
        
    elif args.action == 'all':
        # Run all steps
        print("Running full pipeline...")
        
        # Collect data
        collector = FPLDataCollector()
        collector.collect_all_data()
        
        # Process data
        processor = FPLDataProcessor(args.lookback)
        processor.process_all_data()
        
        # Train model
        train_model(epochs=args.epochs)
        
        # Predict team
        predict_team_for_gameweek(args.gameweek, args.budget, args.lookback)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()