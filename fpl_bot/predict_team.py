import os
import pandas as pd
import numpy as np
from .models.cnn_model import FPLPredictionModel
from .utils.data_processing import FPLDataProcessor
from .utils.team_optimizer import FPLTeamOptimizer
from .utils.general_utils import get_data
from .utils.team_state_manager import TeamStateManager
import json
from datetime import datetime

def predict_team_for_gameweek(gameweek, budget=100.0, lookback=3, n_features=14, cutoff_gw=None, apply_transfers=False):
    """
    Predict optimal team for a specific gameweek
    
    Parameters:
    -----------
    gameweek : int
        Gameweek number to predict for
    budget : float
        Total budget available (default: 100.0)
    lookback : int
        Number of gameweeks to use for input features
    n_features : int
        Number of features for the model
    cutoff_gw : int
        If provided, only use data up to this gameweek for training and prediction
    apply_transfers : bool
        Whether to apply the suggested transfers to the team state
        
    Returns:
    --------
    team_dict : dict
        Dictionary with selected team information
    """
    print(f"Predicting team for Gameweek {gameweek}...")
    if cutoff_gw:
        print(f"Using only data up to Gameweek {cutoff_gw}")
    
    # Initialize data processor
    data_processor = FPLDataProcessor(cutoff_gw=cutoff_gw)
    
    # Initialize team state manager
    team_state_manager = TeamStateManager()
    
    # Load previous team state if it exists
    previous_state = team_state_manager.load_team_state()
    current_team_ids = []
    free_transfers = 1
    
    if previous_state:
        prev_gameweek = previous_state.get("gameweek", 0)
        free_transfers = previous_state.get("free_transfers", 1)
        
        # Check if we're predicting for the next gameweek
        if gameweek == prev_gameweek + 1:
            current_team_ids = [player["id"] for player in previous_state["team"]["squad"]]
            print(f"Loaded previous team from gameweek {prev_gameweek}")
            print(f"Available free transfers: {free_transfers}")
        else:
            print(f"Warning: Previous state is for gameweek {prev_gameweek}, not {gameweek-1}.")
            print("Will create a new team from scratch.")
    else:
        print("No previous team state found. Creating a new team.")
    
    # Load latest bootstrap static data for players
    bootstrap = data_processor.load_latest_data("bootstrap_static")
    players_raw = pd.DataFrame(bootstrap["elements"])
    
    # Get current gameweek if not specified
    if gameweek is None:
        current_gw = next((gw for gw in bootstrap["events"] if gw["is_current"]), None)
        if current_gw is None:
            current_gw = next((gw for gw in bootstrap["events"] if gw["is_next"]), None)
        gameweek = current_gw["id"]
        print(f"Current gameweek is {gameweek}.")
    
    # Prepare player data
    players_df = data_processor.create_player_features()
    
    # Prepare data for prediction
    X, y, player_ids, X_pred, pred_player_ids = data_processor.prepare_training_data(
        lookback=lookback, prediction_gw=gameweek
    )
    
    # Load the trained model
    model = FPLPredictionModel(lookback=lookback, n_features=n_features)
    try:
        model.load()
    except:
        print("No saved model found. Training a new model...")
        from .train_model import train_model
        train_model()
        model.load()
    
    # Make predictions
    predictions = model.predict(X_pred).flatten()
    
    # Create predictions dataframe
    predictions_df = pd.DataFrame({
        'id': pred_player_ids,
        'predicted_points': predictions
    })
    
    # Filter out injured players and those not expected to play
    availability_df = players_raw[['id', 'chance_of_playing_next_round', 'status']].copy()
    availability_df['chance_of_playing_next_round'] = availability_df['chance_of_playing_next_round'].fillna(100)
    
    # Filter out players with low chance of playing
    available_players = availability_df[availability_df['chance_of_playing_next_round'] > 50]
    predictions_df = predictions_df[predictions_df['id'].isin(available_players['id'])]
    
    # Create team optimizer with current team constraints
    optimizer = FPLTeamOptimizer(
        total_budget=budget,
        current_team_ids=current_team_ids,
        free_transfers=free_transfers
    )
    
    # Convert player_id to numeric in players_df if needed
    players_df['id'] = players_df['id'].astype(int)
    
    # Optimize team selection with transfers
    if current_team_ids:
        selected_team, transfers, transfer_cost = optimizer.optimize_transfers(players_df, predictions_df)
    else:
        # Build initial team from scratch
        selected_team = optimizer.optimize_team(players_df, predictions_df)
        transfers = []
        transfer_cost = 0
    
    # Select playing XI
    playing_xi, captain, vice_captain, formation = optimizer.select_playing_xi(selected_team)
    
    # Create directories for saving results
    os.makedirs('data/predictions', exist_ok=True)
    
    # Save predicted team
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    selected_team.to_csv(f'data/predictions/team_gw{gameweek}_{timestamp}.csv', index=False)
    playing_xi.to_csv(f'data/predictions/playing_xi_gw{gameweek}_{timestamp}.csv', index=False)
    
    # Create a dictionary with team information
    team_dict = {
        'gameweek': gameweek,
        'timestamp': timestamp,
        'total_predicted_points': float(playing_xi['predicted_points'].sum()) - transfer_cost,
        'formation': f"{formation['DEF']}-{formation['MID']}-{formation['FWD']}",
        'total_cost': float(selected_team['now_cost'].sum() / 10),
        'captain': {
            'name': captain['web_name'],
            'team': captain['team_name'],
            'position': captain['position'],
            'predicted_points': float(captain['predicted_points'])
        },
        'vice_captain': {
            'name': vice_captain['web_name'],
            'team': vice_captain['team_name'],
            'position': vice_captain['position'],
            'predicted_points': float(vice_captain['predicted_points'])
        },
        'transfers': {
            'count': len(transfers),
            'free_transfers': free_transfers,
            'cost': transfer_cost,
            'details': [{
                'out': {
                    'name': out['web_name'],
                    'team': out['team_name'],
                    'position': out['position'],
                    'cost': float(out['now_cost'] / 10)
                },
                'in': {
                    'name': incoming['web_name'], 
                    'team': incoming['team_name'],
                    'position': incoming['position'],
                    'cost': float(incoming['now_cost'] / 10),
                    'predicted_points': float(incoming['predicted_points'])
                }
            } for out, incoming in transfers]
        },
        'squad': []
    }
    
    # Add squad information
    for _, player in selected_team.iterrows():
        is_in_xi = player.name in playing_xi.index
        team_dict['squad'].append({
            'id': int(player['id']),
            'name': player['web_name'],
            'team': player['team_name'],
            'position': player['position'],
            'cost': float(player['now_cost'] / 10),
            'predicted_points': float(player['predicted_points']),
            'in_starting_xi': is_in_xi,
            'is_captain': bool(player.name in playing_xi.index and playing_xi.loc[player.name, 'is_captain']),
            'is_vice_captain': bool(player.name in playing_xi.index and playing_xi.loc[player.name, 'is_vice_captain']),
        })
    
    def convert_numpy_to_python(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        import numpy as np
        if isinstance(obj, dict):
            return {k: convert_numpy_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_to_python(i) for i in obj]
        elif isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    # Convert team_dict to use only Python native types
    team_dict = convert_numpy_to_python(team_dict)

    # Save as JSON
    with open(f'data/predictions/team_info_gw{gameweek}_{timestamp}.json', 'w') as f:
        json.dump(team_dict, f, indent=2)
    
    # If we want to apply transfers, save the team state
    if apply_transfers:
        team_state_manager.save_team_state(team_dict, gameweek)
        print("Applied transfers and updated team state.")
    
    # Print results
    print(f"Team prediction for Gameweek {gameweek} complete!")
    print(f"Formation: {formation['DEF']}-{formation['MID']}-{formation['FWD']}")
    print(f"Captain: {captain['web_name']} ({captain['team_name']}) - Predicted points: {captain['predicted_points']:.2f}")
    print(f"Vice-captain: {vice_captain['web_name']} ({vice_captain['team_name']})")
    
    if transfers:
        print(f"\nSuggested Transfers ({len(transfers)} of {free_transfers} free):")
        for out, incoming in transfers:
            print(f"  OUT: {out['web_name']} ({out['team_name']}, {out['position']})")
            print(f"  IN:  {incoming['web_name']} ({incoming['team_name']}, {incoming['position']}) - Predicted points: {incoming['predicted_points']:.2f}")
        
        if len(transfers) > free_transfers:
            print(f"Transfer cost: -{transfer_cost} points")
    
    print(f"\nTotal predicted points: {playing_xi['predicted_points'].sum():.2f}" + 
          (f" - {transfer_cost} = {playing_xi['predicted_points'].sum() - transfer_cost:.2f}" if transfer_cost else ""))
    print(f"Total cost: £{team_dict['total_cost']:.1f}m")
    
    # Print the rest of the team details
    print("\n----- STARTING XI -----")
    for position in ['GKP', 'DEF', 'MID', 'FWD']:
        print(f"\n{position}:")
        for player in team_dict['squad']:
            if player['position'] == position and player['in_starting_xi']:
                captain_mark = "(C)" if player['is_captain'] else "(V)" if player['is_vice_captain'] else ""
                print(f"  {player['name']} {captain_mark} - {player['team']} - £{player['cost']}m - {player['predicted_points']:.2f} pts")
    
    print("\n----- BENCH -----")
    for player in team_dict['squad']:
        if not player['in_starting_xi']:
            print(f"  {player['name']} - {player['team']} - {player['position']} - £{player['cost']}m - {player['predicted_points']:.2f} pts")
    
    return team_dict

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict optimal FPL team for a gameweek')
    parser.add_argument('--gameweek', type=int, help='Gameweek number to predict for')
    parser.add_argument('--budget', type=float, default=100.0, help='Available budget (default: 100.0)')
    parser.add_argument('--lookback', type=int, default=3, help='Lookback period for training data (default: 3)')
    parser.add_argument('--n_features', type=int, default=14, help='Number of features for the model (default: 14)')
    parser.add_argument('--cutoff', type=int, help='Only use data up to this gameweek (for historical testing)')
    
    args = parser.parse_args()
    
    predict_team_for_gameweek(args.gameweek, args.budget, args.lookback, args.n_features, args.cutoff)