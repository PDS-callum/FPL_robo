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

def predict_team_for_gameweek(gameweek=None, budget=100.0, lookback=3, n_features=18, 
                             cutoff_gw=None, apply_transfers=False, use_historical_model=False,
                             next_season=False, next_season_teams=None):
    """
    Predict optimal team for a specific gameweek using fixture difficulty
    
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
    use_historical_model : bool
        Whether to use the model trained with historical data
    next_season : bool
        Whether to predict for the first gameweek of the next season
    next_season_teams : list
        List of team names that will be in the next season
        
    Returns:
    --------
    team_dict : dict
        Dictionary with selected team information
    """
    print(f"Predicting team for Gameweek {gameweek}...")
    if cutoff_gw:
        print(f"Using only data up to Gameweek {cutoff_gw}")
    
    # Initialize data processor - use MultiSeasonDataProcessor for historical data support
    from .utils.data_processing import MultiSeasonDataProcessor, FPLDataProcessor
    from .utils.history_data_collector import FPLHistoricalDataCollector
    
    # Get available historical seasons
    history_collector = FPLHistoricalDataCollector()
    available_seasons = history_collector.get_available_seasons()
    
    # When predicting for next season, ensure we're using the most recent season
    if next_season and gameweek == 1:
        historical_seasons = available_seasons[-1:] # Just use the most recent season
        print(f"Predicting for first gameweek of new season using data from {historical_seasons[0]}")
        use_historical_model = True  # Force use of historical model
        
        # Print teams being used for next season if provided
        if next_season_teams:
            print(f"Filtering players to only include these teams: {', '.join(next_season_teams)}")
    else:
        # Use the most recent seasons for context
        historical_seasons = available_seasons[-2:] 
        print(f"Using historical data from seasons: {', '.join(historical_seasons)}")
    
    # Initialize multi-season data processor
    data_processor = MultiSeasonDataProcessor(seasons=historical_seasons, lookback=lookback)
    
    # Also keep a standard processor for current season info
    current_processor = FPLDataProcessor(cutoff_gw=cutoff_gw)
    
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
    
    # Load player data from bootstrap_static
    bootstrap = current_processor.load_latest_data("bootstrap_static")
    players_raw = pd.DataFrame(bootstrap["elements"])
    
    # Filter players by team if next_season_teams is provided
    if next_season and next_season_teams:
        # Get team ID to name mapping from current season
        teams_df = pd.DataFrame(bootstrap["teams"])
        team_name_to_id = {name.lower(): id for id, name in zip(teams_df['id'], teams_df['name'])}
        original_team_names = {name.lower(): name for name in teams_df['name']}
        
        # Get historical team data for better name matching
        all_historical_teams = {}
        latest_season = history_collector.get_latest_season()
        
        # Fetch team data from the last 5 seasons to cover promoted/relegated teams
        seasons_to_check = history_collector.get_available_seasons()[-5:]
        
        print("Collecting historical team data for better team matching...")
        for season in seasons_to_check:
            season_teams_url = f"{history_collector.base_url}/{season}/teams.csv"
            try:
                historical_teams_df = pd.read_csv(season_teams_url)
                # Add to our dict of all historical teams
                for _, team_row in historical_teams_df.iterrows():
                    team_name = team_row['name'].lower()
                    if 'short_name' in team_row:
                        short_name = team_row['short_name'].lower()
                        all_historical_teams[short_name] = {
                            'name': team_row['name'],
                            'id': 1000 + len(all_historical_teams) if team_name not in team_name_to_id else team_name_to_id[team_name]
                        }
                    all_historical_teams[team_name] = {
                        'name': team_row['name'],
                        'id': 1000 + len(all_historical_teams) if team_name not in team_name_to_id else team_name_to_id[team_name]
                    }
                print(f"  - Loaded {len(historical_teams_df)} teams from {season}")
            except Exception as e:
                print(f"  - Failed to load teams from {season}: {e}")
        
        # Create list of team IDs to include
        included_team_ids = []
        unmatched_teams = []
        historical_team_matches = []
        
        # Process each specified team
        for specified_team in next_season_teams:
            specified_team_lower = specified_team.lower()
            matched = False
            
            # Try exact match in current season teams first
            if specified_team_lower in team_name_to_id:
                included_team_ids.append(team_name_to_id[specified_team_lower])
                matched = True
            
            # Try partial match in current season teams
            elif not matched:
                for team_name, team_id in team_name_to_id.items():
                    if specified_team_lower in team_name or team_name in specified_team_lower:
                        included_team_ids.append(team_id)
                        print(f"Matched '{specified_team}' to team '{original_team_names[team_name]}'")
                        matched = True
                        break
            
            # Try exact match in historical teams
            if not matched and specified_team_lower in all_historical_teams:
                historical_team = all_historical_teams[specified_team_lower]
                historical_team_matches.append(historical_team['name'])
                print(f"Matched '{specified_team}' to historical team '{historical_team['name']}'")
                matched = True
            
            # Try partial match in historical teams
            elif not matched:
                for hist_team_name, hist_team_data in all_historical_teams.items():
                    if specified_team_lower in hist_team_name or hist_team_name in specified_team_lower:
                        historical_team_matches.append(hist_team_data['name'])
                        print(f"Matched '{specified_team}' to historical team '{hist_team_data['name']}'")
                        matched = True
                        break
            
            if not matched:
                print(f"Warning: Could not match team '{specified_team}' to any known team")
                unmatched_teams.append(specified_team)
        
        # Display available teams if there were any unmatched teams
        if unmatched_teams:
            print("\nAvailable teams in the current dataset:")
            for i, team_name in enumerate(sorted(original_team_names.values()), 1):
                print(f"  {i}. {team_name}")
                
            print("\nAvailable historical teams:")
            unique_historical_teams = sorted(set(team_data['name'] for team_data in all_historical_teams.values()))
            for i, team_name in enumerate(unique_historical_teams, 1):
                if team_name.lower() not in {t.lower() for t in original_team_names.values()}:
                    print(f"  {i}. {team_name}")
                
            print("\nPlease use these exact team names or close variations when specifying teams.")
        
        # Filter players to only include those from specified teams
        filtered_players = players_raw[players_raw['team'].isin(included_team_ids)]
        
        # Check if we have any players left
        if len(filtered_players) == 0:
            print("Error: No players found from the specified teams. Using all players instead.")
            filtered_players = players_raw
        else:
            print(f"Filtered down to {len(filtered_players)} players from specified teams")
            players_raw = filtered_players
            
        if historical_team_matches:
            print(f"\nNote: The following historical teams were matched but have no players in the current dataset: "
                  f"{', '.join(historical_team_matches)}")
            print("These teams may be promoted for next season or historically relegated.")
    
    # Get current gameweek if not specified
    if gameweek is None:
        current_gw = next((gw for gw in bootstrap["events"] if gw["is_current"]), None)
        if current_gw is None:
            current_gw = next((gw for gw in bootstrap["events"] if gw["is_next"]), None)
        gameweek = current_gw["id"]
        print(f"Current gameweek is {gameweek}.")
    
    # Prepare player data - use current processor for updated player info
    players_df = current_processor.create_player_features()
    
    # Prepare data for prediction using historical data for training
    X, y, player_ids = data_processor.prepare_multi_season_training_data()
    
    # Get prediction data for current players
    # Create X_pred manually since we don't have player_history_features.csv
    X_pred = []
    pred_player_ids = []
    
    # Feature columns to match historical data
    feature_cols = [
        'minutes', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
        'bonus', 'bps', 'influence', 'creativity', 'threat', 'ict_index',
        'rolling_pts_3', 'rolling_mins_3', 'was_home'
    ]
    
    # Create recent form features for current players
    try:
        # Try to get player GW data from API
        gw_data = current_processor.load_latest_data("gw")
        
        # Convert to dataframe
        recent_gw_data = pd.DataFrame()
        for gw_file in gw_data[-lookback:]:  # Use recent gameweeks
            if isinstance(gw_file, list):
                gw_df = pd.DataFrame(gw_file)
                recent_gw_data = pd.concat([recent_gw_data, gw_df])
        
        # Group by player and calculate averages
        player_averages = recent_gw_data.groupby('element').agg({
            'minutes': 'mean',
            'goals_scored': 'mean', 
            'assists': 'mean',
            'clean_sheets': 'mean',
            'goals_conceded': 'mean',
            'bonus': 'mean',
            'bps': 'mean',
            'influence': 'mean',
            'creativity': 'mean',
            'threat': 'mean',
            'ict_index': 'mean',
            'total_points': 'mean',
        }).reset_index()
        
        # Add was_home placeholder (since we don't have fixtures data)
        player_averages['was_home'] = 0.5  # Neutral value for home/away
        
        # Rename total_points to rolling_pts_3
        player_averages['rolling_pts_3'] = player_averages['total_points']
        player_averages['rolling_mins_3'] = player_averages['minutes']
        
        # Normalize the features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        player_averages[feature_cols] = scaler.fit_transform(player_averages[feature_cols].fillna(0))
        
        # Create prediction inputs
        for _, player in player_averages.iterrows():
            player_id = player['element']
            
            # Create a sequence of lookback size with the same values
            # (since we're using averages, repeat them)
            features = np.tile(player[feature_cols].values, (lookback, 1))
            
            X_pred.append(features)
            pred_player_ids.append(player_id)
            
    except Exception as e:
        print(f"Error generating prediction features: {e}")
        print("Using basic features from player stats instead")
        
        # Fallback to using basic player stats
        for _, player in players_raw.iterrows():
            player_id = player['id']
            
            # Get minutes played and handle division safely
            player_minutes = player['minutes'] 
            appearances = player.get('appearances', 0) or 1  # Fallback to 1 if not present or 0
            
            # Calculate average minutes per appearance
            avg_minutes = player_minutes / appearances if appearances > 0 else player_minutes
            
            # Create basic features
            basic_features = np.zeros(len(feature_cols))
            basic_features[0] = avg_minutes  # minutes
            basic_features[11] = player['total_points'] / appearances  # rolling_pts_3
            basic_features[12] = avg_minutes  # rolling_mins_3
            # was_home is already 0 (neutral)
            
            # Repeat for lookback periods
            features = np.tile(basic_features, (lookback, 1))
            
            X_pred.append(features)
            pred_player_ids.append(player_id)
    
    # Convert to numpy arrays
    X_pred = np.array(X_pred) if X_pred else np.empty((0, lookback, len(feature_cols)))
    pred_player_ids = np.array(pred_player_ids)
    
    # Load the trained model
    model = FPLPredictionModel(lookback=lookback, n_features=n_features)
    
    try:
        if use_historical_model:
            model.load(model_type="historical")
            print("Using model trained with historical data for predictions")
        else:
            model.load()
            print("Using standard model for predictions")
    except:
        print("No saved model found. Training a new model...")
        if use_historical_model:
            from .train_model import train_model_with_history
            train_model_with_history()
            model.load(model_type="historical")
        else:
            from .train_model import train_model
            train_model()
            model.load()
    
    # Make predictions if we have prediction data
    if len(X_pred) > 0:
        predictions = model.predict(X_pred).flatten()
        
        # Create predictions dataframe
        predictions_df = pd.DataFrame({
            'id': pred_player_ids,
            'predicted_points': predictions
        })
    else:
        # Fallback if we couldn't create prediction features
        print("Warning: No prediction data available, using current form as predicted points")
        predictions_df = pd.DataFrame({
            'id': players_raw['id'],
            'predicted_points': players_raw['form'].astype(float) * 2  # Simple heuristic
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