import os
import pandas as pd
import numpy as np
from .models.fpl_model import FPLPredictionModel
from .utils.team_optimizer import FPLTeamOptimizer
from .utils.current_season_collector import FPLCurrentSeasonCollector
import json
from datetime import datetime

def predict_team_for_gameweek(gameweek=None, budget=100.0, target='points_scored', 
                             data_dir="data", save_results=True):
    """
    Predict optimal FPL team for a specific gameweek
    
    Parameters:
    -----------
    gameweek : int, optional
        Gameweek to predict for (if None, predicts for next gameweek)
    budget : float
        Available budget in millions
    target : str
        Target variable model to use for predictions
    data_dir : str
        Data directory path
    save_results : bool
        Whether to save prediction results
        
    Returns:
    --------
    prediction_results : dict
        Complete prediction results including team, formation, etc.
    """
    print("="*60)
    print("FPL TEAM PREDICTION")
    print("="*60)
    print(f"Budget: £{budget}m")
    print(f"Target Model: {target}")
    
    # Step 1: Get current season data and determine gameweek
    print("\n📡 Getting current season data...")
    current_collector = FPLCurrentSeasonCollector(data_dir=data_dir)
    
    if gameweek is None:
        current_gw, bootstrap = current_collector.get_current_gameweek()
        if current_gw:
            # Predict for next gameweek
            gameweek = current_gw['id'] + 1
            print(f"📅 Predicting for gameweek {gameweek} (next gameweek)")
        else:
            raise ValueError("Could not determine current gameweek")
    else:
        bootstrap = current_collector.get_bootstrap_static()
        print(f"📅 Predicting for gameweek {gameweek}")
    
    # Step 2: Load trained model
    print(f"\n🤖 Loading trained model for {target}...")
    try:
        # Try to load scaler first to get correct feature count
        import pickle
        scaler_path = os.path.join(data_dir, 'processed', 'scalers.pkl')
        scaler = None
        scalers = {}
        expected_features = 46  # Default
        
        try:
            with open(scaler_path, 'rb') as f:
                scalers = pickle.load(f)
            scaler = scalers.get(f'{target}_scaler')
            expected_features = getattr(scaler, 'n_features_in_', 46) if scaler else 46
            print("✅ Scaler loaded successfully")
        except Exception as scaler_error:
            print(f"⚠️  Could not load scaler: {scaler_error}")
            print("⚠️  Will create features without scaling (less accurate)")
            scaler = None
        
        # Load model with correct feature count
        model = FPLPredictionModel(n_features=expected_features)
        model.load(f'fpl_model_{target}.h5')
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("🔄 Please train a model first using the train command")
        return None
    
    # Step 3: Prepare player data for prediction
    print("\n⚙️  Preparing player data...")
    
    # Get current players from bootstrap
    players_df = pd.DataFrame(bootstrap['elements'])
    teams_df = pd.DataFrame(bootstrap['teams'])
    
    # Load feature names if available
    feature_names = None
    try:
        feature_names_path = os.path.join(data_dir, 'features', f'feature_names_{target}.json')
        with open(feature_names_path, 'r') as f:
            feature_names = json.load(f)
    except Exception as e:
        print(f"⚠️  Could not load feature names: {e}")
        feature_names = None
    
    # Step 4: Create features for current players
    print("🔧 Creating player features...")
    
    # Create features that match training data as closely as possible
    player_features = []
    
    for _, player in players_df.iterrows():
        # Basic features from FPL API (map to training feature names)
        features = {
            'position': player.get('element_type', 0),
            'team': player.get('team', 0),
            'price': player.get('now_cost', 0) / 10 if player.get('now_cost') else 0,
            'selected_by_percent': float(player.get('selected_by_percent', 0)),
            'minutes_played': player.get('minutes', 0),
            'goals_scored': player.get('goals_scored', 0),
            'assists': player.get('assists', 0),
            'clean_sheets': player.get('clean_sheets', 0),
            'goals_conceded': player.get('goals_conceded', 0),
            'yellow_cards': player.get('yellow_cards', 0),
            'red_cards': player.get('red_cards', 0),
            'saves': player.get('saves', 0),
            'bonus': player.get('bonus', 0),
            'bps': player.get('bps', 0),
            'influence': float(player.get('influence', 0)),
            'creativity': float(player.get('creativity', 0)),
            'threat': float(player.get('threat', 0)),
        }
        
        # Use season totals as proxy for rolling averages
        total_points = player.get('total_points', 0)
        total_minutes = player.get('minutes', 0)
        form_val = float(player.get('form', 0))
        
        # Create approximations for missing historical features
        features.update({
            'avg_points_3gw': form_val,  # Form is close to recent average
            'avg_minutes_3gw': total_minutes / max(1, 38),  # Rough season average
            'avg_goals_3gw': player.get('goals_scored', 0) / max(1, 38),
            'avg_assists_3gw': player.get('assists', 0) / max(1, 38),
            'form_3gw': form_val,
            'avg_points_5gw': form_val,
            'avg_minutes_5gw': total_minutes / max(1, 38),
            'avg_goals_5gw': player.get('goals_scored', 0) / max(1, 38),
            'avg_assists_5gw': player.get('assists', 0) / max(1, 38),
            'form_5gw': form_val,
            'total_points_season': total_points,
            'total_minutes_season': total_minutes,
            'total_goals_season': player.get('goals_scored', 0),
            'total_assists_season': player.get('assists', 0),
            'games_played_season': max(1, total_minutes // 90),
            'avg_points_per_game': float(player.get('points_per_game', 0)),
            'points_std': 0,  # Not available from API
            'minutes_consistency': 0,  # Not available from API
        })
        
        # Add team strength information
        team_info = teams_df[teams_df['id'] == player['team']]
        if len(team_info) > 0:
            team_info = team_info.iloc[0]
            features.update({
                'strength_overall_home': team_info.get('strength_overall_home', 1000),
                'strength_overall_away': team_info.get('strength_overall_away', 1000),
                'strength_attack_home': team_info.get('strength_attack_home', 1000),
                'strength_attack_away': team_info.get('strength_attack_away', 1000),
                'strength_defence_home': team_info.get('strength_defence_home', 1000),
                'strength_defence_away': team_info.get('strength_defence_away', 1000),
            })
        else:
            features.update({
                'strength_overall_home': 1000,
                'strength_overall_away': 1000,
                'strength_attack_home': 1000,
                'strength_attack_away': 1000,
                'strength_defence_home': 1000,
                'strength_defence_away': 1000,
            })
        
        # Fixture difficulty (simplified - would need current fixtures for accuracy)
        features.update({
            'avg_home_difficulty': 3,  # Average difficulty
            'avg_away_difficulty': 3,  # Average difficulty  
            'home_difficulty': 3,
            'away_difficulty': 3,
            'is_home': 1,  # Assume home for prediction
        })
        
        player_features.append(features)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(player_features)
    features_df['id'] = players_df['id']
    
    # Step 5: Make predictions
    print("🔮 Making predictions...")
    
    # Prepare features for model (match training feature set exactly)
    if scaler and feature_names and hasattr(scaler, 'n_features_in_'):
        expected_features = scaler.n_features_in_
        # Use only the first N features that match the scaler
        actual_feature_names = feature_names[:expected_features]
        
        # Create feature matrix matching scaler exactly
        X_pred = np.zeros((len(features_df), expected_features))
        for i, feature_name in enumerate(actual_feature_names):
            if feature_name in features_df.columns:
                X_pred[:, i] = features_df[feature_name].fillna(0)
            else:
                X_pred[:, i] = 0
        
        print(f"✅ Created feature matrix: {X_pred.shape} matching scaler features ({expected_features})")
        X_pred = scaler.transform(X_pred)
        print("✅ Applied feature scaling")
        
    elif feature_names:
        # Create feature matrix matching training exactly
        features_to_use = feature_names[:expected_features] if expected_features <= len(feature_names) else feature_names
        X_pred = np.zeros((len(features_df), len(features_to_use)))
        for i, feature_name in enumerate(features_to_use):
            if feature_name in features_df.columns:
                X_pred[:, i] = features_df[feature_name].fillna(0)
            else:
                X_pred[:, i] = 0
        
        print(f"✅ Created feature matrix: {X_pred.shape} matching training features")
        if scaler:
            try:
                X_pred = scaler.transform(X_pred)
                print("✅ Applied feature scaling")
            except Exception as e:
                print(f"⚠️  Scaling failed: {e}, continuing without scaling")
    else:
        # Fallback: use available numeric features
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        # Limit to expected features count
        numeric_cols = numeric_cols[:expected_features]
        X_pred = features_df[numeric_cols].fillna(0).values
        
        # Pad or truncate to expected features
        if X_pred.shape[1] < expected_features:
            padding = np.zeros((X_pred.shape[0], expected_features - X_pred.shape[1]))
            X_pred = np.concatenate([X_pred, padding], axis=1)
        elif X_pred.shape[1] > expected_features:
            X_pred = X_pred[:, :expected_features]
            
        print(f"⚠️  Using fallback features: {X_pred.shape}")
        if scaler:
            try:
                X_pred = scaler.transform(X_pred)
                print("✅ Applied feature scaling")
            except Exception as e:
                print(f"⚠️  Scaling failed: {e}, continuing without scaling")
    
    # Make predictions
    predictions = model.predict(X_pred).flatten()
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        'id': players_df['id'],
        'predicted_points': predictions
    })
    
    print(f"✅ Generated predictions for {len(predictions_df)} players")
    
    # Step 6: Optimize team selection
    print("\n🎯 Optimizing team selection...")
    
    # Filter out injured/unavailable players
    available_players = players_df[
        (players_df['chance_of_playing_next_round'] > 50) |
        (players_df['chance_of_playing_next_round'].isna())
    ].copy()
    
    # Add team names for display
    team_names = {team['id']: team['name'] for _, team in teams_df.iterrows()}
    available_players['team_name'] = available_players['team'].map(team_names)
    
    # Debug: Check dataframe columns and content
    print(f"📊 Available players: {len(available_players)}")
    print(f"📊 Available columns: {list(available_players.columns)}")
    print(f"📊 Predictions columns: {list(predictions_df.columns)}")
    
    # Use team optimizer
    optimizer = FPLTeamOptimizer(total_budget=budget)
    selected_team = optimizer.optimize_team(available_players, predictions_df, budget=budget)
    
    if len(selected_team) < 15:
        print(f"⚠️  Could only select {len(selected_team)} players (need 15)")
        print("This might be due to budget constraints or data issues")
        
        # Try with higher budget as fallback
        if budget < 105:
            print(f"🔄 Retrying with budget of £105m...")
            selected_team = optimizer.optimize_team(available_players, predictions_df, budget=105.0)
    
    # Validate team
    is_valid, errors = optimizer.validate_team(selected_team)
    if not is_valid:
        print("⚠️  Team validation errors:")
        for error in errors:
            print(f"  - {error}")
    
    # Step 7: Select playing XI
    print("⚡ Selecting playing XI...")
    playing_xi, captain, vice_captain, formation = optimizer.select_playing_xi(selected_team)
    
    # Step 8: Prepare results
    total_cost = selected_team['now_cost'].sum() / 10 if 'now_cost' in selected_team.columns else selected_team['price'].sum()
    total_predicted_points = playing_xi['predicted_points'].sum()
    
    prediction_results = {
        'gameweek': gameweek,
        'target_model': target,
        'budget_used': budget,
        'total_cost': total_cost,
        'remaining_budget': budget - total_cost,
        'total_predicted_points': total_predicted_points,
        'formation': f"{formation['DEF']}-{formation['MID']}-{formation['FWD']}",
        'captain': {
            'name': captain['web_name'],
            'team': captain.get('team_name', 'Unknown'),
            'position': captain.get('element_type', 0),
            'predicted_points': float(captain['predicted_points']),
            'cost': captain['now_cost'] / 10
        },
        'vice_captain': {
            'name': vice_captain['web_name'],
            'team': vice_captain.get('team_name', 'Unknown'),
            'position': vice_captain.get('element_type', 0),
            'predicted_points': float(vice_captain['predicted_points']),
            'cost': vice_captain['now_cost'] / 10
        },
        'playing_xi': [],
        'bench': [],
        'team_validation': {
            'is_valid': is_valid,
            'errors': errors
        },
        'prediction_time': datetime.now().isoformat()
    }
    
    # Add playing XI details
    position_names = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    for _, player in playing_xi.iterrows():
        player_info = {
            'name': player['web_name'],
            'team': player.get('team_name', 'Unknown'),
            'position': position_names.get(player.get('element_type', 0), 'Unknown'),
            'predicted_points': float(player['predicted_points']),
            'cost': player['now_cost'] / 10,
            'is_captain': player.get('is_captain', False),
            'is_vice_captain': player.get('is_vice_captain', False)
        }
        prediction_results['playing_xi'].append(player_info)
    
    # Add bench details
    bench_players = selected_team[~selected_team['id'].isin(playing_xi['id'])]
    for _, player in bench_players.iterrows():
        player_info = {
            'name': player['web_name'],
            'team': player.get('team_name', 'Unknown'),
            'position': position_names.get(player.get('element_type', 0), 'Unknown'),
            'predicted_points': float(player['predicted_points']),
            'cost': player['now_cost'] / 10
        }
        prediction_results['bench'].append(player_info)
    
    # Step 9: Display results
    print("\n" + "="*60)
    print("TEAM PREDICTION COMPLETE!")
    print("="*60)
    print(f"🎯 Gameweek: {gameweek}")
    print(f"💰 Total Cost: £{total_cost:.1f}m (£{budget - total_cost:.1f}m remaining)")
    print(f"📊 Total Predicted Points: {total_predicted_points:.1f}")
    print(f"⚽ Formation: {prediction_results['formation']}")
    print(f"👑 Captain: {captain['web_name']} ({captain.get('team_name', 'Unknown')}) - {captain['predicted_points']:.1f} pts")
    print(f"🔸 Vice-Captain: {vice_captain['web_name']} ({vice_captain.get('team_name', 'Unknown')}) - {vice_captain['predicted_points']:.1f} pts")
    
    print("\n----- STARTING XI -----")
    for position in ['GKP', 'DEF', 'MID', 'FWD']:
        position_players = [p for p in prediction_results['playing_xi'] if p['position'] == position]
        if position_players:
            print(f"\n{position}:")
            for player in sorted(position_players, key=lambda x: x['predicted_points'], reverse=True):
                captain_mark = " (C)" if player['is_captain'] else " (VC)" if player['is_vice_captain'] else ""
                print(f"  {player['name']}{captain_mark} - {player['team']} - £{player['cost']}m - {player['predicted_points']:.1f} pts")
    
    print("\n----- BENCH -----")
    for i, player in enumerate(prediction_results['bench'], 1):
        print(f"  {i}. {player['name']} - {player['team']} - {player['position']} - £{player['cost']}m - {player['predicted_points']:.1f} pts")
    
    # Step 10: Save results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_dir = os.path.join(data_dir, "predictions")
        os.makedirs(results_dir, exist_ok=True)
        
        results_file = os.path.join(results_dir, f"team_prediction_gw{gameweek}_{timestamp}.json")
        
        # Convert numpy/pandas types to native Python types for JSON serialization
        def convert_to_json_serializable(obj):
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'to_dict'):  # pandas Series/DataFrame
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            else:
                try:
                    # Try converting to standard Python types
                    return float(obj) if isinstance(obj, (np.integer, np.floating)) else obj
                except:
                    return str(obj)
        
        json_safe_results = convert_to_json_serializable(prediction_results)
        
        with open(results_file, 'w') as f:
            json.dump(json_safe_results, f, indent=2)
        
        # Save team CSV
        team_file = os.path.join(results_dir, f"selected_team_gw{gameweek}_{timestamp}.csv")
        selected_team.to_csv(team_file, index=False)
        print(f"\n💾 Results saved to:")
        print(f"  - {results_file}")
        print(f"  - {team_file}")
        
        # Return JSON-safe results
        return json_safe_results
    else:
        # Even if not saving, convert to JSON-serializable format for return
        def convert_to_json_serializable(obj):
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'to_dict'):  # pandas Series/DataFrame
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            else:
                try:
                    # Try converting to standard Python types
                    return float(obj) if isinstance(obj, (np.integer, np.floating)) else obj
                except:
                    return str(obj)
        
        return convert_to_json_serializable(prediction_results)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict optimal FPL team for a gameweek')
    parser.add_argument('--gameweek', type=int, help='Gameweek to predict for (default: next gameweek)')
    parser.add_argument('--budget', type=float, default=100.0, help='Available budget in millions (default: 100.0)')
    parser.add_argument('--target', type=str, default='points_scored',
                       choices=['points_scored', 'goals_scored', 'assists', 'minutes_played'],
                       help='Target model to use for predictions (default: points_scored)')
    parser.add_argument('--data-dir', default='data', help='Data directory path')
    parser.add_argument('--no-save', action='store_true', help='Do not save prediction results')
    
    args = parser.parse_args()
    
    results = predict_team_for_gameweek(
        gameweek=args.gameweek,
        budget=args.budget,
        target=args.target,
        data_dir=args.data_dir,
        save_results=not args.no_save
    )
