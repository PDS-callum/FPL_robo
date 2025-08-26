import os
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from .models.fpl_model import FPLPredictionModel
from .utils.team_optimizer import FPLTeamOptimizer
from .utils.current_season_collector import FPLCurrentSeasonCollector
from .utils.constants import POSITION_MAP
from .utils.file_utils import convert_to_json_serializable
import json
from datetime import datetime

def predict_team_for_gameweek(
    gameweek: Optional[int] = None,
    budget: float = 100.0,
    target: str = 'points_scored',
    data_dir: str = "data",
    save_results: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Predict optimal FPL team for a specific gameweek.

    Args:
        gameweek (Optional[int]): Gameweek to predict for (if None, predicts for next gameweek).
        budget (float): Available budget in millions.
        target (str): Target variable model to use for predictions.
        data_dir (str): Data directory path.
        save_results (bool): Whether to save prediction results.

    Returns:
        Optional[Dict[str, Any]]: Complete prediction results including team, formation, etc.
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
    
    # Try to load current fixtures for better captain selection
    fixtures_df = pd.DataFrame()
    try:
        # Look for the most recent fixtures file
        current_season_dir = os.path.join(data_dir, 'current_season')
        if os.path.exists(current_season_dir):
            fixture_files = [f for f in os.listdir(current_season_dir) if f.startswith('fixtures_') and f.endswith('.json')]
            if fixture_files:
                latest_fixture_file = sorted(fixture_files)[-1]
                fixture_path = os.path.join(current_season_dir, latest_fixture_file)
                
                with open(fixture_path, 'r') as f:
                    fixtures_data = json.load(f)
                    fixtures_df = pd.DataFrame(fixtures_data)
                    print(f"✅ Loaded fixtures from {latest_fixture_file}")
    except Exception as e:
        print(f"⚠️  Could not load fixtures: {e}")
        print("⚠️  Captain selection will use basic logic without fixture analysis")
    
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
        
        # Enhanced fixture difficulty using loaded fixtures data
        if len(fixtures_df) > 0 and gameweek:
            # Find this team's fixture for the current gameweek
            team_fixture = fixtures_df[
                ((fixtures_df['team_h'] == player['team']) | (fixtures_df['team_a'] == player['team'])) &
                (fixtures_df['event'] == gameweek)
            ]
            
            if len(team_fixture) > 0:
                fixture = team_fixture.iloc[0]
                is_home = fixture['team_h'] == player['team']
                opponent_id = fixture['team_a'] if is_home else fixture['team_h']
                
                # Get opponent team info
                opponent_info = teams_df[teams_df['id'] == opponent_id]
                if len(opponent_info) > 0:
                    opponent = opponent_info.iloc[0]
                    
                    # Set fixture features based on actual data
                    features.update({
                        'avg_home_difficulty': fixture.get('team_h_difficulty', 3),
                        'avg_away_difficulty': fixture.get('team_a_difficulty', 3),
                        'home_difficulty': fixture.get('team_h_difficulty', 3) if is_home else fixture.get('team_a_difficulty', 3),
                        'away_difficulty': fixture.get('team_a_difficulty', 3) if not is_home else fixture.get('team_h_difficulty', 3),
                        'is_home': 1 if is_home else 0,
                        'fixture_difficulty': fixture.get('team_h_difficulty', 3) if is_home else fixture.get('team_a_difficulty', 3),
                        'opponent_strength': opponent.get('strength_overall_away' if is_home else 'strength_overall_home', 1000),
                    })
                else:
                    # Default values if opponent not found
                    features.update({
                        'avg_home_difficulty': 3,
                        'avg_away_difficulty': 3,
                        'home_difficulty': 3,
                        'away_difficulty': 3,
                        'is_home': 1,
                        'fixture_difficulty': 3,
                        'opponent_strength': 1000,
                    })
            else:
                # Default values if no fixture found
                features.update({
                    'avg_home_difficulty': 3,
                    'avg_away_difficulty': 3,
                    'home_difficulty': 3,
                    'away_difficulty': 3,
                    'is_home': 1,
                    'fixture_difficulty': 3,
                    'opponent_strength': 1000,
                })
        else:
            # Fallback to default values when no fixtures available
            features.update({
                'avg_home_difficulty': 3,  # Average difficulty
                'avg_away_difficulty': 3,  # Average difficulty  
                'home_difficulty': 3,
                'away_difficulty': 3,
                'is_home': 1,  # Assume home for prediction
                'fixture_difficulty': 3,
                'opponent_strength': 1000,
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
    
    # Enhance selected team with fixture features for better captain selection
    if len(selected_team) > 0 and len(fixtures_df) > 0 and gameweek:
        print("🔍 Adding fixture features for captain selection...")
        enhanced_features = ['is_home', 'fixture_difficulty', 'opponent_strength']
        
        for feature in enhanced_features:
            if feature in features_df.columns:
                # Merge fixture features into selected team
                feature_mapping = features_df.set_index('id')[feature].to_dict()
                selected_team[feature] = selected_team['id'].map(feature_mapping).fillna(
                    3 if feature == 'fixture_difficulty' else 1000 if feature == 'opponent_strength' else 1
                )
    
    # Check if team optimization failed completely
    if len(selected_team) == 0:
        print("❌ CRITICAL: Team optimization failed completely!")
        print("📊 This indicates the constraints cannot be met with available players.")
        print("📊 Available player summary:")
        if len(available_players) > 0:
            pos_counts = available_players['element_type'].map({1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}).value_counts()
            print(f"   Position distribution: {dict(pos_counts)}")
            print(f"   Price range: £{available_players['now_cost'].min()/10:.1f}m - £{available_players['now_cost'].max()/10:.1f}m")
            print(f"   Prediction range: {predictions_df['predicted_points'].min():.3f} - {predictions_df['predicted_points'].max():.3f}")
            
            # Calculate minimum possible team cost
            min_costs_by_pos = {}
            for pos_code, pos_name in {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}.items():
                pos_players = available_players[available_players['element_type'] == pos_code]
                if len(pos_players) > 0:
                    min_costs_by_pos[pos_name] = pos_players['now_cost'].min() / 10
            
            min_team_cost = (
                min_costs_by_pos.get('GK', 0) * 2 +
                min_costs_by_pos.get('DEF', 0) * 5 +
                min_costs_by_pos.get('MID', 0) * 5 +
                min_costs_by_pos.get('FWD', 0) * 3
            )
            print(f"   Minimum possible team cost: £{min_team_cost:.1f}m")
            print(f"   Your budget: £{budget:.1f}m")
            
            if min_team_cost > budget:
                print(f"❌ BUDGET TOO LOW: Need at least £{min_team_cost:.1f}m to build any valid team")
                print(f"💡 Try increasing budget to at least £{min_team_cost + 5:.0f}m")
        
        print("❌ Cannot proceed with team prediction. Please check data quality and constraints.")
        return None

    # STRICT VALIDATION: Reject any team that doesn't meet FPL requirements
    if len(selected_team) != 15:
        print(f"❌ INVALID TEAM: Selected {len(selected_team)} players (FPL requires exactly 15)")
        print("❌ TEAM SELECTION FAILED - This should never happen with the new strict optimizer")
        
        # Show what we got for debugging
        if len(selected_team) > 0:
            pos_counts = selected_team['position'].value_counts()
            total_cost = selected_team['price'].sum() if 'price' in selected_team.columns else selected_team['now_cost'].sum() / 10
            print(f"   Selected positions: {dict(pos_counts)}")
            print(f"   Total cost: £{total_cost:.1f}m")
            
            # Show what positions we're missing
            required_positions = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
            actual_positions = dict(pos_counts)
            missing = {}
            for pos, req in required_positions.items():
                actual = actual_positions.get(pos, 0)
                if actual < req:
                    missing[pos] = req - actual
            if missing:
                print(f"   Missing positions: {missing}")
        
        return None

    # FINAL VALIDATION: One last check to ensure team meets ALL FPL constraints
    is_valid, errors = optimizer.validate_team(selected_team)
    if not is_valid:
        print("❌ FINAL VALIDATION FAILED:")
        for error in errors:
            print(f"   - {error}")
        
        # Show team composition for debugging
        if len(selected_team) > 0:
            print("📊 Team composition:")
            if 'position' in selected_team.columns:
                pos_breakdown = selected_team['position'].value_counts()
                print(f"   Positions: {dict(pos_breakdown)}")
            
            total_cost = selected_team['price'].sum() if 'price' in selected_team.columns else selected_team['now_cost'].sum() / 10
            print(f"   Total cost: £{total_cost:.1f}m / £{budget:.1f}m")
            
            if 'team' in selected_team.columns:
                team_breakdown = selected_team['team'].value_counts()
                over_limit = team_breakdown[team_breakdown > 3]
                if len(over_limit) > 0:
                    print(f"   Teams over limit: {dict(over_limit)}")
        
        print("❌ Cannot proceed with invalid team - strict FPL constraints must be met")
        return None

    
    # Step 7: Select playing XI
    print("⚡ Selecting playing XI...")
    try:
        playing_xi, captain, vice_captain, formation = optimizer.select_playing_xi(selected_team)
    except Exception as e:
        print(f"❌ Failed to select playing XI: {e}")
        return None
    
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
    for _, player in playing_xi.iterrows():
        player_info = {
            'name': player['web_name'],
            'team': player.get('team_name', 'Unknown'),
            'position': POSITION_MAP.get(player.get('element_type', 0), 'Unknown'),
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
            'position': POSITION_MAP.get(player.get('element_type', 0), 'Unknown'),
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
    for position in ['GK', 'DEF', 'MID', 'FWD']:
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
