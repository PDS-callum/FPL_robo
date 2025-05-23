import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from datetime import datetime

def create_directories(base_dir="data"):
    """Create all necessary directories for data processing"""
    dirs = [
        os.path.join(base_dir, "raw"),
        os.path.join(base_dir, "processed"),
        os.path.join(base_dir, "historical"),
        os.path.join(base_dir, "models"),
        os.path.join(base_dir, "team_state"),
        os.path.join(base_dir, "features")
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        
    return dirs

def calculate_rolling_features(data, group_col, sort_col, value_cols, window_sizes=[3, 5]):
    """
    Calculate rolling averages for specified columns
    
    Parameters:
    -----------
    data : DataFrame
        Data to process
    group_col : str
        Column to group by (usually player_id)
    sort_col : str
        Column to sort by (usually gameweek)
    value_cols : list
        Columns to calculate rolling features for
    window_sizes : list
        List of window sizes for rolling calculations
        
    Returns:
    --------
    DataFrame with added rolling features
    """
    result = data.copy()
    
    # Group by the specified column
    for group, group_data in result.groupby(group_col):
        # Sort the group data
        group_data = group_data.sort_values(sort_col)
        
        # Calculate rolling stats for each value column and window size
        for col in value_cols:
            for window in window_sizes:
                # Calculate rolling mean
                roll_mean = group_data[col].rolling(window, min_periods=1).mean()
                result.loc[group_data.index, f'{col}_rolling_mean_{window}'] = roll_mean
                
                # Calculate rolling max
                roll_max = group_data[col].rolling(window, min_periods=1).max()
                result.loc[group_data.index, f'{col}_rolling_max_{window}'] = roll_max
                
                # For minutes/points, calculate consistency (std dev)
                if col in ['minutes', 'total_points']:
                    roll_std = group_data[col].rolling(window, min_periods=2).std()
                    result.loc[group_data.index, f'{col}_rolling_std_{window}'] = roll_std.fillna(0)
    
    # Fill NaN values with 0
    for col in result.columns:
        if col.startswith(tuple([f'{val}_rolling' for val in value_cols])):
            result[col] = result[col].fillna(0)
    
    return result

def normalize_features(data, feature_cols, method='standard'):
    """
    Normalize features using specified method
    
    Parameters:
    -----------
    data : DataFrame
        Data to normalize
    feature_cols : list
        Columns to normalize
    method : str
        Normalization method ('standard' or 'robust')
        
    Returns:
    --------
    DataFrame with normalized features, scaler
    """
    # Copy data to avoid modifying original
    result = data.copy()
    
    # Choose scaler based on method
    if method == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    
    # Normalize features
    result[feature_cols] = scaler.fit_transform(result[feature_cols].fillna(0))
    
    return result, scaler

def create_player_team_features(players_df, teams_df):
    """
    Create features that combine player and team information
    
    Parameters:
    -----------
    players_df : DataFrame
        Player data
    teams_df : DataFrame
        Team data
        
    Returns:
    --------
    DataFrame with combined features
    """
    result = players_df.copy()
    
    # Map team strength metrics to players
    team_map = {team['id']: team for _, team in teams_df.iterrows()}
    
    # Add team strength features
    result['team_strength'] = result['team'].map(lambda x: team_map.get(x, {}).get('strength', 0))
    result['team_attack_home'] = result['team'].map(lambda x: team_map.get(x, {}).get('strength_attack_home', 0))
    result['team_attack_away'] = result['team'].map(lambda x: team_map.get(x, {}).get('strength_attack_away', 0))
    result['team_defence_home'] = result['team'].map(lambda x: team_map.get(x, {}).get('strength_defence_home', 0))
    result['team_defence_away'] = result['team'].map(lambda x: team_map.get(x, {}).get('strength_defence_away', 0))
    
    # Calculate player value relative to team
    result['value_in_team'] = result.groupby('team')['now_cost'].transform(
        lambda x: (x - x.mean()) / x.std() if len(x) > 1 else 0
    )
    
    # Calculate player points contribution to team
    result['points_contribution'] = result.groupby('team')['total_points'].transform(
        lambda x: x / x.sum() if x.sum() > 0 else 0
    )
    
    return result

def create_fixture_difficulty_features(players_df, fixtures_df, teams_df, lookback=3, lookahead=5):
    """
    Create features based on fixture difficulty
    
    Parameters:
    -----------
    players_df : DataFrame
        Player data
    fixtures_df : DataFrame
        Fixture data
    teams_df : DataFrame
        Team data
    lookback : int
        Number of past fixtures to consider
    lookahead : int
        Number of future fixtures to consider
        
    Returns:
    --------
    DataFrame with fixture difficulty features
    """
    result = players_df.copy()
    
    # Check if we have fixture data
    if fixtures_df is None or fixtures_df.empty:
        print("No fixture data available, skipping fixture difficulty features")
        # Add placeholder columns
        result['avg_fixture_difficulty'] = 3.0  # Medium difficulty
        result['next_fixture_difficulty'] = 3.0
        result['next_is_home'] = 0.5  # 50/50
        return result
    
    # Get team ID to name mapping
    team_id_to_name = {team['id']: team['name'] for _, team in teams_df.iterrows()}
    
    # Process fixtures to get difficulty by team and gameweek
    fixture_difficulty = {}
    
    for _, fixture in fixtures_df.iterrows():
        gameweek = fixture.get('event')
        if gameweek is None or pd.isna(gameweek):
            continue
            
        gameweek = int(gameweek)
        home_team = fixture['team_h']
        away_team = fixture['team_a']
        
        # Get team names for readability in logs
        home_name = team_id_to_name.get(home_team, f"Team {home_team}")
        away_name = team_id_to_name.get(away_team, f"Team {away_team}")
        
        # Store difficulty for home team
        if home_team not in fixture_difficulty:
            fixture_difficulty[home_team] = {}
        
        h_difficulty = fixture.get('team_h_difficulty', 3)
        fixture_difficulty[home_team][gameweek] = {
            'opponent': away_team,
            'opponent_name': away_name,
            'difficulty': h_difficulty,
            'is_home': True
        }
        
        # Store difficulty for away team
        if away_team not in fixture_difficulty:
            fixture_difficulty[away_team] = {}
            
        a_difficulty = fixture.get('team_a_difficulty', 3)
        fixture_difficulty[away_team][gameweek] = {
            'opponent': home_team,
            'opponent_name': home_name,
            'difficulty': a_difficulty,
            'is_home': False
        }
    
    # Get current gameweek
    current_gw = max([gw for team_fixtures in fixture_difficulty.values() 
                      for gw in team_fixtures.keys()], default=1)
    
    # Calculate fixture difficulty features for each player
    for idx, player in result.iterrows():
        team_id = player['team']
        
        # Skip if team not in fixture data
        if team_id not in fixture_difficulty:
            continue
            
        # Get last few gameweeks' difficulty
        past_fixtures = [
            fixture_difficulty[team_id].get(gw, {'difficulty': 3, 'is_home': True})
            for gw in range(current_gw - lookback, current_gw)
            if gw > 0
        ]
        
        # Get next few gameweeks' difficulty
        future_fixtures = [
            fixture_difficulty[team_id].get(gw, {'difficulty': 3, 'is_home': True})
            for gw in range(current_gw, current_gw + lookahead)
        ]
        
        # Calculate average difficulty
        if past_fixtures:
            avg_past_diff = sum(f['difficulty'] for f in past_fixtures) / len(past_fixtures)
            result.loc[idx, 'past_fixture_difficulty'] = avg_past_diff
        else:
            result.loc[idx, 'past_fixture_difficulty'] = 3.0
            
        if future_fixtures:
            avg_future_diff = sum(f['difficulty'] for f in future_fixtures) / len(future_fixtures)
            result.loc[idx, 'avg_fixture_difficulty'] = avg_future_diff
            
            # Next fixture info
            result.loc[idx, 'next_fixture_difficulty'] = future_fixtures[0]['difficulty']
            result.loc[idx, 'next_is_home'] = 1 if future_fixtures[0]['is_home'] else 0
            
            # Difficulty trend (increasing or decreasing)
            if len(future_fixtures) > 1:
                diff_trend = future_fixtures[0]['difficulty'] - future_fixtures[-1]['difficulty']
                result.loc[idx, 'fixture_difficulty_trend'] = diff_trend
        else:
            # Default values if no future fixtures
            result.loc[idx, 'avg_fixture_difficulty'] = 3.0
            result.loc[idx, 'next_fixture_difficulty'] = 3.0
            result.loc[idx, 'next_is_home'] = 0.5
            result.loc[idx, 'fixture_difficulty_trend'] = 0
    
    # Fill any missing values
    for col in ['past_fixture_difficulty', 'avg_fixture_difficulty', 
                'next_fixture_difficulty', 'next_is_home', 'fixture_difficulty_trend']:
        if col in result.columns:
            result[col] = result[col].fillna(3.0 if 'difficulty' in col else 0.5 if col == 'next_is_home' else 0)
    
    return result

def prepare_training_sequences(data, player_col, gameweek_col, feature_cols, 
                               target_col='total_points', lookback=3):
    """
    Create training sequences for CNN model using lookback periods
    
    Parameters:
    -----------
    data : DataFrame
        Data to process
    player_col : str
        Column containing player identifier
    gameweek_col : str
        Column containing gameweek number
    feature_cols : list
        Columns to use as features
    target_col : str
        Column to use as target
    lookback : int
        Number of gameweeks to use for lookback
        
    Returns:
    --------
    X : numpy array
        Input features with shape (n_samples, lookback, n_features)
    y : numpy array
        Target values
    ids : list
        Player IDs corresponding to each sample
    """
    X_data = []
    y_data = []
    player_ids = []
    
    # Process each player separately
    for player_id, player_data in data.groupby(player_col):
        # Sort by gameweek
        player_data = player_data.sort_values(gameweek_col)
        
        # Need at least lookback+1 gameweeks to create a sample
        if len(player_data) <= lookback:
            continue
            
        # Create samples
        for i in range(len(player_data) - lookback):
            # Get feature sequence and target
            features = player_data.iloc[i:i+lookback][feature_cols].values
            next_gw = player_data.iloc[i+lookback]
            
            X_data.append(features)
            y_data.append(next_gw[target_col])
            player_ids.append(player_id)
            
    # Convert to numpy arrays
    X = np.array(X_data)
    y = np.array(y_data)
    
    # Make sure we have the right shape
    assert X.shape[1] == lookback, f"Expected lookback {lookback}, got {X.shape[1]}"
    assert X.shape[2] == len(feature_cols), f"Expected {len(feature_cols)} features, got {X.shape[2]}"
    
    return X, y, player_ids

def save_processed_data(data, filename, output_dir="data/processed"):
    """Save processed data with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"{filename}_{timestamp}.csv")
    data.to_csv(path, index=False)
    print(f"Saved processed data to {path}")
    
    # Also save as latest version without timestamp
    latest_path = os.path.join(output_dir, f"{filename}_latest.csv")
    data.to_csv(latest_path, index=False)
    
    return path
