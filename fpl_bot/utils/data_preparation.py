import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

def create_directories(base_dir="data"):
    """Create all necessary directories for data processing"""
    dirs = [
        os.path.join(base_dir, "raw"),
        os.path.join(base_dir, "processed"),
        os.path.join(base_dir, "historical"),
        os.path.join(base_dir, "models"),
        os.path.join(base_dir, "team_state"),
        os.path.join(base_dir, "features"),
        os.path.join(base_dir, "plots"),
        os.path.join(base_dir, "validation")
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        
    return dirs

class FPLDataPreprocessor:
    """
    Comprehensive data preprocessing for FPL model training
    """
    
    def __init__(self, data_dir="data", lookback=3, validation_split=0.2):
        self.data_dir = data_dir
        self.processed_dir = os.path.join(data_dir, "processed")
        self.lookback = lookback
        self.validation_split = validation_split
        
        # Create directories
        create_directories(data_dir)
        
        # Initialize scalers and feature selectors
        self.feature_scaler = None
        self.target_scaler = None
        self.feature_selector = None
        
        # Feature configuration
        self.base_features = [
            'minutes', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
            'bonus', 'bps', 'influence', 'creativity', 'threat', 'ict_index',
            'was_home', 'team_strength'
        ]
        
        self.rolling_features = [
            'total_points', 'minutes', 'goals_scored', 'assists', 'bonus', 
            'clean_sheets', 'goals_conceded', 'bps', 'influence', 'creativity', 'threat'
        ]
        
        self.window_sizes = [3, 5, 10]
        
    def load_bootstrap_data(self, season="2022-23"):
        """Load bootstrap static data for a specific season"""
        try:
            # Try to find the most recent bootstrap file for the season
            raw_files = os.listdir(os.path.join(self.data_dir, "raw"))
            bootstrap_files = [f for f in raw_files if f.startswith("bootstrap_static") and season.replace("-", "") in f]
            
            if not bootstrap_files:
                print(f"No bootstrap data found for season {season}")
                return None
                
            latest_file = sorted(bootstrap_files)[-1]
            file_path = os.path.join(self.data_dir, "raw", latest_file)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            print(f"Error loading bootstrap data: {e}")
            return None

    def calculate_advanced_rolling_features(self, data, group_col='element', sort_col='round'):
        """
        Calculate advanced rolling statistics for player performance
        """
        print("Calculating advanced rolling features...")
        result = data.copy()
        
        # Ensure we have the required columns
        required_cols = ['total_points', 'minutes', 'goals_scored', 'assists']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols}, filling with zeros")
            for col in missing_cols:
                result[col] = 0
        
        for group_name, group_data in result.groupby(group_col):
            if len(group_data) < 2:
                continue
                
            # Sort by gameweek
            group_data = group_data.sort_values(sort_col)
            idx = group_data.index
            
            for col in self.rolling_features:
                if col not in group_data.columns:
                    continue
                    
                for window in self.window_sizes:
                    # Rolling mean
                    roll_mean = group_data[col].rolling(window, min_periods=1).mean()
                    result.loc[idx, f'{col}_rolling_mean_{window}'] = roll_mean
                    
                    # Rolling max
                    roll_max = group_data[col].rolling(window, min_periods=1).max()
                    result.loc[idx, f'{col}_rolling_max_{window}'] = roll_max
                    
                    # Rolling min
                    roll_min = group_data[col].rolling(window, min_periods=1).min()
                    result.loc[idx, f'{col}_rolling_min_{window}'] = roll_min
                    
                    # Rolling std (consistency metric)
                    roll_std = group_data[col].rolling(window, min_periods=2).std()
                    result.loc[idx, f'{col}_rolling_std_{window}'] = roll_std.fillna(0)
                    
                    # Rolling trend (slope of linear regression)
                    def calculate_trend(values):
                        if len(values) < 2:
                            return 0
                        x = np.arange(len(values))
                        try:
                            slope = np.polyfit(x, values, 1)[0]
                            return slope
                        except:
                            return 0
                    
                    roll_trend = group_data[col].rolling(window, min_periods=2).apply(
                        calculate_trend, raw=True
                    )
                    result.loc[idx, f'{col}_rolling_trend_{window}'] = roll_trend.fillna(0)
            
            # Calculate form metrics
            # Points per minute (efficiency)
            points_per_min = group_data['total_points'] / (group_data['minutes'] + 1)  # +1 to avoid division by zero
            for window in self.window_sizes:
                roll_ppm = points_per_min.rolling(window, min_periods=1).mean()
                result.loc[idx, f'points_per_minute_rolling_{window}'] = roll_ppm
            
            # Goal involvement (goals + assists)
            if 'goals_scored' in group_data.columns and 'assists' in group_data.columns:
                goal_involvement = group_data['goals_scored'] + group_data['assists']
                for window in self.window_sizes:
                    roll_gi = goal_involvement.rolling(window, min_periods=1).mean()
                    result.loc[idx, f'goal_involvement_rolling_{window}'] = roll_gi
        
        # Fill any remaining NaN values
        rolling_cols = [col for col in result.columns if '_rolling_' in col]
        for col in rolling_cols:
            result[col] = result[col].fillna(0)
            
        print(f"Added {len(rolling_cols)} rolling feature columns")
        return result
    
    def create_team_strength_features(self, data, bootstrap_data):
        """
        Create team strength and opposition features
        """
        print("Creating team strength features...")
        result = data.copy()
        
        if not bootstrap_data or 'teams' not in bootstrap_data:
            print("Warning: No team data available, using default values")
            result['team_strength'] = 1000  # Default strength
            result['team_attack_strength'] = 1000
            result['team_defence_strength'] = 1000
            return result
        
        # Create team strength mapping
        teams_df = pd.DataFrame(bootstrap_data['teams'])
        team_strength_map = {}
        
        for _, team in teams_df.iterrows():
            team_id = team['id']
            team_strength_map[team_id] = {
                'strength': team.get('strength', 1000),
                'strength_attack_home': team.get('strength_attack_home', 1000),
                'strength_attack_away': team.get('strength_attack_away', 1000),
                'strength_defence_home': team.get('strength_defence_home', 1000),
                'strength_defence_away': team.get('strength_defence_away', 1000),
                'strength_overall_home': team.get('strength_overall_home', 1000),
                'strength_overall_away': team.get('strength_overall_away', 1000)
            }
        
        # Add team strength features to player data
        result['team_strength'] = result['team'].map(
            lambda x: team_strength_map.get(x, {}).get('strength', 1000)
        )
        
        result['team_attack_home'] = result['team'].map(
            lambda x: team_strength_map.get(x, {}).get('strength_attack_home', 1000)
        )
        
        result['team_attack_away'] = result['team'].map(
            lambda x: team_strength_map.get(x, {}).get('strength_attack_away', 1000)
        )
        
        result['team_defence_home'] = result['team'].map(
            lambda x: team_strength_map.get(x, {}).get('strength_defence_home', 1000)
        )
        
        result['team_defence_away'] = result['team'].map(
            lambda x: team_strength_map.get(x, {}).get('strength_defence_away', 1000)
        )
        
        # Calculate dynamic team strength based on home/away
        result['effective_attack_strength'] = np.where(
            result['was_home'] == 1,
            result['team_attack_home'],
            result['team_attack_away']
        )
        
        result['effective_defence_strength'] = np.where(
            result['was_home'] == 1,
            result['team_defence_home'],
            result['team_defence_away']
        )
        
        return result
    
    def create_position_features(self, data, bootstrap_data):
        """
        Create position-based features and encodings
        """
        print("Creating position features...")
        result = data.copy()
        
        if not bootstrap_data or 'elements' not in bootstrap_data:
            print("Warning: No player data available for position mapping")
            result['position'] = 3  # Default to midfielder
            result['element_type'] = 3
        else:
            # Create player to position mapping
            players_df = pd.DataFrame(bootstrap_data['elements'])
            player_position_map = players_df.set_index('id')['element_type'].to_dict()
            
            # Map positions to players (if element column exists)
            if 'element' in result.columns:
                result['element_type'] = result['element'].map(player_position_map)
            elif 'id' in result.columns:
                result['element_type'] = result['id'].map(player_position_map)
            else:
                result['element_type'] = 3  # Default to midfielder
        
        # Fill missing positions
        result['element_type'] = result['element_type'].fillna(3)
        
        # Create position dummy variables
        position_dummies = pd.get_dummies(result['element_type'], prefix='pos')
        result = pd.concat([result, position_dummies], axis=1)
        
        # Create position-specific features
        # Different expectations for different positions
        result['position_adjusted_points'] = result['total_points'] * np.where(
            result['element_type'] == 1, 0.8,  # Goalkeepers typically score less
            np.where(result['element_type'] == 4, 1.2, 1.0)  # Forwards typically score more
        )
        
        return result
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
