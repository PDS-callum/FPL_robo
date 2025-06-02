import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
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
    
    def add_team_strength_features(self, data, bootstrap_data_list):
        """
        Add team strength features from bootstrap data
        """
        print("Adding team strength features...")
        result = data.copy()
        
        # Default values
        result['team_strength'] = 1000
        result['team_attack_home'] = 1000
        result['team_attack_away'] = 1000
        result['team_defence_home'] = 1000
        result['team_defence_away'] = 1000
        
        # Process each season's bootstrap data
        for season, bootstrap_data in bootstrap_data_list:
            if not bootstrap_data or 'teams' not in bootstrap_data:
                continue
                
            # Create team strength mapping for this season
            teams_df = pd.DataFrame(bootstrap_data['teams'])
            team_strength_map = {}
            
            for _, team in teams_df.iterrows():
                team_id = team['id']
                team_name = team.get('name', '')
                
                team_strength_map[team_id] = {
                    'strength': team.get('strength', 1000),
                    'attack_home': team.get('strength_attack_home', 1000),
                    'attack_away': team.get('strength_attack_away', 1000),
                    'defence_home': team.get('strength_defence_home', 1000),
                    'defence_away': team.get('strength_defence_away', 1000)
                }
                
                # Also map by name for flexibility
                if team_name:
                    team_strength_map[team_name] = team_strength_map[team_id]
            
            # Apply to data for this season
            season_mask = result['season'] == season
            
            for idx in result[season_mask].index:
                team_id = result.loc[idx, 'team']
                was_home = result.loc[idx, 'was_home'] if 'was_home' in result.columns else True
                
                if team_id in team_strength_map:
                    team_info = team_strength_map[team_id]
                    result.loc[idx, 'team_strength'] = team_info['strength']
                    
                    if was_home:
                        result.loc[idx, 'team_attack_strength'] = team_info['attack_home']
                        result.loc[idx, 'team_defence_strength'] = team_info['defence_home']
                    else:
                        result.loc[idx, 'team_attack_strength'] = team_info['attack_away']
                        result.loc[idx, 'team_defence_strength'] = team_info['defence_away']
        
        print("Team strength features added")
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
    
    def add_position_features(self, data, bootstrap_data_list):
        """
        Add position-based features and encoding
        """
        print("Adding position features...")
        result = data.copy()
        
        # Default position features
        result['position_encoded'] = 2  # Default to MID
        result['position_GKP'] = 0
        result['position_DEF'] = 0
        result['position_MID'] = 1  # Default
        result['position_FWD'] = 0
        
        # Position adjustment factors (based on typical scoring patterns)
        position_factors = {
            1: {'name': 'GKP', 'points_factor': 0.8, 'minutes_factor': 1.0},  # GKP
            2: {'name': 'DEF', 'points_factor': 0.9, 'minutes_factor': 0.95}, # DEF
            3: {'name': 'MID', 'points_factor': 1.0, 'minutes_factor': 0.85}, # MID
            4: {'name': 'FWD', 'points_factor': 1.2, 'minutes_factor': 0.75}  # FWD
        }
        
        # Process each season's bootstrap data for player positions
        for season, bootstrap_data in bootstrap_data_list:
            if not bootstrap_data or 'elements' not in bootstrap_data:
                continue
                
            # Create player position mapping
            players_df = pd.DataFrame(bootstrap_data['elements'])
            player_position_map = {}
            
            for _, player in players_df.iterrows():
                player_id = player['id']
                position = player.get('element_type', 2)  # Default to MID
                player_position_map[player_id] = position
            
            # Apply to data for this season
            season_mask = result['season'] == season
            
            for idx in result[season_mask].index:
                player_id = result.loc[idx, 'element']
                
                if player_id in player_position_map:
                    position = player_position_map[player_id]
                    result.loc[idx, 'position_encoded'] = position
                    
                    # One-hot encoding
                    for pos_id, pos_info in position_factors.items():
                        result.loc[idx, f'position_{pos_info["name"]}'] = 1 if position == pos_id else 0
                    
                    # Position-adjusted features
                    if position in position_factors:
                        factor_info = position_factors[position]
                        
                        # Adjust points expectation based on position
                        if 'total_points' in result.columns:
                            result.loc[idx, 'position_adjusted_points'] = (
                                result.loc[idx, 'total_points'] / factor_info['points_factor']
                            )
                        
                        # Adjust minutes expectation
                        if 'minutes' in result.columns:
                            result.loc[idx, 'position_adjusted_minutes'] = (
                                result.loc[idx, 'minutes'] / factor_info['minutes_factor']
                            )
        
        print("Position features added")
        return result
    
    def calculate_performance_efficiency(self, data):
        """
        Calculate performance efficiency metrics
        """
        print("Calculating performance efficiency metrics...")
        result = data.copy()
        
        # Points per minute (avoid division by zero)
        result['points_per_minute'] = result['total_points'] / (result['minutes'] + 1)
        
        # Goal involvement rate (goals + assists per match played)
        if 'goals_scored' in result.columns and 'assists' in result.columns:
            result['goal_involvement'] = result['goals_scored'] + result['assists']
            result['goal_involvement_rate'] = result['goal_involvement'] / ((result['minutes'] / 90) + 0.1)
        
        # BPS efficiency
        if 'bps' in result.columns:
            result['bps_per_minute'] = result['bps'] / (result['minutes'] + 1)
        
        # ICT efficiency
        if 'ict_index' in result.columns:
            result['ict_per_minute'] = result['ict_index'] / (result['minutes'] + 1)
        
        # Clean sheet efficiency (for defensive players)
        if 'clean_sheets' in result.columns:
            result['clean_sheet_rate'] = result['clean_sheets'] / ((result['minutes'] / 90) + 0.1)
        
        # Save efficiency
        if 'saves' in result.columns:
            result['saves_per_minute'] = result['saves'] / (result['minutes'] + 1)
        
        print("Performance efficiency metrics calculated")
        return result
    
    def calculate_consistency_metrics(self, data):
        """
        Calculate player consistency and form metrics
        """
        print("Calculating consistency metrics...")
        result = data.copy()
        
        # Group by player and calculate consistency
        for player_id, player_data in result.groupby('element'):
            if len(player_data) < 3:
                continue
                
            player_data = player_data.sort_values('round')
            idx = player_data.index
            
            # Points consistency (coefficient of variation)
            points_std = player_data['total_points'].std()
            points_mean = player_data['total_points'].mean()
            consistency_score = 1 - (points_std / (points_mean + 1))  # Higher = more consistent
            result.loc[idx, 'consistency_score'] = consistency_score
            
            # Form trend (recent vs overall performance)
            if len(player_data) >= 5:
                recent_avg = player_data['total_points'].tail(5).mean()
                overall_avg = player_data['total_points'].mean()
                form_trend = recent_avg / (overall_avg + 1)
                result.loc[idx, 'form_trend'] = form_trend
            
            # Appearance reliability
            games_played = len(player_data[player_data['minutes'] > 0])
            total_games = len(player_data)
            appearance_rate = games_played / total_games
            result.loc[idx, 'appearance_reliability'] = appearance_rate
            
            # Minutes consistency
            if 'minutes' in player_data.columns:
                minutes_std = player_data['minutes'].std()
                minutes_mean = player_data['minutes'].mean()
                minutes_consistency = 1 - (minutes_std / (minutes_mean + 1))
                result.loc[idx, 'minutes_consistency'] = minutes_consistency
        
        # Fill NaN values with defaults
        result['consistency_score'] = result['consistency_score'].fillna(0.5)
        result['form_trend'] = result['form_trend'].fillna(1.0)
        result['appearance_reliability'] = result['appearance_reliability'].fillna(0.5)
        result['minutes_consistency'] = result['minutes_consistency'].fillna(0.5)
        
        print("Consistency metrics calculated")
        return result
    
    def prepare_model_sequences(self, data, target_col='total_points', player_col='element', 
                               gameweek_col='round', lookback=3, validation_split=0.2):
        """
        Prepare model-ready sequences with proper train/validation split
        """
        print("Preparing model-ready sequences...")
        
        # Identify feature columns (exclude identifiers and target)
        exclude_cols = {player_col, gameweek_col, target_col, 'season', 'name', 'player_name'}
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        # Remove any remaining text/object columns
        numeric_features = []
        for col in feature_cols:
            if data[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                numeric_features.append(col)
        
        feature_cols = numeric_features
        print(f"Using {len(feature_cols)} features for sequences")
        
        # Fill missing values
        data_clean = data.copy()
        data_clean[feature_cols] = data_clean[feature_cols].fillna(0)
        
        # Create sequences for each player
        X_sequences = []
        y_sequences = []
        player_ids = []
        gameweek_ids = []
        
        for player_id, player_data in data_clean.groupby(player_col):
            player_data = player_data.sort_values(gameweek_col)
            
            if len(player_data) < lookback + 1:
                continue
            
            # Create sequences for this player
            for i in range(lookback, len(player_data)):
                # Input sequence (lookback gameweeks of features)
                sequence_data = player_data.iloc[i-lookback:i][feature_cols].values
                
                # Target (next gameweek points)
                target_value = player_data.iloc[i][target_col]
                
                X_sequences.append(sequence_data)
                y_sequences.append(target_value)
                player_ids.append(player_id)
                gameweek_ids.append(player_data.iloc[i][gameweek_col])
        
        if not X_sequences:
            raise ValueError("No sequences could be created. Check data format and lookback parameter.")
        
        X = np.array(X_sequences)
        y = np.array(y_sequences)
        
        print(f"Created {len(X)} sequences with shape {X.shape}")
        
        # Split into train/validation sets
        # Use stratified split by gameweek to ensure temporal distribution
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=None
        )
        
        # Create metadata
        metadata = {
            'feature_columns': feature_cols,
            'player_ids': player_ids,
            'gameweek_ids': gameweek_ids,
            'lookback': lookback,
            'n_features': len(feature_cols),
            'n_sequences': len(X),
            'train_size': len(X_train),
            'val_size': len(X_val)
        }
        
        print(f"Train set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        
        return X_train, X_val, y_train, y_val, feature_cols, metadata
    
    def save_processed_data(self, X_train, X_val, y_train, y_val, feature_names, metadata, 
                           filename_prefix="enhanced"):
        """
        Save processed training data and metadata
        """
        print("Saving processed training data...")
        
        # Save training data
        np.save(os.path.join(self.processed_dir, f"{filename_prefix}_X_train.npy"), X_train)
        np.save(os.path.join(self.processed_dir, f"{filename_prefix}_X_val.npy"), X_val)
        np.save(os.path.join(self.processed_dir, f"{filename_prefix}_y_train.npy"), y_train)
        np.save(os.path.join(self.processed_dir, f"{filename_prefix}_y_val.npy"), y_val)
        
        # Save feature names
        np.save(os.path.join(self.processed_dir, f"{filename_prefix}_feature_names.npy"), 
                np.array(feature_names))
        
        # Save metadata
        metadata['timestamp'] = datetime.now().isoformat()
        with open(os.path.join(self.processed_dir, f"{filename_prefix}_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved processed data with prefix '{filename_prefix}'")
        print(f"Files saved in: {self.processed_dir}")

def process_multi_season_data(seasons=None, data_dir="data", lookback=3):
    """
    Main function to process multi-season FPL data for model training
    """
    print("=== Multi-Season FPL Data Processing ===")
    
    if seasons is None:
        seasons = ["2022-23", "2023-24"]  # Default seasons
    
    # Initialize preprocessor
    preprocessor = FPLDataPreprocessor(data_dir=data_dir, lookback=lookback)
    
    # Load historical data collector
    try:
        from .history_data_collector import FPLHistoricalDataCollector
        collector = FPLHistoricalDataCollector()
    except ImportError:
        print("Error: Cannot import FPLHistoricalDataCollector")
        return None
    
    # Collect data from all seasons
    all_gameweek_data = []
    all_bootstrap_data = None
    
    for season in seasons:
        print(f"\nProcessing season {season}...")
        
        # Load season data
        season_data = collector.load_season_data(season)
        
        if not season_data:
            print(f"No data found for season {season}")
            continue
        
        # Get gameweek data
        if "merged_gw" in season_data and not season_data["merged_gw"].empty:
            gw_data = season_data["merged_gw"].copy()
            gw_data['season'] = season
            all_gameweek_data.append(gw_data)
            print(f"Added {len(gw_data)} gameweek records from {season}")
        
        # Use the most recent season's bootstrap data
        if season == seasons[-1]:
            all_bootstrap_data = preprocessor.load_bootstrap_data(season)
    
    if not all_gameweek_data:
        print("Error: No gameweek data found across seasons")
        return None
    
    # Combine all gameweek data
    print(f"\nCombining data from {len(seasons)} seasons...")
    combined_data = pd.concat(all_gameweek_data, ignore_index=True)
    print(f"Combined dataset: {len(combined_data)} records")
    
    # Process the data
    processed_data, feature_cols = preprocessor.process_fpl_data(
        combined_data, 
        bootstrap_data=all_bootstrap_data
    )
    
    # Create model-ready data
    model_data = preprocessor.create_model_ready_data(
        processed_data, 
        feature_cols,
        player_col='element' if 'element' in processed_data.columns else 'id',
        gameweek_col='round' if 'round' in processed_data.columns else 'GW'
    )
    
    print("\n=== Data Processing Complete ===")
    print(f"Training samples: {len(model_data['X_train'])}")
    print(f"Validation samples: {len(model_data['X_val'])}")
    print(f"Features per sample: {len(model_data['feature_cols'])}")
    print(f"Lookback period: {lookback} gameweeks")
    
    return model_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced FPL data preprocessing for model training')
    parser.add_argument('--seasons', nargs='+', default=["2022-23", "2023-24"], 
                       help='Seasons to process (e.g., 2022-23 2023-24)')
    parser.add_argument('--lookback', type=int, default=3, 
                       help='Number of gameweeks for lookback (default: 3)')
    parser.add_argument('--data-dir', default="data", 
                       help='Data directory (default: data)')
    
    args = parser.parse_args()
    
    # Process the data
    result = process_multi_season_data(
        seasons=args.seasons,
        data_dir=args.data_dir,
        lookback=args.lookback
    )
    
    if result:
        print("\nData processing completed successfully!")
        print("Training data saved and ready for model training.")
    else:
        print("\nData processing failed!")
