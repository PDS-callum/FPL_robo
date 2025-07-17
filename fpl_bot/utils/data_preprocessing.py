import os
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class AdvancedFPLPreprocessor:
    """Advanced preprocessing for FPL data with feature engineering"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.processed_dir = os.path.join(data_dir, "processed")
        self.features_dir = os.path.join(data_dir, "features")
        self.models_dir = os.path.join(data_dir, "models")
        
        os.makedirs(self.models_dir, exist_ok=True)
        
    def create_rolling_features(self, df, player_col='player_id', gw_col='gameweek', 
                               target_cols=['total_points', 'minutes', 'goals_scored', 'assists'], 
                               windows=[3, 5, 10]):
        """Create rolling window features for better trend analysis"""
        print("Creating rolling window features...")
        
        df_sorted = df.sort_values([player_col, gw_col])
        
        for window in windows:
            for col in target_cols:
                if col in df.columns:
                    # Rolling mean
                    df_sorted[f'{col}_rolling_mean_{window}'] = (
                        df_sorted.groupby(player_col)[col]
                        .rolling(window=window, min_periods=1)
                        .mean()
                        .reset_index(level=0, drop=True)
                    )
                    
                    # Rolling standard deviation (volatility)
                    df_sorted[f'{col}_rolling_std_{window}'] = (
                        df_sorted.groupby(player_col)[col]
                        .rolling(window=window, min_periods=1)
                        .std()
                        .reset_index(level=0, drop=True)
                        .fillna(0)
                    )
                    
                    # Rolling max
                    df_sorted[f'{col}_rolling_max_{window}'] = (
                        df_sorted.groupby(player_col)[col]
                        .rolling(window=window, min_periods=1)
                        .max()
                        .reset_index(level=0, drop=True)
                    )
                    
                    # Trend (difference from rolling mean)
                    df_sorted[f'{col}_trend_{window}'] = (
                        df_sorted[col] - df_sorted[f'{col}_rolling_mean_{window}']
                    )
        
        return df_sorted
    
    def create_momentum_features(self, df, player_col='player_id', gw_col='gameweek'):
        """Create momentum and form features"""
        print("Creating momentum features...")
        
        df_sorted = df.sort_values([player_col, gw_col])
        
        # Calculate consecutive games with points
        df_sorted['points_streak'] = 0
        df_sorted['blank_streak'] = 0
        df_sorted['minutes_streak'] = 0
        
        for player_id in df_sorted[player_col].unique():
            player_mask = df_sorted[player_col] == player_id
            player_data = df_sorted[player_mask].copy()
            
            # Points streak
            points_streak = 0
            blank_streak = 0
            minutes_streak = 0
            
            for idx in player_data.index:
                points = df_sorted.loc[idx, 'points_scored'] if 'points_scored' in df_sorted.columns else 0
                minutes = df_sorted.loc[idx, 'minutes_played'] if 'minutes_played' in df_sorted.columns else 0
                
                # Points streak
                if points > 0:
                    points_streak += 1
                    blank_streak = 0
                else:
                    blank_streak += 1
                    points_streak = 0
                
                # Minutes streak (starter vs bench)
                if minutes >= 60:  # Started
                    minutes_streak += 1
                else:
                    minutes_streak = 0
                
                df_sorted.loc[idx, 'points_streak'] = points_streak
                df_sorted.loc[idx, 'blank_streak'] = blank_streak
                df_sorted.loc[idx, 'minutes_streak'] = minutes_streak
        
        # Form indicators
        df_sorted['is_in_form'] = (df_sorted['points_streak'] >= 2).astype(int)
        df_sorted['is_out_of_form'] = (df_sorted['blank_streak'] >= 2).astype(int)
        df_sorted['is_regular_starter'] = (df_sorted['minutes_streak'] >= 3).astype(int)
        
        return df_sorted
    
    def create_opponent_features(self, df, teams_df, fixtures_df):
        """Create features based on opponent strength"""
        print("Creating opponent-based features...")
        
        # Create opponent difficulty mapping
        team_strength = teams_df.set_index('id')[['strength_overall_home', 'strength_overall_away', 
                                                 'strength_attack_home', 'strength_attack_away',
                                                 'strength_defence_home', 'strength_defence_away']].to_dict('index')
        
        df['opponent_strength'] = 0
        df['opponent_attack_strength'] = 0
        df['opponent_defence_strength'] = 0
        df['fixture_difficulty_adjusted'] = 0
        
        for idx, row in df.iterrows():
            gw = row['gameweek']
            team = row['team']
            
            # Find fixture for this team in this gameweek
            gw_fixtures = fixtures_df[fixtures_df['event'] == gw]
            team_fixture = gw_fixtures[
                (gw_fixtures['team_h'] == team) | (gw_fixtures['team_a'] == team)
            ]
            
            if len(team_fixture) > 0:
                fixture = team_fixture.iloc[0]
                is_home = fixture['team_h'] == team
                opponent_id = fixture['team_a'] if is_home else fixture['team_h']
                
                if opponent_id in team_strength:
                    opponent = team_strength[opponent_id]
                    
                    # Opponent strength from player's perspective
                    if is_home:
                        df.at[idx, 'opponent_strength'] = opponent['strength_overall_away']
                        df.at[idx, 'opponent_attack_strength'] = opponent['strength_attack_away']
                        df.at[idx, 'opponent_defence_strength'] = opponent['strength_defence_away']
                        df.at[idx, 'fixture_difficulty_adjusted'] = fixture.get('team_h_difficulty', 3)
                    else:
                        df.at[idx, 'opponent_strength'] = opponent['strength_overall_home']
                        df.at[idx, 'opponent_attack_strength'] = opponent['strength_attack_home']
                        df.at[idx, 'opponent_defence_strength'] = opponent['strength_defence_home']
                        df.at[idx, 'fixture_difficulty_adjusted'] = fixture.get('team_a_difficulty', 3)
        
        return df
    
    def create_position_specific_features(self, df):
        """Create position-specific features"""
        print("Creating position-specific features...")
        
        # Position mappings (1=GK, 2=DEF, 3=MID, 4=FWD)
        position_mapping = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        df['position_name'] = df['position'].map(position_mapping)
        
        # Position-specific expectations
        df['expected_cs_points'] = 0  # Clean sheet points expectation
        df['expected_goal_points'] = 0  # Goal points expectation
        df['expected_assist_points'] = 0  # Assist points expectation
        
        # GK specific
        gk_mask = df['position'] == 1
        df.loc[gk_mask, 'expected_cs_points'] = df.loc[gk_mask, 'clean_sheets'] * 4
        df.loc[gk_mask, 'expected_goal_points'] = df.loc[gk_mask, 'goals_scored'] * 6
        
        # DEF specific
        def_mask = df['position'] == 2
        df.loc[def_mask, 'expected_cs_points'] = df.loc[def_mask, 'clean_sheets'] * 4
        df.loc[def_mask, 'expected_goal_points'] = df.loc[def_mask, 'goals_scored'] * 6
        df.loc[def_mask, 'expected_assist_points'] = df.loc[def_mask, 'assists'] * 3
        
        # MID specific
        mid_mask = df['position'] == 3
        df.loc[mid_mask, 'expected_cs_points'] = df.loc[mid_mask, 'clean_sheets'] * 1
        df.loc[mid_mask, 'expected_goal_points'] = df.loc[mid_mask, 'goals_scored'] * 5
        df.loc[mid_mask, 'expected_assist_points'] = df.loc[mid_mask, 'assists'] * 3
        
        # FWD specific
        fwd_mask = df['position'] == 4
        df.loc[fwd_mask, 'expected_goal_points'] = df.loc[fwd_mask, 'goals_scored'] * 4
        df.loc[fwd_mask, 'expected_assist_points'] = df.loc[fwd_mask, 'assists'] * 3
        
        return df
    
    def create_value_features(self, df):
        """Create value and ownership features"""
        print("Creating value and ownership features...")
        
        # Points per million
        df['points_per_million'] = np.where(df['price'] > 0, df['points_scored'] / df['price'], 0)
        
        # Value categories
        df['price_category'] = pd.cut(df['price'], 
                                     bins=[0, 5, 7, 9, 12, 20], 
                                     labels=['Budget', 'Mid', 'Premium', 'Elite', 'Super_Premium'])
        
        # Ownership categories
        df['ownership_category'] = pd.cut(df['selected_by_percent'], 
                                         bins=[0, 5, 15, 30, 100], 
                                         labels=['Low', 'Medium', 'High', 'Very_High'])
        
        # Differential player (low ownership but high points)
        df['is_differential'] = ((df['selected_by_percent'] < 10) & 
                                (df['points_scored'] > df['points_scored'].quantile(0.75))).astype(int)
        
        return df
    
    def create_team_form_features(self, df, fixtures_df):
        """Create team-level form features"""
        print("Creating team form features...")
        
        # Calculate team form based on recent results
        team_form = {}
        
        for team_id in df['team'].unique():
            team_fixtures = fixtures_df[
                ((fixtures_df['team_h'] == team_id) | (fixtures_df['team_a'] == team_id)) &
                (fixtures_df['finished'] == True)
            ].sort_values('event')
            
            if len(team_fixtures) > 0:
                recent_form = []
                for _, fixture in team_fixtures.tail(5).iterrows():  # Last 5 games
                    home_score = fixture.get('team_h_score', 0)
                    away_score = fixture.get('team_a_score', 0)
                    
                    if fixture['team_h'] == team_id:  # Team playing at home
                        if home_score > away_score:
                            recent_form.append(3)  # Win
                        elif home_score == away_score:
                            recent_form.append(1)  # Draw
                        else:
                            recent_form.append(0)  # Loss
                    else:  # Team playing away
                        if away_score > home_score:
                            recent_form.append(3)  # Win
                        elif home_score == away_score:
                            recent_form.append(1)  # Draw
                        else:
                            recent_form.append(0)  # Loss
                
                team_form[team_id] = {
                    'form_points': sum(recent_form),
                    'recent_wins': sum([1 for x in recent_form if x == 3]),
                    'recent_losses': sum([1 for x in recent_form if x == 0]),
                    'games_played': len(recent_form)
                }
            else:
                team_form[team_id] = {
                    'form_points': 0,
                    'recent_wins': 0,
                    'recent_losses': 0,
                    'games_played': 0
                }
        
        # Add team form to dataframe
        df['team_form_points'] = df['team'].map(lambda x: team_form.get(x, {}).get('form_points', 0))
        df['team_recent_wins'] = df['team'].map(lambda x: team_form.get(x, {}).get('recent_wins', 0))
        df['team_recent_losses'] = df['team'].map(lambda x: team_form.get(x, {}).get('recent_losses', 0))
        
        return df
    
    def create_advanced_features(self, base_dataset, teams_df, fixtures_df):
        """Create all advanced features"""
        print("Creating advanced feature set...")
        
        df = base_dataset.copy()
        
        # Apply all feature engineering steps
        df = self.create_rolling_features(df)
        df = self.create_momentum_features(df)
        df = self.create_opponent_features(df, teams_df, fixtures_df)
        df = self.create_position_specific_features(df)
        df = self.create_value_features(df)
        df = self.create_team_form_features(df, fixtures_df)
        
        return df
    
    def feature_importance_analysis(self, X, y, feature_names, target_name='points_scored'):
        """Analyze feature importance using Random Forest"""
        print(f"Analyzing feature importance for {target_name}...")
        
        # Train Random Forest to get feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save feature importance
        importance_df.to_csv(
            os.path.join(self.features_dir, f"feature_importance_{target_name}.csv"),
            index=False
        )
        
        print(f"Top 10 features for {target_name}:")
        print(importance_df.head(10))
        
        return importance_df
    
    def create_prediction_features(self, current_gw_data, historical_data):
        """Create features for predicting next gameweek"""
        print("Creating features for next gameweek prediction...")
        
        prediction_features = []
        
        for _, player in current_gw_data.iterrows():
            player_id = player['player_id']
            
            # Get historical data for this player
            player_history = historical_data[historical_data['player_id'] == player_id]
            
            if len(player_history) == 0:
                continue
            
            # Calculate recent form (last 5 games)
            recent_games = player_history.tail(5)
            
            feature_row = {
                'player_id': player_id,
                'position': player['position'],
                'team': player['team'],
                'price': player['price'],
                'selected_by_percent': player.get('selected_by_percent', 0),
                
                # Recent form
                'avg_points_last_5': recent_games['points_scored'].mean(),
                'avg_minutes_last_5': recent_games['minutes_played'].mean(),
                'form_last_5': recent_games['points_scored'].sum(),
                'goals_last_5': recent_games['goals_scored'].sum(),
                'assists_last_5': recent_games['assists'].sum(),
                
                # Season stats
                'total_points_season': player_history['points_scored'].sum(),
                'avg_points_per_game': player_history[player_history['minutes_played'] > 0]['points_scored'].mean(),
                'games_played': len(player_history[player_history['minutes_played'] > 0]),
                
                # Consistency
                'points_std': player_history['points_scored'].std(),
                'minutes_consistency': player_history['minutes_played'].std(),
                
                # Current form indicators
                'points_streak': player.get('points_streak', 0),
                'minutes_streak': player.get('minutes_streak', 0),
                'is_in_form': player.get('is_in_form', 0),
                'is_regular_starter': player.get('is_regular_starter', 0),
            }
            
            prediction_features.append(feature_row)
        
        return pd.DataFrame(prediction_features)
    
    def save_feature_engineered_data(self, df, suffix="advanced"):
        """Save feature engineered dataset"""
        output_path = os.path.join(self.processed_dir, f"fpl_dataset_{suffix}.csv")
        df.to_csv(output_path, index=False)
        print(f"Advanced dataset saved to {output_path}")
        
        # Save feature summary
        feature_summary = {
            'total_features': len(df.columns),
            'total_rows': len(df),
            'unique_players': df['player_id'].nunique(),
            'gameweeks_covered': df['gameweek'].nunique(),
            'feature_types': {
                'numerical': len(df.select_dtypes(include=[np.number]).columns),
                'categorical': len(df.select_dtypes(include=['object', 'category']).columns)
            }
        }
        
        with open(os.path.join(self.processed_dir, f"feature_summary_{suffix}.json"), 'w') as f:
            json.dump(feature_summary, f, indent=2)
        
        return output_path
    
    def validate_data_quality(self, df):
        """Validate the quality of processed data"""
        print("Validating data quality...")
        
        quality_report = {
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'outliers': {},
            'correlations': {}
        }
        
        # Check for outliers in key columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in ['points_scored', 'minutes_played', 'price']:
            if col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
                quality_report['outliers'][col] = len(outliers)
        
        # Check correlations between features and target
        if 'points_scored' in df.columns:
            correlations = df[numeric_cols].corr()['points_scored'].abs().sort_values(ascending=False)
            quality_report['correlations'] = correlations.head(10).to_dict()
        
        # Save quality report
        with open(os.path.join(self.processed_dir, "data_quality_report.json"), 'w') as f:
            json.dump(quality_report, f, indent=2, default=str)
        
        print("Data quality report saved")
        return quality_report
