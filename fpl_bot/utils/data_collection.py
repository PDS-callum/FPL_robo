import os
import pandas as pd
import json
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

class FPLDataCollector:
    def __init__(self, data_dir="data"):
        self.base_url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
        self.data_dir = data_dir
        self.history_dir = os.path.join(data_dir, "historical")
        os.makedirs(self.history_dir, exist_ok=True)
        
    def get_available_seasons(self):
        """Return list of available seasons in the format '2023-24'"""
        # This is a static list based on what's available in the repo
        # You could dynamically fetch this, but it would require web scraping the GitHub repo
        return [
            '2016-17', '2017-18', '2018-19', '2019-20', 
            '2020-21', '2021-22', '2022-23', '2023-24'
        ]
    
    def collect_season_data(self, season):
        """Collect all data for a specific season"""
        print(f"Collecting data for season {season}...")
        
        # Create season directory
        season_dir = os.path.join(self.history_dir, season)
        os.makedirs(season_dir, exist_ok=True)
        
        # Collect merged gameweek data
        gw_url = f"{self.base_url}/{season}/gws/merged_gw.csv"
        try:
            # Try multiple strategies to handle malformed CSV
            merged_gw = None
            
            # Strategy 1: Standard read with error handling
            try:
                merged_gw = pd.read_csv(gw_url, encoding='utf-8')
            except (UnicodeDecodeError, pd.errors.ParserError):
                try:
                    merged_gw = pd.read_csv(gw_url, encoding='latin-1')
                except pd.errors.ParserError:
                    # Strategy 2: Use error_bad_lines=False (for older pandas) or on_bad_lines='skip'
                    try:
                        merged_gw = pd.read_csv(gw_url, encoding='utf-8', on_bad_lines='skip')
                    except TypeError:
                        # For older pandas versions
                        merged_gw = pd.read_csv(gw_url, encoding='utf-8', error_bad_lines=False, warn_bad_lines=True)
                    except:
                        # Strategy 3: Quote handling
                        try:
                            merged_gw = pd.read_csv(gw_url, encoding='utf-8', quoting=1, skipinitialspace=True)
                        except:
                            # Strategy 4: Use python engine which is more forgiving
                            merged_gw = pd.read_csv(gw_url, encoding='utf-8', engine='python', on_bad_lines='skip')
        
            if merged_gw is not None:
                merged_gw.to_csv(os.path.join(season_dir, "merged_gw.csv"), index=False, encoding='utf-8')
                print(f"  - Saved merged gameweek data: {merged_gw.shape[0]} rows")
            else:
                print(f"  - Failed to parse merged gameweek data after trying multiple strategies")
        except Exception as e:
            print(f"  - Failed to collect merged gameweek data: {e}")
    
        # Collect players data
        players_url = f"{self.base_url}/{season}/players_raw.csv"
        try:
            # Apply similar error handling for players data
            players = None
            
            try:
                players = pd.read_csv(players_url, encoding='utf-8')
            except (UnicodeDecodeError, pd.errors.ParserError):
                try:
                    players = pd.read_csv(players_url, encoding='latin-1')
                except pd.errors.ParserError:
                    try:
                        players = pd.read_csv(players_url, encoding='utf-8', on_bad_lines='skip')
                    except TypeError:
                        players = pd.read_csv(players_url, encoding='utf-8', error_bad_lines=False, warn_bad_lines=True)
                    except:
                        try:
                            players = pd.read_csv(players_url, encoding='utf-8', quoting=1, skipinitialspace=True)
                        except:
                            players = pd.read_csv(players_url, encoding='utf-8', engine='python', on_bad_lines='skip')
            
            if players is not None:
                players.to_csv(os.path.join(season_dir, "players_raw.csv"), index=False, encoding='utf-8')
                print(f"  - Saved players data: {players.shape[0]} players")
            else:
                print(f"  - Failed to parse players data after trying multiple strategies")
        except Exception as e:
            print(f"  - Failed to collect players data: {e}")
    
        # Collect teams data
        teams_url = f"{self.base_url}/{season}/teams.csv"
        try:
            # Apply similar error handling for teams data
            teams = None
            
            try:
                teams = pd.read_csv(teams_url, encoding='utf-8')
            except (UnicodeDecodeError, pd.errors.ParserError):
                try:
                    teams = pd.read_csv(teams_url, encoding='latin-1')
                except pd.errors.ParserError:
                    try:
                        teams = pd.read_csv(teams_url, encoding='utf-8', on_bad_lines='skip')
                    except TypeError:
                        teams = pd.read_csv(teams_url, encoding='utf-8', error_bad_lines=False, warn_bad_lines=True)
                    except:
                        teams = pd.read_csv(teams_url, encoding='utf-8', engine='python', on_bad_lines='skip')
            
            if teams is not None:
                teams.to_csv(os.path.join(season_dir, "teams.csv"), index=False, encoding='utf-8')
                print(f"  - Saved teams data: {teams.shape[0]} teams")
            else:
                print(f"  - Failed to parse teams data after trying multiple strategies")
        except Exception as e:
            print(f"  - Failed to collect teams data: {e}")
    
        # Collect fixtures data
        fixtures_url = f"{self.base_url}/{season}/fixtures.csv"
        try:
            # Apply similar error handling for fixtures data
            fixtures = None
            
            try:
                fixtures = pd.read_csv(fixtures_url, encoding='utf-8')
            except (UnicodeDecodeError, pd.errors.ParserError):
                try:
                    fixtures = pd.read_csv(fixtures_url, encoding='latin-1')
                except pd.errors.ParserError:
                    try:
                        fixtures = pd.read_csv(fixtures_url, encoding='utf-8', on_bad_lines='skip')
                    except TypeError:
                        fixtures = pd.read_csv(fixtures_url, encoding='utf-8', error_bad_lines=False, warn_bad_lines=True)
                    except:
                        fixtures = pd.read_csv(fixtures_url, engine='python', on_bad_lines='skip')
            
            if fixtures is not None:
                fixtures.to_csv(os.path.join(season_dir, "fixtures.csv"), index=False, encoding='utf-8')
                print(f"  - Saved fixtures data: {fixtures.shape[0]} fixtures")
            else:
                print(f"  - Failed to parse fixtures data after trying multiple strategies")
        except Exception as e:
            print(f"  - Failed to collect fixtures data: {e}")
    
        return {
            "merged_gw": merged_gw if 'merged_gw' in locals() and merged_gw is not None else None,
            "players": players if 'players' in locals() and players is not None else None,
            "teams": teams if 'teams' in locals() and teams is not None else None,
            "fixtures": fixtures if 'fixtures' in locals() and fixtures is not None else None
        }
    
    def collect_all_seasons(self, seasons=None):
        """Collect data for all specified seasons or all available seasons"""
        if seasons is None:
            seasons = self.get_available_seasons()
        
        collected_data = {}
        for season in seasons:
            collected_data[season] = self.collect_season_data(season)
            
        return collected_data

    def get_latest_season(self):
        """Get the most recent season available"""
        seasons = self.get_available_seasons()
        return seasons[-1] if seasons else None
        
    def load_season_data(self, season):
        """Load data for a specific season if it exists locally"""
        season_dir = os.path.join(self.history_dir, season)
        
        if not os.path.exists(season_dir):
            print(f"No local data found for season {season}")
            return None
        
        merged_gw_path = os.path.join(season_dir, "merged_gw.csv")
        players_path = os.path.join(season_dir, "players_raw.csv")
        teams_path = os.path.join(season_dir, "teams.csv")
        fixtures_path = os.path.join(season_dir, "fixtures.csv")
        
        data = {}
        
        if os.path.exists(merged_gw_path):
            data["merged_gw"] = pd.read_csv(merged_gw_path)
    
        if os.path.exists(players_path):
            data["players"] = pd.read_csv(players_path)
        
        if os.path.exists(teams_path):
            data["teams"] = pd.read_csv(teams_path)
        
        if os.path.exists(fixtures_path):
            data["fixtures"] = pd.read_csv(fixtures_path)
        
        return data

class FPLDataProcessor:
    """Class to process FPL data into ML-ready format"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.historical_dir = os.path.join(data_dir, "historical")
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        self.features_dir = os.path.join(data_dir, "features")
        
        # Create directories if they don't exist
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)
        
        # Initialize encoders
        self.label_encoders = {}
        self.scalers = {}
        
    def load_and_combine_seasons(self, seasons=None):
        """Load and combine data from multiple seasons"""
        if seasons is None:
            seasons = [d for d in os.listdir(self.historical_dir) 
                      if os.path.isdir(os.path.join(self.historical_dir, d))]
        
        combined_data = {
            'gameweeks': [],
            'players': [],
            'teams': [],
            'fixtures': []
        }
        
        for season in seasons:
            season_dir = os.path.join(self.historical_dir, season)
            if not os.path.exists(season_dir):
                print(f"Season {season} data not found, skipping...")
                continue
                
            print(f"Loading data for season {season}...")
            
            # Load merged gameweek data
            merged_gw_path = os.path.join(season_dir, "merged_gw.csv")
            if os.path.exists(merged_gw_path):
                gw_data = pd.read_csv(merged_gw_path)
                gw_data['season'] = season
                combined_data['gameweeks'].append(gw_data)
            
            # Load players data
            players_path = os.path.join(season_dir, "players_raw.csv")
            if os.path.exists(players_path):
                players_data = pd.read_csv(players_path)
                players_data['season'] = season
                combined_data['players'].append(players_data)
            
            # Load teams data
            teams_path = os.path.join(season_dir, "teams.csv")
            if os.path.exists(teams_path):
                teams_data = pd.read_csv(teams_path)
                teams_data['season'] = season
                combined_data['teams'].append(teams_data)
            
            # Load fixtures data
            fixtures_path = os.path.join(season_dir, "fixtures.csv")
            if os.path.exists(fixtures_path):
                fixtures_data = pd.read_csv(fixtures_path)
                fixtures_data['season'] = season
                combined_data['fixtures'].append(fixtures_data)
        
        # Combine all seasons
        for key in combined_data:
            if combined_data[key]:
                combined_data[key] = pd.concat(combined_data[key], ignore_index=True)
            else:
                combined_data[key] = pd.DataFrame()
        
        return combined_data
    
    def create_player_features(self, gameweeks_df, players_df):
        """Create comprehensive player features for ML"""
        print("Creating player features...")
        
        # Ensure we have the required columns
        required_cols = ['element', 'GW', 'total_points', 'minutes', 'goals_scored', 
                        'assists', 'clean_sheets', 'goals_conceded', 'yellow_cards', 
                        'red_cards', 'saves', 'bonus', 'bps', 'influence', 'creativity', 'threat']
        
        missing_cols = [col for col in required_cols if col not in gameweeks_df.columns]
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols}")
            for col in missing_cols:
                gameweeks_df[col] = 0
        
        # Sort by player and gameweek
        gameweeks_df = gameweeks_df.sort_values(['element', 'GW'])
        
        features = []
        
        for player_id in gameweeks_df['element'].unique():
            player_data = gameweeks_df[gameweeks_df['element'] == player_id].copy()
            player_info = players_df[players_df['id'] == player_id].iloc[0] if len(players_df[players_df['id'] == player_id]) > 0 else None
            
            if player_info is None:
                continue
                
            for idx, row in player_data.iterrows():
                gw = row['GW']
                
                # Basic player info
                feature_row = {
                    'player_id': player_id,
                    'gameweek': gw,
                    'position': player_info.get('element_type', 0),
                    'team': row.get('team', 0),
                    'price': row.get('value', player_info.get('now_cost', 0)) / 10,  # Convert to millions
                    'selected_by_percent': player_info.get('selected_by_percent', 0),
                    
                    # Current gameweek performance (target variable)
                    'points_scored': row['total_points'],
                    'minutes_played': row['minutes'],
                    'goals_scored': row['goals_scored'],
                    'assists': row['assists'],
                    'clean_sheets': row['clean_sheets'],
                    'goals_conceded': row['goals_conceded'],
                    'yellow_cards': row['yellow_cards'],
                    'red_cards': row['red_cards'],
                    'saves': row['saves'],
                    'bonus': row['bonus'],
                    'bps': row['bps'],
                    'influence': row['influence'],
                    'creativity': row['creativity'],
                    'threat': row['threat'],
                }
                
                # Historical features (last 3, 5, 10 gameweeks)
                prev_data = player_data[player_data['GW'] < gw]
                
                if len(prev_data) > 0:
                    # Last 3 gameweeks
                    last_3 = prev_data.tail(3)
                    feature_row.update({
                        'avg_points_3gw': last_3['total_points'].mean(),
                        'avg_minutes_3gw': last_3['minutes'].mean(),
                        'avg_goals_3gw': last_3['goals_scored'].mean(),
                        'avg_assists_3gw': last_3['assists'].mean(),
                        'form_3gw': last_3['total_points'].sum(),
                    })
                    
                    # Last 5 gameweeks
                    last_5 = prev_data.tail(5)
                    feature_row.update({
                        'avg_points_5gw': last_5['total_points'].mean(),
                        'avg_minutes_5gw': last_5['minutes'].mean(),
                        'avg_goals_5gw': last_5['goals_scored'].mean(),
                        'avg_assists_5gw': last_5['assists'].mean(),
                        'form_5gw': last_5['total_points'].sum(),
                    })
                    
                    # Season totals
                    feature_row.update({
                        'total_points_season': prev_data['total_points'].sum(),
                        'total_minutes_season': prev_data['minutes'].sum(),
                        'total_goals_season': prev_data['goals_scored'].sum(),
                        'total_assists_season': prev_data['assists'].sum(),
                        'games_played_season': len(prev_data[prev_data['minutes'] > 0]),
                        'avg_points_per_game': prev_data[prev_data['minutes'] > 0]['total_points'].mean() if len(prev_data[prev_data['minutes'] > 0]) > 0 else 0,
                    })
                    
                    # Consistency metrics
                    if len(prev_data) > 1:
                        feature_row['points_std'] = prev_data['total_points'].std()
                        feature_row['minutes_consistency'] = prev_data['minutes'].std()
                    else:
                        feature_row['points_std'] = 0
                        feature_row['minutes_consistency'] = 0
                else:
                    # First gameweek - use defaults
                    for key in ['avg_points_3gw', 'avg_minutes_3gw', 'avg_goals_3gw', 'avg_assists_3gw', 'form_3gw',
                               'avg_points_5gw', 'avg_minutes_5gw', 'avg_goals_5gw', 'avg_assists_5gw', 'form_5gw',
                               'total_points_season', 'total_minutes_season', 'total_goals_season', 'total_assists_season',
                               'games_played_season', 'avg_points_per_game', 'points_std', 'minutes_consistency']:
                        feature_row[key] = 0
                
                features.append(feature_row)
        
        return pd.DataFrame(features)
    
    def create_team_features(self, teams_df, fixtures_df):
        """Create team-based features"""
        print("Creating team features...")
        
        team_features = []
        
        for _, team in teams_df.iterrows():
            team_id = team['id']
            
            # Team fixtures for difficulty analysis
            team_fixtures = fixtures_df[
                (fixtures_df['team_h'] == team_id) | (fixtures_df['team_a'] == team_id)
            ].copy()
            
            feature_row = {
                'team_id': team_id,
                'team_name': team['name'],
                'strength_overall_home': team.get('strength_overall_home', 1000),
                'strength_overall_away': team.get('strength_overall_away', 1000),
                'strength_attack_home': team.get('strength_attack_home', 1000),
                'strength_attack_away': team.get('strength_attack_away', 1000),
                'strength_defence_home': team.get('strength_defence_home', 1000),
                'strength_defence_away': team.get('strength_defence_away', 1000),
            }
            
            # Calculate average fixture difficulty
            if len(team_fixtures) > 0:
                home_fixtures = team_fixtures[team_fixtures['team_h'] == team_id]
                away_fixtures = team_fixtures[team_fixtures['team_a'] == team_id]
                
                feature_row['avg_home_difficulty'] = home_fixtures['team_h_difficulty'].mean() if len(home_fixtures) > 0 else 3
                feature_row['avg_away_difficulty'] = away_fixtures['team_a_difficulty'].mean() if len(away_fixtures) > 0 else 3
            else:
                feature_row['avg_home_difficulty'] = 3
                feature_row['avg_away_difficulty'] = 3
            
            team_features.append(feature_row)
        
        return pd.DataFrame(team_features)
    
    def create_fixture_features(self, fixtures_df, teams_df):
        """Create fixture-based features for upcoming gameweeks"""
        print("Creating fixture features...")
        
        if fixtures_df.empty:
            print("Warning: No fixtures data available")
            return pd.DataFrame(columns=['fixture_id', 'gameweek', 'home_team', 'away_team', 
                                       'home_difficulty', 'away_difficulty', 'home_attack_vs_away_defence',
                                       'away_attack_vs_home_defence', 'overall_strength_diff'])
        
        fixture_features = []
        
        for _, fixture in fixtures_df.iterrows():
            # For historical data, we want to include all fixtures, not skip finished ones
            # since we're doing historical analysis
            
            home_team = fixture.get('team_h', 0)
            away_team = fixture.get('team_a', 0)
            
            # Get team strengths
            home_team_info = teams_df[teams_df['id'] == home_team]
            away_team_info = teams_df[teams_df['id'] == away_team]
            
            if len(home_team_info) == 0 or len(away_team_info) == 0:
                # Use default values if team info is missing
                home_team_info = {'strength_attack_home': 1000, 'strength_defence_home': 1000, 'strength_overall_home': 1000}
                away_team_info = {'strength_attack_away': 1000, 'strength_defence_away': 1000, 'strength_overall_away': 1000}
            else:
                home_team_info = home_team_info.iloc[0]
                away_team_info = away_team_info.iloc[0]
            
            # Try different column names for gameweek/event
            gameweek = fixture.get('event', fixture.get('gameweek', fixture.get('GW', 0)))
            
            feature_row = {
                'fixture_id': fixture.get('id', 0),
                'gameweek': gameweek,
                'home_team': home_team,
                'away_team': away_team,
                'home_difficulty': fixture.get('team_h_difficulty', 3),
                'away_difficulty': fixture.get('team_a_difficulty', 3),
                
                # Team strength comparisons
                'home_attack_vs_away_defence': home_team_info.get('strength_attack_home', 1000) - away_team_info.get('strength_defence_away', 1000),
                'away_attack_vs_home_defence': away_team_info.get('strength_attack_away', 1000) - home_team_info.get('strength_defence_home', 1000),
                'overall_strength_diff': home_team_info.get('strength_overall_home', 1000) - away_team_info.get('strength_overall_away', 1000),
            }
            
            fixture_features.append(feature_row)
        
        return pd.DataFrame(fixture_features)
    
    def create_combined_dataset(self, seasons=None):
        """Create the final combined dataset for ML training"""
        print("Creating combined dataset for ML training...")
        
        # Load all data
        combined_data = self.load_and_combine_seasons(seasons)
        
        if combined_data['gameweeks'].empty:
            print("No gameweek data found!")
            return None
        
        # Create individual feature sets
        player_features = self.create_player_features(
            combined_data['gameweeks'], 
            combined_data['players']
        )
        
        team_features = self.create_team_features(
            combined_data['teams'],
            combined_data['fixtures']
        )
        
        # Merge player features with team features
        # Ensure data types match for the merge
        player_features['team'] = pd.to_numeric(player_features['team'], errors='coerce')
        team_features['team_id'] = pd.to_numeric(team_features['team_id'], errors='coerce')
        
        final_dataset = player_features.merge(
            team_features, 
            left_on='team', 
            right_on='team_id', 
            how='left'
        )
        
        # Add fixture difficulty for current gameweek
        fixture_features = self.create_fixture_features(
            combined_data['fixtures'],
            combined_data['teams']
        )
        
        # Check if fixture_features is empty or missing gameweek column
        if fixture_features.empty or 'gameweek' not in fixture_features.columns:
            print("Warning: No fixture features available or missing gameweek column")
            # Add default fixture difficulty columns
            final_dataset['home_difficulty'] = 3  # Default difficulty
            final_dataset['away_difficulty'] = 3
            final_dataset['is_home'] = False
        else:
            # For each player-gameweek, find their team's fixture
            final_dataset['home_difficulty'] = 3  # Default values
            final_dataset['away_difficulty'] = 3
            final_dataset['is_home'] = False
            
            for idx, row in final_dataset.iterrows():
                gw_fixtures = fixture_features[fixture_features['gameweek'] == row['gameweek']]
                
                # Find fixture where player's team is involved
                team_fixture = gw_fixtures[
                    (gw_fixtures['home_team'] == row['team']) | 
                    (gw_fixtures['away_team'] == row['team'])
                ]
                
                if len(team_fixture) > 0:
                    fixture = team_fixture.iloc[0]
                    if fixture['home_team'] == row['team']:
                        final_dataset.at[idx, 'home_difficulty'] = fixture['home_difficulty']
                        final_dataset.at[idx, 'is_home'] = True
                    else:
                        final_dataset.at[idx, 'away_difficulty'] = fixture['away_difficulty']
                        final_dataset.at[idx, 'is_home'] = False
        
        # Clean up and prepare final dataset
        final_dataset = final_dataset.drop(['team_id', 'team_name'], axis=1, errors='ignore')
        
        # Handle missing values
        numeric_columns = final_dataset.select_dtypes(include=[np.number]).columns
        final_dataset[numeric_columns] = final_dataset[numeric_columns].fillna(0)
        
        # Remove rows where we don't have enough historical data (first few gameweeks)
        final_dataset = final_dataset[final_dataset['gameweek'] > 1]  # Skip GW1 as no historical data
        
        print(f"Final dataset shape: {final_dataset.shape}")
        print(f"Features: {list(final_dataset.columns)}")
        
        return final_dataset
    
    def prepare_ml_datasets(self, final_dataset, target_columns=None):
        """Prepare datasets for different ML tasks"""
        if target_columns is None:
            target_columns = ['points_scored', 'minutes_played', 'goals_scored', 'assists']
        
        datasets = {}
        
        for target in target_columns:
            if target not in final_dataset.columns:
                print(f"Warning: Target column {target} not found in dataset")
                continue
            
            # Prepare features and target
            feature_columns = [col for col in final_dataset.columns 
                             if col not in ['player_id', 'gameweek'] + target_columns]
            
            X = final_dataset[feature_columns].copy()
            y = final_dataset[target].copy()
            
            # Encode categorical variables
            categorical_columns = X.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
                else:
                    X[col] = self.label_encoders[col].transform(X[col].astype(str))
            
            # Scale numerical features (including converted boolean features)
            # Convert boolean columns to numeric first so they're included in scaling
            for col in X.columns:
                if X[col].dtype == 'bool':
                    X[col] = X[col].astype(int)
            
            numerical_columns = X.select_dtypes(include=[np.number]).columns
            if f'{target}_scaler' not in self.scalers:
                self.scalers[f'{target}_scaler'] = StandardScaler()
                X[numerical_columns] = self.scalers[f'{target}_scaler'].fit_transform(X[numerical_columns])
            else:
                X[numerical_columns] = self.scalers[f'{target}_scaler'].transform(X[numerical_columns])
            
            datasets[target] = {
                'X': X[numerical_columns],  # Only use the scaled numerical columns
                'y': y,
                'feature_names': list(numerical_columns)  # Only numerical column names
            }
        
        return datasets
    
    def save_processed_data(self, final_dataset, datasets):
        """Save processed data for future use"""
        print("Saving processed data...")
        
        # Save raw processed dataset
        final_dataset.to_csv(
            os.path.join(self.processed_dir, "fpl_ml_dataset.csv"), 
            index=False
        )
        
        # Save individual ML datasets
        for target, data in datasets.items():
            # Save features and targets
            data['X'].to_csv(
                os.path.join(self.features_dir, f"features_{target}.csv"), 
                index=False
            )
            pd.Series(data['y']).to_csv(
                os.path.join(self.features_dir, f"target_{target}.csv"), 
                index=False
            )
            
            # Save feature names
            with open(os.path.join(self.features_dir, f"feature_names_{target}.json"), 'w') as f:
                json.dump(data['feature_names'], f)
        
        # Save encoders and scalers
        import pickle
        with open(os.path.join(self.processed_dir, "label_encoders.pkl"), 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        with open(os.path.join(self.processed_dir, "scalers.pkl"), 'wb') as f:
            pickle.dump(self.scalers, f)
        
        print(f"Data saved to {self.processed_dir} and {self.features_dir}")
    
    def process_all_data(self, seasons=None, target_columns=None):
        """Main method to process all data"""
        print("Starting complete data processing pipeline...")
        
        # Create combined dataset
        final_dataset = self.create_combined_dataset(seasons)
        
        if final_dataset is None:
            print("Failed to create dataset")
            return None
        
        # Prepare ML datasets
        datasets = self.prepare_ml_datasets(final_dataset, target_columns)
        
        # Save everything
        self.save_processed_data(final_dataset, datasets)
        
        print("Data processing complete!")
        return final_dataset, datasets
    
    def load_processed_data(self, target='points_scored'):
        """Load previously processed data"""
        try:
            # Load features and target
            X = pd.read_csv(os.path.join(self.features_dir, f"features_{target}.csv"))
            y = pd.read_csv(os.path.join(self.features_dir, f"target_{target}.csv")).iloc[:, 0]
            
            # Load feature names
            with open(os.path.join(self.features_dir, f"feature_names_{target}.json"), 'r') as f:
                feature_names = json.load(f)
            
            # Load encoders and scalers
            import pickle
            with open(os.path.join(self.processed_dir, "label_encoders.pkl"), 'rb') as f:
                self.label_encoders = pickle.load(f)
            
            with open(os.path.join(self.processed_dir, "scalers.pkl"), 'rb') as f:
                self.scalers = pickle.load(f)
            
            return X, y, feature_names
        
        except Exception as e:
            print(f"Error loading processed data: {e}")
            return None, None, None
    
    def get_data_summary(self):
        """Get a summary of the processed data"""
        try:
            dataset = pd.read_csv(os.path.join(self.processed_dir, "fpl_ml_dataset.csv"))
            
            summary = {
                'total_rows': len(dataset),
                'unique_players': dataset['player_id'].nunique(),
                'gameweeks_covered': dataset['gameweek'].nunique(),
                'seasons_covered': dataset['season'].nunique() if 'season' in dataset.columns else 'Unknown',
                'features_count': len(dataset.columns) - 4,  # Exclude player_id, gameweek, season, target
                'avg_points_per_gw': dataset['points_scored'].mean(),
                'positions_distribution': dataset['position'].value_counts().to_dict(),
            }
            
            return summary
        
        except Exception as e:
            print(f"Error generating summary: {e}")
            return None