import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from .history_data_collector import FPLHistoricalDataCollector

class FPLDataProcessor:
    def __init__(self, data_dir="data", cutoff_gw=None, lookback=3):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        self.cutoff_gw = cutoff_gw
        self.lookback = lookback
        
    def load_latest_data(self, prefix):
        """Load the most recent data file with the given prefix"""
        files = [f for f in os.listdir(self.raw_dir) if f.startswith(prefix)]
        if not files:
            raise FileNotFoundError(f"No files found with prefix {prefix}")
        
        latest_file = sorted(files)[-1]
        filepath = os.path.join(self.raw_dir, latest_file)
        
        with open(filepath, 'r') as f:
            return json.load(f)
        
    def create_player_features(self, cutoff_gw=None):
        """
        Create features for all players based on their historical performance
        Include team strength and fixture difficulty metrics
        """
        bootstrap = self.load_latest_data("bootstrap_static")
        players = bootstrap["elements"]
        teams = {team["id"]: team["name"] for team in bootstrap["teams"]}
        
        # Convert players to DataFrame
        players_df = pd.DataFrame(players)
        
        # Add team name
        players_df["team_name"] = players_df["team"].map(teams)
        
        # Calculate points per minute
        players_df["points_per_minute"] = players_df.apply(
            lambda x: x["total_points"] / x["minutes"] if x["minutes"] > 0 else 0, axis=1
        )
        
        # Calculate value (points per cost)
        players_df["value"] = players_df["total_points"] / (players_df["now_cost"] / 10)
        
        # Form as numeric
        players_df["form_float"] = players_df["form"].astype(float)
        
        # Get team data and build team strength metrics
        teams_df = pd.DataFrame(bootstrap["teams"])
        
        # Add team strength indicators
        # Use the FPL team strength ratings (attack/defense home/away)
        team_metrics = {}
        for _, team in teams_df.iterrows():
            team_id = team['id']
            team_metrics[team_id] = {
                'strength': team['strength'],
                'strength_attack_home': team.get('strength_attack_home', team['strength']),
                'strength_attack_away': team.get('strength_attack_away', team['strength']),
                'strength_defence_home': team.get('strength_defence_home', team['strength']),
                'strength_defence_away': team.get('strength_defence_away', team['strength']),
            }
        
        # Try to get fixtures data to determine opponents
        try:
            fixtures_df = pd.DataFrame(self.load_latest_data("fixtures"))
            
            # Create fixtures lookup for each team and gameweek
            fixtures_by_team_gw = {}
            for _, fixture in fixtures_df.iterrows():
                gameweek = fixture.get('event')
                if gameweek is None or pd.isna(gameweek):
                    continue
                    
                gameweek = int(gameweek)
                home_team = fixture['team_h']
                away_team = fixture['team_a']
                
                # Store fixture info for home team
                if home_team not in fixtures_by_team_gw:
                    fixtures_by_team_gw[home_team] = {}
                fixtures_by_team_gw[home_team][gameweek] = {
                    'opponent': away_team,
                    'is_home': True,
                    'difficulty': fixture.get('team_h_difficulty', 3),  # Default to medium if not available
                }
                
                # Store fixture info for away team
                if away_team not in fixtures_by_team_gw:
                    fixtures_by_team_gw[away_team] = {}
                fixtures_by_team_gw[away_team][gameweek] = {
                    'opponent': home_team,
                    'is_home': False,
                    'difficulty': fixture.get('team_a_difficulty', 3),  # Default to medium if not available
                }
            
            # Check if player_gw_data is available
            if 'player_gw_data' in locals() and 'player_id_to_info' in locals():
                # Process player data and add team/fixture features
                for player_id, player_data in player_gw_data.items():
                    player_info = player_id_to_info[player_id]
                    team_id = player_info['team']
                    position = player_info['element_type']
                    
                    # Add fixture difficulty features for each gameweek
                    for gw, gw_stats in player_data.items():
                        # Skip if we don't have fixture data for this team/gameweek
                        if team_id not in fixtures_by_team_gw or gw not in fixtures_by_team_gw[team_id]:
                            continue
                            
                        fixture = fixtures_by_team_gw[team_id][gw]
                        opponent_id = fixture['opponent']
                        is_home = fixture['is_home']
                        
                        # Get opponent strength metrics based on whether player is home/away
                        if opponent_id in team_metrics:
                            if is_home:
                                # Player is home, opponent is away
                                gw_stats['opponent_attack_strength'] = team_metrics[opponent_id]['strength_attack_away']
                                gw_stats['opponent_defense_strength'] = team_metrics[opponent_id]['strength_defence_away']
                            else:
                                # Player is away, opponent is home
                                gw_stats['opponent_attack_strength'] = team_metrics[opponent_id]['strength_attack_home']
                                gw_stats['opponent_defense_strength'] = team_metrics[opponent_id]['strength_defence_home']
                        
                        gw_stats['opponent_overall_strength'] = team_metrics[opponent_id]['strength']
                    
                    # Add FPL's own difficulty rating
                    gw_stats['fixture_difficulty'] = fixture['difficulty']
                    
                    # Add home/away indicator (1 for home, 0 for away)
                    gw_stats['is_home'] = 1 if is_home else 0
        except FileNotFoundError:
            print("No fixtures data found. Skipping fixture-related feature generation.")
            # Add placeholder fixture difficulty for all players
            players_df['fixture_difficulty'] = 3.0  # Medium difficulty as default
            players_df['is_home'] = 0.5  # Neutral home/away status
        
        # Save processed data
        players_df.to_csv(os.path.join(self.processed_dir, "players_features.csv"), index=False)
        return players_df
    
    def create_fixtures_features(self):
        """Process fixtures data to create features"""
        try:
            fixtures = self.load_latest_data("fixtures")
            bootstrap = self.load_latest_data("bootstrap_static")
            teams = {team["id"]: team["name"] for team in bootstrap["teams"]}
            
            # Convert to DataFrame
            fixtures_df = pd.DataFrame(fixtures)
            
            # Add team names
            fixtures_df["team_h_name"] = fixtures_df["team_h"].map(teams)
            fixtures_df["team_a_name"] = fixtures_df["team_a"].map(teams)
            
            # Convert kickoff time to datetime
            fixtures_df["kickoff_datetime"] = pd.to_datetime(fixtures_df["kickoff_time"])
            
            # Add days until match feature
            today = pd.Timestamp.now()
            fixtures_df["days_until_match"] = (fixtures_df["kickoff_datetime"] - today).dt.days
            
            # Add calculated difficulty metric based on team strengths from bootstrap data
            team_strengths = {team["id"]: team["strength"] for team in bootstrap["teams"]}
            
            # Calculate difficulty based on difference in team strengths
            fixtures_df["team_h_strength"] = fixtures_df["team_h"].map(team_strengths)
            fixtures_df["team_a_strength"] = fixtures_df["team_a"].map(team_strengths)
            fixtures_df["strength_diff"] = fixtures_df["team_h_strength"] - fixtures_df["team_a_strength"]
            
            # Calculate home advantage (can be adjusted based on analysis)
            home_advantage = 10
            fixtures_df["adjusted_diff"] = fixtures_df["strength_diff"] + home_advantage
            
            # Generate simple win probability (logistic function of adjusted strength diff)
            fixtures_df["home_win_prob"] = 1 / (1 + np.exp(-fixtures_df["adjusted_diff"] / 100))
            
            # Save processed data
            fixtures_df.to_csv(os.path.join(self.processed_dir, "fixtures_features.csv"), index=False)
            
            # Also save a version sorted by gameweek for easier reference
            fixtures_by_gw = fixtures_df.sort_values("event")
            fixtures_by_gw.to_csv(os.path.join(self.processed_dir, "fixtures_by_gameweek.csv"), index=False)
            
            return fixtures_df
            
        except FileNotFoundError:
            print("No fixtures data found. Creating empty fixtures DataFrame.")
            # Create an empty fixtures DataFrame with expected columns
            fixtures_df = pd.DataFrame(columns=[
                "id", "event", "team_h", "team_a", "team_h_name", "team_a_name",
                "team_h_difficulty", "team_a_difficulty"
            ])
            fixtures_df.to_csv(os.path.join(self.processed_dir, "fixtures_features.csv"), index=False)
            return fixtures_df
    
    def create_player_history_features(self):
        """Process player history to create features for ML model"""
        try:
            player_histories = self.load_latest_data("player_histories")
        except:
            # If single file is too large, try loading individual player files
            bootstrap = self.load_latest_data("bootstrap_static")
            player_ids = [player["id"] for player in bootstrap["elements"]]
            player_histories = {}
            for player_id in player_ids:
                try:
                    history = self.load_latest_data(f"player_{player_id}")
                    player_histories[player_id] = history
                except:
                    continue
        
        players_data = []
        
        for player_id, history in player_histories.items():
            if "history" not in history or not history["history"]:
                continue
                
            player_history = pd.DataFrame(history["history"])
            
            # Calculate rolling averages
            if len(player_history) >= 3:
                player_history["rolling_pts_3"] = player_history["total_points"].rolling(3).mean()
                player_history["rolling_mins_3"] = player_history["minutes"].rolling(3).mean()
            else:
                player_history["rolling_pts_3"] = player_history["total_points"]
                player_history["rolling_mins_3"] = player_history["minutes"]
            
            # Fill NaN values
            player_history = player_history.fillna(0)
            
            # Add to list
            for _, row in player_history.iterrows():
                players_data.append({
                    "player_id": player_id,
                    "gameweek": row["round"],
                    "total_points": row["total_points"],
                    "minutes": row["minutes"],
                    "goals_scored": row.get("goals_scored", 0),
                    "assists": row.get("assists", 0),
                    "clean_sheets": row.get("clean_sheets", 0),
                    "goals_conceded": row.get("goals_conceded", 0),
                    "bonus": row.get("bonus", 0),
                    "bps": row.get("bps", 0),
                    "influence": row.get("influence", 0),
                    "creativity": row.get("creativity", 0),
                    "threat": row.get("threat", 0),
                    "ict_index": row.get("ict_index", 0),
                    "rolling_pts_3": row["rolling_pts_3"],
                    "rolling_mins_3": row["rolling_mins_3"],
                    "opponent_team": row.get("opponent_team", 0),
                    "was_home": row.get("was_home", False)
                })
        
        # Create DataFrame
        player_history_df = pd.DataFrame(players_data)
        
        # Save processed data
        player_history_df.to_csv(os.path.join(self.processed_dir, "player_history_features.csv"), index=False)
        return player_history_df
    
    def prepare_training_data(self, lookback=3, prediction_gw=None):
        """
        Prepare training data for CNN model with lookback period
        
        Parameters:
        -----------
        lookback : int
            Number of gameweeks to use for input features
        prediction_gw : int
            If provided, prepare data for predicting this gameweek
        
        Returns:
        --------
        X : numpy array
            Input features for CNN with shape (n_samples, lookback, n_features)
        y : numpy array
            Target values (points scored in next gameweek)
        players : list
            List of player IDs corresponding to each sample
        """
        # Load processed data
        player_history = pd.read_csv(os.path.join(self.processed_dir, "player_history_features.csv"))
        players_df = pd.read_csv(os.path.join(self.processed_dir, "players_features.csv"))
        
        # Apply cutoff filter if specified
        if self.cutoff_gw is not None:
            player_history = player_history[player_history['gameweek'] <= self.cutoff_gw]
            
        # Feature columns to use
        feature_cols = [
            'minutes', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
            'bonus', 'bps', 'influence', 'creativity', 'threat', 'ict_index',
            'rolling_pts_3', 'rolling_mins_3', 'was_home'
        ]
        
        # Normalize features
        scaler = StandardScaler()
        player_history[feature_cols] = scaler.fit_transform(player_history[feature_cols].fillna(0))
        
        # Prepare data with lookback
        X_data = []
        y_data = []
        player_ids = []
        
        # Get unique player IDs and gameweeks
        unique_players = player_history['player_id'].unique()
        
        for player_id in unique_players:
            player_data = player_history[player_history['player_id'] == player_id].sort_values('gameweek')
            
            if len(player_data) <= lookback:
                continue
                
            for i in range(len(player_data) - lookback):
                features = player_data.iloc[i:i+lookback][feature_cols].values
                next_gw = player_data.iloc[i+lookback]
                
                X_data.append(features)
                y_data.append(next_gw['total_points'])
                player_ids.append(player_id)
        
        # Convert to numpy arrays
        X = np.array(X_data)
        y = np.array(y_data)
        
        # If preparing for prediction
        if prediction_gw is not None:
            X_pred = []
            pred_player_ids = []
            
            for player_id in unique_players:
                player_data = player_history[player_history['player_id'] == player_id].sort_values('gameweek')
                
                if len(player_data) < lookback:
                    continue
                
                latest_data = player_data.iloc[-lookback:][feature_cols].values
                if len(latest_data) == lookback:
                    X_pred.append(latest_data)
                    pred_player_ids.append(player_id)
            
            X_pred = np.array(X_pred)
            return X, y, player_ids, X_pred, pred_player_ids
        
        return X, y, player_ids
    
    def create_player_position_encoding(self):
        """Create one-hot encoding for player positions"""
        players_df = pd.read_csv(os.path.join(self.processed_dir, "players_features.csv"))
        
        # Create position dummies (GK=1, DEF=2, MID=3, FWD=4)
        position_dummies = pd.get_dummies(players_df['element_type'], prefix='pos')
        players_with_pos = pd.concat([players_df, position_dummies], axis=1)
        
        # Save to file
        players_with_pos.to_csv(os.path.join(self.processed_dir, "players_with_position.csv"), index=False)
        return players_with_pos
    
    def process_all_data(self):
        """Process all data and prepare for training"""
        print("Processing player features...")
        self.create_player_features()
        
        print("Processing fixtures data...")
        self.create_fixtures_features()
        
        print("Processing player history...")
        self.create_player_history_features()
        
        print("Creating position encoding...")
        self.create_player_position_encoding()
        
        print("Preparing training data...")
        X, y, player_ids = self.prepare_training_data(lookback=self.lookback)
        
        print(f"Data processing complete! Created {len(X)} training samples.")
        return X, y, player_ids

class MultiSeasonDataProcessor:
    def __init__(self, data_dir="data", seasons=None, lookback=3, use_historical=True):
        self.data_dir = data_dir
        self.processed_dir = os.path.join(data_dir, "processed")
        self.history_dir = os.path.join(data_dir, "historical")
        os.makedirs(self.processed_dir, exist_ok=True)
        
        self.use_historical = use_historical
        self.lookback = lookback
        
        # Initialize collectors
        self.current_collector = FPLDataProcessor()
        self.history_collector = FPLHistoricalDataCollector()
        
        # Set seasons to process
        if seasons is None and use_historical:
            self.seasons = self.history_collector.get_available_seasons()[-2:]  # Use last 2 seasons by default
        else:
            self.seasons = seasons if seasons else []
            
    def prepare_historical_player_data(self):
        """Prepare and merge player data from multiple seasons"""
        all_player_data = []
        
        for season in self.seasons:
            season_data = self.history_collector.load_season_data(season)
            if not season_data or "merged_gw" not in season_data:
                print(f"Missing merged gameweek data for season {season}")
                continue
                
            merged_gw = season_data["merged_gw"]
            
            # Add season identifier
            merged_gw['season'] = season
            
            # Add to collection
            all_player_data.append(merged_gw)
            
        # Concatenate all data
        if not all_player_data:
            print("No historical player data available")
            return None
            
        combined_data = pd.concat(all_player_data, ignore_index=True)
        
        # Save processed data
        combined_data.to_csv(os.path.join(self.processed_dir, "historical_player_data.csv"), index=False)
        return combined_data
        
    def prepare_multi_season_training_data(self, prediction_gw=None, current_season=None):
        """Prepare training data using both historical and current season data"""
        print("Preparing multi-season training data...")
        
        # First, get raw historical data
        historical_data = self.prepare_historical_player_data()
        
        if historical_data is None:
            print("No historical data available")
            return None, None, None
        
        # Feature columns to use (must match across seasons)
        feature_cols = [
            'minutes', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
            'bonus', 'bps', 'influence', 'creativity', 'threat', 'ict_index',
            'was_home'
        ]
        
        # Ensure all required columns are available
        for col in feature_cols:
            if col not in historical_data.columns:
                print(f"Missing required column: {col}")
                # Try to handle missing columns with reasonable defaults
                if col in ['influence', 'creativity', 'threat', 'ict_index']:
                    # These metrics were added in later seasons
                    historical_data[col] = 0
                elif col == 'was_home':
                    # Convert 'H'/'A' format to boolean if needed
                    if 'was_home' not in historical_data.columns and 'H/A' in historical_data.columns:
                        historical_data['was_home'] = (historical_data['H/A'] == 'H').astype(int)
                    else:
                        historical_data[col] = 0
                else:
                    historical_data[col] = 0
        
        # Add rolling features
        print("Calculating rolling averages...")
        historical_data['season_gw'] = historical_data['GW'] if 'GW' in historical_data.columns else historical_data['round']
        
        # Group by player and season for rolling calculations
        historical_data['player_season'] = historical_data['name'] + '_' + historical_data['season']
        
        # Calculate rolling averages within each player-season group
        all_processed_data = []
        
        for player_season, group in historical_data.groupby('player_season'):
            # Sort by gameweek
            group = group.sort_values('season_gw')
            
            # Calculate rolling averages for points and minutes
            group['rolling_pts_3'] = group['total_points'].rolling(3, min_periods=1).mean().fillna(0)
            group['rolling_mins_3'] = group['minutes'].rolling(3, min_periods=1).mean().fillna(0)
            
            all_processed_data.append(group)
        
        # Combine all processed data
        processed_data = pd.concat(all_processed_data, ignore_index=True)
        
        # Make sure all feature columns are numeric
        for col in feature_cols + ['rolling_pts_3', 'rolling_mins_3']:
            if col in processed_data.columns:
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce').fillna(0)
        
        # Convert categorical was_home to numeric if needed
        if 'was_home' in processed_data.columns and processed_data['was_home'].dtype == 'object':
            processed_data['was_home'] = (processed_data['was_home'] == 'True').astype(int)
        
        # Normalize features
        print("Normalizing features...")
        from sklearn.preprocessing import StandardScaler
        
        combined_features = feature_cols + ['rolling_pts_3', 'rolling_mins_3']
        scaler = StandardScaler()
        processed_data[combined_features] = scaler.fit_transform(processed_data[combined_features])
        
        # Prepare data with lookback
        print("Creating training sequences...")
        X_data = []
        y_data = []
        player_ids = []
        
        # For each player season, create training sequences
        for player_season, group in processed_data.groupby('player_season'):
            # Sort by gameweek
            group = group.sort_values('season_gw')
            
            # Skip if we don't have enough gameweeks
            if len(group) <= self.lookback:
                continue
            
            # Create sequences
            for i in range(len(group) - self.lookback):
                features = group.iloc[i:i+self.lookback][combined_features].values
                next_gw = group.iloc[i+self.lookback]
                
                X_data.append(features)
                y_data.append(next_gw['total_points'])
                
                # Use player name as ID for historical data
                player_name = player_season.split('_')[0]
                player_ids.append(player_name)
        
        # Convert to numpy arrays
        X = np.array(X_data)
        y = np.array(y_data)
        
        print(f"Created {len(X)} training samples from historical data")
        
        # If we need to include current season data as well
        if current_season:
            print("Adding current season data...")
            current_data = self.current_collector.prepare_training_data(
                lookback=self.lookback, 
                prediction_gw=prediction_gw
            )
            
            if current_data and len(current_data) == 3:
                X_current, y_current, player_ids_current = current_data
                
                # Combine with historical data
                X = np.concatenate([X, X_current])
                y = np.concatenate([y, y_current])
                player_ids = player_ids + player_ids_current
                
                print(f"Added {len(X_current)} samples from current season")
        
        return X, y, player_ids
    
    def create_player_mappings_across_seasons(self):
        """Create mappings to track players across different seasons"""
        all_players = {}
        
        # For each season, collect player data
        for season in self.seasons:
            season_data = self.history_collector.load_season_data(season)
            if not season_data or "players" not in season_data:
                continue
                
            players = season_data["players"]
            
            # Use name as a key for matching players across seasons
            for _, player in players.iterrows():
                name = player.get('first_name', '') + ' ' + player.get('second_name', '')
                player_id = player.get('id')
                
                if name not in all_players:
                    all_players[name] = {
                        'seasons': {},
                        'current_id': None
                    }
                    
                # Add this season's ID
                all_players[name]['seasons'][season] = player_id
                
        # Save the mapping
        with open(os.path.join(self.processed_dir, "player_mappings.json"), 'w') as f:
            json.dump(all_players, f, indent=2)
            
        return all_players

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process FPL data for training')
    parser.add_argument('--lookback', type=int, default=3, help='Number of gameweeks to use for input features')
    args = parser.parse_args()

    processor = FPLDataProcessor(lookback=args.lookback)
    processor.process_all_data()