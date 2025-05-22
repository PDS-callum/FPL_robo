import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

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
        
    def create_player_features(self):
        """Create features for each player based on historical performance"""
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
        
        # Save processed data
        players_df.to_csv(os.path.join(self.processed_dir, "players_features.csv"), index=False)
        return players_df
    
    def create_fixtures_features(self):
        """Process fixtures data to create features"""
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
        
        # Save processed data
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

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process FPL data for training')
    parser.add_argument('--lookback', type=int, default=3, help='Number of gameweeks to use for input features')
    args = parser.parse_args()

    processor = FPLDataProcessor(lookback=args.lookback)
    processor.process_all_data()