import os
import json
import pandas as pd
from datetime import datetime

def create_api_compatible_data(season):
    """
    Convert historical data format to match the API format expected by the rest of the system
    
    Parameters:
    -----------
    season : str
        Season to convert (e.g., '2023-24')
    """
    # Paths
    history_dir = os.path.join("data", "historical", season)
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    # Timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Season identifier to include in filenames
    season_id = season.replace('-', '')
    
    # 1. Create bootstrap_static equivalent
    if os.path.exists(os.path.join(history_dir, "players_raw.csv")) and \
       os.path.exists(os.path.join(history_dir, "teams.csv")):
        # Load player and team data
        players_df = pd.read_csv(os.path.join(history_dir, "players_raw.csv"))
        teams_df = pd.read_csv(os.path.join(history_dir, "teams.csv"))
        # Create bootstrap structure
        bootstrap = {
            "elements": players_df.to_dict('records'),
            "teams": teams_df.to_dict('records'),
            "events": []  # Would need to be populated with gameweek data
        }
        # Save as JSON
        with open(os.path.join(raw_dir, f"bootstrap_static_{season_id}_{timestamp}.json"), 'w') as f:
            json.dump(bootstrap, f)
        print(f"  - Created bootstrap_static with {len(players_df)} players and {len(teams_df)} teams")
    # 2. Create gameweek data
    if os.path.exists(os.path.join(history_dir, "merged_gw.csv")):
        gw_df = pd.read_csv(os.path.join(history_dir, "merged_gw.csv"))
        # Group by gameweek
        gw_column = 'GW' if 'GW' in gw_df.columns else 'round'
        for gw, group in gw_df.groupby(gw_column):
            gw_data = {
                "elements": group.to_dict('records')
            }
            # Save as JSON
            with open(os.path.join(raw_dir, f"event_{gw}_{season_id}_{timestamp}.json"), 'w') as f:
                json.dump(gw_data, f)
        print(f"  - Created data for {len(gw_df[gw_column].unique())} gameweeks")
    return True