import os
import pandas as pd
import requests
from datetime import datetime
import json

class FPLHistoricalDataCollector:
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
            # Try first with utf-8 encoding
            try:
                merged_gw = pd.read_csv(gw_url, encoding='utf-8')
            except UnicodeDecodeError:
                # If utf-8 fails, try with latin-1 which is more permissive
                merged_gw = pd.read_csv(gw_url, encoding='latin-1')
            
            merged_gw.to_csv(os.path.join(season_dir, "merged_gw.csv"), index=False, encoding='utf-8')
            print(f"  - Saved merged gameweek data: {merged_gw.shape[0]} rows")
        except Exception as e:
            print(f"  - Failed to collect merged gameweek data: {e}")
        
        # Collect players data
        players_url = f"{self.base_url}/{season}/players_raw.csv"
        try:
            # Try first with utf-8 encoding
            try:
                players = pd.read_csv(players_url, encoding='utf-8')
            except UnicodeDecodeError:
                # If utf-8 fails, try with latin-1 which is more permissive
                players = pd.read_csv(players_url, encoding='latin-1')
                
            players.to_csv(os.path.join(season_dir, "players_raw.csv"), index=False, encoding='utf-8')
            print(f"  - Saved players data: {players.shape[0]} players")
        except Exception as e:
            print(f"  - Failed to collect players data: {e}")
        
        # Collect teams data
        teams_url = f"{self.base_url}/{season}/teams.csv"
        try:
            # Try first with utf-8 encoding
            try:
                teams = pd.read_csv(teams_url, encoding='utf-8')
            except UnicodeDecodeError:
                # If utf-8 fails, try with latin-1 which is more permissive
                teams = pd.read_csv(teams_url, encoding='latin-1')
                
            teams.to_csv(os.path.join(season_dir, "teams.csv"), index=False, encoding='utf-8')
            print(f"  - Saved teams data: {teams.shape[0]} teams")
        except Exception as e:
            print(f"  - Failed to collect teams data: {e}")
            
        return {
            "merged_gw": merged_gw if 'merged_gw' in locals() else None,
            "players": players if 'players' in locals() else None,
            "teams": teams if 'teams' in locals() else None
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
        
        data = {}
        
        if os.path.exists(merged_gw_path):
            data["merged_gw"] = pd.read_csv(merged_gw_path)
        
        if os.path.exists(players_path):
            data["players"] = pd.read_csv(players_path)
            
        if os.path.exists(teams_path):
            data["teams"] = pd.read_csv(teams_path)
            
        return data