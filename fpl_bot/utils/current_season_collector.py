import requests
import pandas as pd
import json
import os
from datetime import datetime, timezone

class FPLCurrentSeasonCollector:
    """
    Collector for current season FPL data using the official API
    """
    
    def __init__(self, data_dir="data"):
        self.base_url = "https://fantasy.premierleague.com/api/"
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.current_season_dir = os.path.join(data_dir, "current_season")
        
        # Create directories
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.current_season_dir, exist_ok=True)
    
    def get_bootstrap_static(self):
        """Get general game information, players, teams, and gameweeks"""
        url = f"{self.base_url}bootstrap-static/"
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        
        # Save to raw directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bootstrap_static_{timestamp}.json"
        filepath = os.path.join(self.raw_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
        
        print(f"Saved bootstrap static data to {filepath}")
        return data
    
    def get_current_gameweek(self):
        """Get information about the current gameweek"""
        bootstrap = self.get_bootstrap_static()
        
        # Find current gameweek
        current_gw = None
        next_gw = None
        
        for event in bootstrap['events']:
            if event['is_current']:
                current_gw = event
            if event['is_next']:
                next_gw = event
        
        # If no current gameweek, use next
        if current_gw is None and next_gw is not None:
            current_gw = next_gw
        
        if current_gw is None:
            # Find the latest finished gameweek
            finished_gws = [gw for gw in bootstrap['events'] if gw['finished']]
            if finished_gws:
                current_gw = max(finished_gws, key=lambda x: x['id'])
        
        return current_gw, bootstrap
    
    def get_gameweek_live_data(self, gameweek_id):
        """Get live data for a specific gameweek"""
        url = f"{self.base_url}event/{gameweek_id}/live/"
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        
        # Save data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gameweek_{gameweek_id}_live_{timestamp}.json"
        filepath = os.path.join(self.current_season_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
        
        print(f"Saved gameweek {gameweek_id} live data to {filepath}")
        return data
    
    def get_fixtures(self):
        """Get all fixtures for the season"""
        url = f"{self.base_url}fixtures/"
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        
        # Save fixtures
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fixtures_{timestamp}.json"
        filepath = os.path.join(self.current_season_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
        
        print(f"Saved fixtures data to {filepath}")
        return data
    
    def collect_current_season_data(self):
        """
        Collect all current season data
        
        Returns:
        --------
        dict with collected data
        """
        print("Collecting current season FPL data...")
        
        # Get bootstrap data (players, teams, gameweeks)
        bootstrap = self.get_bootstrap_static()
        
        # Get fixtures
        fixtures = self.get_fixtures()
        
        # Get live data for completed gameweeks
        completed_gameweeks = {}
        for event in bootstrap['events']:
            if event['finished']:
                gw_id = event['id']
                try:
                    live_data = self.get_gameweek_live_data(gw_id)
                    completed_gameweeks[gw_id] = live_data
                    print(f"Collected live data for gameweek {gw_id}")
                except Exception as e:
                    print(f"Failed to collect gameweek {gw_id}: {e}")
        
        return {
            'bootstrap': bootstrap,
            'fixtures': fixtures,
            'gameweeks': completed_gameweeks
        }
    
    def convert_to_training_format(self, current_season_data):
        """
        Convert current season API data to the same format as historical data
        for training data integration
        
        Returns:
        --------
        dict with DataFrames in historical format
        """
        bootstrap = current_season_data['bootstrap']
        gameweeks_data = current_season_data['gameweeks']
        fixtures = current_season_data['fixtures']
        
        # Convert players data
        players_df = pd.DataFrame(bootstrap['elements'])
        
        # Convert teams data
        teams_df = pd.DataFrame(bootstrap['teams'])
        
        # Convert fixtures data
        fixtures_df = pd.DataFrame(fixtures)
        
        # Convert gameweek data to merged format
        merged_gw_data = []
        
        for gw_id, gw_data in gameweeks_data.items():
            for element_data in gw_data['elements']:
                player_stats = element_data['stats']
                
                # Create row in merged_gw format
                row = {
                    'element': element_data['id'],
                    'GW': gw_id,
                    'total_points': player_stats['total_points'],
                    'minutes': player_stats['minutes'],
                    'goals_scored': player_stats['goals_scored'],
                    'assists': player_stats['assists'],
                    'clean_sheets': player_stats['clean_sheets'],
                    'goals_conceded': player_stats['goals_conceded'],
                    'own_goals': player_stats['own_goals'],
                    'penalties_saved': player_stats['penalties_saved'],
                    'penalties_missed': player_stats['penalties_missed'],
                    'yellow_cards': player_stats['yellow_cards'],
                    'red_cards': player_stats['red_cards'],
                    'saves': player_stats['saves'],
                    'bonus': player_stats['bonus'],
                    'bps': player_stats['bps'],
                    'influence': float(player_stats['influence']),
                    'creativity': float(player_stats['creativity']),
                    'threat': float(player_stats['threat']),
                    'ict_index': float(player_stats['ict_index']),
                    'value': 0,  # Will be filled from bootstrap data
                    'selected': 0,  # Will be filled from bootstrap data
                }
                
                # Get player info from bootstrap for value and selection
                player_info = next((p for p in bootstrap['elements'] if p['id'] == element_data['id']), None)
                if player_info:
                    row['value'] = player_info['now_cost']
                    row['selected'] = player_info['selected_by_percent']
                    row['team'] = player_info['team']
                
                merged_gw_data.append(row)
        
        merged_gw_df = pd.DataFrame(merged_gw_data)
        
        # Save converted data to current season directory
        season_id = f"2024-25"  # Current season
        current_season_path = os.path.join(self.data_dir, "historical", season_id)
        os.makedirs(current_season_path, exist_ok=True)
        
        # Save in historical format for integration
        merged_gw_df.to_csv(os.path.join(current_season_path, "merged_gw.csv"), index=False)
        players_df.to_csv(os.path.join(current_season_path, "players_raw.csv"), index=False)
        teams_df.to_csv(os.path.join(current_season_path, "teams.csv"), index=False)
        fixtures_df.to_csv(os.path.join(current_season_path, "fixtures.csv"), index=False)
        
        print(f"Converted current season data saved to {current_season_path}")
        
        return {
            'merged_gw': merged_gw_df,
            'players': players_df,
            'teams': teams_df,
            'fixtures': fixtures_df
        }
    
    def update_training_data(self):
        """
        Update training data with latest current season data
        
        Returns:
        --------
        dict with updated data ready for training
        """
        # Collect current season data
        current_data = self.collect_current_season_data()
        
        # Convert to training format
        converted_data = self.convert_to_training_format(current_data)
        
        print("Current season data updated and ready for training integration")
        return converted_data
