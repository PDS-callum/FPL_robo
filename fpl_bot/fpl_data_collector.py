import os
import requests
import pandas as pd
import numpy as np
import json
from datetime import datetime
from .utils.general_utils import get_data

class FPLDataCollector:
    def __init__(self, data_dir="data"):
        self.base_url = "https://fantasy.premierleague.com/api/"
        self.data_dir = data_dir
        os.makedirs(os.path.join(data_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "processed"), exist_ok=True)
        
    def collect_bootstrap_static(self):
        """Collect general FPL data including players, teams, and rules"""
        data = get_data(f"{self.base_url}bootstrap-static/")
        self._save_data(data, "bootstrap_static")
        return data
    
    def collect_fixtures(self):
        """Collect all fixtures for the season"""
        data = get_data(f"{self.base_url}fixtures/")
        self._save_data(data, "fixtures")
        return data
    
    def collect_player_history(self):
        """Collect detailed history for all players"""
        bootstrap = self.collect_bootstrap_static()
        players_data = {}
        
        print(f"Collecting data for {len(bootstrap['elements'])} players...")
        for i, player in enumerate(bootstrap['elements']):
            player_id = player['id']
            player_data = get_data(f"{self.base_url}element-summary/{player_id}/")
            players_data[player_id] = player_data
            
            if (i + 1) % 50 == 0 or i + 1 == len(bootstrap['elements']):
                print(f"Processed {i + 1}/{len(bootstrap['elements'])} players")
        
        self._save_data(players_data, "player_histories")
        return players_data
    
    def collect_gameweek_data(self, gameweek):
        """Collect data for a specific gameweek"""
        data = get_data(f"{self.base_url}event/{gameweek}/live/")
        self._save_data(data, f"gameweek_{gameweek}")
        return data
    
    def collect_all_gameweeks(self):
        """Collect data for all completed gameweeks"""
        bootstrap = self.collect_bootstrap_static()
        current_gameweek = next((gw for gw in bootstrap['events'] if gw['is_current']), None)
        
        if current_gameweek is None:
            current_gameweek = next((gw for gw in bootstrap['events'] if gw['is_next']), None)
            if current_gameweek is None:
                return {}
            gw_number = current_gameweek['id'] - 1
        else:
            gw_number = current_gameweek['id']
        
        all_gw_data = {}
        for i in range(1, gw_number + 1):
            print(f"Collecting data for gameweek {i}...")
            all_gw_data[i] = self.collect_gameweek_data(i)
        
        return all_gw_data
    
    def _save_data(self, data, name):
        """Save data to file with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.json"
        filepath = os.path.join(self.data_dir, "raw", filename)
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
        
        print(f"Saved {name} data to {filepath}")
    
    def collect_all_data(self):
        """Collect all available FPL data"""
        print("Starting comprehensive data collection...")
        bootstrap = self.collect_bootstrap_static()
        fixtures = self.collect_fixtures()
        player_histories = self.collect_player_history()
        gameweeks = self.collect_all_gameweeks()
        
        print("Data collection complete!")
        return {
            "bootstrap": bootstrap,
            "fixtures": fixtures,
            "player_histories": player_histories,
            "gameweeks": gameweeks
        }

if __name__ == "__main__":
    collector = FPLDataCollector()
    collector.collect_all_data()