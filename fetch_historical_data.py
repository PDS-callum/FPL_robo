"""
Script to fetch and cache historical FPL data for backtesting

This script demonstrates how to fetch real FPL data from various sources:
1. Official FPL API for current season
2. GitHub repositories for historical seasons
3. Other data providers like Fantasy Nutmeg

Run this script to download and cache historical data before running backtests.
"""

import requests
import json
import csv
import os
from pathlib import Path


def fetch_current_season():
    """Fetch current season data from official FPL API"""
    print("Fetching current season data from FPL API...")
    
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Found {len(data['elements'])} players")
        print(f"✓ Found {len(data['teams'])} teams")
        print(f"✓ Current event: GW{data['events'][0]['id']}")
        
        # Save to file
        os.makedirs('data/current_season', exist_ok=True)
        with open('data/current_season/bootstrap.json', 'w') as f:
            json.dump(data, f, indent=2)
        
        return data
        
    except requests.RequestException as e:
        print(f"✗ Error fetching data: {e}")
        return None


def fetch_historical_from_github(season='2023-24'):
    """
    Fetch historical season data from vaastav's GitHub repository
    
    This is a popular open-source repo with complete FPL historical data:
    https://github.com/vaastav/Fantasy-Premier-League
    """
    print(f"\nFetching {season} historical data from GitHub...")
    
    season_folder = season.replace('-', '-20')  # e.g., '2023-2024'
    base_url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season_folder}"
    
    files_to_fetch = {
        'players_raw': f"{base_url}/players_raw.csv",
        'teams': f"{base_url}/teams.csv",
        'gws': f"{base_url}/gws/merged_gw.csv",  # All gameweeks merged
    }
    
    season_dir = Path(f'data/historical/{season}')
    season_dir.mkdir(parents=True, exist_ok=True)
    
    for name, url in files_to_fetch.items():
        try:
            print(f"  Fetching {name}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save to file
            file_path = season_dir / f"{name}.csv"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            print(f"  ✓ Saved {name} ({len(response.text)} bytes)")
            
        except requests.RequestException as e:
            print(f"  ✗ Error fetching {name}: {e}")
    
    print(f"✓ Historical data for {season} saved to {season_dir}")


def fetch_player_gameweek_data(season='2023-24', player_name='Haaland'):
    """
    Example: Fetch detailed gameweek-by-gameweek data for a specific player
    """
    print(f"\nFetching gameweek data for {player_name}...")
    
    season_folder = season.replace('-', '-20')
    # Player folders use underscores and are in format: Erling_Haaland_439
    # You would need to map player names to their IDs
    
    # Example URL structure:
    # https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2023-2024/players/Erling_Haaland_439/gw.csv
    
    print("Note: You would need to find the player's ID from the main dataset")


def parse_historical_data(season='2023-24'):
    """
    Parse downloaded historical data into a format suitable for backtesting
    """
    season_dir = Path(f'data/historical/{season}')
    
    if not season_dir.exists():
        print(f"No historical data found for {season}. Run fetch_historical_from_github() first.")
        return None
    
    print(f"\nParsing {season} historical data...")
    
    # Parse players data
    players_file = season_dir / 'players_raw.csv'
    if players_file.exists():
        with open(players_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            players = list(reader)
            print(f"✓ Loaded {len(players)} players")
            
            # Show sample player data
            if players:
                sample = players[0]
                print(f"\nSample player data keys: {list(sample.keys())[:10]}...")
    
    return players


def example_backtest_integration():
    """
    Example of how to integrate this with the backtest system
    """
    print("\n" + "="*60)
    print("EXAMPLE: Integrating with Backtest System")
    print("="*60)
    
    example_code = '''
# In bot_execution_service.py, you would:

from fpl_platform.services.fpl_data_service import FPLDataService

def run_backtest_with_real_data(bot_submission_id, season="2023-24"):
    # 1. Load historical data for the season
    data_service = FPLDataService()
    historical_data = data_service.get_player_data_for_backtest(season)
    
    # 2. For each gameweek, use REAL player data
    for gw in range(1, 39):
        # Get actual player stats and prices for this GW
        gw_data = historical_data['gameweeks'][gw]
        
        # Pass real data to bot
        bot_input = {
            'current_team': current_squad,
            'gameweek': gw,
            'bank': bank,
            'available_players': gw_data['players'],  # Real players with real stats
        }
        
        optimization = bot_instance.optimize(bot_input)
        
        # Calculate ACTUAL points based on real gameweek results
        actual_points = calculate_actual_points(
            optimization['lineup'],
            optimization['captain'],
            gw_data['results']  # Real player performances
        )
'''
    
    print(example_code)


if __name__ == '__main__':
    print("="*60)
    print("FPL Historical Data Fetcher")
    print("="*60)
    
    # 1. Fetch current season
    # current_data = fetch_current_season()
    
    # 2. Fetch historical seasons
    # Uncomment to actually fetch data:
    # fetch_historical_from_github('2023-24')
    # fetch_historical_from_github('2022-23')
    
    # 3. Parse the data
    # parse_historical_data('2023-24')
    
    # 4. Show integration example
    example_backtest_integration()
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Uncomment the fetch functions above to download real data")
    print("2. Run this script to cache historical seasons locally")
    print("3. Update bot_execution_service.py to use real data")
    print("4. Integrate with your bot's analyze() and optimize() methods")
    print("\nData sources:")
    print("- Official API: https://fantasy.premierleague.com/api/bootstrap-static/")
    print("- GitHub repo: https://github.com/vaastav/Fantasy-Premier-League")
    print("- Fantasy Nutmeg: https://www.fantasynutmeg.com")
    print("="*60)

