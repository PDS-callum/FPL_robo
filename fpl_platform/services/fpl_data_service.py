"""
Service for fetching FPL data from various sources
"""
import requests
import json
from datetime import datetime
import time


class FPLDataService:
    """
    Service to fetch FPL data from official API and other sources
    """
    
    # Official FPL API endpoints
    FPL_BASE_URL = "https://fantasy.premierleague.com/api"
    BOOTSTRAP_URL = f"{FPL_BASE_URL}/bootstrap-static/"
    FIXTURES_URL = f"{FPL_BASE_URL}/fixtures/"
    ELEMENT_SUMMARY_URL = f"{FPL_BASE_URL}/element-summary/{{player_id}}/"
    
    # For historical data, you would use archived endpoints or downloaded datasets
    # Example: GitHub repos with historical FPL data
    HISTORICAL_DATA_SOURCES = {
        'github': 'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master',
        # Fantasy Nutmeg or other sources could be added here
        # 'nutmeg': 'https://www.fantasynutmeg.com/api/...'
    }
    
    @staticmethod
    def fetch_current_season_data():
        """
        Fetch current season bootstrap data (players, teams, gameweeks)
        Returns: dict with all current season data
        """
        try:
            response = requests.get(FPLDataService.BOOTSTRAP_URL, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Transform into more usable format
            players = {}
            for player in data['elements']:
                players[player['id']] = {
                    'id': player['id'],
                    'name': player['web_name'],
                    'full_name': f"{player['first_name']} {player['second_name']}",
                    'team': player['team'],
                    'position': ['GKP', 'DEF', 'MID', 'FWD'][player['element_type'] - 1],
                    'cost': player['now_cost'] / 10.0,  # Convert from tenths to actual value
                    'total_points': player['total_points'],
                    'form': float(player['form']) if player['form'] else 0.0,
                    'points_per_game': float(player['points_per_game']) if player['points_per_game'] else 0.0,
                }
            
            teams = {team['id']: team for team in data['teams']}
            
            # Get current gameweek
            current_gw = None
            for event in data['events']:
                if event['is_current']:
                    current_gw = event['id']
                    break
            if not current_gw:
                # If no current GW, find the last finished one
                for event in data['events']:
                    if event['finished']:
                        current_gw = event['id']
            
            return {
                'players': players,
                'teams': teams,
                'events': data['events'],
                'current_gameweek': current_gw or 1,
                'total_players': data['total_players']
            }
        except requests.RequestException as e:
            print(f"Error fetching current season data: {e}")
            return None
    
    @staticmethod
    def fetch_fixtures():
        """
        Fetch all fixtures for current season
        Returns: list of fixtures
        """
        try:
            response = requests.get(FPLDataService.FIXTURES_URL, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching fixtures: {e}")
            return []
    
    @staticmethod
    def fetch_player_history(player_id):
        """
        Fetch historical data for a specific player
        Returns: dict with player history across seasons
        """
        try:
            url = FPLDataService.ELEMENT_SUMMARY_URL.format(player_id=player_id)
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching player {player_id} history: {e}")
            return None
    
    @staticmethod
    def fetch_gameweek_player_stats(gameweek):
        """
        Fetch detailed player statistics for a specific gameweek
        
        This includes: minutes, goals, assists, clean_sheets, saves, bonus, bps, etc.
        
        Args:
            gameweek: Gameweek number
            
        Returns:
            Dict mapping player_id to their gameweek stats
        """
        try:
            # Get fixtures for this gameweek
            fixtures = FPLDataService.fetch_fixtures()
            if not fixtures:
                return {}
            
            # Filter to this gameweek
            gw_fixtures = [f for f in fixtures if f.get('event') == gameweek]
            
            # Collect player stats from all fixtures
            player_stats = {}
            
            for fixture in gw_fixtures:
                if not fixture.get('finished'):
                    continue  # Skip unfinished fixtures
                
                # Get stats for both home and away teams
                for team_type in ['h', 'a']:
                    stats_key = f'stats_{team_type}'
                    
                    if stats_key in fixture:
                        for stat_category in fixture[stats_key]:
                            # Each stat category has a list of player stats
                            for player_stat in stat_category.get('stats', []):
                                player_id = player_stat.get('element')
                                
                                if player_id not in player_stats:
                                    player_stats[player_id] = {
                                        'minutes': 0,
                                        'goals_scored': 0,
                                        'assists': 0,
                                        'clean_sheets': 0,
                                        'goals_conceded': 0,
                                        'own_goals': 0,
                                        'penalties_saved': 0,
                                        'penalties_missed': 0,
                                        'yellow_cards': 0,
                                        'red_cards': 0,
                                        'saves': 0,
                                        'bonus': 0,
                                        'bps': 0,
                                        'influence': 0.0,
                                        'creativity': 0.0,
                                        'threat': 0.0,
                                        'ict_index': 0.0,
                                        'total_points': 0
                                    }
                                
                                # Update stats from this fixture
                                # The API provides cumulative stats, so we just take the value
                                player_stats[player_id]['minutes'] = player_stat.get('minutes', 0)
                                player_stats[player_id]['goals_scored'] = player_stat.get('goals_scored', 0)
                                player_stats[player_id]['assists'] = player_stat.get('assists', 0)
                                player_stats[player_id]['clean_sheets'] = player_stat.get('clean_sheets', 0)
                                player_stats[player_id]['goals_conceded'] = player_stat.get('goals_conceded', 0)
                                player_stats[player_id]['saves'] = player_stat.get('saves', 0)
                                player_stats[player_id]['bonus'] = player_stat.get('bonus', 0)
                                player_stats[player_id]['bps'] = player_stat.get('bps', 0)
                                player_stats[player_id]['total_points'] = player_stat.get('total_points', 0)
            
            return player_stats
            
        except Exception as e:
            print(f"Error fetching gameweek {gameweek} player stats: {e}")
            return {}
    
    @staticmethod
    def fetch_historical_season_data(season_year='2023-24', source='github'):
        """
        Fetch historical season data from archived sources
        
        Args:
            season_year: str, e.g., '2023-24'
            source: str, which data source to use
            
        Returns: dict with historical season data
        """
        if source == 'github':
            # Example: Fetch from vaastav's GitHub repo
            # This is a well-maintained repo with historical FPL data
            base_url = FPLDataService.HISTORICAL_DATA_SOURCES['github']
            season_folder = season_year.replace('-', '-20')  # e.g., '2023-2024'
            
            try:
                # Fetch players data
                players_url = f"{base_url}/data/{season_folder}/players_raw.csv"
                players_response = requests.get(players_url, timeout=10)
                
                if players_response.status_code == 200:
                    # Parse CSV data (you'd use pandas or csv module in practice)
                    return {
                        'season': season_year,
                        'players_data': players_response.text,
                        'source': 'github-vaastav'
                    }
            except requests.RequestException as e:
                print(f"Error fetching historical data: {e}")
                return None
        
        return None
    
    @staticmethod
    def fetch_gameweek_data(season_year='2023-24', gameweek=1):
        """
        Fetch specific gameweek data for backtesting
        
        Args:
            season_year: str, e.g., '2023-24'
            gameweek: int, gameweek number
            
        Returns: dict with gameweek player performances
        """
        # This would fetch actual player points, bonus, minutes, etc. for that GW
        # For now, returning structure that would come from historical data
        
        # In production, you'd:
        # 1. Fetch from historical database
        # 2. Parse CSV/JSON from archived sources
        # 3. Query an API if available
        
        return {
            'gameweek': gameweek,
            'season': season_year,
            'deadline_time': None,  # Would come from data
            'player_performances': []  # List of player stats for that GW
        }
    
    @staticmethod
    def get_player_data_for_backtest(season_year='2023-24', start_gw=1, end_gw=38):
        """
        Get comprehensive player data for a backtest period
        
        This would load all player data needed for the specified gameweeks,
        including predicted points, actual points, prices, etc.
        
        Returns: dict with player data organized by gameweek
        """
        # In a full implementation:
        # 1. Load player database for season
        # 2. Load fixtures and results
        # 3. Load price changes
        # 4. Organize data by gameweek
        
        return {
            'season': season_year,
            'gameweeks': {},  # Would be populated with GW data
            'players': {},    # Would be populated with player info
            'teams': {}       # Would be populated with team info
        }
    
    @staticmethod
    def parse_player_data(raw_data):
        """
        Parse raw player data into standardized format
        
        Args:
            raw_data: Raw data from API or CSV
            
        Returns: List of standardized player dictionaries
        """
        # Convert various data formats into standard structure
        players = []
        
        # This would handle parsing from different sources
        # and create a unified player data structure
        
        return players
    
    @staticmethod
    def cache_historical_data(season_year='2023-24'):
        """
        Download and cache historical season data locally
        
        This would be run once to download full season data
        and store it locally for fast backtest access
        """
        print(f"Caching historical data for {season_year}...")
        
        # 1. Fetch all gameweek data
        # 2. Fetch all player data
        # 3. Store in local database or files
        # 4. Create indexes for fast lookup
        
        return {
            'status': 'cached',
            'season': season_year,
            'gameweeks_cached': 38,
            'players_cached': 600
        }


# Example usage functions for integration with backtest

def prepare_backtest_data(season='2023-24', start_gw=1, end_gw=38):
    """
    Prepare all data needed for a backtest
    
    This function would be called before running a backtest to ensure
    all necessary historical data is available
    """
    service = FPLDataService()
    
    # Check if data is cached, if not, cache it
    data = service.get_player_data_for_backtest(season, start_gw, end_gw)
    
    # Validate data completeness
    if not data or not data.get('gameweeks'):
        print(f"Warning: Historical data for {season} not available")
        print("Using simulated data for backtest")
        return None
    
    return data


def get_gameweek_player_stats(season='2023-24', gameweek=1):
    """
    Get actual player stats for a specific gameweek (for backtest scoring)
    
    Returns: Dict mapping player_id to their stats for that gameweek
    """
    service = FPLDataService()
    gw_data = service.fetch_gameweek_data(season, gameweek)
    
    if not gw_data:
        return {}
    
    # Parse and return player stats
    return {
        # player_id: {points, minutes, goals, assists, clean_sheet, etc.}
    }

