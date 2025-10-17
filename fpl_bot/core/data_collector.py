"""
Data collection module for FPL Bot

Handles fetching data from:
- FPL API for current season data
- fpl-data.co.uk for previous season statistics
- Manager-specific data (teams, transfers, history)
"""

import requests
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import time


class DataCollector:
    """Handles all data collection for the FPL Bot"""
    
    def __init__(self):
        self.fpl_base_url = "https://fantasy.premierleague.com/api"
        self.fpl_data_url = "https://www.fpl-data.co.uk/statistics"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'FPL-Bot/2.0.0 (https://github.com/yourusername/fpl-bot)'
        })
        self.authenticated = False
        self.manager_id = None
        
    def get_current_season_data(self) -> Optional[Dict]:
        """Get current season data from FPL API"""
        try:
            response = self.session.get(f"{self.fpl_base_url}/bootstrap-static/")
            response.raise_for_status()
            
            data = response.json()
            print(f"Successfully fetched current season data for {len(data['elements'])} players")
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching current season data: {e}")
            return None
    
    def get_previous_season_data(self, season: str = "2023-24") -> Optional[pd.DataFrame]:
        """Get previous season data from fpl-data.co.uk"""
        try:
            # Try to fetch from the statistics page
            response = self.session.get(self.fpl_data_url)
            response.raise_for_status()
            
            # For now, we'll create a placeholder structure
            # In a real implementation, you'd parse the HTML or use their API if available
            print(f"Note: Previous season data collection needs implementation for {season}")
            print("Currently returning None - will need manual data or alternative source")
            
            return None
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching previous season data: {e}")
            return None
    
    def get_manager_data(self, manager_id: int) -> Optional[Dict]:
        """Get manager's current team and history"""
        try:
            # Get manager entry data
            manager_url = f"{self.fpl_base_url}/entry/{manager_id}/"
            response = self.session.get(manager_url)
            response.raise_for_status()
            
            manager_data = response.json()
            
            # Get manager's current team
            # Try to fetch team for current gameweek, but if 404, try previous gameweeks
            current_gw = self._get_current_gameweek()
            
            team_fetched = False
            if current_gw:
                # Try current gameweek and up to 2 previous gameweeks
                for gw_offset in range(0, 3):
                    try_gw = current_gw - gw_offset
                    if try_gw < 1:
                        break
                    
                    team_url = f"{self.fpl_base_url}/entry/{manager_id}/event/{try_gw}/picks/"
                    team_response = self.session.get(team_url)
                    
                    if team_response.status_code == 200:
                        manager_data['current_team'] = team_response.json()
                        picks_count = len(manager_data['current_team'].get('picks', []))
                        print(f"Successfully fetched team for GW{try_gw}: {picks_count} picks")
                        
                        # Debug: Show what fields are in the response
                        if 'entry_history' in manager_data['current_team']:
                            print(f"  [DEBUG] entry_history keys: {list(manager_data['current_team']['entry_history'].keys())}")
                        if 'transfers' in manager_data['current_team']:
                            print(f"  [DEBUG] transfers object: {manager_data['current_team']['transfers']}")
                        
                        team_fetched = True
                        break
                
                if not team_fetched:
                    print(f"Could not fetch team for GW{current_gw} or previous gameweeks")
            else:
                print("Could not determine current gameweek - team not fetched")
            
            # Get manager's history
            history_url = f"{self.fpl_base_url}/entry/{manager_id}/history/"
            history_response = self.session.get(history_url)
            if history_response.status_code == 200:
                manager_data['history'] = history_response.json()
            
            # Get transfer history
            transfers_url = f"{self.fpl_base_url}/entry/{manager_id}/transfers/"
            transfers_response = self.session.get(transfers_url)
            if transfers_response.status_code == 200:
                manager_data['transfers'] = transfers_response.json()
            
            # Calculate saved transfers using the improved method
            manager_data['saved_transfers'] = self._calculate_saved_transfers(manager_data, current_gw)
            
            print(f"Successfully fetched data for manager ID: {manager_id}")
            return manager_data
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching manager data for {manager_id}: {e}")
            return None
    
    def get_player_detailed_stats(self, player_id: int) -> Optional[Dict]:
        """Get detailed stats for a specific player"""
        try:
            player_url = f"{self.fpl_base_url}/element-summary/{player_id}/"
            response = self.session.get(player_url)
            response.raise_for_status()
            
            player_data = response.json()
            return player_data
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching detailed stats for player {player_id}: {e}")
            return None
    
    def get_fixtures_data(self) -> Optional[Dict]:
        """Get fixtures data for the current season"""
        try:
            fixtures_url = f"{self.fpl_base_url}/fixtures/"
            response = self.session.get(fixtures_url)
            response.raise_for_status()
            
            fixtures_data = response.json()
            return fixtures_data
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching fixtures data: {e}")
            return None
    
    def get_live_gameweek_data(self, gameweek: Optional[int] = None) -> Optional[Dict]:
        """Get live data for a specific gameweek including real-time player stats
        
        Args:
            gameweek: Gameweek number. If None, uses current gameweek.
            
        Returns:
            Dict with live player statistics including BPS, points, minutes played
        """
        try:
            if gameweek is None:
                gameweek = self._get_current_gameweek()
            
            if not gameweek:
                print("Could not determine gameweek for live data")
                return None
            
            live_url = f"{self.fpl_base_url}/event/{gameweek}/live/"
            response = self.session.get(live_url)
            response.raise_for_status()
            
            live_data = response.json()
            print(f"Successfully fetched live data for GW{gameweek} - {len(live_data.get('elements', []))} players")
            return live_data
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching live gameweek data: {e}")
            return None
    
    def get_dream_team(self, gameweek: Optional[int] = None) -> Optional[Dict]:
        """Get the dream team (highest scoring 11) for a specific gameweek
        
        Args:
            gameweek: Gameweek number. If None, uses current gameweek.
            
        Returns:
            Dict with dream team player IDs and formation
        """
        try:
            if gameweek is None:
                gameweek = self._get_current_gameweek()
            
            if not gameweek:
                print("Could not determine gameweek for dream team")
                return None
            
            dream_team_url = f"{self.fpl_base_url}/dream-team/{gameweek}/"
            response = self.session.get(dream_team_url)
            response.raise_for_status()
            
            dream_team_data = response.json()
            print(f"Successfully fetched dream team for GW{gameweek}")
            return dream_team_data
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching dream team: {e}")
            return None
    
    def get_event_status(self) -> Optional[Dict]:
        """Get current event/gameweek status including bonus points status
        
        Returns:
            Dict with status info, bonus points provisional/confirmed status
        """
        try:
            status_url = f"{self.fpl_base_url}/event-status/"
            response = self.session.get(status_url)
            response.raise_for_status()
            
            status_data = response.json()
            print(f"Successfully fetched event status")
            return status_data
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching event status: {e}")
            return None
    
    def get_classic_league_standings(self, league_id: int, page: int = 1) -> Optional[Dict]:
        """Get standings for a classic league
        
        Args:
            league_id: The league ID
            page: Page number for pagination (default 1)
            
        Returns:
            Dict with league standings, manager points, ranks
        """
        try:
            league_url = f"{self.fpl_base_url}/leagues-classic/{league_id}/standings/"
            params = {'page_standings': page} if page > 1 else {}
            
            response = self.session.get(league_url, params=params)
            response.raise_for_status()
            
            league_data = response.json()
            standings_count = len(league_data.get('standings', {}).get('results', []))
            print(f"Successfully fetched league {league_id} standings - {standings_count} managers on page {page}")
            return league_data
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching league standings: {e}")
            return None
    
    def get_h2h_league_standings(self, league_id: int, page: int = 1) -> Optional[Dict]:
        """Get standings for a head-to-head league
        
        Args:
            league_id: The league ID
            page: Page number for pagination (default 1)
            
        Returns:
            Dict with H2H league standings
        """
        try:
            league_url = f"{self.fpl_base_url}/leagues-h2h/{league_id}/standings/"
            params = {'page_standings': page} if page > 1 else {}
            
            response = self.session.get(league_url, params=params)
            response.raise_for_status()
            
            league_data = response.json()
            print(f"Successfully fetched H2H league {league_id} standings")
            return league_data
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching H2H league standings: {e}")
            return None
    
    def get_set_piece_takers(self) -> Optional[Dict]:
        """Extract set piece takers information from bootstrap-static data
        
        Returns:
            Dict mapping team_id to set piece takers info
        """
        try:
            data = self.get_current_season_data()
            if not data:
                return None
            
            teams = data.get('teams', [])
            set_pieces = {}
            
            for team in teams:
                team_id = team['id']
                set_pieces[team_id] = {
                    'team_name': team['name'],
                    'team_short_name': team['short_name'],
                    'corners_and_indirect_freekicks_order': team.get('corners_and_indirect_freekicks_order', []),
                    'direct_freekicks_order': team.get('direct_freekicks_order', []),
                    'penalties_order': team.get('penalties_order', [])
                }
            
            print(f"Successfully extracted set piece takers for {len(set_pieces)} teams")
            return set_pieces
            
        except Exception as e:
            print(f"Error extracting set piece takers: {e}")
            return None
    
    def get_team_strengths(self) -> Optional[pd.DataFrame]:
        """Extract team strength and fixture difficulty ratings from bootstrap-static
        
        Returns:
            DataFrame with team strengths (attack/defence, home/away)
        """
        try:
            data = self.get_current_season_data()
            if not data:
                return None
            
            teams = data.get('teams', [])
            teams_df = pd.DataFrame(teams)
            
            # Select relevant strength columns
            strength_cols = [
                'id', 'name', 'short_name',
                'strength', 
                'strength_overall_home', 'strength_overall_away',
                'strength_attack_home', 'strength_attack_away',
                'strength_defence_home', 'strength_defence_away'
            ]
            
            teams_df = teams_df[strength_cols]
            print(f"Successfully extracted team strengths for {len(teams_df)} teams")
            return teams_df
            
        except Exception as e:
            print(f"Error extracting team strengths: {e}")
            return None
    
    def create_players_dataframe(self, all_players_data: Dict, include_set_pieces: bool = True) -> Optional[pd.DataFrame]:
        """Create a comprehensive DataFrame with all players' season data
        
        Args:
            all_players_data: Data from bootstrap-static endpoint
            include_set_pieces: Whether to add set piece taker flags
            
        Returns:
            DataFrame with player data enriched with team info and set piece roles
        """
        if not all_players_data:
            return None
        
        players = all_players_data['elements']
        teams = all_players_data['teams']
        
        # Create teams lookup
        teams_dict = {team['id']: team for team in teams}
        
        # Convert to DataFrame
        df = pd.DataFrame(players)
        
        # Add team names
        df['team_name'] = df['team'].map(lambda x: teams_dict.get(x, {}).get('name', 'Unknown'))
        df['team_short_name'] = df['team'].map(lambda x: teams_dict.get(x, {}).get('short_name', 'Unknown'))
        
        # Convert costs to proper format
        df['cost'] = df['now_cost'] / 10
        
        # Handle cost changes
        if 'cost_change_start' in df.columns:
            df['cost_start'] = df['cost'] - (df['cost_change_start'] / 10)
        else:
            df['cost_start'] = df['cost']
        
        # Calculate value metrics
        df['points_per_million'] = df['total_points'] / df['cost'].replace(0, 1)
        df['value_rating'] = df['points_per_million'].fillna(0)
        
        # Add position names
        position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        df['position_name'] = df['element_type'].map(position_map)
        
        # Add form and ownership data
        df['form_rating'] = df.get('form', 0).fillna(0)
        df['ownership_percent'] = df.get('selected_by_percent', 0).fillna(0)
        
        # Add availability/injury status information
        df['status'] = df.get('status', 'a').fillna('a')  # a=available, d=doubtful, i=injured, u=unavailable, s=suspended
        df['chance_of_playing_next_round'] = df.get('chance_of_playing_next_round', None)
        df['chance_of_playing_this_round'] = df.get('chance_of_playing_this_round', None)
        df['news'] = df.get('news', '').fillna('')
        
        # Add set piece taker information
        if include_set_pieces:
            df['on_penalties'] = False
            df['penalty_order'] = None
            df['on_corners'] = False
            df['corner_order'] = None
            df['on_freekicks'] = False
            df['freekick_order'] = None
            
            for team_id, team in teams_dict.items():
                # Penalties
                penalties = team.get('penalties_order', [])
                for order, player_id in enumerate(penalties, 1):
                    mask = (df['id'] == player_id) & (df['team'] == team_id)
                    df.loc[mask, 'on_penalties'] = True
                    df.loc[mask, 'penalty_order'] = order
                
                # Corners and indirect free kicks
                corners = team.get('corners_and_indirect_freekicks_order', [])
                for order, player_id in enumerate(corners, 1):
                    mask = (df['id'] == player_id) & (df['team'] == team_id)
                    df.loc[mask, 'on_corners'] = True
                    df.loc[mask, 'corner_order'] = order
                
                # Direct free kicks
                freekicks = team.get('direct_freekicks_order', [])
                for order, player_id in enumerate(freekicks, 1):
                    mask = (df['id'] == player_id) & (df['team'] == team_id)
                    df.loc[mask, 'on_freekicks'] = True
                    df.loc[mask, 'freekick_order'] = order
        
        return df
    
    def get_fixtures(self) -> Optional[pd.DataFrame]:
        """Get fixtures as a DataFrame with team strength information
        
        Returns:
            DataFrame with fixtures enriched with team strength data
        """
        try:
            fixtures_data = self.get_fixtures_data()
            if not fixtures_data:
                return None
            
            fixtures_df = pd.DataFrame(fixtures_data)
            
            # Get team strengths
            team_strengths = self.get_team_strengths()
            if team_strengths is not None:
                # Merge home team strengths
                fixtures_df = fixtures_df.merge(
                    team_strengths[['id', 'name', 'strength_attack_home', 'strength_defence_home']],
                    left_on='team_h',
                    right_on='id',
                    how='left',
                    suffixes=('', '_home')
                )
                fixtures_df.rename(columns={
                    'name': 'team_h_name',
                    'strength_attack_home': 'team_h_attack',
                    'strength_defence_home': 'team_h_defence'
                }, inplace=True)
                fixtures_df.drop(columns=['id'], inplace=True)
                
                # Merge away team strengths
                fixtures_df = fixtures_df.merge(
                    team_strengths[['id', 'name', 'strength_attack_away', 'strength_defence_away']],
                    left_on='team_a',
                    right_on='id',
                    how='left',
                    suffixes=('', '_away')
                )
                fixtures_df.rename(columns={
                    'name': 'team_a_name',
                    'strength_attack_away': 'team_a_attack',
                    'strength_defence_away': 'team_a_defence'
                }, inplace=True)
                fixtures_df.drop(columns=['id'], inplace=True)
            
            return fixtures_df
            
        except Exception as e:
            print(f"Error creating fixtures dataframe: {e}")
            return None
    
    def get_player_data(self, include_set_pieces: bool = True) -> Optional[pd.DataFrame]:
        """Convenience method to get player data as DataFrame
        
        Args:
            include_set_pieces: Whether to include set piece taker information
            
        Returns:
            DataFrame with enriched player data
        """
        try:
            season_data = self.get_current_season_data()
            if not season_data:
                return None
            
            return self.create_players_dataframe(season_data, include_set_pieces=include_set_pieces)
            
        except Exception as e:
            print(f"Error getting player data: {e}")
            return None
    
    def _get_current_gameweek(self) -> Optional[int]:
        """Get the current gameweek number (next unplayed gameweek for optimization)
        
        Returns the next gameweek that hasn't been played yet, which is what we want
        to optimize for. If a gameweek is marked as current but finished, we return
        the next one.
        """
        try:
            response = self.session.get(f"{self.fpl_base_url}/bootstrap-static/")
            response.raise_for_status()
            data = response.json()
            events = data.get('events', [])
            
            # Strategy: Find the first gameweek that hasn't finished yet
            # This is the one we should optimize for
            for event in events:
                # Check if this gameweek is not finished and hasn't started or is in progress
                if not event.get('finished', False):
                    return event.get('id')
            
            # Fallback: if all are finished (shouldn't happen), return the last one
            if events:
                return events[-1].get('id')
            
            return None
        except Exception as e:
            return None
    
    def _calculate_saved_transfers(self, manager_data: Dict, current_gw: int) -> Dict:
        """Calculate free transfers available using API data.
        
        Priority:
        1. Try to get directly from API 'transfers.limit' field (most reliable)
        2. Fall back to calculation if not available
        
        Calculation approach:
        1. Get transfers made this GW from entry_history.event_transfers
        2. Get previous GW transfers from history to determine starting FTs (2 if prev=0, else 1)
        3. Calculate: available = max(0, starting_FTs - transfers_made_this_gw)
        """
        if not current_gw or current_gw <= 1:
            print(f"  [FT Calc] Early season (GW {current_gw}), defaulting to 1 FT")
            return {
                'free_transfers': 1,
                'free_transfers_at_start': 1,
                'total_available': 1,
                'transfers_this_gw': 0
            }

        print(f"  [FT Calc] Calculating FTs for GW{current_gw}...")
        
        # PRIORITY 1: Try to get FTs directly from API
        current_team = manager_data.get('current_team', {})
        if current_team and 'transfers' in current_team:
            transfers_obj = current_team['transfers']
            if 'limit' in transfers_obj and 'made' in transfers_obj:
                api_available_fts = transfers_obj.get('limit', 1)
                api_made = transfers_obj.get('made', 0)
                print(f"  [FT Calc] API Direct - Available FTs: {api_available_fts}, Made: {api_made}")
                
                # Also get previous GW info for context
                try:
                    hist_current = ((manager_data.get('history') or {}).get('current')) or []
                    prev_row = next((row for row in hist_current if row.get('event') == current_gw - 1), None)
                    prev_gw_transfers = int(prev_row.get('event_transfers', 0) or 0) if prev_row else 0
                except:
                    prev_gw_transfers = 0
                
                return {
                    'free_transfers': api_available_fts,
                    'free_transfers_at_start': api_available_fts + api_made,  # Reconstruct starting FTs
                    'total_available': api_available_fts,
                    'transfers_this_gw': api_made,
                    'prev_gw_transfers': prev_gw_transfers
                }
        
        print(f"  [FT Calc] API direct method not available, using history-based calculation...")

        # SIMPLIFIED APPROACH: Look at transfer history pattern
        # Logic: You gain 1 FT per week, capped at 2
        # Available FTs = min(weeks_since_last_transfer, 2) - transfers_made_this_gw
        
        try:
            hist_current = ((manager_data.get('history') or {}).get('current')) or []
            
            # Get transfers made THIS gameweek
            current_row = next((row for row in hist_current if row.get('event') == current_gw), None)
            transfers_made_this_gw = int(current_row.get('event_transfers', 0) or 0) if current_row else 0
            print(f"  [FT Calc] Transfers made this GW: {transfers_made_this_gw}")
            
            # Find LAST gameweek where transfers were made (before current GW)
            # Important: Even if they took a hit, we count from that week (FTs reset to 0, then build up again)
            last_transfer_gw = None
            for row in reversed(hist_current):
                gw = row.get('event')
                if gw and gw < current_gw and row.get('event_transfers', 0) > 0:
                    last_transfer_gw = gw
                    break
            
            if last_transfer_gw:
                weeks_since_last_transfer = current_gw - last_transfer_gw
                print(f"  [FT Calc] Last transfer in GW{last_transfer_gw}, {weeks_since_last_transfer} weeks ago")
            else:
                # No transfers found in history, assume started with 1 FT
                weeks_since_last_transfer = 1
                print(f"  [FT Calc] No previous transfers found in history")
            
            # Calculate FTs at start of this gameweek
            # Rules:
            # 1. You gain 1 FT per week without making transfers
            # 2. Maximum FTs you can bank: 5 (not standard FPL 2, but for multi-week planning)
            # 3. If you take a hit, FTs go to 0 (not negative), then build back up
            MAX_FREE_TRANSFERS = 5
            free_transfers_at_start = min(weeks_since_last_transfer, MAX_FREE_TRANSFERS)
            print(f"  [FT Calc] FTs at start of GW{current_gw}: {free_transfers_at_start} (capped at {MAX_FREE_TRANSFERS})")
            
            # Calculate available FTs NOW
            # If transfers > FTs, you took a hit, but available FTs = 0 (not negative)
            available_free_transfers = max(0, free_transfers_at_start - transfers_made_this_gw)
            print(f"  [FT Calc] Available FTs NOW: {available_free_transfers}")
            
            if transfers_made_this_gw > free_transfers_at_start:
                hits_taken = transfers_made_this_gw - free_transfers_at_start
                print(f"  [FT Calc] ⚠ Hit taken: {hits_taken} extra transfer(s) = -{hits_taken * 4} points")
            
            # Get previous GW transfers for context
            prev_row = next((row for row in hist_current if row.get('event') == current_gw - 1), None)
            prev_gw_transfers = int(prev_row.get('event_transfers', 0) or 0) if prev_row else 0
            
        except Exception as e:
            print(f"  [FT Calc] Error in calculation: {e}")
            # Fallback to safe defaults
            available_free_transfers = 1
            free_transfers_at_start = 1
            transfers_made_this_gw = 0
            prev_gw_transfers = 0

        return {
            'free_transfers': available_free_transfers,
            'free_transfers_at_start': free_transfers_at_start,
            'total_available': available_free_transfers,
            'transfers_this_gw': transfers_made_this_gw,
            'prev_gw_transfers': prev_gw_transfers
        }
    
    def save_data_to_file(self, data: Dict, filename: str) -> bool:
        """Save data to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            print(f"Data saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving data to {filename}: {e}")
            return False
    
    def load_data_from_file(self, filename: str) -> Optional[Dict]:
        """Load data from JSON file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            print(f"Data loaded from {filename}")
            return data
        except Exception as e:
            print(f"Error loading data from {filename}: {e}")
            return None
    
    def authenticate(self, email: str, password: str) -> bool:
        """Authenticate with FPL to enable transfer execution
        
        Args:
            email: FPL account email
            password: FPL account password
            
        Returns:
            True if authentication successful, False otherwise
        """
        try:
            login_url = "https://users.premierleague.com/accounts/login/"
            
            # First, get the login page to retrieve CSRF token
            response = self.session.get(login_url)
            
            # Extract CSRF token from cookies
            csrf_token = self.session.cookies.get('csrftoken')
            
            if not csrf_token:
                print("Error: Could not retrieve CSRF token")
                return False
            
            # Prepare login data
            login_data = {
                'login': email,
                'password': password,
                'redirect_uri': 'https://fantasy.premierleague.com/',
                'app': 'plfpl-web'
            }
            
            # Set headers for login request
            headers = {
                'User-Agent': 'FPL-Bot/2.0.0',
                'Content-Type': 'application/x-www-form-urlencoded',
                'Referer': 'https://fantasy.premierleague.com/',
                'X-CSRFToken': csrf_token
            }
            
            # Attempt login
            response = self.session.post(login_url, data=login_data, headers=headers)
            
            if response.status_code == 200:
                # Verify authentication by checking if we can access my-team endpoint
                verify_url = f"{self.fpl_base_url}/me/"
                verify_response = self.session.get(verify_url)
                
                if verify_response.status_code == 200:
                    user_data = verify_response.json()
                    self.authenticated = True
                    self.manager_id = user_data.get('player', {}).get('entry')
                    print(f"[OK] Authentication successful! Logged in as manager {self.manager_id}")
                    return True
                else:
                    print("[FAIL] Authentication failed: Could not verify login")
                    return False
            else:
                print(f"[FAIL] Authentication failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Authentication error: {e}")
            return False
    
    def execute_transfers(self, transfers_in: List[int], transfers_out: List[int]) -> bool:
        """Execute transfers via FPL API
        
        Args:
            transfers_in: List of player IDs to transfer in
            transfers_out: List of player IDs to transfer out
            
        Returns:
            True if transfers executed successfully, False otherwise
        """
        if not self.authenticated:
            print("[ERROR] Not authenticated. Please authenticate first.")
            return False
        
        if len(transfers_in) != len(transfers_out):
            print("[ERROR] Number of players in must equal number of players out")
            return False
        
        if not transfers_in or not transfers_out:
            print("[ERROR] No transfers specified")
            return False
        
        try:
            # Get current gameweek
            current_gw = self._get_current_gameweek()
            if not current_gw:
                print("[ERROR] Could not determine current gameweek")
                return False
            
            # Get CSRF token
            csrf_token = self.session.cookies.get('csrftoken')
            if not csrf_token:
                print("[ERROR] No CSRF token found. Please re-authenticate.")
                return False
            
            # Prepare transfer data
            # FPL API expects transfers as a list of dicts
            transfer_list = []
            for i in range(len(transfers_in)):
                transfer_list.append({
                    'element_in': transfers_in[i],
                    'element_out': transfers_out[i],
                    'purchase_price': None,  # Will be determined by API
                    'selling_price': None    # Will be determined by API
                })
            
            transfer_data = {
                'transfers': transfer_list,
                'chip': None,
                'entry': self.manager_id,
                'event': current_gw
            }
            
            # Execute transfers
            transfers_url = f"{self.fpl_base_url}/transfers/"
            headers = {
                'User-Agent': 'FPL-Bot/2.0.0',
                'Content-Type': 'application/json',
                'Referer': 'https://fantasy.premierleague.com/',
                'X-CSRFToken': csrf_token
            }
            
            print(f"\n[EXEC] Executing {len(transfers_in)} transfer(s)...")
            response = self.session.post(transfers_url, json=transfer_data, headers=headers)
            
            if response.status_code == 200:
                print(f"[OK] Transfers executed successfully!")
                return True
            else:
                error_data = response.json() if response.content else {}
                print(f"[FAIL] Transfer failed: HTTP {response.status_code}")
                if error_data:
                    print(f"       Error details: {error_data}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Transfer execution error: {e}")
            return False
    
    def get_team_for_execution(self, manager_id: int) -> Optional[Dict]:
        """Get team data needed for transfer execution
        
        Returns team picks with all necessary info for making transfers
        """
        try:
            current_gw = self._get_current_gameweek()
            if not current_gw:
                return None
            
            team_url = f"{self.fpl_base_url}/entry/{manager_id}/event/{current_gw}/picks/"
            response = self.session.get(team_url)
            response.raise_for_status()
            
            return response.json()
        except Exception as e:
            print(f"Error fetching team for execution: {e}")
            return None
