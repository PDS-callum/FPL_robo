"""
Manager analysis module for FPL Bot

Analyzes manager's current team composition, transfer history, and performance
to inform transfer decisions and team optimization.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class ManagerAnalyzer:
    """Analyzes manager's team and performance data"""
    
    def __init__(self, data_collector):
        self.data_collector = data_collector
        self.manager_data = None
        self.current_team = None
        self.players_df = None
    
    def analyze_manager(self, manager_id: int, no_ft_gain_last_gw: bool = False) -> Dict:
        """Perform comprehensive manager analysis"""
        print(f"Analyzing manager {manager_id}...")
        
        # Get manager data
        self.manager_data = self.data_collector.get_manager_data(manager_id, no_ft_gain_last_gw=no_ft_gain_last_gw)
        if not self.manager_data:
            return {}
        
        # Get current season players data
        season_data = self.data_collector.get_current_season_data()
        if season_data:
            self.players_df = self.data_collector.create_players_dataframe(season_data)
        
        # Analyze current team
        team_analysis = self._analyze_current_team()
        
        # Analyze performance
        performance_analysis = self._analyze_performance()
        
        # Analyze transfer history
        transfer_analysis = self._analyze_transfers()
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        return {
            'manager_info': self._get_manager_info(),
            'current_team': self.manager_data.get('current_team', {}),
            'saved_transfers': self.manager_data.get('saved_transfers', {}),
            'team_analysis': team_analysis,
            'performance_analysis': performance_analysis,
            'transfer_analysis': transfer_analysis,
            'recommendations': recommendations
        }
    
    def _get_manager_info(self) -> Dict:
        """Extract basic manager information"""
        if not self.manager_data:
            return {}
        
        return {
            'manager_id': self.manager_data.get('id'),
            'team_name': self.manager_data.get('name'),
            'manager_name': f"{self.manager_data.get('player_first_name', '')} {self.manager_data.get('player_last_name', '')}".strip(),
            'overall_rank': self.manager_data.get('summary_overall_rank'),
            'total_points': self.manager_data.get('summary_overall_points'),
            'current_gameweek_points': self.manager_data.get('summary_event_points'),
            'team_value': self.manager_data.get('last_deadline_value', 0) / 10,
            'bank': self.manager_data.get('last_deadline_bank', 0) / 10,
            'saved_transfers': self.manager_data.get('saved_transfers', {'free_transfers': 1, 'total_available': 1})
        }
    
    def _analyze_current_team(self) -> Dict:
        """Analyze current team composition and performance"""
        if not self.manager_data or 'current_team' not in self.manager_data:
            return {}
        
        team_data = self.manager_data['current_team']
        picks = team_data.get('picks', [])
        
        if not picks or self.players_df is None:
            return {}
        
        # Get player details for current team
        team_players = []
        for pick in picks:
            player_info = self.players_df[self.players_df['id'] == pick['element']]
            if not player_info.empty:
                player = player_info.iloc[0].to_dict()
                player['pick_position'] = pick['position']  # Rename to avoid conflict
                player['is_captain'] = pick['is_captain']
                player['is_vice_captain'] = pick['is_vice_captain']
                team_players.append(player)
        
        # Analyze team composition
        team_analysis = {
            'total_value': sum(p['cost'] for p in team_players),
            'total_points': sum(p['total_points'] for p in team_players),
            'formation': self._get_formation(team_players),
            'players': team_players,
            'captain': next((p for p in team_players if p['is_captain']), None),
            'vice_captain': next((p for p in team_players if p['is_vice_captain']), None)
        }
        
        # Position analysis
        position_counts = {}
        position_points = {}
        position_costs = {}
        
        for player in team_players:
            pos = player['position_name']
            position_counts[pos] = position_counts.get(pos, 0) + 1
            position_points[pos] = position_points.get(pos, 0) + player['total_points']
            position_costs[pos] = position_costs.get(pos, 0) + player['cost']
        
        team_analysis['position_analysis'] = {
            'counts': position_counts,
            'points': position_points,
            'costs': position_costs
        }
        
        return team_analysis
    
    def _get_formation(self, team_players: List[Dict]) -> str:
        """Determine team formation from starting XI"""
        # Group by position (assuming positions 1-11 are starting XI)
        starting_xi = [p for p in team_players if p['pick_position'] <= 11]
        
        def_count = len([p for p in starting_xi if p['position_name'] == 'DEF'])
        mid_count = len([p for p in starting_xi if p['position_name'] == 'MID'])
        fwd_count = len([p for p in starting_xi if p['position_name'] == 'FWD'])
        
        return f"{def_count}-{mid_count}-{fwd_count}"
    
    def _analyze_performance(self) -> Dict:
        """Analyze manager's performance history"""
        if not self.manager_data or 'history' not in self.manager_data:
            return {}
        
        history = self.manager_data['history']
        current_season = history.get('current', [])
        
        if not current_season:
            return {}
        
        df = pd.DataFrame(current_season)
        
        performance = {
            'total_gameweeks': len(df),
            'average_points': df['points'].mean(),
            'best_gameweek': df['points'].max(),
            'worst_gameweek': df['points'].min(),
            'current_rank': df.iloc[-1]['rank'] if len(df) > 0 else None,
            'rank_change': df.iloc[-1]['rank'] - df.iloc[0]['rank'] if len(df) > 1 else 0
        }
        
        # Recent form (last 5 gameweeks)
        recent_games = df.tail(5)
        performance['recent_form'] = {
            'total_points': recent_games['points'].sum(),
            'average_points': recent_games['points'].mean(),
            'gameweeks': len(recent_games)
        }
        
        return performance
    
    def _analyze_transfers(self) -> Dict:
        """Analyze transfer history and patterns"""
        if not self.manager_data or 'transfers' not in self.manager_data:
            return {}
        
        transfers = self.manager_data['transfers']
        
        if not transfers:
            return {'total_transfers': 0, 'transfers_this_season': 0}
        
        # Count transfers this season
        current_season_transfers = [t for t in transfers if t.get('event') is not None]
        
        transfer_analysis = {
            'total_transfers': len(transfers),
            'transfers_this_season': len(current_season_transfers),
            'recent_transfers': transfers[-5:] if transfers else []
        }
        
        return transfer_analysis
    
    def _generate_recommendations(self) -> Dict:
        """Generate recommendations based on analysis"""
        recommendations = {
            'transfer_priorities': [],
            'captain_suggestions': [],
            'team_improvements': []
        }
        
        if not self.manager_data or self.players_df is None:
            return recommendations
        
        # Analyze team for potential improvements
        team_analysis = self._analyze_current_team()
        
        if team_analysis:
            # Find underperforming players
            for player in team_analysis['players']:
                if player['pick_position'] <= 11:  # Starting XI only
                    avg_points_for_position = self._get_avg_points_for_position(player['position_name'])
                    if player['total_points'] < avg_points_for_position * 0.7:  # 30% below average
                        recommendations['transfer_priorities'].append({
                            'player': player['web_name'],
                            'position': player['position_name'],
                            'current_points': player['total_points'],
                            'reason': 'Underperforming compared to position average'
                        })
        
        return recommendations
    
    def _get_avg_points_for_position(self, position: str) -> float:
        """Get average points for players in a specific position"""
        if self.players_df is not None:
            pos_players = self.players_df[self.players_df['position_name'] == position]
            return pos_players['total_points'].mean() if len(pos_players) > 0 else 0
        return 0
    
    def get_team_for_gameweek(self, manager_id: int, gameweek: int) -> Optional[Dict]:
        """Get team composition for a specific gameweek"""
        try:
            team_url = f"{self.data_collector.fpl_base_url}/entry/{manager_id}/event/{gameweek}/picks/"
            response = self.data_collector.session.get(team_url)
            response.raise_for_status()
            
            return response.json()
        except:
            return None
    
    def compare_teams(self, manager_id: int, gameweek1: int, gameweek2: int) -> Dict:
        """Compare team composition between two gameweeks"""
        team1 = self.get_team_for_gameweek(manager_id, gameweek1)
        team2 = self.get_team_for_gameweek(manager_id, gameweek2)
        
        if not team1 or not team2:
            return {}
        
        changes = []
        picks1 = {p['position']: p['element'] for p in team1.get('picks', [])}
        picks2 = {p['position']: p['element'] for p in team2.get('picks', [])}
        
        for position in range(1, 16):
            if picks1.get(position) != picks2.get(position):
                changes.append({
                    'position': position,
                    'player_out': picks1.get(position),
                    'player_in': picks2.get(position)
                })
        
        return {
            'gameweek1': gameweek1,
            'gameweek2': gameweek2,
            'changes': changes,
            'total_changes': len(changes)
        }
