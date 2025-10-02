import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json


class FPLChipManager:
    """
    Manages FPL chip usage decisions and team modifications for chip weeks.
    
    Chips available:
    - Wildcard: Complete team rebuild (2 per season, resets at GW19)
    - Free Hit: One-week team (2 per season, resets at GW19)
    - Triple Captain: Captain gets 3x points (2 per season, resets at GW19)
    - Bench Boost: All 15 players score points (2 per season, resets at GW19)
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.chip_state_file = f"{data_dir}/chip_state.json"
        self.chips_used = []
        self.last_reset_gameweek = 0
        self._load_chip_state()
    
    def _load_chip_state(self):
        """Load chip usage state from file"""
        try:
            if pd.io.common.file_exists(self.chip_state_file):
                with open(self.chip_state_file, 'r') as f:
                    data = json.load(f)
                    self.chips_used = data.get('chips_used', [])
                    self.last_reset_gameweek = data.get('last_reset_gameweek', 0)
            else:
                # Initialize with empty state
                self.chips_used = []
                self.last_reset_gameweek = 0
                self._save_chip_state()
        except Exception as e:
            print(f"⚠️ Could not load chip state: {e}")
            self.chips_used = []
            self.last_reset_gameweek = 0
    
    def _save_chip_state(self):
        """Save chip usage state to file"""
        try:
            data = {
                'chips_used': self.chips_used,
                'last_reset_gameweek': self.last_reset_gameweek
            }
            with open(self.chip_state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save chip state: {e}")
    
    def get_available_chips(self, gameweek: int) -> List[str]:
        """
        Get list of available chips for the given gameweek.
        Also handles chip reset at gameweek 19.
        """
        # Check if we need to reset chips (after GW19)
        if gameweek > 19 and self.last_reset_gameweek < 19:
            print("🔄 Resetting chips after Gameweek 19...")
            self.reset_chips()
        
        available = []
        for chip in ['wildcard', 'free_hit', 'triple_captain', 'bench_boost']:
            if self._is_chip_available(chip, gameweek):
                available.append(chip)
        
        return available
    
    def reset_chips(self):
        """Reset all chips (used at GW19)"""
        self.chips_used = []
        self.last_reset_gameweek = 19
        self._save_chip_state()
        print("✅ All chips have been reset!")
    
    def _is_chip_available(self, chip_name: str, gameweek: int) -> bool:
        """Check if a specific chip is available"""
        # Count how many times this chip has been used
        chip_count = sum(1 for chip in self.chips_used if chip['chip'] == chip_name)
        
        # Each chip can be used twice per season
        return chip_count < 2
    
    def should_use_chip(self, gameweek: int, team_data: pd.DataFrame, 
                       fixtures_data: pd.DataFrame, predictions: pd.DataFrame) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Determine if a chip should be used this gameweek.
        
        Returns:
        - Tuple of (chip_name, chip_config) if chip should be used
        - None if no chip should be used
        """
        available_chips = self.get_available_chips(gameweek)
        
        if not available_chips:
            return None
        
        # Check each available chip in priority order
        for chip in ['wildcard', 'free_hit', 'triple_captain', 'bench_boost']:
            if chip in available_chips:
                chip_decision = self._check_chip_usage(chip, gameweek, team_data, fixtures_data, predictions)
                if chip_decision:
                    return chip_decision
        
        return None
    
    def _check_chip_usage(self, chip_name: str, gameweek: int, team_data: pd.DataFrame, 
                         fixtures_data: pd.DataFrame, predictions: pd.DataFrame) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Check if a specific chip should be used"""
        if chip_name == 'wildcard':
            return self._check_wildcard_usage(gameweek, team_data, fixtures_data, predictions)
        elif chip_name == 'free_hit':
            return self._check_free_hit_usage(gameweek, team_data, fixtures_data, predictions)
        elif chip_name == 'triple_captain':
            return self._check_triple_captain_usage(gameweek, team_data, fixtures_data, predictions)
        elif chip_name == 'bench_boost':
            return self._check_bench_boost_usage(gameweek, team_data, fixtures_data, predictions)
        
        return None
    
    def _check_wildcard_usage(self, gameweek: int, team_data: pd.DataFrame, 
                             fixtures_data: pd.DataFrame, predictions: pd.DataFrame) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Check if wildcard should be used"""
        if gameweek < 2:
            return None  # Too early
        
        if gameweek > 15:
            return None  # Too late for first wildcard
        
        # If no previous team data, don't use wildcard (let normal transfers handle it)
        if team_data.empty or 'id' not in team_data.columns:
            return None
        
        # Check for blank gameweeks (teams not playing)
        blank_teams = self._identify_blank_teams(fixtures_data, gameweek)
        if len(blank_teams) > 0:
            # Count players from blank teams
            blank_players = 0
            if 'team' in team_data.columns:
                # Handle potential NaN values in team column
                team_data_clean = team_data.dropna(subset=['team'])
                blank_players = team_data_clean[team_data_clean['team'].isin(blank_teams)].shape[0]
            
            if blank_players > 3:  # More than 3 players from blank teams
                print(f"🎯 Wildcard recommended: {blank_players} players from blank teams")
                return "wildcard", {
                    "reason": "blank_gameweek",
                    "blank_players": blank_players,
                    "allows_unlimited_transfers": True
                }
        
        # Check for injury crisis
        if 'chance_of_playing_this_round' in team_data.columns:
            injured_players = team_data[team_data['chance_of_playing_this_round'] < 0.5].shape[0]
            if injured_players > 2:  # More than 2 players likely to miss
                print(f"🎯 Wildcard recommended: {injured_players} players injured/doubtful")
                return "wildcard", {
                    "reason": "injury_crisis",
                    "injured_players": injured_players,
                    "allows_unlimited_transfers": True
                }
        
        # Check if team is significantly underperforming based on recent performance
        # This would require historical data - for now, be more conservative
        team_value = self._calculate_team_value(team_data, predictions)
        optimal_value = self._calculate_optimal_team_value(predictions)
        
        # Only use wildcard if team is VERY underperforming (more conservative threshold)
        value_ratio = team_value / optimal_value if optimal_value > 0 else 1.0
        
        if value_ratio < 0.70:  # Team is 30% below optimal (more conservative)
            print(f"🎯 Wildcard recommended: Team value {value_ratio:.2f} below optimal")
            return "wildcard", {
                "reason": "team_underperforming",
                "value_ratio": value_ratio,
                "allows_unlimited_transfers": True
            }
        
        return None
    
    def _check_free_hit_usage(self, gameweek: int, team_data: pd.DataFrame, 
                             fixtures_data: pd.DataFrame, predictions: pd.DataFrame) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Check if free hit should be used"""
        # Free Hit logic: Use during blank gameweeks or double gameweeks
        blank_teams = self._identify_blank_teams(fixtures_data, gameweek)
        double_teams = self._identify_double_gameweek_teams(fixtures_data, gameweek)
        
        # Check if current team has many players from blank teams
        if team_data.empty or 'team' not in team_data.columns:
            blank_players = 0
        else:
            current_team_teams = set(team_data['team'].tolist())
            blank_players = len(current_team_teams.intersection(blank_teams))
        
        if blank_players >= 5:  # 5+ players from blank teams
            print(f"🎯 Free Hit recommended: {blank_players} players from blank teams")
            return "free_hit", {
                "reason": "blank_gameweek",
                "blank_players": blank_players,
                "allows_unlimited_transfers": True
            }
        
        # Check for double gameweek opportunity
        if len(double_teams) > 0:
            double_players = 0
            if 'team' in team_data.columns:
                current_team_teams = set(team_data['team'].tolist())
                double_players = len(current_team_teams.intersection(double_teams))
            
            if double_players < 3:  # Few players from double gameweek teams
                print(f"🎯 Free Hit recommended: Only {double_players} players from double gameweek teams")
                return "free_hit", {
                    "reason": "double_gameweek",
                    "double_players": double_players,
                    "allows_unlimited_transfers": True
                }
        
        return None
    
    def _check_triple_captain_usage(self, gameweek: int, team_data: pd.DataFrame, 
                                  fixtures_data: pd.DataFrame, predictions: pd.DataFrame) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Check if triple captain should be used"""
        # Triple Captain logic: Use when captain has excellent fixture
        captain = self._get_current_captain(team_data)
        if not captain:
            return None
        
        # Check if captain has double gameweek
        double_teams = self._identify_double_gameweek_teams(fixtures_data, gameweek)
        if captain.get('team') in double_teams:
            print(f"🎯 Triple Captain recommended: {captain.get('web_name', 'Captain')} has double gameweek")
            return "triple_captain", {
                "reason": "double_gameweek",
                "captain": captain,
                "multiplier": 3
            }
        
        # Check if captain has very high predicted points
        if 'predicted_points' in captain and captain['predicted_points'] > 8:
            print(f"🎯 Triple Captain recommended: {captain.get('web_name', 'Captain')} has high predicted points ({captain['predicted_points']:.1f})")
            return "triple_captain", {
                "reason": "high_predicted_points",
                "captain": captain,
                "multiplier": 3
            }
        
        return None
    
    def _check_bench_boost_usage(self, gameweek: int, team_data: pd.DataFrame, 
                                fixtures_data: pd.DataFrame, predictions: pd.DataFrame) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Check if bench boost should be used"""
        # Bench Boost logic: Use when bench players have good fixtures
        bench_players = self._get_bench_players(team_data)
        if len(bench_players) == 0:
            return None
        
        # Handle potential NaN values in id column
        bench_players_clean = bench_players.dropna(subset=['id'])
        if len(bench_players_clean) == 0:
            return None
        # Handle potential NaN values in predictions id column
        predictions_clean = predictions.dropna(subset=['id'])
        bench_predictions = predictions_clean[predictions_clean['id'].isin(bench_players_clean['id'])]
        if len(bench_predictions) == 0:
            return None
        
        # Calculate total bench points
        total_bench_points = bench_predictions['predicted_points'].sum()
        
        # Use bench boost if bench has good total points
        if total_bench_points >= 15:  # 15+ points from bench
            print(f"🎯 Bench Boost recommended: {total_bench_points:.1f} points from bench")
            return "bench_boost", {
                "reason": "strong_bench",
                "bench_points": total_bench_points,
                "multiplier": 1
            }
        
        return None
    
    def use_chip(self, chip_name: str, gameweek: int, team_data: pd.DataFrame, 
                 fixtures_data: pd.DataFrame, predictions: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Use a specific chip and return modified team data"""
        chip_decision = self._check_chip_usage(chip_name, gameweek, team_data, fixtures_data, predictions)
        
        if not chip_decision:
            return None
        
        chip_name, chip_config = chip_decision
        
        # Record chip usage
        self.chips_used.append({
            'chip': chip_name,
            'gameweek': gameweek,
            'timestamp': datetime.now().isoformat(),
            'config': chip_config
        })
        
        self._save_chip_state()
        print(f"✅ Recorded {chip_name} usage for gameweek {gameweek}")
        
        # Return chip configuration for team optimization
        return {
            'chip_used': chip_name,
            'chip_config': chip_config,
            'allows_unlimited_transfers': chip_config.get('allows_unlimited_transfers', False)
        }
    
    def _calculate_team_value(self, team_data: pd.DataFrame, predictions: pd.DataFrame) -> float:
        """Calculate current team's predicted value"""
        if team_data.empty or 'id' not in team_data.columns:
            return 0
        # Handle potential NaN values in id column
        team_data_clean = team_data.dropna(subset=['id'])
        if len(team_data_clean) == 0:
            return 0
        # Handle potential NaN values in predictions id column
        predictions_clean = predictions.dropna(subset=['id'])
        team_predictions = predictions_clean[predictions_clean['id'].isin(team_data_clean['id'])]
        return team_predictions['predicted_points'].sum() if len(team_predictions) > 0 else 0
    
    def _calculate_optimal_team_value(self, predictions: pd.DataFrame) -> float:
        """Calculate optimal team's predicted value (simplified)"""
        # Sort by predicted points and take top 15
        top_players = predictions.nlargest(15, 'predicted_points')
        return top_players['predicted_points'].sum()
    
    def _calculate_team_performance_ratio(self, team_data: pd.DataFrame, predictions: pd.DataFrame) -> float:
        """Calculate how well the current team is performing relative to optimal"""
        if team_data.empty or 'id' not in team_data.columns:
            return 1.0  # No team data means no underperformance
        
        team_value = self._calculate_team_value(team_data, predictions)
        optimal_value = self._calculate_optimal_team_value(predictions)
        
        if optimal_value <= 0:
            return 1.0
        
        return team_value / optimal_value
    
    def _identify_blank_teams(self, fixtures_data: pd.DataFrame, gameweek: int) -> set:
        """Identify teams that don't play this gameweek"""
        if len(fixtures_data) == 0:
            return set()
        
        # Handle NaN values in event column
        fixtures_clean = fixtures_data.dropna(subset=['event'])
        gameweek_fixtures = fixtures_clean[fixtures_clean['event'] == gameweek]
        playing_teams = set()
        
        for _, fixture in gameweek_fixtures.iterrows():
            if pd.notna(fixture['team_h']):
                playing_teams.add(int(fixture['team_h']))
            if pd.notna(fixture['team_a']):
                playing_teams.add(int(fixture['team_a']))
        
        # All teams that should be playing but aren't
        all_teams = set(range(1, 21))  # FPL has 20 teams
        return all_teams - playing_teams
    
    def _identify_double_gameweek_teams(self, fixtures_data: pd.DataFrame, gameweek: int) -> set:
        """Identify teams with double gameweeks"""
        if len(fixtures_data) == 0:
            return set()
        
        # Handle NaN values in event column
        fixtures_clean = fixtures_data.dropna(subset=['event'])
        gameweek_fixtures = fixtures_clean[fixtures_clean['event'] == gameweek]
        team_counts = {}
        
        for _, fixture in gameweek_fixtures.iterrows():
            if pd.notna(fixture['team_h']):
                team_counts[fixture['team_h']] = team_counts.get(fixture['team_h'], 0) + 1
            if pd.notna(fixture['team_a']):
                team_counts[fixture['team_a']] = team_counts.get(fixture['team_a'], 0) + 1
        
        # Teams with 2+ fixtures this gameweek
        return {team for team, count in team_counts.items() if count >= 2}
    
    def _get_current_captain(self, team_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Get current team captain"""
        if team_data.empty or 'is_captain' not in team_data.columns:
            return None
        captain_players = team_data[team_data['is_captain']]
        if len(captain_players) > 0:
            return captain_players.iloc[0].to_dict()
        return None
    
    def _get_bench_players(self, team_data: pd.DataFrame) -> pd.DataFrame:
        """Get current team bench players"""
        if team_data.empty or 'predicted_points' not in team_data.columns:
            return pd.DataFrame()
        # This is a simplified version - in reality, you'd need to track which 4 players are on the bench
        # For now, return the 4 players with lowest predicted points
        return team_data.nsmallest(4, 'predicted_points')
    
    def get_chip_usage_summary(self) -> Dict[str, Any]:
        """Get summary of chip usage"""
        return {
            'chips_used': self.chips_used,
            'last_reset_gameweek': self.last_reset_gameweek,
            'total_chips_used': len(self.chips_used)
        }