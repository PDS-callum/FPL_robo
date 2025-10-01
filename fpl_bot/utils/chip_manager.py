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
        self.chip_state_file = "chip_state.json"
        self.chip_state = self._load_chip_state()
        
    def _load_chip_state(self) -> Dict[str, Any]:
        """Load current chip usage state"""
        try:
            with open(self.chip_state_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Initialize with fresh chip state
            return {
                "chips_used": {
                    "wildcard": 0,
                    "free_hit": 0,
                    "triple_captain": 0,
                    "bench_boost": 0
                },
                "last_reset_gameweek": 0,
                "chip_usage_history": []
            }
    
    def _save_chip_state(self):
        """Save current chip usage state"""
        with open(self.chip_state_file, 'w') as f:
            json.dump(self.chip_state, f, indent=2)
    
    def get_available_chips(self, current_gameweek: int) -> Dict[str, bool]:
        """
        Get which chips are available for use.
        
        Args:
            current_gameweek: Current gameweek number
            
        Returns:
            Dict mapping chip names to availability
        """
        # Reset chips at gameweek 19
        if current_gameweek > 19 and self.chip_state["last_reset_gameweek"] < 19:
            self._reset_chips()
        
        # Each chip can be used twice per half-season
        max_uses = 2
        available = {}
        
        for chip in ["wildcard", "free_hit", "triple_captain", "bench_boost"]:
            used_count = self.chip_state["chips_used"].get(chip, 0)
            available[chip] = used_count < max_uses
            
        return available
    
    def _reset_chips(self):
        """Reset chip usage at gameweek 19"""
        print("🔄 Resetting chips at gameweek 19")
        self.chip_state["chips_used"] = {
            "wildcard": 0,
            "free_hit": 0,
            "triple_captain": 0,
            "bench_boost": 0
        }
        self.chip_state["last_reset_gameweek"] = 19
        self._save_chip_state()
    
    def should_use_chip(self, 
                       gameweek: int, 
                       team_data: pd.DataFrame, 
                       fixtures_data: pd.DataFrame,
                       predictions: pd.DataFrame) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Determine if a chip should be used this gameweek.
        
        Args:
            gameweek: Current gameweek
            team_data: Current team data
            fixtures_data: Fixtures data for the gameweek
            predictions: Player predictions
            
        Returns:
            Tuple of (chip_name, chip_config) if chip should be used, None otherwise
        """
        available_chips = self.get_available_chips(gameweek)
        
        # Check each chip in order of priority
        chip_checks = [
            self._check_wildcard_usage,
            self._check_free_hit_usage,
            self._check_triple_captain_usage,
            self._check_bench_boost_usage
        ]
        
        for check_func in chip_checks:
            result = check_func(gameweek, team_data, fixtures_data, predictions, available_chips)
            if result:
                chip_name, chip_config = result
                return chip_name, chip_config
        
        return None
    
    def _check_wildcard_usage(self, 
                            gameweek: int, 
                            team_data: pd.DataFrame, 
                            fixtures_data: pd.DataFrame,
                            predictions: pd.DataFrame,
                            available_chips: Dict[str, bool]) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Check if Wildcard should be used"""
        if not available_chips.get("wildcard", False):
            return None
        
        # Wildcard logic: Use early in season (GW4-8) or when team needs major overhaul
        if gameweek < 4:
            return None  # Too early
        
        if gameweek > 15:
            return None  # Too late for first wildcard
        
        # Check if team needs major changes
        team_value = self._calculate_team_value(team_data, predictions)
        optimal_value = self._calculate_optimal_team_value(predictions)
        
        # Use wildcard if current team is significantly underperforming
        value_ratio = team_value / optimal_value if optimal_value > 0 else 0
        
        if value_ratio < 0.85:  # Team is 15% below optimal
            print(f"🎯 Wildcard recommended: Team value {value_ratio:.2f} below optimal")
            return "wildcard", {
                "reason": "team_underperforming",
                "value_ratio": value_ratio,
                "allows_unlimited_transfers": True
            }
        
        return None
    
    def _check_free_hit_usage(self, 
                             gameweek: int, 
                             team_data: pd.DataFrame, 
                             fixtures_data: pd.DataFrame,
                             predictions: pd.DataFrame,
                             available_chips: Dict[str, bool]) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Check if Free Hit should be used"""
        if not available_chips.get("free_hit", False):
            return None
        
        # Free Hit logic: Use during blank gameweeks or double gameweeks
        blank_teams = self._identify_blank_teams(fixtures_data, gameweek)
        double_teams = self._identify_double_gameweek_teams(fixtures_data, gameweek)
        
        # Check if current team has many players from blank teams
        current_team_teams = set(team_data['team'].tolist())
        blank_players = len(current_team_teams.intersection(blank_teams))
        
        if blank_players >= 5:  # 5+ players from blank teams
            print(f"🎯 Free Hit recommended: {blank_players} players from blank teams")
            return "free_hit", {
                "reason": "blank_gameweek",
                "blank_players": blank_players,
                "blank_teams": list(blank_teams),
                "allows_unlimited_transfers": True
            }
        
        # Check for double gameweek opportunity
        if len(double_teams) >= 3:  # 3+ teams with double gameweeks
            print(f"🎯 Free Hit recommended: {len(double_teams)} teams with double gameweeks")
            return "free_hit", {
                "reason": "double_gameweek",
                "double_teams": list(double_teams),
                "allows_unlimited_transfers": True
            }
        
        return None
    
    def _check_triple_captain_usage(self, 
                                  gameweek: int, 
                                  team_data: pd.DataFrame, 
                                  fixtures_data: pd.DataFrame,
                                  predictions: pd.DataFrame,
                                  available_chips: Dict[str, bool]) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Check if Triple Captain should be used"""
        if not available_chips.get("triple_captain", False):
            return None
        
        # Triple Captain logic: Use when captain has excellent fixture and form
        captain = self._get_current_captain(team_data)
        if captain is None:
            return None
        
        captain_prediction = predictions[predictions['id'] == captain['id']]
        if len(captain_prediction) == 0:
            return None
        
        captain_points = captain_prediction.iloc[0]['predicted_points']
        
        # Use Triple Captain if captain is predicted to score 8+ points
        if captain_points >= 8.0:
            print(f"🎯 Triple Captain recommended: {captain['web_name']} predicted {captain_points:.1f} points")
            return "triple_captain", {
                "reason": "high_captain_prediction",
                "captain_name": captain['web_name'],
                "predicted_points": captain_points,
                "multiplier": 3
            }
        
        # Check for double gameweek captain
        double_teams = self._identify_double_gameweek_teams(fixtures_data, gameweek)
        if captain['team'] in double_teams and captain_points >= 6.0:
            print(f"🎯 Triple Captain recommended: {captain['web_name']} has double gameweek")
            return "triple_captain", {
                "reason": "double_gameweek_captain",
                "captain_name": captain['web_name'],
                "predicted_points": captain_points,
                "multiplier": 3
            }
        
        return None
    
    def _check_bench_boost_usage(self, 
                               gameweek: int, 
                               team_data: pd.DataFrame, 
                               fixtures_data: pd.DataFrame,
                               predictions: pd.DataFrame,
                               available_chips: Dict[str, bool]) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Check if Bench Boost should be used"""
        if not available_chips.get("bench_boost", False):
            return None
        
        # Bench Boost logic: Use when bench players have good fixtures
        bench_players = self._get_bench_players(team_data)
        if len(bench_players) == 0:
            return None
        
        bench_predictions = predictions[predictions['id'].isin(bench_players['id'])]
        if len(bench_predictions) == 0:
            return None
        
        # Calculate total bench points
        total_bench_points = bench_predictions['predicted_points'].sum()
        
        # Use Bench Boost if bench is predicted to score 15+ points
        if total_bench_points >= 15.0:
            print(f"🎯 Bench Boost recommended: Bench predicted {total_bench_points:.1f} points")
            return "bench_boost", {
                "reason": "strong_bench",
                "bench_points": total_bench_points,
                "bench_players": len(bench_players)
            }
        
        # Check for double gameweek bench players
        double_teams = self._identify_double_gameweek_teams(fixtures_data, gameweek)
        bench_teams = set(bench_players['team'].tolist())
        double_bench_teams = bench_teams.intersection(double_teams)
        
        if len(double_bench_teams) >= 2 and total_bench_points >= 12.0:
            print(f"🎯 Bench Boost recommended: {len(double_bench_teams)} bench teams with double gameweeks")
            return "bench_boost", {
                "reason": "double_gameweek_bench",
                "bench_points": total_bench_points,
                "double_bench_teams": list(double_bench_teams)
            }
        
        return None
    
    def apply_chip_to_team(self, 
                          chip_name: str, 
                          chip_config: Dict[str, Any],
                          team_data: pd.DataFrame,
                          predictions: pd.DataFrame,
                          fixtures_data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply chip effects to team selection.
        
        Args:
            chip_name: Name of chip being used
            chip_config: Configuration for the chip
            team_data: Current team data
            predictions: Player predictions
            fixtures_data: Fixtures data
            
        Returns:
            Modified team data
        """
        if chip_name == "wildcard":
            return self._apply_wildcard(team_data, predictions, chip_config)
        elif chip_name == "free_hit":
            return self._apply_free_hit(team_data, predictions, fixtures_data, chip_config)
        elif chip_name == "triple_captain":
            return self._apply_triple_captain(team_data, chip_config)
        elif chip_name == "bench_boost":
            return self._apply_bench_boost(team_data, predictions, chip_config)
        else:
            return team_data
    
    def _apply_wildcard(self, team_data: pd.DataFrame, predictions: pd.DataFrame, chip_config: Dict[str, Any]) -> pd.DataFrame:
        """Apply Wildcard - complete team rebuild"""
        print("🔄 Applying Wildcard - rebuilding entire team")
        # Wildcard allows complete team rebuild, so return original team
        # The team optimizer will handle the rebuild
        return team_data
    
    def _apply_free_hit(self, team_data: pd.DataFrame, predictions: pd.DataFrame, fixtures_data: pd.DataFrame, chip_config: Dict[str, Any]) -> pd.DataFrame:
        """Apply Free Hit - one-week team optimization"""
        print("🔄 Applying Free Hit - optimizing for this gameweek only")
        # Free Hit allows complete team rebuild for one week
        # The team optimizer will handle the rebuild
        return team_data
    
    def _apply_triple_captain(self, team_data: pd.DataFrame, chip_config: Dict[str, Any]) -> pd.DataFrame:
        """Apply Triple Captain - captain gets 3x points"""
        print(f"🔄 Applying Triple Captain - {chip_config['captain_name']} gets 3x points")
        # Triple Captain doesn't change team composition, just affects scoring
        return team_data
    
    def _apply_bench_boost(self, team_data: pd.DataFrame, predictions: pd.DataFrame, chip_config: Dict[str, Any]) -> pd.DataFrame:
        """Apply Bench Boost - all 15 players score points"""
        print(f"🔄 Applying Bench Boost - all 15 players will score points")
        # Bench Boost doesn't change team composition, just affects scoring
        return team_data
    
    def record_chip_usage(self, gameweek: int, chip_name: str, chip_config: Dict[str, Any]):
        """Record chip usage in state"""
        self.chip_state["chips_used"][chip_name] += 1
        self.chip_state["chip_usage_history"].append({
            "gameweek": gameweek,
            "chip": chip_name,
            "config": chip_config,
            "timestamp": datetime.now().isoformat()
        })
        self._save_chip_state()
        print(f"✅ Recorded {chip_name} usage for gameweek {gameweek}")
    
    def _calculate_team_value(self, team_data: pd.DataFrame, predictions: pd.DataFrame) -> float:
        """Calculate current team's predicted value"""
        team_predictions = predictions[predictions['id'].isin(team_data['id'])]
        return team_predictions['predicted_points'].sum() if len(team_predictions) > 0 else 0
    
    def _calculate_optimal_team_value(self, predictions: pd.DataFrame) -> float:
        """Calculate optimal team's predicted value (simplified)"""
        # Sort by predicted points and take top 15
        top_players = predictions.nlargest(15, 'predicted_points')
        return top_players['predicted_points'].sum()
    
    def _identify_blank_teams(self, fixtures_data: pd.DataFrame, gameweek: int) -> set:
        """Identify teams that don't play this gameweek"""
        if len(fixtures_data) == 0:
            return set()
        
        gameweek_fixtures = fixtures_data[fixtures_data['event'] == gameweek]
        playing_teams = set()
        
        for _, fixture in gameweek_fixtures.iterrows():
            playing_teams.add(fixture['team_h'])
            playing_teams.add(fixture['team_a'])
        
        # All teams that should be playing but aren't
        all_teams = set(range(1, 21))  # FPL has 20 teams
        return all_teams - playing_teams
    
    def _identify_double_gameweek_teams(self, fixtures_data: pd.DataFrame, gameweek: int) -> set:
        """Identify teams with double gameweeks"""
        if len(fixtures_data) == 0:
            return set()
        
        gameweek_fixtures = fixtures_data[fixtures_data['event'] == gameweek]
        team_counts = {}
        
        for _, fixture in gameweek_fixtures.iterrows():
            team_counts[fixture['team_h']] = team_counts.get(fixture['team_h'], 0) + 1
            team_counts[fixture['team_a']] = team_counts.get(fixture['team_a'], 0) + 1
        
        # Teams with 2+ fixtures this gameweek
        return {team for team, count in team_counts.items() if count >= 2}
    
    def _get_current_captain(self, team_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Get current team captain"""
        captain_players = team_data[team_data.get('is_captain', False)]
        if len(captain_players) > 0:
            return captain_players.iloc[0].to_dict()
        return None
    
    def _get_bench_players(self, team_data: pd.DataFrame) -> pd.DataFrame:
        """Get current team bench players"""
        # This is a simplified version - in reality, you'd need to track which 4 players are on the bench
        # For now, return the 4 players with lowest predicted points
        return team_data.nsmallest(4, 'predicted_points')
    
    def get_chip_usage_summary(self) -> Dict[str, Any]:
        """Get summary of chip usage"""
        return {
            "chips_used": self.chip_state["chips_used"],
            "total_chips_used": sum(self.chip_state["chips_used"].values()),
            "usage_history": self.chip_state["chip_usage_history"]
        }
