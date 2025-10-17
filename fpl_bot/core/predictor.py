"""
Advanced Points Predictor

Uses multiple factors to predict player points:
- Historical performance with price-quality signal
- Fixture difficulty (opponent strength)
- Home/away advantage
- Recent form
- Set piece roles
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class PointsPredictor:
    """Advanced points prediction using fixture analysis and player stats"""
    
    def __init__(self):
        # Position-specific base points
        self.position_base_points = {
            'GK': 2.0,
            'DEF': 2.0,
            'MID': 2.0,
            'FWD': 2.0
        }
        
        # Clean sheet points by position
        self.clean_sheet_points = {
            'GK': 4,
            'DEF': 4,
            'MID': 1,
            'FWD': 0
        }
        
    def predict_gameweek_points(
        self,
        players_df: pd.DataFrame,
        fixtures_df: pd.DataFrame,
        team_strengths_df: pd.DataFrame,
        gameweek: int
    ) -> pd.DataFrame:
        """Predict points for all players for a specific gameweek"""
        predictions = []
        
        # Get fixtures for this gameweek
        gw_fixtures = fixtures_df[fixtures_df['event'] == gameweek] if fixtures_df is not None else pd.DataFrame()
        
        for _, player in players_df.iterrows():
            # Check if player has a fixture
            fixture_info = self._get_player_fixture(player, gw_fixtures)
            
            if fixture_info is None:
                # No fixture = 0 points
                predictions.append({
                    'player_id': player['id'],
                    'gameweek': gameweek,
                    'predicted_points': 0.0,
                    'confidence': 1.0
                })
                continue
            
            # Predict points
            predicted_points = self._predict_player_points(player, fixture_info)
            confidence = self._calculate_confidence(player)
            
            predictions.append({
                'player_id': player['id'],
                'gameweek': gameweek,
                'predicted_points': predicted_points,
                'confidence': confidence
            })
        
        return pd.DataFrame(predictions)
    
    def _get_player_fixture(self, player: pd.Series, gw_fixtures: pd.DataFrame) -> Optional[Dict]:
        """Get fixture information for a player's team"""
        if gw_fixtures.empty:
            return None
        
        team_id = player['team']
        
        # Check if team is playing home or away
        home_fixture = gw_fixtures[gw_fixtures['team_h'] == team_id]
        away_fixture = gw_fixtures[gw_fixtures['team_a'] == team_id]
        
        if not home_fixture.empty:
            fixture = home_fixture.iloc[0]
            return {
                'is_home': True,
                'opponent_attack': fixture.get('team_a_attack', 3),
                'opponent_defence': fixture.get('team_a_defence', 3),
                'team_attack': fixture.get('team_h_attack', 3),
                'team_defence': fixture.get('team_h_defence', 3)
            }
        elif not away_fixture.empty:
            fixture = away_fixture.iloc[0]
        return {
                'is_home': False,
                'opponent_attack': fixture.get('team_h_attack', 3),
                'opponent_defence': fixture.get('team_h_defence', 3),
                'team_attack': fixture.get('team_a_attack', 3),
                'team_defence': fixture.get('team_a_defence', 3)
            }
        
        return None
    
    def _predict_player_points(self, player: pd.Series, fixture_info: Dict) -> float:
        """Predict points for a single player"""
        position = player['position_name']
        
        # Start with historical average
        total_points = player.get('total_points', 0)
        minutes = max(player.get('minutes', 1), 1)
        games_played = minutes / 90.0
        
        if games_played >= 3:
            avg_points_per_game = total_points / games_played
        else:
            avg_points_per_game = self.position_base_points.get(position, 2.0)
        
        # Blend with recent form
        form = float(player.get('form', avg_points_per_game))
        if games_played >= 3:
            base_points = avg_points_per_game * 0.7 + form * 0.3
        else:
            base_points = form
        
        # Team quality bonus based on team strength (from FPL API)
        # Better teams create more chances and keep more clean sheets
        # Scale by minutes - players who don't play don't benefit from team quality
        team_quality_bonus = self._calculate_team_quality_bonus(
            player, fixture_info, position
        )
        minutes_factor = self._estimate_minutes_factor(player)
        team_quality_bonus *= minutes_factor  # Scale team bonus by playing time
        
        # Ownership signal - if lots of managers own a player, there's a reason
        # High ownership = proven quality, low ownership = punts/differentials
        # Also scale by minutes - low ownership of non-playing players is a warning sign
        ownership_bonus = self._calculate_ownership_bonus(player)
        ownership_bonus *= minutes_factor  # Scale ownership bonus by playing time
        
        base_points += team_quality_bonus + ownership_bonus
        
        # Fixture difficulty
        fixture_factor = self._calculate_fixture_difficulty(player, fixture_info, position)
        
        # Set piece bonus
        set_piece_bonus = self._calculate_set_piece_bonus(player, position)
        
        # Home/away adjustment
        home_away_multiplier = 1.1 if fixture_info['is_home'] else 0.95
        
        # Minutes expectation (already calculated above for bonuses, reuse it)
        # minutes_factor already set above when scaling team/ownership bonuses
        
        # Combine all factors
        predicted_points = (
            base_points * 
            fixture_factor * 
            home_away_multiplier * 
            minutes_factor
        ) + set_piece_bonus
        
        # Clean sheet adjustment for defenders/keepers
        # Scale by minutes - can't keep a clean sheet if you don't play!
        if position in ['GK', 'DEF']:
            clean_sheet_prob = self._calculate_clean_sheet_probability(fixture_info)
            avg_cs_prob = 0.30
            cs_adjustment = (clean_sheet_prob - avg_cs_prob) * self.clean_sheet_points[position]
            cs_adjustment *= minutes_factor  # Scale CS bonus by playing time
            predicted_points += cs_adjustment
        
        return max(0.0, predicted_points)
    
    def _calculate_fixture_difficulty(self, player: pd.Series, fixture_info: Dict, position: str) -> float:
        """Calculate fixture difficulty multiplier"""
        if position in ['GK', 'DEF']:
            opponent_attack = fixture_info.get('opponent_attack', 3)
            difficulty_multiplier = 1.0 + (3 - opponent_attack) * 0.08
        else:
            opponent_defence = fixture_info.get('opponent_defence', 3)
            difficulty_multiplier = 1.0 + (3 - opponent_defence) * 0.08
        
        return max(0.75, min(1.25, difficulty_multiplier))
    
    def _calculate_set_piece_bonus(self, player: pd.Series, position: str) -> float:
        """Calculate bonus from set piece roles"""
        bonus = 0.0
        
        if player.get('on_penalties', False):
            penalty_order = player.get('penalty_order', 1)
            bonus += 0.8 if penalty_order == 1 else 0.3
        
        if player.get('on_corners', False) and position in ['MID', 'DEF']:
            corner_order = player.get('corner_order', 1)
            bonus += 0.5 if corner_order == 1 else 0.2
        
        if player.get('on_freekicks', False) and position in ['MID', 'FWD']:
            freekick_order = player.get('freekick_order', 1)
            bonus += 0.4 if freekick_order == 1 else 0.15
        
        return bonus
    
    def _estimate_minutes_factor(self, player: pd.Series) -> float:
        """Estimate likelihood of playing"""
        minutes_played = player.get('minutes', 0)
        
        if minutes_played > 500:
            return 1.0
        elif minutes_played > 200:
            return 0.85
        elif minutes_played > 50:
            return 0.5
        else:
            return 0.2
    
    def _calculate_clean_sheet_probability(self, fixture_info: Dict) -> float:
        """Calculate clean sheet probability"""
        team_defence = fixture_info.get('team_defence', 3)
        opponent_attack = fixture_info.get('opponent_attack', 3)
        is_home = fixture_info['is_home']
        
        base_prob = 0.30
        defence_factor = (team_defence - 3) * 0.06
        attack_factor = (3 - opponent_attack) * 0.06
        home_factor = 0.08 if is_home else -0.04
        
        probability = base_prob + defence_factor + attack_factor + home_factor
        return max(0.05, min(0.65, probability))
    
    def _calculate_team_quality_bonus(
        self, 
        player: pd.Series, 
        fixture_info: Dict, 
        position: str
    ) -> float:
        """Calculate bonus based on team quality
        
        Uses FPL's own team strength ratings (1000-1400 scale):
        - Stronger teams create more chances (helps attackers)
        - Stronger teams defend better (helps defenders)
        """
        # Average team strength is around 1200
        avg_strength = 1200
        
        if position in ['GK', 'DEF']:
            # Defenders benefit from strong team defence
            team_defence = fixture_info.get('team_defence', avg_strength)
            # Convert to bonus: +/- 1 point per 100 strength difference
            # Top teams (1380) get +1.8pts, poor teams (1060) get -1.4pts
            bonus = (team_defence - avg_strength) / 100.0
        else:
            # Attackers benefit from strong team attack
            team_attack = fixture_info.get('team_attack', avg_strength)
            # Convert to bonus: +/- 1 point per 100 strength difference
            # Top teams (1350) get +1.5pts, poor teams (1100) get -1.0pts
            bonus = (team_attack - avg_strength) / 100.0
        
        return bonus
    
    def _calculate_ownership_bonus(self, player: pd.Series) -> float:
        """Calculate bonus based on ownership %
        
        Highly owned players are owned for a reason - proven quality
        This prevents over-valuing budget over-performers with small sample sizes
        """
        ownership = float(player.get('selected_by_percent', 0))
        
        # Ownership-based bonus
        if ownership >= 30:
            # Elite ownership (template players)
            return 1.0
        elif ownership >= 15:
            # High ownership (popular picks)
            return 0.6
        elif ownership >= 5:
            # Moderate ownership (solid options)
            return 0.3
        else:
            # Low ownership (differentials/punts)
            return 0.0
    
    def _calculate_confidence(self, player: pd.Series) -> float:
        """Calculate prediction confidence based on playing time"""
        minutes = player.get('minutes', 0)
        
        if minutes > 500:
            return 0.9
        elif minutes > 200:
            return 0.7
        elif minutes > 50:
            return 0.4
        else:
            return 0.2
