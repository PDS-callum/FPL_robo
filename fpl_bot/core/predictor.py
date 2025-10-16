"""
Advanced Points Predictor

Uses multiple factors to predict player points:
- Fixture difficulty (opponent strength)
- Home/away advantage
- Recent form
- Set piece roles
- Position-specific scoring patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


class PointsPredictor:
    """Advanced points prediction using fixture analysis and player stats"""
    
    def __init__(self):
        # Position-specific base points (per 90 minutes)
        self.position_base_points = {
            'GK': 2.0,   # Clean sheet potential
            'DEF': 2.0,  # Clean sheet + attacking potential
            'MID': 2.0,  # Balanced
            'FWD': 2.0   # Goal-scoring focus
        }
        
        # Scoring multipliers by position
        self.goal_points = {
            'GK': 10,   # Rare but high value
            'DEF': 6,
            'MID': 5,
            'FWD': 4
        }
        
        self.assist_points = 3
        self.clean_sheet_points = {
            'GK': 4,
            'DEF': 4,
            'MID': 1,
            'FWD': 0
        }
        
        # Bonus and other scoring
        self.bonus_avg = 0.5  # Average bonus points per game
        
    def predict_gameweek_points(
        self,
        players_df: pd.DataFrame,
        fixtures_df: pd.DataFrame,
        team_strengths_df: pd.DataFrame,
        gameweek: int
    ) -> pd.DataFrame:
        """Predict points for all players for a specific gameweek
        
        Args:
            players_df: DataFrame with player data including set piece info
            fixtures_df: DataFrame with fixtures and team strengths
            team_strengths_df: DataFrame with team strength ratings
            gameweek: Gameweek number to predict
            
        Returns:
            DataFrame with player_id, gameweek, predicted_points, confidence
        """
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
                    'confidence': 1.0  # Certain they won't play
                })
                continue
            
            # Predict points based on multiple factors
            predicted_points = self._predict_player_points(
                player, fixture_info, team_strengths_df
            )
            
            # Calculate confidence based on minutes played consistency
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
                'opponent_id': fixture['team_a'],
                'opponent_attack': fixture.get('team_a_attack', 3),
                'opponent_defence': fixture.get('team_a_defence', 3),
                'team_attack': fixture.get('team_h_attack', 3),
                'team_defence': fixture.get('team_h_defence', 3)
            }
        elif not away_fixture.empty:
            fixture = away_fixture.iloc[0]
            return {
                'is_home': False,
                'opponent_id': fixture['team_h'],
                'opponent_attack': fixture.get('team_h_attack', 3),
                'opponent_defence': fixture.get('team_h_defence', 3),
                'team_attack': fixture.get('team_a_attack', 3),
                'team_defence': fixture.get('team_a_defence', 3)
            }
        
        return None
    
    def _predict_player_points(
        self,
        player: pd.Series,
        fixture_info: Dict,
        team_strengths_df: Optional[pd.DataFrame]
    ) -> float:
        """Predict points for a single player in a single fixture"""
        
        position = player['position_name']
        
        # Start with historical average (much stronger anchor to reality)
        total_points = player.get('total_points', 0)
        minutes = max(player.get('minutes', 1), 1)
        games_played = minutes / 90.0
        
        if games_played >= 3:
            # Use actual average points per game as strong baseline
            avg_points_per_game = total_points / max(games_played, 1)
        else:
            # Not enough data, use position average
            avg_points_per_game = self.position_base_points.get(position, 2.0)
        
        # Factor 1: Recent Form adjustment (modify baseline, not replace it)
        form = float(player.get('form', avg_points_per_game))
        if games_played >= 3:
            # Blend historical average with recent form (70% history, 30% recent)
            base_points = avg_points_per_game * 0.7 + form * 0.3
        else:
            base_points = form
        
        # Factor 2: Fixture Difficulty (moderate impact)
        fixture_factor = self._calculate_fixture_difficulty(
            player, fixture_info, position
        )
        
        # Factor 3: Set Piece Bonus
        set_piece_bonus = self._calculate_set_piece_bonus(player, position)
        
        # Factor 4: Home/Away adjustment (smaller impact)
        home_away_multiplier = 1.1 if fixture_info['is_home'] else 0.95
        
        # Factor 5: Minutes expectation
        minutes_factor = self._estimate_minutes_factor(player)
        
        # Combine factors - base points already include ALL historical scoring
        # We just adjust for:
        # 1. Fixture difficulty (is opponent weak/strong?)
        # 2. Home/away
        # 3. Playing time likelihood
        # 4. Set piece bonus (extra expected value)
        predicted_points = (
            base_points * 
            fixture_factor * 
            home_away_multiplier * 
            minutes_factor
        ) + set_piece_bonus
        
        # For defenders/keepers ONLY: add clean sheet expectation
        # (This isn't fully captured in historical average due to fixture variance)
        if position in ['GK', 'DEF']:
            clean_sheet_prob = self._calculate_clean_sheet_probability(
                fixture_info, position
            )
            # Adjust clean sheet value: historical avg already has ~some~ CS points
            # Only add the fixture-specific adjustment
            avg_cs_prob = 0.30  # League average
            cs_adjustment = (clean_sheet_prob - avg_cs_prob) * self.clean_sheet_points[position]
            predicted_points += cs_adjustment
        
        return max(0.0, predicted_points)  # Can't be negative
    
    def _calculate_fixture_difficulty(
        self,
        player: pd.Series,
        fixture_info: Dict,
        position: str
    ) -> float:
        """Calculate fixture difficulty multiplier (1.0 = average, >1 = easier, <1 = harder)
        
        Reduced impact to avoid over-weighting fixtures vs player quality
        """
        if position in ['GK', 'DEF']:
            # Defenders care about opponent attack strength
            opponent_attack = fixture_info.get('opponent_attack', 3)
            difficulty_multiplier = 1.0 + (3 - opponent_attack) * 0.08
        else:
            # Attackers care about opponent defence strength  
            opponent_defence = fixture_info.get('opponent_defence', 3)
            difficulty_multiplier = 1.0 + (3 - opponent_defence) * 0.08
        
        return max(0.75, min(1.25, difficulty_multiplier))
    
    def _calculate_set_piece_bonus(self, player: pd.Series, position: str) -> float:
        """Calculate bonus points from set piece roles"""
        bonus = 0.0
        
        # Penalties are valuable but not overwhelming
        if player.get('on_penalties', False):
            penalty_order = player.get('penalty_order', 1)
            if penalty_order == 1:
                bonus += 0.8  # Reduced from 1.5
            elif penalty_order == 2:
                bonus += 0.3  # Reduced from 0.5
        
        # Corners (mainly for assists)
        if player.get('on_corners', False) and position in ['MID', 'DEF']:
            corner_order = player.get('corner_order', 1)
            if corner_order == 1:
                bonus += 0.5  # Reduced from 0.8
            elif corner_order == 2:
                bonus += 0.2  # Reduced from 0.3
        
        # Free kicks
        if player.get('on_freekicks', False) and position in ['MID', 'FWD']:
            freekick_order = player.get('freekick_order', 1)
            if freekick_order == 1:
                bonus += 0.4  # Reduced from 0.6
            elif freekick_order == 2:
                bonus += 0.15  # Reduced from 0.2
        
        return bonus
    
    def _estimate_minutes_factor(self, player: pd.Series) -> float:
        """Estimate likelihood of playing based on recent minutes"""
        minutes_played = player.get('minutes', 0)
        
        # If they've played a lot, likely to play
        if minutes_played > 500:
            return 1.0  # Regular starter
        elif minutes_played > 200:
            return 0.85  # Rotational player
        elif minutes_played > 50:
            return 0.5  # Occasional player
        else:
            return 0.2  # Rarely plays
    
    def _calculate_clean_sheet_probability(
        self,
        fixture_info: Dict,
        position: str
    ) -> float:
        """Calculate probability of clean sheet
        
        More conservative - based on league average clean sheet rates
        """
        
        # Based on team defensive strength and opponent attack
        team_defence = fixture_info.get('team_defence', 3)
        opponent_attack = fixture_info.get('opponent_attack', 3)
        is_home = fixture_info['is_home']
        
        # Base probability - league average is around 30%
        base_prob = 0.30
        
        # Adjust for strengths (reduced impact)
        defence_factor = (team_defence - 3) * 0.06  # +/- 6% per strength point
        attack_factor = (3 - opponent_attack) * 0.06  # +/- 6% per opponent weakness
        home_factor = 0.08 if is_home else -0.04
        
        probability = base_prob + defence_factor + attack_factor + home_factor
        
        return max(0.05, min(0.65, probability))  # Realistic range: 5% to 65%
    
    def _calculate_confidence(self, player: pd.Series) -> float:
        """Calculate prediction confidence (0-1)
        
        Higher confidence for:
        - Players with consistent minutes
        - Players with stable form
        - Regular starters
        """
        minutes = player.get('minutes', 0)
        
        # Confidence based on playing time consistency
        if minutes > 500:
            return 0.9  # Very confident about regular starters
        elif minutes > 200:
            return 0.7  # Moderate confidence about rotational players
        elif minutes > 50:
            return 0.4  # Low confidence about fringe players
        else:
            return 0.2  # Very uncertain about rarely-used players

