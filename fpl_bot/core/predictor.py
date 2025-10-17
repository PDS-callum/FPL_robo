"""
Advanced Points Predictor - Rewritten

Based on approaches from successful FPL bots:
- Uses xG/xA (expected goals/assists) when available
- Better fixture difficulty using actual team performance
- Realistic clean sheet probabilities
- Improved minutes prediction
- Double gameweek support
- Evidence-based scoring models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


class PointsPredictor:
    """
    Advanced points prediction using evidence-based models
    
    Key improvements over naive approaches:
    1. Uses underlying stats (xG, xA, ICT) instead of just points
    2. Realistic fixture difficulty based on actual goals scored/conceded
    3. Position-specific scoring models calibrated to real data
    4. Proper clean sheet probability modeling
    5. Double gameweek support
    6. Minutes prediction based on rotation patterns
    """
    
    def __init__(self):
        # Realistic scoring points (from FPL rules)
        self.POINTS_PER_GOAL = {'GK': 6, 'DEF': 6, 'MID': 5, 'FWD': 4}
        self.POINTS_PER_ASSIST = 3
        self.POINTS_PER_CS = {'GK': 4, 'DEF': 4, 'MID': 1, 'FWD': 0}
        self.APPEARANCE_POINTS = 2  # 60+ minutes
        self.POINTS_PER_SAVE = 1.0 / 3.0  # 1 point per 3 saves (GKs only)
        
        # Bonus points (simplified - harder to predict accurately)
        self.AVG_BONUS_PER_GOAL = 1.5  # Goals often get bonus
        self.AVG_BONUS_PER_CS = 0.3    # Defenders in CS sometimes get bonus
        
        # Premier League averages (empirical data)
        self.AVG_GOALS_PER_GAME = 2.8
        self.AVG_HOME_GOALS = 1.6
        self.AVG_AWAY_GOALS = 1.2
        self.AVG_CS_RATE = 0.35  # ~35% of games end in clean sheet
        
        # Position-specific appearance rates (top players)
        self.POSITION_APPEARANCE_RATE = {
            'GK': 0.95,   # GKs rarely rotated
            'DEF': 0.85,  # Defenders fairly stable
            'MID': 0.80,  # Midfielders moderate rotation
            'FWD': 0.75   # Forwards more rotation
        }
        
    def predict_gameweek_points(
        self,
        players_df: pd.DataFrame,
        fixtures_df: pd.DataFrame,
        team_strengths_df: pd.DataFrame,
        gameweek: int
    ) -> pd.DataFrame:
        """
        Predict points for all players for a specific gameweek
        
        Approach:
        1. Get fixtures for gameweek (including doubles)
        2. For each fixture, predict:
           - Expected goals/assists (using xG if available, else historical)
           - Clean sheet probability
           - Minutes expectation
        3. Convert predictions to FPL points
        4. Add bonus points estimation
        """
        predictions = []
        
        # Get fixtures for this gameweek
        gw_fixtures = fixtures_df[fixtures_df['event'] == gameweek] if fixtures_df is not None else pd.DataFrame()
        
        # Build team performance lookup for better predictions
        team_stats = self._build_team_stats(players_df, fixtures_df, gameweek)
        
        for _, player in players_df.iterrows():
            # Get all fixtures for this player's team (handles double gameweeks)
            player_fixtures = self._get_player_fixtures(player, gw_fixtures)
            
            if not player_fixtures:
                # No fixture = 0 points (blank gameweek)
                predictions.append({
                    'player_id': player['id'],
                    'gameweek': gameweek,
                    'predicted_points': 0.0,
                    'confidence': 1.0
                })
                continue
            
            # Predict points across all fixtures
            total_predicted_points = 0.0
            for fixture_info in player_fixtures:
                fixture_points = self._predict_player_fixture_points(
                    player, fixture_info, team_stats
                )
                total_predicted_points += fixture_points
            
            # Calculate confidence
            confidence = self._calculate_confidence(player, len(player_fixtures))
            
            predictions.append({
                'player_id': player['id'],
                'gameweek': gameweek,
                'predicted_points': total_predicted_points,
                'confidence': confidence
            })
        
        return pd.DataFrame(predictions)
    
    def _get_player_fixtures(self, player: pd.Series, gw_fixtures: pd.DataFrame) -> List[Dict]:
        """
        Get all fixture information for a player's team
        Handles both single and double gameweeks
        """
        if gw_fixtures.empty:
            return []
        
        team_id = player['team']
        fixtures = []
        
        # Find all fixtures for this team
        team_fixtures = gw_fixtures[
            (gw_fixtures['team_h'] == team_id) | 
            (gw_fixtures['team_a'] == team_id)
        ]
        
        for _, fixture in team_fixtures.iterrows():
            is_home = fixture['team_h'] == team_id
            
            if is_home:
                fixture_info = {
                    'is_home': True,
                    'team_id': team_id,
                    'opponent_id': fixture['team_a'],
                    'opponent_attack': fixture.get('team_a_attack', 1200),
                    'opponent_defence': fixture.get('team_a_defence', 1200),
                    'team_attack': fixture.get('team_h_attack', 1200),
                    'team_defence': fixture.get('team_h_defence', 1200),
                }
            else:
                fixture_info = {
                    'is_home': False,
                    'team_id': team_id,
                    'opponent_id': fixture['team_h'],
                    'opponent_attack': fixture.get('team_h_attack', 1200),
                    'opponent_defence': fixture.get('team_h_defence', 1200),
                    'team_attack': fixture.get('team_a_attack', 1200),
                    'team_defence': fixture.get('team_a_defence', 1200),
                }
            
            fixtures.append(fixture_info)
        
        return fixtures
    
    def _build_team_stats(
        self, 
        players_df: pd.DataFrame, 
        fixtures_df: pd.DataFrame,
        current_gw: int
    ) -> Dict:
        """
        Build team-level statistics for better predictions
        
        Returns dict with:
        - avg_goals_scored: Average goals scored per game
        - avg_goals_conceded: Average goals conceded per game
        - clean_sheet_rate: % of games with clean sheet
        """
        team_stats = {}
        
        # Get unique teams
        teams = players_df['team'].unique()
        
        # Helper to safely convert to float for aggregations
        def safe_float_for_sum(value, default=0.0):
            try:
                return float(value) if value not in [None, '', 'None'] else default
            except (ValueError, TypeError):
                return default
        
        for team_id in teams:
            # Get team's players
            team_players = players_df[players_df['team'] == team_id].copy()
            
            # Ensure numeric types for aggregation
            if 'goals_scored' in team_players.columns:
                team_players['goals_scored'] = team_players['goals_scored'].apply(safe_float_for_sum)
            if 'minutes' in team_players.columns:
                team_players['minutes'] = team_players['minutes'].apply(safe_float_for_sum)
            
            # Calculate team attack strength
            # Use goalkeeper starts as true indicator of team games played
            gk_players = team_players[team_players['position_name'] == 'GK'].copy()
            if len(gk_players) > 0 and 'starts' in gk_players.columns:
                gk_players['starts'] = gk_players['starts'].apply(safe_float_for_sum)
                team_games = gk_players['starts'].max()
            else:
                # Fallback: estimate from max player minutes
                max_mins = team_players['minutes'].max() if 'minutes' in team_players.columns else 630
                team_games = max(max_mins / 90, 7)
            
            team_games = max(team_games, 1)  # Ensure at least 1 game
            
            # Sum goals from ALL players (this is team total)
            total_goals = team_players['goals_scored'].sum() if 'goals_scored' in team_players.columns else 0
            goals_per_game = total_goals / team_games
            
            # For defense, look at goalkeeper stats (most reliable)
            goalkeepers = team_players[team_players['position_name'] == 'GK'].copy()
            if len(goalkeepers) > 0:
                # Ensure numeric types
                if 'goals_conceded' in goalkeepers.columns:
                    goalkeepers['goals_conceded'] = goalkeepers['goals_conceded'].apply(safe_float_for_sum)
                if 'clean_sheets' in goalkeepers.columns:
                    goalkeepers['clean_sheets'] = goalkeepers['clean_sheets'].apply(safe_float_for_sum)
                if 'starts' in goalkeepers.columns:
                    goalkeepers['starts'] = goalkeepers['starts'].apply(safe_float_for_sum)
                
                # Use the goalkeeper with most starts
                main_gk = goalkeepers.loc[goalkeepers['starts'].idxmax()]
                games_played = main_gk.get('starts', 1)
                games_played = max(games_played, 1)
                
                avg_conceded = main_gk.get('goals_conceded', games_played * 1.4) / games_played
                clean_sheets = main_gk.get('clean_sheets', 0)
                cs_rate = clean_sheets / games_played
            else:
                # Fallback to defenders
                defenders = team_players[team_players['position_name'] == 'DEF'].copy()
                if len(defenders) > 0:
                    if 'goals_conceded' in defenders.columns:
                        defenders['goals_conceded'] = defenders['goals_conceded'].apply(safe_float_for_sum)
                    avg_conceded = defenders['goals_conceded'].mean()
                    cs_rate = self.AVG_CS_RATE
                else:
                    avg_conceded = 1.4
                    cs_rate = self.AVG_CS_RATE
            
            team_stats[team_id] = {
                'avg_goals_scored': min(3.0, max(0.5, goals_per_game)),
                'avg_goals_conceded': min(3.0, max(0.5, avg_conceded)),
                'clean_sheet_rate': min(0.6, max(0.1, cs_rate))
            }
        
        return team_stats
    
    def _predict_player_fixture_points(
        self,
        player: pd.Series,
        fixture_info: Dict,
        team_stats: Dict
    ) -> float:
        """
        Predict points for a single fixture
        
        Method:
        1. Predict minutes (will they play?)
        2. Predict attacking returns (goals + assists)
        3. Predict clean sheet (defenders/GKs)
        4. Add appearance points + bonus
        """
        position = player['position_name']
        
        # 1. Minutes prediction
        minutes_prob = self._predict_minutes_probability(player, position)
        
        if minutes_prob < 0.1:
            return 0.0  # Unlikely to play
        
        # 2. Predict attacking returns
        expected_goal_involvement = self._predict_goal_involvement(
            player, fixture_info, team_stats, position
        )
        
        # 3. Convert to points
        points = 0.0
        
        # Appearance points (if >60 min, which we approximate with minutes_prob)
        points += self.APPEARANCE_POINTS * minutes_prob
        
        # Goal involvement points
        expected_goals = expected_goal_involvement['goals']
        expected_assists = expected_goal_involvement['assists']
        
        # CRITICAL FIX: Scale by minutes probability!
        # Players who don't play shouldn't get full attacking points
        goal_points = expected_goals * self.POINTS_PER_GOAL[position] * minutes_prob
        assist_points = expected_assists * self.POINTS_PER_ASSIST * minutes_prob
        
        points += goal_points + assist_points
        
        # 4. Clean sheet points (defenders and GKs only)
        if position in ['GK', 'DEF']:
            cs_prob = self._predict_clean_sheet_probability(
                fixture_info, team_stats
            )
            cs_points = cs_prob * self.POINTS_PER_CS[position] * minutes_prob
            points += cs_points
        
        # 5. Goalkeeper save points
        if position == 'GK':
            expected_saves = self._predict_goalkeeper_saves(fixture_info, team_stats)
            save_points = expected_saves * self.POINTS_PER_SAVE * minutes_prob
            points += save_points
        
        # 6. Bonus points estimation (simplified)
        # Better players in good fixtures get more bonus
        # Note: expected_goals already scaled by minutes_prob above, so don't double-scale
        bonus_points = (
            (expected_goals / minutes_prob if minutes_prob > 0 else 0) * self.AVG_BONUS_PER_GOAL * minutes_prob +
            (cs_prob * self.AVG_BONUS_PER_CS if position in ['GK', 'DEF'] else 0) * minutes_prob
        )
        points += bonus_points
        
        # Scale everything by minutes probability
        # (already done above for most components)
        
        return max(0.0, points)
    
    def _predict_minutes_probability(self, player: pd.Series, position: str) -> float:
        """
        Predict probability of playing significant minutes (60+)
        
        Factors:
        - Historical minutes played
        - Recent starts
        - Position (GKs rarely rotated, forwards more often)
        - Injury status (already filtered out, but check news)
        """
        # Helper to safely convert API values to int
        def safe_int(value, default=0):
            try:
                return int(float(value)) if value not in [None, '', 'None'] else default
            except (ValueError, TypeError):
                return default
        
        # Get historical data (API may return strings)
        total_minutes = safe_int(player.get('minutes', 0))
        
        # Different API versions may use different field names
        starts = safe_int(player.get('starts', 0))
        if starts == 0:
            starts = safe_int(player.get('starts_per_90', 0))  # Alternative field name
        
        # Appearances might be in different fields
        appearances = safe_int(player.get('appearances', 0))
        if appearances == 0:
            # Estimate from minutes (90 min = 1 appearance)
            appearances = max(total_minutes / 60, 1) if total_minutes > 0 else 1
        appearances = max(appearances, max(starts, 1))  # Ensure starts <= appearances
        
        # Calculate minutes per appearance
        if appearances > 0:
            minutes_per_app = total_minutes / appearances
        else:
            # New player - use position average
            return self.POSITION_APPEARANCE_RATE.get(position, 0.5) * 0.5
        
        # Base probability on playing time
        if minutes_per_app >= 75:
            # Regular starter
            base_prob = 0.95
        elif minutes_per_app >= 60:
            # Frequent starter
            base_prob = 0.85
        elif minutes_per_app >= 30:
            # Squad rotation / super sub
            base_prob = 0.60
        elif minutes_per_app >= 10:
            # Fringe player
            base_prob = 0.30
        else:
            # Rarely plays
            base_prob = 0.10
        
        # Adjust for position (GKs more reliable)
        position_factor = self.POSITION_APPEARANCE_RATE.get(position, 0.80)
        
        # Combine
        probability = base_prob * (position_factor / 0.80)  # Normalize to MID baseline
        
        # Cap between 0.05 and 0.98
        return min(0.98, max(0.05, probability))
    
    def _predict_goal_involvement(
        self,
        player: pd.Series,
        fixture_info: Dict,
        team_stats: Dict,
        position: str
    ) -> Dict[str, float]:
        """
        Predict expected goals and assists for this fixture
        
        Uses xG/xA if available, otherwise falls back to historical rates
        adjusted for fixture difficulty
        """
        # Try to use expected stats first (xG, xA)
        # FPL API fields: expected_goals_per_90, expected_assists_per_90
        # Note: These might be named 'expected_goals' and 'expected_assists' in some API versions
        # Also note: API often returns these as strings, so we need careful type conversion
        xG_per_90 = 0
        xA_per_90 = 0
        
        # Helper function to safely convert to float
        def safe_float(value, default=0.0):
            try:
                return float(value) if value not in [None, '', 'None'] else default
            except (ValueError, TypeError):
                return default
        
        # Try different field name variations
        # CRITICAL: Only use per-90 stats if player has meaningful sample size
        total_minutes = max(safe_float(player.get('minutes', 1)), 1)
        has_enough_data = total_minutes >= 180  # At least 2 full games
        
        xG_per_90_raw = safe_float(player.get('expected_goals_per_90', 0))
        if xG_per_90_raw > 0 and has_enough_data:
            # Use API's per-90 stats (only if enough data!)
            xG_per_90 = xG_per_90_raw
            xA_per_90 = safe_float(player.get('expected_assists_per_90', 0))
        else:
            # Try total expected stats (need to convert to per 90)
            xG_total = safe_float(player.get('expected_goals', 0))
            xA_total = safe_float(player.get('expected_assists', 0))
            
            if (xG_total > 0 or xA_total > 0) and has_enough_data:
                # Convert total to per-90 (only if enough data!)
                xG_per_90 = (xG_total / total_minutes) * 90
                xA_per_90 = (xA_total / total_minutes) * 90
            else:
                # Not enough data - use position baseline
                xG_per_90 = 0
                xA_per_90 = 0
        
        # Fall back to actual stats if no xG available
        if xG_per_90 == 0 and xA_per_90 == 0:
            total_minutes = max(safe_float(player.get('minutes', 1)), 1)
            goals = safe_float(player.get('goals_scored', 0))
            assists = safe_float(player.get('assists', 0))
            
            # Same rule: need meaningful sample size (180+ mins = 2 full games)
            if total_minutes >= 180:
                xG_per_90 = (goals / total_minutes) * 90
                xA_per_90 = (assists / total_minutes) * 90
            else:
                # Not enough data - use very conservative baseline
                xG_per_90 = 0.1 if position in ['MID', 'FWD'] else 0.05
                xA_per_90 = 0.1 if position == 'MID' else 0.05
        
        # Adjust for fixture difficulty
        fixture_multiplier = self._calculate_fixture_multiplier(
            fixture_info, team_stats, position
        )
        
        expected_goals = xG_per_90 * fixture_multiplier
        expected_assists = xA_per_90 * fixture_multiplier
        
        # Add set piece bonus
        set_piece_boost = self._calculate_set_piece_boost(player, position)
        expected_goals += set_piece_boost['goals']
        expected_assists += set_piece_boost['assists']
        
        return {
            'goals': expected_goals,
            'assists': expected_assists
        }
    
    def _calculate_fixture_multiplier(
        self,
        fixture_info: Dict,
        team_stats: Dict,
        position: str
    ) -> float:
        """
        Calculate fixture difficulty multiplier
        
        Better approach than FPL strength ratings:
        - Use actual goals scored/conceded
        - Account for home/away
        - Different for attackers vs defenders
        """
        team_id = fixture_info['team_id']
        opponent_id = fixture_info['opponent_id']
        is_home = fixture_info['is_home']
        
        # Get team stats
        team_attack = team_stats.get(team_id, {}).get('avg_goals_scored', self.AVG_GOALS_PER_GAME / 2)
        opponent_defence = team_stats.get(opponent_id, {}).get('avg_goals_conceded', self.AVG_GOALS_PER_GAME / 2)
        
        # Expected goals for this matchup
        if is_home:
            expected_team_goals = (team_attack + opponent_defence) / 2 * 1.15  # Home advantage
        else:
            expected_team_goals = (team_attack + opponent_defence) / 2 * 0.90  # Away disadvantage
        
        # Compare to league average
        multiplier = expected_team_goals / (self.AVG_GOALS_PER_GAME / 2)
        
        # Position-specific adjustments
        if position in ['GK', 'DEF']:
            # Defenders benefit less from easy fixtures (inverted)
            # Good defense vs bad attack = less action = fewer bonus points
            # But more clean sheets (handled separately)
            multiplier = 0.8 + (multiplier * 0.2)  # Flatten the curve for defenders
        
        # Cap the multiplier to reasonable bounds
        return min(2.5, max(0.4, multiplier))
    
    def _calculate_set_piece_boost(self, player: pd.Series, position: str) -> Dict[str, float]:
        """
        Calculate boost from set piece roles
        
        Penalties are the most valuable, followed by corners and free kicks
        """
        boost = {'goals': 0.0, 'assists': 0.0}
        
        # Penalties (huge boost - ~50% of pens are scored)
        if player.get('on_penalties', False):
            penalty_order = player.get('penalty_order', 1)
            if penalty_order == 1:
                # First choice pen taker - expect ~0.2 pens per game, 75% conversion
                boost['goals'] += 0.15
            elif penalty_order == 2:
                # Backup - rare
                boost['goals'] += 0.03
        
        # Corners (good for assists, small goal threat)
        if player.get('on_corners', False) and position in ['MID', 'DEF']:
            corner_order = player.get('corner_order', 1)
            if corner_order == 1:
                boost['assists'] += 0.10
                if position == 'MID':
                    boost['goals'] += 0.03  # Direct from corner
            elif corner_order == 2:
                boost['assists'] += 0.04
        
        # Free kicks (direct scoring threat)
        if player.get('on_freekicks', False) and position in ['MID', 'FWD', 'DEF']:
            freekick_order = player.get('freekick_order', 1)
            if freekick_order == 1:
                boost['goals'] += 0.08
                boost['assists'] += 0.05
            elif freekick_order == 2:
                boost['goals'] += 0.03
                boost['assists'] += 0.02
        
        return boost
    
    def _predict_clean_sheet_probability(
        self,
        fixture_info: Dict,
        team_stats: Dict
    ) -> float:
        """
        Predict clean sheet probability using team defensive stats
        
        Better than fixed probabilities - uses actual team performance
        """
        team_id = fixture_info['team_id']
        opponent_id = fixture_info['opponent_id']
        is_home = fixture_info['is_home']
        
        # Get team defensive stats
        team_cs_rate = team_stats.get(team_id, {}).get('clean_sheet_rate', self.AVG_CS_RATE)
        opponent_attack = team_stats.get(opponent_id, {}).get('avg_goals_scored', self.AVG_GOALS_PER_GAME / 2)
        
        # Base probability from team's historical CS rate
        base_prob = team_cs_rate
        
        # Adjust for opponent strength
        # Weak opponent (0.8 goals/game) = +15% CS chance
        # Strong opponent (2.0 goals/game) = -15% CS chance
        avg_attack = self.AVG_GOALS_PER_GAME / 2
        opponent_factor = (avg_attack - opponent_attack) * 0.10
        
        # Home advantage for clean sheets
        home_factor = 0.08 if is_home else -0.05
        
        # Combine
        probability = base_prob + opponent_factor + home_factor
        
        # Realistic bounds (even best teams don't get 80% CS, even worst get some)
        return min(0.65, max(0.05, probability))
    
    def _predict_goalkeeper_saves(
        self,
        fixture_info: Dict,
        team_stats: Dict
    ) -> float:
        """
        Predict expected saves for a goalkeeper
        
        More shots against = more saves
        But better defense = fewer shots
        """
        opponent_id = fixture_info['opponent_id']
        
        # Opponent attack strength
        opponent_goals = team_stats.get(opponent_id, {}).get('avg_goals_scored', self.AVG_GOALS_PER_GAME / 2)
        
        # Rough model: ~5 shots per goal expected
        # Strong attack (2.0 goals) = ~10 shots = ~6 saves (assuming 60% save rate)
        expected_shots = opponent_goals * 5
        expected_saves = expected_shots * 0.6  # Average save percentage
        
        # Cap at reasonable bounds
        return min(10.0, max(1.0, expected_saves))
    
    def _calculate_confidence(self, player: pd.Series, num_fixtures: int) -> float:
        """
        Calculate prediction confidence
        
        More data = more confidence
        More fixtures = slightly less confidence per fixture
        """
        minutes = player.get('minutes', 0)
        
        # Base confidence on data available
        if minutes > 800:
            base_confidence = 0.90
        elif minutes > 400:
            base_confidence = 0.75
        elif minutes > 180:
            base_confidence = 0.60
        elif minutes > 90:
            base_confidence = 0.45
        else:
            base_confidence = 0.25
        
        # Double gameweeks are less predictable (rotation risk)
        if num_fixtures > 1:
            base_confidence *= 0.90
        
        return base_confidence
