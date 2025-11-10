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
from scipy.stats import poisson


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
    
    def _detect_double_gameweeks(self, fixtures_df: pd.DataFrame, gameweeks: List[int]) -> Dict[int, List[int]]:
        """
        Detect double gameweeks for all teams across the planning horizon
        
        Args:
            fixtures_df: DataFrame with fixture data
            gameweeks: List of gameweeks to analyze
            
        Returns:
            Dict mapping gameweek to list of team IDs with double gameweeks
        """
        double_gw_teams = {}
        
        for gw in gameweeks:
            gw_fixtures = fixtures_df[fixtures_df['event'] == gw] if fixtures_df is not None else pd.DataFrame()
            
            if gw_fixtures.empty:
                continue
            
            # Count fixtures per team for this gameweek
            team_fixture_counts = {}
            
            for _, fixture in gw_fixtures.iterrows():
                team_h = fixture.get('team_h')
                team_a = fixture.get('team_a')
                
                if team_h:
                    team_fixture_counts[team_h] = team_fixture_counts.get(team_h, 0) + 1
                if team_a:
                    team_fixture_counts[team_a] = team_fixture_counts.get(team_a, 0) + 1
            
            # Find teams with 2+ fixtures (double gameweek)
            double_gw_teams[gw] = [team_id for team_id, count in team_fixture_counts.items() if count >= 2]
        
        return double_gw_teams
    
    def _get_player_fixtures(self, player: pd.Series, gw_fixtures: pd.DataFrame) -> List[Dict]:
        """
        Get all fixture information for a player's team with enhanced double gameweek handling
        Handles both single and double gameweeks with improved analysis
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
        
        # Check if this is a double gameweek
        is_double_gw = len(team_fixtures) > 1
        
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
                    'is_double_gw': is_double_gw,
                    'fixture_order': len(fixtures) + 1,  # Track order in double GW
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
                    'is_double_gw': is_double_gw,
                    'fixture_order': len(fixtures) + 1,  # Track order in double GW
                }
            
            fixtures.append(fixture_info)
        
        return fixtures
    
    def _calculate_recent_form(
        self,
        team_id: int,
        fixtures_df: pd.DataFrame,
        current_gw: int,
        team_stats: Dict
    ) -> float:
        """
        Calculate recent form factor based on last 5 finished games
        
        Returns:
            Form factor: 1.0 = neutral, >1.0 = good form, <1.0 = poor form
        """
        if fixtures_df is None or fixtures_df.empty:
            return 1.0
        
        # Get last 5 finished fixtures for this team (before current_gw)
        team_fixtures = fixtures_df[
            (fixtures_df['team_h'] == team_id) | (fixtures_df['team_a'] == team_id)
        ].copy()
        
        # Filter for finished fixtures before current gameweek
        finished_fixtures = team_fixtures[
            (team_fixtures['finished'] == True) & 
            (team_fixtures['event'] < current_gw)
        ].copy()
        
        if len(finished_fixtures) == 0:
            return 1.0  # No data, assume neutral form
        
        # Sort by gameweek descending and take last 5
        finished_fixtures = finished_fixtures.sort_values('event', ascending=False)
        recent_fixtures = finished_fixtures.head(5)
        
        if len(recent_fixtures) == 0:
            return 1.0
        
        # Calculate goals scored and conceded in recent fixtures
        recent_goals_scored = 0
        recent_goals_conceded = 0
        recent_games = 0
        
        for _, fixture in recent_fixtures.iterrows():
            # Safely get scores
            team_h_score = fixture.get('team_h_score')
            team_a_score = fixture.get('team_a_score')
            
            # Skip if scores not available
            if team_h_score is None or team_a_score is None:
                continue
            
            is_home = fixture['team_h'] == team_id
            
            if is_home:
                recent_goals_scored += team_h_score
                recent_goals_conceded += team_a_score
            else:
                recent_goals_scored += team_a_score
                recent_goals_conceded += team_h_score
            
            recent_games += 1
        
        if recent_games == 0:
            return 1.0
        
        # Calculate recent averages
        recent_goals_per_game = recent_goals_scored / recent_games
        recent_conceded_per_game = recent_goals_conceded / recent_games
        
        # Get season averages from team_stats
        season_goals_per_game = team_stats.get('avg_goals_scored', self.AVG_GOALS_PER_GAME / 2)
        season_conceded_per_game = team_stats.get('avg_goals_conceded', self.AVG_GOALS_PER_GAME / 2)
        
        # Calculate form factor: ratio of recent to season average
        # For attack: recent_goals / season_goals
        # For defense: season_conceded / recent_conceded (inverted because lower is better)
        attack_form = recent_goals_per_game / max(season_goals_per_game, 0.1)
        defence_form = season_conceded_per_game / max(recent_conceded_per_game, 0.1)
        
        # Combine attack and defense form (weighted average)
        # Attack slightly more important for form factor
        form_factor = (attack_form * 0.6 + defence_form * 0.4)
        
        # Cap form factor to reasonable bounds (0.7 to 1.3)
        return min(1.3, max(0.7, form_factor))
    
    def _build_team_stats(
        self, 
        players_df: pd.DataFrame, 
        fixtures_df: pd.DataFrame,
        current_gw: int
    ) -> Dict:
        """
        Build enhanced team-level statistics for better predictions
        
        Returns dict with:
        - avg_goals_scored: Average goals scored per game
        - avg_goals_conceded: Average goals conceded per game
        - clean_sheet_rate: % of games with clean sheet
        - home/away specific stats
        - recent form factors
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
            # Try to use fixture-level data first (more accurate)
            goals_per_game = None
            avg_conceded = None
            cs_rate = None
            team_games = 0
            
            if fixtures_df is not None and not fixtures_df.empty:
                # Get finished fixtures for this team
                team_fixtures = fixtures_df[
                    ((fixtures_df['team_h'] == team_id) | (fixtures_df['team_a'] == team_id)) &
                    (fixtures_df['finished'] == True) &
                    (fixtures_df['event'] < current_gw)
                ].copy()
                
                if len(team_fixtures) > 0:
                    # Calculate goals scored and conceded from fixture results
                    goals_scored = 0
                    goals_conceded = 0
                    clean_sheets = 0
                    
                    for _, fixture in team_fixtures.iterrows():
                        team_h_score = fixture.get('team_h_score')
                        team_a_score = fixture.get('team_a_score')
                        
                        # Skip if scores not available
                        if team_h_score is None or team_a_score is None:
                            continue
                        
                        is_home = fixture['team_h'] == team_id
                        
                        if is_home:
                            goals_scored += team_h_score
                            goals_conceded += team_a_score
                            if team_a_score == 0:
                                clean_sheets += 1
                        else:
                            goals_scored += team_a_score
                            goals_conceded += team_h_score
                            if team_h_score == 0:
                                clean_sheets += 1
                        
                        team_games += 1
                    
                    if team_games > 0:
                        goals_per_game = goals_scored / team_games
                        avg_conceded = goals_conceded / team_games
                        cs_rate = clean_sheets / team_games
            
            # Fallback to player aggregation if fixture data unavailable or insufficient
            if goals_per_game is None:
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
            
            # Fallback to player aggregation for defense if fixture data unavailable
            if avg_conceded is None:
                # Get team's players
                team_players = players_df[players_df['team'] == team_id].copy()
                
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
            
            # Ensure we have valid values
            if goals_per_game is None:
                goals_per_game = self.AVG_GOALS_PER_GAME / 2
            if avg_conceded is None:
                avg_conceded = self.AVG_GOALS_PER_GAME / 2
            if cs_rate is None:
                cs_rate = self.AVG_CS_RATE
            
            # Calculate home/away specific stats (simplified approach)
            # In a full implementation, you'd analyze actual home/away fixture results
            home_advantage_factor = 1.15  # Default home advantage
            away_disadvantage_factor = 0.90  # Default away disadvantage
            
            # Estimate home/away performance (simplified)
            goals_per_game_home = goals_per_game * home_advantage_factor
            goals_per_game_away = goals_per_game * away_disadvantage_factor
            
            # Store basic stats first (needed for form calculation)
            basic_stats = {
                'avg_goals_scored': min(3.0, max(0.5, goals_per_game)),
                'avg_goals_conceded': min(3.0, max(0.5, avg_conceded))
            }
            
            # Calculate recent form factor using fixture data
            recent_form_factor = self._calculate_recent_form(
                team_id, fixtures_df, current_gw, basic_stats
            )
            
            team_stats[team_id] = {
                'avg_goals_scored': basic_stats['avg_goals_scored'],
                'avg_goals_scored_home': min(3.0, max(0.5, goals_per_game_home)),
                'avg_goals_scored_away': min(3.0, max(0.5, goals_per_game_away)),
                'avg_goals_conceded': basic_stats['avg_goals_conceded'],
                'avg_goals_conceded_home': min(3.0, max(0.5, avg_conceded * away_disadvantage_factor)),
                'avg_goals_conceded_away': min(3.0, max(0.5, avg_conceded * home_advantage_factor)),
                'clean_sheet_rate': min(0.6, max(0.1, cs_rate)),
                'home_advantage_factor': home_advantage_factor,
                'away_disadvantage_factor': away_disadvantage_factor,
                'recent_form_factor': recent_form_factor
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
        
        # 1. Minutes prediction with double gameweek considerations
        minutes_prob = self._predict_minutes_probability(player, position)
        
        # Adjust for double gameweek rotation risk (position-specific)
        if fixture_info.get('is_double_gw', False):
            # Position-specific rotation risk
            # Get player's starts to determine if "nailed starter"
            def safe_int(value, default=0):
                try:
                    return int(float(value)) if value not in [None, '', 'None'] else default
                except (ValueError, TypeError):
                    return default
            
            starts = safe_int(player.get('starts', 0))
            is_nailed_starter = starts > 20  # More than 20 starts = regular starter
            
            # Position-specific rotation risk
            if position == 'GK':
                rotation_risk = 0.0  # GKs rarely rotated
            elif position == 'DEF':
                rotation_risk = 0.05 if is_nailed_starter else 0.10
            elif position == 'MID':
                rotation_risk = 0.15 if is_nailed_starter else 0.20
            else:  # FWD
                rotation_risk = 0.20 if is_nailed_starter else 0.25
            
            # Additional risk if fixtures are within 48 hours (not tracked in current data)
            # For now, assume standard double GW spacing
            minutes_prob *= (1.0 - rotation_risk)
        
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
        # Store unscaled values for bonus calculation
        expected_goals_unscaled = expected_goal_involvement['goals']
        expected_assists_unscaled = expected_goal_involvement['assists']
        
        # Scale by minutes probability for actual points
        expected_goals = expected_goals_unscaled * minutes_prob
        expected_assists = expected_assists_unscaled * minutes_prob
        
        # CRITICAL FIX: Scale by minutes probability!
        # Players who don't play shouldn't get full attacking points
        goal_points = expected_goals * self.POINTS_PER_GOAL[position]
        assist_points = expected_assists * self.POINTS_PER_ASSIST
        
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
        
        # 6. Bonus points estimation (FIXED: use unscaled goals for bonus calculation)
        # Bonus points should reflect actual goal involvement potential, then scale by minutes
        bonus_points = (
            expected_goals_unscaled * self.AVG_BONUS_PER_GOAL * minutes_prob +
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
        
        # Factor in chance_of_playing_next_round from FPL API
        chance_of_playing = player.get('chance_of_playing_next_round')
        if chance_of_playing is not None and chance_of_playing != '':
            try:
                chance_value = float(chance_of_playing)
                if chance_value <= 0:
                    # Injured or suspended
                    probability *= 0.1
                elif chance_value < 50:
                    # Doubtful
                    probability *= 0.5
                elif chance_value < 75:
                    # 50% chance, reduce probability
                    probability *= 0.75
                # 75%+ chance, no adjustment needed
            except (ValueError, TypeError):
                pass  # Invalid value, ignore
        
        # Factor in injury news
        news = player.get('news', '')
        if news and isinstance(news, str) and news.strip():
            news_lower = news.lower()
            # Check for injury-related keywords
            injury_keywords = ['injured', 'injury', 'doubt', 'doubtful', 'fitness', 'hamstring', 
                             'knee', 'ankle', 'calf', 'muscle', 'strain', 'suspended', 'ban']
            if any(keyword in news_lower for keyword in injury_keywords):
                # Reduce probability based on news severity
                if 'confirmed' in news_lower or 'ruled out' in news_lower:
                    probability *= 0.1  # Confirmed injury
                elif 'doubt' in news_lower or 'doubtful' in news_lower:
                    probability *= 0.5  # Doubtful
                else:
                    probability *= 0.75  # Minor concern
        
        # Consider recent minutes trend (if we have enough data)
        # For now, we use the overall minutes_per_app, but could enhance with last 3-5 games
        # This would require tracking historical gameweek data which isn't in current scope
        
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
        
        # Add set piece bonus (fixture-adjusted)
        set_piece_boost = self._calculate_set_piece_boost(player, position, fixture_info, team_stats)
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
        Calculate fixture difficulty multiplier with improved team-specific analysis
        
        Enhanced approach:
        - Use team-specific home/away performance
        - Account for recent form trends
        - Better opponent strength assessment
        - Position-specific adjustments
        """
        team_id = fixture_info['team_id']
        opponent_id = fixture_info['opponent_id']
        is_home = fixture_info['is_home']
        
        # Get team stats with home/away specific data
        team_stats_team = team_stats.get(team_id, {})
        opponent_stats = team_stats.get(opponent_id, {})
        
        # Base attack/defense stats
        team_attack = team_stats_team.get('avg_goals_scored', self.AVG_GOALS_PER_GAME / 2)
        opponent_defence = opponent_stats.get('avg_goals_conceded', self.AVG_GOALS_PER_GAME / 2)
        
        # Get home/away specific performance if available
        team_home_attack = team_stats_team.get('avg_goals_scored_home', team_attack)
        team_away_attack = team_stats_team.get('avg_goals_scored_away', team_attack)
        opponent_home_defence = opponent_stats.get('avg_goals_conceded_home', opponent_defence)
        opponent_away_defence = opponent_stats.get('avg_goals_conceded_away', opponent_defence)
        
        # Use appropriate home/away stats
        if is_home:
            effective_team_attack = team_home_attack
            effective_opponent_defence = opponent_away_defence
            home_advantage_factor = team_stats_team.get('home_advantage_factor', 1.15)
        else:
            effective_team_attack = team_away_attack
            effective_opponent_defence = opponent_home_defence
            home_advantage_factor = team_stats_team.get('away_disadvantage_factor', 0.90)
        
        # FIX: Convert "goals conceded" to "defensive weakness" (align scales)
        # opponent_defence = goals conceded per game (lower = stronger defense)
        # We want this in same direction as attack (higher = more goals expected)
        league_avg_goals = self.AVG_GOALS_PER_GAME / 2
        opponent_defensive_weakness = league_avg_goals - effective_opponent_defence
        
        # Calculate expected goals for this specific matchup
        # Now both metrics are aligned: higher = more goals expected
        expected_team_goals = (effective_team_attack + opponent_defensive_weakness) / 2 * home_advantage_factor
        
        # Apply recent form weighting if available
        # IMPORTANT: Form should enhance predictions, not override them
        # Use more conservative weighting to prevent form from completely overriding league position
        team_form_factor = team_stats_team.get('recent_form_factor', 1.0)
        opponent_form_factor = opponent_stats.get('recent_form_factor', 1.0)
        
        # Conservative form weighting: 50% recent form, 50% overall (was 70/30)
        # This prevents temporary form swings from completely overriding underlying team quality
        form_adjusted_goals = expected_team_goals * (0.5 * team_form_factor + 0.5 * 1.0)
        
        # Apply opponent form factor conservatively as well
        # Stronger teams should still be harder even if in poor form
        form_adjusted_goals = form_adjusted_goals * (0.5 * opponent_form_factor + 0.5 * 1.0)
        
        # Compare to league average
        multiplier = form_adjusted_goals / (self.AVG_GOALS_PER_GAME / 2)
        
        # Position-specific adjustments with improved logic
        if position in ['GK', 'DEF']:
            # Defenders: benefit more from clean sheets than attacking returns
            # Use inverted logic - easier fixtures = more clean sheets = higher multiplier
            # But cap the benefit to be more realistic
            multiplier = 0.6 + (multiplier * 0.4)  # More conservative flattening
        elif position == 'MID':
            # Midfielders: balanced benefit from both attack and defense
            multiplier = 0.8 + (multiplier * 0.2)  # Slight flattening
        else:  # FWD
            # Forwards: full benefit from attacking fixtures
            multiplier = multiplier  # No flattening for forwards
        
        # Cap the multiplier to realistic bounds
        return min(2.0, max(0.3, multiplier))
    
    def _calculate_set_piece_boost(
        self, 
        player: pd.Series, 
        position: str,
        fixture_info: Optional[Dict] = None,
        team_stats: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Calculate boost from set piece roles with fixture-adjusted scaling
        
        Enhanced approach:
        - Validates set piece data quality
        - Uses fallback mechanisms for missing data
        - Applies position-specific adjustments
        - Scales by opponent defensive strength (except penalties)
        """
        boost = {'goals': 0.0, 'assists': 0.0}
        
        # Helper function to safely get set piece data
        def safe_get_set_piece(value, default=False):
            try:
                if value is None or value == '' or value == 'None':
                    return default
                return bool(value)
            except (ValueError, TypeError):
                return default
        
        def safe_get_order(value, default=1):
            try:
                if value is None or value == '' or value == 'None':
                    return default
                return int(value)
            except (ValueError, TypeError):
                return default
        
        # Calculate opponent defensive weakness factor for scaling
        opponent_defensive_factor = 1.0  # Default: no adjustment
        if fixture_info and team_stats:
            opponent_id = fixture_info.get('opponent_id')
            if opponent_id and opponent_id in team_stats:
                opponent_stats = team_stats[opponent_id]
                opponent_defence = opponent_stats.get('avg_goals_conceded', self.AVG_GOALS_PER_GAME / 2)
                league_avg_goals = self.AVG_GOALS_PER_GAME / 2
                
                # Stronger defense (lower goals conceded) = fewer set piece opportunities
                # Weaker defense (higher goals conceded) = more set piece opportunities
                # Factor: 1.0 for average defense, >1.0 for weak defense, <1.0 for strong defense
                opponent_defensive_factor = 1.0 + (opponent_defence - league_avg_goals) / league_avg_goals
                # Cap between 0.7 and 1.3
                opponent_defensive_factor = min(1.3, max(0.7, opponent_defensive_factor))
        
        # Penalties (fixture-independent - referee decision)
        is_penalty_taker = safe_get_set_piece(player.get('on_penalties', False))
        if is_penalty_taker:
            penalty_order = safe_get_order(player.get('penalty_order', 1))
            if penalty_order == 1:
                # First choice pen taker - expect ~0.2 pens per game, 75% conversion
                boost['goals'] += 0.15  # Not scaled by fixture
            elif penalty_order == 2:
                # Backup - rare but still valuable
                boost['goals'] += 0.05  # Not scaled by fixture
            elif penalty_order == 3:
                # Third choice - very rare
                boost['goals'] += 0.02  # Not scaled by fixture
        
        # Corners (good for assists, small goal threat) - scale by opponent defense
        is_corner_taker = safe_get_set_piece(player.get('on_corners', False))
        if is_corner_taker and position in ['MID', 'DEF']:
            corner_order = safe_get_order(player.get('corner_order', 1))
            base_assists = 0.12 if corner_order == 1 else 0.06
            base_goals = 0.04 if (corner_order == 1 and position == 'MID') else (0.02 if (corner_order == 2 and position == 'MID') else 0.0)
            
            # Scale by opponent defensive weakness
            boost['assists'] += base_assists * opponent_defensive_factor
            boost['goals'] += base_goals * opponent_defensive_factor
        
        # Free kicks (direct scoring threat) - scale by opponent defense
        is_freekick_taker = safe_get_set_piece(player.get('on_freekicks', False))
        if is_freekick_taker and position in ['MID', 'FWD', 'DEF']:
            freekick_order = safe_get_order(player.get('freekick_order', 1))
            base_goals = 0.10 if freekick_order == 1 else 0.04
            base_assists = 0.06 if freekick_order == 1 else 0.03
            
            # Scale by opponent defensive weakness
            boost['goals'] += base_goals * opponent_defensive_factor
            boost['assists'] += base_assists * opponent_defensive_factor
        
        # Additional validation: Ensure boosts are realistic
        # Cap individual boosts to prevent unrealistic predictions
        boost['goals'] = min(boost['goals'], 0.25)  # Max 0.25 goals per game from set pieces
        boost['assists'] = min(boost['assists'], 0.20)  # Max 0.20 assists per game from set pieces
        
        return boost
    
    def _predict_clean_sheet_probability(
        self,
        fixture_info: Dict,
        team_stats: Dict
    ) -> float:
        """
        Predict clean sheet probability using Poisson distribution model
        
        Improved approach:
        - Uses Poisson distribution for realistic probability modeling
        - Calculates expected goals conceded based on team defence and opponent attack
        - Accounts for home/away factors
        - Uses form adjustments
        """
        team_id = fixture_info['team_id']
        opponent_id = fixture_info['opponent_id']
        is_home = fixture_info['is_home']
        
        # Get team defensive stats
        team_stats_data = team_stats.get(team_id, {})
        opponent_stats_data = team_stats.get(opponent_id, {})
        
        team_defence = team_stats_data.get('avg_goals_conceded', self.AVG_GOALS_PER_GAME / 2)
        
        # Get opponent attack stats (home/away specific if available)
        if is_home:
            opponent_attack = opponent_stats_data.get('avg_goals_scored_away', 
                                                     opponent_stats_data.get('avg_goals_scored', self.AVG_GOALS_PER_GAME / 2))
            # Home advantage: reduce expected goals conceded
            home_factor = team_stats_data.get('home_advantage_factor', 1.15)
            # Invert for defence: home advantage means conceding less
            home_defence_multiplier = 1.0 / home_factor
        else:
            opponent_attack = opponent_stats_data.get('avg_goals_scored_home',
                                                     opponent_stats_data.get('avg_goals_scored', self.AVG_GOALS_PER_GAME / 2))
            # Away disadvantage: increase expected goals conceded
            away_factor = team_stats_data.get('away_disadvantage_factor', 0.90)
            # Invert for defence: away disadvantage means conceding more
            home_defence_multiplier = 1.0 / away_factor
        
        # Calculate expected goals conceded using Poisson model
        # Base: team's defensive record adjusted by opponent's attack strength
        league_avg_goals = self.AVG_GOALS_PER_GAME / 2
        opponent_attack_ratio = opponent_attack / league_avg_goals
        
        # Expected goals conceded = team_defence * opponent_attack_ratio * home_factor
        expected_goals_conceded = team_defence * opponent_attack_ratio * home_defence_multiplier
        
        # Apply recent form adjustment if available
        recent_form_factor = team_stats_data.get('recent_form_factor', 1.0)
        opponent_form_factor = opponent_stats_data.get('recent_form_factor', 1.0)
        
        # Form affects both teams: better form = more goals scored/conceded
        # Weight recent form (70% recent, 30% overall)
        form_adjusted_goals_conceded = expected_goals_conceded * (
            0.7 * recent_form_factor + 0.3 * 1.0
        ) * (
            0.7 * opponent_form_factor + 0.3 * 1.0
        )
        
        # Use Poisson distribution: P(0 goals) = probability of clean sheet
        # Ensure lambda is positive and reasonable
        lambda_param = max(0.1, min(3.0, form_adjusted_goals_conceded))
        cs_prob = poisson.pmf(0, lambda_param)
        
        # Apply realistic bounds
        # Best defensive teams: max 70% CS probability
        # Worst defensive teams: min 5% CS probability
        return min(0.70, max(0.05, cs_prob))
    
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
