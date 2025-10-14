"""
Prediction module for FPL Bot

Predicts player performance for the next gameweek based on:
- Current season form and statistics
- Fixture difficulty
- Player form and trends
- Team performance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import math


class Predictor:
    """Predicts player performance for upcoming gameweeks"""
    
    def __init__(self, data_collector):
        self.data_collector = data_collector
        self.players_df = None
        self.fixtures_df = None
        self.current_gameweek = None
        self.teams_data = None  # Store team data for fixture difficulty calculations
        
    def predict_next_gameweek(self, players_df: pd.DataFrame, fixtures_df: pd.DataFrame = None, teams_data: List[Dict] = None) -> pd.DataFrame:
        """Predict player performance for the next gameweek"""
        self.players_df = players_df.copy()
        self.current_gameweek = self._get_current_gameweek()
        
        if fixtures_df is not None:
            self.fixtures_df = fixtures_df.copy()
        
        if teams_data is not None:
            self.teams_data = {team['id']: team for team in teams_data}
        
        # Calculate predictions
        predictions = []
        
        for _, player in self.players_df.iterrows():
            prediction = self._predict_player_points(player)
            predictions.append(prediction)
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame(predictions)
        
        # Add confidence scores
        predictions_df['confidence'] = self._calculate_confidence(predictions_df)
        
        # Sort by predicted points
        predictions_df = predictions_df.sort_values('predicted_points', ascending=False)
        
        return predictions_df
    
    def predict_multi_gameweek(self, players_df: pd.DataFrame, fixtures_df: pd.DataFrame, teams_data: List[Dict], horizon: int = 7) -> Dict:
        """
        Predict player performance across multiple gameweeks with matchup analysis
        
        Returns:
            Dict with:
            - player_predictions: Dict mapping player_id to gameweek predictions
            - favorable_matchups: List of identified favorable matchup windows
        """
        try:
            self.players_df = players_df.copy()
            if isinstance(fixtures_df, pd.DataFrame):
                self.fixtures_df = fixtures_df.copy()
            else:
                self.fixtures_df = pd.DataFrame()
            self.teams_data = {team['id']: team for team in teams_data} if teams_data else {}
            self.current_gameweek = self._get_current_gameweek()
            
            if not self.current_gameweek:
                print("  WARNING: Could not determine current gameweek")
                return {'player_predictions': {}, 'favorable_matchups': []}
            
            # Step 1: Identify favorable matchup windows for teams
            favorable_matchups = self._identify_favorable_matchup_windows(horizon)
        except Exception as e:
            print(f"  ERROR in predict_multi_gameweek setup: {e}")
            import traceback
            traceback.print_exc()
            return {'player_predictions': {}, 'favorable_matchups': []}
        
        # Step 2: Predict player points for each gameweek, factoring in matchup windows
        player_predictions = {}
        
        try:
            for idx, (_, player) in enumerate(self.players_df.iterrows()):
                # Handle both 'id' (from raw data) and 'player_id' (from predictions DF)
                player_id = player.get('player_id', player.get('id'))
                if player_id is None:
                    continue
                    
                team_id = player.get('team')
                if team_id is None:
                    continue
                
                # Get player's matchup window bonus (if their team has favorable fixtures)
                matchup_bonus = self._get_matchup_window_bonus(team_id, favorable_matchups)
                
                # Predict for each gameweek in horizon
                gw_predictions = {}
                for offset in range(horizon):
                    gw = self.current_gameweek + offset
                    prediction = self._predict_player_for_gw(player, gw, matchup_bonus.get(gw, 0))
                    gw_predictions[gw] = prediction
                
                player_predictions[player_id] = {
                    'player_id': player_id,
                    'web_name': player.get('web_name', f'Player_{player_id}'),
                    'team': player.get('team_name', 'Unknown'),
                    'position': player.get('position_name', 'MID'),
                    'cost': player.get('cost', 0),
                    'gameweek_predictions': gw_predictions,
                    'total_horizon_points': sum(gw_predictions.values()),
                    'in_favorable_window': team_id in [m['team_id'] for m in favorable_matchups]
                }
            
            print(f"  Generated predictions for {len(player_predictions)} players over {horizon} gameweeks")
        except Exception as e:
            print(f"  ERROR in player predictions: {e}")
            import traceback
            traceback.print_exc()
            return {'player_predictions': {}, 'favorable_matchups': []}
        
        return {
            'player_predictions': player_predictions,
            'favorable_matchups': favorable_matchups
        }
    
    def _predict_player_points(self, player: pd.Series) -> Dict:
        """Predict points for a single player"""
        # Base prediction from current form
        base_points = self._calculate_base_prediction(player)
        
        # Adjust for fixture difficulty
        fixture_adjustment = self._calculate_fixture_adjustment(player)
        
        # Adjust for form trends
        form_adjustment = self._calculate_form_adjustment(player)
        
        # Adjust for team performance
        team_adjustment = self._calculate_team_adjustment(player)
        
        # Calculate final prediction
        predicted_points = base_points + fixture_adjustment + form_adjustment + team_adjustment
        
        # CALIBRATION: Apply position-specific caps (realistic maximums)
        position = player['position_name']
        cost = player.get('cost', 0)
        
        # Position caps (typical great gameweek)
        position_caps = {
            'GK': 8.0,   # Clean sheet + saves + bonus
            'DEF': 10.0, # Clean sheet + attacking return + bonus  
            'MID': 12.0, # Goal + assist + bonus
            'FWD': 11.0  # Goals + bonus
        }
        
        # Premium players (>£10m) get +2 to cap
        max_prediction = position_caps.get(position, 10.0)
        if cost >= 10.0:
            max_prediction += 2.0
        
        predicted_points = min(predicted_points, max_prediction)
        
        # CALIBRATION: Uncertainty scaling for low-minutes players
        minutes = player.get('minutes', 0)
        try:
            minutes = float(minutes) if minutes is not None else 0
        except (ValueError, TypeError):
            minutes = 0
            
        if minutes < 300:  # Less than ~3.5 full games
            predicted_points *= 0.85  # 15% haircut for uncertainty
        
        # Ensure minimum of 0 points
        predicted_points = max(0, predicted_points)
        
        return {
            'player_id': player['id'],
            'web_name': player['web_name'],
            'team': player['team'],  # Team ID needed for fixture analysis
            'team_name': player['team_name'],
            'position_name': player['position_name'],
            'cost': player['cost'],
            'predicted_points': round(predicted_points, 1),
            'base_points': round(base_points, 1),
            'fixture_adjustment': round(fixture_adjustment, 1),
            'form_adjustment': round(form_adjustment, 1),
            'team_adjustment': round(team_adjustment, 1),
            'value_prediction': round(predicted_points / player['cost'], 2)
        }
    
    def _calculate_base_prediction(self, player: pd.Series) -> float:
        """
        Calculate base prediction from player's current season performance
        
        CALIBRATED: Scaled to 65% of raw value for realistic expectations
        """
        # Use points per game as base
        ppg = player.get('points_per_game', 0)
        try:
            ppg = float(ppg) if ppg is not None else 0
        except (ValueError, TypeError):
            ppg = 0
        
        # Adjust based on minutes played
        minutes = player.get('minutes', 0)
        try:
            minutes = float(minutes) if minutes is not None else 0
        except (ValueError, TypeError):
            minutes = 0
            
        if minutes < 90:  # Less than full game average
            ppg *= (minutes / 90)
        
        # Weight recent form more heavily
        form = player.get('form', 0)
        # Convert form to float if it's a string
        try:
            form = float(form) if form is not None else 0
        except (ValueError, TypeError):
            form = 0
            
        if form > 0:
            # Blend PPG with form (70% PPG, 30% form)
            raw_base = (ppg * 0.7) + (form * 0.3)
        else:
            raw_base = ppg
        
        # IMPROVEMENT: Adaptive scaling based on consistency
        # Consistent performers get less regression, volatile players get more
        ict_index = player.get('ict_index', 0)
        try:
            ict_index = float(ict_index) if ict_index is not None else 0
        except (ValueError, TypeError):
            ict_index = 0
        
        # High ICT index (>80) suggests consistent involvement
        # High form (>5) with decent ICT suggests hot streak worth trusting
        if ict_index > 80 and form > 5:
            # Consistent, in-form players - less regression
            scaling_factor = 0.75
        elif form > 6:
            # Very hot form - trust it more
            scaling_factor = 0.72
        elif ppg > 5.5 and minutes > 600:
            # Regular high performers with minutes
            scaling_factor = 0.70
        else:
            # Standard regression for inconsistent/lower performers
            scaling_factor = 0.65
        
        base_points = raw_base * scaling_factor
        
        return base_points
    
    def _calculate_fixture_adjustment(self, player: pd.Series) -> float:
        """Adjust prediction based on fixture difficulty"""
        if self.fixtures_df is None:
            return 0
        
        # Get next fixture for player's team
        next_fixture = self._get_next_fixture(player['team'])
        
        if next_fixture is None:
            return 0
        
        # Calculate fixture difficulty
        difficulty = self._get_fixture_difficulty(next_fixture, player['team'])
        
        # Adjust based on position
        position = player['position_name']
        
        if position == 'GK':
            # Goalkeepers benefit from easy fixtures (clean sheets)
            adjustment = (5 - difficulty) * 0.5
        elif position == 'DEF':
            # Defenders also benefit from easy fixtures
            adjustment = (5 - difficulty) * 0.4
        elif position == 'MID':
            # Midfielders have moderate adjustment
            adjustment = (5 - difficulty) * 0.2
        else:  # FWD
            # Forwards benefit less from fixture difficulty
            adjustment = (5 - difficulty) * 0.1
        
        return adjustment
    
    def _calculate_form_adjustment(self, player: pd.Series) -> float:
        """
        Adjust prediction based on recent form trends
        
        CALIBRATED: Capped at ±2.5 points, reduced multiplier for regression to mean
        """
        # Get player's detailed stats for recent games
        detailed_stats = self.data_collector.get_player_detailed_stats(player['id'])
        
        if not detailed_stats or 'history' not in detailed_stats:
            return 0
        
        history = detailed_stats['history']
        if len(history) < 3:
            return 0
        
        # Calculate form trend from last 3 games
        recent_points = [game['total_points'] for game in history[-3:]]
        
        # Calculate trend (positive if improving, negative if declining)
        if len(recent_points) >= 2:
            trend = recent_points[-1] - recent_points[0]
            # CALIBRATION: Reduced multiplier (0.15 instead of 0.3) and cap at ±2.5
            # Hot streaks don't persist - regression to mean
            adjustment = trend * 0.15
            adjustment = max(-2.5, min(2.5, adjustment))
        else:
            adjustment = 0
        
        return adjustment
    
    def _calculate_team_adjustment(self, player: pd.Series) -> float:
        """Adjust prediction based on team performance"""
        if self.players_df is None:
            return 0
        
        # Get team's average performance
        team_players = self.players_df[self.players_df['team_name'] == player['team_name']]
        
        if len(team_players) == 0:
            return 0
        
        # Calculate team form
        team_avg_points = team_players['total_points'].mean()
        league_avg_points = self.players_df['total_points'].mean()
        
        # Adjust based on team performance vs league average
        team_factor = (team_avg_points - league_avg_points) / league_avg_points
        
        # Scale the adjustment
        adjustment = team_factor * 0.5
        
        return adjustment
    
    def _calculate_confidence(self, predictions_df: pd.DataFrame) -> List[float]:
        """Calculate confidence scores for predictions"""
        confidence_scores = []
        
        for _, prediction in predictions_df.iterrows():
            # Base confidence on player's consistency
            base_confidence = 0.5
            
            # Increase confidence for players with more games
            minutes = self.players_df[self.players_df['id'] == prediction['player_id']]['minutes'].iloc[0]
            try:
                minutes = float(minutes) if minutes is not None else 0
            except (ValueError, TypeError):
                minutes = 0
                
            if minutes > 1000:  # More than ~11 full games
                base_confidence += 0.2
            elif minutes > 500:  # More than ~5 full games
                base_confidence += 0.1
            
            # Increase confidence for players with consistent form
            form = self.players_df[self.players_df['id'] == prediction['player_id']]['form'].iloc[0]
            try:
                form = float(form) if form is not None else 0
            except (ValueError, TypeError):
                form = 0
                
            if form > 5:
                base_confidence += 0.1
            elif form < 2:
                base_confidence -= 0.1
            
            # Cap confidence between 0.1 and 0.9
            confidence = max(0.1, min(0.9, base_confidence))
            confidence_scores.append(confidence)
        
        return confidence_scores
    
    def _get_current_gameweek(self) -> Optional[int]:
        """Get current gameweek number"""
        try:
            response = self.data_collector.session.get(f"{self.data_collector.fpl_base_url}/bootstrap-static/")
            response.raise_for_status()
            data = response.json()
            events = data.get('events', [])
            
            # Prefer current if not finished
            current_ev = next((e for e in events if e.get('is_current', False)), None)
            if current_ev:
                if current_ev.get('finished', False):
                    # Use next unplayed week if current is finished
                    nxt = next((e for e in events if e.get('is_next', False) and not e.get('finished', False)), None)
                    if nxt:
                        return nxt.get('id')
                    # Fallback: first not finished event
                    rem = next((e for e in events if not e.get('finished', False)), None)
                    if rem:
                        return rem.get('id')
                # Current week is ongoing
                return current_ev.get('id')

            # No explicit current: choose next upcoming or first not finished
            nxt = next((e for e in events if e.get('is_next', False) and not e.get('finished', False)), None)
            if nxt:
                return nxt.get('id')
            rem = next((e for e in events if not e.get('finished', False)), None)
            if rem:
                return rem.get('id')

            # Last resort: use current-event field if present
            cur = data.get('current-event')
            if cur:
                return cur
            return None
        except:
            return None
    
    def _get_next_fixture(self, team_id: int) -> Optional[Dict]:
        """Get next fixture for a team"""
        if self.fixtures_df is None:
            return None
        
        # Filter fixtures for the team
        team_fixtures = self.fixtures_df[
            (self.fixtures_df['team_a'] == team_id) | (self.fixtures_df['team_h'] == team_id)
        ]
        
        # Get next unplayed fixture
        next_fixture = team_fixtures[team_fixtures['finished'] == False].iloc[0] if len(team_fixtures[team_fixtures['finished'] == False]) > 0 else None
        
        return next_fixture.to_dict() if next_fixture is not None else None
    
    def _get_fixture_difficulty(self, fixture: Dict, team_id: int) -> float:
        """
        Get fixture difficulty (1-5, where 5 is hardest) based on:
        - Opponent strength ratings (attack/defense)
        - League positions
        - Home vs away advantage
        """
        if self.teams_data is None:
            return 3.0  # Default to medium difficulty if no team data
        
        # Determine if playing home or away
        is_home = fixture.get('team_h') == team_id
        opponent_id = fixture.get('team_a') if is_home else fixture.get('team_h')
        
        # Get team and opponent data
        team_data = self.teams_data.get(team_id)
        opponent_data = self.teams_data.get(opponent_id)
        
        if not team_data or not opponent_data:
            return 3.0  # Default if data missing
        
        # Get strength ratings (FPL uses 800-1400 scale, need to normalize)
        # For attacking team: opponent's defense strength matters
        # For defensive team: opponent's attack strength matters
        if is_home:
            opponent_attack = opponent_data.get('strength_attack_away', 1000)
            opponent_defense = opponent_data.get('strength_defence_away', 1000)
            team_attack = team_data.get('strength_attack_home', 1000)
            team_defense = team_data.get('strength_defence_home', 1000)
        else:
            opponent_attack = opponent_data.get('strength_attack_home', 1000)
            opponent_defense = opponent_data.get('strength_defence_home', 1000)
            team_attack = team_data.get('strength_attack_away', 1000)
            team_defense = team_data.get('strength_defence_away', 1000)
        
        # Normalize from 800-1400 to 1-5 scale
        # 800 = 1 (very weak), 1100 = 3 (average), 1400 = 5 (very strong)
        try:
            opponent_attack_norm = ((opponent_attack - 800) / 150) + 1
            opponent_defense_norm = ((opponent_defense - 800) / 150) + 1
            
            # Clamp to 1-5 range
            opponent_attack_norm = max(1.0, min(5.0, opponent_attack_norm))
            opponent_defense_norm = max(1.0, min(5.0, opponent_defense_norm))
        except (ValueError, TypeError):
            opponent_attack_norm = 3.0
            opponent_defense_norm = 3.0
        
        # Calculate fixture difficulty based on opponent's strength
        # Higher opponent strength = harder fixture
        # Blend attack and defense strength (60% defense matters for difficulty, 40% attack)
        difficulty = (opponent_defense_norm * 0.6 + opponent_attack_norm * 0.4)
        
        # Adjust based on league positions if available
        try:
            team_position = team_data.get('position', 10)
            opponent_position = opponent_data.get('position', 10)
            
            # If opponent is higher up the table (lower position number), increase difficulty
            position_difference = team_position - opponent_position
            
            # Add up to ±0.5 difficulty based on position difference
            # If opponent is 10 places higher, add 0.5 difficulty
            position_adjustment = (position_difference / 20) * 0.5
            difficulty += position_adjustment
        except (TypeError, ValueError):
            pass  # Skip position adjustment if data not available
        
        # Consider home advantage (home fixtures are slightly easier)
        if is_home:
            difficulty -= 0.3
        else:
            difficulty += 0.3
        
        # Clamp to 1-5 range
        difficulty = max(1.0, min(5.0, difficulty))
        
        return difficulty
    
    def get_top_predictions(self, predictions_df: pd.DataFrame, position: str = None, limit: int = 10) -> pd.DataFrame:
        """Get top predictions, optionally filtered by position"""
        if position:
            filtered = predictions_df[predictions_df['position_name'] == position]
        else:
            filtered = predictions_df
        
        return filtered.head(limit)
    
    def get_value_predictions(self, predictions_df: pd.DataFrame, min_predicted_points: float = 3.0) -> pd.DataFrame:
        """Get players with good predicted value (points per cost)"""
        value_players = predictions_df[
            predictions_df['predicted_points'] >= min_predicted_points
        ].copy()
        
        # Sort by value prediction
        value_players = value_players.sort_values('value_prediction', ascending=False)
        
        return value_players
    
    def predict_captain_options(self, predictions_df: pd.DataFrame, current_team: List[int]) -> pd.DataFrame:
        """Get best captain options from current team"""
        # Filter to current team players only
        team_predictions = predictions_df[predictions_df['player_id'].isin(current_team)]
        
        # Sort by predicted points (captain gets double points)
        captain_options = team_predictions.sort_values('predicted_points', ascending=False)
        
        return captain_options.head(5)  # Top 5 captain options
    
    def get_fixture_difficulty_explanation(self, team_id: int) -> Dict:
        """Get detailed explanation of next fixture difficulty for a team"""
        if self.fixtures_df is None or self.teams_data is None:
            return {'error': 'Fixtures or teams data not available'}
        
        # Get next fixture
        next_fixture = self._get_next_fixture(team_id)
        if next_fixture is None:
            return {'error': 'No upcoming fixture found'}
        
        # Get difficulty
        difficulty = self._get_fixture_difficulty(next_fixture, team_id)
        
        # Get opponent info
        is_home = next_fixture.get('team_h') == team_id
        opponent_id = next_fixture.get('team_a') if is_home else next_fixture.get('team_h')
        team_data = self.teams_data.get(team_id, {})
        opponent_data = self.teams_data.get(opponent_id, {})
        
        return {
            'difficulty': round(difficulty, 2),
            'difficulty_rating': self._get_difficulty_rating(difficulty),
            'is_home': is_home,
            'venue': 'Home' if is_home else 'Away',
            'opponent_name': opponent_data.get('name', 'Unknown'),
            'opponent_position': opponent_data.get('position', 'N/A'),
            'team_position': team_data.get('position', 'N/A'),
            'opponent_strength_attack': opponent_data.get('strength_attack_home' if not is_home else 'strength_attack_away', 0),
            'opponent_strength_defence': opponent_data.get('strength_defence_home' if not is_home else 'strength_defence_away', 0)
        }
    
    def _get_difficulty_rating(self, difficulty: float) -> str:
        """Convert numerical difficulty to rating"""
        if difficulty <= 2.0:
            return 'Very Easy'
        elif difficulty <= 2.5:
            return 'Easy'
        elif difficulty <= 3.5:
            return 'Medium'
        elif difficulty <= 4.0:
            return 'Hard'
        else:
            return 'Very Hard'
    
    def _identify_favorable_matchup_windows(self, horizon: int) -> List[Dict]:
        """
        Identify windows where strong teams have favorable matchups against weaker opposition
        
        A favorable matchup window is defined as:
        - A top 10 team (by league position)
        - Playing 3+ consecutive games against teams in bottom half (pos 11+)
        - Average fixture difficulty <= 2.5
        
        Returns list of matchup windows with team info and gameweek range
        """
        if self.fixtures_df.empty or not self.teams_data:
            return []
        
        matchup_windows = []
        current_gw = self.current_gameweek
        end_gw = current_gw + horizon - 1
        
        # Only analyze top 10 teams (these are the teams worth targeting)
        top_teams = [t for t in self.teams_data.values() if t.get('position', 21) <= 10]
        
        for team in top_teams:
            team_id = team['id']
            
            # Get team's fixtures in the horizon
            team_fixtures = self.fixtures_df[
                ((self.fixtures_df['team_h'] == team_id) | (self.fixtures_df['team_a'] == team_id)) &
                (self.fixtures_df['event'] >= current_gw) &
                (self.fixtures_df['event'] <= end_gw) &
                (self.fixtures_df['finished'] == False)
            ].sort_values('event')
            
            if len(team_fixtures) < 3:
                continue
            
            # Analyze fixture sequence for favorable runs
            fixture_sequence = []
            for _, fixture in team_fixtures.iterrows():
                is_home = fixture['team_h'] == team_id
                opponent_id = fixture['team_a'] if is_home else fixture['team_h']
                opponent = self.teams_data.get(opponent_id, {})
                
                difficulty = self._get_fixture_difficulty(fixture.to_dict(), team_id)
                opponent_position = opponent.get('position', 15)
                
                fixture_sequence.append({
                    'gw': fixture['event'],
                    'opponent_id': opponent_id,
                    'opponent_name': opponent.get('name', 'Unknown'),
                    'opponent_position': opponent_position,
                    'difficulty': difficulty,
                    'is_home': is_home
                })
            
            # Find runs of 3+ favorable fixtures
            current_run = []
            for fixture in fixture_sequence:
                # Favorable if: difficulty <= 2.5 AND opponent in bottom half
                is_favorable = fixture['difficulty'] <= 2.5 and fixture['opponent_position'] >= 11
                
                if is_favorable:
                    current_run.append(fixture)
                else:
                    # Check if we have a valid run
                    if len(current_run) >= 3:
                        avg_difficulty = sum(f['difficulty'] for f in current_run) / len(current_run)
                        matchup_windows.append({
                            'team_id': team_id,
                            'team_name': team['name'],
                            'team_position': team.get('position', 10),
                            'start_gw': current_run[0]['gw'],
                            'end_gw': current_run[-1]['gw'],
                            'length': len(current_run),
                            'avg_difficulty': avg_difficulty,
                            'fixtures': current_run,
                            'quality_score': self._calculate_window_quality_score(team, current_run)
                        })
                    current_run = []
            
            # Check final run
            if len(current_run) >= 3:
                avg_difficulty = sum(f['difficulty'] for f in current_run) / len(current_run)
                matchup_windows.append({
                    'team_id': team_id,
                    'team_name': team['name'],
                    'team_position': team.get('position', 10),
                    'start_gw': current_run[0]['gw'],
                    'end_gw': current_run[-1]['gw'],
                    'length': len(current_run),
                    'avg_difficulty': avg_difficulty,
                    'fixtures': current_run,
                    'quality_score': self._calculate_window_quality_score(team, current_run)
                })
        
        # Sort by quality score (best windows first)
        matchup_windows.sort(key=lambda x: x['quality_score'], reverse=True)
        
        return matchup_windows
    
    def _calculate_window_quality_score(self, team: Dict, fixtures: List[Dict]) -> float:
        """
        Calculate quality score for a matchup window
        
        Factors:
        - Team strength (higher position = higher score)
        - Number of fixtures
        - Average difficulty (lower = higher score)
        """
        team_position = team.get('position', 15)
        team_score = max(0, 10 - team_position) * 2  # 0-18 points
        
        length_score = len(fixtures) * 3  # 9-21+ points for 3-7 fixtures
        
        avg_difficulty = sum(f['difficulty'] for f in fixtures) / len(fixtures)
        difficulty_score = (3.0 - avg_difficulty) * 5  # 0-10 points
        
        return team_score + length_score + difficulty_score
    
    def _get_matchup_window_bonus(self, team_id: int, favorable_matchups: List[Dict]) -> Dict[int, float]:
        """
        Get bonus points for each gameweek if team is in a favorable matchup window
        
        Returns dict mapping gameweek to bonus points
        """
        bonus_by_gw = {}
        
        for window in favorable_matchups:
            if window['team_id'] == team_id:
                # Apply bonus for each gameweek in the window
                for gw in range(window['start_gw'], window['end_gw'] + 1):
                    # Bonus increases with window quality
                    quality_score = window['quality_score']
                    bonus_by_gw[gw] = min(1.5, quality_score / 20)  # Cap at 1.5 points bonus
        
        return bonus_by_gw
    
    def _predict_player_for_gw(self, player: pd.Series, gw: int, matchup_bonus: float = 0) -> float:
        """
        Predict player points for a specific gameweek
        
        Args:
            player: Player data (can be from raw players_df or predictions_df)
            gw: Gameweek number
            matchup_bonus: Bonus points if in favorable matchup window
        """
        # Get base prediction - use predicted_points if available (from predictions_df)
        # otherwise calculate it
        if 'predicted_points' in player and pd.notna(player['predicted_points']):
            base_points = player['predicted_points']
        elif 'base_points' in player and pd.notna(player['base_points']):
            base_points = player['base_points']
        else:
            base_points = self._calculate_base_prediction(player)
        
        # Get fixture for this gameweek
        if self.fixtures_df.empty:
            return max(0, base_points + matchup_bonus)
        
        team_id = player.get('team')
        if team_id is None:
            return max(0, base_points + matchup_bonus)
            
        fixture = self.fixtures_df[
            ((self.fixtures_df['team_h'] == team_id) | (self.fixtures_df['team_a'] == team_id)) &
            (self.fixtures_df['event'] == gw)
        ]
        
        if fixture.empty:
            # No fixture this gameweek (blank gameweek)
            return 0
        
        fixture = fixture.iloc[0]
        
        # Calculate fixture adjustment
        fixture_difficulty = self._get_fixture_difficulty(fixture.to_dict(), team_id)
        position = player.get('position_name', 'MID')
        
        # Position-specific fixture adjustment
        if position == 'GK':
            fixture_adj = (5 - fixture_difficulty) * 0.5
        elif position == 'DEF':
            fixture_adj = (5 - fixture_difficulty) * 0.4
        elif position == 'MID':
            fixture_adj = (5 - fixture_difficulty) * 0.2
        else:  # FWD
            fixture_adj = (5 - fixture_difficulty) * 0.1
        
        # Combine: base + fixture + matchup window bonus
        total_prediction = base_points + fixture_adj + matchup_bonus
        
        # Apply calibration caps
        cost = player.get('cost', 0)
        position_caps = {
            'GK': 8.0, 'DEF': 10.0, 'MID': 12.0, 'FWD': 11.0
        }
        max_prediction = position_caps.get(position, 10.0)
        if cost >= 10.0:
            max_prediction += 2.0
        
        return max(0, min(total_prediction, max_prediction))
