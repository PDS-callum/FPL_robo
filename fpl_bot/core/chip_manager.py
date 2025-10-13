"""
Chip management module for FPL Bot

Manages FPL chips (Wildcard, Free Hit, Triple Captain, Bench Boost) with:
- Christmas reset functionality
- Optimal chip usage recommendations
- Chip availability tracking
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date


class ChipManager:
    """Manages FPL chips and provides usage recommendations"""
    
    def __init__(self, data_collector):
        self.data_collector = data_collector
        self.chips = {
            'wildcard': {'available': True, 'used': False, 'reset_date': None},
            'free_hit': {'available': True, 'used': False, 'reset_date': None},
            'triple_captain': {'available': True, 'used': False, 'reset_date': None},
            'bench_boost': {'available': True, 'used': False, 'reset_date': None}
        }
        self.christmas_reset_date = date(2024, 12, 25)  # Adjust as needed for current season
        self.christmas_gameweek = 19  # Gameweek when chips reset (typically GW19)
    
    def get_chip_status(self, manager_id: int) -> Dict:
        """Get current chip status for a manager"""
        try:
            # Get manager data to check chip usage
            manager_data = self.data_collector.get_manager_data(manager_id)
            if not manager_data or 'history' not in manager_data:
                return self.chips
            
            # Check chip usage from history
            history = manager_data['history']
            current_season = history.get('current', [])
            
            # Track chip usage
            chip_usage = {
                'wildcard': False,
                'free_hit': False,
                'triple_captain': False,
                'bench_boost': False
            }
            
            # Check if chips were used this season
            for gameweek in current_season:
                if gameweek.get('event_transfers_cost', 0) == 0 and gameweek.get('event_transfers', 0) > 0:
                    # Likely wildcard used (free transfers but no cost)
                    chip_usage['wildcard'] = True
            
            # Update chip status
            current_date = date.today()
            
            for chip_name in self.chips:
                if chip_usage[chip_name]:
                    self.chips[chip_name]['used'] = True
                    self.chips[chip_name]['available'] = False
                elif self._should_reset_chip(chip_name, current_date):
                    self.chips[chip_name]['available'] = True
                    self.chips[chip_name]['used'] = False
                    self.chips[chip_name]['reset_date'] = None
            
            return self.chips
            
        except Exception as e:
            print(f"Error getting chip status: {e}")
            return self.chips
    
    def _should_reset_chip(self, chip_name: str, current_date: date) -> bool:
        """Check if a chip should be reset (Christmas reset)"""
        # All chips reset at Christmas
        if current_date >= self.christmas_reset_date:
            return True
        return False
    
    def _get_lookahead_window(self, current_gw: int) -> Tuple[int, int]:
        """
        Calculate the appropriate lookahead window for chip planning
        
        Returns:
            Tuple[int, int]: (start_gw, end_gw) for planning horizon
            
        Logic:
        - Before GW19: Show opportunities from current GW up to GW19
        - GW19 or after: Show opportunities from current GW for next ~8-10 weeks
        """
        if current_gw < self.christmas_gameweek:
            # Before Christmas - plan up to GW19
            start_gw = current_gw
            end_gw = self.christmas_gameweek
        else:
            # After Christmas - plan for next 8-10 gameweeks (rest of season)
            start_gw = current_gw
            end_gw = min(current_gw + 10, 38)  # Don't go beyond GW38
        
        return start_gw, end_gw
    
    def recommend_chip_usage(self, 
                           predictions_df: pd.DataFrame,
                           current_team: List[int],
                           fixtures_df: pd.DataFrame = None) -> Dict:
        """Recommend optimal chip usage for the current gameweek"""
        chip_recommendations = {}
        
        for chip_name in self.chips:
            if self.chips[chip_name]['available']:
                recommendation = self._get_chip_recommendation(
                    chip_name, predictions_df, current_team, fixtures_df
                )
                chip_recommendations[chip_name] = recommendation
        
        return chip_recommendations
    
    def _get_chip_recommendation(self, 
                               chip_name: str,
                               predictions_df: pd.DataFrame,
                               current_team: List[int],
                               fixtures_df: pd.DataFrame = None) -> Dict:
        """Get recommendation for a specific chip"""
        if chip_name == 'wildcard':
            return self._recommend_wildcard(predictions_df, current_team)
        elif chip_name == 'free_hit':
            return self._recommend_free_hit(predictions_df, current_team, fixtures_df)
        elif chip_name == 'triple_captain':
            return self._recommend_triple_captain(predictions_df, current_team)
        elif chip_name == 'bench_boost':
            return self._recommend_bench_boost(predictions_df, current_team)
        else:
            return {'recommended': False, 'reason': 'Unknown chip type'}
    
    def _recommend_wildcard(self, predictions_df: pd.DataFrame, current_team: List[int]) -> Dict:
        """Recommend whether to use Wildcard"""
        # Get current team performance
        current_team_df = predictions_df[predictions_df['player_id'].isin(current_team)]
        current_total_points = current_team_df['predicted_points'].sum()
        
        # Get optimal team from predictions
        optimal_team = self._get_optimal_team(predictions_df)
        optimal_total_points = optimal_team['predicted_points'].sum()
        
        # Calculate potential improvement
        improvement = optimal_total_points - current_total_points
        
        # Recommend if significant improvement is possible
        if improvement > 15:  # Threshold for wildcard recommendation
            return {
                'recommended': True,
                'reason': f'Team overhaul could gain {improvement:.1f} points',
                'current_points': current_total_points,
                'optimal_points': optimal_total_points,
                'expected_benefit': improvement
            }
        else:
            return {
                'recommended': False,
                'reason': f'Improvement only {improvement:.1f} points - not worth wildcard',
                'current_points': current_total_points,
                'optimal_points': optimal_total_points,
                'expected_benefit': improvement
            }
    
    def _recommend_free_hit(self, predictions_df: pd.DataFrame, current_team: List[int], fixtures_df: pd.DataFrame = None) -> Dict:
        """Recommend whether to use Free Hit"""
        # Free Hit is typically used for:
        # 1. Blank gameweeks (few teams playing)
        # 2. Double gameweeks (teams playing twice)
        # 3. When current team has many players not playing
        
        # For now, we'll implement a simplified version since we don't have reliable fixtures data
        # In a real implementation, you'd check actual fixture data for blank/double gameweeks
        
        # Check if we have enough fixtures data to make a proper recommendation
        if fixtures_df is not None and not fixtures_df.empty:
            # This is where you'd implement proper blank/double gameweek detection
            # For now, we'll be conservative and not recommend Free Hit unless we have clear data
            return {
                'recommended': False,
                'reason': 'No blank/double gameweek detected',
                'expected_benefit': 0
            }
        
        return {
            'recommended': False,
            'reason': 'No significant advantage identified for Free Hit',
            'expected_benefit': 0
        }
    
    def _recommend_triple_captain(self, predictions_df: pd.DataFrame, current_team: List[int]) -> Dict:
        """
        Recommend whether to use Triple Captain with future fixture planning
        
        Strategy:
        1. Identify premium players in current team (top performers)
        2. Look ahead at their fixtures for next 5 gameweeks
        3. Find the best week to use TC (player quality + fixture difficulty)
        4. Only recommend TC NOW if this is the best week
        5. Otherwise suggest SAVE for better opportunity
        """
        # Get captain options from current team
        captain_options = self._get_captain_options(predictions_df, current_team)
        
        if captain_options.empty:
            return {
                'recommended': False,
                'reason': 'No high-scoring captain options identified',
                'expected_benefit': 0
            }
        
        # Get fixtures data for lookahead
        fixtures_data = self.data_collector.get_fixtures_data()
        teams_data = self.data_collector.get_current_season_data()
        
        if not fixtures_data or not teams_data:
            # Fallback to old logic if no fixture data
            return self._recommend_triple_captain_basic(captain_options)
        
        # Analyze fixtures for premium players
        # Use top 10 to ensure we catch new signings and emerging players
        tc_analysis = self._analyze_tc_timing(
            captain_options.head(10),  # Top 10 players for comprehensive analysis
            fixtures_data,
            teams_data.get('teams', [])
        )
        
        if not tc_analysis:
            return self._recommend_triple_captain_basic(captain_options)
        
        best_week = tc_analysis['best_week']
        current_week_score = tc_analysis['current_week_score']
        
        # STRATEGIC TC CRITERIA:
        # 1. Elite player (predicted 7+ points)
        # 2. Easy fixture (difficulty <= 2.8)
        # 3. TC Score > 9.0 (quality + fixture combo)
        # Only use TC when ALL criteria met
        
        is_elite_player = best_week['predicted_points'] >= 7.0
        is_easy_fixture = best_week['difficulty'] <= 2.8
        is_high_tc_score = best_week['gameweek_score'] >= 9.0
        
        # Check if best week meets TC criteria
        meets_tc_criteria = is_elite_player and is_easy_fixture and is_high_tc_score
        
        # Decision logic
        if best_week['gameweek'] == 'current' and meets_tc_criteria:
            # This week is the best AND meets criteria - USE IT
            return {
                'recommended': True,
                'reason': f'{best_week["player"]} has optimal TC conditions: {best_week["predicted_points"]:.1f} pts vs {best_week["opponent"]} ({best_week["fixture_rating"]}, {best_week["venue"]})',
                'recommended_captain': best_week['player'],
                'predicted_points': best_week['predicted_points'],
                'expected_benefit': best_week['predicted_points'],
                'fixture_rating': best_week['fixture_rating'],
                'opponent': best_week['opponent'],
                'venue': best_week['venue'],
                'timing': 'NOW',
                'planning_details': tc_analysis  # Include for reporting
            }
        elif not meets_tc_criteria and best_week['gameweek'] != 'current':
            # Best week is future but doesn't meet criteria either
            future_meets_criteria = best_week['predicted_points'] >= 7.0 and best_week['difficulty'] <= 2.8
            
            if future_meets_criteria:
                # Future week is better and meets criteria - SAVE
                return {
                    'recommended': False,
                    'reason': f'SAVE TC: {best_week["player"]} vs {best_week["opponent"]} in GW{best_week["gameweek"]} ({best_week["fixture_rating"]}, {best_week["venue"]})',
                    'expected_benefit': 0,
                    'save_for': f'Gameweek {best_week["gameweek"]}',
                    'save_for_player': best_week['player'],
                    'save_for_opponent': best_week['opponent'],
                    'future_predicted': best_week['predicted_points'],
                    'fixture_rating': best_week['fixture_rating'],
                    'timing': f'GW{best_week["gameweek"]}'
                }
            else:
                # No good TC opportunity in next 5 weeks
                return {
                    'recommended': False,
                    'reason': f'No ideal TC opportunity found. Best: {best_week["player"]} vs {best_week["opponent"]} (Diff: {best_week["difficulty"]:.1f}) - waiting for elite player + easy fixture combo',
                    'expected_benefit': 0,
                    'planning_details': tc_analysis  # Include for reporting
                }
        elif best_week['gameweek'] == 'current' and not meets_tc_criteria:
            # This week is best but doesn't meet criteria
            # Check if future weeks are better
            future_weeks = [w for w in tc_analysis['all_weeks'] if w['gameweek'] != 'current']
            
            if future_weeks:
                best_future = future_weeks[0]
                future_meets = best_future['predicted_points'] >= 7.0 and best_future['difficulty'] <= 2.8
                
                if future_meets or best_future['gameweek_score'] > current_week_score + 2:
                    # Future is better or meets criteria
                    return {
                        'recommended': False,
                        'reason': f'SAVE TC: {best_future["player"]} vs {best_future["opponent"]} in GW{best_future["gameweek"]} is better ({best_future["fixture_rating"]}, {best_future["venue"]})',
                        'expected_benefit': 0,
                        'save_for': f'Gameweek {best_future["gameweek"]}',
                        'save_for_player': best_future['player'],
                        'save_for_opponent': best_future['opponent'],
                        'future_predicted': best_future['predicted_points'],
                        'fixture_rating': best_future['fixture_rating'],
                        'timing': f'GW{best_future["gameweek"]}'
                    }
            
            # No better future week found
            return {
                'recommended': False,
                'reason': f'Current best is {best_week["player"]} ({best_week["predicted_points"]:.1f} pts) vs {best_week["opponent"]} but fixture not ideal (Diff: {best_week["difficulty"]:.1f}). Waiting for better opportunity.',
                'expected_benefit': 0,
                'planning_details': tc_analysis  # Include for reporting
            }
        else:
            # Future week is significantly better
            return {
                'recommended': False,
                'reason': f'SAVE TC: {best_week["player"]} vs {best_week["opponent"]} in GW{best_week["gameweek"]} ({best_week["fixture_rating"]}, {best_week["venue"]})',
                'expected_benefit': 0,
                'save_for': f'Gameweek {best_week["gameweek"]}',
                'save_for_player': best_week['player'],
                'save_for_opponent': best_week['opponent'],
                'future_predicted': best_week['predicted_points'],
                'fixture_rating': best_week['fixture_rating'],
                'timing': f'GW{best_week["gameweek"]}',
                'planning_details': tc_analysis  # Include for reporting
            }
    
    def _recommend_triple_captain_basic(self, captain_options: pd.DataFrame) -> Dict:
        """Fallback TC recommendation without fixture lookahead"""
        best_captain = captain_options.iloc[0]
        expected_benefit = best_captain['predicted_points']
        
        # Basic logic: only recommend if predicted 8+ points  
        if best_captain['predicted_points'] >= 8:
                return {
                    'recommended': True,
                    'reason': f'{best_captain["web_name"]} has high predicted points ({best_captain["predicted_points"]:.1f})',
                    'recommended_captain': best_captain['web_name'],
                'predicted_points': best_captain['predicted_points'],
                'expected_benefit': expected_benefit,
                'timing': 'NOW'
                }
        
        return {
            'recommended': False,
            'reason': f'Best captain only predicted {best_captain["predicted_points"]:.1f} points',
            'expected_benefit': 0
        }
    
    def _recommend_bench_boost(self, predictions_df: pd.DataFrame, current_team: List[int]) -> Dict:
        """
        Recommend whether to use Bench Boost with strategic planning
        
        Strategy:
        1. Identify bench players (last 4 in squad)
        2. Look ahead at their fixtures for next 5 gameweeks
        3. Find week where bench will score most points
        4. Only recommend BB NOW if this is the best week
        5. Otherwise suggest SAVE for better fixtures
        """
        # Get bench players (positions 12-15)
        current_team_df = predictions_df[predictions_df['player_id'].isin(current_team)]
        bench_players = current_team_df.tail(4)  # Last 4 players as bench
        
        if len(bench_players) < 3:
            return {
                'recommended': False,
                'reason': 'Not enough bench players',
                'expected_benefit': 0
            }
        
        # Get fixtures data for lookahead
        fixtures_data = self.data_collector.get_fixtures_data()
        teams_data = self.data_collector.get_current_season_data()
        
        if not fixtures_data or not teams_data:
            # Fallback to basic logic if no fixture data
            return self._recommend_bench_boost_basic(bench_players)
        
        # Analyze bench boost timing
        bb_analysis = self._analyze_bench_boost_timing(
            bench_players,
            fixtures_data,
            teams_data.get('teams', [])
        )
        
        if not bb_analysis:
            return self._recommend_bench_boost_basic(bench_players)
        
        best_week = bb_analysis['best_week']
        current_week_score = bb_analysis['current_week_score']
        
        # STRATEGIC BB CRITERIA:
        # 1. Bench predicted 8+ points
        # 2. No significantly better week ahead (>2 points difference)
        
        is_good_bench = best_week['bench_score'] >= 8.0
        is_best_week = best_week['gameweek'] == 'current'
        future_much_better = (best_week['bench_score'] - current_week_score) > 2.0
        
        if is_best_week and is_good_bench:
            # This week is best - USE IT
            return {
                'recommended': True,
                'reason': f'Optimal BB week: bench predicted {best_week["bench_score"]:.1f} pts with favorable fixtures',
                'bench_points': best_week['bench_score'],
                'expected_benefit': best_week['bench_score'],
                'timing': 'NOW',
                'planning_details': bb_analysis  # Include for reporting
            }
        elif future_much_better:
            # Future week is significantly better - SAVE
            return {
                'recommended': False,
                'reason': f'SAVE BB: Bench will score more in GW{best_week["gameweek"]} ({best_week["bench_score"]:.1f} pts with better fixtures)',
                'expected_benefit': 0,
                'save_for': f'Gameweek {best_week["gameweek"]}',
                'future_predicted': best_week["bench_score"],
                'timing': f'GW{best_week["gameweek"]}',
                'planning_details': bb_analysis  # Include for reporting
            }
        elif is_good_bench:
            # Current week is good enough
            return {
                'recommended': True,
                'reason': f'Strong bench could add {current_week_score:.1f} points, no significantly better weeks ahead',
                'bench_points': current_week_score,
                'expected_benefit': current_week_score,
                'timing': 'NOW',
                'planning_details': bb_analysis  # Include for reporting
            }
        else:
            # Bench not strong enough
            return {
                'recommended': False,
                'reason': f'Bench only predicted {current_week_score:.1f} points - waiting for stronger bench week',
                'expected_benefit': 0,
                'planning_details': bb_analysis  # Include for reporting
            }
    
    def _recommend_bench_boost_basic(self, bench_players: pd.DataFrame) -> Dict:
        """Fallback BB recommendation without fixture lookahead"""
        bench_points = bench_players['predicted_points'].sum()
        expected_benefit = bench_points
        
        if bench_points >= 8:
            return {
                'recommended': True,
                'reason': f'Strong bench could add {bench_points:.1f} points',
                'bench_points': bench_points,
                'expected_benefit': expected_benefit,
                'timing': 'NOW'
            }
        
        return {
            'recommended': False,
            'reason': f'Bench only predicted {bench_points:.1f} points',
            'expected_benefit': 0
        }
    
    def _analyze_bench_boost_timing(self, bench_players: pd.DataFrame, fixtures_data: List[Dict], teams_data: List[Dict]) -> Optional[Dict]:
        """
        Analyze the best timing for Bench Boost over next 5 gameweeks
        
        Returns dict with:
        - best_week: The optimal week to use BB
        - current_week_score: Score for using BB this week
        - analysis: Breakdown of each week
        """
        try:
            # Get current gameweek
            current_gw = self._get_current_gameweek()
            if not current_gw:
                return None
            
            # Create teams lookup
            teams_dict = {team['id']: team for team in teams_data}
            
            # Analyze bench players across next N weeks
            weekly_analysis = {}
            
            for _, player in bench_players.iterrows():
                player_team_id = player.get('team')
                if not player_team_id:
                    continue
                
                # Get player's upcoming fixtures (dynamic window based on Christmas reset)
                upcoming_fixtures = self._get_upcoming_fixtures(
                    player_team_id,
                    fixtures_data,
                    current_gw
                )
                
                # Predict points for each gameweek based on fixture
                for fixture in upcoming_fixtures:
                    gw = fixture['event']
                    gw_key = gw if gw != current_gw else 'current'
                    
                    # Calculate fixture difficulty
                    is_home = fixture['team_h'] == player_team_id
                    opponent_id = fixture['team_a'] if is_home else fixture['team_h']
                    opponent = teams_dict.get(opponent_id, {})
                    
                    difficulty = self._calculate_fixture_difficulty_simple(
                        is_home,
                        opponent,
                        teams_dict.get(player_team_id, {})
                    )
                    
                    # Adjust player prediction for this specific fixture
                    # Easy fixtures help, hard fixtures hurt
                    base_prediction = player['predicted_points']
                    position = player.get('position_name', 'MID')
                    
                    # Fixture adjustment
                    if position in ['DEF', 'GK']:
                        fixture_adj = (3.0 - difficulty) * 0.4  # Max ±0.8
                    elif position == 'MID':
                        fixture_adj = (3.0 - difficulty) * 0.25  # Max ±0.5
                    else:  # FWD
                        fixture_adj = (3.0 - difficulty) * 0.15  # Max ±0.3
                    
                    adjusted_prediction = max(0, base_prediction + fixture_adj)
                    
                    # Add to weekly total
                    if gw_key not in weekly_analysis:
                        weekly_analysis[gw_key] = {
                            'gameweek': gw_key,
                            'bench_score': 0,
                            'player_count': 0,
                            'bench_players': []  # Track which players contribute
                        }
                    
                    weekly_analysis[gw_key]['bench_score'] += adjusted_prediction
                    weekly_analysis[gw_key]['player_count'] += 1
                    
                    # Store player info for this gameweek
                    weekly_analysis[gw_key]['bench_players'].append({
                        'web_name': player['web_name'],
                        'predicted_points': adjusted_prediction,
                        'position_name': player.get('position_name', 'UNK')
                    })
            
            if not weekly_analysis:
                return None
            
            # Convert to list and sort by bench score
            analysis_list = list(weekly_analysis.values())
            analysis_list.sort(key=lambda x: x['bench_score'], reverse=True)
            
            best_week = analysis_list[0]
            current_week_score = weekly_analysis.get('current', {}).get('bench_score', 0)
            
            return {
                'best_week': best_week,
                'current_week_score': current_week_score,
                'all_weeks': analysis_list[:5]
            }
            
        except Exception as e:
            print(f"Error analyzing BB timing: {e}")
            return None
    
    def _analyze_tc_timing(self, captain_options: pd.DataFrame, fixtures_data: List[Dict], teams_data: List[Dict]) -> Optional[Dict]:
        """
        Analyze the best timing for Triple Captain over next 5 gameweeks
        
        Returns dict with:
        - best_week: The optimal week to use TC
        - current_week_score: Score for using TC this week
        - analysis: Breakdown of each week
        """
        try:
            # Get current gameweek
            current_gw = self._get_current_gameweek()
            if not current_gw:
                return None
            
            # Create teams lookup
            teams_dict = {team['id']: team for team in teams_data}
            
            # Analyze each captain option across next N weeks
            analysis = []
            
            for _, player in captain_options.iterrows():
                player_team_id = player.get('team')
                player_name = player.get('web_name', 'Unknown')
                if not player_team_id:
                    continue
                
                # Get player's upcoming fixtures (dynamic window based on Christmas reset)
                upcoming_fixtures = self._get_upcoming_fixtures(
                    player_team_id,
                    fixtures_data,
                    current_gw
                )
                
                # Evaluate each gameweek
                for fixture in upcoming_fixtures:
                    gw = fixture['event']
                    is_current_week = (gw == current_gw)
                    
                    # Calculate fixture difficulty
                    is_home = fixture['team_h'] == player_team_id
                    opponent_id = fixture['team_a'] if is_home else fixture['team_h']
                    
                    opponent = teams_dict.get(opponent_id, {})
                    opponent_name = opponent.get('name', 'Unknown')
                    
                    # Get fixture difficulty (1-5 scale, lower is easier)
                    difficulty = self._calculate_fixture_difficulty_simple(
                        is_home,
                        opponent,
                        teams_dict.get(player_team_id, {})
                    )
                    
                    # Calculate TC score: player quality + fixture advantage
                    # Base score from predicted points (indicator of player quality)
                    player_quality = player['predicted_points']
                    
                    # CALIBRATED: Fixture bonus more realistic, position-specific
                    # Research shows: Easy fixtures add 0.5-1.5 pts typically, NOT 3-5 pts!
                    player_position = player.get('position_name', 'MID')
                    
                    if player_position in ['DEF', 'GK']:
                        # Defenders/GKs benefit most from easy fixtures (clean sheets)
                        fixture_bonus = max(0, (3.5 - difficulty) * 0.8)  # Max +2.0 pts
                    elif player_position == 'MID':
                        # Midfielders moderate benefit
                        fixture_bonus = max(0, (3.5 - difficulty) * 0.5)  # Max +1.25 pts
                    else:  # FWD
                        # Forwards least affected (elite forwards score regardless)
                        fixture_bonus = max(0, (3.5 - difficulty) * 0.3)  # Max +0.75 pts
                    
                    tc_score = player_quality + fixture_bonus
                    
                    # Determine fixture rating (human readable)
                    # Adjusted thresholds based on typical difficulty distribution
                    if difficulty <= 2.0:
                        fixture_rating = "Very Easy"
                    elif difficulty <= 2.8:
                        fixture_rating = "Easy"
                    elif difficulty <= 3.5:
                        fixture_rating = "Medium"
                    elif difficulty <= 4.2:
                        fixture_rating = "Hard"
                    else:
                        fixture_rating = "Very Hard"
                    
                    analysis.append({
                        'player': player['web_name'],
                        'gameweek': gw if gw != current_gw else 'current',
                        'opponent': opponent_name,
                        'opponent_position': opponent.get('position', 'N/A'),
                        'is_home': is_home,
                        'venue': 'Home' if is_home else 'Away',
                        'difficulty': difficulty,
                        'fixture_rating': fixture_rating,
                        'predicted_points': player['predicted_points'],
                        'fixture_bonus': fixture_bonus,
                        'gameweek_score': tc_score
                    })
            
            if not analysis:
                return None
            
            # Sort by TC score (best opportunity first)
            analysis.sort(key=lambda x: x['gameweek_score'], reverse=True)
            
            best_week = analysis[0]
            
            # Find current week score (or next unplayed gameweek if current has no fixtures)
            current_week_options = [a for a in analysis if a['gameweek'] == 'current']
            
            if not current_week_options:
                # No fixtures for current GW (already finished), use NEXT gameweek as baseline
                next_gw = min([a['gameweek'] for a in analysis if a['gameweek'] != 'current'])
                current_week_options = [a for a in analysis if a['gameweek'] == next_gw]
                # Mark next GW as 'current' for decision logic
                for a in current_week_options:
                    a['gameweek'] = 'current'
            
            current_week_score = max([a['gameweek_score'] for a in current_week_options]) if current_week_options else 0
            
            # Build balanced fixture list: top fixtures overall + ensure each player represented
            balanced_fixtures = []
            seen_players = set()
            
            # First pass: Add top fixtures (prioritizes best opportunities)
            for fixture in analysis:
                if len(balanced_fixtures) >= 15:  # Limit to reasonable number
                    break
                balanced_fixtures.append(fixture)
                seen_players.add(fixture['player'])
            
            # Second pass: Add at least one fixture for each player not yet shown
            for fixture in analysis:
                if len(balanced_fixtures) >= 20:  # Hard cap
                    break
                if fixture['player'] not in seen_players:
                    balanced_fixtures.append(fixture)
                    seen_players.add(fixture['player'])
            
            return {
                'best_week': best_week,
                'current_week_score': current_week_score,
                'all_weeks': balanced_fixtures
            }
            
        except Exception as e:
            print(f"Error analyzing TC timing: {e}")
            return None
    
    def _get_upcoming_fixtures(self, team_id: int, fixtures_data: List[Dict], current_gw: int, end_gw: int = None) -> List[Dict]:
        """
        Get upcoming fixtures for a team within planning horizon
        
        Args:
            team_id: Team ID to get fixtures for
            fixtures_data: All fixtures data
            current_gw: Current gameweek
            end_gw: End of planning window (if None, uses dynamic calculation)
        """
        # Get planning window
        if end_gw is None:
            _, end_gw = self._get_lookahead_window(current_gw)
        
        upcoming = []
        
        for fixture in fixtures_data:
            # Check if this fixture involves the team
            if fixture.get('team_h') != team_id and fixture.get('team_a') != team_id:
                continue
            
            # Check if fixture is in the lookahead window
            fixture_gw = fixture.get('event')
            if not fixture_gw or fixture_gw < current_gw or fixture_gw > end_gw:
                continue
            
            # Check if fixture hasn't been played yet
            if fixture.get('finished', False):
                continue
            
            upcoming.append(fixture)
        
        return sorted(upcoming, key=lambda x: x.get('event', 999))
    
    def _calculate_fixture_difficulty_simple(self, is_home: bool, opponent: Dict, team: Dict) -> float:
        """
        Calculate fixture difficulty based on league position and FPL strength ratings
        1 = Very Easy, 5 = Very Hard
        
        Priority:
        1. League position (most important - 60% weight)
        2. FPL strength ratings (underlying stats - 30% weight)
        3. Home/away advantage (10% weight)
        """
        # LEAGUE POSITION COMPONENT (60% weight)
        try:
            opponent_pos = int(opponent.get('position', 10))
            team_pos = int(team.get('position', 10))
            
            # Convert position to difficulty score (1-5 scale)
            # Top teams (1-6) = 4-5 difficulty
            # Mid teams (7-14) = 2.5-4 difficulty
            # Bottom teams (15-20) = 1-2.5 difficulty
            if opponent_pos <= 6:
                position_difficulty = 4.0 + (7 - opponent_pos) * 0.2  # 4.2 to 5.2
            elif opponent_pos <= 14:
                position_difficulty = 2.5 + (14 - opponent_pos) * 0.19  # 2.5 to 4.0
            else:
                position_difficulty = 1.0 + (20 - opponent_pos) * 0.25  # 1.0 to 2.5
            
            # Clamp to 1-5
            position_difficulty = max(1.0, min(5.0, position_difficulty))
        except (ValueError, TypeError):
            position_difficulty = 3.0  # Default to medium
        
        # FPL STRENGTH COMPONENT (30% weight)
        # Get opponent's strength ratings
        if is_home:
            opponent_attack = opponent.get('strength_attack_away', 1000)
            opponent_defense = opponent.get('strength_defence_away', 1000)
        else:
            opponent_attack = opponent.get('strength_attack_home', 1000)
            opponent_defense = opponent.get('strength_defence_home', 1000)
        
        try:
            opponent_attack = int(opponent_attack)
            opponent_defense = int(opponent_defense)
            
            # Normalize from 800-1400 to 1-5 scale
            # 800 = 1 (very weak), 1100 = 3 (average), 1400 = 5 (very strong)
            opponent_attack_norm = ((opponent_attack - 800) / 150) + 1
            opponent_defense_norm = ((opponent_defense - 800) / 150) + 1
            
            # Clamp to 1-5 range
            opponent_attack_norm = max(1.0, min(5.0, opponent_attack_norm))
            opponent_defense_norm = max(1.0, min(5.0, opponent_defense_norm))
            
            # Combine (60% defense, 40% attack for FPL component)
            strength_difficulty = (opponent_defense_norm * 0.6 + opponent_attack_norm * 0.4)
        except (ValueError, TypeError):
            strength_difficulty = 3.0
        
        # HOME/AWAY COMPONENT (10% weight)
        home_advantage = -0.5 if is_home else 0.5  # Home easier, away harder
        
        # COMBINE ALL COMPONENTS
        # 60% position, 30% strength, 10% home/away
        difficulty = (position_difficulty * 0.6) + (strength_difficulty * 0.3) + (home_advantage * 0.1)
        
        # Clamp final result to 1-5
        return max(1.0, min(5.0, difficulty))
    
    def _get_current_gameweek(self) -> Optional[int]:
        """Get current gameweek number"""
        try:
            season_data = self.data_collector.get_current_season_data()
            if season_data and 'events' in season_data:
                for event in season_data['events']:
                    if event.get('is_current', False):
                        return event.get('id')
            return None
        except:
            return None
    
    def _get_optimal_team(self, predictions_df: pd.DataFrame, budget: float = 100.0) -> pd.DataFrame:
        """Get optimal team within budget constraints"""
        # Simple implementation - in reality, you'd use more sophisticated optimization
        
        # Sort players by value prediction
        sorted_players = predictions_df.sort_values('value_prediction', ascending=False)
        
        optimal_team = []
        total_cost = 0
        
        # Position requirements
        position_limits = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        position_counts = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        
        for _, player in sorted_players.iterrows():
            position = player['position_name']
            
            # Check position limits
            if position_counts[position] < position_limits[position]:
                # Check budget
                if total_cost + player['cost'] <= budget:
                    optimal_team.append(player)
                    total_cost += player['cost']
                    position_counts[position] += 1
            
            # Stop if we have a full team
            if len(optimal_team) >= 15:
                break
        
        return pd.DataFrame(optimal_team)
    
    def _get_captain_options(self, predictions_df: pd.DataFrame, current_team: List[int]) -> pd.DataFrame:
        """Get best captain options from current team"""
        current_team_df = predictions_df[predictions_df['player_id'].isin(current_team)]
        
        # Filter to starting XI only (positions 1-11)
        # Note: This assumes the current_team_df has pick_position column
        # We need to get this from the manager analyzer instead
        starting_xi = current_team_df  # For now, use all players
        
        # Sort by predicted points
        captain_options = starting_xi.sort_values('predicted_points', ascending=False)
        
        return captain_options
    
    def _get_teams_playing_next_gw(self, fixtures_df: pd.DataFrame) -> List[str]:
        """Get teams playing in the next gameweek"""
        if fixtures_df is None or fixtures_df.empty:
            return []
        
        # For now, return empty list since we don't have proper fixtures data
        # This will prevent false Free Hit recommendations
        return []
    
    def use_chip(self, chip_name: str) -> bool:
        """Mark a chip as used"""
        if chip_name in self.chips and self.chips[chip_name]['available']:
            self.chips[chip_name]['used'] = True
            self.chips[chip_name]['available'] = False
            print(f"Chip '{chip_name}' marked as used")
            return True
        else:
            print(f"Chip '{chip_name}' is not available")
            return False
    
    def reset_chips_for_christmas(self) -> Dict:
        """Reset all chips for Christmas (if applicable)"""
        current_date = date.today()
        
        if current_date >= self.christmas_reset_date:
            reset_chips = []
            for chip_name in self.chips:
                if self.chips[chip_name]['used']:
                    self.chips[chip_name]['used'] = False
                    self.chips[chip_name]['available'] = True
                    self.chips[chip_name]['reset_date'] = current_date
                    reset_chips.append(chip_name)
            
            print(f"Chips reset for Christmas: {', '.join(reset_chips)}")
            return {'reset': True, 'chips_reset': reset_chips}
        else:
            print("Christmas reset not yet applicable")
            return {'reset': False, 'reason': 'Christmas reset date not reached'}
    
    def get_chip_usage_history(self, manager_id: int) -> Dict:
        """Get history of chip usage for a manager"""
        try:
            manager_data = self.data_collector.get_manager_data(manager_id)
            if not manager_data or 'history' not in manager_data:
                return {}
            
            history = manager_data['history']
            current_season = history.get('current', [])
            
            chip_history = []
            for gameweek in current_season:
                # This is a simplified implementation
                # In reality, you'd need to parse chip usage from the API
                if gameweek.get('event_transfers_cost', 0) == 0 and gameweek.get('event_transfers', 0) > 0:
                    chip_history.append({
                        'gameweek': gameweek['event'],
                        'chip': 'wildcard',
                        'points': gameweek['points']
                    })
            
            return {'chip_history': chip_history}
            
        except Exception as e:
            print(f"Error getting chip usage history: {e}")
            return {}
