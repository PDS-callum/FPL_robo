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
        """Get current chip status for a manager
        
        Checks the FPL API for chip usage history to determine availability.
        The API returns chips as a list in manager_data['chips'].
        """
        try:
            # Get manager data to check chip usage
            manager_data = self.data_collector.get_manager_data(manager_id)
            if not manager_data:
                # If we can't get manager data, assume all chips are available (default)
                return self.chips
            
            # Determine chip usage from explicit chip history if available
            # The FPL API returns chips as a top-level 'chips' list in manager_data
            # Each chip has: {'name': 'wildcard', 'event': <gameweek_number>, 'time': '<timestamp>'}
            chips_list = manager_data.get('chips', [])
            
            # If no chips list found at top level, check if it's nested elsewhere
            if not chips_list and 'history' in manager_data:
                # Some API responses might have chips in history
                chips_list = manager_data.get('history', {}).get('chips', [])
            
            # Debug: Log what chips we found
            if chips_list:
                print(f"[Chip Status] Found {len(chips_list)} chip(s) used:")
                for chip in chips_list:
                    print(f"  - {chip.get('name', 'unknown')} in GW{chip.get('event', '?')}")
            else:
                print(f"[Chip Status] No chips found in manager_data. Available keys: {list(manager_data.keys())[:10]}")
            
            current_gw = self._get_current_gameweek() or 1
            used_wc_pre = False
            used_wc_post = False
            used_tc = False
            used_bb = False
            used_fh = False

            for chip in chips_list:
                name = chip.get('name', '').lower()
                gw = int(chip.get('event', 0) or 0)
                if name == 'wildcard':
                    if gw < self.christmas_gameweek:
                        used_wc_pre = True
                    else:
                        used_wc_post = True
                elif name == '3xc':
                    used_tc = True
                elif name == 'bboost':
                    used_bb = True
                elif name == 'freehit':
                    used_fh = True

            # Reset chips for post-Christmas period
            # Availability logic:
            # - Before GW19: wildcard available if not used pre-Christmas
            # - GW19 or later: wildcard available if not used post-Christmas
            # Other chips: available if not used (no seasonal reset modeled here)
            is_pre_christmas = current_gw < self.christmas_gameweek

            # Wildcard
            if is_pre_christmas:
                self.chips['wildcard']['used'] = used_wc_pre
                self.chips['wildcard']['available'] = not used_wc_pre
            else:
                self.chips['wildcard']['used'] = used_wc_post
                self.chips['wildcard']['available'] = not used_wc_post

            # Triple Captain
            self.chips['triple_captain']['used'] = used_tc
            self.chips['triple_captain']['available'] = not used_tc

            # Bench Boost
            self.chips['bench_boost']['used'] = used_bb
            self.chips['bench_boost']['available'] = not used_bb

            # Free Hit
            self.chips['free_hit']['used'] = used_fh
            self.chips['free_hit']['available'] = not used_fh
            
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
                           fixtures_df: pd.DataFrame = None,
                           weeks_ahead: int = 9) -> Dict:
        """Recommend optimal chip usage for the current gameweek and future weeks
        
        Args:
            predictions_df: DataFrame with player predictions
            current_team: List of player IDs in current team
            fixtures_df: DataFrame with fixtures (optional)
            weeks_ahead: Number of future gameweeks to analyze (default 9)
        
        Returns:
            Dict with chip recommendations including weekly breakdown
        """
        chip_recommendations = {}
        
        # Get current gameweek
        current_gw = self._get_current_gameweek() or 1
        
        # First, get weekly recommendations for all chips
        all_chip_weekly_recs = {}
        for chip_name in self.chips:
            if self.chips[chip_name]['available']:
                weekly_recommendations = self._get_weekly_chip_recommendations(
                    chip_name, predictions_df, current_team, fixtures_df, current_gw, weeks_ahead
                )
                all_chip_weekly_recs[chip_name] = weekly_recommendations
        
        # Coordinate chip recommendations to avoid overlaps
        # Assign each chip to a unique gameweek
        assigned_weeks = {}  # {gameweek: chip_name}
        chip_priorities = {
            'wildcard': 1,      # Highest priority (use early if needed)
            'free_hit': 2,      # Second priority (for blank/double GWs)
            'triple_captain': 3, # Third priority
            'bench_boost': 4     # Lowest priority
        }
        
        # Sort chips by priority
        available_chips = sorted(
            [chip for chip in all_chip_weekly_recs.keys()],
            key=lambda x: chip_priorities.get(x, 99)
        )
        
        # Assign each chip to its best available week
        for chip_name in available_chips:
            weekly_recs = all_chip_weekly_recs[chip_name]
            
            # Find the best week for this chip that hasn't been assigned yet
            # Sort by score descending
            sorted_weeks = sorted(
                weekly_recs,
                key=lambda x: x.get('score', 0),
                reverse=True
            )
            
            best_week = None
            for week_rec in sorted_weeks:
                gw = week_rec['gameweek']
                if gw not in assigned_weeks:
                    best_week = week_rec
                    assigned_weeks[gw] = chip_name
                    break
            
            # If no unassigned week found, use the best week anyway (overlap)
            if not best_week and sorted_weeks:
                best_week = sorted_weeks[0]
                assigned_weeks[best_week['gameweek']] = chip_name
            
            # Skip if no best week found (shouldn't happen, but safety check)
            if not best_week:
                continue
            
            # Update weekly recommendations: mark only assigned week as recommended
            for week_rec in weekly_recs:
                if week_rec['gameweek'] == best_week['gameweek']:
                    week_rec['recommended'] = True
                    # Update reason to indicate this is the assigned week
                    # IMPORTANT: Preserve suggested_team and other fields - don't overwrite the entire dict
                    if chip_name == 'triple_captain':
                        week_rec['reason'] = f"⭐ Best TC week: {week_rec.get('recommended_captain', 'Unknown')} vs {week_rec.get('opponent', 'Unknown')} ({week_rec.get('fixture_rating', 'Unknown')}, {week_rec.get('venue', '')})"
                    elif chip_name == 'bench_boost':
                        week_rec['reason'] = f"⭐ Best BB week: Bench predicted {week_rec.get('bench_points', 0):.1f} points"
                    elif chip_name == 'wildcard':
                        suggested_team_info = ""
                        if week_rec.get('suggested_team') and len(week_rec.get('suggested_team', [])) > 0:
                            suggested_team_info = f" - Suggested team: {week_rec.get('suggested_team_points', 0):.1f} pts"
                            print(f"[Wildcard Coordination] GW{week_rec['gameweek']} has suggested team with {len(week_rec.get('suggested_team', []))} players")
                        week_rec['reason'] = f"⭐ Best WC week: Use for team overhaul{suggested_team_info}"
                    elif chip_name == 'free_hit':
                        suggested_team_info = ""
                        if week_rec.get('suggested_team') and len(week_rec.get('suggested_team', [])) > 0:
                            suggested_team_info = f" - Suggested team: {week_rec.get('suggested_team_points', 0):.1f} pts"
                            print(f"[Free Hit Coordination] GW{week_rec['gameweek']} has suggested team with {len(week_rec.get('suggested_team', []))} players")
                        week_rec['reason'] = f"⭐ Best FH week: Use for blank/double gameweek{suggested_team_info}"
                else:
                    week_rec['recommended'] = False
                    # Update reason to show why this week is not recommended
                    if week_rec.get('score', 0) > 0:
                        week_rec['reason'] = f"Not assigned. Assigned to GW{best_week['gameweek']} (score: {best_week.get('score', 0):.1f} vs this week: {week_rec.get('score', 0):.1f})"
            
            # Sort weekly recommendations by gameweek for display
            all_chip_weekly_recs[chip_name].sort(key=lambda x: x.get('gameweek', 999))
        
        # Build final recommendations with coordinated weekly assignments
        for chip_name in self.chips:
            if self.chips[chip_name]['available']:
                # Get single recommendation (current behavior)
                recommendation = self._get_chip_recommendation(
                    chip_name, predictions_df, current_team, fixtures_df
                )
                
                # Use the coordinated weekly recommendations
                weekly_recommendations = all_chip_weekly_recs.get(chip_name, [])
                
                # Update main recommendation based on weekly recommendations
                # Find the recommended week (if any)
                recommended_week_rec = next(
                    (rec for rec in weekly_recommendations if rec.get('recommended')),
                    None
                )
                
                if recommended_week_rec:
                    # Update main recommendation to reflect the weekly assignment
                    recommendation['recommended'] = True
                    recommendation['reason'] = recommended_week_rec.get('reason', recommendation.get('reason', ''))
                    recommendation['expected_benefit'] = recommended_week_rec.get('expected_benefit', recommendation.get('expected_benefit', 0))
                    recommendation['timing'] = f"GW{recommended_week_rec.get('gameweek', '?')}"
                    
                    # Include chip-specific details from weekly recommendation
                    if chip_name in ['free_hit', 'wildcard'] and recommended_week_rec.get('suggested_team'):
                        recommendation['suggested_team'] = recommended_week_rec.get('suggested_team')
                        recommendation['suggested_team_points'] = recommended_week_rec.get('suggested_team_points', 0)
                    elif chip_name == 'triple_captain':
                        recommendation['recommended_captain'] = recommended_week_rec.get('recommended_captain')
                        recommendation['opponent'] = recommended_week_rec.get('opponent')
                        recommendation['fixture_rating'] = recommended_week_rec.get('fixture_rating')
                        recommendation['venue'] = recommended_week_rec.get('venue')
                        recommendation['predicted_points'] = recommended_week_rec.get('predicted_points')
                    elif chip_name == 'bench_boost':
                        recommendation['bench_points'] = recommended_week_rec.get('bench_points')
                else:
                    # If no week is recommended, ensure main recommendation reflects that
                    recommendation['recommended'] = False
                    if chip_name == 'free_hit':
                        recommendation['reason'] = 'No blank/double gameweek identified in the next 9 gameweeks'
                
                chip_recommendations[chip_name] = {
                    **recommendation,
                    'weekly_recommendations': weekly_recommendations
                }
        
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
        # Handle both 'player_id' and 'id' column names
        player_id_col = 'player_id' if 'player_id' in predictions_df.columns else 'id'
        current_team_df = predictions_df[predictions_df[player_id_col].isin(current_team)]
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
        # Handle both 'player_id' and 'id' column names
        player_id_col = 'player_id' if 'player_id' in predictions_df.columns else 'id'
        current_team_df = predictions_df[predictions_df[player_id_col].isin(current_team)]
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
    
    def _get_weekly_chip_recommendations(self, 
                                        chip_name: str,
                                        predictions_df: pd.DataFrame,
                                        current_team: List[int],
                                        fixtures_df: pd.DataFrame,
                                        current_gw: int,
                                        weeks_ahead: int) -> List[Dict]:
        """Get chip recommendations for each of the next N gameweeks
        
        Args:
            chip_name: Name of the chip
            predictions_df: DataFrame with player predictions
            current_team: List of player IDs in current team
            fixtures_df: DataFrame with fixtures
            current_gw: Current gameweek
            weeks_ahead: Number of weeks to analyze
        
        Returns:
            List of recommendations, one per gameweek
        """
        weekly_recs = []
        
        # Get fixtures and teams data
        fixtures_data = self.data_collector.get_fixtures_data()
        teams_data = self.data_collector.get_current_season_data()
        
        if not fixtures_data or not teams_data:
            # If no fixture data, return basic recommendations
            for gw in range(current_gw, min(current_gw + weeks_ahead, 39)):
                weekly_recs.append({
                    'gameweek': gw,
                    'recommended': False,
                    'reason': 'Insufficient fixture data for analysis',
                    'expected_benefit': 0
                })
            return weekly_recs
        
        teams_dict = {team['id']: team for team in teams_data.get('teams', [])}
        
        # Pre-build fixture lookup by gameweek and team for O(1) access
        # This avoids repeated linear searches through fixtures_data
        fixture_lookup = {}  # {(gw, team_id): fixture}
        for fixture in fixtures_data:
            gw = fixture.get('event')
            if gw and current_gw <= gw < current_gw + weeks_ahead:
                team_h = fixture.get('team_h')
                team_a = fixture.get('team_a')
                if team_h:
                    fixture_lookup[(gw, team_h)] = fixture
                if team_a:
                    fixture_lookup[(gw, team_a)] = fixture
        
        # First pass: Analyze all gameweeks and collect scores
        all_week_scores = []
        for gw in range(current_gw, min(current_gw + weeks_ahead, 39)):
            if chip_name == 'triple_captain':
                rec = self._get_tc_recommendation_for_gw(
                    gw, predictions_df, current_team, fixture_lookup, teams_dict
                )
            elif chip_name == 'bench_boost':
                rec = self._get_bb_recommendation_for_gw(
                    gw, predictions_df, current_team, fixture_lookup, teams_dict
                )
            elif chip_name == 'wildcard':
                rec = self._get_wc_recommendation_for_gw(
                    gw, predictions_df, current_team
                )
            elif chip_name == 'free_hit':
                rec = self._get_fh_recommendation_for_gw(
                    gw, predictions_df, current_team, fixture_lookup, teams_dict
                )
            else:
                rec = {
                    'gameweek': gw,
                    'recommended': False,
                    'reason': 'Chip type not supported for weekly analysis',
                    'expected_benefit': 0,
                    'score': 0
                }
            
            rec['gameweek'] = gw
            # Calculate a score for ranking (higher is better)
            # Always calculate score regardless of initial recommendation status
            if chip_name == 'triple_captain':
                # Use tc_score if available, otherwise use predicted_points
                rec['score'] = rec.get('tc_score', rec.get('predicted_points', 0))
            elif chip_name == 'bench_boost':
                rec['score'] = rec.get('bench_points', 0)
            elif chip_name == 'wildcard':
                # Use score if available, otherwise expected_benefit
                rec['score'] = rec.get('score', rec.get('expected_benefit', 0))
            elif chip_name == 'free_hit':
                # Use score if available, otherwise expected_benefit
                rec['score'] = rec.get('score', rec.get('expected_benefit', 0))
            else:
                rec['score'] = rec.get('score', 0)
            
            all_week_scores.append(rec)
        
        # Find the best week for this chip (highest score)
        if all_week_scores:
            # Sort by score descending to find best week
            all_week_scores.sort(key=lambda x: x.get('score', 0), reverse=True)
            best_week_gw = all_week_scores[0]['gameweek']
            
            # Mark only the best week as recommended
            best_rec = all_week_scores[0]
            for rec in all_week_scores:
                if rec['gameweek'] == best_week_gw:
                    rec['recommended'] = True
                    # Update reason to indicate this is the best week
                    if chip_name == 'triple_captain':
                        rec['reason'] = f"⭐ Best TC week: {rec.get('recommended_captain', 'Unknown')} vs {rec.get('opponent', 'Unknown')} ({rec.get('fixture_rating', 'Unknown')}, {rec.get('venue', '')})"
                    elif chip_name == 'bench_boost':
                        rec['reason'] = f"⭐ Best BB week: Bench predicted {rec.get('bench_points', 0):.1f} points"
                    elif chip_name == 'wildcard':
                        rec['reason'] = f"⭐ Best WC week: Use for team overhaul"
                    elif chip_name == 'free_hit':
                        rec['reason'] = f"⭐ Best FH week: Use for blank/double gameweek"
                else:
                    rec['recommended'] = False
                    # Update reason to show why this week is not recommended
                    if rec.get('score', 0) > 0:
                        rec['reason'] = f"Not best week. Best is GW{best_week_gw} (score: {best_rec.get('score', 0):.1f} vs this week: {rec.get('score', 0):.1f})"
            
            # Sort by gameweek ascending for display
            all_week_scores.sort(key=lambda x: x.get('gameweek', 999))
        
        return all_week_scores
    
    def _get_tc_recommendation_for_gw(self, gw: int, predictions_df: pd.DataFrame, 
                                     current_team: List[int], fixture_lookup: Dict,
                                     teams_dict: Dict) -> Dict:
        """Get Triple Captain recommendation for a specific gameweek"""
        # Filter predictions to only this gameweek
        if 'gameweek' in predictions_df.columns:
            gw_predictions_df = predictions_df[predictions_df['gameweek'] == gw].copy()
        else:
            # If no gameweek column, use all predictions (fallback)
            gw_predictions_df = predictions_df.copy()
        
        if len(gw_predictions_df) == 0:
            return {
                'recommended': False,
                'reason': f'No predictions available for GW{gw}',
                'expected_benefit': 0,
                'tc_score': 0,
                'score': 0
            }
        
        # Get captain options for THIS gameweek
        player_id_col = 'player_id' if 'player_id' in gw_predictions_df.columns else 'id'
        current_team_df = gw_predictions_df[gw_predictions_df[player_id_col].isin(current_team)]
        captain_options = current_team_df.sort_values('predicted_points', ascending=False).head(5)
        
        if captain_options.empty:
            return {
                'recommended': False,
                'reason': 'No captain options available',
                'expected_benefit': 0,
                'tc_score': 0,
                'score': 0
            }
        
        # Find best captain option for this gameweek
        best_option = None
        best_score = 0
        
        for _, player in captain_options.iterrows():
            player_team_id = player.get('team')
            if not player_team_id:
                continue
            
            # Get fixture for this gameweek using pre-built lookup
            fixture = fixture_lookup.get((gw, player_team_id))
            if not fixture:
                continue
            
            # Calculate difficulty and score
            is_home = fixture['team_h'] == player_team_id
            opponent_id = fixture['team_a'] if is_home else fixture['team_h']
            opponent = teams_dict.get(opponent_id, {})
            
            difficulty = self._calculate_fixture_difficulty_simple(
                is_home, opponent, teams_dict.get(player_team_id, {})
            )
            
            player_quality = float(player.get('predicted_points', 0))
            player_position = player.get('position_name', 'MID')
            
            if player_position in ['DEF', 'GK']:
                fixture_bonus = max(0, (3.5 - difficulty) * 0.8)
            elif player_position == 'MID':
                fixture_bonus = max(0, (3.5 - difficulty) * 0.5)
            else:
                fixture_bonus = max(0, (3.5 - difficulty) * 0.3)
            
            tc_score = player_quality + fixture_bonus
            
            if tc_score > best_score:
                best_score = tc_score
                best_option = {
                    'player': player.get('web_name', 'Unknown'),
                    'predicted_points': player_quality,
                    'opponent': opponent.get('name', 'Unknown'),
                    'difficulty': difficulty,
                    'fixture_rating': self._get_fixture_rating(difficulty),
                    'venue': 'Home' if is_home else 'Away',
                    'tc_score': tc_score
                }
        
        if not best_option:
            return {
                'recommended': False,
                'reason': 'No fixtures found for this gameweek',
                'expected_benefit': 0,
                'tc_score': 0,
                'score': 0
            }
        
        # Determine if recommended (but we'll override this in the calling function)
        is_elite = best_option['predicted_points'] >= 7.0
        is_easy = best_option['difficulty'] <= 2.8
        is_high_score = best_option['tc_score'] >= 9.0
        
        recommended = is_elite and is_easy and is_high_score
        
        return {
            'recommended': recommended,  # Will be overridden by caller
            'reason': f"{best_option['player']} vs {best_option['opponent']} ({best_option['fixture_rating']}, {best_option['venue']})",
            'expected_benefit': best_option['predicted_points'],
            'recommended_captain': best_option['player'],
            'opponent': best_option['opponent'],
            'fixture_rating': best_option['fixture_rating'],
            'venue': best_option['venue'],
            'predicted_points': best_option['predicted_points'],
            'tc_score': best_option['tc_score']  # Always include for ranking
        }
    
    def _get_bb_recommendation_for_gw(self, gw: int, predictions_df: pd.DataFrame,
                                     current_team: List[int], fixture_lookup: Dict,
                                     teams_dict: Dict) -> Dict:
        """Get Bench Boost recommendation for a specific gameweek"""
        # Filter predictions to only this gameweek
        if 'gameweek' in predictions_df.columns:
            gw_predictions_df = predictions_df[predictions_df['gameweek'] == gw].copy()
        else:
            # If no gameweek column, use all predictions (fallback)
            gw_predictions_df = predictions_df.copy()
        
        if len(gw_predictions_df) == 0:
            return {
                'recommended': False,
                'reason': f'No predictions available for GW{gw}',
                'expected_benefit': 0,
                'bench_points': 0,
                'score': 0
            }
        
        # Get bench players for THIS gameweek
        player_id_col = 'player_id' if 'player_id' in gw_predictions_df.columns else 'id'
        current_team_df = gw_predictions_df[gw_predictions_df[player_id_col].isin(current_team)]
        bench_players = current_team_df.tail(4)
        
        if len(bench_players) < 3:
            return {
                'recommended': False,
                'reason': 'Not enough bench players',
                'expected_benefit': 0,
                'bench_points': 0,
                'score': 0
            }
        
        # Calculate bench points for this gameweek
        bench_score = 0
        for _, player in bench_players.iterrows():
            player_team_id = player.get('team')
            if not player_team_id:
                continue
            
            fixture = fixture_lookup.get((gw, player_team_id))
            if not fixture:
                continue
            
            base_prediction = float(player.get('predicted_points', 0))
            position = player.get('position_name', 'MID')
            
            # Adjust for fixture difficulty
            is_home = fixture['team_h'] == player_team_id
            opponent_id = fixture['team_a'] if is_home else fixture['team_h']
            opponent = teams_dict.get(opponent_id, {})
            
            difficulty = self._calculate_fixture_difficulty_simple(
                is_home, opponent, teams_dict.get(player_team_id, {})
            )
            
            if position in ['DEF', 'GK']:
                fixture_adj = (3.0 - difficulty) * 0.4
            elif position == 'MID':
                fixture_adj = (3.0 - difficulty) * 0.25
            else:
                fixture_adj = (3.0 - difficulty) * 0.15
            
            bench_score += max(0, base_prediction + fixture_adj)
        
        recommended = bench_score >= 8.0  # Will be overridden by caller
        
        return {
            'recommended': recommended,  # Will be overridden by caller
            'reason': f'Bench predicted {bench_score:.1f} points',
            'expected_benefit': bench_score,
            'bench_points': bench_score  # Always include for ranking
        }
    
    def _get_wc_recommendation_for_gw(self, gw: int, predictions_df: pd.DataFrame,
                                      current_team: List[int]) -> Dict:
        """Get Wildcard recommendation for a specific gameweek"""
        # Filter predictions to only this gameweek
        # CRITICAL: predictions_df must have a 'gameweek' column for this to work correctly
        # If it doesn't, all gameweeks will show the same improvement
        if 'gameweek' not in predictions_df.columns:
            # This is a problem - predictions aren't gameweek-specific
            # Return a warning instead of incorrect results
            return {
                'recommended': False,
                'reason': f'Predictions not gameweek-specific - cannot calculate GW{gw} improvement',
                'expected_benefit': 0,
                'score': 0
            }
        
        # Filter to only this gameweek
        gw_predictions_df = predictions_df[predictions_df['gameweek'] == gw].copy()
        
        if len(gw_predictions_df) == 0:
            return {
                'recommended': False,
                'reason': f'No predictions available for GW{gw}',
                'expected_benefit': 0,
                'score': 0
            }
        
        # Verify we have predictions for this gameweek
        if 'predicted_points' not in gw_predictions_df.columns:
            return {
                'recommended': False,
                'reason': f'No predicted_points column for GW{gw}',
                'expected_benefit': 0,
                'score': 0
            }
        
        # Calculate potential improvement from wildcard for THIS gameweek
        player_id_col = 'player_id' if 'player_id' in gw_predictions_df.columns else 'id'
        current_team_df = gw_predictions_df[gw_predictions_df[player_id_col].isin(current_team)]
        # current_team_df is a DataFrame, so access the column directly
        if len(current_team_df) > 0 and 'predicted_points' in current_team_df.columns:
            current_total_points = pd.to_numeric(current_team_df['predicted_points'], errors='coerce').fillna(0).sum()
        else:
            current_total_points = 0.0
        
        # Get optimal team for THIS gameweek (simplified - just top players by value)
        # Note: We're using gameweek-specific predictions, so the optimal team should vary by gameweek
        # based on fixture difficulty and player form for that specific gameweek
        optimal_team_df = self._get_optimal_team(gw_predictions_df)
        # optimal_team_df is a DataFrame, so sum the predicted_points column
        if len(optimal_team_df) > 0 and 'predicted_points' in optimal_team_df.columns:
            # Sum predicted_points from the optimal team for THIS gameweek
            optimal_total_points = pd.to_numeric(optimal_team_df['predicted_points'], errors='coerce').fillna(0).sum()
        else:
            optimal_total_points = 0.0
        
        improvement = optimal_total_points - current_total_points
        score = max(0, improvement)  # Score based on potential improvement
        
        # Generate suggested Wildcard team for this gameweek
        # Unlike Free Hit, Wildcard team persists beyond the gameweek
        suggested_team = None
        suggested_team_points = 0.0
        
        if len(optimal_team_df) > 0:
            # Calculate total predicted points for the suggested team
            if 'predicted_points' in optimal_team_df.columns:
                suggested_team_points = pd.to_numeric(
                    optimal_team_df['predicted_points'], 
                    errors='coerce'
                ).fillna(0).sum()
            
            # Format suggested team for display
            suggested_team = []
            player_id_col_team = 'player_id' if 'player_id' in optimal_team_df.columns else 'id'
            
            for _, player in optimal_team_df.iterrows():
                player_info = {
                    'player_id': int(player.get(player_id_col_team, 0)),
                    'name': player.get('web_name', player.get('name', 'Unknown')),
                    'position': player.get('position_name', 'MID'),
                    'team': player.get('team', 0),
                    'predicted_points': float(player.get('predicted_points', 0)),
                    'cost': float(player.get('cost', player.get('now_cost', 0) / 10 if 'now_cost' in player else 5.0))
                }
                suggested_team.append(player_info)
        
        result = {
            'recommended': False,  # Will be overridden by caller
            'reason': f'Wildcard available - potential improvement: {improvement:.1f} points',
            'expected_benefit': improvement,
            'score': score,
            'suggested_team': suggested_team if suggested_team else [],  # List of player dicts
            'suggested_team_points': suggested_team_points  # Total predicted points
        }
        
        # Debug: Log if suggested team was generated
        if suggested_team and len(suggested_team) > 0:
            print(f"[Wildcard GW{gw}] Generated suggested team with {len(suggested_team)} players, {suggested_team_points:.1f} predicted points")
        else:
            print(f"[Wildcard GW{gw}] No suggested team generated. Optimal team DF length: {len(optimal_team_df)}")
        
        return result
    
    def _get_fh_recommendation_for_gw(self, gw: int, predictions_df: pd.DataFrame,
                                      current_team: List[int], fixture_lookup: Dict,
                                      teams_dict: Dict) -> Dict:
        """Get Free Hit recommendation for a specific gameweek
        
        Returns a recommendation with a suggested Free Hit team if beneficial.
        """
        # Filter predictions to only this gameweek
        if 'gameweek' not in predictions_df.columns:
            return {
                'recommended': False,
                'reason': 'Predictions not gameweek-specific - cannot calculate Free Hit team',
                'expected_benefit': 0,
                'score': 0
            }
        
        gw_predictions_df = predictions_df[predictions_df['gameweek'] == gw].copy()
        
        if len(gw_predictions_df) == 0:
            return {
                'recommended': False,
                'reason': f'No predictions available for GW{gw}',
                'expected_benefit': 0,
                'score': 0
            }
        
        # Check for blank/double gameweeks using fixture lookup
        # Count how many teams are playing this gameweek
        teams_playing = set()
        team_fixture_counts = {}
        seen_fixtures = set()  # Track fixtures to avoid double-counting
        
        # Use fixture_lookup to efficiently find fixtures for this gameweek
        # Each fixture appears twice in lookup (once for home, once for away)
        # So we track seen fixtures to count correctly
        for (fixture_gw, team_id), fixture in fixture_lookup.items():
            if fixture_gw == gw:
                teams_playing.add(team_id)
                # Use fixture ID to avoid double-counting the same fixture
                fixture_id = fixture.get('id')
                if fixture_id and fixture_id not in seen_fixtures:
                    seen_fixtures.add(fixture_id)
                    # Count this fixture for both teams
                    team_h = fixture.get('team_h')
                    team_a = fixture.get('team_a')
                    if team_h:
                        team_fixture_counts[team_h] = team_fixture_counts.get(team_h, 0) + 1
                    if team_a:
                        team_fixture_counts[team_a] = team_fixture_counts.get(team_a, 0) + 1
        
        total_teams = 20
        teams_playing_count = len(teams_playing)
        
        # Blank gameweek (few teams playing) or double gameweek (some teams playing twice)
        # Score higher for blank gameweeks (more benefit from Free Hit)
        if teams_playing_count < total_teams:
            # Blank gameweek - high score
            score = (total_teams - teams_playing_count) * 2  # More teams blanked = higher score
            reason = f'Blank gameweek - only {teams_playing_count}/{total_teams} teams playing'
        else:
            # Check for double gameweek
            double_gw_teams = sum(1 for count in team_fixture_counts.values() if count > 1)
            if double_gw_teams > 0:
                score = double_gw_teams * 1.5  # Some benefit from double gameweek
                reason = f'Double gameweek - {double_gw_teams} teams playing twice'
            else:
                score = 0
                reason = 'Normal gameweek - Free Hit not particularly beneficial'
        
        # If Free Hit is beneficial (score > 0), generate a suggested team
        suggested_team = None
        suggested_team_points = 0.0
        if score > 0:
            # Generate optimal Free Hit team for this gameweek
            # Filter to only players from teams that are playing this gameweek
            player_id_col = 'player_id' if 'player_id' in gw_predictions_df.columns else 'id'
            
            # Filter predictions to only include players from teams playing this gameweek
            # Check what column name is used for team ID
            team_col = None
            for col in ['team', 'team_id', 'team_code']:
                if col in gw_predictions_df.columns:
                    team_col = col
                    break
            
            if team_col:
                playing_teams_predictions = gw_predictions_df[
                    gw_predictions_df[team_col].isin(teams_playing)
                ].copy()
            else:
                # If no team column found, use all predictions (fallback)
                playing_teams_predictions = gw_predictions_df.copy()
            
            if len(playing_teams_predictions) > 0:
                # Get optimal team from players who are playing
                optimal_fh_team_df = self._get_optimal_team(playing_teams_predictions)
                
                if len(optimal_fh_team_df) > 0:
                    # Calculate total predicted points for the suggested team
                    if 'predicted_points' in optimal_fh_team_df.columns:
                        suggested_team_points = pd.to_numeric(
                            optimal_fh_team_df['predicted_points'], 
                            errors='coerce'
                        ).fillna(0).sum()
                    
                    # Format suggested team for display
                    # Include player name, position, team, predicted points, and cost
                    suggested_team = []
                    player_id_col_team = 'player_id' if 'player_id' in optimal_fh_team_df.columns else 'id'
                    
                    for _, player in optimal_fh_team_df.iterrows():
                        player_info = {
                            'player_id': int(player.get(player_id_col_team, 0)),
                            'name': player.get('web_name', player.get('name', 'Unknown')),
                            'position': player.get('position_name', 'MID'),
                            'team': player.get('team', 0),
                            'predicted_points': float(player.get('predicted_points', 0)),
                            'cost': float(player.get('cost', player.get('now_cost', 0) / 10 if 'now_cost' in player else 5.0))
                        }
                        suggested_team.append(player_info)
        
        result = {
            'recommended': False,  # Will be overridden by caller
            'reason': reason,
            'expected_benefit': score,
            'score': score,
            'suggested_team': suggested_team if suggested_team else [],  # List of player dicts (always include, even if empty)
            'suggested_team_points': suggested_team_points  # Total predicted points
        }
        
        # Debug: Log if suggested team was generated
        if score > 0:
            if suggested_team and len(suggested_team) > 0:
                print(f"[Free Hit GW{gw}] Generated suggested team with {len(suggested_team)} players, {suggested_team_points:.1f} predicted points")
            else:
                print(f"[Free Hit GW{gw}] Score > 0 but no suggested team generated. Teams playing: {len(teams_playing)}, Predictions available: {len(gw_predictions_df)}")
        
        return result
    
    def _get_fixture_for_team_gw(self, team_id: int, gw: int, fixtures_data: List[Dict] = None, fixture_lookup: Dict = None) -> Optional[Dict]:
        """Get fixture for a specific team and gameweek
        
        Args:
            team_id: Team ID
            gw: Gameweek
            fixtures_data: List of fixtures (deprecated, use fixture_lookup instead)
            fixture_lookup: Pre-built lookup dict {(gw, team_id): fixture} for O(1) access
        """
        # Use fixture_lookup if provided (much faster)
        if fixture_lookup is not None:
            return fixture_lookup.get((gw, team_id))
        
        # Fallback to linear search (slower, for backwards compatibility)
        if fixtures_data:
            for fixture in fixtures_data:
                if fixture.get('event') == gw:
                    if fixture.get('team_h') == team_id or fixture.get('team_a') == team_id:
                        return fixture
        return None
    
    def _get_fixture_rating(self, difficulty: float) -> str:
        """Get human-readable fixture rating"""
        if difficulty <= 2.0:
            return "Very Easy"
        elif difficulty <= 2.8:
            return "Easy"
        elif difficulty <= 3.5:
            return "Medium"
        elif difficulty <= 4.2:
            return "Hard"
        else:
            return "Very Hard"
    
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
        difficulty = (position_difficulty * 0.55) + (strength_difficulty * 0.35) + (home_advantage * 0.10)
        
        # Clamp final result to 1-5
        return max(1.0, min(5.0, difficulty))
    
    def _get_current_gameweek(self) -> Optional[int]:
        """Get current gameweek number"""
        try:
            season_data = self.data_collector.get_current_season_data()
            if season_data and 'events' in season_data:
                events = season_data['events']
                current_event = next((e for e in events if e.get('is_current', False)), None)
                if current_event:
                    if current_event.get('finished', False):
                        # Prefer next unplayed GW
                        for e in events:
                            if not e.get('finished', False) and e.get('is_next', False):
                                return e.get('id')
                        for e in events:
                            if not e.get('finished', False):
                                return e.get('id')
                    return current_event.get('id')
                # Fallback: first not finished event
                for e in events:
                    if not e.get('finished', False):
                        return e.get('id')
            return None
        except:
            return None
    
    def _get_optimal_team(self, predictions_df: pd.DataFrame, budget: float = 100.0) -> pd.DataFrame:
        """Get optimal team within budget constraints"""
        # Simple implementation - in reality, you'd use more sophisticated optimization
        
        # Always work with a copy to avoid modifying the original
        predictions_df = predictions_df.copy()
        
        # Calculate value_prediction if it doesn't exist
        if 'value_prediction' not in predictions_df.columns:
            # Calculate value as predicted_points per million
            if 'predicted_points' in predictions_df.columns and 'cost' in predictions_df.columns:
                # Ensure numeric types
                predicted_pts = pd.to_numeric(predictions_df['predicted_points'], errors='coerce').fillna(0)
                cost = pd.to_numeric(predictions_df['cost'], errors='coerce').replace(0, 1)
                predictions_df['value_prediction'] = predicted_pts / cost
            elif 'points_per_million' in predictions_df.columns:
                predictions_df['value_prediction'] = pd.to_numeric(predictions_df['points_per_million'], errors='coerce').fillna(0)
            else:
                # Fallback: use total_points / cost
                if 'total_points' in predictions_df.columns and 'cost' in predictions_df.columns:
                    total_pts = pd.to_numeric(predictions_df['total_points'], errors='coerce').fillna(0)
                    cost = pd.to_numeric(predictions_df['cost'], errors='coerce').replace(0, 1)
                    predictions_df['value_prediction'] = total_pts / cost
                else:
                    # Last resort: use predicted_points or 0
                    predictions_df['value_prediction'] = pd.to_numeric(predictions_df.get('predicted_points', 0), errors='coerce').fillna(0)
        
        # Sort players by value prediction
        sorted_players = predictions_df.sort_values('value_prediction', ascending=False)
        
        # Ensure cost column exists
        if 'cost' not in sorted_players.columns:
            sorted_players = sorted_players.copy()
            if 'now_cost' in sorted_players.columns:
                sorted_players['cost'] = pd.to_numeric(sorted_players['now_cost'], errors='coerce').fillna(50) / 10
            else:
                sorted_players['cost'] = 5.0  # Default cost if unknown
        else:
            # Ensure cost is numeric
            sorted_players = sorted_players.copy()
            sorted_players['cost'] = pd.to_numeric(sorted_players['cost'], errors='coerce').fillna(5.0)
        
        optimal_team = []
        total_cost = 0
        
        # Position requirements
        position_limits = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        position_counts = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        
        for _, player in sorted_players.iterrows():
            position = player.get('position_name', 'MID')  # Default to MID if missing
            
            # Check position limits
            if position_counts[position] < position_limits[position]:
                # Check budget - ensure cost is numeric
                player_cost = float(player.get('cost', 5.0))
                if total_cost + player_cost <= budget:
                    optimal_team.append(player)
                    total_cost += player_cost
                    position_counts[position] += 1
            
            # Stop if we have a full team
            if len(optimal_team) >= 15:
                break
        
        return pd.DataFrame(optimal_team)
    
    def _get_captain_options(self, predictions_df: pd.DataFrame, current_team: List[int]) -> pd.DataFrame:
        """Get best captain options from current team"""
        # Handle both 'player_id' and 'id' column names
        player_id_col = 'player_id' if 'player_id' in predictions_df.columns else 'id'
        current_team_df = predictions_df[predictions_df[player_id_col].isin(current_team)]
        
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
