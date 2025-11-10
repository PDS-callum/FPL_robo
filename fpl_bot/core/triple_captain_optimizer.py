"""
Triple Captain Timing Optimizer

Fast approach to find optimal triple captain timing by analyzing captain options
directly without full team optimization for each scenario.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class TripleCaptainOptimizer:
    """Optimizes triple captain timing using fast prediction-based approach"""
    
    def __init__(self, team_optimizer, data_collector):
        """
        Initialize triple captain optimizer
        
        Args:
            team_optimizer: TeamOptimizer instance
            data_collector: DataCollector instance
        """
        self.team_optimizer = team_optimizer
        self.data_collector = data_collector
        self.predictor = self.team_optimizer.predictor
    
    def find_optimal_triple_captain_timing(
        self,
        current_team: List[int],
        current_budget: float,
        free_transfers: int,
        current_gw: int = 8,
        end_gw: int = 19,
        verbose: bool = False,
        current_team_value: Optional[float] = None,
        max_gws_to_test: int = 6,
        risk_aversion: float = 0.5,
        min_chance_of_playing: int = 90
    ) -> Dict:
        """
        Find optimal triple captain timing using fast prediction-based approach
        
        Analyzes captain predictions for each gameweek without full team optimization.
        Much faster than brute force approach.
        
        Args:
            current_team: List of current player IDs
            current_budget: Available budget in millions
            free_transfers: Free transfers available
            current_gw: Current gameweek
            end_gw: Last gameweek to consider (default 19)
            verbose: Show detailed output
            current_team_value: Selling value of current team
            max_gws_to_test: Maximum number of gameweeks to test (default 6 for performance)
            risk_aversion: Risk aversion parameter (0=aggressive, 1=conservative, default 0.5)
            min_chance_of_playing: Minimum % chance of playing (default 90 for faster analysis)
            
        Returns:
            Dict with recommended timing, expected benefit, and analysis
        """
        # Limit the number of gameweeks to test for performance
        actual_end_gw = min(end_gw, current_gw + max_gws_to_test - 1)
        
        if verbose:
            print("\n" + "="*60)
            print("TRIPLE CAPTAIN TIMING OPTIMIZER")
            print("="*60)
            print(f"Analyzing optimal triple captain timing from GW{current_gw} to GW{actual_end_gw}")
        
        # Fast approach: Generate predictions once, analyze captain options
        scenarios = self._analyze_triple_captain_fast(
            current_team, current_gw, actual_end_gw, verbose, 
            risk_aversion, min_chance_of_playing
        )
        
        if not scenarios:
            return {'error': 'Could not generate triple captain scenarios'}
        
        # Find best scenario (for new fast approach, we compare differently)
        # Check if this is the new fast approach by looking for any scenario with normal_captain_points
        non_baseline_scenarios = [s for s in scenarios if not s.get('is_baseline')]
        is_fast_approach = non_baseline_scenarios and 'normal_captain_points' in non_baseline_scenarios[0] if non_baseline_scenarios else True
        
        if is_fast_approach:
            # Fast approach: compare TC benefits
            best = max(non_baseline_scenarios, 
                      key=lambda x: x.get('benefit', 0), default=None) if non_baseline_scenarios else None
            baseline = next((s for s in scenarios if s.get('is_baseline')), None)
        else:
            # Old approach (should not be used anymore)
            best = max(scenarios, key=lambda x: x.get('total_points', 0))
            baseline = next((s for s in scenarios if s.get('is_baseline')), scenarios[0])
        
        if verbose and best:
            self._print_summary(best, baseline, scenarios)
        
        # Calculate expected benefit based on approach
        if baseline is not None and best is not None:
            if 'normal_captain_points' in best:
                # Fast approach - benefit is the extra captain point
                expected_benefit = best.get('benefit', 0)
                total_with_tc = best.get('tc_captain_points', 0)
                total_without_tc = best.get('normal_captain_points', 0)
            else:
                # Old approach
                expected_benefit = best['total_points'] - baseline['total_points']
                total_with_tc = best['total_points']
                total_without_tc = baseline['total_points']
        else:
            expected_benefit = 0
            total_with_tc = 0
            total_without_tc = 0
        
        return {
            'recommended_gw': best['gameweek'] if best else None,
            'expected_benefit': expected_benefit,
            'total_points_with_tc': total_with_tc,
            'total_points_without_tc': total_without_tc,
            'captain_info': best.get('captain_info') if best else None,
            'all_scenarios': scenarios,
            'tc_plan': best.get('plan') if best else None
        }
    
    def _test_all_triple_captain_timings(
        self,
        current_team: List[int],
        current_budget: float,
        free_transfers: int,
        start_gw: int,
        end_gw: int,
        verbose: bool = False,
        current_team_value: Optional[float] = None,
        risk_aversion: float = 0.5,
        min_chance_of_playing: int = 90
    ) -> List[Dict]:
        """Test triple captain at each possible gameweek"""
        scenarios = []
        
        # Baseline: No triple captain
        if verbose:
            print(f"\nTesting baseline (no triple captain)...", end=' ', flush=True)
        
        base_result = self.team_optimizer.optimize_team(
            current_team=current_team,
            current_budget=current_budget,
            free_transfers=free_transfers,
            triple_captain_gw=None,
            verbose=False,
            current_team_value=current_team_value,
            horizon_gws=6,  # Plan 6 weeks ahead for faster testing
            risk_aversion=risk_aversion,
            min_chance_of_playing=min_chance_of_playing
        )
        
        baseline_points = self._calculate_total_points(base_result)
        if verbose:
            print(f"{baseline_points:.0f} pts")
        
        scenarios.append({
            'gameweek': None,
            'total_points': baseline_points,
            'is_baseline': True,
            'plan': base_result
        })
        
        # Test triple captain at each GW
        num_gws = end_gw - start_gw + 1
        for idx, tc_gw in enumerate(range(start_gw, end_gw + 1), 1):
            if verbose:
                print(f"Testing triple captain GW{tc_gw} ({idx}/{num_gws})...", end=' ', flush=True)
            
            try:
                result = self.team_optimizer.optimize_team(
                    current_team=current_team,
                    current_budget=current_budget,
                    free_transfers=free_transfers,
                    triple_captain_gw=tc_gw,
                    verbose=False,
                    current_team_value=current_team_value,
                    horizon_gws=6,  # Plan 6 weeks ahead for faster testing
                    risk_aversion=risk_aversion,
                    min_chance_of_playing=min_chance_of_playing
                )
                
                total_pts = self._calculate_total_points(result)
                benefit = total_pts - baseline_points
                
                # Extract captain info for this GW
                captain_info = self._extract_captain_info(result, tc_gw)
                
                if verbose:
                    print(f"{total_pts:.0f} pts (+{benefit:.0f})")
                
                scenarios.append({
                    'gameweek': tc_gw,
                    'total_points': total_pts,
                    'benefit': benefit,
                    'captain_info': captain_info,
                    'plan': result
                })
            except Exception as e:
                if verbose:
                    print(f"FAILED: {e}")
        
        return scenarios
    
    def _analyze_triple_captain_fast(
        self,
        current_team: List[int],
        current_gw: int,
        end_gw: int,
        verbose: bool,
        risk_aversion: float,
        min_chance_of_playing: int
    ) -> List[Dict]:
        """Fast triple captain analysis without full team optimization"""
        scenarios = []
        gameweeks = list(range(current_gw, end_gw + 1))
        
        # Get player data
        players_df = self.data_collector.get_player_data(include_set_pieces=True)
        if players_df is None:
            return []
        
        # Filter by playing chance
        players_df['is_available'] = (
            (players_df['chance_of_playing_next_round'].isna()) |
            (players_df['chance_of_playing_next_round'] >= min_chance_of_playing)
        )
        players_df = players_df[players_df['is_available']].copy()
        
        # Get fixtures and team strengths
        fixtures_df = self.data_collector.get_fixtures()
        team_strengths_df = self.data_collector.get_team_strengths()
        
        # Generate predictions once for all gameweeks
        if verbose:
            print("\nGenerating predictions...", end=' ', flush=True)
        
        all_predictions = []
        for gw in gameweeks:
            gw_predictions = self.predictor.predict_gameweek_points(
                players_df, fixtures_df, team_strengths_df, gw
            )
            all_predictions.append(gw_predictions)
        
        predictions_df = pd.DataFrame(pd.concat(all_predictions, ignore_index=True))
        
        # Apply risk aversion
        if risk_aversion > 0:
            predictions_df['predicted_points'] = (
                predictions_df['predicted_points'] * 
                (predictions_df['confidence'] ** risk_aversion)
            )
        
        if verbose:
            print("Done")
        
        # Analyze each gameweek
        baseline_captain_points = []
        tc_captain_points = []
        
        for gw in gameweeks:
            # Get predictions for this gameweek
            gw_preds = predictions_df[predictions_df['gameweek'] == gw].copy()
            gw_preds = gw_preds.sort_values('predicted_points', ascending=False)
            
            # Find best captain (top predicted points)
            best_captain = gw_preds.iloc[0] if len(gw_preds) > 0 else None
            
            if best_captain is not None:
                # Get player details
                player_id = best_captain['player_id']
                player_row = players_df[players_df['id'] == player_id]
                player_name = player_row['web_name'].iloc[0] if len(player_row) > 0 else f"Player {player_id}"
                player_pos = player_row['position_name'].iloc[0] if len(player_row) > 0 else 'MID'
                
                normal_points = best_captain['predicted_points'] * 2  # Normal captain
                tc_points = best_captain['predicted_points'] * 3  # Triple captain
                benefit = best_captain['predicted_points']  # Extra point from TC
                
                baseline_captain_points.append(normal_points)
                tc_captain_points.append(tc_points)
                
                scenarios.append({
                    'gameweek': gw,
                    'normal_captain_points': normal_points,
                    'tc_captain_points': tc_points,
                    'benefit': benefit,
                    'captain_info': {
                        'player_id': player_id,
                        'player_name': player_name,
                        'position': player_pos,
                        'predicted_points': best_captain['predicted_points'],
                        'captain_value': tc_points
                    }
                })
        
        # Create baseline scenario (no triple captain)
        if scenarios:
            scenarios.insert(0, {
                'gameweek': None,
                'is_baseline': True,
                'total_captain_points': sum(baseline_captain_points)
            })
        
        return scenarios
    
    def _calculate_total_points(self, optimization_result: Dict) -> float:
        """Sum total expected points across all gameweeks"""
        if not optimization_result or 'weekly_plans' not in optimization_result:
            return 0.0
        
        total = 0.0
        for gw_plan in optimization_result['weekly_plans']:
            # Use tracked expected points (includes captain bonus and transfer costs)
            total += gw_plan.get('expected_points', 0)
        
        return total
    
    def _extract_captain_info(self, optimization_result: Dict, tc_gw: int) -> Optional[Dict]:
        """Extract captain information for the triple captain gameweek"""
        if not optimization_result or 'weekly_plans' not in optimization_result:
            return None
        
        for gw_plan in optimization_result['weekly_plans']:
            if gw_plan['gameweek'] == tc_gw:
                captain = gw_plan.get('captain', {})
                return {
                    'player_id': captain.get('player_id'),
                    'player_name': captain.get('player_name'),
                    'predicted_points': captain.get('predicted_points', 0),
                    'position': captain.get('position'),
                    'team': captain.get('team'),
                    'captain_value': captain.get('predicted_points', 0) * 3  # Triple points
                }
        
        return None
    
    def _print_summary(
        self,
        best: Dict,
        baseline: Dict,
        all_scenarios: List[Dict]
    ):
        """Print formatted summary of triple captain analysis"""
        print("\n" + "="*60)
        print("TRIPLE CAPTAIN TIMING RECOMMENDATION")
        print("="*60)
        
        if 'normal_captain_points' in best:
            # Fast approach
            print(f"\n[RECOMMENDED] Use Triple Captain in GW{best['gameweek']}")
            print(f"   Expected benefit: +{best.get('benefit', 0):.1f} points")
            print(f"   TC Captain points: {best.get('tc_captain_points', 0):.1f}")
            print(f"   Normal Captain points: {best.get('normal_captain_points', 0):.1f}")
        else:
            # Old approach
            print(f"\n[RECOMMENDED] Use Triple Captain in GW{best['gameweek']}")
            print(f"   Expected benefit: +{best.get('benefit', 0):.1f} points")
            print(f"   Total points (with TC): {best.get('total_points', 0):.0f}")
            print(f"   Total points (without TC): {baseline.get('total_points', 0):.0f}")
        
        if best and best.get('captain_info'):
            cap_info = best['captain_info']
            print(f"\n   Captain: {cap_info.get('player_name', 'Unknown')} ({cap_info.get('position', 'MID')})")
            print(f"   Predicted points: {cap_info.get('predicted_points', 0):.1f}")
            print(f"   Triple captain value: {cap_info.get('captain_value', 0):.1f} pts")
        
        # Show alternative options within 2 points of best (for fast approach)
        if best and 'benefit' in best:
            alternatives = [s for s in all_scenarios if not s.get('is_baseline') and 
                           s['gameweek'] != best['gameweek'] and
                           best['benefit'] - s['benefit'] <= 2]
            
            if alternatives:
                print(f"\nAlternative Options (within 2 points benefit):")
                for alt in sorted(alternatives, key=lambda x: x.get('benefit', 0), reverse=True)[:3]:
                    diff = best.get('benefit', 0) - alt.get('benefit', 0)
                    print(f"   - GW{alt['gameweek']}: +{alt.get('benefit', 0):.1f} benefit (-{diff:.1f} from best)")
        
        print("="*60)

