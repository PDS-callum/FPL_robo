"""
Wildcard Timing Optimizer

Uses hybrid approach:
1. Brute force testing of wildcard at each GW (mathematical optimum)
2. Fixture run analysis (strategic context)
3. Synthesized recommendation with reasoning
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class WildcardOptimizer:
    """Optimizes wildcard timing using brute force + fixture intelligence"""
    
    def __init__(self, team_optimizer, data_collector):
        """
        Initialize wildcard optimizer
        
        Args:
            team_optimizer: TeamOptimizer instance
            data_collector: DataCollector instance
        """
        self.team_optimizer = team_optimizer
        self.data_collector = data_collector
        
        # Fixture difficulty thresholds
        self.EASY_FIXTURE = 3.0
        self.MIN_RUN_LENGTH = 3
    
    def find_optimal_wildcard_timing(
        self,
        current_team: List[int],
        current_budget: float,
        free_transfers: int,
        current_gw: int = 8,
        end_gw: int = 19,
        verbose: bool = False
    ) -> Dict:
        """
        Find optimal wildcard timing using hybrid approach
        
        Phase 1: Brute force test wildcard at each GW
        Phase 2: Analyze fixture runs for strategic context
        Phase 3: Synthesize recommendation with reasoning
        
        Args:
            current_team: List of current player IDs
            current_budget: Available budget in millions
            free_transfers: Free transfers available
            current_gw: Current gameweek
            end_gw: Last gameweek to consider (default 19)
            verbose: Show detailed output
            
        Returns:
            Dict with recommended timing, expected benefit, and strategic analysis
        """
        if verbose:
            print("\n" + "="*60)
            print("WILDCARD TIMING OPTIMIZER")
            print("="*60)
            print(f"Analyzing optimal wildcard timing from GW{current_gw} to GW{end_gw}")
        
        # Phase 1: Brute force testing
        scenarios = self._test_all_wildcard_timings(
            current_team, current_budget, free_transfers, current_gw, end_gw, verbose
        )
        
        if not scenarios:
            return {'error': 'Could not generate wildcard scenarios'}
        
        # Phase 2: Fixture analysis
        if verbose:
            print("\nAnalyzing fixture runs...")
        fixture_runs = self._identify_fixture_runs(current_gw, end_gw)
        
        # Find best scenario
        best = max(scenarios, key=lambda x: x['total_points'])
        baseline = next((s for s in scenarios if s.get('is_baseline')), scenarios[0])
        
        # Phase 3: Generate strategic explanation
        explanation = self._explain_timing(best['gameweek'], fixture_runs, current_gw)
        alternatives = self._find_alternatives(scenarios, best['total_points'])
        
        if verbose:
            self._print_summary(best, baseline, fixture_runs, alternatives)
        
        return {
            'recommended_gw': best['gameweek'],
            'expected_benefit': best['total_points'] - baseline['total_points'],
            'total_points_with_wildcard': best['total_points'],
            'total_points_without_wildcard': baseline['total_points'],
            'strategic_analysis': explanation,
            'fixture_runs': fixture_runs,
            'alternative_options': alternatives,
            'all_scenarios': scenarios,
            'wildcard_plan': best.get('plan')
        }
    
    def _test_all_wildcard_timings(
        self,
        current_team: List[int],
        current_budget: float,
        free_transfers: int,
        start_gw: int,
        end_gw: int,
        verbose: bool = False
    ) -> List[Dict]:
        """Test wildcard at each possible gameweek"""
        scenarios = []
        
        # Baseline: No wildcard
        if verbose:
            print(f"\nTesting baseline (no wildcard)...", end=' ')
        
        base_result = self.team_optimizer.optimize_team(
            current_team=current_team,
            current_budget=current_budget,
            free_transfers=free_transfers,
            wildcard_gw=None,
            verbose=False
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
        
        # Test wildcard at each GW
        for wc_gw in range(start_gw, end_gw + 1):
            if verbose:
                print(f"Testing GW{wc_gw}...", end=' ')
            
            try:
                result = self.team_optimizer.optimize_team(
                    current_team=current_team,
                    current_budget=current_budget,
                    free_transfers=free_transfers,
                    wildcard_gw=wc_gw,
                    verbose=False
                )
                
                total_pts = self._calculate_total_points(result)
                benefit = total_pts - baseline_points
                
                if verbose:
                    print(f"{total_pts:.0f} pts (+{benefit:.0f})")
                
                scenarios.append({
                    'gameweek': wc_gw,
                    'total_points': total_pts,
                    'benefit': benefit,
                    'plan': result
                })
            except Exception as e:
                if verbose:
                    print(f"FAILED: {e}")
        
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
    
    def _identify_fixture_runs(self, start_gw: int, end_gw: int) -> List[Dict]:
        """
        Identify teams with good fixture runs
        
        Returns list of fixture runs sorted by quality
        """
        fixtures_df = self.data_collector.get_fixtures()
        teams_df = self.data_collector.get_team_strengths()
        
        if fixtures_df is None or teams_df is None:
            return []
        
        fixture_runs = []
        
        # Focus on top 10 teams (better assets)
        top_teams = teams_df.nlargest(10, 'strength')['id'].tolist() if 'strength' in teams_df.columns else teams_df['id'].tolist()
        
        for team_id in top_teams:
            # Get team fixtures in range
            team_fixtures = fixtures_df[
                ((fixtures_df['team_h'] == team_id) | (fixtures_df['team_a'] == team_id)) &
                (fixtures_df['event'] >= start_gw) & 
                (fixtures_df['event'] <= end_gw)
            ].copy()
            
            if team_fixtures.empty:
                continue
            
            # Calculate difficulty for each fixture
            team_fixtures = team_fixtures.sort_values('event')
            difficulties = []
            
            for _, fixture in team_fixtures.iterrows():
                difficulty = self._calculate_fixture_difficulty_rating(fixture, team_id, teams_df)
                difficulties.append({
                    'gw': fixture['event'],
                    'difficulty': difficulty
                })
            
            # Find runs of 3+ consecutive easy fixtures
            runs = self._find_consecutive_easy_runs(difficulties)
            
            # Add team info
            team_name = teams_df[teams_df['id'] == team_id]['name'].iloc[0] if len(teams_df[teams_df['id'] == team_id]) > 0 else f"Team {team_id}"
            team_strength = teams_df[teams_df['id'] == team_id]['strength'].iloc[0] if 'strength' in teams_df.columns else 1200
            
            for run in runs:
                fixture_runs.append({
                    'team_id': team_id,
                    'team_name': team_name,
                    'team_strength': team_strength,
                    'start_gw': run['start_gw'],
                    'end_gw': run['end_gw'],
                    'length': run['length'],
                    'avg_difficulty': run['avg_difficulty']
                })
        
        # Sort by quality: team strength * length * ease
        fixture_runs.sort(
            key=lambda x: (x['team_strength'] / 1200) * x['length'] * (5 - x['avg_difficulty']),
            reverse=True
        )
        
        return fixture_runs[:10]  # Top 10 fixture runs
    
    def _calculate_fixture_difficulty_rating(
        self, 
        fixture: pd.Series, 
        team_id: int,
        teams_df: pd.DataFrame
    ) -> float:
        """
        Calculate fixture difficulty rating (1-5 scale)
        1 = Very Easy, 5 = Very Hard
        
        Based on opponent strength
        """
        is_home = fixture['team_h'] == team_id
        opponent_id = fixture['team_a'] if is_home else fixture['team_h']
        
        # Get opponent strength
        opponent_data = teams_df[teams_df['id'] == opponent_id]
        if opponent_data.empty:
            return 3.0  # Average
        
        # Use FPL strength ratings (typically 1000-1400)
        opponent_strength = opponent_data['strength'].iloc[0] if 'strength' in opponent_data.columns else 1200
        
        # Convert to 1-5 scale
        # Strong opponent (1400) = 5 (hard)
        # Weak opponent (1000) = 1 (easy)
        # Average (1200) = 3
        difficulty = 1 + ((opponent_strength - 1000) / 100)
        
        # Home advantage
        if is_home:
            difficulty -= 0.3
        else:
            difficulty += 0.3
        
        return max(1.0, min(5.0, difficulty))
    
    def _find_consecutive_easy_runs(self, difficulties: List[Dict]) -> List[Dict]:
        """Find runs of 3+ consecutive easy fixtures"""
        runs = []
        current_run = []
        
        for item in difficulties:
            if item['difficulty'] < self.EASY_FIXTURE:
                current_run.append(item)
            else:
                # Run ended
                if len(current_run) >= self.MIN_RUN_LENGTH:
                    runs.append({
                        'start_gw': current_run[0]['gw'],
                        'end_gw': current_run[-1]['gw'],
                        'length': len(current_run),
                        'avg_difficulty': sum(x['difficulty'] for x in current_run) / len(current_run)
                    })
                current_run = []
        
        # Check final run
        if len(current_run) >= self.MIN_RUN_LENGTH:
            runs.append({
                'start_gw': current_run[0]['gw'],
                'end_gw': current_run[-1]['gw'],
                'length': len(current_run),
                'avg_difficulty': sum(x['difficulty'] for x in current_run) / len(current_run)
            })
        
        return runs
    
    def _explain_timing(
        self, 
        recommended_gw: int, 
        fixture_runs: List[Dict],
        current_gw: int
    ) -> Dict:
        """Generate strategic explanation for wildcard timing"""
        
        # Find fixture runs that align with wildcard timing
        # Wildcard should be 0-2 GWs before fixture run starts
        relevant_runs = [
            run for run in fixture_runs 
            if recommended_gw <= run['start_gw'] <= recommended_gw + 2
        ]
        
        # Generate reasons
        reasons = []
        
        if relevant_runs:
            top_run = relevant_runs[0]
            reasons.append(
                f"Targets {top_run['team_name']} fixture run "
                f"(GW{top_run['start_gw']}-{top_run['end_gw']}, "
                f"{top_run['length']} games, avg difficulty {top_run['avg_difficulty']:.1f})"
            )
        
        if recommended_gw <= current_gw + 3:
            reasons.append("Early wildcard allows time to recover team value")
        elif recommended_gw >= 15:
            reasons.append("Late wildcard to prepare for final run-in")
        
        return {
            'primary_reason': reasons[0] if reasons else "Mathematically optimal timing",
            'all_reasons': reasons,
            'aligned_fixture_runs': relevant_runs[:3]  # Top 3 relevant runs
        }
    
    def _find_alternatives(
        self, 
        scenarios: List[Dict], 
        best_points: float
    ) -> List[Dict]:
        """Find alternative GWs within 10 points of best"""
        alternatives = []
        
        for scenario in scenarios:
            if scenario.get('is_baseline'):
                continue
            
            point_diff = best_points - scenario['total_points']
            
            # Within 10 points of best
            if 0 < point_diff <= 10:
                alternatives.append({
                    'gameweek': scenario['gameweek'],
                    'total_points': scenario['total_points'],
                    'points_from_best': -point_diff
                })
        
        return sorted(alternatives, key=lambda x: x['points_from_best'], reverse=True)
    
    def _print_summary(
        self,
        best: Dict,
        baseline: Dict,
        fixture_runs: List[Dict],
        alternatives: List[Dict]
    ):
        """Print formatted summary of wildcard analysis"""
        print("\n" + "="*60)
        print("🃏 WILDCARD TIMING RECOMMENDATION")
        print("="*60)
        
        print(f"\n✅ Recommended: Use Wildcard in GW{best['gameweek']}")
        print(f"   Expected benefit: +{best['benefit']:.1f} points")
        print(f"   Total points (GW-end): {best['total_points']:.0f}")
        print(f"   Without wildcard: {baseline['total_points']:.0f}")
        
        if fixture_runs:
            print(f"\n📅 Strategic Factors:")
            for run in fixture_runs[:3]:
                print(f"   ✓ {run['team_name']}: "
                      f"GW{run['start_gw']}-{run['end_gw']} "
                      f"({run['length']} easy games, avg diff {run['avg_difficulty']:.1f})")
        
        if alternatives:
            print(f"\n🔄 Alternative Options:")
            for alt in alternatives[:3]:
                diff = best['total_points'] - alt['total_points']
                print(f"   • GW{alt['gameweek']}: "
                      f"{alt['total_points']:.0f} pts (-{diff:.0f} from best)")
        
        print("="*60)

