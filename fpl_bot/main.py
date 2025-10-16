"""
Main FPL Bot interface

Provides a clean interface to run the FPL bot with all functionality:
- Data collection and analysis
- Manager team analysis
- Transfer optimization
- Chip management
- Recommendations and reports
"""

import argparse
import json
from typing import Dict, Optional, List
from datetime import datetime
import pandas as pd
import threading
import webbrowser
import time

from .core.data_collector import DataCollector
from .core.manager_analyzer import ManagerAnalyzer
from .core.predictor import Predictor
from .core.transfer_optimizer import TransferOptimizer
from .core.chip_manager import ChipManager
from .core.multi_period_planner import MultiPeriodPlanner
from .ui.app import run_ui, set_report


class FPLBot:
    """Main FPL Bot class that orchestrates all functionality"""
    
    def __init__(self):
        self.data_collector = DataCollector()
        self.manager_analyzer = ManagerAnalyzer(self.data_collector)
        self.predictor = Predictor(self.data_collector)
        self.transfer_optimizer = TransferOptimizer(self.predictor)
        self.chip_manager = ChipManager(self.data_collector)
        self.multi_period_planner = MultiPeriodPlanner(
            self.data_collector, self.predictor, self.chip_manager
        )
        
        # Cache for data
        self.current_season_data = None
        self.players_df = None
        self.fixtures_df = None
        
    def run_analysis(self, manager_id: int) -> Dict:
        """Run complete FPL analysis for a manager"""
        print(f"Starting FPL Bot analysis for manager {manager_id}")
        print("=" * 60)
        
        # Step 1: Collect data
        print("\nStep 1: Collecting data...")
        self._collect_all_data()
        
        # Step 2: Analyze manager
        print("\nStep 2: Analyzing manager...")
        manager_analysis = self.manager_analyzer.analyze_manager(manager_id)
        
        # Step 3: Generate predictions
        print("\nStep 3: Generating predictions...")
        teams_data = self.current_season_data.get('teams', []) if self.current_season_data else []
        predictions = self.predictor.predict_next_gameweek(self.players_df, self.fixtures_df, teams_data)
        
        # Step 4: Optimize transfers
        print("\nStep 4: Optimizing transfers...")
        current_team = self._get_current_team_ids(manager_analysis)
        saved_transfers = manager_analysis.get('manager_info', {}).get('saved_transfers', {'free_transfers': 1})
        
        # Debug: show what saved_transfers contains
        print(f"[DEBUG] saved_transfers dict: {saved_transfers}")
        
        free_transfers = saved_transfers.get('free_transfers', 1)
        transfers_made_this_gw = saved_transfers.get('transfers_this_gw', 0)
        
        print(f"Available free transfers: {free_transfers}")
        transfer_analysis = self.transfer_optimizer.optimize_transfers(
            current_team, predictions, 
            budget=manager_analysis.get('manager_info', {}).get('bank', 1.0),
            free_transfers=free_transfers
        )
        
        # Ensure transfer_analysis is not None
        if transfer_analysis is None:
            transfer_analysis = {'optimized_team': {'transfers_made': 0, 'message': 'Transfer analysis failed'}}
        
        # Step 5: Multi-Period Planning & Chip Optimization
        # NEW WORKFLOW:
        # 1. ML model identifies favorable matchup windows (strong teams vs weak opposition)
        # 2. ML model predicts player points for each gameweek considering matchups
        # 3. MIP optimizes team selection towards optimal team each GW
        # 4. MIP integrates chip decisions to maximize total points over horizon
        print("\nStep 5: Creating multi-gameweek strategic plan...")
        chip_status = self.chip_manager.get_chip_status(manager_id)
        current_gw = self._get_current_gameweek()
        
        multi_period_plan = None
        if current_gw:
            try:
                # Get team AFTER current GW transfers for planning
                optimized_team_data = transfer_analysis.get('optimized_team', {})
                if optimized_team_data.get('transfers_made', 0) > 0:
                    new_team_players = optimized_team_data.get('team_players', [])
                    planning_team = [p['player_id'] for p in new_team_players]
                    players_in = optimized_team_data.get('players_in', [])
                    if players_in:
                        print(f"Planning with NEW team (includes: {', '.join([p['web_name'] for p in players_in])})")
                else:
                    planning_team = current_team
                
                # Run multi-period optimization
                multi_period_plan = self.multi_period_planner.plan_gameweeks(
                    current_team=planning_team,
                    current_gw=current_gw,
                    budget=manager_analysis.get('manager_info', {}).get('bank', 1.0),
                    free_transfers=free_transfers,
                    predictions_df=predictions,
                    fixtures_df=self.fixtures_df,
                    teams_data=teams_data,
                    chip_status=chip_status
                )
            except Exception as e:
                print(f"Warning: Multi-period planning failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Extract chip recommendations from multi-period plan
        chip_recommendations = multi_period_plan['chip_plan'] if multi_period_plan else {}
        
        # Step 6: Generate recommendations
        print("\nStep 6: Generating recommendations...")
        recommendations = self._generate_recommendations(
            manager_analysis, transfer_analysis, chip_recommendations, predictions, multi_period_plan
        )
        
        # Step 7: Create report
        print("\nStep 7: Creating report...")
        report = self._create_report(
            manager_analysis, transfer_analysis, chip_recommendations, 
            predictions, recommendations, multi_period_plan
        )
        
        print("\n" + "=" * 60)
        print("Analysis complete!")
        print("=" * 60)
        
        return report
    
    def _collect_all_data(self):
        """Collect all necessary data"""
        # Get current season data
        self.current_season_data = self.data_collector.get_current_season_data()
        if self.current_season_data:
            self.players_df = self.data_collector.create_players_dataframe(self.current_season_data)
        
        # Get fixtures data
        self.fixtures_df = pd.DataFrame(self.data_collector.get_fixtures_data() or [])
        
        print(f"Collected data for {len(self.players_df)} players")
    
    def _get_current_team_ids(self, manager_analysis: Dict) -> list:
        """Extract current team player IDs from manager analysis"""
        team_analysis = manager_analysis.get('team_analysis', {})
        players = team_analysis.get('players', [])
        return [player['id'] for player in players]
    
    def _get_current_gameweek(self) -> Optional[int]:
        """Get current gameweek number"""
        try:
            if self.current_season_data and 'events' in self.current_season_data:
                events = self.current_season_data['events']
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
                cur = self.current_season_data.get('current-event')
                if cur:
                    return cur
            return None
        except:
            return None
    
    def _generate_recommendations(self, 
                                manager_analysis: Dict,
                                transfer_analysis: Dict,
                                chip_recommendations: Dict,
                                predictions: pd.DataFrame,
                                multi_period_plan: Optional[Dict] = None) -> Dict:
        """Generate comprehensive recommendations with definitive decisions"""
        recommendations = {
            'transfer_decision': None,
            'chip_decision': None,
            'captain_decision': None,
            'confidence_scores': {}
        }
        
        # Transfer decision
        best_scenario = transfer_analysis.get('best_scenario')
        if best_scenario and best_scenario.get('num_transfers', 0) > 0:
            # Calculate confidence based on net points gained
            net_gain = best_scenario['net_points_gained']
            confidence = min(100, max(0, (net_gain / 10) * 100))  # Scale to 0-100%
            
            recommendations['transfer_decision'] = {
                'action': 'MAKE_TRANSFERS',
                'num_transfers': best_scenario['num_transfers'],
                'net_points_gained': best_scenario['net_points_gained'],
                'players_out': [p['web_name'] for p in best_scenario['players_out']],
                'players_in': [p['web_name'] for p in best_scenario['players_in']],
                'transfer_cost': best_scenario['transfer_cost'],
                'confidence': round(confidence, 1)
            }
            recommendations['confidence_scores']['transfers'] = round(confidence, 1)
        else:
            recommendations['transfer_decision'] = {
                'action': 'NO_TRANSFERS',
                'reason': 'No beneficial transfers identified',
                'confidence': 100.0
            }
            recommendations['confidence_scores']['transfers'] = 100.0
        
        # Chip decision from multi-period plan (if available)
        current_gw = multi_period_plan.get('start_gw') if multi_period_plan else None
        
        # Check if multi-period plan recommends chip usage THIS week
        use_chip_now = False
        chip_this_week = None
        
        if multi_period_plan and current_gw:
            for chip_name, chip_info in chip_recommendations.items():
                if chip_info.get('recommended') and chip_info.get('best_gw') == current_gw:
                    use_chip_now = True
                    chip_this_week = chip_name
                    best_chip_score = chip_info.get('expected_benefit', 0)
                    break
        
        if use_chip_now and chip_this_week:
            # Multi-period plan recommends using chip NOW
            chip_details = chip_recommendations[chip_this_week].get('details', {})
            reason = f"Optimal timing based on 5-GW analysis"
            if chip_this_week == 'triple_captain':
                captain = chip_details.get('captain_name', 'best player')
                reason = f"Best captain option ({captain}) peaks this week"
            
            recommendations['chip_decision'] = {
                'action': f'USE_{chip_this_week.upper()}',
                'chip': chip_this_week,
                'expected_benefit': best_chip_score,
                'reason': reason,
                'confidence': min(100, (best_chip_score / 20) * 100)
            }
            recommendations['confidence_scores']['chip'] = recommendations['chip_decision']['confidence']
        else:
            # Don't use chip this week - either save for later or no good option
            future_chip_gw = None
            future_chip_name = None
            
            if multi_period_plan:
                for chip_name, chip_info in chip_recommendations.items():
                    if chip_info.get('recommended') and chip_info.get('best_gw', 0) > current_gw:
                        future_chip_gw = chip_info['best_gw']
                        future_chip_name = chip_name
                        break
            
            if future_chip_gw:
                recommendations['chip_decision'] = {
                    'action': 'NO_CHIP',
                    'reason': f'Save {future_chip_name.replace("_", " ").title()} for GW{future_chip_gw} (better opportunity)',
                    'confidence': 95.0,
                    'save_for': f'GW{future_chip_gw}'
                }
            else:
                recommendations['chip_decision'] = {
                    'action': 'NO_CHIP',
                    'reason': 'No chip provides sufficient benefit this gameweek',
                    'confidence': 90.0
                }
            recommendations['confidence_scores']['chip'] = recommendations['chip_decision']['confidence']
        
        # Captain decision (always pick one)
        captain_options = self.predictor.predict_captain_options(predictions, self._get_current_team_ids(manager_analysis))
        if not captain_options.empty:
            best_captain = captain_options.iloc[0]
            second_best = captain_options.iloc[1] if len(captain_options) > 1 else None
            
            # Calculate confidence based on gap to second best
            if second_best is not None:
                gap = best_captain['predicted_points'] - second_best['predicted_points']
                confidence = min(100, 70 + (gap * 5))  # Base 70% + gap bonus
            else:
                confidence = 95.0
            
            recommendations['captain_decision'] = {
                'action': 'CAPTAIN',
                'player': best_captain['web_name'],
                'predicted_points': round(best_captain['predicted_points'], 1),
                'alternatives': captain_options.head(3)[['web_name', 'predicted_points']].to_dict('records'),
                'confidence': round(confidence, 1)
            }
            recommendations['confidence_scores']['captain'] = round(confidence, 1)
        
        return recommendations
    
    def _create_report(self, 
                      manager_analysis: Dict,
                      transfer_analysis: Dict,
                      chip_recommendations: Dict,
                      predictions: pd.DataFrame,
                      recommendations: Dict,
                      multi_period_plan: Optional[Dict] = None) -> Dict:
        """Create comprehensive analysis report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = {
            'timestamp': timestamp,
            'manager_info': manager_analysis.get('manager_info', {}),
            'team_analysis': manager_analysis.get('team_analysis', {}),
            'performance_analysis': manager_analysis.get('performance_analysis', {}),
            'predictions_summary': {
                'total_players_analyzed': len(predictions),
                'top_predictions': predictions.head(10)[['web_name', 'team_name', 'predicted_points']].to_dict('records')
            },
            'transfer_analysis': transfer_analysis,
            'optimized_team': transfer_analysis.get('optimized_team', {}),
            'chip_recommendations': chip_recommendations,
            'recommendations': recommendations,
            'next_steps': self._generate_next_steps(recommendations),
            'multi_period_plan': multi_period_plan  # 5-GW strategic plan with chip optimization
        }
        
        return report
    
    def _generate_next_steps(self, recommendations: Dict) -> list:
        """Generate definitive actions to execute"""
        actions = []
        
        # Transfer action
        transfer_decision = recommendations.get('transfer_decision') or {}
        if transfer_decision.get('action') == 'MAKE_TRANSFERS':
            num = transfer_decision['num_transfers']
            gain = transfer_decision['net_points_gained']
            conf = transfer_decision['confidence']
            
            # Build transfer details string
            players_out = transfer_decision.get('players_out', [])
            players_in = transfer_decision.get('players_in', [])
            
            if players_out and players_in:
                # Encode player names to ASCII for Windows console compatibility
                def safe_name(name):
                    """Convert Unicode names to ASCII-safe versions"""
                    if isinstance(name, str):
                        # Try to encode as ASCII, replacing problematic characters
                        return name.encode('ascii', 'replace').decode('ascii').replace('?', '')
                    return str(name)
                
                # Create readable transfer list
                if num == 1:
                    out_name = safe_name(players_out[0])
                    in_name = safe_name(players_in[0])
                    transfer_detail = f" (OUT: {out_name}, IN: {in_name})"
                else:
                    # For multiple transfers, show as list
                    out_names = ", ".join([safe_name(n) for n in players_out])
                    in_names = ", ".join([safe_name(n) for n in players_in])
                    transfer_detail = f" (OUT: {out_names} | IN: {in_names})"
            else:
                transfer_detail = ""
            
            actions.append({
                'priority': 1,
                'action': 'EXECUTE_TRANSFERS',
                'description': f"Make {num} transfer(s) for {gain:.1f} net points gain{transfer_detail}",
                'confidence': conf,
                'executable': True
            })
        else:
            actions.append({
                'priority': 1,
                'action': 'HOLD_TRANSFERS',
                'description': 'Keep current team - no beneficial transfers available',
                'confidence': 100.0,
                'executable': True
            })
        
        # Captain action (always required)
        captain_decision = recommendations.get('captain_decision') or {}
        if captain_decision.get('action') == 'CAPTAIN':
            player = captain_decision['player']
            points = captain_decision['predicted_points']
            conf = captain_decision['confidence']
            actions.append({
                'priority': 2,
                'action': 'SET_CAPTAIN',
                'description': f"Captain {player} ({points:.1f} predicted points)",
                'confidence': conf,
                'executable': True
            })
        
        # Chip action
        chip_decision = recommendations.get('chip_decision') or {}
        if chip_decision.get('action') != 'NO_CHIP':
            chip = chip_decision['chip']
            benefit = chip_decision.get('expected_benefit', 0)
            conf = chip_decision['confidence']
            actions.append({
                'priority': 3,
                'action': chip_decision['action'],
                'description': f"Activate {chip.upper()} chip ({benefit:.1f} expected benefit)",
                'confidence': conf,
                'executable': True
            })
        else:
            actions.append({
                'priority': 3,
                'action': 'NO_CHIP_USED',
                'description': 'Do not use any chip this gameweek',
                'confidence': chip_decision['confidence'],
                'executable': True
            })
        
        return actions
    
    def _print_multi_period_plan(self, plan: Dict):
        """Print the multi-gameweek strategic plan"""
        
        num_weeks = plan.get('horizon', 7)
        
        print("\n" + "="*70)
        print(f"{num_weeks}-GAMEWEEK STRATEGIC PLAN (GW{plan['start_gw']}-{plan['end_gw']})")
        print("="*70)
        
        # Week-by-Week Breakdown
        print("\nWEEK-BY-WEEK PLAN:")
        print("="*70)
        
        team_evolution = plan.get('team_evolution', {})
        chip_plan = plan.get('chip_plan', {})
        
        for offset in range(plan.get('horizon', 5)):
            gw = plan['start_gw'] + offset
            
            if gw not in team_evolution:
                continue
            
            week_plan = team_evolution[gw]
            
            # Header
            is_current = (offset == 0)
            marker = ">>> " if is_current else "    "
            print(f"\n{marker}GAMEWEEK {gw}{'  (THIS WEEK)' if is_current else ''}")
            print(f"{marker}{'-'*66}")
            
            # Transfers
            transfers = week_plan.get('transfers', [])
            num_transfers = week_plan.get('num_transfers', 0)
            
            if num_transfers > 0:
                print(f"{marker}Transfers: {num_transfers}")
                for t in transfers:
                    out_name = t.get('out_name', 'Unknown')
                    in_name = t.get('in_name', 'Unknown')
                    gain = t.get('gain', 0)
                    print(f"{marker}  OUT: {out_name} -> IN: {in_name} (+{gain:.1f} pts)")
                
                cost = week_plan.get('transfer_cost', 0)
                if cost > 0:
                    print(f"{marker}  Cost: {cost} pts hit")
                
                # Show FTs after this transfer
                ft_used = week_plan.get('free_transfers_used', 1)
                print(f"{marker}Free Transfers Used: {ft_used}")
            else:
                # No transfers - show FT status
                ft_available = week_plan.get('free_transfers_available', 1)
                
                if ft_available >= 5:
                    print(f"{marker}Transfers: None (already at max 5 FTs)")
                elif offset > 0:
                    print(f"{marker}Transfers: None (banking FT - will have {ft_available} total)")
                else:
                    print(f"{marker}Transfers: None")
                
                print(f"{marker}Free Transfers Available: {ft_available}")
            
            # Chip usage
            chip_this_week = None
            for chip_name, chip_info in chip_plan.items():
                if chip_info.get('best_gw') == gw and chip_info.get('recommended'):
                    chip_this_week = chip_name
                    details = chip_info.get('details', {})
                    
                    if chip_name == 'triple_captain':
                        captain = details.get('captain_name', 'Unknown')
                        pts = details.get('captain_points', 0)
                        print(f"{marker}Chip: TRIPLE CAPTAIN on {captain} (2x{pts:.1f} = {pts*2:.1f} pts)")
                    elif chip_name == 'bench_boost':
                        bench_pts = details.get('bench_points', 0)
                        print(f"{marker}Chip: BENCH BOOST ({bench_pts:.1f} pts from bench)")
                    elif chip_name == 'wildcard':
                        after_pts = details.get('expected_points_after', week_plan.get('expected_points', 0))
                        before_pts = details.get('expected_points_before', 0)
                        delta = after_pts - before_pts
                        print(f"{marker}Chip: WILDCARD (squad refresh, +{delta:.1f} pts vs prior)")
                    elif chip_name == 'free_hit':
                        after_pts = details.get('expected_points_after', week_plan.get('expected_points', 0))
                        before_pts = details.get('expected_points_before', 0)
                        delta = after_pts - before_pts
                        print(f"{marker}Chip: FREE HIT (one-week XI, +{delta:.1f} pts vs prior)")
                    break
            
            if not chip_this_week and offset == 0:
                print(f"{marker}Chip: None")
            
            # Expected points for this GW (if available in plan)
            if 'expected_points' in week_plan:
                print(f"{marker}Expected Points: {week_plan['expected_points']:.1f}")
        
        # Fixture Runs
        print("\nFIXTURE RUN OPPORTUNITIES:")
        fixture_runs = plan.get('fixture_runs', [])
        if fixture_runs:
            premium_runs = [r for r in fixture_runs if r.get('team_position', 20) <= 6]
            if premium_runs:
                print("Top teams with favorable fixture runs:")
                for i, run in enumerate(premium_runs[:3], 1):
                    fixtures_str = ", ".join([
                        f"vs {f['opponent']} ({'H' if f['is_home'] else 'A'})"
                        for f in run['fixtures'][:4]  # Show first 4
                    ])
                    print(f"{i}. {run['team_name']} [#{run['team_position']}] (GW{run['start_gw']}-{run['end_gw']}): {fixtures_str}")
                    print(f"   Avg Difficulty: {run['avg_difficulty']:.2f} - {run['recommendation']}")
            else:
                print("No premium teams (top 6) have significant fixture runs")
                other_runs = fixture_runs[:2]
                if other_runs:
                    print("\nOther fixture runs (not recommended):")
                    for run in other_runs:
                        print(f"  {run['team_name']} [#{run['team_position']}] - {run['recommendation']}")
        else:
            print("No significant fixture runs identified")
        
        # Chip Timing
        print(f"\n{'='*70}")
        print("CHIP TIMING RECOMMENDATIONS:")
        print("="*70)
        
        chip_plan = plan.get('chip_plan', {})
        has_recommendations = False
        
        for chip_name in ['triple_captain', 'bench_boost', 'wildcard', 'free_hit']:
            if chip_name not in chip_plan:
                continue
            
            chip_info = chip_plan[chip_name]
            if not chip_info.get('best_gw') and chip_name != 'wildcard':
                continue
            
            has_recommendations = True
            status = "[RECOMMENDED]" if chip_info.get('recommended') else "[Possible]"
            
            # Build detailed message
            if chip_name == 'triple_captain':
                details = chip_info.get('details', {})
                captain = details.get('captain_name', 'Unknown')
                captain_pts = details.get('captain_points', 0)
                print(f"\nTRIPLE CAPTAIN: GW{chip_info['best_gw']} {status}")
                print(f"  Captain: {captain}")
                print(f"  Expected Points: {captain_pts:.1f} pts (2x = {chip_info['expected_benefit']:.1f} pts total)")
                
                # Show alternatives
                alternatives = details.get('all_options', [])
                if len(alternatives) > 1:
                    print(f"  Alternatives:")
                    for alt in alternatives[1:4]:  # Show next 3
                        print(f"    - {alt['web_name']}: {alt['predicted_points']:.1f} pts")
            
            elif chip_name == 'bench_boost':
                details = chip_info.get('details', {})
                bench_pts = details.get('bench_points', 0)
                bench_players = details.get('bench_players', [])
                
                print(f"\nBENCH BOOST: GW{chip_info['best_gw']} {status}")
                print(f"  Bench Total: {bench_pts:.1f} pts")
                
                if bench_players:
                    print(f"  Bench Players:")
                    for bp in bench_players:
                        print(f"    - {bp['web_name']} ({bp['position']}): {bp['predicted_points']:.1f} pts")
            elif chip_name == 'wildcard':
                # Print wildcard even if not recommended, if a best_gw was set
                if chip_info.get('best_gw'):
                    details = chip_info.get('details', {})
                    after_pts = details.get('expected_points_after', 0)
                    before_pts = details.get('expected_points_before', 0)
                    delta = details.get('delta', max(0, after_pts - before_pts))
                    print(f"\nWILDCARD: GW{chip_info['best_gw']} {status}")
                    print(f"  Expected Points After: {after_pts:.1f}")
                    print(f"  Prior Week Points: {before_pts:.1f}")
                    print(f"  Delta: +{delta:.1f} pts")
                else:
                    # Not chosen within horizon
                    print(f"\nWILDCARD: Not recommended in the next {num_weeks} GWs")
            elif chip_name == 'free_hit':
                if chip_info.get('best_gw'):
                    details = chip_info.get('details', {})
                    after_pts = details.get('expected_points_after', 0)
                    before_pts = details.get('expected_points_before', 0)
                    delta = details.get('delta', max(0, after_pts - before_pts))
                    print(f"\nFREE HIT: GW{chip_info['best_gw']} {status}")
                    print(f"  Expected Points: {after_pts:.1f}")
                    print(f"  Baseline (prior week): {before_pts:.1f}")
                    print(f"  Delta: +{delta:.1f} pts")
        
        if not has_recommendations:
            print("No chip usage recommended in next 5 gameweeks")
        
        # Key Recommendations
        recommendations = plan.get('recommendations', {})
        if recommendations:
            immediate = recommendations.get('immediate_actions', [])
            chip_recs = recommendations.get('chip_recommendations', [])
            fixture_opps = recommendations.get('fixture_opportunities', [])
            
            if immediate or chip_recs or fixture_opps:
                print(f"\n{'='*70}")
                print("STRATEGIC RECOMMENDATIONS:")
                print("="*70)
                
                if immediate:
                    print("\nImmediate Actions (This Week):")
                    for action in immediate:
                        print(f"  - {action}")
                
                if chip_recs:
                    print("\nChip Strategy:")
                    for rec in chip_recs:
                        print(f"  - {rec}")
                
                if fixture_opps:
                    print("\nFixture Opportunities:")
                    for opp in fixture_opps:
                        print(f"  - {opp}")
        
        print()
    
    def _print_chip_planning(self, chip_recommendations: Dict, new_players: List[str] = None):
        """Print detailed chip planning analysis
        
        Args:
            chip_recommendations: Chip recommendations dict
            new_players: List of web_names of newly transferred in players
        """
        # Get current gameweek to determine planning horizon
        current_gw = self._get_current_gameweek()
        christmas_gw = 19
        
        # Determine planning window message
        if current_gw and current_gw < christmas_gw:
            window_text = f"Up to GW{christmas_gw} (Before Christmas)"
            num_to_show = min(christmas_gw - current_gw + 1, 15)  # Show all up to Christmas
        else:
            window_text = "Next 10 Gameweeks (After Christmas)"
            num_to_show = 10
        
        print(f"\n{'=' * 60}")
        print(f"CHIP PLANNING ANALYSIS ({window_text})")
        print(f"{'=' * 60}")
        
        # Triple Captain Planning
        tc_rec = chip_recommendations.get('triple_captain', {})
        if tc_rec and 'planning_details' in tc_rec:
            details = tc_rec['planning_details']
            print(f"\nTRIPLE CAPTAIN OPPORTUNITIES:")
            
            opportunities = details.get('all_weeks', [])[:num_to_show]
            if opportunities:
                print(f"{'GW':<4} {'Player':<15} {'Opponent':<15} {'Venue':<6} {'Diff':<5} {'Score':<6}")
                print("-" * 60)
                for opp in opportunities:
                    gw = opp.get('gameweek', 'N/A')
                    if gw == 'current':
                        gw = '**'  # Current week marker
                    player = opp.get('player', 'N/A')[:14]
                    opponent = opp.get('opponent', 'N/A')[:14]
                    venue = opp.get('venue', 'N/A')[:5]
                    diff = opp.get('difficulty', 0)
                    score = opp.get('gameweek_score', 0)
                    rating = opp.get('fixture_rating', '')
                    
                    # Highlight if meets TC criteria
                    is_elite = opp.get('predicted_points', 0) >= 7.0
                    is_easy = diff <= 2.8
                    meets_criteria = is_elite and is_easy and score >= 9.0
                    marker = "*" if meets_criteria else " "
                    
                    # Mark if this is a new signing
                    is_new = (new_players and player.strip() in [p.strip() for p in new_players])
                    new_marker = "[NEW]" if is_new else ""
                    
                    print(f"{gw:<4} {player:<15} {opponent:<15} {venue:<6} {diff:<5.1f} {score:<6.1f} {marker} {rating:<10} {new_marker}")
                
                print("\n* = Meets TC criteria (Elite player + Easy fixture + Score > 9.0)")
                print("** = Current gameweek")
                
                # Show new players' fixtures separately if they're not in top opportunities
                if new_players:
                    all_weeks = details.get('all_weeks', [])
                    new_player_fixtures = [opp for opp in all_weeks
                                          if opp.get('player', '').strip() in [p.strip() for p in new_players]]
                    
                    # Only show if new players have fixtures but aren't in top opportunities already
                    top_players = set([opp.get('player', '').strip() for opp in opportunities])
                    new_not_shown = [p for p in new_players if p.strip() not in top_players]
                    
                    if new_not_shown and new_player_fixtures:
                        print(f"\nNEW SIGNINGS' FIXTURES:")
                        print(f"{'GW':<4} {'Player':<15} {'Opponent':<15} {'Venue':<6} {'Diff':<5} {'Score':<6}")
                        print("-" * 60)
                        
                        shown_count = 0
                        for opp in new_player_fixtures[:num_to_show]:
                            player = opp.get('player', 'N/A')[:14]
                            if player.strip() not in new_not_shown:
                                continue
                                
                            gw = opp.get('gameweek', 'N/A')
                            if gw == 'current':
                                gw = '**'
                            opponent = opp.get('opponent', 'N/A')[:14]
                            venue = opp.get('venue', 'N/A')[:5]
                            diff = opp.get('difficulty', 0)
                            score = opp.get('gameweek_score', 0)
                            rating = opp.get('fixture_rating', '')
                            
                            print(f"{gw:<4} {player:<15} {opponent:<15} {venue:<6} {diff:<5.1f} {score:<6.1f}   {rating}")
                            shown_count += 1
                
                # Show recommendation
                if tc_rec.get('recommended'):
                    print(f"\n> RECOMMEND: Use TC NOW")
                else:
                    save_for = tc_rec.get('save_for', 'Better opportunity')
                    print(f"\n> RECOMMEND: {save_for}")
            else:
                print("No TC opportunities found in next 5 gameweeks")
        
        # Bench Boost Planning
        bb_rec = chip_recommendations.get('bench_boost', {})
        if bb_rec and 'planning_details' in bb_rec:
            details = bb_rec['planning_details']
            print(f"\n\nBENCH BOOST OPPORTUNITIES:")
            
            opportunities = details.get('all_weeks', [])[:num_to_show]
            if opportunities:
                print(f"{'GW':<6} {'Bench Score':<12} {'Players':<8}")
                print("-" * 30)
                for opp in opportunities:
                    gw = opp.get('gameweek', 'N/A')
                    if gw == 'current':
                        gw = '**'  # Current week marker
                    score = opp.get('bench_score', 0)
                    count = opp.get('player_count', 0)
                    
                    # Highlight if good BB week
                    is_good = score >= 8.0
                    marker = "*" if is_good else " "
                    
                    # Check if any new players contribute to this week
                    has_new_bench = ""
                    if new_players and 'bench_players' in opp:
                        bench_names = [p.get('web_name', '') for p in opp.get('bench_players', [])]
                        if any(new_p in bench_names for new_p in new_players):
                            has_new_bench = "[NEW]"
                    
                    print(f"{str(gw):<6} {score:<12.1f} {count:<8} {marker} {has_new_bench}")
                
                print("\n* = Good BB week (Bench 8+ points)")
                print("** = Current gameweek")
                print("[NEW] = Week includes new signing(s) on bench")
                
                # Show top bench week details
                if opportunities:
                    best_week = opportunities[0]
                    bench_players = best_week.get('bench_players', [])
                    if bench_players:
                        print(f"\nBest BB Week (GW{best_week['gameweek']}) Bench:")
                        for bp in sorted(bench_players, key=lambda x: x.get('predicted_points', 0), reverse=True):
                            name = bp.get('web_name', 'Unknown')
                            pts = bp.get('predicted_points', 0)
                            pos = bp.get('position_name', 'UNK')
                            is_new = (new_players and name in new_players)
                            new_tag = " [NEW]" if is_new else ""
                            print(f"  {name} ({pos}): {pts:.1f} pts{new_tag}")
                
                # Show recommendation
                if bb_rec.get('recommended'):
                    print(f"\n> RECOMMEND: Use BB NOW")
                else:
                    save_for = bb_rec.get('save_for', 'Better opportunity')
                    print(f"\n> RECOMMEND: {save_for}")
            else:
                print("No BB opportunities analyzed")
    
    def print_terminal_status(self, report: Dict):
        """Print minimal status info to terminal (debug info only)"""
        print("\n" + "=" * 60)
        print("FPL BOT - Analysis Complete")
        print("=" * 60)
        
        manager_info = report.get('manager_info', {})
        if manager_info:
            print(f"Manager: {manager_info.get('manager_name', 'Unknown')}")
            print(f"Team: {manager_info.get('team_name', 'Unknown')}")
        
        plan = report.get('multi_period_plan') or {}
        current_gw = plan.get('start_gw')
        if current_gw:
            print(f"Current GW: {current_gw}")
        
        print("\nView full analysis at: http://127.0.0.1:5000")
        print("=" * 60 + "\n")
    
    def print_summary(self, report: Dict):
        """Print a detailed summary of the analysis (legacy/debug mode)"""
        print("\n" + "=" * 60)
        print("FPL BOT SUMMARY")
        print("=" * 60)

        # Manager info (condensed, ASCII-safe)
        manager_info = report.get('manager_info', {})
        if manager_info:
            name = str(manager_info.get('manager_name', 'Unknown'))
            team = str(manager_info.get('team_name', 'Unknown'))
            rank = manager_info.get('overall_rank', 'Unknown')
            total = manager_info.get('total_points', 'Unknown')
            value = manager_info.get('team_value', 0)
            bank = manager_info.get('bank', 0)
            saved_transfers = manager_info.get('saved_transfers', {})
            ft = saved_transfers.get('free_transfers', 1)
            print(f"Manager: {name}  |  Team: {team}")
            print(f"Rank: {rank}  |  Points: {total}  |  Value: {value:.1f}m  |  Bank: {bank:.1f}m  |  FTs: {ft}")
        else:
            print("Manager info not available")

        # Helper for safe encoding
        def safe_encode(name):
            return name.encode('ascii', 'replace').decode('ascii').replace('?', '') if isinstance(name, str) else str(name)
        
        # Get recommendations from multi-period plan (primary source)
        plan = report.get('multi_period_plan') or {}
        team_evolution = plan.get('team_evolution', {})
        current_gw = plan.get('start_gw')
        player_projections = plan.get('player_projections', {})  # Get player info for lookups
        
        # Use GW8 (current week) from multi-period plan if available
        this_week_plan = team_evolution.get(current_gw) if current_gw else None
        
        # Fallback to standalone recommendations if no multi-period plan
        recommendations = report.get('recommendations', {}) or {}
        transfer_decision = recommendations.get('transfer_decision', {})
        chip_decision = recommendations.get('chip_decision', {})
        captain_decision = recommendations.get('captain_decision', {})

        print("\nThis Week")
        
        # Initialize "this week" variables for Next Steps section
        this_week_num_transfers = 0
        this_week_transfers_list = []
        this_week_chip = None
        this_week_captain_name = 'Unknown'
        this_week_captain_pts = 0
        this_week_vice_captain_name = 'Unknown'
        
        # Use multi-period plan for this week if available (more accurate)
        # Save these values as they'll be overwritten by the loop
        if this_week_plan:
            num_transfers = this_week_plan.get('num_transfers', 0)
            transfers_list = this_week_plan.get('transfers', [])
            chip_this_week = this_week_plan.get('chip')
            captain_name = this_week_plan.get('captain_name', 'Unknown')
            captain_pts = this_week_plan.get('expected_points', 0)
            vice_captain_name = this_week_plan.get('vice_captain_name', 'Unknown')
            
            # Save for "Next Steps" section (loop will overwrite these variables)
            this_week_num_transfers = num_transfers
            this_week_transfers_list = transfers_list
            this_week_chip = chip_this_week
            this_week_captain_name = captain_name
            this_week_captain_pts = captain_pts
            this_week_vice_captain_name = vice_captain_name
            
            # Transfers
            if num_transfers > 0 and transfers_list:
                # Calculate expected gain
                total_gain = sum(t.get('gain', 0) for t in transfers_list)
                outs = ", ".join([safe_encode(t.get('out_name', '?')) for t in transfers_list[:3]])
                ins = ", ".join([safe_encode(t.get('in_name', '?')) for t in transfers_list[:3]])
                detail = f" (OUT: {outs} | IN: {ins})" if outs and ins else ""
                if len(transfers_list) > 3:
                    detail += f" + {len(transfers_list)-3} more"
                print(f"- Transfers: Make {num_transfers} for +{total_gain:.1f} pts{detail}")
            else:
                print("- Transfers: Hold (no transfers planned)")
            
            # Chip
            if chip_this_week:
                chip_display = chip_this_week.replace('_', ' ').title()
                print(f"- Chip: Use {chip_display}")
            else:
                print("- Chip: None")
            
            # Captain and Vice Captain
            captain_safe = safe_encode(captain_name)
            vice_captain_safe = safe_encode(vice_captain_name)
            # Get vice-captain predicted points if available
            vice_pts = 0
            if vice_captain_name and vice_captain_name != 'Unknown' and current_gw:
                # Try to find vice captain's predicted points
                for pid in this_week_plan.get('starting_xi', []):
                    if player_projections.get(pid, {}).get('web_name') == vice_captain_name:
                        vice_pts = player_projections.get(pid, {}).get('gameweek_predictions', {}).get(current_gw, 0)
                        break
            
            if vice_pts > 0:
                print(f"- Captain: {captain_safe} ({captain_pts:.1f} pts) | Vice: {vice_captain_safe} ({vice_pts:.1f} pts)")
            else:
                print(f"- Captain: {captain_safe} ({captain_pts:.1f} pts) | Vice: {vice_captain_safe}")
        
        else:
            # Fallback: Use standalone recommendations (old behavior)
            # Transfers (concise)
            if transfer_decision.get('action') == 'MAKE_TRANSFERS':
                num = transfer_decision.get('num_transfers', 0)
                gain = transfer_decision.get('net_points_gained', 0)
                outs = ", ".join([safe_encode(n) for n in transfer_decision.get('players_out', [])[:3]])
                ins = ", ".join([safe_encode(n) for n in transfer_decision.get('players_in', [])[:3]])
                detail = f" (OUT: {outs} | IN: {ins})" if outs and ins else ""
                print(f"- Transfers: Make {num} for +{gain:.1f} pts{detail}")
            else:
                print("- Transfers: Hold (no beneficial moves)")

            # Chip now
            if chip_decision and chip_decision.get('action') != 'NO_CHIP':
                chip = chip_decision.get('chip', 'chip')
                benefit = chip_decision.get('expected_benefit', 0)
                print(f"- Chip: Use {chip.replace('_',' ').title()} (+{benefit:.1f} pts)")
            else:
                print("- Chip: None")

            # Captain
            if captain_decision:
                print(f"- Captain: {captain_decision.get('player', 'Unknown')} ({captain_decision.get('predicted_points', 0):.1f} pts)")

        # Chip plan (next weeks)
        chip_plan = (report.get('multi_period_plan') or {}).get('chip_plan', {})
        if chip_plan:
            # Collect recommended chips with GW
            items = []
            for key in ['wildcard', 'free_hit', 'triple_captain', 'bench_boost']:
                info = chip_plan.get(key)
                if not info:
                    continue
                if info.get('best_gw') and info.get('recommended'):
                    if key == 'triple_captain':
                        cap = info.get('details', {}).get('captain_name', 'Best')
                        items.append(f"GW{info['best_gw']}: Triple Captain ({cap})")
                    elif key == 'bench_boost':
                        items.append(f"GW{info['best_gw']}: Bench Boost")
                    elif key == 'wildcard':
                        items.append(f"GW{info['best_gw']}: Wildcard")
                    elif key == 'free_hit':
                        items.append(f"GW{info['best_gw']}: Free Hit")
            if items:
                print("\nChip Plan")
                for line in items:
                    print(f"- {line}")

        # Planned actions per GW in horizon
        # Note: plan, team_evolution, and player_projections already defined above
        if plan and team_evolution:
            start_gw = plan.get('start_gw')
            horizon = plan.get('horizon', 0)
            print("\nMulti-Gameweek Plan Summary")
            print("-" * 80)
            # Track FTs manually based on FPL rules
            manager_info = report.get('manager_info', {}) or {}
            saved_transfers = manager_info.get('saved_transfers', {})
            current_fts = saved_transfers.get('free_transfers', 1)
            
            for offset in range(horizon):
                gw = start_gw + offset
                week = team_evolution.get(gw)
                if not week:
                    continue
                
                # Core info
                num_transfers = week.get('num_transfers', 0)
                transfer_cost = week.get('transfer_cost', 0)
                expected_pts = week.get('expected_points', 0)
                budget = week.get('budget_remaining', 0)
                captain_name = week.get('captain_name', 'Unknown')
                vice_captain_name = week.get('vice_captain_name', 'Unknown')
                
                # Chip info
                chip_this_week = None
                for cname, info in chip_plan.items():
                    if info.get('best_gw') == gw and info.get('recommended'):
                        chip_this_week = cname.replace('_', ' ').title()
                        break
                
                # Use tracked FTs (updated at end of previous iteration)
                fts_available = current_fts
                
                # Build summary line
                if num_transfers > 0:
                    if chip_this_week in ['Wildcard', 'Free Hit']:
                        transfer_str = f"{num_transfers} transfers (FREE - {chip_this_week})"
                    else:
                        # Recalculate transfer cost based on FTs available
                        # This ensures display matches actual FT tracking
                        actual_ft_used = min(num_transfers, fts_available) if fts_available is not None else 0
                        hits_taken = max(0, num_transfers - actual_ft_used)
                        actual_transfer_cost = hits_taken * 4
                        
                        if actual_transfer_cost > 0:
                            transfer_str = f"{num_transfers} transfers (-{hits_taken}x4 = -{actual_transfer_cost} pts)"
                        else:
                            transfer_str = f"{num_transfers} transfer(s) (Free)"
                else:
                    transfer_str = "HOLD"
                
                # Format FTs info - track properly according to FPL rules
                if fts_available is not None:
                    # Handle chips first (they affect FT tracking differently)
                    if chip_this_week in ['Wildcard', 'Free Hit']:
                        # Both WC and FH: Don't gain +1 FT at end of week (FTs stay at current level)
                        next_ft = fts_available  # Stay at current level
                        ft_str = f"FT: {fts_available} ({chip_this_week} active, next: {next_ft})"
                        current_fts = next_ft
                    elif num_transfers == 0:
                        # No transfers made = banking FT
                        next_ft = min(5, fts_available + 1)
                        ft_str = f"FT: {fts_available} (banking -> {next_ft})"
                        current_fts = next_ft
                    else:
                        # Regular transfers (no chip active)
                        actual_ft_used = min(num_transfers, fts_available)
                        hits_taken = max(0, num_transfers - fts_available)
                        
                        # Next week: current - used + 1, capped at 5, min 1
                        next_ft = min(5, max(1, fts_available - actual_ft_used + 1))
                        
                        if hits_taken > 0:
                            ft_str = f"FT: {fts_available} (using {actual_ft_used} FT + {hits_taken} hit(s), next: {next_ft})"
                        else:
                            ft_str = f"FT: {fts_available} (using {actual_ft_used}, next: {next_ft})"
                        current_fts = next_ft
                else:
                    ft_str = ""
                
                # Chip indicator
                chip_str = f"[{chip_this_week.upper()}]" if chip_this_week else ""
                
                # Safe encode captain name for Windows console
                def safe_encode(name):
                    if isinstance(name, str):
                        return name.encode('ascii', 'replace').decode('ascii').replace('?', '')
                    return str(name)
                
                captain_safe = safe_encode(captain_name)
                vice_captain_safe = safe_encode(vice_captain_name)
                
                # Get captain and vice-captain predicted points for display
                captain_id = week.get('captain_id')
                vice_id = week.get('vice_captain_id')
                cap_proj = player_projections.get(captain_id, {})
                vice_proj = player_projections.get(vice_id, {})
                cap_pts_gw = cap_proj.get('gameweek_predictions', {}).get(gw, 0) if cap_proj else 0
                vice_pts_gw = vice_proj.get('gameweek_predictions', {}).get(gw, 0) if vice_proj else 0
                
                # Format captain/vice string with points
                if cap_pts_gw > 0 and vice_pts_gw > 0:
                    captain_str = f"(C) {captain_safe} ({cap_pts_gw:.1f}) (VC) {vice_captain_safe} ({vice_pts_gw:.1f})"
                else:
                    captain_str = f"(C) {captain_safe} (VC) {vice_captain_safe}"
                
                # Build full line
                parts = [
                    f"GW{gw}:",
                    transfer_str,
                    f"| {ft_str}" if ft_str else "",
                    f"| Exp: {expected_pts:.1f} pts",
                    f"| {captain_str}",
                    f"| {chip_str}" if chip_str else "",
                    f"| Bank: ${budget:.1f}m" if offset == 0 or num_transfers > 0 else ""
                ]
                line = " ".join([p for p in parts if p])
                print(f"  {line}")
                
                # Show ALL transfer details
                transfers_list = week.get('transfers', [])
                if transfers_list and num_transfers > 0:
                    for i, transfer in enumerate(transfers_list):
                        out_name = safe_encode(transfer.get('out_name', 'Unknown'))
                        in_name = safe_encode(transfer.get('in_name', 'Unknown'))
                        gain = transfer.get('gain', 0)
                        cost_change = transfer.get('cost_change', 0)
                        
                        # Format the transfer line
                        gain_str = f"+{gain:.1f}" if gain >= 0 else f"{gain:.1f}"
                        cost_str = f" (${cost_change:+.1f}m)" if abs(cost_change) > 0.05 else ""
                        print(f"      OUT: {out_name} | IN: {in_name} ({gain_str} pts{cost_str})")
                
                # Show Free Hit temporary swaps (if any)
                fh_players = week.get('fh_players_in', [])
                if fh_players and chip_this_week == 'Free Hit':
                    # Get permanent squad and starting XI to identify who's benched
                    permanent_squad = week.get('team', [])
                    starting_xi = week.get('starting_xi', [])
                    
                    # Find players from permanent squad who are NOT in the starting XI (benched for FH)
                    benched_for_fh = [pid for pid in permanent_squad if pid not in starting_xi]
                    
                    # Pair benched players with FH temporary players
                    print(f"      Free Hit Temporary Swaps ({len(fh_players)} changes):")
                    for idx, fh_player in enumerate(fh_players[:8]):
                        fh_name = safe_encode(fh_player.get('name', 'Unknown'))
                        fh_pts = fh_player.get('predicted_points', 0)
                        
                        # Try to find the benched player (if available)
                        if idx < len(benched_for_fh) and player_projections:
                            benched_pid = benched_for_fh[idx]
                            benched_player = player_projections.get(benched_pid, {})
                            benched_name = benched_player.get('web_name', 'Unknown')
                            benched_safe = safe_encode(benched_name)
                            # Get benched player's predicted points for this GW (they're benched so 0 pts)
                            benched_gw_pts = benched_player.get('gameweek_predictions', {}).get(gw, 0)
                            
                            gain = fh_pts - benched_gw_pts  # Compare FH player vs benched player
                            gain_str = f"+{gain:.1f}" if gain >= 0 else f"{gain:.1f}"
                            print(f"        OUT: {benched_safe} | IN: {fh_name} ({gain_str} pts)")
                        else:
                            # No specific pairing available, just show the FH player
                            print(f"        IN: {fh_name} (+{fh_pts:.1f} pts)")
                    
                    if len(fh_players) > 8:
                        remaining = len(fh_players) - 8
                        print(f"        ... and {remaining} more temporary swaps")

        # Next Steps - Use multi-period plan if available, otherwise fallback
        if this_week_plan:
            # Build action steps from this week's plan (use saved variables, not loop variables)
            actions = []
            
            # Transfer action
            if this_week_num_transfers > 0 and this_week_transfers_list:
                total_gain = sum(t.get('gain', 0) for t in this_week_transfers_list)
                transfer_desc = f"Make {this_week_num_transfers} transfer(s) for {total_gain:.1f} net points gain"
                if len(this_week_transfers_list) <= 3:
                    names_out = ", ".join([safe_encode(t.get('out_name', '?')) for t in this_week_transfers_list])
                    names_in = ", ".join([safe_encode(t.get('in_name', '?')) for t in this_week_transfers_list])
                    transfer_desc += f" (OUT: {names_out}, IN: {names_in})"
                actions.append({'priority': 1, 'description': transfer_desc})
            
            # Captain action (always show)
            if this_week_captain_name and this_week_captain_name != 'Unknown':
                captain_safe = safe_encode(this_week_captain_name)
                actions.append({'priority': 2, 'description': f"Captain {captain_safe} ({this_week_captain_pts:.1f} predicted points)"})
            
            # Chip action
            if this_week_chip:
                chip_display = this_week_chip.replace('_', ' ').title()
                actions.append({'priority': 3, 'description': f"Activate {chip_display} chip"})
            
            # If holding with no chip, make it clear
            if this_week_num_transfers == 0 and not this_week_chip:
                actions.insert(0, {'priority': 0, 'description': "Hold team - no transfers needed this week"})
            
            if actions:
                print("\nNext Steps")
                for action in sorted(actions, key=lambda a: a.get('priority', 99)):
                    print(f"- {action.get('description','')}")
        else:
            # Fallback: Use standalone next_steps from report
            actions = report.get('next_steps', [])
            if actions:
                print("\nNext Steps")
                for action in sorted(actions, key=lambda a: a.get('priority', 99))[:3]:
                    print(f"- {action.get('description','')}")


def main():
    """Main entry point for the FPL Bot CLI"""
    parser = argparse.ArgumentParser(
        description='FPL Bot - Fantasy Premier League prediction and optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python -m fpl_bot.main 789800
        """
    )
    parser.add_argument('manager_id', type=int, help='FPL Manager ID')
    
    args = parser.parse_args()
    
    # Start web UI server in background thread
    port = 5000
    print(f"\nStarting web UI on http://127.0.0.1:{port}")
    print("   (Terminal will show debug info only)\n")
    ui_thread = threading.Thread(
        target=run_ui, 
        kwargs={'host': '127.0.0.1', 'port': port, 'debug': False},
        daemon=True
    )
    ui_thread.start()
    time.sleep(1)  # Give server time to start
    
    # Try to open browser
    try:
        webbrowser.open(f'http://127.0.0.1:{port}')
    except:
        pass  # Silently fail if browser can't open
    
    # Initialize bot
    bot = FPLBot()
    
    try:
        report = bot.run_analysis(args.manager_id)
        
        # Feed report to UI
        set_report(report)
        bot.print_terminal_status(report)
        
        # Keep program alive so UI stays accessible
        print("\n[Press Ctrl+C to exit]")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nShutting down...")
            
    except Exception as e:
        print(f"Error running FPL Bot: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
