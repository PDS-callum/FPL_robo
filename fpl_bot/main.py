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

from .core.data_collector import DataCollector
from .core.manager_analyzer import ManagerAnalyzer
from .core.predictor import Predictor
from .core.transfer_optimizer import TransferOptimizer
from .core.chip_manager import ChipManager
from .core.multi_period_planner import MultiPeriodPlanner


class FPLBot:
    """Main FPL Bot class that orchestrates all functionality"""
    
    def __init__(self, auto_execute: bool = False):
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
        
        # Auto-execution settings
        self.auto_execute = auto_execute
        self.authenticated = False
        
    def run_analysis(self, manager_id: int, save_results: bool = True, fpl_email: str = None, fpl_password: str = None) -> Dict:
        """Run complete FPL analysis for a manager"""
        print(f"Starting FPL Bot analysis for manager {manager_id}")
        print("=" * 60)
        
        # If auto-execute is enabled, authenticate first
        if self.auto_execute:
            if not fpl_email or not fpl_password:
                print("\n[!] Auto-execute mode enabled but no credentials provided!")
                print("    Please provide --email and --password for authentication.")
                self.auto_execute = False
            else:
                print("\n[AUTH] Authenticating with FPL...")
                if self.data_collector.authenticate(fpl_email, fpl_password):
                    self.authenticated = True
                else:
                    print("[!] Authentication failed. Auto-execute disabled.")
                    self.auto_execute = False
        
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
        
        # Auto-execute transfers if enabled and no transfers made this week
        if self.auto_execute and self.authenticated and transfers_made_this_gw == 0:
            self._execute_transfers_if_beneficial(transfer_analysis, manager_analysis)
        
        # Step 5: Multi-Period Planning & Chip Optimization
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
        
        # Save results if requested
        if save_results:
            self._save_results(report, manager_id)
        
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
                for event in self.current_season_data['events']:
                    if event.get('is_current', False):
                        return event.get('id')
            return None
        except:
            return None
    
    def _execute_transfers_if_beneficial(self, transfer_analysis: Dict, manager_analysis: Dict):
        """Execute transfers if they are beneficial
        
        Safety checks:
        1. Only execute if net points gained > 0
        2. Only execute if using free transfers (no hits unless confidence is very high)
        3. Provide summary before executing
        """
        optimized_team = transfer_analysis.get('optimized_team', {})
        transfers_made = optimized_team.get('transfers_made', 0)
        
        if transfers_made == 0:
            print("\n✓ No beneficial transfers found - keeping current team")
            return
        
        net_gain = optimized_team.get('net_points_gained', 0)
        transfer_cost = optimized_team.get('transfer_cost', 0)
        
        # Safety check: Don't take hits unless extremely beneficial
        if transfer_cost > 0 and net_gain < 10:
            print(f"\n[!] Transfer would cost {transfer_cost} points for {net_gain:.1f} net gain")
            print("    Skipping execution - not beneficial enough for a hit")
            return
        
        # Get transfer details
        players_out = optimized_team.get('players_out', [])
        players_in = optimized_team.get('players_in', [])
        
        # Show transfer summary
        print("\n" + "="*60)
        print("AUTONOMOUS TRANSFER EXECUTION")
        print("="*60)
        print(f"Transfers to make: {transfers_made}")
        print(f"Expected net gain: {net_gain:.1f} points")
        if transfer_cost > 0:
            print(f"Transfer cost: {transfer_cost} points")
        print("\nTransfers:")
        for i in range(len(players_out)):
            out = players_out[i]
            in_player = players_in[i]
            print(f"  OUT: {out['web_name']} ({out['position_name']}) £{out['cost']:.1f}m")
            print(f"  IN:  {in_player['web_name']} ({in_player['position_name']}) £{in_player['cost']:.1f}m")
            print()
        
        # Extract player IDs for API call
        player_ids_out = [p['player_id'] for p in players_out]
        player_ids_in = [p['player_id'] for p in players_in]
        
        # Execute the transfers
        print("Executing transfers...")
        success = self.data_collector.execute_transfers(player_ids_in, player_ids_out)
        
        if success:
            print("\n[SUCCESS] TRANSFERS COMPLETED SUCCESSFULLY!")
            print("="*60)
        else:
            print("\n[FAILED] TRANSFER EXECUTION FAILED")
            print("Please check the error message above and make transfers manually.")
            print("="*60)
    
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
        transfer_decision = recommendations.get('transfer_decision', {})
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
                    return name.encode('ascii', 'ignore').decode('ascii') if isinstance(name, str) else str(name)
                
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
        captain_decision = recommendations.get('captain_decision', {})
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
        chip_decision = recommendations.get('chip_decision', {})
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
    
    def _save_results(self, report: Dict, manager_id: int):
        """Save analysis results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fpl_analysis_{manager_id}_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Results saved to {filename}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
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
            if not chip_info.get('best_gw'):
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
    
    def print_summary(self, report: Dict):
        """Print a summary of the analysis"""
        print("\n" + "=" * 60)
        print("FPL BOT ANALYSIS SUMMARY")
        print("=" * 60)
        
        # Manager info
        manager_info = report.get('manager_info', {})
        if manager_info:
            print(f"\nManager: {manager_info.get('manager_name', 'Unknown')}")
            print(f"Team: {manager_info.get('team_name', 'Unknown')}")
            print(f"Overall Rank: {manager_info.get('overall_rank', 'Unknown')}")
            print(f"Total Points: {manager_info.get('total_points', 'Unknown')}")
            print(f"Team Value: £{manager_info.get('team_value', 0):.1f}m")
            print(f"Bank: £{manager_info.get('bank', 0):.1f}m")
            
            # Show saved transfers info
            saved_transfers = manager_info.get('saved_transfers', {})
            if saved_transfers:
                free_transfers = saved_transfers.get('free_transfers', 1)
                transfers_this_gw = saved_transfers.get('transfers_this_gw', 0)
                print(f"Free Transfers Available: {free_transfers}")
                if transfers_this_gw > 0:
                    print(f"Transfers Made This Week: {transfers_this_gw}")
        else:
            print("\nManager info not available")
        
        # Transfer recommendations and optimized team
        transfer_analysis = report.get('transfer_analysis', {})
        optimized_team = transfer_analysis.get('optimized_team', {}) if transfer_analysis else {}
        
        if optimized_team and optimized_team.get('transfers_made', 0) > 0:
            print(f"\nTRANSFER DECISION:")
            print(f"{optimized_team['message']}")
            if optimized_team.get('players_out'):
                players_out_names = [p['web_name'].encode('ascii', 'ignore').decode('ascii') for p in optimized_team['players_out']]
                print(f"Players out: {', '.join(players_out_names)}")
            if optimized_team.get('players_in'):
                players_in_names = [p['web_name'].encode('ascii', 'ignore').decode('ascii') for p in optimized_team['players_in']]
                print(f"Players in: {', '.join(players_in_names)}")
            
            # Show transfer cost if any
            transfer_cost = optimized_team.get('transfer_cost', 0)
            if transfer_cost > 0:
                print(f"Transfer Cost: {transfer_cost} points")
            
            print(f"\nNEW TEAM:")
            print(f"Formation: {optimized_team.get('formation', 'Unknown')}")
            print(f"Total Cost: £{optimized_team.get('total_cost', 0):.1f}m")
            print(f"Predicted Points: {optimized_team.get('total_predicted_points', 0):.1f}")
            
            # Show team players
            team_players = optimized_team.get('team_players', [])
            if team_players:
                print(f"\nTeam Players:")
                for player in team_players:
                    # Handle Unicode characters in player names
                    player_name = player['web_name'].encode('ascii', 'ignore').decode('ascii')
                    print(f"  {player_name} ({player['position_name']}) - £{player['cost']:.1f}m - {player['predicted_points']:.1f} pts")
        else:
            print(f"\nTRANSFER DECISION: No beneficial transfers found - keeping current team")
            if optimized_team:
                print(f"Current Team Predicted Points: {optimized_team.get('total_predicted_points', 0):.1f}")
                print(f"Formation: {optimized_team.get('formation', 'Unknown')}")
        
        # Recommendations with confidence
        recommendations = report.get('recommendations', {})
        
        # Chip decision
        chip_decision = recommendations.get('chip_decision', {}) if recommendations else {}
        if chip_decision:
            action = chip_decision.get('action', 'NO_CHIP')
            confidence = chip_decision.get('confidence', 0)
            print(f"\nCHIP DECISION: {action} (Confidence: {confidence:.0f}%)")
            if action != 'NO_CHIP':
                print(f"Reason: {chip_decision.get('reason', '')}")
                print(f"Expected Benefit: {chip_decision.get('expected_benefit', 0):.1f} points")
                timing = chip_decision.get('timing', 'NOW')
                if timing != 'NOW':
                    print(f"Timing: {timing}")
            elif 'save_for' in chip_decision:
                # Chip should be saved for better week
                print(f"Reason: {chip_decision.get('reason', '')}")
                print(f"Save For: {chip_decision['save_for']}")
                if 'save_for_player' in chip_decision:
                    print(f"Target: {chip_decision['save_for_player']} vs {chip_decision.get('save_for_opponent', 'TBD')}")
                print(f"Future Expected: {chip_decision.get('future_predicted', 0):.1f} points")
            else:
                print(f"Reason: {chip_decision.get('reason', 'No chip benefit identified')}")
        else:
            print(f"\nCHIP DECISION: NO_CHIP")
        
        # Captain decision
        captain_decision = recommendations.get('captain_decision', {}) if recommendations else {}
        if captain_decision:
            player = captain_decision.get('player', 'Unknown')
            points = captain_decision.get('predicted_points', 0)
            confidence = captain_decision.get('confidence', 0)
            print(f"\nCAPTAIN DECISION: {player} (Confidence: {confidence:.0f}%)")
            print(f"Predicted Points: {points:.1f}")
            
            alternatives = captain_decision.get('alternatives', [])
            if len(alternatives) > 1:
                print(f"Alternatives:")
                for alt in alternatives[1:3]:  # Show next 2
                    print(f"  - {alt['web_name']}: {alt['predicted_points']:.1f} points")
        else:
            print(f"\nCAPTAIN DECISION: Not available")
        
        # Action plan
        actions = report.get('next_steps', [])
        if actions:
            print(f"\nACTION PLAN:")
            for action in actions:
                priority = action.get('priority', 0)
                desc = action.get('description', '')
                conf = action.get('confidence', 0)
                print(f"{priority}. {desc} (Confidence: {conf:.0f}%)")
        
        # Multi-Period Strategic Plan (new default)
        multi_period_plan = report.get('multi_period_plan')
        if multi_period_plan:
            self._print_multi_period_plan(multi_period_plan)


def main():
    """Main entry point for the FPL Bot CLI"""
    parser = argparse.ArgumentParser(
        description='FPL Bot - Fantasy Premier League prediction and optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard analysis with 7-GW strategic planning
  python -m fpl_bot.main 789800 --summary-only
  
  # Autonomous mode - bot will execute transfers automatically
  python -m fpl_bot.main 789800 --auto-execute --email your@email.com --password yourpass
  
  # Set credentials via environment variables (recommended for security)
  set FPL_EMAIL=your@email.com
  set FPL_PASSWORD=yourpassword
  python -m fpl_bot.main 789800 --auto-execute
        """
    )
    parser.add_argument('manager_id', type=int, help='FPL Manager ID')
    parser.add_argument('--no-save', action='store_true', help='Do not save results to file')
    parser.add_argument('--summary-only', action='store_true', help='Show only summary output')
    parser.add_argument('--auto-execute', action='store_true', 
                       help='Automatically execute recommended transfers (requires authentication)')
    parser.add_argument('--email', type=str, 
                       help='FPL account email (or set FPL_EMAIL env var)')
    parser.add_argument('--password', type=str, 
                       help='FPL account password (or set FPL_PASSWORD env var)')
    
    args = parser.parse_args()
    
    # Get credentials from args or environment variables
    import os
    fpl_email = args.email or os.getenv('FPL_EMAIL')
    fpl_password = args.password or os.getenv('FPL_PASSWORD')
    
    # Initialize bot with auto-execute flag
    bot = FPLBot(auto_execute=args.auto_execute)
    
    try:
        report = bot.run_analysis(
            args.manager_id, 
            save_results=not args.no_save,
            fpl_email=fpl_email,
            fpl_password=fpl_password
        )
        
        if args.summary_only:
            bot.print_summary(report)
        else:
            # Print full report
            print(json.dumps(report, indent=2, default=str))
            
    except Exception as e:
        print(f"Error running FPL Bot: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
