"""
Main FPL Bot interface

Provides a clean interface to run the FPL bot with all functionality:
- Data collection and analysis
- Manager team analysis
- Chip management
"""

import argparse
import json
from typing import Dict, Optional
from datetime import datetime
import pandas as pd
import threading
import webbrowser
import time

from .core.data_collector import DataCollector
from .core.manager_analyzer import ManagerAnalyzer
from .core.chip_manager import ChipManager
from .core.team_optimizer import TeamOptimizer
from .core.wildcard_optimizer import WildcardOptimizer
from .ui.app import run_ui, set_report


class FPLBot:
    """Main FPL Bot class that orchestrates all functionality"""
    
    def __init__(self):
        self.data_collector = DataCollector()
        self.manager_analyzer = ManagerAnalyzer(self.data_collector)
        self.chip_manager = ChipManager(self.data_collector)
        self.team_optimizer = TeamOptimizer(self.data_collector)
        self.wildcard_optimizer = WildcardOptimizer(self.team_optimizer, self.data_collector)
        
        # Cache for data
        self.current_season_data = None
        self.players_df = None
        self.fixtures_df = None
        
    def run_analysis(self, manager_id: int, optimize: bool = True, verbose: bool = False, risk_aversion: float = 0.5, min_chance_of_playing: int = 75, analyze_wildcard: bool = False, budget_correction: float = 0.0, no_ft_gain_last_gw: bool = False, no_hits: bool = False) -> Dict:
        """Run FPL analysis and optimization for a manager
        
        Args:
            manager_id: FPL manager ID
            optimize: Whether to run team optimization (default True)
            verbose: Whether to show detailed terminal output (default False)
            risk_aversion: Risk aversion (0=aggressive, 1=conservative, default 0.5)
            min_chance_of_playing: Minimum % chance of playing to consider player (default 75)
            analyze_wildcard: Whether to analyze optimal wildcard timing (default False, slower)
            budget_correction: Manual budget adjustment in millions (e.g., -2.5 to reduce by £2.5m)
            no_ft_gain_last_gw: Flag indicating wildcard/free hit was played last week (no FT gained)
            no_hits: Prevent optimizer from taking point hits (only use free transfers)
            
        Returns:
            Dict with analysis and optimization results
        """
        if verbose:
            print(f"Starting FPL Bot analysis for manager {manager_id}")
            print("=" * 60)
        else:
            print(f"FPL Bot starting for manager {manager_id}...")
        
        # Step 1: Collect data
        if verbose:
            print("\nStep 1: Collecting data...")
        else:
            print("Collecting data...", end=" ", flush=True)
        self._collect_all_data()
        if not verbose:
            print("[OK]")
        
        # Step 2: Analyze manager
        if verbose:
            print("\nStep 2: Analyzing manager...")
        else:
            print("Analyzing manager...", end=" ", flush=True)
        manager_analysis = self.manager_analyzer.analyze_manager(manager_id, no_ft_gain_last_gw=no_ft_gain_last_gw)
        if not verbose:
            print("[OK]")
        
        # Step 3: Optimize team (if requested)
        optimization_result = None
        if optimize:
            if verbose:
                print("\nStep 3: Optimizing team...")
            else:
                print("Running MIP optimizer...", end=" ", flush=True)
            optimization_result = self._optimize_team(manager_analysis, verbose=verbose, risk_aversion=risk_aversion, min_chance_of_playing=min_chance_of_playing, budget_correction=budget_correction, no_hits=no_hits)
            if not verbose:
                print("[OK]")
        
        # Step 4: Analyze wildcard timing (if requested)
        wildcard_analysis = None
        if analyze_wildcard and optimization_result:
            if verbose:
                print("\nStep 4: Analyzing wildcard timing...")
            else:
                print("Analyzing wildcard timing...", end=" ", flush=True)
            wildcard_analysis = self._analyze_wildcard(manager_analysis, verbose=verbose, risk_aversion=risk_aversion, min_chance_of_playing=min_chance_of_playing, budget_correction=budget_correction)
            if not verbose:
                print("[OK]")
            
            # If wildcard is recommended for the current or next GW, use the wildcard plan instead
            if wildcard_analysis and wildcard_analysis.get('recommended_gw'):
                current_gw = optimization_result['weekly_plans'][0]['gameweek']
                recommended_gw = wildcard_analysis['recommended_gw']
                
                # If wildcard recommended within next 2 GWs, use that plan
                if recommended_gw <= current_gw + 1:
                    if verbose:
                        print(f"\n[INFO] Wildcard recommended for GW{recommended_gw}, using wildcard plan for display")
                    # Get the wildcard plan from the analysis
                    wildcard_plan = wildcard_analysis.get('wildcard_plan')
                    if wildcard_plan:
                        optimization_result = wildcard_plan
        
        # Create report
        report = self._create_report(manager_analysis, optimization_result, wildcard_analysis)
        
        if not verbose:
            print("Analysis complete!")
        
        return report
    
    def _collect_all_data(self):
        """Collect all necessary FPL data"""
        try:
            self.current_season_data = self.data_collector.get_current_season_data()
            self.players_df = self.data_collector.get_player_data()
            self.fixtures_df = self.data_collector.get_fixtures()
            print(f"[OK] Data collected: {len(self.players_df)} players, {len(self.fixtures_df)} fixtures")
        except Exception as e:
            print(f"[ERROR] Error collecting data: {e}")
            raise
    
    def _optimize_team(self, manager_analysis: Dict, verbose: bool = False, risk_aversion: float = 0.5, min_chance_of_playing: int = 75, budget_correction: float = 0.0, no_hits: bool = False) -> Optional[Dict]:
        """Run team optimization
        
        Args:
            manager_analysis: Manager analysis results
            verbose: Whether to show detailed output
            risk_aversion: Risk aversion parameter (0=aggressive, 1=conservative)
            min_chance_of_playing: Minimum % chance of playing to consider player
            budget_correction: Manual budget adjustment in millions
            no_hits: Prevent optimizer from taking point hits
            
        Returns:
            Optimization results or None if failed
        """
        try:
            # Get current team IDs
            current_team = []
            team_data = manager_analysis.get('current_team', {})
            picks = team_data.get('picks', [])
            for pick in picks:
                current_team.append(pick['element'])
            
            if not current_team:
                if verbose:
                    print("✗ Could not extract current team")
                return None
            
            # Get budget and free transfers
            manager_info = manager_analysis.get('manager_info', {})
            budget = manager_info.get('bank', 0.0)
            
            saved_transfers = manager_analysis.get('saved_transfers', {})
            free_transfers = saved_transfers.get('free_transfers', 1)
            
            # Always show free transfers (important for user to verify)
            if not verbose:
                print(f"Free transfers: {free_transfers}")
            
            if verbose:
                print(f"Current team: {len(current_team)} players")
                print(f"Budget: £{budget:.1f}m")
                print(f"Free transfers: {free_transfers}")
                print(f"  [DEBUG] saved_transfers dict: {saved_transfers}")
            
            # Get team value (selling price if you sold all players)
            team_value = manager_info.get('team_value', 0.0)
            
            # Apply budget correction if specified
            if budget_correction != 0.0:
                team_value += budget_correction
                if verbose:
                    print(f"\n[BUDGET CORRECTION] Adjusting team value by £{budget_correction:+.1f}m")
                    print(f"  Adjusted team value: £{team_value:.1f}m")
            
            # Run optimization
            result = self.team_optimizer.optimize_team(
                current_team=current_team,
                current_budget=budget,
                free_transfers=free_transfers,
                horizon_gws=None,  # Optimize until GW19
                verbose=verbose,
                risk_aversion=risk_aversion,
                min_chance_of_playing=min_chance_of_playing,
                current_team_value=team_value,
                no_hits=no_hits
            )
            
            return result
            
        except Exception as e:
            if verbose:
                print(f"[ERROR] Error during optimization: {e}")
                import traceback
                traceback.print_exc()
            return None
    
    def _analyze_wildcard(self, manager_analysis: Dict, verbose: bool = False, risk_aversion: float = 0.5, min_chance_of_playing: int = 75, budget_correction: float = 0.0) -> Optional[Dict]:
        """Analyze optimal wildcard timing
        
        Args:
            manager_analysis: Manager analysis results
            verbose: Whether to show detailed output
            risk_aversion: Risk aversion parameter
            min_chance_of_playing: Minimum % chance of playing
            budget_correction: Manual budget adjustment in millions
            
        Returns:
            Wildcard timing analysis or None if failed
        """
        try:
            # Get current team IDs
            current_team = []
            team_data = manager_analysis.get('current_team', {})
            picks = team_data.get('picks', [])
            for pick in picks:
                current_team.append(pick['element'])
            
            if not current_team:
                if verbose:
                    print("✗ Could not extract current team")
                return None
            
            # Get budget and free transfers
            manager_info = manager_analysis.get('manager_info', {})
            budget = manager_info.get('bank', 0.0)
            saved_transfers = manager_analysis.get('saved_transfers', {})
            free_transfers = saved_transfers.get('free_transfers', 1)
            
            # Get current gameweek
            season_data = self.data_collector.get_current_season_data()
            events = season_data.get('events', [])
            current_gw = None
            for event in events:
                if not event.get('finished', False):
                    current_gw = event.get('id')
                    break
            
            if not current_gw:
                if verbose:
                    print("[ERROR] Could not determine current gameweek")
                return None
            
            if verbose:
                print(f"Analyzing wildcard timing from GW{current_gw} to GW19...")
                print(f"Budget: £{budget:.1f}m")
                print(f"Free transfers: {free_transfers}")
            
            # Get team value (selling price)
            team_value = manager_info.get('team_value', 0.0)
            
            # Apply budget correction if specified
            if budget_correction != 0.0:
                team_value += budget_correction
                if verbose:
                    print(f"\n[BUDGET CORRECTION] Adjusting team value by £{budget_correction:+.1f}m")
                    print(f"  Adjusted team value: £{team_value:.1f}m")
            
            # Run wildcard analysis
            result = self.wildcard_optimizer.find_optimal_wildcard_timing(
                current_team=current_team,
                current_budget=budget,
                free_transfers=free_transfers,
                current_gw=current_gw,
                end_gw=19,
                verbose=verbose,
                current_team_value=team_value
            )
            
            return result
            
        except Exception as e:
            if verbose:
                print(f"[ERROR] Error during wildcard analysis: {e}")
                import traceback
                traceback.print_exc()
            return None
    
    def _create_report(self, manager_analysis: Dict, optimization_result: Optional[Dict] = None, wildcard_analysis: Optional[Dict] = None) -> Dict:
        """Create analysis report
        
        Args:
            manager_analysis: Manager analysis results
            optimization_result: Team optimization results (optional)
            wildcard_analysis: Wildcard timing analysis (optional)
            
        Returns:
            Complete report dict
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'manager_analysis': manager_analysis,
            'optimization': optimization_result,
            'wildcard_analysis': wildcard_analysis,
            'data': {
                'players': len(self.players_df) if self.players_df is not None else 0,
                'fixtures': len(self.fixtures_df) if self.fixtures_df is not None else 0
            }
        }
        return report
    
    def print_summary(self, report: Dict):
        """Print a summary of the analysis"""
        print("\n" + "=" * 60)
        print("FPL BOT ANALYSIS SUMMARY")
        print("=" * 60)
        
        manager_info = report.get('manager_analysis', {}).get('manager_info', {})
        print(f"\nManager: {manager_info.get('name', 'Unknown')}")
        print(f"Team: {manager_info.get('team_name', 'Unknown')}")
        print(f"Overall Rank: {manager_info.get('overall_rank', 'N/A'):,}")
        print(f"Total Points: {manager_info.get('total_points', 0):,}")
        print(f"Bank: £{manager_info.get('bank', 0.0):.1f}m")
        
        # Print optimization results
        optimization = report.get('optimization')
        if optimization and optimization.get('status') == 'Optimal':
            print("\n" + "=" * 60)
            print("MULTI-GAMEWEEK OPTIMIZATION PLAN")
            print("=" * 60)
            
            summary = optimization.get('summary', {})
            print(f"\nPlanning Horizon: {summary.get('planning_horizon', 'N/A')}")
            print(f"Expected Total Points: {optimization.get('objective_value', 0):.1f}")
            
            # Print week-by-week plan
            weekly_plans = optimization.get('weekly_plans', [])
            
            for plan in weekly_plans:
                gw = plan['gameweek']
                transfers = plan['transfers']
                captain = plan['captain']
                
                print("\n" + "-" * 60)
                print(f"GAMEWEEK {gw}")
                print("-" * 60)
                
                # Transfers
                if transfers['count'] > 0:
                    print(f"\nTransfers: {transfers['count']}")
                    if plan['hits_taken'] > 0:
                        print(f"Hits Taken: {plan['hits_taken']} (-{plan['points_cost']} pts)")
                    
                    if transfers['in']:
                        print(f"\n  {'IN:':<4} {'Player':<25} {'Pos':<5} {'Team':<5} {'Cost':<6}")
                        for t in transfers['in']:
                            print(f"  {'→':<4} {t['player_name']:<25} {t['position']:<5} {t['team']:<5} £{t['cost']:<5.1f}")
                    
                    if transfers['out']:
                        print(f"\n  {'OUT:':<4} {'Player':<25} {'Pos':<5} {'Team':<5} {'Cost':<6}")
                        for t in transfers['out']:
                            print(f"  {'←':<4} {t['player_name']:<25} {t['position']:<5} {t['team']:<5} £{t['cost']:<5.1f}")
                else:
                    print("\nNo transfers")
                
                # Captain
                captain_name = captain.get('player_name')
                if captain_name:
                    print(f"\nCaptain: {captain_name}")
                
                # Squad breakdown
                squad = plan['squad']
                print(f"\nStarting 11: {len(squad['starting_11'])} players")
                print(f"Bench: {len(squad['bench'])} players")
        
        print("\n" + "=" * 60)


def main():
    """Main entry point for FPL Bot CLI"""
    parser = argparse.ArgumentParser(description='FPL Bot - Fantasy Premier League Analysis Tool')
    parser.add_argument('manager_id', type=int, help='Your FPL Manager ID')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    parser.add_argument('--output', '-o', type=str, help='Save report to file')
    parser.add_argument('--no-ui', action='store_true', help='Skip launching the web UI')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed terminal output')
    parser.add_argument('--port', type=int, default=5000, help='Port for web UI (default: 5000)')
    parser.add_argument('--risk', type=float, default=0.5, help='Risk aversion: 0=aggressive, 1=conservative (default: 0.5)')
    parser.add_argument('--min-playing-chance', type=int, default=75, help='Minimum %% chance of playing to consider a player (default: 75)')
    parser.add_argument('--analyze-wildcard', action='store_true', help='Analyze optimal wildcard timing (slower, ~5-10 min)')
    parser.add_argument('--budget-correction', type=float, default=0.0, help='Budget correction in millions (e.g., -2.5 to reduce budget by £2.5m)')
    parser.add_argument('--no-ft-gain-last-gw', action='store_true', help='Indicate that a wildcard/free hit was played last week, so no FT was gained')
    parser.add_argument('--no-hits', action='store_true', help='Prevent optimizer from recommending point hits (only use free transfers)')
    
    args = parser.parse_args()
    
    # Start web UI in background thread (unless disabled)
    if not args.no_ui and not args.json:
        print(f"\n🌐 Starting web UI on http://localhost:{args.port}")
        ui_thread = threading.Thread(target=run_ui, kwargs={'port': args.port}, daemon=True)
        ui_thread.start()
        
        # Give the server a moment to start
        time.sleep(1)
        
        # Open browser
        webbrowser.open(f'http://localhost:{args.port}')
        print("[OK] Web UI launched in your browser\n")
    
    # Initialize and run bot
    bot = FPLBot()
    report = bot.run_analysis(
        args.manager_id, 
        verbose=args.verbose, 
        risk_aversion=args.risk, 
        min_chance_of_playing=args.min_playing_chance,
        analyze_wildcard=args.analyze_wildcard,
        budget_correction=args.budget_correction,
        no_ft_gain_last_gw=args.no_ft_gain_last_gw,
        no_hits=args.no_hits
    )
    
    # Update UI with report
    if not args.no_ui:
        set_report(report)
        print(f"\n💫 Results available at: http://localhost:{args.port}")
    
    # Output results
    if args.json:
        print(json.dumps(report, indent=2))
    elif args.verbose:
        bot.print_summary(report)
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n[OK] Report saved to {args.output}")
    
    # Keep the program running if UI is active
    if not args.no_ui and not args.json:
        print("\n📌 Press Ctrl+C to exit")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n👋 Shutting down FPL Bot...")


if __name__ == '__main__':
    main()
