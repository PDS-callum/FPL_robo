"""
Weekly Summary Dashboard for FPL Bot.
Provides rich console output with team changes, performance metrics, and recommendations.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from .performance_tracker import FPLPerformanceTracker


class FPLWeeklyDashboard:
    """
    Generates rich weekly summary dashboards for FPL Bot.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.performance_tracker = FPLPerformanceTracker(data_dir)
    
    def generate_weekly_summary(self, gameweek: int, team_prediction: Dict[str, Any], 
                               previous_team: Optional[Dict[str, Any]] = None,
                               transfers_made: List[Dict[str, Any]] = None) -> str:
        """
        Generate a comprehensive weekly summary dashboard.
        
        Parameters:
        -----------
        gameweek : int
            Current gameweek number
        team_prediction : dict
            Current week's team prediction
        previous_team : dict, optional
            Previous week's team for comparison
        transfers_made : list, optional
            List of transfers made this week
            
        Returns:
        --------
        str
            Formatted dashboard string
        """
        dashboard = []
        
        # Header
        dashboard.append(self._create_header(gameweek))
        
        # Team Overview
        dashboard.append(self._create_team_overview(team_prediction))
        
        # Transfer Analysis
        if transfers_made:
            dashboard.append(self._create_transfer_analysis(transfers_made))
        
        # Team Changes
        if previous_team:
            dashboard.append(self._create_team_changes(team_prediction, previous_team, transfers_made))
        
        # Performance Metrics
        dashboard.append(self._create_performance_metrics(gameweek))
        
        # Recommendations
        dashboard.append(self._create_recommendations(team_prediction, transfers_made))
        
        # Footer
        dashboard.append(self._create_footer())
        
        return "\n".join(dashboard)
    
    def _create_header(self, gameweek: int) -> str:
        """Create dashboard header."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           🏆 FPL BOT WEEKLY DASHBOARD 🏆                    ║
║                                                                              ║
║  📅 Gameweek {gameweek:2d} Summary                    🕐 Generated: {timestamp}  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    
    def _create_team_overview(self, team_prediction: Dict[str, Any]) -> str:
        """Create team overview section."""
        total_points = team_prediction.get('total_predicted_points', 0)
        total_cost = team_prediction.get('total_cost', 0)
        formation = team_prediction.get('formation', 'Unknown')
        captain = team_prediction.get('captain', {}).get('name', 'Unknown')
        vice_captain = team_prediction.get('vice_captain', {}).get('name', 'Unknown')
        
        # Calculate value metrics
        value_per_million = total_points / total_cost if total_cost > 0 else 0
        
        return f"""
┌─ 📊 TEAM OVERVIEW ──────────────────────────────────────────────────────────┐
│                                                                              │
│  🎯 Predicted Points: {total_points:5.1f} pts    💰 Team Cost: £{total_cost:5.1f}m    ⚡ Value: {value_per_million:4.1f} pts/£m  │
│  📐 Formation: {formation:>8s}        👑 Captain: {captain:>20s}  │
│  🥈 Vice Captain: {vice_captain:>15s}                                                      │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
"""
    
    def _create_transfer_analysis(self, transfers_made: List[Dict[str, Any]]) -> str:
        """Create transfer analysis section."""
        if not transfers_made:
            return ""
        
        total_gain = sum(t.get('points_gain', 0) for t in transfers_made)
        total_cost = sum(t.get('transfer_cost', 0) for t in transfers_made)
        net_gain = total_gain - total_cost
        
        transfer_lines = []
        for i, transfer in enumerate(transfers_made, 1):
            out_player = transfer.get('transfer_out', {}).get('name', 'Unknown')
            in_player = transfer.get('transfer_in', {}).get('name', 'Unknown')
            points_gain = transfer.get('points_gain', 0)
            cost_diff = transfer.get('cost_diff', 0)
            
            # Add priority indicators
            priority_emoji = ""
            if transfer.get('priority_type') == 'injured':
                priority_emoji = " 🚑"
            elif transfer.get('priority_type') == 'unavailable':
                priority_emoji = " ❌"
            
            transfer_lines.append(
                f"│  {i}. {out_player:>15s} → {in_player:>15s}  "
                f"({points_gain:+.1f} pts, £{cost_diff:+.1f}m){priority_emoji}"
            )
        
        return f"""
┌─ 💱 TRANSFER ANALYSIS ──────────────────────────────────────────────────────┐
│                                                                              │
│  📈 Total Transfers: {len(transfers_made)}        🎯 Points Gain: {total_gain:+.1f} pts    💸 Transfer Cost: {total_cost} pts  │
│  ⚖️  Net Gain: {net_gain:+.1f} pts                                                              │
│                                                                              │
{chr(10).join(transfer_lines)}
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
"""
    
    def _create_team_changes(self, current_team: Dict[str, Any], 
                           previous_team: Dict[str, Any], 
                           transfers_made: List[Dict[str, Any]] = None) -> str:
        """Create team changes section."""
        changes_lines = []
        
        # If we have transfer information, use that instead of comparing teams
        if transfers_made:
            for transfer in transfers_made:
                out_player = transfer['transfer_out']
                in_player = transfer['transfer_in']
                
                # Show the actual transfer made
                changes_lines.append(f"│  🔄 Transfer: {out_player['name']} ({out_player['position']}) → {in_player['name']} ({in_player['position']})")
                
                # Add smart transfer insights if available
                if transfer.get('smart_analysis'):
                    smart_analysis = transfer['smart_analysis']
                    rating = smart_analysis.get('overall_rating', 0)
                    recommendation = smart_analysis.get('recommendation', 'HOLD')
                    changes_lines.append(f"│     🧠 Smart Rating: {rating:.1f}/100 - {recommendation}")
        else:
            # Fallback to comparing teams if no transfer info available
            current_players = {p['name'] for p in current_team.get('playing_xi', [])}
            previous_players = {p['name'] for p in previous_team.get('playing_xi', [])}
            
            new_players = current_players - previous_players
            dropped_players = previous_players - current_players
            
            if new_players:
                changes_lines.append(f"│  ➕ New to XI: {', '.join(sorted(new_players))}")
            if dropped_players:
                changes_lines.append(f"│  ➖ Dropped: {', '.join(sorted(dropped_players))}")
        
        if not changes_lines:
            changes_lines.append("│  🔄 No changes to starting XI")
        
        return f"""
┌─ 🔄 TEAM CHANGES ───────────────────────────────────────────────────────────┐
│                                                                              │
{chr(10).join(changes_lines)}
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
"""
    
    def _create_performance_metrics(self, gameweek: int) -> str:
        """Create performance metrics section."""
        performance = self.performance_tracker.get_weekly_performance(gameweek)
        summary = self.performance_tracker.get_performance_summary()
        
        if not performance.get('has_actual_results'):
            return f"""
┌─ 📈 PERFORMANCE METRICS ────────────────────────────────────────────────────┐
│                                                                              │
│  ⏳ Waiting for gameweek {gameweek} results to calculate accuracy...                    │
│  📊 Overall Accuracy: {summary.get('average_accuracy', 0):.1f}% (last {summary.get('total_predictions', 0)} predictions)  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
"""
        
        accuracy = performance.get('accuracy', 0)
        points_diff = performance.get('points_difference', 0)
        rating = performance.get('performance_rating', 'Unknown')
        
        return f"""
┌─ 📈 PERFORMANCE METRICS ────────────────────────────────────────────────────┐
│                                                                              │
│  🎯 This Week's Accuracy: {accuracy:.1f}% ({rating})                        │
│  📊 Points Difference: {points_diff:+.1f} pts                               │
│  📈 Overall Accuracy: {summary.get('average_accuracy', 0):.1f}% (last {summary.get('total_predictions', 0)} predictions)  │
│  🔄 Recent Trend: {summary.get('recent_accuracy', 0):.1f}% (last 5 GWs)     │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
"""
    
    def _create_recommendations(self, team_prediction: Dict[str, Any], 
                              transfers_made: List[Dict[str, Any]]) -> str:
        """Create recommendations section."""
        recommendations = []
        
        # Analyze team composition
        playing_xi = team_prediction.get('playing_xi', [])
        if playing_xi:
            # Check for injured players
            injured_players = [p for p in playing_xi if p.get('status') == 'i']
            if injured_players:
                recommendations.append(f"🚑 Consider replacing injured players: {', '.join([p['name'] for p in injured_players])}")
            
            # Check for low availability players
            low_availability = [p for p in playing_xi if p.get('chance_of_playing') is not None and p.get('chance_of_playing', 100) < 75]
            if low_availability:
                recommendations.append(f"⚠️  Monitor low availability players: {', '.join([p['name'] for p in low_availability])}")
            
            # Check team distribution
            teams = {}
            for player in playing_xi:
                team = player.get('team', 'Unknown')
                teams[team] = teams.get(team, 0) + 1
            
            over_represented = [team for team, count in teams.items() if count > 3]
            if over_represented:
                recommendations.append(f"⚖️  Consider diversifying from teams: {', '.join(over_represented)}")
        
        # Transfer recommendations
        if transfers_made:
            successful_transfers = [t for t in transfers_made if t.get('points_gain', 0) > 0]
            if len(successful_transfers) == len(transfers_made):
                recommendations.append("✅ All transfers look promising based on predictions")
            elif len(successful_transfers) > 0:
                recommendations.append("⚠️  Some transfers may not provide expected value")
            else:
                recommendations.append("❌ Consider reviewing transfer strategy")
        
        # Budget recommendations
        total_cost = team_prediction.get('total_cost', 0)
        budget_remaining = team_prediction.get('budget_remaining', 0)
        if budget_remaining > 5:
            recommendations.append(f"💰 Consider upgrading players with {budget_remaining:.1f}m remaining budget")
        elif budget_remaining < 0:
            recommendations.append("⚠️  Team exceeds budget - review selections")
        
        if not recommendations:
            recommendations.append("✅ Team looks well-balanced for this gameweek")
        
        rec_lines = []
        for i, rec in enumerate(recommendations, 1):
            rec_lines.append(f"│  {i}. {rec}")
        
        return f"""
┌─ 💡 RECOMMENDATIONS ────────────────────────────────────────────────────────┐
│                                                                              │
{chr(10).join(rec_lines)}
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
"""
    
    def _create_footer(self) -> str:
        """Create dashboard footer."""
        return """
╔══════════════════════════════════════════════════════════════════════════════╗
║  🤖 Generated by FPL Bot - Good luck with your fantasy team! 🍀             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    
    def print_weekly_summary(self, gameweek: int, team_prediction: Dict[str, Any], 
                           previous_team: Optional[Dict[str, Any]] = None,
                           transfers_made: List[Dict[str, Any]] = None) -> None:
        """Print the weekly summary dashboard to console."""
        summary = self.generate_weekly_summary(gameweek, team_prediction, previous_team, transfers_made)
        print(summary)
    
    def save_weekly_summary(self, gameweek: int, team_prediction: Dict[str, Any], 
                          previous_team: Optional[Dict[str, Any]] = None,
                          transfers_made: List[Dict[str, Any]] = None) -> str:
        """Save weekly summary to file."""
        summary = self.generate_weekly_summary(gameweek, team_prediction, previous_team, transfers_made)
        
        # Create weekly reports directory
        reports_dir = os.path.join(self.data_dir, "weekly_reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        # Save as text file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"weekly_summary_gw{gameweek}_{timestamp}.txt"
        filepath = os.path.join(reports_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        return filepath


