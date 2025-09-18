"""
Weekly Reports Generator for FPL Bot.
Creates HTML and PDF reports with charts and analysis.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from .performance_tracker import FPLPerformanceTracker
from .weekly_dashboard import FPLWeeklyDashboard


class FPLWeeklyReports:
    """
    Generates comprehensive weekly reports in HTML and PDF formats.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.performance_tracker = FPLPerformanceTracker(data_dir)
        self.dashboard = FPLWeeklyDashboard(data_dir)
        self.reports_dir = os.path.join(data_dir, "weekly_reports")
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def generate_html_report(self, gameweek: int, team_prediction: Dict[str, Any],
                           previous_team: Optional[Dict[str, Any]] = None,
                           transfers_made: List[Dict[str, Any]] = None) -> str:
        """Generate HTML weekly report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"weekly_report_gw{gameweek}_{timestamp}.html"
        filepath = os.path.join(self.reports_dir, filename)
        
        html_content = self._create_html_content(gameweek, team_prediction, previous_team, transfers_made)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return filepath
    
    def _create_html_content(self, gameweek: int, team_prediction: Dict[str, Any],
                           previous_team: Optional[Dict[str, Any]] = None,
                           transfers_made: List[Dict[str, Any]] = None) -> str:
        """Create HTML content for weekly report."""
        performance_summary = self.performance_tracker.get_performance_summary()
        weekly_performance = self.performance_tracker.get_weekly_performance(gameweek)
        
        # Team overview data
        total_points = team_prediction.get('total_predicted_points', 0)
        total_cost = team_prediction.get('total_cost', 0)
        formation = team_prediction.get('formation', 'Unknown')
        captain = team_prediction.get('captain', {}).get('name', 'Unknown')
        vice_captain = team_prediction.get('vice_captain', {}).get('name', 'Unknown')
        
        # Transfer data
        transfer_summary = self._get_transfer_summary(transfers_made) if transfers_made else {}
        
        # Performance trends
        trends = self.performance_tracker.generate_performance_trends()
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FPL Bot - Gameweek {gameweek} Report</title>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>🏆 FPL Bot Weekly Report</h1>
            <h2>Gameweek {gameweek} Analysis</h2>
            <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </header>
        
        <div class="grid">
            <div class="card team-overview">
                <h3>📊 Team Overview</h3>
                <div class="metrics">
                    <div class="metric">
                        <span class="label">Predicted Points:</span>
                        <span class="value">{total_points:.1f}</span>
                    </div>
                    <div class="metric">
                        <span class="label">Team Cost:</span>
                        <span class="value">£{total_cost:.1f}m</span>
                    </div>
                    <div class="metric">
                        <span class="label">Formation:</span>
                        <span class="value">{formation}</span>
                    </div>
                    <div class="metric">
                        <span class="label">Captain:</span>
                        <span class="value">{captain}</span>
                    </div>
                    <div class="metric">
                        <span class="label">Vice Captain:</span>
                        <span class="value">{vice_captain}</span>
                    </div>
                </div>
            </div>
            
            <div class="card performance">
                <h3>📈 Performance Metrics</h3>
                <div class="metrics">
                    <div class="metric">
                        <span class="label">Overall Accuracy:</span>
                        <span class="value">{performance_summary.get('average_accuracy', 0):.1f}%</span>
                    </div>
                    <div class="metric">
                        <span class="label">Recent Accuracy:</span>
                        <span class="value">{performance_summary.get('recent_accuracy', 0):.1f}%</span>
                    </div>
                    <div class="metric">
                        <span class="label">Prediction Bias:</span>
                        <span class="value">{performance_summary.get('prediction_bias', 0):+.1f}</span>
                    </div>
                    <div class="metric">
                        <span class="label">Total Predictions:</span>
                        <span class="value">{performance_summary.get('total_predictions', 0)}</span>
                    </div>
                </div>
            </div>
        </div>
        
        {self._create_transfer_section(transfer_summary)}
        {self._create_team_section(team_prediction, previous_team)}
        {self._create_performance_trends_section(trends)}
        {self._create_recommendations_section(team_prediction, transfers_made)}
    </div>
</body>
</html>
"""
        return html
    
    def _get_css_styles(self) -> str:
        """Get CSS styles for HTML report."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header h2 {
            font-size: 1.5em;
            margin-bottom: 10px;
        }
        
        .timestamp {
            font-size: 0.9em;
            opacity: 0.9;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }
        
        .card h3 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.3em;
        }
        
        .metrics {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
        }
        
        .metric .label {
            font-weight: 500;
            color: #666;
        }
        
        .metric .value {
            font-weight: bold;
            color: #333;
            font-size: 1.1em;
        }
        
        .section {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        .section h3 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.3em;
        }
        
        .team-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .player-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 3px solid #28a745;
        }
        
        .player-card.injured {
            border-left-color: #dc3545;
            background: #fff5f5;
        }
        
        .player-card.doubtful {
            border-left-color: #ffc107;
            background: #fffbf0;
        }
        
        .player-name {
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }
        
        .player-details {
            font-size: 0.9em;
            color: #666;
        }
        
        .transfer-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 3px solid #17a2b8;
        }
        
        .transfer-direction {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }
        
        .transfer-arrow {
            font-size: 1.2em;
            color: #17a2b8;
        }
        
        .transfer-gain {
            font-weight: bold;
            color: #28a745;
        }
        
        .transfer-loss {
            font-weight: bold;
            color: #dc3545;
        }
        
        .recommendation {
            background: #e7f3ff;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 3px solid #007bff;
        }
        
        .recommendation.warning {
            background: #fff3cd;
            border-left-color: #ffc107;
        }
        
        .recommendation.danger {
            background: #f8d7da;
            border-left-color: #dc3545;
        }
        
        .chart-placeholder {
            background: #f8f9fa;
            padding: 40px;
            text-align: center;
            border-radius: 8px;
            color: #666;
            border: 2px dashed #dee2e6;
        }
        
        @media (max-width: 768px) {
            .grid {
                grid-template-columns: 1fr;
            }
            
            .team-grid {
                grid-template-columns: 1fr;
            }
        }
        """
    
    def _create_transfer_section(self, transfer_summary: Dict[str, Any]) -> str:
        """Create transfer analysis section."""
        if not transfer_summary:
            return ""
        
        transfers = transfer_summary.get('transfers', [])
        total_gain = transfer_summary.get('total_gain', 0)
        total_cost = transfer_summary.get('total_cost', 0)
        
        transfer_html = f"""
        <div class="section">
            <h3>💱 Transfer Analysis</h3>
            <div class="metrics">
                <div class="metric">
                    <span class="label">Total Transfers:</span>
                    <span class="value">{len(transfers)}</span>
                </div>
                <div class="metric">
                    <span class="label">Points Gain:</span>
                    <span class="value {'transfer-gain' if total_gain > 0 else 'transfer-loss'}">{total_gain:+.1f}</span>
                </div>
                <div class="metric">
                    <span class="label">Transfer Cost:</span>
                    <span class="value">{total_cost} pts</span>
                </div>
            </div>
        """
        
        for transfer in transfers:
            out_player = transfer.get('transfer_out', {}).get('name', 'Unknown')
            in_player = transfer.get('transfer_in', {}).get('name', 'Unknown')
            points_gain = transfer.get('points_gain', 0)
            cost_diff = transfer.get('cost_diff', 0)
            
            transfer_html += f"""
            <div class="transfer-item">
                <div class="transfer-direction">
                    <span>{out_player}</span>
                    <span class="transfer-arrow">→</span>
                    <span>{in_player}</span>
                </div>
                <div class="player-details">
                    Points: {points_gain:+.1f} | Cost: £{cost_diff:+.1f}m
                </div>
            </div>
            """
        
        transfer_html += "</div>"
        return transfer_html
    
    def _create_team_section(self, team_prediction: Dict[str, Any], 
                           previous_team: Optional[Dict[str, Any]] = None) -> str:
        """Create team composition section."""
        playing_xi = team_prediction.get('playing_xi', [])
        
        team_html = f"""
        <div class="section">
            <h3>⚽ Team Composition</h3>
            <div class="team-grid">
        """
        
        for player in playing_xi:
            name = player.get('name', 'Unknown')
            position = player.get('position', 'Unknown')
            points = player.get('predicted_points', 0)
            cost = player.get('cost', 0)
            status = player.get('status', 'a')
            chance = player.get('chance_of_playing', 100)
            
            # Determine card class based on status
            card_class = "player-card"
            if status == 'i' or (chance is not None and chance < 25):
                card_class += " injured"
            elif status == 'd' or (chance is not None and chance < 75):
                card_class += " doubtful"
            
            team_html += f"""
                <div class="{card_class}">
                    <div class="player-name">{name}</div>
                    <div class="player-details">
                        {position} | {points:.1f} pts | £{cost:.1f}m
                        {f" | {chance}% chance" if chance is not None and chance < 100 else ""}
                    </div>
                </div>
            """
        
        team_html += """
            </div>
        </div>
        """
        return team_html
    
    def _create_performance_trends_section(self, trends: Dict[str, Any]) -> str:
        """Create performance trends section."""
        if 'message' in trends:
            return f"""
            <div class="section">
                <h3>📈 Performance Trends</h3>
                <p>{trends['message']}</p>
            </div>
            """
        
        return f"""
        <div class="section">
            <h3>📈 Performance Trends</h3>
            <div class="chart-placeholder">
                <p>📊 Performance trend chart would be displayed here</p>
                <p>Trend Direction: {trends.get('trend_direction', 'Unknown')}</p>
                <p>Consistency Score: {trends.get('consistency_score', 0):.1f}%</p>
            </div>
        </div>
        """
    
    def _create_recommendations_section(self, team_prediction: Dict[str, Any], 
                                      transfers_made: List[Dict[str, Any]]) -> str:
        """Create recommendations section."""
        recommendations = self._generate_recommendations(team_prediction, transfers_made)
        
        rec_html = """
        <div class="section">
            <h3>💡 Recommendations</h3>
        """
        
        for rec in recommendations:
            rec_type = rec.get('type', 'info')
            message = rec.get('message', '')
            
            rec_html += f"""
            <div class="recommendation {rec_type}">
                {message}
            </div>
            """
        
        rec_html += "</div>"
        return rec_html
    
    def _generate_recommendations(self, team_prediction: Dict[str, Any], 
                                transfers_made: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate recommendations based on team and transfers."""
        recommendations = []
        
        # Check for injured players
        playing_xi = team_prediction.get('playing_xi', [])
        injured_players = [p for p in playing_xi if p.get('status') == 'i']
        if injured_players:
            recommendations.append({
                'type': 'danger',
                'message': f"🚑 Replace injured players: {', '.join([p['name'] for p in injured_players])}"
            })
        
        # Check budget utilization
        total_cost = team_prediction.get('total_cost', 0)
        budget_remaining = team_prediction.get('budget_remaining', 0)
        if budget_remaining > 5:
            recommendations.append({
                'type': 'info',
                'message': f"💰 Consider upgrading players with {budget_remaining:.1f}m remaining budget"
            })
        
        # Check transfer success
        if transfers_made:
            successful_transfers = [t for t in transfers_made if t.get('points_gain', 0) > 0]
            if len(successful_transfers) == len(transfers_made):
                recommendations.append({
                    'type': 'info',
                    'message': "✅ All transfers look promising based on predictions"
                })
        
        return recommendations
    
    def _get_transfer_summary(self, transfers_made: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get transfer summary data."""
        if not transfers_made:
            return {}
        
        total_gain = sum(t.get('points_gain', 0) for t in transfers_made)
        total_cost = sum(t.get('transfer_cost', 0) for t in transfers_made)
        
        return {
            'transfers': transfers_made,
            'total_gain': total_gain,
            'total_cost': total_cost,
            'net_gain': total_gain - total_cost
        }
    
    def generate_pdf_report(self, gameweek: int, team_prediction: Dict[str, Any],
                          previous_team: Optional[Dict[str, Any]] = None,
                          transfers_made: List[Dict[str, Any]] = None) -> str:
        """Generate PDF weekly report (placeholder - would require additional dependencies)."""
        # This would require libraries like weasyprint or reportlab
        # For now, we'll return the HTML file path
        html_file = self.generate_html_report(gameweek, team_prediction, previous_team, transfers_made)
        
        # In a real implementation, you would convert HTML to PDF here
        print(f"📄 PDF generation not implemented. HTML report available at: {html_file}")
        
        return html_file


