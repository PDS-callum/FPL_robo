"""
Smart Transfer Logic for FPL Bot.
Enhanced transfer decisions based on fixture difficulty, form trends, and injury risk.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from .constants import POSITION_MAP


class FPLSmartTransferAnalyzer:
    """
    Analyzes transfer decisions with advanced factors beyond simple point predictions.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.fixture_difficulty_cache = {}
        self.form_trends_cache = {}
    
    def analyze_transfer_opportunity(self, out_player: Dict[str, Any], 
                                   in_player: Dict[str, Any], 
                                   gameweek: int) -> Dict[str, Any]:
        """
        Analyze a potential transfer with smart factors.
        
        Parameters:
        -----------
        out_player : dict
            Player to transfer out
        in_player : dict
            Player to transfer in
        gameweek : int
            Current gameweek
            
        Returns:
        --------
        dict
            Comprehensive transfer analysis
        """
        analysis = {
            'basic_value': in_player.get('predicted_points', 0) - out_player.get('predicted_points', 0),
            'cost_difference': in_player.get('cost', 0) - out_player.get('cost', 0),
            'fixture_analysis': self._analyze_fixture_difficulty(in_player, out_player, gameweek),
            'form_analysis': self._analyze_form_trends(in_player, out_player),
            'injury_risk': self._analyze_injury_risk(in_player, out_player),
            'team_balance': self._analyze_team_balance_impact(in_player, out_player),
            'value_for_money': self._calculate_value_for_money(in_player, out_player),
            'overall_rating': 0.0,
            'recommendation': 'HOLD'
        }
        
        # Calculate overall rating
        analysis['overall_rating'] = self._calculate_overall_rating(analysis)
        analysis['recommendation'] = self._get_transfer_recommendation(analysis)
        
        return analysis
    
    def _analyze_fixture_difficulty(self, in_player: Dict[str, Any], 
                                  out_player: Dict[str, Any], 
                                  gameweek: int) -> Dict[str, Any]:
        """Analyze fixture difficulty for both players."""
        # This would ideally use actual fixture data
        # For now, we'll use a simplified approach based on team strength
        
        in_team = in_player.get('team', 'Unknown')
        out_team = out_player.get('team', 'Unknown')
        
        # Simplified team strength ratings (would be loaded from actual data)
        team_strength = {
            'Manchester City': 9, 'Arsenal': 8, 'Liverpool': 8, 'Chelsea': 7,
            'Tottenham': 7, 'Manchester United': 6, 'Newcastle': 6,
            'Brighton': 5, 'West Ham': 5, 'Aston Villa': 5,
            'Crystal Palace': 4, 'Fulham': 4, 'Brentford': 4,
            'Wolves': 4, 'Everton': 3, 'Nottingham Forest': 3,
            'Bournemouth': 3, 'Luton': 2, 'Sheffield United': 2
        }
        
        in_strength = team_strength.get(in_team, 5)
        out_strength = team_strength.get(out_team, 5)
        
        # Higher strength = easier fixtures (simplified)
        in_fixture_difficulty = 10 - in_strength
        out_fixture_difficulty = 10 - out_strength
        
        return {
            'in_player_difficulty': in_fixture_difficulty,
            'out_player_difficulty': out_fixture_difficulty,
            'difficulty_advantage': out_fixture_difficulty - in_fixture_difficulty,
            'in_team_strength': in_strength,
            'out_team_strength': out_strength
        }
    
    def _analyze_form_trends(self, in_player: Dict[str, Any], 
                           out_player: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze form trends for both players."""
        # This would ideally use historical form data
        # For now, we'll use simplified metrics
        
        in_form = in_player.get('form', 0)
        out_form = out_player.get('form', 0)
        
        # Calculate form momentum (simplified)
        in_momentum = self._calculate_form_momentum(in_player)
        out_momentum = self._calculate_form_momentum(out_player)
        
        return {
            'in_form': in_form,
            'out_form': out_form,
            'form_difference': in_form - out_form,
            'in_momentum': in_momentum,
            'out_momentum': out_momentum,
            'momentum_advantage': in_momentum - out_momentum
        }
    
    def _calculate_form_momentum(self, player: Dict[str, Any]) -> float:
        """Calculate form momentum based on recent performance."""
        # This would ideally use actual recent gameweek data
        # For now, we'll use a simplified calculation
        
        form = player.get('form', 0)
        total_points = player.get('total_points', 0)
        games_played = player.get('games_played', 1)
        
        # Simple momentum calculation
        if games_played > 0:
            avg_points = total_points / games_played
            momentum = (form - avg_points) / max(avg_points, 1) * 100
        else:
            momentum = 0
        
        return max(-100, min(100, momentum))  # Clamp between -100 and 100
    
    def _analyze_injury_risk(self, in_player: Dict[str, Any], 
                           out_player: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze injury risk for both players."""
        in_status = in_player.get('status', 'a')
        out_status = out_player.get('status', 'a')
        in_chance = in_player.get('chance_of_playing', 100)
        out_chance = out_player.get('chance_of_playing', 100)
        
        # Calculate injury risk scores
        in_risk = self._calculate_injury_risk(in_status, in_chance)
        out_risk = self._calculate_injury_risk(out_status, out_chance)
        
        return {
            'in_risk_score': in_risk,
            'out_risk_score': out_risk,
            'risk_difference': in_risk - out_risk,
            'in_status': in_status,
            'out_status': out_status,
            'in_availability': in_chance,
            'out_availability': out_chance
        }
    
    def _calculate_injury_risk(self, status: str, chance: float) -> float:
        """Calculate injury risk score (0-100, higher = more risk)."""
        if status == 'i':  # Injured
            return 100
        elif status == 'u':  # Unavailable
            return 90
        elif status == 'd':  # Doubtful
            return 70
        elif chance is not None and chance < 25:
            return 80
        elif chance is not None and chance < 50:
            return 60
        elif chance is not None and chance < 75:
            return 30
        else:
            return 10  # Low risk
    
    def _analyze_team_balance_impact(self, in_player: Dict[str, Any], 
                                   out_player: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze impact on team balance and formation."""
        in_position = in_player.get('position', 'Unknown')
        out_position = out_player.get('position', 'Unknown')
        
        # Check if positions match (required for valid transfer)
        position_match = in_position == out_position
        
        # Calculate positional value (some positions are more valuable)
        position_values = {'GK': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}
        in_pos_value = position_values.get(in_position, 2)
        out_pos_value = position_values.get(out_position, 2)
        
        return {
            'position_match': position_match,
            'in_position': in_position,
            'out_position': out_position,
            'position_value_change': in_pos_value - out_pos_value,
            'formation_impact': 'None' if position_match else 'Invalid'
        }
    
    def _calculate_value_for_money(self, in_player: Dict[str, Any], 
                                 out_player: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate value for money metrics."""
        in_cost = in_player.get('cost', 0)
        out_cost = out_player.get('cost', 0)
        in_points = in_player.get('predicted_points', 0)
        out_points = out_player.get('predicted_points', 0)
        
        # Calculate points per million
        in_value = in_points / max(in_cost, 0.1) if in_cost > 0 else 0
        out_value = out_points / max(out_cost, 0.1) if out_cost > 0 else 0
        
        # Calculate cost efficiency
        cost_efficiency = (in_value - out_value) / max(out_value, 0.1) * 100 if out_value > 0 else 0
        
        return {
            'in_value_per_million': in_value,
            'out_value_per_million': out_value,
            'value_difference': in_value - out_value,
            'cost_efficiency': cost_efficiency,
            'budget_impact': in_cost - out_cost
        }
    
    def _calculate_overall_rating(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall transfer rating (0-100)."""
        # Weighted scoring system
        weights = {
            'basic_value': 0.25,
            'fixture_advantage': 0.20,
            'form_advantage': 0.15,
            'injury_risk_reduction': 0.15,
            'value_efficiency': 0.15,
            'team_balance': 0.10
        }
        
        # Normalize scores to 0-100 range
        basic_value_score = min(100, max(0, analysis['basic_value'] * 10 + 50))
        
        fixture_advantage = analysis['fixture_analysis']['difficulty_advantage']
        fixture_score = min(100, max(0, fixture_advantage * 5 + 50))
        
        form_advantage = analysis['form_analysis']['form_difference']
        form_score = min(100, max(0, form_advantage * 10 + 50))
        
        injury_risk_reduction = -analysis['injury_risk']['risk_difference']
        injury_score = min(100, max(0, injury_risk_reduction + 50))
        
        value_efficiency = analysis['value_for_money']['cost_efficiency']
        value_score = min(100, max(0, value_efficiency + 50))
        
        team_balance_score = 100 if analysis['team_balance']['position_match'] else 0
        
        # Calculate weighted average
        overall_rating = (
            basic_value_score * weights['basic_value'] +
            fixture_score * weights['fixture_advantage'] +
            form_score * weights['form_advantage'] +
            injury_score * weights['injury_risk_reduction'] +
            value_score * weights['value_efficiency'] +
            team_balance_score * weights['team_balance']
        )
        
        return round(overall_rating, 1)
    
    def _get_transfer_recommendation(self, analysis: Dict[str, Any]) -> str:
        """Get transfer recommendation based on analysis."""
        rating = analysis['overall_rating']
        
        if rating >= 80:
            return 'STRONG BUY'
        elif rating >= 70:
            return 'BUY'
        elif rating >= 60:
            return 'WEAK BUY'
        elif rating >= 40:
            return 'HOLD'
        elif rating >= 30:
            return 'WEAK SELL'
        else:
            return 'SELL'
    
    def get_transfer_insights(self, transfer_analysis: Dict[str, Any]) -> List[str]:
        """Get human-readable insights from transfer analysis."""
        insights = []
        
        # Basic value insight
        basic_value = transfer_analysis['basic_value']
        if basic_value > 2:
            insights.append(f"📈 Strong points gain expected: +{basic_value:.1f} points")
        elif basic_value > 0:
            insights.append(f"📊 Modest points gain: +{basic_value:.1f} points")
        elif basic_value < -2:
            insights.append(f"📉 Points loss expected: {basic_value:.1f} points")
        
        # Fixture insight
        fixture_adv = transfer_analysis['fixture_analysis']['difficulty_advantage']
        if fixture_adv > 1:
            insights.append(f"🏆 Better fixture run for incoming player")
        elif fixture_adv < -1:
            insights.append(f"⚠️  Tougher fixtures for incoming player")
        
        # Form insight
        form_adv = transfer_analysis['form_analysis']['form_difference']
        if form_adv > 1:
            insights.append(f"🔥 Incoming player in better form")
        elif form_adv < -1:
            insights.append(f"❄️  Outgoing player in better form")
        
        # Injury risk insight
        risk_diff = transfer_analysis['injury_risk']['risk_difference']
        if risk_diff < -20:
            insights.append(f"🛡️  Significantly reduces injury risk")
        elif risk_diff > 20:
            insights.append(f"⚠️  Increases injury risk")
        
        # Value insight
        value_diff = transfer_analysis['value_for_money']['value_difference']
        if value_diff > 0.5:
            insights.append(f"💰 Better value for money")
        elif value_diff < -0.5:
            insights.append(f"💸 Lower value for money")
        
        return insights
    
    def rank_transfer_options(self, current_team: List[Dict[str, Any]], 
                            available_players: List[Dict[str, Any]], 
                            gameweek: int) -> List[Dict[str, Any]]:
        """Rank all possible transfer options."""
        transfer_options = []
        
        for out_player in current_team:
            for in_player in available_players:
                if (in_player['name'] != out_player['name'] and 
                    in_player.get('position') == out_player.get('position')):
                    
                    analysis = self.analyze_transfer_opportunity(out_player, in_player, gameweek)
                    
                    transfer_options.append({
                        'out_player': out_player,
                        'in_player': in_player,
                        'analysis': analysis,
                        'insights': self.get_transfer_insights(analysis)
                    })
        
        # Sort by overall rating
        transfer_options.sort(key=lambda x: x['analysis']['overall_rating'], reverse=True)
        
        return transfer_options


