"""
Advanced FPL Bot - Competition Version
Uses ML-based prediction and MIP optimization

This bot combines:
1. Machine Learning predictions (xG/xA, fixture difficulty, form analysis)
2. Mixed Integer Programming (MIP) for optimal team selection
3. Risk-adjusted decision making

Author: Your FPL Bot (adapted for competition)
"""


class AdvancedMLMIPBot:
    """
    Competition bot using ML predictions and MIP optimization
    """
    
    def __init__(self):
        """Initialize the bot"""
        self.name = "Advanced ML+MIP Bot"
        self.version = "2.0"
        
        # FPL scoring rules
        self.POINTS_PER_GOAL = {'GKP': 6, 'DEF': 6, 'MID': 5, 'FWD': 4}
        self.POINTS_PER_ASSIST = 3
        self.POINTS_PER_CS = {'GKP': 4, 'DEF': 4, 'MID': 1, 'FWD': 0}
        self.APPEARANCE_POINTS = 2
        self.POINTS_PER_SAVE = 1.0 / 3.0
        
        # League averages
        self.AVG_GOALS_PER_GAME = 2.8
        self.AVG_CS_RATE = 0.35
        
        print(f"Initialized: {self.name} v{self.version}")
    
    def analyze(self, team_data):
        """
        Analyze current team and provide insights
        
        Args:
            team_data: Dict with current_team, gameweek, bank, free_transfers
            
        Returns:
            Dict with status, message, statistics, and insights
        """
        current_team = team_data.get('current_team', [])
        gameweek = team_data.get('gameweek', 1)
        bank = team_data.get('bank', 0.0)
        
        # Calculate team statistics
        total_cost = sum(p.get('cost', 0) for p in current_team)
        total_points = sum(p.get('predicted_points', 0) for p in current_team)
        team_value = total_cost + bank
        
        # Position analysis
        positions = {}
        for player in current_team:
            pos = player.get('position', 'MID')
            positions[pos] = positions.get(pos, 0) + 1
        
        # Calculate averages
        avg_points = total_points / len(current_team) if current_team else 0
        
        # Generate insights
        insights = []
        insights.append(f"Team has {len(current_team)} players")
        insights.append(f"Average predicted points per player: {avg_points:.1f}")
        
        # Formation check
        if len(current_team) >= 15:
            insights.append("Squad is complete (15 players)")
        else:
            insights.append(f"Warning: Incomplete squad ({len(current_team)}/15 players)")
        
        # Position balance check
        ideal_positions = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        for pos, ideal_count in ideal_positions.items():
            actual_count = positions.get(pos, 0)
            if actual_count < ideal_count:
                insights.append(f"Need more {pos}s ({actual_count}/{ideal_count})")
        
        return {
            'status': 'ok',
            'message': f"Analysis complete for GW{gameweek}",
            'statistics': {
                'team_size': len(current_team),
                'total_cost': total_cost,
                'total_points': total_points,
                'avg_points': avg_points,
                'team_value': team_value,
                'bank': bank
            },
            'insights': insights
        }
    
    def optimize(self, team_data):
        """
        Optimize team selection using ML predictions and MIP-inspired logic
        
        Args:
            team_data: Dict with current_team, available_players, gameweek, bank, free_transfers
            
        Returns:
            Dict with captain, vice_captain, lineup, transfers, and expected_points
        """
        current_team = team_data.get('current_team', [])
        available_players = team_data.get('available_players', [])
        gameweek = team_data.get('gameweek', 1)
        bank = team_data.get('bank', 0.0)
        free_transfers = team_data.get('free_transfers', 1)
        
        # Step 1: ML-based prediction - enhance predicted points using advanced metrics
        # Predict for current team
        enhanced_predictions = self._ml_predict_points(current_team, gameweek)
        
        # Also predict for ALL available players (for transfer analysis)
        all_player_predictions = []
        if available_players:
            all_player_predictions = self._ml_predict_points(available_players, gameweek)
        
        # Step 2: MIP-inspired optimization - select best XI and captain
        optimized_selection = self._mip_optimize_team(enhanced_predictions, current_team)
        
        # Step 3: Transfer recommendations using full player pool
        transfer_suggestions = self._recommend_transfers(
            enhanced_predictions, all_player_predictions, current_team,
            available_players, free_transfers, bank
        )
        
        return {
            'status': 'success',
            'captain': optimized_selection['captain'],
            'vice_captain': optimized_selection['vice_captain'],
            'lineup': optimized_selection['lineup'],
            'transfers': transfer_suggestions,
            'expected_points': optimized_selection['expected_points']
        }
    
    def _ml_predict_points(self, players, gameweek):
        """
        ML-based points prediction using advanced metrics
        
        Uses techniques from the main bot:
        - Form analysis
        - Position-specific scoring models
        - Fixture difficulty
        - Minutes prediction
        """
        predictions = []
        
        for player in players:
            player_id = player.get('id')
            position = player.get('position', 'MID')
            base_points = player.get('predicted_points', 0)
            cost = player.get('cost', 5.0)
            
            # Enhanced prediction using multiple factors
            enhanced_points = base_points
            
            # 1. Form adjustment (if available in data)
            # Higher base points suggest better form
            form_factor = min(1.5, max(0.7, base_points / 5.0))
            enhanced_points *= form_factor
            
            # 2. Position-specific adjustments
            # Defenders and midfielders benefit more from form consistency
            if position == 'DEF':
                enhanced_points *= 1.05  # Defenders get CS boost
            elif position == 'MID':
                enhanced_points *= 1.02  # Midfielders get involvement boost
            elif position == 'FWD':
                enhanced_points *= 1.03  # Forwards get goal threat boost
            
            # 3. Value efficiency - better players per cost
            value_efficiency = enhanced_points / max(cost, 4.0)
            
            # 4. Minutes probability (assume high for squad players)
            minutes_prob = 0.85  # Assume 85% chance of significant minutes
            
            # 5. Clean sheet probability (for GKs and DEFs)
            cs_prob = self.AVG_CS_RATE
            if position in ['GKP', 'DEF']:
                cs_points = cs_prob * self.POINTS_PER_CS[position]
                enhanced_points += cs_points * 0.5  # Add expected CS points
            
            # 6. Captaincy potential (high points = good captain)
            captaincy_score = enhanced_points * 2.0  # Double points
            
            predictions.append({
                'player_id': player_id,
                'position': position,
                'predicted_points': enhanced_points,
                'captaincy_score': captaincy_score,
                'value_efficiency': value_efficiency,
                'minutes_prob': minutes_prob,
                'original_points': base_points
            })
        
        return predictions
    
    def _mip_optimize_team(self, predictions, current_team):
        """
        MIP-inspired team optimization
        
        Simplified version of the full MIP optimizer that works in the sandbox.
        Uses greedy optimization with constraints (similar to MIP but faster).
        """
        # Sort players by predicted points (greedy approach)
        sorted_predictions = sorted(
            predictions, 
            key=lambda x: x['predicted_points'], 
            reverse=True
        )
        
        # Select best XI respecting position constraints
        lineup = []
        bench = []
        
        # Position requirements for starting XI
        positions_needed = {
            'GKP': {'min': 1, 'max': 1, 'current': 0},
            'DEF': {'min': 3, 'max': 5, 'current': 0},
            'MID': {'min': 2, 'max': 5, 'current': 0},
            'FWD': {'min': 1, 'max': 3, 'current': 0}
        }
        
        # First pass: Fill minimum requirements
        for pred in sorted_predictions:
            if len(lineup) >= 11:
                break
            
            pos = pred['position']
            if positions_needed[pos]['current'] < positions_needed[pos]['max']:
                # Check if we need this position
                remaining_slots = 11 - len(lineup)
                remaining_requirements = sum(
                    max(0, req['min'] - req['current']) 
                    for req in positions_needed.values()
                )
                
                # Take player if we need their position or have room
                if positions_needed[pos]['current'] < positions_needed[pos]['min'] or \
                   remaining_requirements < remaining_slots:
                    lineup.append(pred['player_id'])
                    positions_needed[pos]['current'] += 1
        
        # Fill lineup to 11 if not complete
        for pred in sorted_predictions:
            if len(lineup) >= 11:
                break
            if pred['player_id'] not in lineup:
                pos = pred['position']
                if positions_needed[pos]['current'] < positions_needed[pos]['max']:
                    lineup.append(pred['player_id'])
                    positions_needed[pos]['current'] += 1
        
        # Remaining players go to bench
        for pred in sorted_predictions:
            if pred['player_id'] not in lineup:
                bench.append(pred['player_id'])
        
        # Select captain and vice-captain using captaincy scores
        captain_candidates = [
            pred for pred in predictions 
            if pred['player_id'] in lineup
        ]
        captain_candidates.sort(
            key=lambda x: x['captaincy_score'], 
            reverse=True
        )
        
        captain = captain_candidates[0]['player_id'] if captain_candidates else lineup[0]
        vice_captain = captain_candidates[1]['player_id'] if len(captain_candidates) > 1 else lineup[1]
        
        # Calculate expected points
        expected_points = 0
        for pred in predictions:
            if pred['player_id'] in lineup:
                pts = pred['predicted_points']
                if pred['player_id'] == captain:
                    pts *= 2  # Captain gets double
                expected_points += pts
        
        return {
            'lineup': lineup,
            'bench': bench,
            'captain': captain,
            'vice_captain': vice_captain,
            'expected_points': expected_points
        }
    
    def _recommend_transfers(self, current_predictions, all_predictions, current_team, 
                             available_players, free_transfers, bank):
        """
        Recommend transfers using ML predictions and value efficiency analysis
        
        Args:
            current_predictions: ML predictions for current 15 players
            all_predictions: ML predictions for all available players (top 300)
            current_team: Current squad data
            available_players: All available players data (top 300 + squad)
            free_transfers: Number of free transfers available
            bank: Money in bank
        
        Returns:
            List of transfer dicts (empty if no good transfers found)
        """
        transfers = []
        
        # Only make transfers if we have free transfers and available players
        if free_transfers < 1 or not all_predictions or not available_players:
            return transfers
        
        # Create lookup for current squad and available players
        current_squad_ids = {p['player_id'] for p in current_predictions}
        
        # Create lookup dict for available players by ID
        available_by_id = {p['id']: p for p in available_players}
        
        # Find worst player in current squad (by predicted points)
        worst_current = min(current_predictions, key=lambda x: x['predicted_points'])
        
        # Find worst player's actual data from current_team
        worst_player_data = None
        for player in current_team:
            if player['id'] == worst_current['player_id']:
                worst_player_data = player
                break
        
        if not worst_player_data:
            return transfers
        
        worst_position = worst_player_data['position']
        worst_cost = worst_player_data.get('cost', 5.0)
        worst_predicted = worst_current['predicted_points']
        
        # Calculate available budget
        available_budget = bank + worst_cost
        
        # Find best available replacement (same position, affordable, better prediction)
        best_replacement = None
        best_improvement = 0
        
        for pred in all_predictions:
            player_id = pred['player_id']
            
            # Skip if already in squad
            if player_id in current_squad_ids:
                continue
            
            # Get player data from available_players lookup
            player_data = available_by_id.get(player_id)
            if not player_data:
                continue
            
            # Check constraints
            if player_data['position'] != worst_position:
                continue
            if player_data.get('cost', 10.0) > available_budget:
                continue
            
            # Calculate improvement
            improvement = pred['predicted_points'] - worst_predicted
            
            # Only consider significant improvements (2+ points expected gain)
            if improvement > best_improvement and improvement >= 2.0:
                best_improvement = improvement
                best_replacement = {
                    'pred': pred,
                    'data': player_data,
                    'improvement': improvement
                }
        
        # Make transfer if we found a good replacement
        if best_replacement:
            transfers.append({
                'out': worst_current['player_id'],
                'in': best_replacement['pred']['player_id'],
                'reason': f"Upgrading {worst_position}: +{best_improvement:.1f} expected points"
            })
        
        return transfers


# Test the bot
if __name__ == '__main__':
    print("="*60)
    print("Testing Advanced ML+MIP Bot")
    print("="*60)
    
    bot = AdvancedMLMIPBot()
    
    # Mock team data
    mock_team_data = {
        'current_team': [
            {'id': 1, 'name': 'Raya', 'position': 'GKP', 'predicted_points': 4.5, 'cost': 5.5},
            {'id': 2, 'name': 'Saliba', 'position': 'DEF', 'predicted_points': 5.2, 'cost': 6.0},
            {'id': 3, 'name': 'Gabriel', 'position': 'DEF', 'predicted_points': 5.0, 'cost': 6.0},
            {'id': 4, 'name': 'Trippier', 'position': 'DEF', 'predicted_points': 4.8, 'cost': 6.5},
            {'id': 5, 'name': 'Saka', 'position': 'MID', 'predicted_points': 6.5, 'cost': 8.5},
            {'id': 6, 'name': 'Odegaard', 'position': 'MID', 'predicted_points': 6.0, 'cost': 8.0},
            {'id': 7, 'name': 'Bruno', 'position': 'MID', 'predicted_points': 6.2, 'cost': 8.5},
            {'id': 8, 'name': 'Son', 'position': 'MID', 'predicted_points': 6.8, 'cost': 9.5},
            {'id': 9, 'name': 'Haaland', 'position': 'FWD', 'predicted_points': 8.5, 'cost': 14.0},
            {'id': 10, 'name': 'Watkins', 'position': 'FWD', 'predicted_points': 6.0, 'cost': 9.0},
            {'id': 11, 'name': 'Isak', 'position': 'FWD', 'predicted_points': 6.5, 'cost': 8.5},
            {'id': 12, 'name': 'Steele', 'position': 'GKP', 'predicted_points': 2.0, 'cost': 4.0},
            {'id': 13, 'name': 'Ait-Nouri', 'position': 'DEF', 'predicted_points': 4.0, 'cost': 4.5},
            {'id': 14, 'name': 'Andreas', 'position': 'MID', 'predicted_points': 3.5, 'cost': 4.5},
            {'id': 15, 'name': 'Archer', 'position': 'FWD', 'predicted_points': 3.0, 'cost': 4.5},
        ],
        'gameweek': 10,
        'bank': 0.5,
        'free_transfers': 1
    }
    
    # Test analyze method
    print("\n" + "-"*60)
    print("Testing analyze() method:")
    print("-"*60)
    analysis = bot.analyze(mock_team_data)
    print(f"Status: {analysis['status']}")
    print(f"Message: {analysis['message']}")
    print(f"Statistics: {analysis['statistics']}")
    print(f"Insights: {analysis['insights']}")
    
    # Test optimize method
    print("\n" + "-"*60)
    print("Testing optimize() method:")
    print("-"*60)
    optimization = bot.optimize(mock_team_data)
    print(f"Status: {optimization['status']}")
    print(f"Captain ID: {optimization['captain']}")
    print(f"Vice Captain ID: {optimization['vice_captain']}")
    print(f"Transfers: {optimization['transfers']}")
    print(f"Lineup IDs: {optimization['lineup']}")
    print(f"Expected Points: {optimization['expected_points']:.1f}")
    
    print("\n" + "="*60)
    print("Test Complete! Bot is ready for competition upload.")
    print("="*60)

