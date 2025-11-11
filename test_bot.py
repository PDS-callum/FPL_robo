"""
Test Bot for FPL Platform Competition
A simple bot to verify the upload and validation system works correctly
"""


class SimpleTestBot:
    """
    A basic FPL bot for testing the competition platform
    This bot uses simple strategies to demonstrate functionality
    """
    
    def __init__(self):
        """Initialize the test bot"""
        self.name = "SimpleTestBot"
        self.version = "1.0"
        self.strategy = "test"
    
    def analyze(self, data):
        """
        Analyze current team and provide insights
        
        This method demonstrates:
        - Data processing
        - Simple calculations
        - Insight generation
        """
        # Extract data with safe defaults
        current_team = data.get('current_team', [])
        gameweek = data.get('gameweek', 1)
        team_value = data.get('team_value', 100.0)
        bank = data.get('bank', 0.0)
        
        # Perform basic analysis
        team_size = len(current_team)
        total_cost = sum(player.get('cost', 0) for player in current_team)
        total_points = sum(player.get('points', 0) for player in current_team)
        avg_points = total_points / max(team_size, 1)
        
        # Generate insights
        insights = []
        insights.append(f"Team has {team_size} players")
        insights.append(f"Average points per player: {avg_points:.1f}")
        
        if bank > 0.5:
            insights.append(f"You have £{bank}m in the bank - consider upgrading")
        
        if team_size < 15:
            insights.append("Warning: Incomplete squad")
        
        return {
            'status': 'ok',
            'bot_name': self.name,
            'bot_version': self.version,
            'gameweek': gameweek,
            'message': f'Analysis complete for GW{gameweek}',
            'statistics': {
                'team_size': team_size,
                'total_cost': total_cost,
                'total_points': total_points,
                'avg_points': avg_points,
                'team_value': team_value,
                'bank': bank
            },
            'insights': insights
        }
    
    def optimize(self, data):
        """
        Optimize team selection and transfers
        
        This method demonstrates:
        - Transfer logic
        - Captain selection
        - Team lineup creation
        """
        # Extract data
        current_team = data.get('current_team', [])
        all_players = data.get('all_players', [])
        free_transfers = data.get('free_transfers', 1)
        budget = data.get('budget', 0.0)
        predictions = data.get('predictions', {})
        gameweek = data.get('gameweek', 1)
        
        # Sort current team by predicted points
        team_with_predictions = []
        for player in current_team:
            player_id = player.get('id')
            predicted_points = predictions.get(player_id, 0)
            team_with_predictions.append({
                'player': player,
                'id': player_id,
                'predicted_points': predicted_points
            })
        
        # Sort by predicted points (descending)
        team_with_predictions.sort(key=lambda x: x['predicted_points'], reverse=True)
        
        # Select captain and vice captain (highest predicted points)
        captain_id = None
        vice_captain_id = None
        
        if len(team_with_predictions) > 0:
            captain_id = team_with_predictions[0]['id']
        if len(team_with_predictions) > 1:
            vice_captain_id = team_with_predictions[1]['id']
        
        # Simple transfer logic
        transfers = []
        
        if free_transfers > 0 and len(team_with_predictions) > 0:
            # Get the worst player (lowest predicted points)
            worst_player = team_with_predictions[-1]
            
            # Find a better replacement if budget allows
            current_ids = [p['id'] for p in current_team]
            
            # Simple logic: look for players not in team with higher predictions
            for available_player in all_players[:10]:  # Check top 10 available
                player_id = available_player.get('id')
                if player_id not in current_ids:
                    available_prediction = predictions.get(player_id, 0)
                    
                    # If significantly better and affordable, suggest transfer
                    if available_prediction > worst_player['predicted_points'] + 1:
                        player_cost = available_player.get('cost', 0)
                        worst_cost = worst_player['player'].get('cost', 0)
                        
                        if player_cost <= worst_cost + budget:
                            transfers.append({
                                'out': worst_player['id'],
                                'in': player_id,
                                'reason': 'Predicted points upgrade'
                            })
                            break  # Only suggest one transfer for this test
        
        # Select lineup (best 11 by predicted points)
        lineup_ids = [item['id'] for item in team_with_predictions[:11]]
        bench_ids = [item['id'] for item in team_with_predictions[11:15]]
        
        # Determine formation (simplified - would need position data in reality)
        formation = '4-4-2'  # Default formation
        
        return {
            'status': 'success',
            'bot_name': self.name,
            'gameweek': gameweek,
            'transfers': transfers,
            'transfer_count': len(transfers),
            'captain': captain_id,
            'vice_captain': vice_captain_id,
            'lineup': lineup_ids,
            'bench': bench_ids,
            'formation': formation,
            'expected_points': sum(item['predicted_points'] for item in team_with_predictions[:11]),
            'confidence': 0.8,
            'notes': [
                f"Captain selected: highest predicted points",
                f"Transfers suggested: {len(transfers)}",
                f"Formation: {formation}"
            ]
        }


# This section is for local testing only
# It will not run on the platform
if __name__ == '__main__':
    print("="*60)
    print("Testing SimpleTestBot")
    print("="*60)
    
    # Create bot instance
    bot = SimpleTestBot()
    print(f"\nBot initialized: {bot.name} v{bot.version}")
    
    # Mock test data
    test_data = {
        'current_team': [
            {'id': 1, 'name': 'Player 1', 'cost': 10.0, 'points': 50},
            {'id': 2, 'name': 'Player 2', 'cost': 8.5, 'points': 45},
            {'id': 3, 'name': 'Player 3', 'cost': 6.0, 'points': 30},
        ],
        'all_players': [
            {'id': 1, 'cost': 10.0},
            {'id': 2, 'cost': 8.5},
            {'id': 3, 'cost': 6.0},
            {'id': 4, 'cost': 9.0},
        ],
        'gameweek': 10,
        'team_value': 100.0,
        'bank': 0.5,
        'free_transfers': 1,
        'budget': 0.5,
        'predictions': {
            1: 6.5,
            2: 5.5,
            3: 4.0,
            4: 7.5
        }
    }
    
    # Test analyze method
    print("\n" + "-"*60)
    print("Testing analyze() method:")
    print("-"*60)
    analysis = bot.analyze(test_data)
    print(f"Status: {analysis['status']}")
    print(f"Message: {analysis['message']}")
    print(f"Statistics: {analysis['statistics']}")
    print(f"Insights: {analysis['insights']}")
    
    # Test optimize method
    print("\n" + "-"*60)
    print("Testing optimize() method:")
    print("-"*60)
    optimization = bot.optimize(test_data)
    print(f"Status: {optimization['status']}")
    print(f"Captain ID: {optimization['captain']}")
    print(f"Vice Captain ID: {optimization['vice_captain']}")
    print(f"Transfers: {optimization['transfers']}")
    print(f"Lineup IDs: {optimization['lineup']}")
    print(f"Expected Points: {optimization['expected_points']}")
    
    print("\n" + "="*60)
    print("Test Complete! Bot is ready for upload.")
    print("="*60)

