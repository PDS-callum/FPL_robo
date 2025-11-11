"""
Sample FPL Bot Template
Upload this file to the FPL Platform competition to get started!

Your bot must implement the 'analyze' and 'optimize' methods.
"""


class MyFPLBot:
    """
    A simple FPL bot that makes basic decisions
    Customize this class with your own strategy!
    """
    
    def __init__(self):
        """Initialize your bot with any necessary state"""
        self.team = []
        self.budget = 100.0
        self.strategy = "balanced"  # You can define your own strategy variables
    
    def analyze(self, data):
        """
        Analyze current team and game data
        
        Parameters:
            data (dict): Contains information about:
                - current_team: List of current players
                - player_stats: Statistics for all players
                - fixtures: Upcoming fixtures
                - gameweek: Current gameweek number
                - team_value: Current team value
                - bank: Money in the bank
        
        Returns:
            dict: Analysis results with insights
        """
        current_team = data.get('current_team', [])
        gameweek = data.get('gameweek', 1)
        team_value = data.get('team_value', 100.0)
        
        # Calculate some basic statistics
        total_value = sum(player.get('cost', 0) for player in current_team)
        avg_points = sum(player.get('points', 0) for player in current_team) / max(len(current_team), 1)
        
        # Identify potential issues
        insights = []
        
        # Check for injured players
        injured_count = sum(1 for player in current_team if player.get('status') == 'injured')
        if injured_count > 0:
            insights.append(f'{injured_count} player(s) injured - consider transfers')
        
        # Check for players with difficult fixtures
        difficult_fixtures = sum(1 for player in current_team if player.get('fixture_difficulty', 0) > 3)
        if difficult_fixtures > 5:
            insights.append('Many players have difficult fixtures ahead')
        
        # Check for high-performing players not in team
        # (This is simplified - you'd want to implement better logic)
        insights.append('Consider premium options if budget allows')
        
        return {
            'status': 'ok',
            'gameweek': gameweek,
            'message': f'Team analyzed for GW{gameweek}',
            'statistics': {
                'total_value': total_value,
                'avg_points': avg_points,
                'injured_count': injured_count
            },
            'insights': insights
        }
    
    def optimize(self, data):
        """
        Optimize team selection and make transfer decisions
        
        Parameters:
            data (dict): Contains:
                - current_team: Your current 15 players
                - available_players: ALL FPL players with stats (see note below!)
                - free_transfers: Number of free transfers available
                - bank: Money in the bank
                - gameweek: Current gameweek number
        
        IMPORTANT - NO DATA LEAKAGE:
            When predicting GW5, 'available_players' contains stats from GW1-4 ONLY!
            Each player includes:
                - id, name, position, cost, team
                - points_so_far: Cumulative points from completed GWs only
                - games_played: Games played so far
                - predicted_points: Average points from last 5 completed games
                - in_squad: True if player is in your current squad
        
        Returns:
            dict: Transfer decisions and team lineup
                - transfers: List of {out: player_id, in: player_id} (optional)
                - captain: Player ID for captain
                - vice_captain: Player ID for vice captain
                - lineup: List of 11 player IDs for starting lineup
                - bench: List of 4 player IDs for bench (in order) (optional)
        """
        current_team = data.get('current_team', [])
        available_players = data.get('available_players', [])
        free_transfers = data.get('free_transfers', 1)
        bank = data.get('bank', 0)
        gameweek = data.get('gameweek', 1)
        
        # Sort players by form (predicted_points based on past performance only!)
        sorted_players = sorted(
            available_players,
            key=lambda x: x.get('predicted_points', 0),
            reverse=True
        )
        
        # Select captain (highest predicted points in current team)
        current_team_sorted = sorted(
            current_team,
            key=lambda x: x.get('predicted_points', 0),
            reverse=True
        )
        
        captain_id = current_team_sorted[0]['id'] if current_team_sorted else None
        vice_captain_id = current_team_sorted[1]['id'] if len(current_team_sorted) > 1 else None
        
        # Simple transfer logic: replace lowest scorer if we have free transfers
        transfers = []
        if free_transfers > 0 and len(current_team) > 0 and available_players:
            # Find worst performing player in current team
            worst_player = min(current_team, key=lambda x: x.get('predicted_points', 0))
            
            # Find best available player not in squad that we can afford
            current_team_ids = [p['id'] for p in current_team]
            affordable_players = [
                p for p in sorted_players 
                if not p.get('in_squad', False)  # Not in our squad
                and p.get('position') == worst_player.get('position')  # Same position
                and p.get('cost', 0) <= worst_player.get('cost', 0) + bank  # Can afford
            ]
            
            if affordable_players:
                best_available = affordable_players[0]
                
                # Only make transfer if it's a significant upgrade (2+ points better)
                if best_available.get('predicted_points', 0) > worst_player.get('predicted_points', 0) + 2:
                    transfers.append({
                        'out': worst_player['id'],
                        'in': best_available['id'],
                        'reason': f"Upgrading {worst_player.get('name')} to {best_available.get('name')}"
                    })
        
        # Select best 11 for lineup (simplified - you should consider formation constraints)
        lineup_ids = [p['id'] for p in current_team_sorted[:11]]
        bench_ids = [p['id'] for p in current_team_sorted[11:15]]
        
        return {
            'transfers': transfers,
            'captain': captain_id,
            'vice_captain': vice_captain_id,
            'lineup': lineup_ids,
            'bench': bench_ids,
            'formation': '4-4-2',  # You should calculate this based on actual positions
            'confidence': 0.75  # Optional: your confidence in these decisions
        }


# Example of how to test your bot locally (won't run on platform)
if __name__ == '__main__':
    print("Testing FPL Bot...")
    bot = MyFPLBot()
    
    # Mock data for testing
    test_data = {
        'current_team': [
            {'id': 1, 'name': 'Player 1', 'cost': 10.0, 'points': 50},
            {'id': 2, 'name': 'Player 2', 'cost': 8.5, 'points': 45},
        ],
        'all_players': [
            {'id': 1, 'cost': 10.0},
            {'id': 2, 'cost': 8.5},
            {'id': 3, 'cost': 9.0},
        ],
        'gameweek': 10,
        'free_transfers': 1,
        'budget': 0.5,
        'predictions': {1: 6.5, 2: 5.5, 3: 7.0}
    }
    
    # Test analyze
    analysis = bot.analyze(test_data)
    print("\nAnalysis:")
    print(analysis)
    
    # Test optimize
    optimization = bot.optimize(test_data)
    print("\nOptimization:")
    print(optimization)
    
    print("\nBot test complete!")

