"""
Simple Example FPL Bot for FPL Vibes Competition
================================================

A well-commented example bot showing:
- Required method signatures
- Basic decision-making logic
- Strategic chip usage
- Position-based team selection
- Transfer logic

Use this as a starting template for your own bot!

Test locally with: python tools/bot_tester.py examples/simple_example_bot.py
"""


class SimpleBot:
    """
    A straightforward FPL bot that makes sensible decisions:
    - Selects captain based on predicted points
    - Builds a balanced lineup respecting positions
    - Makes smart transfers based on value
    - Uses chips strategically
    """
    
    def __init__(self):
        """Initialize the bot (optional)"""
        self.name = "Simple Bot"
        self.version = "1.0"
    
    def analyze(self, data):
        """
        Analyze the current team and game state.
        
        This method is called to provide insights about your team.
        Return a dictionary with status, message, and insights.
        
        Parameters:
            data (dict): Contains current_team, gameweek, chips_available, etc.
        
        Returns:
            dict: Analysis results
        """
        current_team = data.get('current_team', [])
        gameweek = data.get('gameweek', 1)
        chips_available = data.get('chips_available', [])
        bank = data.get('bank', 0)
        
        insights = []
        
        # Basic team info
        insights.append(f"Gameweek {gameweek} analysis")
        insights.append(f"Squad: {len(current_team)} players, £{bank:.1f}m in bank")
        
        # Analyze team value
        if current_team:
            total_cost = sum(p.get('cost', 0) for p in current_team)
            insights.append(f"Team value: £{total_cost:.1f}m")
            
            # Find top performers
            sorted_by_points = sorted(
                current_team,
                key=lambda x: x.get('predicted_points', 0),
                reverse=True
            )
            
            top_player = sorted_by_points[0]
            insights.append(f"Top performer: {top_player.get('name', 'Unknown')}")
        
        # Chip recommendations
        if chips_available:
            insights.append(f"Available chips: {', '.join(chips_available)}")
        
        return {
            'status': 'ok',
            'message': f'Team analyzed for GW{gameweek}',
            'insights': insights
        }
    
    def optimize(self, data):
        """
        Optimize team selection, transfers, and make strategic decisions.
        
        This is the main decision-making method. Return a dictionary with:
        - captain: Player ID for captain
        - vice_captain: Player ID for vice captain
        - lineup: List of 11 player IDs for starting lineup
        - transfers (optional): List of {out: player_id, in: player_id}
        - chip (optional): Chip to activate this week
        
        Parameters:
            data (dict): Contains:
                - current_team: Your 15 players
                - available_players: All players you can transfer in
                - gameweek: Current gameweek number
                - bank: Money in the bank (millions)
                - free_transfers: Number of free transfers
                - chips_available: Chips you can use
        
        Returns:
            dict: Optimization decisions
        """
        current_team = data.get('current_team', [])
        available_players = data.get('available_players', [])
        gameweek = data.get('gameweek', 1)
        bank = data.get('bank', 0)
        free_transfers = data.get('free_transfers', 1)
        chips_available = data.get('chips_available', [])
        
        # Safety check
        if not current_team:
            return {
                'captain': 1,
                'vice_captain': 2,
                'lineup': list(range(1, 12))
            }
        
        # Sort team by predicted points
        sorted_team = sorted(
            current_team,
            key=lambda x: x.get('predicted_points', 0),
            reverse=True
        )
        
        # === CAPTAIN SELECTION ===
        # Choose the player with highest predicted points
        captain_id = sorted_team[0]['id']
        vice_captain_id = sorted_team[1]['id'] if len(sorted_team) > 1 else captain_id
        
        # === LINEUP SELECTION ===
        # Build a valid formation (e.g., 1 GK, 4 DEF, 4 MID, 2 FWD)
        lineup_ids = self._select_lineup(current_team)
        
        # === TRANSFER LOGIC ===
        transfers = []
        
        # Only make transfers if we have free transfers and find good opportunities
        if free_transfers > 0 and available_players:
            # Find lowest performing player
            worst_player = sorted_team[-1]
            worst_points = worst_player.get('predicted_points', 0)
            
            # Find best available replacement in same position
            same_position = [
                p for p in available_players 
                if p.get('position') == worst_player.get('position')
                and p.get('cost', 100) <= worst_player.get('cost', 0) + bank
            ]
            
            if same_position:
                best_replacement = max(
                    same_position,
                    key=lambda x: x.get('predicted_points', 0)
                )
                
                # Make transfer if it's a significant upgrade
                if best_replacement.get('predicted_points', 0) > worst_points + 2:
                    transfers.append({
                        'out': worst_player['id'],
                        'in': best_replacement['id']
                    })
        
        # === CHIP STRATEGY ===
        chip = None
        
        # Triple Captain: Use on your best captain early-ish
        if gameweek == 8 and 'triple_captain' in chips_available:
            chip = 'triple_captain'
        
        # Wildcard: Use when you need a team overhaul (GW15 or second one after GW20)
        elif gameweek == 15 and 'wildcard_1' in chips_available:
            chip = 'wildcard'
        elif gameweek == 25 and 'wildcard_2' in chips_available:
            chip = 'wildcard'
        
        # Bench Boost: Use when you have a strong bench (around GW20)
        elif gameweek == 20 and 'bench_boost' in chips_available:
            # Check if bench is decent
            bench_players = sorted_team[11:15] if len(sorted_team) >= 15 else []
            bench_points = sum(p.get('predicted_points', 0) for p in bench_players)
            if bench_points > 15:  # At least 15 expected points from bench
                chip = 'bench_boost'
        
        # Free Hit: Use for tricky weeks or blank gameweeks (GW28)
        elif gameweek == 28 and 'free_hit' in chips_available:
            chip = 'free_hit'
        
        # === BUILD RESULT ===
        result = {
            'captain': captain_id,
            'vice_captain': vice_captain_id,
            'lineup': lineup_ids
        }
        
        # Add transfers if we're making any
        if transfers:
            result['transfers'] = transfers
        
        # Add chip if we're using one
        if chip:
            result['chip'] = chip
        
        return result
    
    def _select_lineup(self, current_team):
        """
        Select 11 players for the starting lineup.
        Ensures valid formation with 1 GK, 3+ DEF, 2+ MID, 1+ FWD.
        
        Parameters:
            current_team: List of 15 players
        
        Returns:
            List of 11 player IDs
        """
        # Group by position
        by_position = {
            'GK': [],
            'DEF': [],
            'MID': [],
            'FWD': []
        }
        
        for player in current_team:
            pos = player.get('position', 'MID')
            by_position[pos].append(player)
        
        # Sort each position by predicted points
        for pos in by_position:
            by_position[pos].sort(
                key=lambda x: x.get('predicted_points', 0),
                reverse=True
            )
        
        # Build lineup: 1 GK, 4 DEF, 4 MID, 2 FWD (442 formation)
        lineup = []
        
        # Add 1 goalkeeper
        if by_position['GK']:
            lineup.append(by_position['GK'][0]['id'])
        
        # Add 4 defenders (or as many as we have)
        lineup.extend([p['id'] for p in by_position['DEF'][:4]])
        
        # Add 4 midfielders (or as many as we have)
        lineup.extend([p['id'] for p in by_position['MID'][:4]])
        
        # Add 2 forwards (or as many as we have)
        lineup.extend([p['id'] for p in by_position['FWD'][:2]])
        
        # If we don't have 11, fill with remaining best players
        if len(lineup) < 11:
            all_sorted = sorted(
                current_team,
                key=lambda x: x.get('predicted_points', 0),
                reverse=True
            )
            for player in all_sorted:
                if player['id'] not in lineup:
                    lineup.append(player['id'])
                    if len(lineup) == 11:
                        break
        
        return lineup[:11]  # Ensure exactly 11 players

