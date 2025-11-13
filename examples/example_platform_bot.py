"""
Platform-Inspired FPL Bot
A reference implementation demonstrating all bot features including chip usage

This bot uses form-based analysis and strategic chip timing to maximize points.
"""


class PlatformBot:
    """
    A sophisticated FPL bot that demonstrates:
    - Form-based player analysis
    - Strategic captaincy selection
    - Intelligent lineup optimization
    - Smart chip usage timing
    - Transfer management
    """
    
    def __init__(self):
        self.name = "Platform Reference Bot"
        self.version = "1.0"
    
    def analyze(self, data):
        """
        Analyze the current team and game state
        
        Returns insights about team performance, chip availability, and recommendations
        """
        current_team = data.get('current_team', [])
        gameweek = data.get('gameweek', 1)
        chips_available = data.get('chips_available', [])
        bank = data.get('bank', 0.0)
        free_transfers = data.get('free_transfers', 1)
        
        insights = []
        
        # Team value analysis
        if current_team:
            total_value = sum(p.get('cost', 0) for p in current_team)
            avg_predicted = sum(p.get('predicted_points', 0) for p in current_team) / len(current_team)
            
            insights.append(f"Team value: £{total_value:.1f}m (Bank: £{bank:.1f}m)")
            insights.append(f"Average predicted points: {avg_predicted:.1f}")
        
        # Top performers
        if current_team:
            sorted_by_form = sorted(
                current_team, 
                key=lambda x: x.get('predicted_points', 0),
                reverse=True
            )
            top_3 = sorted_by_form[:3]
            insights.append(f"Top performers: {', '.join(p['name'] for p in top_3)}")
        
        # Transfer opportunity analysis
        if free_transfers > 0:
            insights.append(f"Free transfers available: {free_transfers}")
        
        # Chip recommendations
        if chips_available:
            chip_count = len(chips_available)
            insights.append(f"Chips available: {chip_count} ({', '.join(chips_available)})")
            
            # Strategic chip timing recommendations
            if 'triple_captain' in chips_available and gameweek >= 3:
                insights.append("Consider Triple Captain if your captain has great fixtures")
            
            if 'wildcard_1' in chips_available and gameweek >= 5:
                insights.append("Wildcard available for team overhaul")
            
            if 'bench_boost' in chips_available and gameweek >= 8:
                insights.append("Bench Boost ready - best used with strong bench")
            
            if 'free_hit' in chips_available and gameweek >= 12:
                insights.append("Free Hit available for difficult fixture weeks")
        
        return {
            'status': 'ok',
            'message': f'Analysis complete for GW{gameweek}',
            'insights': insights
        }
    
    def optimize(self, data):
        """
        Optimize team selection, captaincy, and chip usage
        
        Uses form-based analysis to make strategic decisions
        """
        current_team = data.get('current_team', [])
        available_players = data.get('available_players', [])
        gameweek = data.get('gameweek', 1)
        bank = data.get('bank', 0.0)
        free_transfers = data.get('free_transfers', 1)
        chips_available = data.get('chips_available', [])
        
        if not current_team:
            # Fallback for empty team
            return self._default_selection()
        
        # Sort team by predicted points (form)
        sorted_team = sorted(
            current_team,
            key=lambda x: x.get('predicted_points', 0),
            reverse=True
        )
        
        # === CAPTAINCY SELECTION ===
        # Select captain and vice captain (top 2 by predicted points)
        captain_id = sorted_team[0]['id']
        vice_captain_id = sorted_team[1]['id'] if len(sorted_team) > 1 else captain_id
        
        # === LINEUP OPTIMIZATION ===
        # Build optimal lineup respecting formation rules
        lineup_ids = self._build_optimal_lineup(sorted_team)
        
        # === CHIP STRATEGY ===
        chip_to_use = self._evaluate_chip_usage(
            gameweek, 
            chips_available, 
            sorted_team,
            current_team,
            available_players,
            bank
        )
        
        # Build result
        result = {
            'captain': captain_id,
            'vice_captain': vice_captain_id,
            'lineup': lineup_ids
        }
        
        # Add chip if using one
        if chip_to_use:
            result['chip'] = chip_to_use
        
        return result
    
    def _build_optimal_lineup(self, sorted_team):
        """
        Build optimal 11-player lineup respecting formation rules
        
        Formation rules:
        - Exactly 1 GKP
        - 3-5 DEF
        - 2-5 MID  
        - 1-3 FWD
        - Total = 11 players
        
        Strategy: Pick best players by predicted points within position constraints
        """
        lineup = []
        position_counts = {'GKP': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        
        # First pass: Add best goalkeeper
        for player in sorted_team:
            if player['position'] == 'GKP' and position_counts['GKP'] < 1:
                lineup.append(player['id'])
                position_counts['GKP'] += 1
                break
        
        # Second pass: Add best outfield players respecting constraints
        # Target formation: 1-4-4-2 (balanced)
        target_formation = {'GKP': 1, 'DEF': 4, 'MID': 4, 'FWD': 2}
        
        for player in sorted_team:
            if len(lineup) >= 11:
                break
            
            pos = player['position']
            
            # Skip if already selected
            if player['id'] in lineup:
                continue
            
            # Check if we can add this position
            if position_counts[pos] < target_formation[pos]:
                lineup.append(player['id'])
                position_counts[pos] += 1
        
        # If we don't have 11 yet, add remaining best players within max limits
        max_limits = {'GKP': 1, 'DEF': 5, 'MID': 5, 'FWD': 3}
        
        for player in sorted_team:
            if len(lineup) >= 11:
                break
            
            if player['id'] in lineup:
                continue
            
            pos = player['position']
            if position_counts[pos] < max_limits[pos]:
                lineup.append(player['id'])
                position_counts[pos] += 1
        
        # Ensure we have minimum requirements
        min_requirements = {'GKP': 1, 'DEF': 3, 'MID': 2, 'FWD': 1}
        
        # If we don't meet minimums, force add players
        for player in sorted_team:
            if player['id'] in lineup:
                continue
            
            pos = player['position']
            if position_counts[pos] < min_requirements[pos]:
                # Remove a player from an over-represented position if needed
                if len(lineup) >= 11:
                    # Find position with most players above minimum
                    for remove_pos in ['MID', 'DEF', 'FWD']:
                        if position_counts[remove_pos] > min_requirements[remove_pos]:
                            # Remove worst player from this position
                            for pid in lineup[::-1]:  # Reverse to remove worst first
                                p = next((x for x in sorted_team if x['id'] == pid), None)
                                if p and p['position'] == remove_pos:
                                    lineup.remove(pid)
                                    position_counts[remove_pos] -= 1
                                    break
                            break
                
                lineup.append(player['id'])
                position_counts[pos] += 1
        
        return lineup[:11]  # Ensure exactly 11
    
    def _evaluate_chip_usage(self, gameweek, chips_available, sorted_team, 
                            current_team, available_players, bank):
        """
        Evaluate whether to use a chip this gameweek
        
        Strategic chip timing:
        - Triple Captain: Early (GW3-5) on premium captain with good fixtures
        - Wildcard: Mid-season (GW8-12) for team rebuild
        - Bench Boost: When bench is strong (GW15-20)
        - Free Hit: Difficult fixture weeks (GW25+)
        """
        
        if not chips_available:
            return None
        
        # === TRIPLE CAPTAIN STRATEGY ===
        # Use early on your best player
        if 'triple_captain' in chips_available:
            # Use in GW4-6 when captain is established
            if 4 <= gameweek <= 6:
                # Check if captain is premium (predicted points > 7)
                if sorted_team and sorted_team[0].get('predicted_points', 0) > 7.0:
                    return 'triple_captain'
        
        # === WILDCARD STRATEGY ===
        # Use for major team overhaul
        if 'wildcard_1' in chips_available or 'wildcard_2' in chips_available:
            # Early wildcard (GW8-12) or late wildcard (GW25-30)
            if 8 <= gameweek <= 12 or 25 <= gameweek <= 30:
                # Check if team needs overhaul (average predicted points < 5.5)
                if current_team:
                    avg_predicted = sum(p.get('predicted_points', 0) for p in current_team) / len(current_team)
                    if avg_predicted < 5.5:
                        return 'wildcard'
            
            # Always use wildcard if we're very late in season and it's still available
            if gameweek >= 35 and 'wildcard_2' in chips_available:
                return 'wildcard'
        
        # === BENCH BOOST STRATEGY ===
        # Use when bench is strong
        if 'bench_boost' in chips_available:
            # Use in GW15-25 (mid-season, potentially double gameweeks)
            if 15 <= gameweek <= 25:
                # Calculate bench strength
                lineup_ids = self._build_optimal_lineup(sorted_team)
                bench_players = [p for p in current_team if p['id'] not in lineup_ids]
                
                if bench_players:
                    bench_strength = sum(p.get('predicted_points', 0) for p in bench_players)
                    
                    # Use if bench averages > 4 points per player
                    if bench_strength / len(bench_players) > 4.0:
                        return 'bench_boost'
        
        # === FREE HIT STRATEGY ===
        # Use for difficult fixture weeks
        if 'free_hit' in chips_available:
            # Use late season (GW28+) when you need temporary changes
            if gameweek >= 28:
                # Check if team has low predicted points (bad fixtures)
                if current_team:
                    avg_predicted = sum(p.get('predicted_points', 0) for p in current_team) / len(current_team)
                    if avg_predicted < 5.0:
                        return 'free_hit'
        
        return None
    
    def _default_selection(self):
        """Fallback selection when team is empty"""
        return {
            'captain': 1,
            'vice_captain': 2,
            'lineup': list(range(1, 12))
        }

