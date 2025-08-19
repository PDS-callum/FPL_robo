import pandas as pd
import numpy as np
from itertools import combinations
from .constants import POSITION_MAP, BUDGET_LIMIT, TEAM_SIZE, PLAYING_TEAM_SIZE

class FPLTeamOptimizer:
    def __init__(self, total_budget=100.0):
        """
        Initialize FPL Team Optimizer with budget constraints
        
        Parameters:
        -----------
        total_budget : float
            Total budget available (default 100.0 million)
        """
        self.total_budget = total_budget
        self.formation_constraints = {
            'GK': {'min': 2, 'max': 2},  # Exactly 2 goalkeepers
            'DEF': {'min': 5, 'max': 5},  # Exactly 5 defenders
            'MID': {'min': 5, 'max': 5},  # Exactly 5 midfielders
            'FWD': {'min': 3, 'max': 3}   # Exactly 3 forwards
        }
        self.max_players_per_team = 3
        
    def optimize_team(self, players_df, predictions_df, budget=None):
        """
        Optimize team selection based on predictions and FPL constraints
        
        Parameters:
        -----------
        players_df : pd.DataFrame
            Player information including price, position, team
        predictions_df : pd.DataFrame
            Player predictions (player_id, predicted_points)
        budget : float, optional
            Override default budget
            
        Returns:
        --------
        selected_team : pd.DataFrame
            Optimized team of 15 players
        """
        if budget is None:
            budget = self.total_budget
            
        # Merge player info with predictions
        team_data = players_df.merge(predictions_df, on='id', how='inner')
        
        print(f"🔍 Debug: Merged {len(team_data)} players from {len(players_df)} available")
        
        # Convert position codes to names if needed
        if 'element_type' in team_data.columns:
            team_data['position'] = team_data['element_type'].map(POSITION_MAP)
            # Remove players with invalid positions
            team_data = team_data.dropna(subset=['position'])
        elif 'position' not in team_data.columns:
            raise ValueError("Position information not found in player data")
        
        # Ensure position column has no NaN values
        team_data = team_data.dropna(subset=['position'])
        
        print(f"🔍 Debug: After position filtering, {len(team_data)} players remain")
        
        # Filter out players with no predictions or invalid data
        print(f"🔍 Debug: Predictions range: {team_data['predicted_points'].min():.3f} to {team_data['predicted_points'].max():.3f}")
        print(f"🔍 Debug: NaN predictions: {team_data['predicted_points'].isna().sum()}")
        print(f"🔍 Debug: Zero predictions: {(team_data['predicted_points'] == 0).sum()}")
        
        team_data = team_data.dropna(subset=['predicted_points', 'now_cost'])
        
        # Normalize predictions to be positive by adding offset
        min_pred = team_data['predicted_points'].min()
        if min_pred < 0:
            print(f"🔍 Debug: Adding offset of {-min_pred + 0.1:.3f} to make all predictions positive")
            team_data['predicted_points'] = team_data['predicted_points'] - min_pred + 0.1
        
        print(f"🔍 Debug: After filtering, {len(team_data)} players remain")
        if len(team_data) > 0:
            print(f"🔍 Debug: Price range: £{team_data['now_cost'].min()/10:.1f}m - £{team_data['now_cost'].max()/10:.1f}m")
            print(f"🔍 Debug: Position counts: {team_data['position'].value_counts().to_dict()}")
        
        # Convert price to millions (FPL API gives prices in tenths of millions)
        team_data['price'] = team_data['now_cost'] / 10.0
        
        # Calculate value (points per million)
        team_data['value'] = team_data['predicted_points'] / team_data['price']
        
        print(f"🔍 Debug: Value range: {team_data['value'].min():.3f} to {team_data['value'].max():.3f}")
        print(f"🔍 Debug: Sample values:")
        print(team_data[['web_name', 'position', 'price', 'predicted_points', 'value']].head(10))
        
        # Use strict greedy algorithm for team selection
        selected_team = self._greedy_team_selection(team_data, budget)
        
        # Final validation before returning
        if len(selected_team) == 0:
            print("❌ CRITICAL ERROR: Team selection completely failed")
            print("📋 Available players by position:")
            if len(team_data) > 0:
                pos_counts = team_data['position'].value_counts()
                for pos, count in pos_counts.items():
                    min_price = team_data[team_data['position'] == pos]['price'].min()
                    print(f"   {pos}: {count} players (cheapest: £{min_price:.1f}m)")
            return selected_team
        
        # Validate the selected team
        is_valid, errors = self.validate_team(selected_team)
        
        if not is_valid:
            print("❌ TEAM VALIDATION FAILED:")
            for error in errors:
                print(f"   - {error}")
            print("🔄 Attempting to fix team...")
            
            # Try multiple fixing strategies
            fixed_team = self._comprehensive_team_fix(selected_team, team_data, budget)
            if len(fixed_team) > 0:
                is_valid_fixed, fix_errors = self.validate_team(fixed_team)
                if is_valid_fixed:
                    print("✅ Team successfully fixed!")
                    return fixed_team
                else:
                    print("❌ Team fix failed, remaining errors:")
                    for error in fix_errors:
                        print(f"   - {error}")
            
            # Last resort: try emergency selection with strict constraints
            print("🚨 Attempting emergency team selection...")
            emergency_team = self._strict_emergency_selection(team_data, budget)
            if len(emergency_team) > 0:
                is_emergency_valid, _ = self.validate_team(emergency_team)
                if is_emergency_valid:
                    print("✅ Emergency team selection successful!")
                    return emergency_team
            
            print("❌ CRITICAL: Could not create any valid team with current constraints")
            print("❌ This indicates insufficient player data or impossible budget constraints")
            return pd.DataFrame()  # Return empty to signal complete failure
        
        print("✅ Team validation passed!")
        return selected_team
    
    def _greedy_team_selection(self, players_df, budget):
        """
        Strict greedy algorithm for team selection with FPL constraints
        Uses position-by-position selection to guarantee valid team
        """
        selected_players = []
        remaining_budget = budget
        position_counts = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        team_counts = {}
        
        print(f"🔍 Debug: Starting team selection with budget £{budget:.1f}m")
        print(f"🔍 Debug: Player pool: {len(players_df)} players")
        print(f"🔍 Debug: Required formation: {self.formation_constraints}")
        
        # Check if we have enough players in each position
        position_availability = players_df['position'].value_counts()
        print(f"🔍 Debug: Position availability: {dict(position_availability)}")
        
        for position, constraints in self.formation_constraints.items():
            available_count = position_availability.get(position, 0)
            required_count = constraints['max']
            if available_count < required_count:
                print(f"❌ CRITICAL: Only {available_count} {position} players available, need {required_count}")
                return pd.DataFrame()  # Cannot build valid team
        
        # Get player IDs already selected to avoid duplicates
        selected_ids = set()
        
        # Select players position by position to ensure exact requirements
        for position, constraints in self.formation_constraints.items():
            needed = constraints['max']
            print(f"\n🎯 Selecting {needed} {position} players...")
            
            # Get available players for this position
            available = players_df[
                (players_df['position'] == position) &
                (~players_df['id'].isin(selected_ids))
            ].copy()
            
            if len(available) < needed:
                print(f"❌ ERROR: Only {len(available)} {position} players available, need {needed}")
                # Try to continue with what we have
                needed = len(available)
            
            # Sort by value for this position
            available = available.sort_values('value', ascending=False)
            
            # Select players for this position using a more careful approach
            position_selected = 0
            
            # First try: select best value players that fit constraints
            for _, player in available.iterrows():
                if position_selected >= needed:
                    break
                    
                team_id = player['team']
                price = player['price']
                
                # Check all constraints
                if (price <= remaining_budget and
                    team_counts.get(team_id, 0) < self.max_players_per_team):
                    
                    selected_players.append(player)
                    selected_ids.add(player['id'])
                    remaining_budget -= price
                    position_counts[position] += 1
                    team_counts[team_id] = team_counts.get(team_id, 0) + 1
                    position_selected += 1
                    
                    print(f"✅ Selected {player['web_name']} - £{price:.1f}m (Budget: £{remaining_budget:.1f}m)")
            
            # If we couldn't select enough players, try with cheapest available
            if position_selected < needed:
                print(f"⚠️ Only selected {position_selected}/{needed} {position}, trying cheaper options...")
                
                # Sort by price for fallback
                available_cheap = available[
                    ~available['id'].isin(selected_ids)
                ].sort_values('price')
                
                for _, player in available_cheap.iterrows():
                    if position_selected >= needed:
                        break
                        
                    team_id = player['team']
                    price = player['price']
                    
                    # More lenient team constraint for completion
                    team_constraint_ok = team_counts.get(team_id, 0) < self.max_players_per_team
                    
                    if price <= remaining_budget and team_constraint_ok:
                        selected_players.append(player)
                        selected_ids.add(player['id'])
                        remaining_budget -= price
                        position_counts[position] += 1
                        team_counts[team_id] = team_counts.get(team_id, 0) + 1
                        position_selected += 1
                        
                        print(f"✅ Fallback selected {player['web_name']} - £{price:.1f}m")
            
            print(f"📊 {position} selection complete: {position_selected}/{needed} players selected")
        
        result_df = pd.DataFrame(selected_players)
        
        # Final validation
        total_players = len(result_df)
        total_cost = result_df['price'].sum() if len(result_df) > 0 else 0
        
        print(f"\n📋 Team Selection Summary:")
        print(f"   Total players: {total_players}/15")
        print(f"   Total cost: £{total_cost:.1f}m/£{budget:.1f}m")
        print(f"   Position breakdown: {dict(result_df['position'].value_counts()) if len(result_df) > 0 else {}}")
        
        # Strict validation - reject invalid teams and try emergency selection if needed
        if total_players != 15:
            print(f"❌ INVALID TEAM: Expected 15 players, got {total_players}")
            return self._strict_emergency_selection(players_df, budget)
        
        if total_cost > budget + 0.1:  # Small tolerance for rounding
            print(f"❌ INVALID TEAM: Cost £{total_cost:.1f}m exceeds budget £{budget:.1f}m")
            return self._strict_emergency_selection(players_df, budget)
        
        # Check position requirements
        actual_positions = dict(result_df['position'].value_counts())
        for position, constraints in self.formation_constraints.items():
            actual_count = actual_positions.get(position, 0)
            if actual_count != constraints['max']:
                print(f"❌ INVALID TEAM: {position} has {actual_count} players, need {constraints['max']}")
                return self._strict_emergency_selection(players_df, budget)
        
        # Check team constraint
        team_counts_final = result_df['team'].value_counts()
        violating_teams = team_counts_final[team_counts_final > self.max_players_per_team]
        if len(violating_teams) > 0:
            print(f"❌ INVALID TEAM: Teams with too many players: {dict(violating_teams)}")
            return self._strict_emergency_selection(players_df, budget)
        
        print("✅ Team selection successful and valid!")
        return result_df

    def select_playing_xi(self, full_team):
        """
        Select playing XI from the 15-player squad based on formation and predictions
        
        Parameters:
        -----------
        full_team : pd.DataFrame
            Full team of 15 players
            
        Returns:
        --------
        playing_xi : pd.DataFrame
            11 players for starting lineup
        captain : pd.Series
            Captain selection
        vice_captain : pd.Series
            Vice-captain selection
        formation : dict
            Formation used (e.g., {'DEF': 4, 'MID': 4, 'FWD': 2})
        """
        # Sort by predicted points within each position
        team_sorted = full_team.sort_values(['position', 'predicted_points'], ascending=[True, False])
        
        # Always select the best goalkeeper
        gk = team_sorted[team_sorted['position'] == 'GK'].iloc[0]
        
        # Select outfield players based on best formation
        def_players = team_sorted[team_sorted['position'] == 'DEF'].iloc[:5]
        mid_players = team_sorted[team_sorted['position'] == 'MID'].iloc[:5]
        fwd_players = team_sorted[team_sorted['position'] == 'FWD'].iloc[:3]
        
        # Try different formations and pick the best one
        formations = [
            {'DEF': 3, 'MID': 5, 'FWD': 2},  # 3-5-2
            {'DEF': 3, 'MID': 4, 'FWD': 3},  # 3-4-3
            {'DEF': 4, 'MID': 5, 'FWD': 1},  # 4-5-1
            {'DEF': 4, 'MID': 4, 'FWD': 2},  # 4-4-2
            {'DEF': 4, 'MID': 3, 'FWD': 3},  # 4-3-3
            {'DEF': 5, 'MID': 4, 'FWD': 1},  # 5-4-1
            {'DEF': 5, 'MID': 3, 'FWD': 2},  # 5-3-2
        ]
        
        best_formation = None
        best_total_points = -1
        best_xi = None
        
        for formation in formations:
            try:
                # Select players for this formation
                selected_def = def_players.head(formation['DEF'])
                selected_mid = mid_players.head(formation['MID'])
                selected_fwd = fwd_players.head(formation['FWD'])
                
                # Calculate total predicted points
                xi_players = pd.concat([pd.DataFrame([gk]), selected_def, selected_mid, selected_fwd])
                total_points = xi_players['predicted_points'].sum()
                
                if total_points > best_total_points:
                    best_total_points = total_points
                    best_formation = formation
                    best_xi = xi_players
                    
            except Exception:
                continue
        
        if best_xi is None:
            # Fallback: just select top 11 players
            best_xi = team_sorted.head(11)
            best_formation = {'DEF': 4, 'MID': 4, 'FWD': 2}
        
        # Select captain and vice-captain (highest predicted points)
        captain = best_xi.nlargest(1, 'predicted_points').iloc[0]
        vice_captain = best_xi[best_xi['id'] != captain['id']].nlargest(1, 'predicted_points').iloc[0]
        
        # Mark captain and vice-captain
        best_xi = best_xi.copy()
        best_xi['is_captain'] = best_xi['id'] == captain['id']
        best_xi['is_vice_captain'] = best_xi['id'] == vice_captain['id']
        
        return best_xi, captain, vice_captain, best_formation
    
    def validate_team(self, team_df):
        """
        Validate team against FPL constraints with detailed reporting
        
        Returns:
        --------
        is_valid : bool
            Whether team meets all constraints
        errors : list
            List of constraint violations
        """
        errors = []
        
        if len(team_df) == 0:
            errors.append("Team is empty")
            return False, errors
        
        # Check team size
        team_size = len(team_df)
        if team_size != 15:
            errors.append(f"Team must have exactly 15 players, got {team_size}")
        
        # Check position constraints
        if 'position' in team_df.columns:
            position_counts = team_df['position'].value_counts()
            for pos, constraints in self.formation_constraints.items():
                count = position_counts.get(pos, 0)
                expected = constraints['max']
                if count != expected:
                    errors.append(f"Need exactly {expected} {pos}, got {count}")
            
            # Special check for goalkeepers (critical for FPL)
            gk_count = position_counts.get('GK', 0)
            if gk_count < 1:
                errors.append(f"CRITICAL: Team has no goalkeepers! Need exactly 2 GK, got {gk_count}")
            elif gk_count != 2:
                errors.append(f"CRITICAL: Team has {gk_count} goalkeepers! Need exactly 2 GK")
        else:
            errors.append("No position information available")
        
        # Check budget constraint
        if 'price' in team_df.columns:
            total_cost = team_df['price'].sum()
        elif 'now_cost' in team_df.columns:
            total_cost = team_df['now_cost'].sum() / 10
        else:
            errors.append("No price information available")
            total_cost = 0
        
        if total_cost > self.total_budget + 0.1:  # Small tolerance for rounding
            errors.append(f"Team cost £{total_cost:.1f}m exceeds budget £{self.total_budget:.1f}m")
        
        # Check max players per team constraint
        if 'team' in team_df.columns:
            team_counts = team_df['team'].value_counts()
            violating_teams = team_counts[team_counts > self.max_players_per_team]
            if len(violating_teams) > 0:
                for team_id, count in violating_teams.items():
                    errors.append(f"Team {team_id} has {count} players (max {self.max_players_per_team})")
        
        # Check for duplicate players
        if 'id' in team_df.columns:
            duplicate_count = len(team_df) - len(team_df['id'].unique())
            if duplicate_count > 0:
                errors.append(f"Team contains {duplicate_count} duplicate players")
        
        return len(errors) == 0, errors
    
    def _comprehensive_team_fix(self, selected_team, all_players, budget):
        """
        Comprehensive attempt to fix an invalid team by addressing all constraint violations
        """
        if len(selected_team) == 0:
            return pd.DataFrame()
        
        print("🔧 Starting comprehensive team fix...")
        working_team = selected_team.copy()
        
        # Step 1: Fix team size
        if len(working_team) != 15:
            print(f"🔧 Fixing team size: {len(working_team)} → 15 players")
            if len(working_team) > 15:
                # Remove worst value players
                working_team = working_team.nlargest(15, 'value')
            elif len(working_team) < 15:
                # Add cheapest players by position to complete the team
                working_team = self._complete_team_to_15(working_team, all_players, budget)
        
        # Step 2: Fix position constraints
        working_team = self._fix_position_constraints(working_team, all_players, budget)
        
        # Step 3: Fix budget constraint
        working_team = self._fix_budget_constraint(working_team, all_players, budget)
        
        # Step 4: Fix team constraint (max 3 per team)
        working_team = self._fix_team_constraint(working_team, all_players, budget)
        
        # Step 5: Remove any duplicates
        if 'id' in working_team.columns:
            working_team = working_team.drop_duplicates(subset=['id'])
        
        # Final validation
        if len(working_team) == 15:
            is_valid, errors = self.validate_team(working_team)
            if is_valid:
                print("✅ Comprehensive team fix successful!")
                return working_team
            else:
                print(f"❌ Comprehensive fix failed, remaining errors: {errors}")
        
        return pd.DataFrame()
    
    def _complete_team_to_15(self, current_team, all_players, budget):
        """Complete team to exactly 15 players by adding cheapest valid options"""
        if len(current_team) >= 15:
            return current_team
        
        remaining_budget = budget - current_team['price'].sum()
        selected_ids = set(current_team['id'])
        current_positions = current_team['position'].value_counts()
        
        # Determine what positions we still need
        needed_positions = []
        for pos, constraints in self.formation_constraints.items():
            current_count = current_positions.get(pos, 0)
            needed = constraints['max'] - current_count
            needed_positions.extend([pos] * needed)
        
        # Add cheapest players for missing positions
        for pos in needed_positions:
            if len(current_team) >= 15:
                break
                
            available = all_players[
                (all_players['position'] == pos) &
                (~all_players['id'].isin(selected_ids)) &
                (all_players['price'] <= remaining_budget)
            ].sort_values('price')
            
            if len(available) > 0:
                player = available.iloc[0]
                current_team = pd.concat([current_team, pd.DataFrame([player])], ignore_index=True)
                selected_ids.add(player['id'])
                remaining_budget -= player['price']
        
        return current_team
    
    def _fix_position_constraints(self, team, all_players, budget):
        """Fix position constraint violations"""
        position_counts = team['position'].value_counts()
        
        for pos, constraints in self.formation_constraints.items():
            current_count = position_counts.get(pos, 0)
            required_count = constraints['max']
            
            if current_count != required_count:
                print(f"🔧 Fixing {pos} count: {current_count} → {required_count}")
                
                if current_count > required_count:
                    # Remove excess players (worst value first)
                    pos_players = team[team['position'] == pos].sort_values('value')
                    to_remove = current_count - required_count
                    remove_ids = pos_players.head(to_remove)['id'].tolist()
                    team = team[~team['id'].isin(remove_ids)]
                
                elif current_count < required_count:
                    # Add players for this position
                    needed = required_count - current_count
                    selected_ids = set(team['id'])
                    remaining_budget = budget - team['price'].sum()
                    
                    available = all_players[
                        (all_players['position'] == pos) &
                        (~all_players['id'].isin(selected_ids)) &
                        (all_players['price'] <= remaining_budget)
                    ].sort_values('value', ascending=False)
                    
                    for i in range(min(needed, len(available))):
                        player = available.iloc[i]
                        team = pd.concat([team, pd.DataFrame([player])], ignore_index=True)
                        remaining_budget -= player['price']
        
        return team
    
    def _fix_budget_constraint(self, team, all_players, budget):
        """Fix budget constraint violations"""
        total_cost = team['price'].sum()
        
        if total_cost > budget:
            print(f"🔧 Fixing budget: £{total_cost:.1f}m → £{budget:.1f}m")
            overspend = total_cost - budget
            
            # Sort by worst value (lowest points per pound)
            team_sorted = team.sort_values('value')
            
            for _, expensive_player in team_sorted.iterrows():
                if total_cost <= budget:
                    break
                
                # Find cheaper replacement in same position
                position = expensive_player['position']
                max_price = expensive_player['price'] - 0.1  # Must be cheaper
                
                replacements = all_players[
                    (all_players['position'] == position) &
                    (all_players['price'] <= max_price) &
                    (~all_players['id'].isin(team['id']))
                ].sort_values('value', ascending=False)
                
                if len(replacements) > 0:
                    replacement = replacements.iloc[0]
                    
                    # Make the replacement
                    team = team[team['id'] != expensive_player['id']]
                    team = pd.concat([team, pd.DataFrame([replacement])], ignore_index=True)
                    
                    price_diff = expensive_player['price'] - replacement['price']
                    total_cost -= price_diff
                    
                    print(f"🔄 Replaced {expensive_player['web_name']} with {replacement['web_name']} (saved £{price_diff:.1f}m)")
        
        return team
    
    def _fix_team_constraint(self, team, all_players, budget):
        """Fix max players per team constraint violations"""
        team_counts = team['team'].value_counts()
        violating_teams = team_counts[team_counts > self.max_players_per_team]
        
        for team_id, count in violating_teams.items():
            print(f"🔧 Fixing team constraint: Team {team_id} has {count} players (max {self.max_players_per_team})")
            
            # Get players from this team, sorted by worst value
            team_players = team[team['team'] == team_id].sort_values('value')
            excess_count = count - self.max_players_per_team
            
            # Remove excess players and try to replace them
            for i in range(excess_count):
                if i >= len(team_players):
                    break
                    
                player_to_remove = team_players.iloc[i]
                position = player_to_remove['position']
                max_price = player_to_remove['price'] + 0.5  # Allow slightly more expensive replacement
                
                # Find replacement from different team
                replacements = all_players[
                    (all_players['position'] == position) &
                    (all_players['team'] != team_id) &
                    (all_players['price'] <= max_price) &
                    (~all_players['id'].isin(team['id']))
                ]
                
                # Check that replacement team won't violate constraint
                current_team_counts = team['team'].value_counts()
                valid_replacements = []
                
                for _, replacement in replacements.iterrows():
                    repl_team_id = replacement['team']
                    if current_team_counts.get(repl_team_id, 0) < self.max_players_per_team:
                        valid_replacements.append(replacement)
                
                if valid_replacements:
                    # Choose best value replacement
                    replacement = max(valid_replacements, key=lambda x: x['value'])
                    
                    # Make the replacement
                    team = team[team['id'] != player_to_remove['id']]
                    team = pd.concat([team, pd.DataFrame([replacement])], ignore_index=True)
                    
                    print(f"🔄 Replaced {player_to_remove['web_name']} with {replacement['web_name']}")
                else:
                    # Just remove the player if no replacement found
                    team = team[team['id'] != player_to_remove['id']]
                    print(f"❌ Removed {player_to_remove['web_name']} (no valid replacement)")
        
        return team
    
    def _strict_emergency_selection(self, players_df, budget):
        """
        Ultra-strict emergency team selection that guarantees FPL constraints
        Uses cheapest valid players for each position
        """
        print("🚨 Starting strict emergency team selection...")
        
        selected_players = []
        remaining_budget = budget
        selected_ids = set()
        team_counts = {}
        
        # Sort players by position and price to ensure we can build a valid team
        for position, constraints in self.formation_constraints.items():
            needed = constraints['max']
            print(f"🎯 Selecting {needed} {position} players...")
            
            # Get all available players for this position, sorted by price
            available_players = players_df[
                (players_df['position'] == position) &
                (~players_df['id'].isin(selected_ids))
            ].sort_values('price').copy()
            
            selected_for_position = 0
            
            for _, player in available_players.iterrows():
                if selected_for_position >= needed:
                    break
                
                player_price = player['price']
                player_team = player['team']
                
                # Check all constraints strictly
                budget_ok = player_price <= remaining_budget
                team_ok = team_counts.get(player_team, 0) < self.max_players_per_team
                
                if budget_ok and team_ok:
                    selected_players.append(player.to_dict())
                    selected_ids.add(player['id'])
                    remaining_budget -= player_price
                    team_counts[player_team] = team_counts.get(player_team, 0) + 1
                    selected_for_position += 1
                    
                    print(f"✅ Selected {player['web_name']} - £{player_price:.1f}m (Budget: £{remaining_budget:.1f}m)")
            
            # Check if we got enough players for this position
            if selected_for_position < needed:
                print(f"❌ Could not select enough {position} players: {selected_for_position}/{needed}")
                print(f"❌ This usually means budget is too low or not enough valid players")
                return pd.DataFrame()
        
        result_df = pd.DataFrame(selected_players)
        
        # Final validation
        if len(result_df) == 15:
            is_valid, errors = self.validate_team(result_df)
            if is_valid:
                print("✅ Strict emergency selection successful!")
                return result_df
            else:
                print(f"❌ Even strict selection failed validation: {errors}")
        else:
            print(f"❌ Strict selection got {len(result_df)} players instead of 15")
        
        return pd.DataFrame()

    def _attempt_team_fix(self, selected_team, all_players, budget):
        """
        Attempt to fix an invalid team by making minimal changes
        """
        if len(selected_team) == 0:
            return pd.DataFrame()
        
        # Check budget issue first
        total_cost = selected_team['price'].sum()
        if total_cost > budget:
            print(f"🔧 Fixing budget: £{total_cost:.1f}m > £{budget:.1f}m")
            
            # Sort by value (worst value first) to replace
            team_sorted = selected_team.sort_values('value')
            
            for _, expensive_player in team_sorted.iterrows():
                if total_cost <= budget:
                    break
                
                # Find a cheaper replacement in the same position
                position = expensive_player['position']
                cheaper_options = all_players[
                    (all_players['position'] == position) &
                    (all_players['price'] < expensive_player['price']) &
                    (~all_players['id'].isin(selected_team['id']))
                ].sort_values('value', ascending=False)
                
                if len(cheaper_options) > 0:
                    replacement = cheaper_options.iloc[0]
                    
                    # Replace the expensive player
                    selected_team = selected_team[selected_team['id'] != expensive_player['id']]
                    selected_team = pd.concat([selected_team, pd.DataFrame([replacement])], ignore_index=True)
                    
                    price_diff = expensive_player['price'] - replacement['price']
                    total_cost -= price_diff
                    
                    print(f"🔄 Replaced {expensive_player['web_name']} with {replacement['web_name']} (saved £{price_diff:.1f}m)")
        
        return selected_team
