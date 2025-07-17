import pandas as pd
import numpy as np
from itertools import combinations

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
        position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        if 'element_type' in team_data.columns:
            team_data['position'] = team_data['element_type'].map(position_map)
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
        
        # Use greedy algorithm for team selection (faster than integer programming)
        selected_team = self._greedy_team_selection(team_data, budget)
        
        return selected_team
    
    def _greedy_team_selection(self, players_df, budget):
        """
        Greedy algorithm for team selection with FPL constraints
        """
        selected_players = []
        remaining_budget = budget
        position_counts = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        team_counts = {}
        
        # Sort players by value (points per million) in descending order
        players_sorted = players_df.sort_values('value', ascending=False).copy()
        
        print(f"🔍 Debug: Top 5 players by value:")
        print(players_sorted[['web_name', 'position', 'price', 'predicted_points', 'value']].head())
        
        # First pass: select players greedily while respecting constraints
        for i, (_, player) in enumerate(players_sorted.iterrows()):
            position = player['position']
            team_id = player['team']
            price = player['price']
            
            # Debug first few iterations
            if i < 10:
                print(f"🔍 Debug: Checking {player['web_name']} ({position}) - £{price:.1f}m")
                print(f"  Position slots: {position_counts[position]}/{self.formation_constraints[position]['max']}")
                print(f"  Budget: £{remaining_budget:.1f}m")
                print(f"  Team constraint: {team_counts.get(team_id, 0)}/{self.max_players_per_team}")
            
            # Check constraints
            if (position_counts[position] < self.formation_constraints[position]['max'] and
                price <= remaining_budget and
                team_counts.get(team_id, 0) < self.max_players_per_team):
                
                # Add player to team
                selected_players.append(player)
                remaining_budget -= price
                position_counts[position] += 1
                team_counts[team_id] = team_counts.get(team_id, 0) + 1
                
                print(f"✅ Selected {player['web_name']} ({position}) - £{price:.1f}m")
                print(f"   Budget remaining: £{remaining_budget:.1f}m")
                print(f"   Position counts: {position_counts}")
                
                # Check if team is complete
                if sum(position_counts.values()) == 15:
                    break
        
        # Check if we have a complete team
        if sum(position_counts.values()) < 15:
            # Fill remaining positions with cheapest available players
            for position, count in position_counts.items():
                needed = self.formation_constraints[position]['max'] - count
                if needed > 0:
                    available_players = players_df[
                        (players_df['position'] == position) &
                        (~players_df['id'].isin([p['id'] for p in selected_players]))
                    ].sort_values('price')
                    
                    for _, player in available_players.iterrows():
                        if (needed > 0 and 
                            player['price'] <= remaining_budget and
                            team_counts.get(player['team'], 0) < self.max_players_per_team):
                            
                            selected_players.append(player)
                            remaining_budget -= player['price']
                            team_counts[player['team']] = team_counts.get(player['team'], 0) + 1
                            needed -= 1
        
        return pd.DataFrame(selected_players)
    
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
        Validate team against FPL constraints
        
        Returns:
        --------
        is_valid : bool
            Whether team meets all constraints
        errors : list
            List of constraint violations
        """
        errors = []
        
        # Check team size
        if len(team_df) != 15:
            errors.append(f"Team must have exactly 15 players, got {len(team_df)}")
        
        # Check position constraints
        position_counts = team_df['position'].value_counts()
        for pos, constraints in self.formation_constraints.items():
            count = position_counts.get(pos, 0)
            if count != constraints['max']:
                errors.append(f"Need exactly {constraints['max']} {pos}, got {count}")
        
        # Check budget constraint
        total_cost = team_df['price'].sum() if 'price' in team_df.columns else team_df['now_cost'].sum() / 10
        if total_cost > self.total_budget:
            errors.append(f"Team cost {total_cost:.1f}m exceeds budget {self.total_budget}m")
        
        # Check max players per team constraint
        team_counts = team_df['team'].value_counts()
        violating_teams = team_counts[team_counts > self.max_players_per_team]
        if len(violating_teams) > 0:
            for team_id, count in violating_teams.items():
                errors.append(f"Team {team_id} has {count} players (max {self.max_players_per_team})")
        
        return len(errors) == 0, errors
