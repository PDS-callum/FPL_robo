import pandas as pd
import numpy as np
import pulp

class FPLTeamOptimizer:
    def __init__(self, total_budget=100.0):
        self.total_budget = total_budget
        self.squad_size = 15
        self.formation_constraints = {
            'GKP': {'min': 2, 'max': 2},
            'DEF': {'min': 5, 'max': 5},
            'MID': {'min': 5, 'max': 5},
            'FWD': {'min': 3, 'max': 3}
        }
        self.team_constraints = {'min': 1, 'max': 3}  # Min and max players from each team
    
    def optimize_team(self, players_df, predictions_df):
        """
        Optimize FPL team selection based on predicted points
        
        Parameters:
        -----------
        players_df : DataFrame
            DataFrame with player information
        predictions_df : DataFrame
            DataFrame with predicted points for each player
            
        Returns:
        --------
        selected_team : DataFrame
            DataFrame with selected team
        """
        # Merge player data with predictions
        merged_df = players_df.merge(predictions_df, on='id', how='inner')
        
        # Create mapping of player positions
        pos_map = {
            1: 'GKP',
            2: 'DEF',
            3: 'MID',
            4: 'FWD'
        }
        merged_df['position'] = merged_df['element_type'].map(pos_map)
        
        # Create optimization model
        prob = pulp.LpProblem("FPL_Team_Selection", pulp.LpMaximize)
        
        # Create decision variables for each player (1 if selected, 0 if not)
        player_vars = pulp.LpVariable.dicts("player", 
                                          merged_df.index, 
                                          cat=pulp.LpBinary)
        
        # Objective function: maximize predicted points
        prob += pulp.lpSum([merged_df.loc[i, 'predicted_points'] * player_vars[i] 
                          for i in merged_df.index])
        
        # Constraint: total budget
        prob += pulp.lpSum([merged_df.loc[i, 'now_cost'] / 10 * player_vars[i] 
                          for i in merged_df.index]) <= self.total_budget
        
        # Constraint: squad size
        prob += pulp.lpSum([player_vars[i] for i in merged_df.index]) == self.squad_size
        
        # Constraint: position limits
        for position, limits in self.formation_constraints.items():
            prob += pulp.lpSum([player_vars[i] for i in merged_df.index 
                              if merged_df.loc[i, 'position'] == position]) >= limits['min']
            prob += pulp.lpSum([player_vars[i] for i in merged_df.index 
                              if merged_df.loc[i, 'position'] == position]) <= limits['max']
        
        # Constraint: max 3 players from the same team
        for team in merged_df['team'].unique():
            prob += pulp.lpSum([player_vars[i] for i in merged_df.index 
                              if merged_df.loc[i, 'team'] == team]) <= self.team_constraints['max']
        
        # Solve the problem
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        # Get the selected players
        selected_indices = [i for i in merged_df.index if player_vars[i].value() == 1]
        selected_team = merged_df.loc[selected_indices].copy()
        
        # Sort by position and predicted points
        position_order = {'GKP': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}
        selected_team['pos_order'] = selected_team['position'].map(position_order)
        selected_team = selected_team.sort_values(['pos_order', 'predicted_points'], ascending=[True, False])
        
        return selected_team
    
    def select_playing_xi(self, full_team):
        """
        Select the best playing XI from a full squad of 15 players
        
        Parameters:
        -----------
        full_team : DataFrame
            DataFrame with the full 15-player squad
            
        Returns:
        --------
        playing_xi : DataFrame
            DataFrame with the best 11 players
        captain : Series
            Player selected as captain
        vice_captain : Series
            Player selected as vice-captain
        """
        # Create valid formation constraints
        valid_formations = [
            {'GKP': 1, 'DEF': 3, 'MID': 4, 'FWD': 3},  # 3-4-3
            {'GKP': 1, 'DEF': 3, 'MID': 5, 'FWD': 2},  # 3-5-2
            {'GKP': 1, 'DEF': 4, 'MID': 3, 'FWD': 3},  # 4-3-3
            {'GKP': 1, 'DEF': 4, 'MID': 4, 'FWD': 2},  # 4-4-2
            {'GKP': 1, 'DEF': 4, 'MID': 5, 'FWD': 1},  # 4-5-1
            {'GKP': 1, 'DEF': 5, 'MID': 3, 'FWD': 2},  # 5-3-2
            {'GKP': 1, 'DEF': 5, 'MID': 4, 'FWD': 1},  # 5-4-1
        ]
        
        best_score = 0
        best_xi = None
        best_formation = None
        
        # Try each formation and find the best one
        for formation in valid_formations:
            prob = pulp.LpProblem("FPL_XI_Selection", pulp.LpMaximize)
            
            # Decision variables
            player_vars = pulp.LpVariable.dicts("player", 
                                              full_team.index, 
                                              cat=pulp.LpBinary)
            
            # Objective function
            prob += pulp.lpSum([full_team.loc[i, 'predicted_points'] * player_vars[i] 
                              for i in full_team.index])
            
            # Constraint: exactly 11 players
            prob += pulp.lpSum([player_vars[i] for i in full_team.index]) == 11
            
            # Constraint: formation
            for position, count in formation.items():
                prob += pulp.lpSum([player_vars[i] for i in full_team.index 
                                  if full_team.loc[i, 'position'] == position]) == count
            
            # Solve
            prob.solve(pulp.PULP_CBC_CMD(msg=False))
            
            # Calculate total score
            score = pulp.value(prob.objective)
            
            if score > best_score:
                best_score = score
                best_xi = [i for i in full_team.index if player_vars[i].value() == 1]
                best_formation = formation
        
        # Get the best XI
        playing_xi = full_team.loc[best_xi].copy()
        
        # Select captain and vice-captain (players with highest predicted points)
        sorted_xi = playing_xi.sort_values('predicted_points', ascending=False)
        captain = sorted_xi.iloc[0]
        vice_captain = sorted_xi.iloc[1]
        
        # Mark captain and vice-captain
        playing_xi['is_captain'] = False
        playing_xi['is_vice_captain'] = False
        playing_xi.loc[captain.name, 'is_captain'] = True
        playing_xi.loc[vice_captain.name, 'is_vice_captain'] = True
        
        return playing_xi, captain, vice_captain, best_formation