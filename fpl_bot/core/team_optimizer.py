"""
Team Optimizer using Mixed Integer Programming (MIP)

Optimizes team selection and transfers across multiple gameweeks to maximize
total points from current gameweek to GW19 or end of season.

Includes robust optimization to handle prediction uncertainty.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pulp import *
from .predictor import PointsPredictor


class TeamOptimizer:
    """MIP-based team optimizer for multi-gameweek planning"""
    
    def __init__(self, data_collector):
        """Initialize optimizer with data collector
        
        Args:
            data_collector: DataCollector instance for fetching FPL data
        """
        self.data_collector = data_collector
        self.predictor = PointsPredictor()
        
        # FPL constraints
        self.SQUAD_SIZE = 15
        self.STARTING_11 = 11
        self.BENCH_SIZE = 4
        self.TOTAL_BUDGET = 100.0
        
        # Position limits for squad
        self.MIN_GK = 2
        self.MAX_GK = 2
        self.MIN_DEF = 5
        self.MAX_DEF = 5
        self.MIN_MID = 5
        self.MAX_MID = 5
        self.MIN_FWD = 3
        self.MAX_FWD = 3
        
        # Position limits for starting 11
        self.STARTING_GK = 1
        self.STARTING_MIN_DEF = 3
        self.STARTING_MIN_MID = 2
        self.STARTING_MIN_FWD = 1
        
        # Players per team limit
        self.MAX_PLAYERS_PER_TEAM = 3
        
        # Transfer constraints
        self.FREE_TRANSFERS_PER_GW = 1
        self.MAX_FREE_TRANSFERS = 5  # Can bank up to 5
        self.TRANSFER_COST = 4  # Points cost per transfer beyond free transfers
        
    def optimize_team(
        self,
        current_team: List[int],
        current_budget: float,
        free_transfers: int,
        horizon_gws: Optional[int] = None,
        verbose: bool = False,
        risk_aversion: float = 0.5,
        min_chance_of_playing: int = 75,
        wildcard_gw: Optional[int] = None
    ) -> Dict:
        """Optimize team selection and transfers over multiple gameweeks
        
        Args:
            current_team: List of player IDs currently in squad
            current_budget: Available budget in millions
            free_transfers: Number of free transfers available
            horizon_gws: Number of gameweeks to optimize (None = until GW19 or end of season)
            verbose: Whether to show detailed output
            risk_aversion: Risk aversion parameter (0=aggressive, 1=conservative)
                          Conservative = weight predictions by confidence
            min_chance_of_playing: Minimum % chance of playing to consider a player (default 75)
                                  Players below this threshold are excluded from transfers
            wildcard_gw: Gameweek to use wildcard (unlimited free transfers, default None)
                          
        Returns:
            Dict with optimization results including transfers and expected points
        """
        if verbose:
            print("\n" + "="*60)
            print("TEAM OPTIMIZER - MIP-based Multi-Gameweek Planning")
            print("="*60)
        
        # Get current gameweek
        season_data = self.data_collector.get_current_season_data()
        if not season_data:
            return {'error': 'Could not fetch season data'}
        
        current_gw = self._get_current_gameweek(season_data)
        if not current_gw:
            return {'error': 'Could not determine current gameweek'}
        
        # Determine planning horizon
        # Limit to 6 gameweeks ahead to avoid over-optimization on uncertain future predictions
        if horizon_gws is None:
            # Default: plan 6 weeks ahead (balances near-term optimization with some foresight)
            target_gw = min(current_gw + 5, self._get_last_gameweek(season_data))
        else:
            target_gw = min(current_gw + horizon_gws - 1, self._get_last_gameweek(season_data))
        
        gameweeks = list(range(current_gw, target_gw + 1))
        if verbose:
            print(f"\nOptimizing from GW{current_gw} to GW{target_gw} ({len(gameweeks)} gameweeks)")
        
        # Get player data
        players_df = self.data_collector.get_player_data(include_set_pieces=True)
        if players_df is None:
            return {'error': 'Could not fetch player data'}
        
        # Filter out players with low chance of playing (injured/doubtful)
        # Keep players where:
        # 1. chance_of_playing_next_round is null (no injury concern) OR
        # 2. chance_of_playing_next_round >= 75% (likely to play)
        # Also keep current team players to allow for valid optimization
        initial_player_count = len(players_df)
        players_df['is_available'] = (
            (players_df['chance_of_playing_next_round'].isna()) |  # No injury news
            (players_df['chance_of_playing_next_round'] >= min_chance_of_playing) |  # Above threshold
            (players_df['id'].isin(current_team))                   # Current team players (allow transfers out)
        )
        
        # Store excluded players before filtering
        excluded_players = players_df[~players_df['is_available']].copy()
        
        # Filter to available players
        players_df = players_df[players_df['is_available']].copy()
        
        if verbose:
            filtered_count = len(excluded_players)
            print(f"\nPlayer availability filter (min {min_chance_of_playing}% chance of playing):")
            print(f"  • Excluded {filtered_count} injured/doubtful players")
            print(f"  • {len(players_df)} available players for consideration")
            
            # Show excluded players if any
            if filtered_count > 0 and filtered_count <= 20:
                print(f"\n  [!] Excluded due to injury concerns:")
                for _, player in excluded_players.iterrows():
                    chance = player.get('chance_of_playing_next_round', 'N/A')
                    news = player.get('news', '')[:50]
                    print(f"    - {player['web_name']} ({chance}% chance) - {news}")
            elif filtered_count > 20:
                print(f"\n  [!] Showing top 10 excluded players:")
                for _, player in excluded_players.head(10).iterrows():
                    chance = player.get('chance_of_playing_next_round', 'N/A')
                    news = player.get('news', '')[:50]
                    print(f"    - {player['web_name']} ({chance}% chance) - {news}")
        
        # Get fixtures and team strengths for predictions
        fixtures_df = self.data_collector.get_fixtures()
        team_strengths_df = self.data_collector.get_team_strengths()
        
        # Generate points predictions for each gameweek using advanced predictor
        predictions = self._predict_points(
            players_df, fixtures_df, team_strengths_df, gameweeks, 
            risk_aversion=risk_aversion, verbose=verbose
        )
        
        # Solve MIP optimization
        result = self._solve_mip(
            players_df=players_df,
            current_team=current_team,
            current_budget=current_budget,
            free_transfers=free_transfers,
            gameweeks=gameweeks,
            predictions=predictions,
            verbose=verbose,
            wildcard_gw=wildcard_gw
        )
        
        return result
    
    def _predict_points(
        self,
        players_df: pd.DataFrame,
        fixtures_df: pd.DataFrame,
        team_strengths_df: pd.DataFrame,
        gameweeks: List[int],
        risk_aversion: float = 0.5,
        verbose: bool = False
    ) -> pd.DataFrame:
        """Generate advanced points predictions using fixture analysis
        
        Uses the PointsPredictor class for sophisticated predictions based on:
        - Fixture difficulty
        - Home/away advantage
        - Set piece roles
        - Recent form
        - Position-specific scoring patterns
        
        Args:
            players_df: DataFrame with player data
            fixtures_df: DataFrame with fixtures
            team_strengths_df: DataFrame with team strengths
            gameweeks: List of gameweeks to predict
            risk_aversion: 0-1, higher = more conservative (weight by confidence)
            verbose: Whether to show detailed output
            
        Returns:
            DataFrame with columns: player_id, gameweek, predicted_points, confidence
        """
        if verbose:
            print(f"\nGenerating advanced predictions for {len(players_df)} players over {len(gameweeks)} gameweeks...")
            print(f"Risk aversion: {risk_aversion:.2f} (0=aggressive, 1=conservative)")
        
        all_predictions = []
        
        for gw in gameweeks:
            # Use advanced predictor
            gw_predictions = self.predictor.predict_gameweek_points(
                players_df, fixtures_df, team_strengths_df, gw
            )
            all_predictions.append(gw_predictions)
        
        predictions_df = pd.DataFrame(pd.concat(all_predictions, ignore_index=True))
        
        # Apply risk aversion: conservative approach weights predictions by confidence
        # risk_aversion=0: use raw predictions (aggressive)
        # risk_aversion=1: heavily discount uncertain predictions (conservative)
        if risk_aversion > 0:
            predictions_df['predicted_points'] = (
                predictions_df['predicted_points'] * 
                (predictions_df['confidence'] ** risk_aversion)
            )
        
        # Apply time decay to predictions for distant gameweeks
        # Predictions become less reliable further into the future
        # This prevents over-commitment to uncertain long-term scenarios
        first_gw = gameweeks[0]
        for idx, gw in enumerate(gameweeks):
            weeks_ahead = gw - first_gw
            # Decay factor: 100% for current GW, 97% for GW+1, 94% for GW+2, etc.
            # Minimum 85% (at 10 weeks out)
            decay_factor = max(0.85, 1.0 - (weeks_ahead * 0.015))
            
            # Apply decay to this gameweek's predictions
            mask = predictions_df['gameweek'] == gw
            predictions_df.loc[mask, 'predicted_points'] *= decay_factor
        
        if verbose:
            print(f"[OK] Generated {len(predictions_df)} advanced predictions")
            avg_confidence = predictions_df['confidence'].mean()
            print(f"  Average prediction confidence: {avg_confidence:.2f}")
            print(f"  Time decay applied: 100% (current) -> {max(0.85, 1.0 - ((len(gameweeks)-1) * 0.015)):.0%} (GW+{len(gameweeks)-1})")
        
        return predictions_df
    
    def _solve_mip(
        self,
        players_df: pd.DataFrame,
        current_team: List[int],
        current_budget: float,
        free_transfers: int,
        gameweeks: List[int],
        predictions: pd.DataFrame,
        verbose: bool = False,
        wildcard_gw: Optional[int] = None
    ) -> Dict:
        """Solve the MIP optimization problem
        
        Args:
            players_df: DataFrame with all players
            current_team: Current squad player IDs
            current_budget: Available budget
            free_transfers: Free transfers available
            gameweeks: List of gameweeks to optimize
            predictions: Predicted points for each player/gameweek
            wildcard_gw: Gameweek to use wildcard (unlimited free transfers)
            
        Returns:
            Dict with optimal team selections and transfers
        """
        if verbose:
            print("\nSolving MIP optimization...")
        
        # Create the optimization problem
        prob = LpProblem("FPL_Team_Optimization", LpMaximize)
        
        # Players and gameweeks
        player_ids = players_df['id'].tolist()
        n_players = len(player_ids)
        n_gameweeks = len(gameweeks)
        
        # Create player lookup
        player_dict = players_df.set_index('id').to_dict('index')
        
        # Decision variables
        # squad[i,t] = 1 if player i is in squad for gameweek t
        squad = LpVariable.dicts("squad",
                                 ((i, t) for i in player_ids for t in gameweeks),
                                 cat='Binary')
        
        # starting[i,t] = 1 if player i is in starting 11 for gameweek t
        starting = LpVariable.dicts("starting",
                                    ((i, t) for i in player_ids for t in gameweeks),
                                    cat='Binary')
        
        # captain[i,t] = 1 if player i is captain for gameweek t
        captain = LpVariable.dicts("captain",
                                   ((i, t) for i in player_ids for t in gameweeks),
                                   cat='Binary')
        
        # transfer_in[i,t] = 1 if player i is transferred in before gameweek t
        transfer_in = LpVariable.dicts("transfer_in",
                                       ((i, t) for i in player_ids for t in gameweeks),
                                       cat='Binary')
        
        # transfer_out[i,t] = 1 if player i is transferred out before gameweek t
        transfer_out = LpVariable.dicts("transfer_out",
                                        ((i, t) for i in player_ids for t in gameweeks),
                                        cat='Binary')
        
        # hits_taken[t] = number of extra transfers (beyond free) for gameweek t
        hits_taken = LpVariable.dicts("hits",
                                      gameweeks,
                                      lowBound=0,
                                      cat='Integer')
        
        # Objective: Maximize total predicted points only
        # Budget is handled as a constraint, not an objective
        objective = 0
        
        for t in gameweeks:
            for i in player_ids:
                pred_points = predictions[
                    (predictions['player_id'] == i) & 
                    (predictions['gameweek'] == t)
                ]['predicted_points'].values
                
                if len(pred_points) > 0:
                    pts = pred_points[0]
                    # Regular starting points
                    objective += pts * starting[i, t]
                    # Captain bonus (double points)
                    objective += pts * captain[i, t]
                    
                    # Bench value: Give bench players a small fraction of their points
                    # This prevents optimizer from treating bad bench players as "free"
                    # Use 10% of predicted points for squad players not in starting 11
                    bench_value = 0.1 * pts * (squad[i, t] - starting[i, t])
                    objective += bench_value
            
            # Subtract transfer costs (4 pts per hit)
            objective -= self.TRANSFER_COST * hits_taken[t]
        
        prob += objective, "Total_Points"
        
        # Constraints
        if verbose:
            print("Adding constraints...")
        
        # 1. Squad size constraint (15 players)
        for t in gameweeks:
            prob += lpSum([squad[i, t] for i in player_ids]) == self.SQUAD_SIZE, f"Squad_Size_GW{t}"
        
        # 2. Starting 11 constraint
        for t in gameweeks:
            prob += lpSum([starting[i, t] for i in player_ids]) == self.STARTING_11, f"Starting_11_GW{t}"
        
        # 3. Can only start players in squad
        for i in player_ids:
            for t in gameweeks:
                prob += starting[i, t] <= squad[i, t], f"Start_Only_Squad_{i}_GW{t}"
        
        # 4. Position constraints for squad
        for t in gameweeks:
            for position, min_count, max_count in [
                (1, self.MIN_GK, self.MAX_GK),
                (2, self.MIN_DEF, self.MAX_DEF),
                (3, self.MIN_MID, self.MAX_MID),
                (4, self.MIN_FWD, self.MAX_FWD)
            ]:
                position_players = [i for i in player_ids if player_dict[i]['element_type'] == position]
                prob += lpSum([squad[i, t] for i in position_players]) >= min_count, \
                        f"Squad_Min_Pos{position}_GW{t}"
                prob += lpSum([squad[i, t] for i in position_players]) <= max_count, \
                        f"Squad_Max_Pos{position}_GW{t}"
        
        # 5. Position constraints for starting 11
        for t in gameweeks:
            # Exactly 1 GK
            gk_players = [i for i in player_ids if player_dict[i]['element_type'] == 1]
            prob += lpSum([starting[i, t] for i in gk_players]) == self.STARTING_GK, \
                    f"Starting_GK_GW{t}"
            
            # At least minimums for outfield positions
            for position, min_count in [(2, self.STARTING_MIN_DEF), 
                                        (3, self.STARTING_MIN_MID),
                                        (4, self.STARTING_MIN_FWD)]:
                position_players = [i for i in player_ids if player_dict[i]['element_type'] == position]
                prob += lpSum([starting[i, t] for i in position_players]) >= min_count, \
                        f"Starting_Min_Pos{position}_GW{t}"
        
        # 6. Max players per team (3)
        teams = players_df['team'].unique()
        for t in gameweeks:
            for team in teams:
                team_players = [i for i in player_ids if player_dict[i]['team'] == team]
                prob += lpSum([squad[i, t] for i in team_players]) <= self.MAX_PLAYERS_PER_TEAM, \
                        f"Max_Per_Team_{team}_GW{t}"
        
        # 7. Captain constraints (exactly 1 captain, must be starting)
        for t in gameweeks:
            prob += lpSum([captain[i, t] for i in player_ids]) == 1, f"One_Captain_GW{t}"
            for i in player_ids:
                prob += captain[i, t] <= starting[i, t], f"Captain_Must_Start_{i}_GW{t}"
        
        # 8. Initial squad constraint (first gameweek must match current team or transfers)
        first_gw = gameweeks[0]
        for i in player_ids:
            if i in current_team:
                # If in current team, either keep or transfer out
                prob += squad[i, first_gw] + transfer_out[i, first_gw] == 1, \
                        f"Current_Player_{i}_FirstGW"
            else:
                # If not in current team, can only be in squad if transferred in
                prob += squad[i, first_gw] <= transfer_in[i, first_gw], \
                        f"New_Player_{i}_FirstGW"
        
        # 9. Squad evolution constraint - properly track who's in squad each week
        # For each player after first gameweek:
        # squad[i,t] = squad[i,t-1] - transfer_out[i,t] + transfer_in[i,t]
        for i in player_ids:
            for idx, t in enumerate(gameweeks):
                if idx > 0:
                    t_prev = gameweeks[idx - 1]
                    # Player is in squad at t if:
                    # - Was in squad at t-1 AND not transferred out at t, OR
                    # - Was transferred in at t
                    prob += squad[i, t] == squad[i, t_prev] - transfer_out[i, t] + transfer_in[i, t], \
                            f"Squad_Evolution_{i}_GW{t}"
        
        # 9b. Cannot transfer in AND out the same player in the same gameweek
        for i in player_ids:
            for t in gameweeks:
                prob += transfer_in[i, t] + transfer_out[i, t] <= 1, \
                        f"No_Same_Player_In_And_Out_{i}_GW{t}"
        
        # 10. Transfer balance (transfers in = transfers out)
        for t in gameweeks:
            prob += lpSum([transfer_in[i, t] for i in player_ids]) == \
                    lpSum([transfer_out[i, t] for i in player_ids]), \
                    f"Transfer_Balance_GW{t}"
        
        # 10b. Position-specific transfer balance (maintain squad composition)
        # For each position, transfers in must equal transfers out
        for t in gameweeks:
            for position in [1, 2, 3, 4]:  # GK, DEF, MID, FWD
                position_players = [i for i in player_ids if player_dict[i]['element_type'] == position]
                prob += lpSum([transfer_in[i, t] for i in position_players]) == \
                        lpSum([transfer_out[i, t] for i in position_players]), \
                        f"Position_Transfer_Balance_Pos{position}_GW{t}"
        
        # 11. Transfer costs and free transfers
        # Wildcard week: unlimited free transfers (hits = 0)
        # Other weeks: normal free transfer logic
        for idx, t in enumerate(gameweeks):
            n_transfers = lpSum([transfer_in[i, t] for i in player_ids])
            
            if wildcard_gw and t == wildcard_gw:
                # WILDCARD WEEK: All transfers are free!
                prob += hits_taken[t] == 0, f"Wildcard_Free_Transfers_GW{t}"
            elif idx == 0:
                # First gameweek: use provided free transfers
                prob += hits_taken[t] >= n_transfers - free_transfers, \
                        f"Hits_Calculation_GW{t}"
            else:
                # Subsequent gameweeks: 1 free transfer per week (simplified)
                # TODO: Model banking of free transfers properly
                prob += hits_taken[t] >= n_transfers - self.FREE_TRANSFERS_PER_GW, \
                        f"Hits_Calculation_GW{t}"
        
        # 12. Budget constraint - only enforce maximum (not minimum)
        # Just ensure squad doesn't exceed available budget
        # Don't force minimum value - let transfer penalty handle preventing downgrades
        for t in gameweeks:
            squad_cost = lpSum([
                squad[i, t] * player_dict[i]['now_cost'] / 10.0
                for i in player_ids
            ])
            # Squad cannot exceed budget + bank
            prob += squad_cost <= self.TOTAL_BUDGET + current_budget, \
                    f"Budget_Constraint_GW{t}"
        
        # Solve
        if verbose:
            print("Solving optimization problem (this may take a minute)...")
        solver = PULP_CBC_CMD(msg=0)  # Use CBC solver, suppress output
        prob.solve(solver)
        
        # Check solution status
        status = LpStatus[prob.status]
        if verbose:
            print(f"\nOptimization Status: {status}")
        
        if status != 'Optimal':
            return {
                'status': status,
                'error': 'Could not find optimal solution'
            }
        
        # Extract solution
        result = self._extract_solution(
            squad, starting, captain, transfer_in, transfer_out, hits_taken,
            player_ids, gameweeks, players_df, current_team, predictions
        )
        
        result['status'] = 'Optimal'
        result['objective_value'] = value(prob.objective)
        
        if verbose:
            print(f"\n[OK] Optimization complete!")
            print(f"Expected total points: {result['objective_value']:.1f}")
        
        return result
    
    def _extract_solution(
        self,
        squad, starting, captain, transfer_in, transfer_out, hits_taken,
        player_ids, gameweeks, players_df, current_team, predictions
    ) -> Dict:
        """Extract the complete multi-week solution from decision variables"""
        
        player_dict = players_df.set_index('id').to_dict('index')
        
        # Extract plan for each gameweek
        weekly_plans = []
        
        for gw in gameweeks:
            # Extract transfers for this gameweek
            transfers_in = []
            transfers_out = []
            
            for i in player_ids:
                if value(transfer_in[i, gw]) == 1:
                    player = player_dict[i]
                    transfers_in.append({
                        'player_id': i,
                        'player_name': player['web_name'],
                        'position': player['position_name'],
                        'team': player['team_short_name'],
                        'cost': player['now_cost'] / 10.0
                    })
                
                if value(transfer_out[i, gw]) == 1:
                    player = player_dict[i]
                    transfers_out.append({
                        'player_id': i,
                        'player_name': player['web_name'],
                        'position': player['position_name'],
                        'team': player['team_short_name'],
                        'cost': player['now_cost'] / 10.0
                    })
            
            # Extract squad for this gameweek
            squad_players = []
            starting_players = []
            bench_players = []
            captain_id = None
            vice_captain_id = None
            
            for i in player_ids:
                if value(squad[i, gw]) == 1:
                    player = player_dict[i]
                    player_info = {
                        'player_id': i,
                        'player_name': player['web_name'],
                        'position': player['position_name'],
                        'team': player['team_short_name'],
                        'cost': player['now_cost'] / 10.0
                    }
                    
                    squad_players.append(player_info)
                    
                    if value(starting[i, gw]) == 1:
                        starting_players.append(player_info)
                    else:
                        bench_players.append(player_info)
                    
                    if value(captain[i, gw]) == 1:
                        captain_id = i
            
            # Calculate hits for this gameweek (MUST be before using it!)
            n_transfers = len(transfers_in)
            hits = int(value(hits_taken[gw]))
            
            # Calculate expected points for this gameweek
            expected_points = 0.0
            for player_info in starting_players:
                player_id = player_info['player_id']
                # Get predicted points for this player this GW
                pred = predictions[
                    (predictions['player_id'] == player_id) & 
                    (predictions['gameweek'] == gw)
                ]['predicted_points'].values
                if len(pred) > 0:
                    expected_points += pred[0]
            
            # Add captain bonus
            if captain_id:
                captain_pred = predictions[
                    (predictions['player_id'] == captain_id) & 
                    (predictions['gameweek'] == gw)
                ]['predicted_points'].values
                if len(captain_pred) > 0:
                    expected_points += captain_pred[0]  # Captain gets double (already counted once)
            
            # Subtract transfer costs
            expected_points -= hits * self.TRANSFER_COST
            
            # Build gameweek plan
            gw_plan = {
                'gameweek': gw,
                'transfers': {
                    'in': transfers_in,
                    'out': transfers_out,
                    'count': n_transfers
                },
                'hits_taken': hits,
                'points_cost': hits * self.TRANSFER_COST,
                'expected_points': expected_points,  # NEW: Track expected points
                'squad': {
                    'all': squad_players,
                    'starting_11': starting_players,
                    'bench': bench_players
                },
                'captain': {
                    'player_id': captain_id,
                    'player_name': player_dict[captain_id]['web_name'] if captain_id else None
                }
            }
            
            weekly_plans.append(gw_plan)
        
        # Create summary for immediate next gameweek (first in plan)
        first_gw_plan = weekly_plans[0]
        
        return {
            'weekly_plans': weekly_plans,
            'summary': {
                'planning_horizon': f"GW{gameweeks[0]} to GW{gameweeks[-1]}",
                'total_gameweeks': len(gameweeks),
                'immediate_action': first_gw_plan
            }
        }
    
    def _get_current_gameweek(self, season_data: Dict) -> Optional[int]:
        """Extract next unplayed gameweek from season data
        
        Returns the gameweek that hasn't finished yet (the one to optimize for)
        """
        events = season_data.get('events', [])
        
        # Find the first gameweek that hasn't finished
        for event in events:
            if not event.get('finished', False):
                return event.get('id')
        
        # Fallback: return last gameweek if all are finished
        if events:
            return events[-1].get('id')
        
        return None
    
    def _get_last_gameweek(self, season_data: Dict) -> int:
        """Get the last gameweek of the season"""
        events = season_data.get('events', [])
        if events:
            return max(e['id'] for e in events)
        return 38  # Default to 38

