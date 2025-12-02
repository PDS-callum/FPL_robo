"""
Debug script to investigate transfer recommendations for Sarr/Enzo and Mateta/Ekitike

This script will:
1. Get predictions for all 4 players across the next 7 gameweeks
2. Show fixture difficulty and predicted points
3. Analyze why the optimizer might prefer certain transfers
"""

import sys
import pandas as pd
from bot.core.data_collector import DataCollector
from bot.core.predictor import PointsPredictor
from bot.core.manager_analyzer import ManagerAnalyzer
from bot.core.team_optimizer import TeamOptimizer

def debug_transfer_recommendations(manager_id, player_names=None):
    """Debug transfer recommendations for specific players"""
    
    if player_names is None:
        player_names = ['Sarr', 'Enzo', 'Mateta', 'Ekitike']
    
    print(f"Debugging transfer recommendations for manager {manager_id}")
    print(f"Players: {', '.join(player_names)}")
    print("=" * 100)
    
    # Initialize components
    data_collector = DataCollector()
    predictor = PointsPredictor()
    manager_analyzer = ManagerAnalyzer(data_collector)
    team_optimizer = TeamOptimizer(data_collector)
    
    # Get data
    print("\n1. Collecting data...")
    players_df = data_collector.get_player_data(include_set_pieces=True)
    fixtures_df = data_collector.get_fixtures()
    team_strengths_df = data_collector.get_team_strengths()
    season_data = data_collector.get_current_season_data()
    
    # Get current gameweek
    current_gw = None
    for event in season_data.get('events', []):
        if not event.get('finished', False):
            current_gw = event.get('id')
            break
    
    if not current_gw:
        print("ERROR: Could not determine current gameweek")
        return
    
    print(f"Current gameweek: {current_gw}")
    
    # Get manager team
    print("\n2. Getting manager team...")
    manager_analysis = manager_analyzer.analyze_manager(manager_id)
    current_team = []
    team_data = manager_analysis.get('current_team', {})
    picks = team_data.get('picks', [])
    for pick in picks:
        current_team.append(pick['element'])
    
    print(f"Current team has {len(current_team)} players")
    
    # Find the players
    print(f"\n3. Finding players: {', '.join(player_names)}...")
    players = {}
    for name in player_names:
        player_matches = players_df[players_df['web_name'].str.contains(name, case=False, na=False)]
        if player_matches.empty:
            print(f"WARNING: Player '{name}' not found")
            continue
        
        # If multiple matches, take the first one
        player = player_matches.iloc[0]
        players[name] = player
        
        in_team = "[IN TEAM]" if player['id'] in current_team else "[NOT IN TEAM]"
        print(f"\n{name} (ID: {player['id']}) {in_team}:")
        print(f"  Position: {player['position_name']}")
        print(f"  Team: {player['team_short_name']}")
        print(f"  Cost: £{player['now_cost']/10:.1f}m")
        print(f"  Total Points: {player.get('total_points', 0)}")
        print(f"  Form: {player.get('form', 'N/A')}")
        print(f"  Minutes: {player.get('minutes', 0)}")
        print(f"  Goals: {player.get('goals_scored', 0)}")
        print(f"  Assists: {player.get('assists', 0)}")
    
    if not players:
        print("ERROR: No players found")
        return
    
    # Generate predictions for next 7 gameweeks
    print(f"\n4. Generating predictions for GW{current_gw} to GW{current_gw+6}...")
    gameweeks = list(range(current_gw, current_gw + 7))
    
    all_predictions = []
    for gw in gameweeks:
        gw_predictions = predictor.predict_gameweek_points(
            players_df, fixtures_df, team_strengths_df, gw
        )
        all_predictions.append(gw_predictions)
    
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    
    # Analyze predictions for each player
    print(f"\n5. Prediction analysis across next 7 gameweeks:")
    print("=" * 100)
    
    for name, player in players.items():
        player_id = player['id']
        player_preds = predictions_df[predictions_df['player_id'] == player_id].sort_values('gameweek')
        
        if player_preds.empty:
            print(f"\nWARNING: No predictions found for {name}")
            continue
        
        print(f"\n{name} ({player['position_name']}, {player['team_short_name']}):")
        print(f"{'GW':<6} {'Opponent':<15} {'H/A':<5} {'Difficulty':<12} {'Pred Pts':<10} {'Confidence':<10}")
        print("-" * 100)
        
        total_predicted = 0
        for _, pred in player_preds.iterrows():
            gw = int(pred['gameweek'])
            
            # Get fixture info
            player_team = player['team']
            fixture = fixtures_df[
                ((fixtures_df['team_h'] == player_team) | (fixtures_df['team_a'] == player_team)) &
                (fixtures_df['event'] == gw)
            ]
            
            if fixture.empty:
                opponent = "No fixture"
                venue = "N/A"
                difficulty = "N/A"
            else:
                fixture = fixture.iloc[0]
                is_home = fixture['team_h'] == player_team
                opponent_id = fixture['team_a'] if is_home else fixture['team_h']
                opponent_data = players_df[players_df['team'] == opponent_id]
                opponent = opponent_data['team_short_name'].iloc[0] if not opponent_data.empty else "Unknown"
                venue = "Home" if is_home else "Away"
                
                # Get difficulty
                if is_home:
                    difficulty = fixture.get('team_h_difficulty', 'N/A')
                else:
                    difficulty = fixture.get('team_a_difficulty', 'N/A')
            
            predicted_pts = pred['predicted_points']
            confidence = pred['confidence']
            total_predicted += predicted_pts
            
            print(f"GW{gw:<4} {opponent:<15} {venue:<5} {str(difficulty):<12} {predicted_pts:<10.2f} {confidence:<10.2f}")
        
        print(f"\nTotal predicted points (7 GWs): {total_predicted:.2f}")
        print(f"Average per GW: {total_predicted/7:.2f}")
    
    # Check team constraints
    print(f"\n6. Team composition check:")
    print("=" * 100)
    
    # Count players per team in current squad
    team_counts = {}
    for player_id in current_team:
        player_data = players_df[players_df['id'] == player_id]
        if not player_data.empty:
            team = player_data.iloc[0]['team_short_name']
            team_counts[team] = team_counts.get(team, 0) + 1
    
    print("Current team composition:")
    for team, count in sorted(team_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {team}: {count} player{'s' if count != 1 else ''}")
    
    # Check if any teams have 3 players (can't add more)
    teams_at_limit = [team for team, count in team_counts.items() if count >= 3]
    if teams_at_limit:
        print(f"\n[!] Teams at 3-player limit: {', '.join(teams_at_limit)}")
        print("   Cannot add players from these teams without removing one first!")
    
    # Check bench positions
    print(f"\n7. Bench analysis:")
    print("=" * 100)
    
    # Get picks with positions
    picks_with_positions = sorted(team_data.get('picks', []), key=lambda x: x.get('position', 99))
    
    print("Current squad (by position):")
    for pick in picks_with_positions:
        player_id = pick['element']
        position = pick.get('position', 99)
        player_data = players_df[players_df['id'] == player_id]
        
        if not player_data.empty:
            player = player_data.iloc[0]
            try:
                player_name = player['web_name']
                player_pos = player['position_name']
                player_team = player['team_short_name']
                
                # Get prediction for current GW
                player_pred = predictions_df[
                    (predictions_df['player_id'] == player_id) & 
                    (predictions_df['gameweek'] == current_gw)
                ]
                pred_pts = player_pred.iloc[0]['predicted_points'] if not player_pred.empty else 0
                
                location = "Starting XI" if position <= 11 else f"Bench ({position - 11})"
                
                # Highlight our target players
                marker = ""
                if any(name.lower() in player_name.lower() for name in player_names):
                    marker = " [*]"
                
                # Safely encode player name
                safe_name = player_name.encode('ascii', 'replace').decode('ascii')
                print(f"  {position:2d}. {safe_name:20s} ({player_pos:3s}, {player_team:3s}) - {location:15s} - {pred_pts:.1f} pts{marker}")
            except Exception as e:
                print(f"  {position:2d}. [Name error] - {str(e)[:30]}")
    
    # Compare transfer options
    print(f"\n8. Transfer comparison:")
    print("=" * 100)
    
    # Check for common transfer pairs
    transfer_pairs = []
    
    # Sarr vs Enzo
    if 'Sarr' in players and 'Enzo' in players:
        transfer_pairs.append(('Sarr', 'Enzo'))
    
    # Mateta vs Ekitike
    if 'Mateta' in players and 'Ekitike' in players:
        transfer_pairs.append(('Ekitike', 'Mateta'))
    
    for out_name, in_name in transfer_pairs:
        out_player = players[out_name]
        in_player = players[in_name]
        
        out_preds = predictions_df[predictions_df['player_id'] == out_player['id']]
        in_preds = predictions_df[predictions_df['player_id'] == in_player['id']]
        
        out_total = out_preds['predicted_points'].sum()
        in_total = in_preds['predicted_points'].sum()
        
        difference = in_total - out_total
        
        print(f"\n{out_name} OUT -> {in_name} IN:")
        print(f"  {out_name} total predicted (7 GWs): {out_total:.2f}")
        print(f"  {in_name} total predicted (7 GWs): {in_total:.2f}")
        print(f"  Difference: {difference:+.2f} points")
        print(f"  Cost difference: £{(in_player['now_cost'] - out_player['now_cost'])/10:.1f}m")
        
        if difference > 4:
            print(f"  [+] Transfer worth a hit ({difference:.2f} > 4)")
        elif difference > 0:
            print(f"  ~ Marginal gain with free transfer")
        else:
            print(f"  [-] Transfer not beneficial")
    
    # Determine which transfer is better
    print(f"\n9. Transfer recommendation analysis:")
    print("=" * 100)
    
    if len(transfer_pairs) >= 2:
        # Compare the two main transfers
        sarr_enzo = None
        ekitike_mateta = None
        
        for out_name, in_name in transfer_pairs:
            out_preds = predictions_df[predictions_df['player_id'] == players[out_name]['id']]
            in_preds = predictions_df[predictions_df['player_id'] == players[in_name]['id']]
            
            out_total = out_preds['predicted_points'].sum()
            in_total = in_preds['predicted_points'].sum()
            difference = in_total - out_total
            
            if out_name == 'Sarr' and in_name == 'Enzo':
                sarr_enzo = difference
            elif out_name == 'Ekitike' and in_name == 'Mateta':
                ekitike_mateta = difference
        
        if sarr_enzo is not None and ekitike_mateta is not None:
            print(f"Transfer value comparison:")
            print(f"  Sarr -> Enzo: {sarr_enzo:+.2f} points over 7 GWs")
            print(f"  Ekitike -> Mateta: {ekitike_mateta:+.2f} points over 7 GWs")
            print()
            
            if sarr_enzo > ekitike_mateta:
                print(f"  [!] ISSUE IDENTIFIED: Bot prefers Sarr -> Enzo")
                print(f"      Difference: {sarr_enzo - ekitike_mateta:.2f} points")
                print(f"      This explains why it's suggesting this transfer!")
                print()
                print(f"  Possible reasons:")
                print(f"    1. Enzo has better underlying stats (xG/xA)")
                print(f"    2. Enzo has better fixtures in later gameweeks")
                print(f"    3. Ekitike is on bench so his low points don't matter")
                print(f"    4. Budget constraints require Sarr out first")
            else:
                print(f"  [+] Ekitike -> Mateta is better by {ekitike_mateta - sarr_enzo:.2f} points")
                print(f"    But bot is suggesting Sarr -> Enzo?")
                print(f"    Check for team limits or budget constraints below...")
    
    # Analyze why predictions differ
    print(f"\n10. Detailed analysis of prediction factors:")
    print("=" * 100)
    
    team_stats = predictor._build_team_stats(players_df, fixtures_df, current_gw)
    
    for name, player in players.items():
        print(f"\n{name}:")
        
        # Get first fixture
        player_team = player['team']
        first_fixture = fixtures_df[
            ((fixtures_df['team_h'] == player_team) | (fixtures_df['team_a'] == player_team)) &
            (fixtures_df['event'] == current_gw)
        ]
        
        if first_fixture.empty:
            print("  No fixture in current gameweek")
            continue
        
        first_fixture = first_fixture.iloc[0]
        fixture_info = predictor._get_player_fixtures(player, first_fixture)
        
        if not fixture_info:
            print("  Could not get fixture info")
            continue
        
        fixture_info = fixture_info[0]
        
        # Analyze prediction components
        position = player['position_name']
        minutes_prob = predictor._predict_minutes_probability(player, position)
        goal_involvement = predictor._predict_goal_involvement(player, fixture_info, team_stats, position)
        
        print(f"  Minutes probability: {minutes_prob:.2%}")
        print(f"  Expected goals (per 90): {goal_involvement['goals']:.3f}")
        print(f"  Expected assists (per 90): {goal_involvement['assists']:.3f}")
        
        if position in ['DEF', 'GK']:
            cs_prob = predictor._predict_clean_sheet_probability(fixture_info, team_stats)
            print(f"  Clean sheet probability: {cs_prob:.2%}")
        
        # Show underlying stats
        print(f"\n  Underlying stats:")
        print(f"    Total minutes: {player.get('minutes', 0)}")
        print(f"    Starts: {player.get('starts', 0)}")
        print(f"    Goals scored: {player.get('goals_scored', 0)}")
        print(f"    Assists: {player.get('assists', 0)}")
        print(f"    Expected goals (xG): {player.get('expected_goals', 'N/A')}")
        print(f"    Expected assists (xA): {player.get('expected_assists', 'N/A')}")
        print(f"    Expected goals per 90: {player.get('expected_goals_per_90', 'N/A')}")
        print(f"    Expected assists per 90: {player.get('expected_assists_per_90', 'N/A')}")
    
    # Run actual optimizer to see what it suggests
    print(f"\n11. Running optimizer to see actual recommendation:")
    print("=" * 100)
    
    try:
        manager_info = manager_analysis.get('manager_info', {})
        budget = manager_info.get('bank', 0.0)
        team_value = manager_info.get('team_value', 0.0)
        saved_transfers = manager_analysis.get('saved_transfers', {})
        free_transfers = saved_transfers.get('free_transfers', 1)
        
        print(f"Budget: £{budget:.1f}m")
        print(f"Team value: £{team_value:.1f}m")
        print(f"Free transfers: {free_transfers}")
        print("\nRunning optimizer (7 gameweek horizon)...")
        
        optimization_result = team_optimizer.optimize_team(
            current_team=current_team,
            current_budget=budget,
            free_transfers=free_transfers,
            horizon_gws=7,
            verbose=False,
            risk_aversion=0.5,
            min_chance_of_playing=75,
            current_team_value=team_value,
            no_hits=False
        )
        
        if optimization_result and optimization_result.get('status') == 'Optimal':
            # Get first gameweek plan
            first_gw_plan = optimization_result['weekly_plans'][0]
            
            print(f"\n[OK] Optimizer completed!")
            print(f"Expected total points: {optimization_result.get('objective_value', 0):.1f}")
            print(f"\nGW{first_gw_plan['gameweek']} suggested transfers:")
            
            transfers_in = first_gw_plan['transfers']['in']
            transfers_out = first_gw_plan['transfers']['out']
            
            if len(transfers_in) > 0:
                print("\n  Transfers IN:")
                for t in transfers_in:
                    print(f"    -> {t['player_name']:20s} ({t['position']:3s}, {t['team']:3s}) - £{t['cost']:.1f}m")
                
                print("\n  Transfers OUT:")
                for t in transfers_out:
                    print(f"    <- {t['player_name']:20s} ({t['position']:3s}, {t['team']:3s}) - £{t['cost']:.1f}m")
                
                # Check if our target players are involved
                transfer_in_names = [t['player_name'] for t in transfers_in]
                transfer_out_names = [t['player_name'] for t in transfers_out]
                
                print("\n  Analysis:")
                if any('Enzo' in name for name in transfer_in_names):
                    print("    [!] Bot IS suggesting Enzo IN")
                if any('Sarr' in name for name in transfer_out_names):
                    print("    [!] Bot IS suggesting Sarr OUT")
                if any('Mateta' in name for name in transfer_in_names):
                    print("    [+] Bot IS suggesting Mateta IN")
                if any('Ekitike' in name for name in transfer_out_names):
                    print("    [+] Bot IS suggesting Ekitike OUT")
                
                # Check future weeks for planned transfers
                print("\n  Future gameweek plans:")
                for plan in optimization_result['weekly_plans'][1:4]:  # Show next 3 GWs
                    if plan['transfers']['count'] > 0:
                        gw = plan['gameweek']
                        ins = [t['player_name'] for t in plan['transfers']['in']]
                        outs = [t['player_name'] for t in plan['transfers']['out']]
                        print(f"    GW{gw}: {', '.join(outs)} OUT -> {', '.join(ins)} IN")
            else:
                print("  No transfers suggested")
        else:
            print(f"\n[ERROR] Optimizer failed: {optimization_result.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"\n[ERROR] Error running optimizer: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 100)
    print("Debug complete!")
    print("\nTo investigate further, check:")
    print("  1. Are Sarr and Mateta both from Crystal Palace? (team limit)")
    print("  2. Is Ekitike on the bench? (bench players have 10% value)")
    print("  3. Does Enzo have much better xG/xA than Sarr?")
    print("  4. Are there budget constraints requiring Sarr out first?")


if __name__ == '__main__':
    manager_id = 789800  # Default manager ID
    player_names = None  # Will use default: Sarr, Enzo, Mateta, Ekitike
    
    if len(sys.argv) > 1:
        manager_id = int(sys.argv[1])
    
    if len(sys.argv) > 2:
        # Allow custom player names as arguments
        player_names = sys.argv[2:]
    
    debug_transfer_recommendations(manager_id, player_names)

