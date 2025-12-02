"""
Debug script to investigate why a specific trade is being recommended

This script will:
1. Run the optimizer
2. Extract the trade recommendation
3. Compare predicted points for players involved
4. Show detailed analysis of why the trade is recommended
"""

import sys
from pathlib import Path

# Add bot directory to path
bot_path = Path(__file__).parent / 'bot'
sys.path.insert(0, str(bot_path))

from bot.core.data_collector import DataCollector
from bot.core.team_optimizer import TeamOptimizer
from bot.core.manager_analyzer import ManagerAnalyzer
import pandas as pd


def find_player_by_name(players_df, name_pattern):
    """Find players matching a name pattern"""
    name_lower = name_pattern.lower()
    matches = players_df[
        players_df['web_name'].str.lower().str.contains(name_lower, na=False) |
        players_df['first_name'].str.lower().str.contains(name_lower, na=False) |
        players_df['second_name'].str.lower().str.contains(name_lower, na=False)
    ]
    return matches


def analyze_trade_recommendation(manager_id, verbose=True):
    """Analyze why a trade is being recommended"""
    
    print("="*80)
    print("TRADE RECOMMENDATION DEBUGGER")
    print("="*80)
    
    # Initialize components
    data_collector = DataCollector()
    manager_analyzer = ManagerAnalyzer(data_collector)
    team_optimizer = TeamOptimizer(data_collector)
    
    # Get manager data
    print(f"\n[1] Fetching manager data for ID {manager_id}...")
    manager_analysis = manager_analyzer.analyze_manager(manager_id)
    
    if not manager_analysis:
        print("ERROR: Could not fetch manager data")
        return
    
    current_team = manager_analysis['current_team']
    current_budget = manager_analysis['bank']
    free_transfers = manager_analysis['free_transfers']
    
    print(f"  Current team: {len(current_team)} players")
    print(f"  Bank: £{current_budget:.1f}m")
    print(f"  Free transfers: {free_transfers}")
    
    # Get player data
    print(f"\n[2] Fetching player data...")
    players_df = data_collector.get_player_data(include_set_pieces=True)
    fixtures_df = data_collector.get_fixtures()
    team_strengths_df = data_collector.get_team_strengths()
    season_data = data_collector.get_current_season_data()
    
    current_gw = team_optimizer._get_current_gameweek(season_data)
    print(f"  Current gameweek: {current_gw}")
    print(f"  Total players: {len(players_df)}")
    
    # Find the players mentioned
    print(f"\n[3] Searching for players...")
    players_to_find = ['van de ven', 'bet', 'de cupyer', 'nmecha']
    found_players = {}
    
    for name_pattern in players_to_find:
        matches = find_player_by_name(players_df, name_pattern)
        if len(matches) > 0:
            print(f"\n  Found matches for '{name_pattern}':")
            for idx, player in matches.iterrows():
                player_id = player['id']
                is_in_team = player_id in current_team
                print(f"    - {player['web_name']} (ID: {player_id}, Team: {player['team_short_name']}, "
                      f"Position: {player['position_name']}, Cost: £{player['now_cost']/10:.1f}m, "
                      f"In team: {is_in_team})")
                found_players[name_pattern] = {
                    'player_id': player_id,
                    'name': player['web_name'],
                    'team': player['team_short_name'],
                    'position': player['position_name'],
                    'cost': player['now_cost'] / 10.0,
                    'in_team': is_in_team,
                    'data': player
                }
        else:
            print(f"  No matches found for '{name_pattern}'")
    
    # Run optimizer
    print(f"\n[4] Running optimizer...")
    print("  (This may take a minute...)")
    
    result = team_optimizer.optimize_team(
        current_team=current_team,
        current_budget=current_budget,
        free_transfers=free_transfers,
        verbose=verbose,
        risk_aversion=0.5,
        min_chance_of_playing=75
    )
    
    if result.get('error'):
        print(f"  ERROR: {result['error']}")
        return
    
    # Extract first gameweek transfers
    first_gw_plan = result.get('weekly_plans', [{}])[0]
    transfers = first_gw_plan.get('transfers', {})
    transfers_in = transfers.get('in', [])
    transfers_out = transfers.get('out', [])
    
    print(f"\n[5] Trade Recommendations:")
    print(f"  Transfers OUT: {len(transfers_out)}")
    for t_out in transfers_out:
        print(f"    - {t_out['player_name']} ({t_out['position']}, {t_out['team']}, £{t_out['cost']:.1f}m)")
    
    print(f"  Transfers IN: {len(transfers_in)}")
    for t_in in transfers_in:
        print(f"    - {t_in['player_name']} ({t_in['position']}, {t_in['team']}, £{t_in['cost']:.1f}m)")
    
    # Check if the specific trade is being recommended
    print(f"\n[6] Checking for specific trade (van de ven + bet -> de cupyer + nmecha)...")
    
    vdv_id = found_players.get('van de ven', {}).get('player_id')
    bet_id = found_players.get('bet', {}).get('player_id')
    dcup_id = found_players.get('de cupyer', {}).get('player_id')
    nmecha_id = found_players.get('nmecha', {}).get('player_id')
    
    transfer_out_ids = [t['player_id'] for t in transfers_out]
    transfer_in_ids = [t['player_id'] for t in transfers_in]
    
    is_vdv_out = vdv_id and vdv_id in transfer_out_ids
    is_bet_out = bet_id and bet_id in transfer_out_ids
    is_dcup_in = dcup_id and dcup_id in transfer_in_ids
    is_nmecha_in = nmecha_id and nmecha_id in transfer_in_ids
    
    if (is_vdv_out and is_bet_out and is_dcup_in and is_nmecha_in):
        print("  ✓ CONFIRMED: This exact trade is being recommended!")
    else:
        print("  ✗ This exact trade is NOT being recommended")
        if is_vdv_out:
            print("    - van de ven IS being transferred out")
        if is_bet_out:
            print("    - bet IS being transferred out")
        if is_dcup_in:
            print("    - de cupyer IS being transferred in")
        if is_nmecha_in:
            print("    - nmecha IS being transferred in")
    
    # Generate predictions for comparison
    print(f"\n[7] Generating predicted points for comparison...")
    gameweeks = list(range(current_gw, min(current_gw + 7, 39)))
    
    predictions = team_optimizer._predict_points(
        players_df, fixtures_df, team_strengths_df, gameweeks, 
        risk_aversion=0.5, verbose=False
    )
    
    # Compare predicted points
    print(f"\n[8] Predicted Points Comparison (next 7 gameweeks):")
    print("="*80)
    
    players_to_compare = []
    if vdv_id:
        players_to_compare.append(('van de ven', vdv_id, is_vdv_out))
    if bet_id:
        players_to_compare.append(('bet', bet_id, is_bet_out))
    if dcup_id:
        players_to_compare.append(('de cupyer', dcup_id, is_dcup_in))
    if nmecha_id:
        players_to_compare.append(('nmecha', nmecha_id, is_nmecha_in))
    
    comparison_data = []
    for name, pid, is_traded in players_to_compare:
        player_preds = predictions[predictions['player_id'] == pid]
        total_points = player_preds['predicted_points'].sum()
        avg_points = player_preds['predicted_points'].mean() if len(player_preds) > 0 else 0
        
        player_data = players_df[players_df['id'] == pid].iloc[0]
        cost = player_data['now_cost'] / 10.0
        position = player_data['position_name']
        team = player_data['team_short_name']
        
        comparison_data.append({
            'name': name,
            'player_id': pid,
            'position': position,
            'team': team,
            'cost': cost,
            'total_points_7gw': total_points,
            'avg_points_per_gw': avg_points,
            'points_per_million': total_points / cost if cost > 0 else 0,
            'is_traded': is_traded
        })
        
        print(f"\n  {name.upper()} (ID: {pid})")
        print(f"    Position: {position}, Team: {team}, Cost: £{cost:.1f}m")
        print(f"    Total predicted points (7 GWs): {total_points:.2f}")
        print(f"    Average per gameweek: {avg_points:.2f}")
        print(f"    Points per million: {total_points/cost:.2f}" if cost > 0 else "    Points per million: N/A")
        print(f"    Status: {'TRADED OUT' if is_traded and pid in transfer_out_ids else 'TRADED IN' if is_traded and pid in transfer_in_ids else 'KEPT'}")
        
        # Show per-gameweek breakdown
        print(f"    Per-gameweek breakdown:")
        for gw in gameweeks[:5]:  # Show first 5 GWs
            gw_pred = player_preds[player_preds['gameweek'] == gw]
            if len(gw_pred) > 0:
                pts = gw_pred.iloc[0]['predicted_points']
                conf = gw_pred.iloc[0]['confidence']
                print(f"      GW{gw}: {pts:.2f} pts (confidence: {conf:.2f})")
    
    # Calculate trade value
    print(f"\n[9] Trade Value Analysis:")
    print("="*80)
    
    if vdv_id and bet_id and dcup_id and nmecha_id:
        vdv_data = next((d for d in comparison_data if d['name'] == 'van de ven'), None)
        bet_data = next((d for d in comparison_data if d['name'] == 'bet'), None)
        dcup_data = next((d for d in comparison_data if d['name'] == 'de cupyer'), None)
        nmecha_data = next((d for d in comparison_data if d['name'] == 'nmecha'), None)
        
        if all([vdv_data, bet_data, dcup_data, nmecha_data]):
            out_total = vdv_data['total_points_7gw'] + bet_data['total_points_7gw']
            in_total = dcup_data['total_points_7gw'] + nmecha_data['total_points_7gw']
            net_gain = in_total - out_total
            
            out_cost = vdv_data['cost'] + bet_data['cost']
            in_cost = dcup_data['cost'] + nmecha_data['cost']
            cost_diff = in_cost - out_cost
            
            print(f"\n  Trading OUT:")
            print(f"    van de ven: {vdv_data['total_points_7gw']:.2f} pts (£{vdv_data['cost']:.1f}m)")
            print(f"    bet: {bet_data['total_points_7gw']:.2f} pts (£{bet_data['cost']:.1f}m)")
            print(f"    TOTAL: {out_total:.2f} pts (£{out_cost:.1f}m)")
            
            print(f"\n  Trading IN:")
            print(f"    de cupyer: {dcup_data['total_points_7gw']:.2f} pts (£{dcup_data['cost']:.1f}m)")
            print(f"    nmecha: {nmecha_data['total_points_7gw']:.2f} pts (£{nmecha_data['cost']:.1f}m)")
            print(f"    TOTAL: {in_total:.2f} pts (£{in_cost:.1f}m)")
            
            print(f"\n  NET RESULT:")
            print(f"    Points gain: {net_gain:+.2f} pts")
            print(f"    Cost difference: {cost_diff:+.1f}m")
            
            if net_gain > 0:
                print(f"\n  ✓ Trade appears beneficial: +{net_gain:.2f} points over 7 gameweeks")
            else:
                print(f"\n  ✗ Trade appears detrimental: {net_gain:.2f} points over 7 gameweeks")
                print(f"    This suggests the optimizer may be making a mistake or")
                print(f"    considering factors beyond just predicted points (e.g., budget, formation, etc.)")
    
    # Check position constraints
    print(f"\n[10] Position Analysis:")
    print("="*80)
    
    if vdv_id and bet_id and dcup_id and nmecha_id:
        vdv_pos = players_df[players_df['id'] == vdv_id].iloc[0]['position_name']
        bet_pos = players_df[players_df['id'] == bet_id].iloc[0]['position_name']
        dcup_pos = players_df[players_df['id'] == dcup_id].iloc[0]['position_name']
        nmecha_pos = players_df[players_df['id'] == nmecha_id].iloc[0]['position_name']
        
        print(f"\n  Position changes:")
        print(f"    OUT: {vdv_pos} (van de ven) + {bet_pos} (bet)")
        print(f"    IN: {dcup_pos} (de cupyer) + {nmecha_pos} (nmecha)")
        
        # Check if this maintains position balance
        out_positions = [vdv_pos, bet_pos]
        in_positions = [dcup_pos, nmecha_pos]
        
        if sorted(out_positions) == sorted(in_positions):
            print(f"    ✓ Position balance maintained")
        else:
            print(f"    ⚠ Position balance changed - this may affect formation")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug trade recommendations')
    parser.add_argument('manager_id', type=int, help='FPL Manager ID')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    analyze_trade_recommendation(args.manager_id, verbose=args.verbose)

