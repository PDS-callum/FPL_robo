"""
Focused analysis: Why does optimizer prefer Sarr -> Enzo over Ekitike -> Mateta?

This will:
1. Get 7-GW predictions for all 4 players
2. Calculate the actual value comparison
3. Run the optimizer and see what it suggests
4. Determine if it's a bench bias issue or legitimate value difference
"""

import sys
import pandas as pd
from bot.core.data_collector import DataCollector
from bot.core.predictor import PointsPredictor
from bot.core.manager_analyzer import ManagerAnalyzer
from bot.core.team_optimizer import TeamOptimizer

# Set UTF-8 encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'ignore')

def analyze_transfer_choice(manager_id=789800):
    print("="*80)
    print("TRANSFER CHOICE ANALYSIS: Sarr->Enzo vs Ekitike->Mateta")
    print("="*80)
    
    # Initialize
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
    
    current_gw = None
    for event in season_data.get('events', []):
        if not event.get('finished', False):
            current_gw = event.get('id')
            break
    
    print(f"Current GW: {current_gw}")
    
    # Get manager team
    print("\n2. Analyzing manager team...")
    manager_analysis = manager_analyzer.analyze_manager(manager_id)
    current_team = [pick['element'] for pick in manager_analysis.get('current_team', {}).get('picks', [])]
    
    # Find the players
    sarr = players_df[players_df['web_name'].str.contains('Sarr', case=False) & (players_df['team_short_name'] == 'CRY')]
    sarr = sarr.iloc[0] if not sarr.empty else None
    
    ekitike = players_df[players_df['web_name'].str.contains('Ekitik', case=False)]
    ekitike = ekitike.iloc[0] if not ekitike.empty else None
    
    enzo = players_df[players_df['web_name'] == 'Enzo']
    enzo = enzo.iloc[0] if not enzo.empty else None
    
    mateta = players_df[players_df['web_name'] == 'Mateta']
    mateta = mateta.iloc[0] if not mateta.empty else None
    
    if not all([sarr is not None, ekitike is not None, enzo is not None, mateta is not None]):
        print("ERROR: Could not find all players")
        print(f"Sarr: {sarr is not None}, Ekitike: {ekitike is not None}, Enzo: {enzo is not None}, Mateta: {mateta is not None}")
        return
    
    print(f"\nFound players:")
    print(f"  Sarr (CRY MID): ID {sarr['id']}, £{sarr['now_cost']/10:.1f}m")
    print(f"  Ekitike (LIV FWD): ID {ekitike['id']}, £{ekitike['now_cost']/10:.1f}m")
    print(f"  Enzo (CHE MID): ID {enzo['id']}, £{enzo['now_cost']/10:.1f}m")
    print(f"  Mateta (CRY FWD): ID {mateta['id']}, £{mateta['now_cost']/10:.1f}m")
    
    # Generate predictions for 7 gameweeks
    print(f"\n3. Generating predictions (GW{current_gw} to GW{current_gw+6})...")
    gameweeks = list(range(current_gw, current_gw + 7))
    
    all_predictions = []
    for gw in gameweeks:
        gw_predictions = predictor.predict_gameweek_points(
            players_df, fixtures_df, team_strengths_df, gw
        )
        all_predictions.append(gw_predictions)
    
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    
    # Get predictions for each player
    sarr_preds = predictions_df[predictions_df['player_id'] == sarr['id']]
    ekitike_preds = predictions_df[predictions_df['player_id'] == ekitike['id']]
    enzo_preds = predictions_df[predictions_df['player_id'] == enzo['id']]
    mateta_preds = predictions_df[predictions_df['player_id'] == mateta['id']]
    
    sarr_total = sarr_preds['predicted_points'].sum()
    ekitike_total = ekitike_preds['predicted_points'].sum()
    enzo_total = enzo_preds['predicted_points'].sum()
    mateta_total = mateta_preds['predicted_points'].sum()
    
    print(f"\n4. 7-Gameweek Predictions:")
    print(f"  Sarr:    {sarr_total:.2f} points")
    print(f"  Ekitike: {ekitike_total:.2f} points")
    print(f"  Enzo:    {enzo_total:.2f} points")
    print(f"  Mateta:  {mateta_total:.2f} points")
    
    # Calculate transfer value
    print(f"\n5. Transfer Value Analysis:")
    print(f"\nOption A: Sarr OUT -> Enzo IN")
    print(f"  Current: Sarr {sarr_total:.2f} pts")
    print(f"  New:     Enzo {enzo_total:.2f} pts")
    print(f"  Gain:    {enzo_total - sarr_total:+.2f} pts over 7 GWs")
    print(f"  Cost:    £{(enzo['now_cost'] - sarr['now_cost'])/10:+.1f}m")
    
    print(f"\nOption B: Ekitike OUT -> Mateta IN")
    print(f"  Current: Ekitike {ekitike_total:.2f} pts")
    print(f"  New:     Mateta {mateta_total:.2f} pts")
    print(f"  Gain:    {mateta_total - ekitike_total:+.2f} pts over 7 GWs")
    print(f"  Cost:    £{(mateta['now_cost'] - ekitike['now_cost'])/10:+.1f}m")
    
    # Verdict
    option_a_gain = enzo_total - sarr_total
    option_b_gain = mateta_total - ekitike_total
    
    print(f"\n6. Verdict:")
    if option_b_gain > option_a_gain:
        print(f"  Option B is BETTER by {option_b_gain - option_a_gain:.2f} points!")
        print(f"  [!] If optimizer suggests Option A, there's a problem")
    else:
        print(f"  Option A is better by {option_a_gain - option_b_gain:.2f} points")
        print(f"  Optimizer is working correctly")
    
    # Run optimizer to see what it actually suggests
    print(f"\n7. Running optimizer...")
    manager_info = manager_analysis.get('manager_info', {})
    budget = manager_info.get('bank', 0.0)
    team_value = manager_info.get('team_value', 0.0)
    free_transfers = manager_analysis.get('saved_transfers', {}).get('free_transfers', 1)
    
    result = team_optimizer.optimize_team(
        current_team=current_team,
        current_budget=budget,
        free_transfers=free_transfers,
        horizon_gws=7,
        verbose=False,
        current_team_value=team_value
    )
    
    if result and result.get('status') == 'Optimal':
        first_gw = result['weekly_plans'][0]
        transfers_in = first_gw['transfers']['in']
        transfers_out = first_gw['transfers']['out']
        
        print(f"\n8. Optimizer Recommendation (GW{first_gw['gameweek']}):")
        if len(transfers_in) > 0:
            for t_out, t_in in zip(transfers_out, transfers_in):
                print(f"  {t_out['player_name']} OUT -> {t_in['player_name']} IN")
                
                # Check which option it matches
                if 'Enzo' in t_in['player_name']:
                    print(f"    ^ This is OPTION A (Sarr/Enzo)")
                elif 'Mateta' in t_in['player_name']:
                    print(f"    ^ This is OPTION B (Ekitike/Mateta)")
        else:
            print("  No transfers suggested")
        
        # Show starting XI to see if Ekitike would start after Mateta transfer
        print(f"\n9. Predicted Starting XI:")
        starting_11 = sorted(first_gw['squad']['starting_11'], 
                           key=lambda x: x['predicted_points'], reverse=True)
        for i, player in enumerate(starting_11, 1):
            marker = ""
            if 'Mateta' in player['player_name'] or 'Enzo' in player['player_name']:
                marker = " [NEW]"
            elif 'Sarr' in player['player_name'] or 'Ekitik' in player['player_name']:
                marker = " [OLD]"
            print(f"  {i:2d}. {player['player_name']:20s} ({player['position']:3s}) - {player['predicted_points']:.1f} pts{marker}")
        
        print(f"\nBench:")
        bench = first_gw['squad']['bench']
        for i, player in enumerate(bench, 1):
            marker = ""
            if 'Mateta' in player['player_name'] or 'Enzo' in player['player_name']:
                marker = " [NEW]"
            elif 'Sarr' in player['player_name'] or 'Ekitik' in player['player_name']:
                marker = " [OLD]"
            print(f"  {i}. {player['player_name']:20s} ({player['position']:3s}) - {player['predicted_points']:.1f} pts{marker}")
        
        # Check future gameweeks for Mateta transfer
        print(f"\n10. Future Gameweek Plans:")
        print("=" * 80)
        for plan in result['weekly_plans'][1:min(4, len(result['weekly_plans']))]:
            gw = plan['gameweek']
            transfers = plan['transfers']
            if transfers['count'] > 0:
                print(f"\nGW{gw}: {transfers['count']} transfer(s)")
                for t_in, t_out in zip(transfers['in'], transfers['out']):
                    print(f"  {t_out['player_name']} OUT -> {t_in['player_name']} IN")
                    if 'Mateta' in t_in['player_name']:
                        print(f"    ^ MATETA COMING IN HERE!")
            else:
                print(f"\nGW{gw}: No transfers")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    manager_id = 789800
    if len(sys.argv) > 1:
        manager_id = int(sys.argv[1])
    
    analyze_transfer_choice(manager_id)

