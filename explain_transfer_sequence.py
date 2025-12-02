"""
Explain why optimizer chooses Sarr->Enzo first, then Ekitike->Mateta

Shows gameweek-by-gameweek breakdown to see which sequence is truly better
"""

import sys
import pandas as pd
from bot.core.data_collector import DataCollector
from bot.core.predictor import PointsPredictor
from bot.core.manager_analyzer import ManagerAnalyzer

if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'ignore')

def explain_sequence(manager_id=789800):
    print("="*90)
    print("TRANSFER SEQUENCE ANALYSIS")
    print("="*90)
    
    # Initialize
    data_collector = DataCollector()
    predictor = PointsPredictor()
    
    # Get data
    players_df = data_collector.get_player_data(include_set_pieces=True)
    fixtures_df = data_collector.get_fixtures()
    team_strengths_df = data_collector.get_team_strengths()
    season_data = data_collector.get_current_season_data()
    
    current_gw = None
    for event in season_data.get('events', []):
        if not event.get('finished', False):
            current_gw = event.get('id')
            break
    
    # Find players
    sarr = players_df[players_df['web_name'].str.contains('Sarr', case=False) & (players_df['team_short_name'] == 'CRY')].iloc[0]
    ekitike = players_df[players_df['web_name'].str.contains('Ekitik', case=False)].iloc[0]
    enzo = players_df[players_df['web_name'] == 'Enzo'].iloc[0]
    mateta = players_df[players_df['web_name'] == 'Mateta'].iloc[0]
    
    # Generate GW-by-GW predictions
    gameweeks = list(range(current_gw, current_gw + 7))
    
    print(f"\nGameweek-by-Gameweek Predictions:")
    print(f"{'GW':<6} {'Sarr':<8} {'Enzo':<8} {'Ekitike':<8} {'Mateta':<8}")
    print("-" * 90)
    
    sarr_by_gw = []
    enzo_by_gw = []
    ekitike_by_gw = []
    mateta_by_gw = []
    
    for gw in gameweeks:
        gw_preds = predictor.predict_gameweek_points(
            players_df, fixtures_df, team_strengths_df, gw
        )
        
        sarr_pts = gw_preds[gw_preds['player_id'] == sarr['id']]['predicted_points'].values[0]
        enzo_pts = gw_preds[gw_preds['player_id'] == enzo['id']]['predicted_points'].values[0]
        ekitike_pts = gw_preds[gw_preds['player_id'] == ekitike['id']]['predicted_points'].values[0]
        mateta_pts = gw_preds[gw_preds['player_id'] == mateta['id']]['predicted_points'].values[0]
        
        sarr_by_gw.append(sarr_pts)
        enzo_by_gw.append(enzo_pts)
        ekitike_by_gw.append(ekitike_pts)
        mateta_by_gw.append(mateta_pts)
        
        print(f"GW{gw:<4} {sarr_pts:<8.2f} {enzo_pts:<8.2f} {ekitike_pts:<8.2f} {mateta_pts:<8.2f}")
    
    print("-" * 90)
    print(f"{'TOTAL':<6} {sum(sarr_by_gw):<8.2f} {sum(enzo_by_gw):<8.2f} {sum(ekitike_by_gw):<8.2f} {sum(mateta_by_gw):<8.2f}")
    
    # Calculate both sequences
    print(f"\n" + "="*90)
    print("SEQUENCE A: Optimizer's Choice")
    print("  GW12: Sarr OUT -> Enzo IN")
    print("  GW13: Ekitike OUT -> Mateta IN")
    print("="*90)
    
    seq_a_total = 0
    print(f"\n{'GW':<6} {'Player':<12} {'Points':<8} {'Notes'}")
    print("-" * 90)
    
    for i, gw in enumerate(gameweeks):
        if i == 0:  # GW12
            enzo_contribution = enzo_by_gw[i]
            ekitike_contribution = ekitike_by_gw[i]
            gw_total = enzo_contribution + ekitike_contribution
            print(f"GW{gw:<4} {'Enzo':<12} {enzo_contribution:<8.2f} New transfer")
            print(f"       {'Ekitike':<12} {ekitike_contribution:<8.2f} Kept (will transfer out GW13)")
            print(f"       {'TOTAL':<12} {gw_total:<8.2f}")
            seq_a_total += gw_total
        else:  # GW13+
            enzo_contribution = enzo_by_gw[i]
            mateta_contribution = mateta_by_gw[i]
            gw_total = enzo_contribution + mateta_contribution
            print(f"GW{gw:<4} {'Enzo':<12} {enzo_contribution:<8.2f}")
            print(f"       {'Mateta':<12} {mateta_contribution:<8.2f} {'New in GW13' if i == 1 else ''}")
            print(f"       {'TOTAL':<12} {gw_total:<8.2f}")
            seq_a_total += gw_total
    
    print("-" * 90)
    print(f"SEQUENCE A TOTAL: {seq_a_total:.2f} points")
    
    # Sequence B
    print(f"\n" + "="*90)
    print("SEQUENCE B: User's Preference")
    print("  GW12: Ekitike OUT -> Mateta IN")
    print("  GW13: Sarr OUT -> Enzo IN")
    print("="*90)
    
    seq_b_total = 0
    print(f"\n{'GW':<6} {'Player':<12} {'Points':<8} {'Notes'}")
    print("-" * 90)
    
    for i, gw in enumerate(gameweeks):
        if i == 0:  # GW12
            sarr_contribution = sarr_by_gw[i]
            mateta_contribution = mateta_by_gw[i]
            gw_total = sarr_contribution + mateta_contribution
            print(f"GW{gw:<4} {'Sarr':<12} {sarr_contribution:<8.2f} Kept (will transfer out GW13)")
            print(f"       {'Mateta':<12} {mateta_contribution:<8.2f} New transfer")
            print(f"       {'TOTAL':<12} {gw_total:<8.2f}")
            seq_b_total += gw_total
        else:  # GW13+
            enzo_contribution = enzo_by_gw[i]
            mateta_contribution = mateta_by_gw[i]
            gw_total = enzo_contribution + mateta_contribution
            print(f"GW{gw:<4} {'Enzo':<12} {enzo_contribution:<8.2f} {'New in GW13' if i == 1 else ''}")
            print(f"       {'Mateta':<12} {mateta_contribution:<8.2f}")
            print(f"       {'TOTAL':<12} {gw_total:<8.2f}")
            seq_b_total += gw_total
    
    print("-" * 90)
    print(f"SEQUENCE B TOTAL: {seq_b_total:.2f} points")
    
    # Verdict
    print(f"\n" + "="*90)
    print("VERDICT:")
    print("="*90)
    
    diff = seq_b_total - seq_a_total
    if diff > 0:
        print(f"\nSequence B is BETTER by {diff:.2f} points!")
        print(f"The optimizer should recommend Ekitike -> Mateta first.")
    elif diff < 0:
        print(f"\nSequence A is BETTER by {-diff:.2f} points!")
        print(f"The optimizer is correct to recommend Sarr -> Enzo first.")
        print(f"\nWhy? Look at GW12:")
        gw12_diff = (enzo_by_gw[0] + ekitike_by_gw[0]) - (sarr_by_gw[0] + mateta_by_gw[0])
        print(f"  Seq A GW12: Enzo ({enzo_by_gw[0]:.2f}) + Ekitike ({ekitike_by_gw[0]:.2f}) = {enzo_by_gw[0] + ekitike_by_gw[0]:.2f}")
        print(f"  Seq B GW12: Sarr ({sarr_by_gw[0]:.2f}) + Mateta ({mateta_by_gw[0]:.2f}) = {sarr_by_gw[0] + mateta_by_gw[0]:.2f}")
        print(f"  GW12 difference: {gw12_diff:+.2f} in favor of Sequence A")
        
        # Check if there's a specific GW where the difference matters
        print(f"\nDetailed per-GW comparison:")
        for i, gw in enumerate(gameweeks):
            if i == 0:
                seq_a_gw = enzo_by_gw[i] + ekitike_by_gw[i]
                seq_b_gw = sarr_by_gw[i] + mateta_by_gw[i]
            else:
                seq_a_gw = enzo_by_gw[i] + mateta_by_gw[i]
                seq_b_gw = enzo_by_gw[i] + mateta_by_gw[i]
            
            diff_gw = seq_a_gw - seq_b_gw
            if abs(diff_gw) > 0.01:
                print(f"  GW{gw}: Seq A = {seq_a_gw:.2f}, Seq B = {seq_b_gw:.2f}, Diff = {diff_gw:+.2f}")
    else:
        print(f"\nBoth sequences are EQUAL!")
    
    print("\n" + "="*90)

if __name__ == '__main__':
    manager_id = 789800
    if len(sys.argv) > 1:
        manager_id = int(sys.argv[1])
    
    explain_sequence(manager_id)


