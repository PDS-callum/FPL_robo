"""
Check ACTUAL performance vs our predictions

User says PL app shows Sarr's average points > Enzo's average points
But we're predicting Enzo > Sarr

Let's check the actual data!
"""

import sys
from bot.core.data_collector import DataCollector

if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'ignore')

def check_actual_performance():
    print("="*100)
    print("ACTUAL PERFORMANCE CHECK")
    print("="*100)
    
    data_collector = DataCollector()
    players_df = data_collector.get_player_data(include_set_pieces=True)
    season_data = data_collector.get_current_season_data()
    
    # Get current gameweek
    current_gw = None
    for event in season_data.get('events', []):
        if not event.get('finished', False):
            current_gw = event.get('id')
            break
    
    games_played = current_gw - 1 if current_gw else 11
    
    # Find players
    sarr = players_df[players_df['web_name'].str.contains('Sarr', case=False) & (players_df['team_short_name'] == 'CRY')].iloc[0]
    enzo = players_df[players_df['web_name'] == 'Enzo'].iloc[0]
    mateta = players_df[players_df['web_name'] == 'Mateta'].iloc[0]
    ekitike = players_df[players_df['web_name'].str.contains('Ekitik', case=False)].iloc[0]
    
    print(f"\nGameweeks played so far: ~{games_played}")
    print("\n" + "="*100)
    print("ACTUAL SEASON PERFORMANCE (from FPL API):")
    print("="*100)
    
    for name, player in [('Sarr', sarr), ('Enzo', enzo), ('Mateta', mateta), ('Ekitike', ekitike)]:
        total_points = float(player.get('total_points', 0))
        minutes = float(player.get('minutes', 1))
        games_played_est = minutes / 90 if minutes > 0 else 1
        
        # Points per game (using estimated games from minutes)
        ppg = total_points / games_played_est if games_played_est > 0 else 0
        
        # Points per 90 minutes
        pp90 = (total_points / minutes * 90) if minutes > 0 else 0
        
        # Form (last few games - from API)
        form = float(player.get('form', 0))
        
        print(f"\n{name} ({player['team_short_name']} {player['position_name']}):")
        print(f"  Total points: {total_points:.0f}")
        print(f"  Minutes: {minutes:.0f}")
        print(f"  Est. games played: {games_played_est:.1f}")
        print(f"  Points per game: {ppg:.2f}")
        print(f"  Points per 90 min: {pp90:.2f}")
        print(f"  Form (recent): {form}")
        print(f"  Points per million: {float(player.get('points_per_game', 0)):.2f} (API)")
    
    print("\n" + "="*100)
    print("COMPARISON:")
    print("="*100)
    
    sarr_ppg = float(sarr.get('total_points', 0)) / (float(sarr.get('minutes', 1)) / 90)
    enzo_ppg = float(enzo.get('total_points', 0)) / (float(enzo.get('minutes', 1)) / 90)
    
    sarr_form = float(sarr.get('form', 0))
    enzo_form = float(enzo.get('form', 0))
    
    print(f"\nSarr vs Enzo (Points per 90):")
    print(f"  Sarr: {sarr_ppg:.2f} pts/90")
    print(f"  Enzo: {enzo_ppg:.2f} pts/90")
    if sarr_ppg > enzo_ppg:
        print(f"  ^ Sarr is {((sarr_ppg/enzo_ppg - 1)*100):.1f}% better!")
    else:
        print(f"  ^ Enzo is {((enzo_ppg/sarr_ppg - 1)*100):.1f}% better!")
    
    print(f"\nSarr vs Enzo (Form - recent games):")
    print(f"  Sarr form: {sarr_form}")
    print(f"  Enzo form: {enzo_form}")
    if sarr_form > enzo_form:
        print(f"  ^ Sarr is in better form!")
    else:
        print(f"  ^ Enzo is in better form!")
    
    mateta_ppg = float(mateta.get('total_points', 0)) / (float(mateta.get('minutes', 1)) / 90)
    ekitike_ppg = float(ekitike.get('total_points', 0)) / (float(ekitike.get('minutes', 1)) / 90)
    
    print(f"\nEkitike vs Mateta (Points per 90):")
    print(f"  Ekitike: {ekitike_ppg:.2f} pts/90")
    print(f"  Mateta: {mateta_ppg:.2f} pts/90")
    if mateta_ppg > ekitike_ppg:
        print(f"  ^ Mateta is {((mateta_ppg/ekitike_ppg - 1)*100):.1f}% better!")
    else:
        print(f"  ^ Ekitike is {((ekitike_ppg/mateta_ppg - 1)*100):.1f}% better!")
    
    print("\n" + "="*100)
    print("ISSUE IDENTIFIED:")
    print("="*100)
    
    if sarr_ppg > enzo_ppg and sarr_form > enzo_form:
        print(f"\n[!] Sarr is ACTUALLY BETTER than Enzo in:")
        print(f"    1. Points per 90: {sarr_ppg:.2f} vs {enzo_ppg:.2f}")
        print(f"    2. Recent form: {sarr_form} vs {enzo_form}")
        print(f"\n    Yet our predictor says Enzo will score MORE going forward!")
        print(f"    This suggests the predictor is over-weighting xG/xA")
        print(f"    and under-weighting actual performance.")
        print(f"\n    The optimizer's choice of Sarr->Enzo is WRONG!")
        print(f"    It should prefer Ekitike->Mateta.")

if __name__ == '__main__':
    check_actual_performance()


