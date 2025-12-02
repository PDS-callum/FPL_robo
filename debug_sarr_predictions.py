"""
Debug why Sarr's predictions seem wrong

User points:
1. Sarr has easy fixtures (playing worst teams)
2. Sarr and Mateta are BOTH on Crystal Palace, playing the SAME fixtures
3. Yet Sarr is predicted much lower (4.20) than Mateta (5.05) in GW12
4. Why is Sarr predicted lower than Enzo when Sarr has easier fixtures?
"""

import sys
import pandas as pd
from bot.core.data_collector import DataCollector
from bot.core.predictor import PointsPredictor

if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'ignore')

def debug_sarr_predictions():
    print("="*100)
    print("SARR PREDICTION DEBUG")
    print("="*100)
    
    data_collector = DataCollector()
    predictor = PointsPredictor()
    
    players_df = data_collector.get_player_data(include_set_pieces=True)
    fixtures_df = data_collector.get_fixtures()
    team_strengths_df = data_collector.get_team_strengths()
    season_data = data_collector.get_current_season_data()
    
    current_gw = None
    for event in season_data.get('events', []):
        if not event.get('finished', False):
            current_gw = event.get('id')
            break
    
    # Find Sarr, Mateta, Enzo, Ekitike
    sarr = players_df[players_df['web_name'].str.contains('Sarr', case=False) & (players_df['team_short_name'] == 'CRY')].iloc[0]
    mateta = players_df[players_df['web_name'] == 'Mateta'].iloc[0]
    enzo = players_df[players_df['web_name'] == 'Enzo'].iloc[0]
    ekitike = players_df[players_df['web_name'].str.contains('Ekitik', case=False)].iloc[0]
    
    print(f"\n1. Player Details:")
    print(f"\nSarr (CRY MID):")
    print(f"  Total points: {sarr.get('total_points', 0)}")
    print(f"  Minutes: {sarr.get('minutes', 0)}")
    print(f"  Goals: {sarr.get('goals_scored', 0)}")
    print(f"  Assists: {sarr.get('assists', 0)}")
    print(f"  Form: {sarr.get('form', 0)}")
    print(f"  xG per 90: {sarr.get('expected_goals_per_90', 'N/A')}")
    print(f"  xA per 90: {sarr.get('expected_assists_per_90', 'N/A')}")
    print(f"  On penalties: {sarr.get('on_penalties', False)}")
    
    print(f"\nMateta (CRY FWD):")
    print(f"  Total points: {mateta.get('total_points', 0)}")
    print(f"  Minutes: {mateta.get('minutes', 0)}")
    print(f"  Goals: {mateta.get('goals_scored', 0)}")
    print(f"  Assists: {mateta.get('assists', 0)}")
    print(f"  Form: {mateta.get('form', 0)}")
    print(f"  xG per 90: {mateta.get('expected_goals_per_90', 'N/A')}")
    print(f"  xA per 90: {mateta.get('expected_assists_per_90', 'N/A')}")
    print(f"  On penalties: {mateta.get('on_penalties', False)}")
    
    print(f"\nEnzo (CHE MID):")
    print(f"  Total points: {enzo.get('total_points', 0)}")
    print(f"  Minutes: {enzo.get('minutes', 0)}")
    print(f"  Goals: {enzo.get('goals_scored', 0)}")
    print(f"  Assists: {enzo.get('assists', 0)}")
    print(f"  Form: {enzo.get('form', 0)}")
    print(f"  xG per 90: {enzo.get('expected_goals_per_90', 'N/A')}")
    print(f"  xA per 90: {enzo.get('expected_assists_per_90', 'N/A')}")
    print(f"  On penalties: {enzo.get('on_penalties', False)}")
    
    # Check fixtures
    print(f"\n2. GW{current_gw} Fixtures:")
    
    # CRY fixture (Sarr and Mateta)
    cry_fixture = fixtures_df[
        ((fixtures_df['team_h'] == sarr['team']) | (fixtures_df['team_a'] == sarr['team'])) &
        (fixtures_df['event'] == current_gw)
    ]
    
    if not cry_fixture.empty:
        fixture = cry_fixture.iloc[0]
        is_home = fixture['team_h'] == sarr['team']
        opponent_id = fixture['team_a'] if is_home else fixture['team_h']
        opponent = players_df[players_df['team'] == opponent_id]['team_short_name'].iloc[0]
        difficulty = fixture.get('team_h_difficulty' if is_home else 'team_a_difficulty', 'N/A')
        
        print(f"\nCrystal Palace (Sarr & Mateta):")
        print(f"  Opponent: {opponent}")
        print(f"  Venue: {'Home' if is_home else 'Away'}")
        print(f"  FPL Difficulty: {difficulty}")
        
        # Get opponent league position
        opponent_data = season_data.get('teams', [])
        opponent_team = next((t for t in opponent_data if t['id'] == opponent_id), None)
        if opponent_team:
            print(f"  Opponent position: {opponent_team.get('position', 'N/A')}")
    
    # CHE fixture (Enzo)
    che_fixture = fixtures_df[
        ((fixtures_df['team_h'] == enzo['team']) | (fixtures_df['team_a'] == enzo['team'])) &
        (fixtures_df['event'] == current_gw)
    ]
    
    if not che_fixture.empty:
        fixture = che_fixture.iloc[0]
        is_home = fixture['team_h'] == enzo['team']
        opponent_id = fixture['team_a'] if is_home else fixture['team_h']
        opponent = players_df[players_df['team'] == opponent_id]['team_short_name'].iloc[0]
        difficulty = fixture.get('team_h_difficulty' if is_home else 'team_a_difficulty', 'N/A')
        
        print(f"\nChelsea (Enzo):")
        print(f"  Opponent: {opponent}")
        print(f"  Venue: {'Home' if is_home else 'Away'}")
        print(f"  FPL Difficulty: {difficulty}")
        
        opponent_team = next((t for t in opponent_data if t['id'] == opponent_id), None)
        if opponent_team:
            print(f"  Opponent position: {opponent_team.get('position', 'N/A')}")
    
    # Generate predictions
    print(f"\n3. Detailed Prediction Breakdown (GW{current_gw}):")
    
    team_stats = predictor._build_team_stats(players_df, fixtures_df, current_gw)
    
    for player_name, player in [('Sarr', sarr), ('Mateta', mateta), ('Enzo', enzo)]:
        print(f"\n{player_name}:")
        
        # Get fixture
        if player_name in ['Sarr', 'Mateta']:
            fixture_data = cry_fixture.iloc[0]
        else:
            fixture_data = che_fixture.iloc[0]
        
        fixture_info = predictor._get_player_fixtures(player, pd.DataFrame([fixture_data]))
        if not fixture_info:
            print("  No fixture info")
            continue
        
        fixture_info = fixture_info[0]
        position = player['position_name']
        
        # Minutes probability
        minutes_prob = predictor._predict_minutes_probability(player, position)
        print(f"  Minutes probability: {minutes_prob:.2%}")
        
        # Goal involvement
        goal_involvement = predictor._predict_goal_involvement(player, fixture_info, team_stats, position)
        print(f"  Expected goals (per 90): {goal_involvement['goals']:.3f}")
        print(f"  Expected assists (per 90): {goal_involvement['assists']:.3f}")
        
        # Clean sheet (if defender)
        if position in ['DEF', 'GK']:
            cs_prob = predictor._predict_clean_sheet_probability(fixture_info, team_stats)
            print(f"  Clean sheet probability: {cs_prob:.2%}")
        
        # Calculate points
        appearance_pts = 2 * minutes_prob
        goal_pts = goal_involvement['goals'] * predictor.POINTS_PER_GOAL[position] * minutes_prob
        assist_pts = goal_involvement['assists'] * 3 * minutes_prob
        
        if position in ['DEF', 'GK']:
            cs_prob = predictor._predict_clean_sheet_probability(fixture_info, team_stats)
            cs_pts = cs_prob * predictor.POINTS_PER_CS[position] * minutes_prob
        else:
            cs_pts = 0
        
        bonus_pts = (goal_involvement['goals'] / minutes_prob if minutes_prob > 0 else 0) * 1.5 * minutes_prob
        if position in ['DEF', 'GK']:
            bonus_pts += cs_prob * 0.3 * minutes_prob
        
        total = appearance_pts + goal_pts + assist_pts + cs_pts + bonus_pts
        
        print(f"\n  Point breakdown:")
        print(f"    Appearance: {appearance_pts:.2f}")
        print(f"    Goals: {goal_pts:.2f}")
        print(f"    Assists: {assist_pts:.2f}")
        print(f"    Clean sheet: {cs_pts:.2f}")
        print(f"    Bonus: {bonus_pts:.2f}")
        print(f"    TOTAL: {total:.2f}")
    
    # Show actual predictions
    print(f"\n4. Actual Predictor Output (GW{current_gw}):")
    gw_preds = predictor.predict_gameweek_points(players_df, fixtures_df, team_strengths_df, current_gw)
    
    for player_name, player_id in [('Sarr', sarr['id']), ('Mateta', mateta['id']), ('Enzo', enzo['id'])]:
        pred = gw_preds[gw_preds['player_id'] == player_id].iloc[0]
        print(f"  {player_name}: {pred['predicted_points']:.2f} pts (confidence: {pred['confidence']:.2f})")
    
    print("\n" + "="*100)
    print("ANALYSIS:")
    print("="*100)
    
    sarr_pred = gw_preds[gw_preds['player_id'] == sarr['id']].iloc[0]['predicted_points']
    mateta_pred = gw_preds[gw_preds['player_id'] == mateta['id']].iloc[0]['predicted_points']
    enzo_pred = gw_preds[gw_preds['player_id'] == enzo['id']].iloc[0]['predicted_points']
    
    print(f"\nSarr and Mateta play for the SAME TEAM (CRY) with SAME FIXTURE")
    print(f"  Sarr predicted: {sarr_pred:.2f}")
    print(f"  Mateta predicted: {mateta_pred:.2f}")
    print(f"  Difference: {mateta_pred - sarr_pred:.2f} (Mateta {((mateta_pred/sarr_pred - 1)*100):.1f}% higher)")
    
    print(f"\nWhy is Mateta predicted {mateta_pred - sarr_pred:.2f} pts higher for the SAME fixture?")
    print(f"  Key factors:")
    print(f"    1. Position: Mateta is FWD (4 pts/goal), Sarr is MID (5 pts/goal)")
    print(f"    2. Mateta xG per 90: {mateta.get('expected_goals_per_90', 'N/A')}")
    print(f"    3. Sarr xG per 90: {sarr.get('expected_goals_per_90', 'N/A')}")
    print(f"    4. Minutes: Mateta {mateta.get('minutes', 0)}, Sarr {sarr.get('minutes', 0)}")
    
    if float(sarr.get('minutes', 0)) < 180:
        print(f"\n  [!] ISSUE: Sarr has < 180 minutes!")
        print(f"      Predictor uses conservative baseline for players with < 180 mins")
        print(f"      This might be undervaluing Sarr significantly")

if __name__ == '__main__':
    debug_sarr_predictions()


