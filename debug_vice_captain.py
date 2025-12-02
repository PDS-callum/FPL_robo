"""
Debug script to investigate why Calafiori is selected as vice captain over Mateta
for manager 789800 in gameweek 12
"""

import sys
import pandas as pd
from bot.core.data_collector import DataCollector
from bot.core.predictor import PointsPredictor
from bot.core.manager_analyzer import ManagerAnalyzer

def debug_vice_captain(manager_id=789800, gameweek=12):
    """Debug vice captain selection for specific manager and gameweek"""
    
    print(f"Debugging vice captain selection for manager {manager_id} in GW{gameweek}")
    print("=" * 80)
    
    # Initialize components
    data_collector = DataCollector()
    predictor = PointsPredictor()
    manager_analyzer = ManagerAnalyzer(data_collector)
    
    # Get data
    print("\n1. Collecting data...")
    players_df = data_collector.get_player_data(include_set_pieces=True)
    fixtures_df = data_collector.get_fixtures()
    team_strengths_df = data_collector.get_team_strengths()
    
    # Get manager team
    print("\n2. Getting manager team...")
    manager_analysis = manager_analyzer.analyze_manager(manager_id)
    current_team = []
    team_data = manager_analysis.get('current_team', {})
    picks = team_data.get('picks', [])
    for pick in picks:
        current_team.append(pick['element'])
    
    # Find Calafiori and Mateta
    print("\n3. Finding Calafiori and Mateta...")
    calafiori = players_df[players_df['web_name'].str.contains('Calafiori', case=False, na=False)]
    mateta = players_df[players_df['web_name'].str.contains('Mateta', case=False, na=False)]
    
    if calafiori.empty:
        print("ERROR: Calafiori not found in players data")
        return
    if mateta.empty:
        print("ERROR: Mateta not found in players data")
        return
    
    calafiori = calafiori.iloc[0]
    mateta = mateta.iloc[0]
    
    print(f"\nCalafiori (ID: {calafiori['id']}):")
    print(f"  Position: {calafiori['position_name']}")
    print(f"  Team: {calafiori['team_short_name']}")
    print(f"  Cost: £{calafiori['now_cost']/10:.1f}m")
    print(f"  Total Points: {calafiori.get('total_points', 0)}")
    print(f"  Goals: {calafiori.get('goals_scored', 0)}")
    print(f"  Assists: {calafiori.get('assists', 0)}")
    print(f"  Minutes: {calafiori.get('minutes', 0)}")
    
    print(f"\nMateta (ID: {mateta['id']}):")
    print(f"  Position: {mateta['position_name']}")
    print(f"  Team: {mateta['team_short_name']}")
    print(f"  Cost: £{mateta['now_cost']/10:.1f}m")
    print(f"  Total Points: {mateta.get('total_points', 0)}")
    print(f"  Goals: {mateta.get('goals_scored', 0)}")
    print(f"  Assists: {mateta.get('assists', 0)}")
    print(f"  Minutes: {mateta.get('minutes', 0)}")
    
    # Check if they're in the team
    print(f"\n4. Checking if players are in team...")
    calafiori_in_team = calafiori['id'] in current_team
    mateta_in_team = mateta['id'] in current_team
    print(f"  Calafiori in team: {calafiori_in_team}")
    print(f"  Mateta in team: {mateta_in_team}")
    
    # Generate predictions
    print(f"\n5. Generating predictions for GW{gameweek}...")
    predictions_df = predictor.predict_gameweek_points(
        players_df, fixtures_df, team_strengths_df, gameweek
    )
    
    # Get predictions for these players
    calafiori_pred = predictions_df[predictions_df['player_id'] == calafiori['id']]
    mateta_pred = predictions_df[predictions_df['player_id'] == mateta['id']]
    
    if calafiori_pred.empty:
        print("ERROR: No prediction found for Calafiori")
        return
    if mateta_pred.empty:
        print("ERROR: No prediction found for Mateta")
        return
    
    calafiori_pred = calafiori_pred.iloc[0]
    mateta_pred = mateta_pred.iloc[0]
    
    print(f"\n6. Predictions for GW{gameweek}:")
    print(f"\nCalafiori:")
    print(f"  Predicted Points: {calafiori_pred['predicted_points']:.2f}")
    print(f"  Confidence: {calafiori_pred['confidence']:.2f}")
    
    print(f"\nMateta:")
    print(f"  Predicted Points: {mateta_pred['predicted_points']:.2f}")
    print(f"  Confidence: {mateta_pred['confidence']:.2f}")
    
    # Get fixture information
    print(f"\n7. Fixture analysis...")
    
    # Get Calafiori's fixture
    calafiori_team = calafiori['team']
    calafiori_fixtures = fixtures_df[
        ((fixtures_df['team_h'] == calafiori_team) | (fixtures_df['team_a'] == calafiori_team)) &
        (fixtures_df['event'] == gameweek)
    ]
    
    if not calafiori_fixtures.empty:
        fixture = calafiori_fixtures.iloc[0]
        is_home = fixture['team_h'] == calafiori_team
        opponent_id = fixture['team_a'] if is_home else fixture['team_h']
        opponent = players_df[players_df['team'] == opponent_id]['team_short_name'].iloc[0] if not players_df[players_df['team'] == opponent_id].empty else "Unknown"
        print(f"\nCalafiori's fixture:")
        print(f"  Opponent: {opponent}")
        print(f"  Venue: {'Home' if is_home else 'Away'}")
        print(f"  Difficulty (H): {fixture.get('team_h_difficulty', 'N/A')}")
        print(f"  Difficulty (A): {fixture.get('team_a_difficulty', 'N/A')}")
    
    # Get Mateta's fixture
    mateta_team = mateta['team']
    mateta_fixtures = fixtures_df[
        ((fixtures_df['team_h'] == mateta_team) | (fixtures_df['team_a'] == mateta_team)) &
        (fixtures_df['event'] == gameweek)
    ]
    
    if not mateta_fixtures.empty:
        fixture = mateta_fixtures.iloc[0]
        is_home = fixture['team_h'] == mateta_team
        opponent_id = fixture['team_a'] if is_home else fixture['team_h']
        opponent = players_df[players_df['team'] == opponent_id]['team_short_name'].iloc[0] if not players_df[players_df['team'] == opponent_id].empty else "Unknown"
        print(f"\nMateta's fixture:")
        print(f"  Opponent: {opponent}")
        print(f"  Venue: {'Home' if is_home else 'Away'}")
        print(f"  Difficulty (H): {fixture.get('team_h_difficulty', 'N/A')}")
        print(f"  Difficulty (A): {fixture.get('team_a_difficulty', 'N/A')}")
    
    # Calculate detailed breakdown
    print(f"\n8. Detailed prediction breakdown...")
    
    # For Calafiori (defender)
    print(f"\nCalafiori breakdown:")
    team_stats = predictor._build_team_stats(players_df, fixtures_df, gameweek)
    if not calafiori_fixtures.empty:
        fixture_info = predictor._get_player_fixtures(calafiori, calafiori_fixtures)
        if fixture_info:
            fixture_info = fixture_info[0]
            minutes_prob = predictor._predict_minutes_probability(calafiori, 'DEF')
            goal_involvement = predictor._predict_goal_involvement(calafiori, fixture_info, team_stats, 'DEF')
            cs_prob = predictor._predict_clean_sheet_probability(fixture_info, team_stats)
            
            print(f"  Minutes probability: {minutes_prob:.2f}")
            print(f"  Expected goals: {goal_involvement['goals']:.3f}")
            print(f"  Expected assists: {goal_involvement['assists']:.3f}")
            print(f"  Clean sheet probability: {cs_prob:.2f}")
            print(f"  Clean sheet points: {cs_prob * 4 * minutes_prob:.2f}")
            
            # Debug: Show opponent attack stats
            opponent_id = fixture_info['opponent_id']
            opponent_stats = team_stats.get(opponent_id, {})
            avg_attack = predictor.AVG_GOALS_PER_GAME / 2
            opponent_attack = opponent_stats.get('avg_goals_scored_away', opponent_stats.get('avg_goals_scored', avg_attack))
            attack_ratio = opponent_attack / avg_attack
            print(f"  Opponent attack: {opponent_attack:.2f} goals/game")
            print(f"  Attack ratio: {attack_ratio:.2f}")
            print(f"  Base CS rate: {team_stats.get(calafiori['team'], {}).get('clean_sheet_rate', 0):.2f}")
            
            # Calculate total
            appearance_pts = 2 * minutes_prob
            goal_pts = goal_involvement['goals'] * 6 * minutes_prob
            assist_pts = goal_involvement['assists'] * 3 * minutes_prob
            cs_pts = cs_prob * 4 * minutes_prob
            bonus_pts = (goal_involvement['goals'] / minutes_prob if minutes_prob > 0 else 0) * 1.5 * minutes_prob + cs_prob * 0.3 * minutes_prob
            
            total_manual = appearance_pts + goal_pts + assist_pts + cs_pts + bonus_pts
            print(f"  Appearance points: {appearance_pts:.2f}")
            print(f"  Goal points: {goal_pts:.2f}")
            print(f"  Assist points: {assist_pts:.2f}")
            print(f"  CS points: {cs_pts:.2f}")
            print(f"  Bonus points: {bonus_pts:.2f}")
            print(f"  Total (manual): {total_manual:.2f}")
            print(f"  Total (predicted): {calafiori_pred['predicted_points']:.2f}")
    
    # For Mateta (forward)
    print(f"\nMateta breakdown:")
    if not mateta_fixtures.empty:
        fixture_info = predictor._get_player_fixtures(mateta, mateta_fixtures)
        if fixture_info:
            fixture_info = fixture_info[0]
            minutes_prob = predictor._predict_minutes_probability(mateta, 'FWD')
            goal_involvement = predictor._predict_goal_involvement(mateta, fixture_info, team_stats, 'FWD')
            
            print(f"  Minutes probability: {minutes_prob:.2f}")
            print(f"  Expected goals: {goal_involvement['goals']:.3f}")
            print(f"  Expected assists: {goal_involvement['assists']:.3f}")
            
            # Calculate total
            appearance_pts = 2 * minutes_prob
            goal_pts = goal_involvement['goals'] * 4 * minutes_prob
            assist_pts = goal_involvement['assists'] * 3 * minutes_prob
            bonus_pts = (goal_involvement['goals'] / minutes_prob if minutes_prob > 0 else 0) * 1.5 * minutes_prob
            
            total_manual = appearance_pts + goal_pts + assist_pts + bonus_pts
            print(f"  Appearance points: {appearance_pts:.2f}")
            print(f"  Goal points: {goal_pts:.2f}")
            print(f"  Assist points: {assist_pts:.2f}")
            print(f"  Bonus points: {bonus_pts:.2f}")
            print(f"  Total (manual): {total_manual:.2f}")
            print(f"  Total (predicted): {mateta_pred['predicted_points']:.2f}")
    
    # Summary
    print(f"\n9. Summary:")
    print(f"  Calafiori predicted: {calafiori_pred['predicted_points']:.2f} points")
    print(f"  Mateta predicted: {mateta_pred['predicted_points']:.2f} points")
    print(f"  Difference: {calafiori_pred['predicted_points'] - mateta_pred['predicted_points']:.2f} points")
    
    if calafiori_pred['predicted_points'] > mateta_pred['predicted_points']:
        print(f"\n  ⚠️  ISSUE: Calafiori has higher predicted points than Mateta!")
        print(f"  This explains why he's selected as vice captain.")
    else:
        print(f"\n  ✓ Mateta has higher predicted points, so there may be another issue.")

if __name__ == '__main__':
    manager_id = 789800
    gameweek = 12
    
    if len(sys.argv) > 1:
        manager_id = int(sys.argv[1])
    if len(sys.argv) > 2:
        gameweek = int(sys.argv[2])
    
    debug_vice_captain(manager_id, gameweek)

