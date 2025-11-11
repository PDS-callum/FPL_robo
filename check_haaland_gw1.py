"""
Quick script to check Haaland's GW1 bonus points from FPL API
"""

from fpl_platform.services.fpl_data_service import FPLDataService

# Get current season data to find Haaland
print('Fetching current season data...')
season_data = FPLDataService.fetch_current_season_data()

if season_data:
    # Find Haaland
    haaland = None
    for pid, player in season_data['players'].items():
        if 'Haaland' in player['full_name']:
            haaland = player
            print(f"Found: {player['full_name']} (ID: {pid})")
            break
    
    if haaland:
        haaland_id = haaland['id']
        
        # Fetch his detailed stats
        print(f"\nFetching Haaland (ID: {haaland_id}) gameweek history...")
        player_history = FPLDataService.fetch_player_history(haaland_id)
        
        if player_history and 'history' in player_history:
            gw1_stats = None
            for gw_data in player_history['history']:
                if gw_data['round'] == 1:
                    gw1_stats = gw_data
                    break
            
            if gw1_stats:
                print(f"\n{'='*50}")
                print(f"Haaland GW1 Stats (2024/25 Season)")
                print(f"{'='*50}")
                print(f"Minutes: {gw1_stats.get('minutes', 0)}")
                print(f"Goals: {gw1_stats.get('goals_scored', 0)}")
                print(f"Assists: {gw1_stats.get('assists', 0)}")
                print(f"Bonus Points: {gw1_stats.get('bonus', 0)}")
                print(f"BPS (Bonus Point System): {gw1_stats.get('bps', 0)}")
                print(f"Total FPL Points: {gw1_stats.get('total_points', 0)}")
                print(f"{'='*50}")
            else:
                print('No GW1 data found')
        else:
            print('Could not fetch player history')
    else:
        print('Haaland not found')
else:
    print('Could not fetch season data')

