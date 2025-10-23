"""Check now_cost vs team_value"""
import sys
sys.path.insert(0, '.')

from fpl_bot.core.data_collector import DataCollector

dc = DataCollector()

# Get bootstrap-static data
season_data = dc.get_current_season_data()
players_df = dc.create_players_dataframe(season_data)

# Get manager data
manager_data = dc.get_manager_data(789800)
picks = manager_data.get('current_team', {}).get('picks', [])

print("="*80)
print("NOW_COST INVESTIGATION")
print("="*80)

# Get current team player IDs
current_team_ids = [pick['element'] for pick in picks]

print(f"\nCurrent team ({len(current_team_ids)} players):")
print(f"{'Player':<20} {'now_cost':<12} {'Buying Price'}")
print("-"*80)

total_now_cost = 0
for player_id in current_team_ids:
    player_row = players_df[players_df['id'] == player_id]
    if not player_row.empty:
        player = player_row.iloc[0]
        now_cost = player['now_cost'] / 10.0  # API stores as 10x
        safe_name = player['web_name'].encode('ascii', 'ignore').decode()
        print(f"{safe_name:<20} £{now_cost:.1f}m")
        total_now_cost += now_cost

print("-"*80)
print(f"{'TOTAL now_cost':<20} £{total_now_cost:.1f}m")

# Compare to team_value from manager endpoint
team_value = manager_data.get('last_deadline_value', 0) / 10.0
bank = manager_data.get('last_deadline_bank', 0) / 10.0

print(f"\nFrom manager endpoint:")
print(f"  last_deadline_value (SELLING value): £{team_value:.1f}m")
print(f"  last_deadline_bank: £{bank:.1f}m")

print(f"\n" + "="*80)
print("ANALYSIS")
print("="*80)
print(f"  Sum of now_cost (buying price): £{total_now_cost:.1f}m")
print(f"  Team value (selling price): £{team_value:.1f}m")
print(f"  Profit/Loss: £{team_value - total_now_cost:.1f}m")

print(f"\nBudget available:")
print(f"  If using now_cost: £{total_now_cost:.1f}m + £{bank:.1f}m = £{total_now_cost + bank:.1f}m")
print(f"  If using team_value: £{team_value:.1f}m + £{bank:.1f}m = £{team_value + bank:.1f}m")

print(f"\n" + "="*80)
print("WHAT OPTIMIZER CURRENTLY DOES:")
print(f"  Uses: current_team_value (calculated from now_cost) = £{total_now_cost:.1f}m")
print(f"  Should use: team_value (from API) = £{team_value:.1f}m")
print(f"  Missing: £{team_value - total_now_cost:.1f}m in available budget!")
print("="*80)

