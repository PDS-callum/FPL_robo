"""Show exactly what the user would see in the report"""
import sys
sys.path.insert(0, '.')

from fpl_bot.main import FPLBot

bot = FPLBot()

print("Running bot (no wildcard analysis for speed)...")
report = bot.run_analysis(789800, verbose=False, analyze_wildcard=False)

if report and report.get('optimization'):
    opt = report['optimization']
    gw8 = opt['weekly_plans'][0]
    
    print("\n" + "="*80)
    print("WHAT USER SEES IN REPORT/UI")
    print("="*80)
    
    print(f"\nGW{gw8['gameweek']} Plan:")
    print(f"  Transfers: {gw8['transfers']['count']}")
    print(f"  Hits: {gw8['hits_taken']}")
    print(f"  Points cost: {gw8['points_cost']}")
    
    print(f"\n  Transfers OUT:")
    total_out_cost = 0
    for t in gw8['transfers']['out']:
        safe_name = t['player_name'].encode('ascii', 'ignore').decode()
        print(f"    {safe_name:<20} {t['position']:<4} £{t['cost']:.1f}m")
        total_out_cost += t['cost']
    
    print(f"\n  Transfers IN:")
    total_in_cost = 0
    for t in gw8['transfers']['in']:
        safe_name = t['player_name'].encode('ascii', 'ignore').decode()
        print(f"    {safe_name:<20} {t['position']:<4} £{t['cost']:.1f}m")
        total_in_cost += t['cost']
    
    print(f"\n  SQUAD (all 15):")
    squad_total = 0
    for p in gw8['squad']['all']:
        safe_name = p['player_name'].encode('ascii', 'ignore').decode()
        print(f"    {safe_name:<20} {p['position']:<4} £{p['cost']:.1f}m")
        squad_total += p['cost']
    
    print(f"\n" + "="*80)
    print(f"BUDGET SUMMARY")
    print(f"="*80)
    print(f"  Total OUT: £{total_out_cost:.1f}m ({len(gw8['transfers']['out'])} players)")
    print(f"  Total IN: £{total_in_cost:.1f}m ({len(gw8['transfers']['in'])} players)")
    print(f"  Net difference: £{total_in_cost - total_out_cost:.1f}m")
    print(f"\n  Final squad value: £{squad_total:.1f}m")
    print(f"  Bank available: £6.1m")
    print(f"  Current team value: £94.1m")
    print(f"  Money available: £{94.1 + 6.1:.1f}m")
    
    if squad_total > 100.2:
        print(f"\n  *** OVER BUDGET by £{squad_total - 100.2:.1f}m ***")
    else:
        print(f"\n  Under budget by £{100.2 - squad_total:.1f}m")
    
    print("\n" + "="*80)
    print("Where you might see '2.5m over':")
    print(f"  If comparing to £100.0m baseline: Over by £{squad_total - 100.0:.1f}m")
    print(f"  If forgetting bank (£6.1m): Would appear over by £{squad_total - 94.1:.1f}m")
    print("="*80)

