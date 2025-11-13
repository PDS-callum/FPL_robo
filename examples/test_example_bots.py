"""
Test script for example bots
Validates bot structure and chip usage
"""


def test_bot(bot_class_name, bot_file):
    """Test a bot implementation"""
    print(f"\n{'='*60}")
    print(f"Testing: {bot_class_name}")
    print(f"File: {bot_file}")
    print('='*60)
    
    try:
        # Read and compile bot code
        with open(bot_file, 'r', encoding='utf-8') as f:
            bot_code = f.read()
        
        compile(bot_code, '<string>', 'exec')
        print("[PASS] Bot code compiles successfully")
        
        # Execute and get class
        namespace = {}
        exec(bot_code, namespace)
        
        bot_class = namespace.get(bot_class_name)
        if not bot_class:
            print(f"[FAIL] Bot class '{bot_class_name}' not found")
            return False
        
        print(f"[PASS] Found bot class: {bot_class_name}")
        
        # Check required methods
        bot = bot_class()
        
        if not hasattr(bot, 'analyze'):
            print("[FAIL] Missing 'analyze' method")
            return False
        print("[PASS] Has 'analyze' method")
        
        if not hasattr(bot, 'optimize'):
            print("[FAIL] Missing 'optimize' method")
            return False
        print("[PASS] Has 'optimize' method")
        
        # Test with sample data
        test_data = {
            'current_team': [
                {'id': i, 'name': f'Player{i}', 
                 'position': ['GKP', 'DEF', 'DEF', 'DEF', 'DEF', 'MID', 'MID', 
                             'MID', 'MID', 'FWD', 'FWD', 'GKP', 'DEF', 'MID', 'FWD'][i-1],
                 'predicted_points': 8.0 - (i * 0.2),
                 'cost': 5.0 + i * 0.3}
                for i in range(1, 16)
            ],
            'available_players': [],
            'gameweek': 5,
            'bank': 0.5,
            'free_transfers': 1,
            'chips_available': ['wildcard_1', 'bench_boost', 'free_hit', 'triple_captain']
        }
        
        # Test analyze
        analysis = bot.analyze(test_data)
        if analysis.get('status') == 'ok':
            print("[PASS] analyze() returned valid response")
            print(f"  Insights: {len(analysis.get('insights', []))} items")
        else:
            print("[FAIL] analyze() returned invalid response")
            return False
        
        # Test optimize
        optimization = bot.optimize(test_data)
        
        required_keys = ['captain', 'vice_captain', 'lineup']
        missing_keys = [k for k in required_keys if k not in optimization]
        
        if missing_keys:
            print(f"[FAIL] optimize() missing keys: {missing_keys}")
            return False
        
        print("[PASS] optimize() returned valid response")
        print(f"  Captain: {optimization['captain']}")
        print(f"  Vice Captain: {optimization['vice_captain']}")
        print(f"  Lineup size: {len(optimization['lineup'])}")
        print(f"  Chip: {optimization.get('chip', 'None')}")
        
        # Test chip usage across different gameweeks
        print("\n  Testing chip usage across gameweeks:")
        chip_usage = {}
        
        for gw in [3, 5, 8, 12, 15, 20, 25, 30]:
            test_data['gameweek'] = gw
            test_data['chips_available'] = ['wildcard_1', 'bench_boost', 'free_hit', 'triple_captain']
            
            result = bot.optimize(test_data)
            chip = result.get('chip')
            
            if chip:
                chip_usage[gw] = chip
                print(f"    GW{gw}: {chip}")
        
        if chip_usage:
            print(f"[PASS] Bot uses chips strategically ({len(chip_usage)} gameweeks)")
        else:
            print("[INFO] Bot doesn't use chips in tested gameweeks")
        
        print(f"\n[SUCCESS] {bot_class_name} passed all tests!")
        return True
        
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test all example bots"""
    print("\n" + "="*60)
    print("EXAMPLE BOT VALIDATION")
    print("="*60)
    
    bots = [
        ('PlatformBot', 'example_platform_bot.py'),
        ('SimpleBot', 'simple_example_bot.py')
    ]
    
    results = []
    for bot_name, bot_file in bots:
        try:
            result = test_bot(bot_name, bot_file)
            results.append((bot_name, result))
        except Exception as e:
            print(f"\n[ERROR] {bot_name} failed with exception: {e}")
            results.append((bot_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for bot_name, result in results:
        status = "[PASSED]" if result else "[FAILED]"
        print(f"{status} {bot_name}")
    
    print(f"\nTotal: {passed}/{total} bots validated")
    
    if passed == total:
        print("\n*** All example bots are ready for competition! ***")
    
    print("="*60)
    
    return passed == total


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)

