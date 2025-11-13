#!/usr/bin/env python3
"""
FPL Vibes Bot Tester
====================

Test your bot locally before submitting to the platform!

Usage:
    python tools/bot_tester.py your_bot.py

This script will:
1. Validate your bot code (check for restricted imports)
2. Test that it has the required methods
3. Run sample analyze() and optimize() calls
4. Report any errors or warnings

"""

import sys
import re
import traceback
from pathlib import Path


# SECURITY: These imports are RESTRICTED on the platform
RESTRICTED_IMPORTS = [
    # System access
    'os', 'subprocess', 'sys', 'eval', 'exec', 'compile',
    '__import__', 'open', 'file', 'input', 'raw_input',
    # Network access - MUST BE BLOCKED to prevent external data sources
    'requests', 'urllib', 'urllib2', 'urllib3', 'http', 'httplib',
    'socket', 'socketserver', 'ftplib', 'telnetlib', 'smtplib',
    'poplib', 'imaplib', 'nntplib', 'ssl', 'asyncio',
    # Additional security
    'pickle', 'shelve', 'marshal', 'dbm', 'sqlite3',
    'multiprocessing', 'threading', 'thread'
]


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text):
    """Print a styled header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")


def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}[PASS] {text}{Colors.END}")


def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}[FAIL] {text}{Colors.END}")


def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}[WARN] {text}{Colors.END}")


def print_info(text):
    """Print info message"""
    print(f"  {text}")


def validate_imports(code):
    """Check for restricted imports in the code"""
    print_header("STEP 1: Validating Imports")
    
    errors = []
    warnings = []
    
    # Check for restricted imports
    for restricted in RESTRICTED_IMPORTS:
        # Pattern for various import styles
        patterns = [
            rf'\bimport\s+{restricted}\b',
            rf'\bfrom\s+{restricted}\s+import\b',
            rf'\bimport\s+\w+\.{restricted}\b',
        ]
        
        for pattern in patterns:
            if re.search(pattern, code):
                errors.append(f"Restricted import detected: {restricted}")
                break
    
    # Check for eval/exec/compile usage
    dangerous_funcs = ['eval', 'exec', 'compile', '__import__']
    for func in dangerous_funcs:
        if re.search(rf'\b{func}\s*\(', code):
            errors.append(f"Dangerous function detected: {func}()")
    
    if errors:
        for error in errors:
            print_error(error)
        print_error("\nYour bot uses restricted imports or functions!")
        print_info("The platform will reject this code for security reasons.")
        return False
    else:
        print_success("No restricted imports found")
        print_success("Security validation passed")
        return True


def validate_class_structure(code, bot_class):
    """Check that the bot has required methods"""
    print_header("STEP 2: Validating Bot Structure")
    
    errors = []
    
    # Check for required methods
    required_methods = ['analyze', 'optimize']
    
    for method in required_methods:
        if not hasattr(bot_class, method):
            errors.append(f"Missing required method: {method}()")
        elif not callable(getattr(bot_class, method)):
            errors.append(f"Method exists but is not callable: {method}")
    
    if errors:
        for error in errors:
            print_error(error)
        return False
    else:
        print_success("Bot has required methods: analyze() and optimize()")
        
        # Check method signatures
        import inspect
        for method_name in required_methods:
            method = getattr(bot_class, method_name)
            sig = inspect.signature(method)
            params = list(sig.parameters.keys())
            
            if 'self' in params:
                params.remove('self')
            
            if len(params) >= 1:
                print_success(f"  {method_name}() accepts data parameter")
            else:
                print_warning(f"  {method_name}() should accept a 'data' parameter")
        
        return True


def test_bot_execution(bot_class):
    """Test that the bot can execute without errors"""
    print_header("STEP 3: Testing Bot Execution")
    
    try:
        # Instantiate the bot
        bot_instance = bot_class()
        print_success("Bot instantiated successfully")
    except Exception as e:
        print_error(f"Failed to instantiate bot: {e}")
        traceback.print_exc()
        return False
    
    # Create sample data
    sample_data = {
        'current_team': [
            {'id': i, 'name': f'Player {i}', 'position': 'FWD', 'cost': 8.0, 'predicted_points': 5 + i*0.5}
            for i in range(1, 16)
        ],
        'available_players': [
            {'id': i, 'name': f'Available Player {i}', 'position': 'MID', 'cost': 6.0, 'predicted_points': 4.0}
            for i in range(100, 120)
        ],
        'gameweek': 1,
        'bank': 0.5,
        'free_transfers': 1,
        'chips_available': ['wildcard_1', 'bench_boost', 'triple_captain', 'free_hit']
    }
    
    # Test analyze()
    print("\nTesting analyze() method...")
    try:
        result = bot_instance.analyze(sample_data)
        
        if isinstance(result, dict):
            print_success("analyze() returned a dictionary")
            
            if 'status' in result:
                print_info(f"  Status: {result['status']}")
            else:
                print_warning("  No 'status' field in result (optional but recommended)")
            
            if 'message' in result:
                print_info(f"  Message: {result['message']}")
            
            if 'insights' in result:
                print_info(f"  Insights: {len(result.get('insights', []))} items")
        else:
            print_warning(f"analyze() returned {type(result).__name__}, expected dict")
    
    except Exception as e:
        print_error(f"analyze() raised an exception: {e}")
        traceback.print_exc()
        return False
    
    # Test optimize()
    print("\nTesting optimize() method...")
    try:
        result = bot_instance.optimize(sample_data)
        
        if isinstance(result, dict):
            print_success("optimize() returned a dictionary")
            
            # Check required fields
            required_fields = ['captain', 'vice_captain', 'lineup']
            for field in required_fields:
                if field in result:
                    print_success(f"  Has required field: '{field}'")
                else:
                    print_error(f"  Missing required field: '{field}'")
            
            # Validate lineup
            if 'lineup' in result:
                lineup = result['lineup']
                if isinstance(lineup, list):
                    print_info(f"  Lineup contains {len(lineup)} players")
                    if len(lineup) != 11:
                        print_warning(f"  Lineup should have exactly 11 players, got {len(lineup)}")
                else:
                    print_error(f"  Lineup should be a list, got {type(lineup).__name__}")
            
            # Check optional chip field
            if 'chip' in result:
                print_info(f"  Chip selected: {result['chip']}")
            
            # Check optional transfers field
            if 'transfers' in result:
                transfers = result['transfers']
                if isinstance(transfers, list):
                    print_info(f"  Transfers: {len(transfers)} planned")
                else:
                    print_warning(f"  Transfers should be a list")
        
        else:
            print_error(f"optimize() returned {type(result).__name__}, expected dict")
            return False
    
    except Exception as e:
        print_error(f"optimize() raised an exception: {e}")
        traceback.print_exc()
        return False
    
    return True


def main():
    """Main test runner"""
    print_header("FPL Vibes Bot Tester")
    
    if len(sys.argv) < 2:
        print_error("Usage: python tools/bot_tester.py your_bot.py")
        print_info("Example: python tools/bot_tester.py examples/simple_example_bot.py")
        sys.exit(1)
    
    bot_file = Path(sys.argv[1])
    
    if not bot_file.exists():
        print_error(f"File not found: {bot_file}")
        sys.exit(1)
    
    print_info(f"Testing bot: {bot_file}")
    
    # Read the bot code
    try:
        with open(bot_file, 'r', encoding='utf-8') as f:
            code = f.read()
    except Exception as e:
        print_error(f"Failed to read file: {e}")
        sys.exit(1)
    
    # Step 1: Validate imports
    if not validate_imports(code):
        print_header("TEST FAILED")
        print_error("Your bot contains security violations.")
        print_info("\nFix the issues above and try again.")
        sys.exit(1)
    
    # Load the bot module
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("bot_module", bot_file)
        bot_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bot_module)
    except Exception as e:
        print_error(f"Failed to load bot module: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Find the bot class
    bot_classes = []
    for name in dir(bot_module):
        obj = getattr(bot_module, name)
        if isinstance(obj, type) and hasattr(obj, 'optimize') and hasattr(obj, 'analyze'):
            bot_classes.append((name, obj))
    
    if not bot_classes:
        print_error("No valid bot class found in the file")
        print_info("Make sure your bot class has both analyze() and optimize() methods")
        sys.exit(1)
    
    if len(bot_classes) > 1:
        print_warning(f"Multiple bot classes found: {[name for name, _ in bot_classes]}")
        print_info("The platform will use the first one found")
    
    bot_class_name, bot_class = bot_classes[0]
    print_info(f"Found bot class: {bot_class_name}")
    
    # Step 2: Validate structure
    if not validate_class_structure(code, bot_class):
        print_header("TEST FAILED")
        print_error("Your bot is missing required methods.")
        sys.exit(1)
    
    # Step 3: Test execution
    if not test_bot_execution(bot_class):
        print_header("TEST FAILED")
        print_error("Your bot encountered errors during execution.")
        sys.exit(1)
    
    # Success!
    print_header("ALL TESTS PASSED!")
    print_success("Your bot is ready to submit to the platform!")
    print_info("\nNext steps:")
    print_info("1. Go to the FPL Vibes competition page")
    print_info("2. Click 'Submit Bot' or 'Edit Bot'")
    print_info("3. Copy and paste your bot code")
    print_info("4. Submit for validation")
    print_info("5. Run backtests to evaluate performance")
    print_info("6. Activate when you're ready to compete!")
    print()


if __name__ == '__main__':
    main()

