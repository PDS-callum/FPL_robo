"""
Test script for competition feature
"""
from fpl_platform.app import create_app
from fpl_platform.database import db
from fpl_platform.models import User, BotSubmission, BacktestResult, SeasonEntry
from fpl_platform.services.bot_execution_service import BotExecutionService

def test_competition_feature():
    """Test the competition feature"""
    app = create_app()
    
    with app.app_context():
        print("\n" + "="*60)
        print("TESTING COMPETITION FEATURE")
        print("="*60)
        
        # Test 1: Check tables exist
        print("\n[TEST 1] Checking database tables...")
        try:
            tables = db.engine.table_names()
            required_tables = ['bot_submissions', 'backtest_results', 'season_entries']
            
            for table in required_tables:
                if table in tables:
                    print(f"  [OK] Table '{table}' exists")
                else:
                    print(f"  [FAIL] Table '{table}' missing")
            
        except Exception as e:
            print(f"  [ERROR] {e}")
        
        # Test 2: Test bot validation
        print("\n[TEST 2] Testing bot validation...")
        
        # Valid bot code
        valid_code = """
class TestBot:
    def __init__(self):
        pass
    
    def analyze(self, data):
        return {'status': 'ok'}
    
    def optimize(self, data):
        return {'transfers': [], 'captain': None}
"""
        
        is_valid, error, class_name = BotExecutionService.validate_bot_code(valid_code)
        if is_valid:
            print(f"  [OK] Valid bot code accepted (class: {class_name})")
        else:
            print(f"  [FAIL] Valid bot code rejected: {error}")
        
        # Invalid bot code (missing method)
        invalid_code = """
class InvalidBot:
    def __init__(self):
        pass
    
    def analyze(self, data):
        return {'status': 'ok'}
"""
        
        is_valid, error, class_name = BotExecutionService.validate_bot_code(invalid_code)
        if not is_valid:
            print(f"  [OK] Invalid bot code rejected: {error}")
        else:
            print(f"  [FAIL] Invalid bot code was accepted")
        
        # Test 3: Test restricted keywords
        print("\n[TEST 3] Testing security restrictions...")
        
        restricted_code = """
import os

class BadBot:
    def analyze(self, data):
        return {'status': 'ok'}
    
    def optimize(self, data):
        return {'transfers': []}
"""
        
        is_valid, error, class_name = BotExecutionService.validate_bot_code(restricted_code)
        if not is_valid and 'os' in error.lower():
            print(f"  [OK] Restricted import blocked: {error}")
        else:
            print(f"  [FAIL] Restricted import was not blocked")
        
        # Test 4: Check models
        print("\n[TEST 4] Testing model creation...")
        
        try:
            # Check if we can query (won't create actual data)
            submission_count = BotSubmission.query.count()
            backtest_count = BacktestResult.query.count()
            entry_count = SeasonEntry.query.count()
            
            print(f"  [OK] BotSubmission model works ({submission_count} submissions)")
            print(f"  [OK] BacktestResult model works ({backtest_count} backtests)")
            print(f"  [OK] SeasonEntry model works ({entry_count} entries)")
            
        except Exception as e:
            print(f"  [ERROR] Model error: {e}")
        
        # Test 5: Test routes registration
        print("\n[TEST 5] Testing route registration...")
        
        competition_routes = [
            '/competition/',
            '/competition/submit',
            '/competition/leaderboard',
            '/competition/documentation',
            '/competition/my-entry',
        ]
        
        for route in competition_routes:
            # Check if route exists in app
            has_route = any(rule.rule == route for rule in app.url_map.iter_rules())
            if has_route:
                print(f"  [OK] Route registered: {route}")
            else:
                print(f"  [FAIL] Route missing: {route}")
        
        print("\n" + "="*60)
        print("TESTING COMPLETE")
        print("="*60)
        print("\nTo test in browser:")
        print("1. Run: python run_platform.py")
        print("2. Navigate to: http://127.0.0.1:5001/competition/")
        print("3. Log in with your credentials")
        print("4. Test features:")
        print("   - Submit a new bot")
        print("   - Run backtest")
        print("   - View leaderboard")
        print("   - Check profile page for active entry")
        print("="*60 + "\n")

if __name__ == '__main__':
    test_competition_feature()

