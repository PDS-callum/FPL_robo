"""
Standalone test for bot security validation
Tests the validation logic without requiring full Flask app context
"""
import re


# Simplified validation logic (extracted from BotExecutionService)
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


def validate_bot_code_simple(code):
    """Simplified validation for testing"""
    for restricted in RESTRICTED_IMPORTS:
        pattern = r'\b' + re.escape(restricted) + r'\b'
        if re.search(pattern, code):
            return False, f"Restricted keyword '{restricted}' found in code"
    return True, None


def test_network_imports_blocked():
    """Test that network imports are blocked"""
    print("\n=== Testing Network Import Blocking ===")
    
    tests = [
        ("requests", "import requests\nclass Bot:\n    pass"),
        ("urllib", "import urllib\nclass Bot:\n    pass"),
        ("socket", "import socket\nclass Bot:\n    pass"),
        ("http", "import http\nclass Bot:\n    pass"),
        ("httplib", "import httplib\nclass Bot:\n    pass"),
    ]
    
    passed = 0
    failed = 0
    
    for name, code in tests:
        is_valid, error = validate_bot_code_simple(code)
        if not is_valid and name in error.lower():
            print(f"  [PASS] {name} import correctly blocked")
            passed += 1
        else:
            print(f"  [FAIL] {name} import NOT blocked (SECURITY ISSUE!)")
            failed += 1
    
    return passed, failed


def test_file_system_blocked():
    """Test that file system access is blocked"""
    print("\n=== Testing File System Blocking ===")
    
    tests = [
        ("open", "class Bot:\n    def test(self):\n        open('file.txt')"),
        ("os", "import os\nclass Bot:\n    pass"),
        ("subprocess", "import subprocess\nclass Bot:\n    pass"),
    ]
    
    passed = 0
    failed = 0
    
    for name, code in tests:
        is_valid, error = validate_bot_code_simple(code)
        if not is_valid and name in error.lower():
            print(f"  [PASS] {name} correctly blocked")
            passed += 1
        else:
            print(f"  [FAIL] {name} NOT blocked (SECURITY ISSUE!)")
            failed += 1
    
    return passed, failed


def test_dangerous_functions_blocked():
    """Test that dangerous functions are blocked"""
    print("\n=== Testing Dangerous Functions Blocking ===")
    
    tests = [
        ("eval", "class Bot:\n    def test(self):\n        eval('1+1')"),
        ("exec", "class Bot:\n    def test(self):\n        exec('print(1)')"),
        ("__import__", "class Bot:\n    def test(self):\n        __import__('os')"),
    ]
    
    passed = 0
    failed = 0
    
    for name, code in tests:
        is_valid, error = validate_bot_code_simple(code)
        if not is_valid and name in error.lower():
            print(f"  [PASS] {name} correctly blocked")
            passed += 1
        else:
            print(f"  [FAIL] {name} NOT blocked (SECURITY ISSUE!)")
            failed += 1
    
    return passed, failed


def test_threading_blocked():
    """Test that threading is blocked"""
    print("\n=== Testing Threading/Concurrency Blocking ===")
    
    tests = [
        ("threading", "import threading\nclass Bot:\n    pass"),
        ("multiprocessing", "import multiprocessing\nclass Bot:\n    pass"),
        ("asyncio", "import asyncio\nclass Bot:\n    pass"),
    ]
    
    passed = 0
    failed = 0
    
    for name, code in tests:
        is_valid, error = validate_bot_code_simple(code)
        if not is_valid and name in error.lower():
            print(f"  [PASS] {name} correctly blocked")
            passed += 1
        else:
            print(f"  [FAIL] {name} NOT blocked (SECURITY ISSUE!)")
            failed += 1
    
    return passed, failed


def test_valid_bot_accepted():
    """Test that valid bots are accepted"""
    print("\n=== Testing Valid Bot Acceptance ===")
    
    valid_bot = """
class SimpleBot:
    def analyze(self, data):
        current_team = data.get('current_team', [])
        total_predicted = sum(p.get('predicted_points', 0) for p in current_team)
        return {
            'status': 'ok',
            'insights': [f"Total predicted: {total_predicted}"]
        }
    
    def optimize(self, data):
        current_team = data.get('current_team', [])
        sorted_team = sorted(
            current_team,
            key=lambda x: x.get('predicted_points', 0),
            reverse=True
        )
        return {
            'captain': sorted_team[0]['id'],
            'vice_captain': sorted_team[1]['id'],
            'lineup': [p['id'] for p in sorted_team[:11]]
        }
"""
    
    is_valid, error = validate_bot_code_simple(valid_bot)
    
    if is_valid:
        print(f"  [PASS] Valid bot correctly accepted")
        return 1, 0
    else:
        print(f"  [FAIL] Valid bot incorrectly rejected: {error}")
        return 0, 1


def test_safe_operations_allowed():
    """Test that safe operations are allowed"""
    print("\n=== Testing Safe Operations ===")
    
    safe_operations = [
        ("sorted", "sorted([1,2,3])"),
        ("sum", "sum([1,2,3])"),
        ("max", "max([1,2,3])"),
        ("min", "min([1,2,3])"),
        ("len", "len([1,2,3])"),
        ("enumerate", "enumerate([1,2,3])"),
        ("zip", "zip([1,2], [3,4])"),
        ("map", "map(lambda x: x*2, [1,2,3])"),
        ("filter", "filter(lambda x: x>0, [1,2,3])"),
    ]
    
    passed = 0
    failed = 0
    
    for name, operation in safe_operations:
        code = f"class Bot:\n    def test(self):\n        {operation}"
        is_valid, error = validate_bot_code_simple(code)
        
        if is_valid:
            print(f"  [PASS] {name} correctly allowed")
            passed += 1
        else:
            print(f"  [FAIL] {name} incorrectly blocked: {error}")
            failed += 1
    
    return passed, failed


def main():
    """Run all tests"""
    print("=" * 60)
    print("FPL Platform Bot Security Validation Tests")
    print("=" * 60)
    
    total_passed = 0
    total_failed = 0
    
    # Run all test suites
    test_suites = [
        test_network_imports_blocked,
        test_file_system_blocked,
        test_dangerous_functions_blocked,
        test_threading_blocked,
        test_valid_bot_accepted,
        test_safe_operations_allowed,
    ]
    
    for test_suite in test_suites:
        passed, failed = test_suite()
        total_passed += passed
        total_failed += failed
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"  Passed: {total_passed}")
    print(f"  Failed: {total_failed}")
    print(f"  Total:  {total_passed + total_failed}")
    
    if total_failed == 0:
        print("\n[SUCCESS] ALL TESTS PASSED! Security measures are working correctly.")
        return 0
    else:
        print(f"\n[WARNING] {total_failed} TESTS FAILED! Security issues detected.")
        return 1


if __name__ == '__main__':
    exit(main())

