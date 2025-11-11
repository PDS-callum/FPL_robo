#!/usr/bin/env python3
"""Quick test to verify platform is working"""
import requests
import time
import sys

print("Testing FPL Platform...")

# Give server time to start if it's running
time.sleep(1)

try:
    # Test health endpoint
    response = requests.get('http://127.0.0.1:5001/api/health', timeout=5)
    if response.status_code == 200:
        print("[OK] Platform is RUNNING and RESPONSIVE!")
        print(f"  Health check: {response.json()}")
        
        # Test welcome page
        response2 = requests.get('http://127.0.0.1:5001/welcome', timeout=5)
        if response2.status_code == 200:
            print("[OK] Welcome page loads successfully!")
            print("\n==> Platform is fully operational!")
            print("==> Open your browser to: http://127.0.0.1:5001/welcome")
            sys.exit(0)
    else:
        print(f"[ERROR] Unexpected status code: {response.status_code}")
        sys.exit(1)
        
except requests.exceptions.ConnectionError:
    print("[ERROR] Cannot connect - server may not be running")
    print("\nTo start the platform, run:")
    print("  python run_platform.py")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Error: {e}")
    sys.exit(1)

