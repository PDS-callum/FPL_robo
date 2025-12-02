"""
Quick runner for transfer debugging
Usage: python run_transfer_debug.py YOUR_MANAGER_ID
"""

import sys
from debug_transfer_logic import debug_transfer_recommendations

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python run_transfer_debug.py YOUR_MANAGER_ID")
        print("\nExample: python run_transfer_debug.py 789800")
        sys.exit(1)
    
    manager_id = int(sys.argv[1])
    
    # Default players to analyze
    players = ['Sarr', 'Enzo', 'Mateta', 'Ekitike']
    
    print(f"\n{'='*100}")
    print(f"TRANSFER DEBUG ANALYSIS")
    print(f"{'='*100}\n")
    
    debug_transfer_recommendations(manager_id, players)


