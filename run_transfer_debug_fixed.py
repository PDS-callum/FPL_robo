"""
Quick runner for transfer debugging - Fixed version with correct player search
"""

import sys
import os

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'ignore')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'ignore')

from debug_transfer_logic import debug_transfer_recommendations

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python run_transfer_debug_fixed.py YOUR_MANAGER_ID")
        print("\nExample: python run_transfer_debug_fixed.py 789800")
        sys.exit(1)
    
    manager_id = int(sys.argv[1])
    
    # Search for the correct players
    # Ismaïla Sarr is at Crystal Palace, not Malo Sarr at Chelsea
    # Ekitike might be "Hugo Ekitike" or just "Ekitike"
    players = ['Isma', 'Enzo', 'Mateta', 'Ekitike']  # Use "Isma" to get Ismaïla Sarr
    
    print(f"\n{'='*100}")
    print(f"TRANSFER DEBUG ANALYSIS - FIXED")
    print(f"{'='*100}\n")
    print(f"Searching for: Ismaila Sarr (CRY), Enzo (CHE), Mateta (CRY), Ekitike")
    print()
    
    debug_transfer_recommendations(manager_id, players)


