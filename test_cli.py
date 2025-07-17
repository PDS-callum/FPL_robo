#!/usr/bin/env python3
"""
Test script for the new CLI functionality
"""

import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # Test import
    try:
        from fpl_bot.main import main
        from fpl_bot.utils.data_collection import FPLDataCollector, FPLDataProcessor
        print("✅ All imports successful!")
        
        # Test class initialization
        collector = FPLDataCollector()
        processor = FPLDataProcessor()
        print("✅ Class initialization successful!")
        
        # Test available seasons
        seasons = collector.get_available_seasons()
        print(f"✅ Available seasons: {seasons}")
        
        print("\n🎉 CLI setup is working correctly!")
        print("\nYou can now use the following commands:")
        print("python -m fpl_bot.main collect --seasons 2023-24")
        print("python -m fpl_bot.main process --seasons 2023-24")
        print("python -m fpl_bot.main collect-and-process --seasons 2023-24")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
