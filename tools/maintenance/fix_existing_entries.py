"""
Quick script to fix existing season entries with incorrect gameweek data

Run this script from the project root:
    python fix_existing_entries.py
"""
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fpl_platform.app import create_app
from fpl_platform.database import db
from fpl_platform.models import SeasonEntry
from fpl_platform.services.fpl_data_service import FPLDataService

def fix_existing_entries():
    """
    Update all existing season entries with the correct current gameweek
    and recalculate ranks
    """
    app = create_app()
    
    with app.app_context():
        print("Fetching current gameweek from FPL API...")
        
        # Get current gameweek from API
        season_data = FPLDataService.fetch_current_season_data()
        
        if season_data:
            current_gw = season_data['current_gameweek']
            print(f"✓ Current gameweek is GW{current_gw}")
        else:
            print("✗ Failed to fetch from API, using GW1 as fallback")
            current_gw = 1
        
        # Update all entries
        entries = SeasonEntry.query.all()
        
        if not entries:
            print("No season entries found in database.")
            return
        
        print(f"\nFound {len(entries)} season entries to update...")
        
        for entry in entries:
            # Update gameweek if it's wrong (i.e., set to 1 when it should be current)
            if entry.current_gameweek == 1 and entry.total_points == 0:
                entry.current_gameweek = current_gw
                entry.entry_gameweek = current_gw
                print(f"  - Updated entry for user {entry.user_id}: GW1 → GW{current_gw}")
        
        # Recalculate ranks for all active entries by season
        seasons = db.session.query(SeasonEntry.season).distinct().all()
        
        for (season,) in seasons:
            print(f"\nRecalculating ranks for season {season}...")
            
            season_entries = SeasonEntry.query.filter_by(
                season=season,
                is_active=True
            ).order_by(SeasonEntry.total_points.desc()).all()
            
            for rank, entry in enumerate(season_entries, start=1):
                entry.current_rank = rank
                print(f"  - Rank {rank}: User {entry.user_id} with {entry.total_points} points")
        
        # Commit changes
        db.session.commit()
        print("\n✓ All entries updated successfully!")
        print("\nYou can now refresh your browser to see the updated values.")

if __name__ == '__main__':
    fix_existing_entries()

