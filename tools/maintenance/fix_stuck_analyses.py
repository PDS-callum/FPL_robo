#!/usr/bin/env python3
"""
Fix stuck analyses that are in 'running' status
This can happen if the bot crashes without updating the database
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fpl_platform.app import create_app
from fpl_platform.database import db
from fpl_platform.models.analysis import Analysis
from datetime import datetime, timedelta

def fix_stuck_analyses():
    """Find and fix analyses stuck in 'running' status"""
    app = create_app()
    
    with app.app_context():
        # Find all analyses in 'running' status that were created more than 10 minutes ago
        ten_minutes_ago = datetime.utcnow() - timedelta(minutes=10)
        
        stuck_analyses = Analysis.query.filter(
            Analysis.status == 'running',
            Analysis.created_at < ten_minutes_ago
        ).all()
        
        if not stuck_analyses:
            print("✓ No stuck analyses found!")
            return
        
        print(f"Found {len(stuck_analyses)} stuck analyses:")
        
        for analysis in stuck_analyses:
            print(f"\n  Analysis ID: {analysis.id}")
            print(f"  User ID: {analysis.user_id}")
            print(f"  Team ID: {analysis.team_id}")
            print(f"  Created: {analysis.created_at}")
            print(f"  Status: {analysis.status}")
            
            # Update to failed status
            analysis.status = 'failed'
            analysis.error_message = 'Analysis timed out or crashed. Please try running analysis again.'
            analysis.completed_at = datetime.utcnow()
        
        # Commit changes
        try:
            db.session.commit()
            print(f"\n✓ Successfully fixed {len(stuck_analyses)} stuck analyses!")
            print("  Status changed to 'failed' with error message.")
            print("  Users can now run new analyses.")
        except Exception as e:
            db.session.rollback()
            print(f"\n✗ Error fixing analyses: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    print("FPL Vibes - Fix Stuck Analyses")
    print("=" * 60)
    fix_stuck_analyses()

