#!/usr/bin/env python3
"""
Direct SQLite fix for stuck analyses
No Flask imports needed - direct database access
"""

import sqlite3
from datetime import datetime, timedelta

def fix_stuck_analyses():
    """Fix analyses stuck in 'running' status using direct SQLite access"""
    db_path = 'instance/fpl_platform.db'
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Find stuck analyses (running for more than 10 minutes)
        ten_minutes_ago = (datetime.utcnow() - timedelta(minutes=10)).isoformat()
        
        cursor.execute("""
            SELECT id, user_id, team_id, created_at, status
            FROM analyses
            WHERE status = 'running'
            AND created_at < ?
        """, (ten_minutes_ago,))
        
        stuck_analyses = cursor.fetchall()
        
        if not stuck_analyses:
            print("[OK] No stuck analyses found!")
            conn.close()
            return
        
        print(f"Found {len(stuck_analyses)} stuck analyses:")
        
        for analysis in stuck_analyses:
            analysis_id, user_id, team_id, created_at, status = analysis
            print(f"\n  Analysis ID: {analysis_id}")
            print(f"  User ID: {user_id}")
            print(f"  Team ID: {team_id}")
            print(f"  Created: {created_at}")
            print(f"  Status: {status}")
        
        # Update to failed status
        current_time = datetime.utcnow().isoformat()
        cursor.execute("""
            UPDATE analyses
            SET status = 'failed',
                error_message = 'Analysis timed out or crashed. Please try running analysis again.',
                completed_at = ?
            WHERE status = 'running'
            AND created_at < ?
        """, (current_time, ten_minutes_ago))
        
        conn.commit()
        
        print(f"\n[SUCCESS] Fixed {len(stuck_analyses)} stuck analyses!")
        print("  Status changed to 'failed' with error message.")
        print("  Users can now run new analyses.")
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"\n[ERROR] Database error: {e}")
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("FPL Vibes - Fix Stuck Analyses (Direct SQLite)")
    print("=" * 60)
    fix_stuck_analyses()

