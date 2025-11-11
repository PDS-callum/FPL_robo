"""
Database migration script to add competition tables
"""
import sys
from fpl_platform.app import create_app
from fpl_platform.database import db

def migrate_database():
    """Add competition tables to database"""
    app = create_app()
    
    with app.app_context():
        print("Creating competition tables...")
        
        # Import models to register them
        from fpl_platform.models import (
            User, Team, Analysis, 
            BotSubmission, BacktestResult, SeasonEntry
        )
        
        # Create all tables
        db.create_all()
        
        print("[OK] Competition tables created successfully!")
        print("\nNew tables:")
        print("  - bot_submissions")
        print("  - backtest_results")
        print("  - season_entries")
        
        return True

if __name__ == '__main__':
    try:
        migrate_database()
        print("\n[OK] Migration completed successfully!")
    except Exception as e:
        print(f"\n[ERROR] Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

