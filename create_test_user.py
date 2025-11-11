#!/usr/bin/env python3
"""
Create a test user for the FPL Platform
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fpl_platform.app import create_app
from fpl_platform.database import db
from fpl_platform.models.user import User

def create_test_user():
    """Create a test user in the database"""
    app = create_app('development')
    
    with app.app_context():
        # Check if user already exists
        existing = User.query.filter_by(username='testuser').first()
        if existing:
            print("[INFO] Test user already exists!")
            print("\nLogin credentials:")
            print("  Username: testuser")
            print("  Email: test@example.com")
            print("  Password: testpass123")
            return
        
        # Create new test user
        user = User(
            username='testuser',
            email='test@example.com',
            first_name='Test',
            last_name='User'
        )
        user.set_password('testpass123')
        
        try:
            db.session.add(user)
            db.session.commit()
            
            print("[SUCCESS] Test user created!")
            print("\nLogin credentials:")
            print("  Username: testuser")
            print("  Email: test@example.com") 
            print("  Password: testpass123")
            print("\nYou can now login at: http://127.0.0.1:5001/auth/login")
            
        except Exception as e:
            db.session.rollback()
            print(f"[ERROR] Failed to create user: {e}")

if __name__ == '__main__':
    create_test_user()

