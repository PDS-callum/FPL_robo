#!/usr/bin/env python3
"""
Password Reset Utility

Reset a user's password in the FPL Platform database.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fpl_platform.app import create_app
from fpl_platform.database import db
from fpl_platform.models.user import User


def reset_password(username_or_email, new_password):
    """Reset password for a user"""
    app = create_app('development')
    
    with app.app_context():
        # Find user by username or email
        user = User.query.filter(
            (User.username == username_or_email) | (User.email == username_or_email)
        ).first()
        
        if not user:
            print(f"❌ User not found: {username_or_email}")
            print("\nAvailable users:")
            users = User.query.all()
            if users:
                for u in users:
                    print(f"  - Username: {u.username}, Email: {u.email}")
            else:
                print("  (No users found in database)")
            return False
        
        # Reset password
        user.set_password(new_password)
        db.session.commit()
        
        print(f"✅ Password reset successful!")
        print(f"   User: {user.username} ({user.email})")
        print(f"   New password: {new_password}")
        return True


def list_users():
    """List all users in the database"""
    app = create_app('development')
    
    with app.app_context():
        users = User.query.all()
        
        if not users:
            print("No users found in database.")
            return
        
        print(f"\nFound {len(users)} user(s):\n")
        for user in users:
            print(f"  ID: {user.id}")
            print(f"  Username: {user.username}")
            print(f"  Email: {user.email}")
            print(f"  Created: {user.created_at}")
            print(f"  Last Login: {user.last_login or 'Never'}")
            print()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("""
Password Reset Utility for FPL Platform
========================================

Usage:
  python tools/reset_password.py <username_or_email> <new_password>
  python tools/reset_password.py --list

Examples:
  python tools/reset_password.py admin newpassword123
  python tools/reset_password.py user@example.com newpassword123
  python tools/reset_password.py --list

Note: This will reset the password without requiring the old password.
        """)
        sys.exit(1)
    
    if sys.argv[1] == '--list':
        list_users()
    elif len(sys.argv) < 3:
        print("❌ Error: Please provide both username/email and new password")
        print("   Usage: python tools/reset_password.py <username_or_email> <new_password>")
        sys.exit(1)
    else:
        username_or_email = sys.argv[1]
        new_password = sys.argv[2]
        
        if len(new_password) < 6:
            print("❌ Error: Password must be at least 6 characters")
            sys.exit(1)
        
        reset_password(username_or_email, new_password)


