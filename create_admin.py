#!/usr/bin/env python3
"""Create admin user for testing"""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fpl_platform.app import create_app
from fpl_platform.database import db
from fpl_platform.models.user import User

app = create_app('development')

with app.app_context():
    # Check if admin exists
    admin = User.query.filter_by(username='admin').first()
    if admin:
        print("[INFO] Admin user already exists!")
    else:
        # Create admin
        admin = User(username='admin', email='admin@fpl.local', first_name='Admin', last_name='User')
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()
        print("[SUCCESS] Admin user created!")
    
    print("\nAdmin Login Credentials:")
    print("  Username: admin")
    print("  Password: admin123")
    print("\nLogin at: http://127.0.0.1:5001/auth/login")

