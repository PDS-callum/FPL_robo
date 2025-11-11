"""
Authentication service
"""
from datetime import datetime
from flask_login import login_user, logout_user
from fpl_platform.database import db
from fpl_platform.models.user import User


class AuthService:
    """Handle authentication operations"""
    
    @staticmethod
    def register_user(username, email, password, first_name=None, last_name=None):
        """
        Register a new user
        
        Args:
            username: Unique username
            email: Unique email address
            password: Plain text password (will be hashed)
            first_name: Optional first name
            last_name: Optional last name
            
        Returns:
            tuple: (user, error_message)
        """
        # Check if username exists
        if User.query.filter_by(username=username).first():
            return None, "Username already exists"
        
        # Check if email exists
        if User.query.filter_by(email=email).first():
            return None, "Email already registered"
        
        # Create new user
        user = User(
            username=username,
            email=email,
            first_name=first_name,
            last_name=last_name
        )
        user.set_password(password)
        
        try:
            db.session.add(user)
            db.session.commit()
            return user, None
        except Exception as e:
            db.session.rollback()
            return None, f"Registration failed: {str(e)}"
    
    @staticmethod
    def login(username_or_email, password, remember=False):
        """
        Login user
        
        Args:
            username_or_email: Username or email address
            password: Plain text password
            remember: Remember user session
            
        Returns:
            tuple: (success, error_message)
        """
        # Try to find user by username or email
        user = User.query.filter(
            (User.username == username_or_email) | (User.email == username_or_email)
        ).first()
        
        if not user:
            return False, "Invalid username or email"
        
        if not user.check_password(password):
            return False, "Invalid password"
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.session.commit()
        
        # Log in user
        login_user(user, remember=remember)
        return True, None
    
    @staticmethod
    def logout():
        """Logout current user"""
        logout_user()
    
    @staticmethod
    def update_profile(user, **kwargs):
        """
        Update user profile
        
        Args:
            user: User instance
            **kwargs: Fields to update
            
        Returns:
            tuple: (success, error_message)
        """
        try:
            for key, value in kwargs.items():
                if hasattr(user, key) and key not in ['id', 'password_hash']:
                    setattr(user, key, value)
            
            db.session.commit()
            return True, None
        except Exception as e:
            db.session.rollback()
            return False, f"Update failed: {str(e)}"
    
    @staticmethod
    def change_password(user, old_password, new_password):
        """
        Change user password
        
        Args:
            user: User instance
            old_password: Current password
            new_password: New password
            
        Returns:
            tuple: (success, error_message)
        """
        if not user.check_password(old_password):
            return False, "Current password is incorrect"
        
        try:
            user.set_password(new_password)
            db.session.commit()
            return True, None
        except Exception as e:
            db.session.rollback()
            return False, f"Password change failed: {str(e)}"

