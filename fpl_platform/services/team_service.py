"""
Team management service
"""
from datetime import datetime
from fpl_platform.database import db
from fpl_platform.models.team import Team


class TeamService:
    """Handle team operations"""
    
    @staticmethod
    def create_team(user, fpl_manager_id, **kwargs):
        """
        Create a new team for a user
        
        Args:
            user: User instance
            fpl_manager_id: FPL Manager ID
            **kwargs: Additional team fields
            
        Returns:
            tuple: (team, error_message)
        """
        # Check if team already exists for this user
        existing = Team.query.filter_by(
            user_id=user.id,
            fpl_manager_id=fpl_manager_id
        ).first()
        
        if existing:
            return None, "Team already exists"
        
        # Create team
        team = Team(
            user_id=user.id,
            fpl_manager_id=fpl_manager_id,
            **kwargs
        )
        
        # If this is the first team, make it primary
        if user.teams.count() == 0:
            team.is_primary = True
        
        try:
            db.session.add(team)
            db.session.commit()
            return team, None
        except Exception as e:
            db.session.rollback()
            return None, f"Team creation failed: {str(e)}"
    
    @staticmethod
    def get_user_teams(user):
        """Get all teams for a user"""
        return user.teams.all()
    
    @staticmethod
    def get_primary_team(user):
        """Get user's primary team"""
        return user.teams.filter_by(is_primary=True).first()
    
    @staticmethod
    def set_primary_team(user, team_id):
        """
        Set a team as primary
        
        Args:
            user: User instance
            team_id: Team ID to set as primary
            
        Returns:
            tuple: (success, error_message)
        """
        # Get team
        team = Team.query.filter_by(id=team_id, user_id=user.id).first()
        if not team:
            return False, "Team not found"
        
        try:
            # Unset all primary flags
            Team.query.filter_by(user_id=user.id).update({'is_primary': False})
            
            # Set new primary
            team.is_primary = True
            db.session.commit()
            return True, None
        except Exception as e:
            db.session.rollback()
            return False, f"Failed to set primary team: {str(e)}"
    
    @staticmethod
    def update_team(team, **kwargs):
        """
        Update team information
        
        Args:
            team: Team instance
            **kwargs: Fields to update
            
        Returns:
            tuple: (success, error_message)
        """
        try:
            for key, value in kwargs.items():
                if hasattr(team, key) and key not in ['id', 'user_id']:
                    setattr(team, key, value)
            
            team.updated_at = datetime.utcnow()
            db.session.commit()
            return True, None
        except Exception as e:
            db.session.rollback()
            return False, f"Update failed: {str(e)}"
    
    @staticmethod
    def delete_team(user, team_id):
        """
        Delete a team
        
        Args:
            user: User instance
            team_id: Team ID to delete
            
        Returns:
            tuple: (success, error_message)
        """
        team = Team.query.filter_by(id=team_id, user_id=user.id).first()
        if not team:
            return False, "Team not found"
        
        try:
            db.session.delete(team)
            db.session.commit()
            return True, None
        except Exception as e:
            db.session.rollback()
            return False, f"Delete failed: {str(e)}"
    
    @staticmethod
    def sync_team_from_fpl(team, manager_info):
        """
        Update team with information from FPL API
        
        Args:
            team: Team instance
            manager_info: Manager info dict from FPL API
            
        Returns:
            tuple: (success, error_message)
        """
        try:
            team.team_name = manager_info.get('team_name')
            team.manager_name = manager_info.get('name')
            team.overall_rank = manager_info.get('overall_rank')
            team.total_points = manager_info.get('total_points')
            team.team_value = manager_info.get('team_value', 0.0)
            team.bank = manager_info.get('bank', 0.0)
            team.last_synced = datetime.utcnow()
            
            db.session.commit()
            return True, None
        except Exception as e:
            db.session.rollback()
            return False, f"Sync failed: {str(e)}"

