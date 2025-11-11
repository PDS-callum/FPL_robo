"""
Team model
"""
from datetime import datetime
from fpl_platform.database import db


class Team(db.Model):
    """FPL Team model"""
    
    __tablename__ = 'teams'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    
    # FPL Information
    fpl_manager_id = db.Column(db.Integer, nullable=False, index=True)
    team_name = db.Column(db.String(100))
    manager_name = db.Column(db.String(100))
    
    # Team Stats (cached from FPL API)
    overall_rank = db.Column(db.Integer)
    total_points = db.Column(db.Integer)
    team_value = db.Column(db.Float)  # in millions
    bank = db.Column(db.Float)  # in millions
    
    # Settings
    is_primary = db.Column(db.Boolean, default=False)
    auto_run_enabled = db.Column(db.Boolean, default=False)
    auto_execute_enabled = db.Column(db.Boolean, default=False)
    
    # Bot Configuration
    risk_aversion = db.Column(db.Float, default=0.5)  # 0=aggressive, 1=conservative
    min_playing_chance = db.Column(db.Integer, default=75)
    analyze_wildcard = db.Column(db.Boolean, default=False)
    no_hits = db.Column(db.Boolean, default=False)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_synced = db.Column(db.DateTime)
    
    # Relationships
    analyses = db.relationship('Analysis', backref='team', lazy='dynamic', cascade='all, delete-orphan', order_by='Analysis.created_at.desc()')
    
    def __repr__(self):
        return f'<Team {self.team_name} (ID: {self.fpl_manager_id})>'
    
    def to_dict(self):
        """Convert team to dictionary"""
        return {
            'id': self.id,
            'fpl_manager_id': self.fpl_manager_id,
            'team_name': self.team_name,
            'manager_name': self.manager_name,
            'overall_rank': self.overall_rank,
            'total_points': self.total_points,
            'team_value': self.team_value,
            'bank': self.bank,
            'is_primary': self.is_primary,
            'auto_run_enabled': self.auto_run_enabled,
            'auto_execute_enabled': self.auto_execute_enabled,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_synced': self.last_synced.isoformat() if self.last_synced else None,
        }

