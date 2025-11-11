"""
Competition models for bot battles
"""
from datetime import datetime
from fpl_platform.database import db
import json


class BotSubmission(db.Model):
    """User-submitted bot code"""
    
    __tablename__ = 'bot_submissions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    
    # Bot information
    bot_name = db.Column(db.String(100), nullable=False)
    bot_description = db.Column(db.Text)
    bot_code = db.Column(db.Text, nullable=False)  # Python class code
    
    # Validation
    is_validated = db.Column(db.Boolean, default=False)
    validation_error = db.Column(db.Text)
    
    # Status
    is_active = db.Column(db.Boolean, default=False)  # Is this bot active for current season?
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    backtests = db.relationship('BacktestResult', backref='bot_submission', lazy='dynamic', cascade='all, delete-orphan')
    season_entry = db.relationship('SeasonEntry', backref='bot_submission', uselist=False, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<BotSubmission {self.bot_name} by User {self.user_id}>'
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'bot_name': self.bot_name,
            'bot_description': self.bot_description,
            'is_validated': self.is_validated,
            'is_active': self.is_active,
            'validation_error': self.validation_error,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class BacktestResult(db.Model):
    """Results from backtesting a bot on previous season"""
    
    __tablename__ = 'backtest_results'
    
    id = db.Column(db.Integer, primary_key=True)
    bot_submission_id = db.Column(db.Integer, db.ForeignKey('bot_submissions.id'), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    
    # Backtest configuration
    season = db.Column(db.String(20))  # e.g., "2023-24"
    start_gameweek = db.Column(db.Integer)
    end_gameweek = db.Column(db.Integer)
    
    # Results
    status = db.Column(db.String(20), default='pending')  # pending, running, completed, failed
    total_points = db.Column(db.Integer)
    final_rank = db.Column(db.Integer)
    team_value = db.Column(db.Float)
    transfers_made = db.Column(db.Integer)
    hits_taken = db.Column(db.Integer)
    chips_used = db.Column(db.String(200))  # JSON list of chips and when used
    
    # Detailed results (stored as JSON)
    gameweek_results = db.Column(db.Text)  # Points, transfers, etc. per gameweek
    
    # Performance metrics
    average_points_per_gw = db.Column(db.Float)
    best_gameweek = db.Column(db.Integer)
    best_gameweek_points = db.Column(db.Integer)
    worst_gameweek = db.Column(db.Integer)
    worst_gameweek_points = db.Column(db.Integer)
    
    # Execution time
    execution_time_seconds = db.Column(db.Float)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    completed_at = db.Column(db.DateTime)
    
    # Error handling
    error_message = db.Column(db.Text)
    
    def __repr__(self):
        return f'<BacktestResult {self.id} for Bot {self.bot_submission_id}>'
    
    def set_gameweek_results(self, data):
        """Store gameweek results as JSON"""
        self.gameweek_results = json.dumps(data)
    
    def get_gameweek_results(self):
        """Retrieve gameweek results from JSON"""
        if self.gameweek_results:
            return json.loads(self.gameweek_results)
        return None
    
    def to_dict(self, include_details=False):
        """Convert to dictionary"""
        data = {
            'id': self.id,
            'bot_submission_id': self.bot_submission_id,
            'season': self.season,
            'status': self.status,
            'total_points': self.total_points,
            'final_rank': self.final_rank,
            'team_value': self.team_value,
            'transfers_made': self.transfers_made,
            'hits_taken': self.hits_taken,
            'average_points_per_gw': self.average_points_per_gw,
            'best_gameweek': self.best_gameweek,
            'best_gameweek_points': self.best_gameweek_points,
            'execution_time_seconds': self.execution_time_seconds,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
        }
        
        if include_details:
            data['gameweek_results'] = self.get_gameweek_results()
        
        return data


class SeasonEntry(db.Model):
    """Active bot entry for current season competition"""
    
    __tablename__ = 'season_entries'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, unique=True, index=True)
    bot_submission_id = db.Column(db.Integer, db.ForeignKey('bot_submissions.id'), nullable=False, unique=True)
    
    # Season information
    season = db.Column(db.String(20))  # e.g., "2024-25"
    entry_gameweek = db.Column(db.Integer)  # When they entered
    
    # Current performance
    current_gameweek = db.Column(db.Integer)
    total_points = db.Column(db.Integer, default=0)
    current_rank = db.Column(db.Integer)
    team_value = db.Column(db.Float)
    bank = db.Column(db.Float)
    
    # Team data (stored as JSON)
    current_squad = db.Column(db.Text)  # Current 15 players
    
    # Performance tracking (stored as JSON)
    gameweek_history = db.Column(db.Text)  # Points, transfers, etc. per gameweek
    
    # Statistics
    total_transfers = db.Column(db.Integer, default=0)
    total_hits = db.Column(db.Integer, default=0)
    chips_used = db.Column(db.String(200))  # JSON list
    
    # Status
    is_active = db.Column(db.Boolean, default=True)
    last_update = db.Column(db.DateTime)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    def __repr__(self):
        return f'<SeasonEntry User {self.user_id} with Bot {self.bot_submission_id}>'
    
    def set_current_squad(self, data):
        """Store current squad as JSON"""
        self.current_squad = json.dumps(data)
    
    def get_current_squad(self):
        """Retrieve current squad from JSON"""
        if self.current_squad:
            return json.loads(self.current_squad)
        return None
    
    def set_gameweek_history(self, data):
        """Store gameweek history as JSON"""
        self.gameweek_history = json.dumps(data)
    
    def get_gameweek_history(self):
        """Retrieve gameweek history from JSON"""
        if self.gameweek_history:
            return json.loads(self.gameweek_history)
        return []
    
    def add_gameweek_result(self, gameweek, points, transfers, hits, captain, squad):
        """Add a new gameweek result"""
        history = self.get_gameweek_history()
        history.append({
            'gameweek': gameweek,
            'points': points,
            'transfers': transfers,
            'hits': hits,
            'captain': captain,
            'total_points': self.total_points + points,
        })
        self.set_gameweek_history(history)
        self.total_points += points
        self.total_transfers += transfers
        self.total_hits += hits
        self.current_gameweek = gameweek
    
    def to_dict(self, include_details=False):
        """Convert to dictionary"""
        data = {
            'id': self.id,
            'user_id': self.user_id,
            'bot_submission_id': self.bot_submission_id,
            'season': self.season,
            'entry_gameweek': self.entry_gameweek,
            'current_gameweek': self.current_gameweek,
            'total_points': self.total_points,
            'current_rank': self.current_rank,
            'team_value': self.team_value,
            'total_transfers': self.total_transfers,
            'total_hits': self.total_hits,
            'is_active': self.is_active,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }
        
        if include_details:
            data['current_squad'] = self.get_current_squad()
            data['gameweek_history'] = self.get_gameweek_history()
        
        return data

