"""
Analysis model for storing bot analysis results
"""
from datetime import datetime
from fpl_platform.database import db
import json
import numpy as np


class Analysis(db.Model):
    """FPL Analysis results from bot runs"""
    
    __tablename__ = 'analyses'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    team_id = db.Column(db.Integer, db.ForeignKey('teams.id'), nullable=False, index=True)
    
    # Analysis metadata
    gameweek = db.Column(db.Integer, index=True)
    status = db.Column(db.String(20), default='pending')  # pending, running, completed, failed
    
    # Results (stored as JSON)
    result_data = db.Column(db.Text)  # Full JSON report from bot
    
    # Summary fields (for quick access without parsing JSON)
    expected_points = db.Column(db.Float)
    transfers_count = db.Column(db.Integer)
    hits_taken = db.Column(db.Integer)
    captain_name = db.Column(db.String(100))
    chip_recommendation = db.Column(db.String(50))
    
    # Execution tracking
    auto_executed = db.Column(db.Boolean, default=False)
    execution_status = db.Column(db.String(20))  # success, failed, skipped
    execution_message = db.Column(db.Text)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    completed_at = db.Column(db.DateTime)
    
    # Error handling
    error_message = db.Column(db.Text)
    
    def __repr__(self):
        return f'<Analysis {self.id} for Team {self.team_id} GW{self.gameweek}>'
    
    def set_result_data(self, data):
        """Store result data as JSON"""
        # Custom JSON encoder to handle numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
        
        self.result_data = json.dumps(data, cls=NumpyEncoder)
    
    def get_result_data(self):
        """Retrieve result data from JSON"""
        if self.result_data:
            return json.loads(self.result_data)
        return None
    
    def extract_summary(self, report):
        """Extract summary fields from bot report"""
        try:
            print(f"[DEBUG] Extracting summary from report. Keys: {list(report.keys()) if report else 'None'}")
            
            # Extract optimization summary
            optimization = report.get('optimization')
            if optimization and optimization is not None:
                print(f"[DEBUG] Found optimization data. Keys: {list(optimization.keys())}")
                self.expected_points = optimization.get('objective_value')
                
                # Get first week's plan
                weekly_plans = optimization.get('weekly_plans', [])
                print(f"[DEBUG] Found {len(weekly_plans)} weekly plans")
                if weekly_plans:
                    first_week = weekly_plans[0]
                    self.gameweek = first_week.get('gameweek')
                    print(f"[DEBUG] Extracted gameweek: {self.gameweek}")
                    
                    transfers = first_week.get('transfers', {})
                    self.transfers_count = transfers.get('count', 0)
                    self.hits_taken = first_week.get('hits_taken', 0)
                    
                    captain = first_week.get('captain', {})
                    self.captain_name = captain.get('player_name')
                    print(f"[DEBUG] Extracted: GW{self.gameweek}, {self.transfers_count} transfers, captain: {self.captain_name}")
            else:
                print("[DEBUG] No optimization data found in report!")
            
            # Extract wildcard analysis if available
            wildcard = report.get('wildcard_analysis', {})
            if wildcard and wildcard.get('recommended_gw'):
                self.chip_recommendation = f"Wildcard GW{wildcard['recommended_gw']}"
            
        except Exception as e:
            print(f"[ERROR] Error extracting summary: {e}")
            import traceback
            traceback.print_exc()
    
    def to_dict(self, include_full_data=False):
        """Convert analysis to dictionary"""
        data = {
            'id': self.id,
            'team_id': self.team_id,
            'gameweek': self.gameweek,
            'status': self.status,
            'expected_points': self.expected_points,
            'transfers_count': self.transfers_count,
            'hits_taken': self.hits_taken,
            'captain_name': self.captain_name,
            'chip_recommendation': self.chip_recommendation,
            'auto_executed': self.auto_executed,
            'execution_status': self.execution_status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
        }
        
        if include_full_data:
            data['result_data'] = self.get_result_data()
        
        return data

