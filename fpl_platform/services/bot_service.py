"""
Bot integration service
"""
import sys
import os
from datetime import datetime
from pathlib import Path

# Add bot directory to path
bot_path = Path(__file__).parent.parent.parent / 'bot'
sys.path.insert(0, str(bot_path))

from bot.main import FPLBot
from fpl_platform.database import db
from fpl_platform.models.analysis import Analysis
from fpl_platform.services.team_service import TeamService


class BotService:
    """Handle bot operations and integration"""
    
    @staticmethod
    def run_analysis(team, verbose=False, save_to_db=True):
        """
        Run bot analysis for a team
        
        Args:
            team: Team instance
            verbose: Verbose output
            save_to_db: Save results to database
            
        Returns:
            tuple: (analysis, report, error_message)
        """
        analysis = None
        
        if save_to_db:
            # Create analysis record
            analysis = Analysis(
                user_id=team.user_id,
                team_id=team.id,
                status='running'
            )
            db.session.add(analysis)
            db.session.commit()
        
        try:
            # Initialize bot
            print(f"[BOT] Starting analysis for team {team.fpl_manager_id}")
            bot = FPLBot()
            
            # Run analysis with team settings (with explicit error handling)
            try:
                report = bot.run_analysis(
                    manager_id=team.fpl_manager_id,
                    optimize=True,
                    verbose=verbose,
                    risk_aversion=team.risk_aversion,
                    min_chance_of_playing=team.min_playing_chance,
                    analyze_wildcard=team.analyze_wildcard,
                    no_hits=team.no_hits
                )
                print(f"[BOT] Analysis completed. Report type: {type(report)}, Has data: {report is not None}")
            except Exception as bot_error:
                print(f"[BOT ERROR] Bot execution failed: {bot_error}")
                import traceback
                traceback.print_exc()
                # Re-raise to be caught by outer exception handler
                raise
            
            if save_to_db and analysis:
                # Update analysis with results
                print(f"[BOT] Saving analysis results to database...")
                analysis.status = 'completed'
                analysis.completed_at = datetime.utcnow()
                analysis.set_result_data(report)
                analysis.extract_summary(report)
                
                # Sync team data
                manager_info = report.get('manager_analysis', {}).get('manager_info', {})
                if manager_info:
                    TeamService.sync_team_from_fpl(team, manager_info)
                
                db.session.commit()
                print(f"[BOT] Analysis saved successfully! Analysis ID: {analysis.id}")
            
            return analysis, report, None
            
        except Exception as e:
            print(f"[BOT ERROR] Exception during analysis: {e}")
            import traceback
            traceback.print_exc()
            
            if save_to_db and analysis:
                analysis.status = 'failed'
                analysis.error_message = str(e)
                analysis.completed_at = datetime.utcnow()
                db.session.commit()
            
            return analysis, None, str(e)
    
    @staticmethod
    def get_team_analyses(team, limit=10):
        """
        Get recent analyses for a team
        
        Args:
            team: Team instance
            limit: Maximum number of analyses to return
            
        Returns:
            list: List of Analysis instances
        """
        return team.analyses.limit(limit).all()
    
    @staticmethod
    def get_analysis_by_id(analysis_id, user):
        """
        Get a specific analysis by ID
        
        Args:
            analysis_id: Analysis ID
            user: User instance (for authorization)
            
        Returns:
            Analysis instance or None
        """
        return Analysis.query.filter_by(
            id=analysis_id,
            user_id=user.id
        ).first()
    
    @staticmethod
    def delete_analysis(analysis_id, user):
        """
        Delete an analysis
        
        Args:
            analysis_id: Analysis ID
            user: User instance (for authorization)
            
        Returns:
            tuple: (success, error_message)
        """
        analysis = Analysis.query.filter_by(
            id=analysis_id,
            user_id=user.id
        ).first()
        
        if not analysis:
            return False, "Analysis not found"
        
        try:
            db.session.delete(analysis)
            db.session.commit()
            return True, None
        except Exception as e:
            db.session.rollback()
            return False, f"Delete failed: {str(e)}"
    
    @staticmethod
    def execute_transfers(team, analysis, fpl_email=None, fpl_password=None):
        """
        Execute transfers recommended by the bot
        
        Args:
            team: Team instance
            analysis: Analysis instance
            fpl_email: FPL email (optional, uses team/user credentials)
            fpl_password: FPL password (optional)
            
        Returns:
            tuple: (success, message)
        """
        # This is a placeholder for future implementation
        # Would integrate with FPL API to execute transfers
        return False, "Auto-execution not yet implemented in platform"

