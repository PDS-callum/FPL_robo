"""
API routes for programmatic access
"""
from flask import Blueprint, jsonify, request
from flask_login import login_required, current_user
from fpl_platform.services.bot_service import BotService
from fpl_platform.services.team_service import TeamService
from fpl_platform.models.team import Team

api_bp = Blueprint('api', __name__)


@api_bp.route('/teams', methods=['GET'])
@login_required
def get_teams():
    """Get all teams for current user"""
    teams = TeamService.get_user_teams(current_user)
    return jsonify({
        'teams': [team.to_dict() for team in teams]
    })


@api_bp.route('/teams/<int:team_id>', methods=['GET'])
@login_required
def get_team(team_id):
    """Get specific team details"""
    team = Team.query.filter_by(id=team_id, user_id=current_user.id).first()
    
    if not team:
        return jsonify({'error': 'Team not found'}), 404
    
    return jsonify(team.to_dict())


@api_bp.route('/teams/<int:team_id>/analyses', methods=['GET'])
@login_required
def get_team_analyses(team_id):
    """Get analyses for a team"""
    team = Team.query.filter_by(id=team_id, user_id=current_user.id).first()
    
    if not team:
        return jsonify({'error': 'Team not found'}), 404
    
    limit = request.args.get('limit', 10, type=int)
    analyses = BotService.get_team_analyses(team, limit=limit)
    
    return jsonify({
        'analyses': [analysis.to_dict() for analysis in analyses]
    })


@api_bp.route('/analyses/<int:analysis_id>', methods=['GET'])
@login_required
def get_analysis(analysis_id):
    """Get specific analysis with full data"""
    analysis = BotService.get_analysis_by_id(analysis_id, current_user)
    
    if not analysis:
        return jsonify({'error': 'Analysis not found'}), 404
    
    include_full = request.args.get('full', 'false').lower() == 'true'
    
    return jsonify(analysis.to_dict(include_full_data=include_full))


@api_bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})

