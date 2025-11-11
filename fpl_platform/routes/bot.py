"""
Bot operation routes
"""
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app, make_response
from flask_login import login_required, current_user
from fpl_platform.services.bot_service import BotService
from fpl_platform.models.team import Team
import threading

bot_bp = Blueprint('bot', __name__)


@bot_bp.route('/run/<int:team_id>')
@login_required
def run_analysis(team_id):
    """Run bot analysis for a team"""
    team = Team.query.filter_by(id=team_id, user_id=current_user.id).first_or_404()
    
    return render_template('bot_run.html', team=team)


@bot_bp.route('/run/<int:team_id>/execute', methods=['POST'])
@login_required
def execute_analysis(team_id):
    """Execute bot analysis"""
    team = Team.query.filter_by(id=team_id, user_id=current_user.id).first_or_404()
    
    # Get app reference for threading
    app = current_app._get_current_object()
    
    # Run analysis in background
    def run_bot():
        with app.app_context():
            BotService.run_analysis(team, verbose=False, save_to_db=True)
    
    thread = threading.Thread(target=run_bot)
    thread.start()
    
    flash('Bot analysis started! Check back in a few moments.', 'info')
    return redirect(url_for('bot.run_analysis', team_id=team.id))


@bot_bp.route('/analysis/<int:analysis_id>')
@login_required
def view_analysis(analysis_id):
    """View analysis results"""
    analysis = BotService.get_analysis_by_id(analysis_id, current_user)
    
    if not analysis:
        flash('Analysis not found', 'danger')
        return redirect(url_for('dashboard.index'))
    
    report = analysis.get_result_data()
    
    print(f"[VIEW] Viewing analysis {analysis_id}")
    print(f"[VIEW] Status: {analysis.status}")
    print(f"[VIEW] Report exists: {report is not None}")
    if report:
        print(f"[VIEW] Report keys: {list(report.keys())}")
        opt = report.get('optimization', {})
        if opt:
            weekly_plans = opt.get('weekly_plans', [])
            print(f"[VIEW] Weekly plans count: {len(weekly_plans)}")
            if weekly_plans:
                print(f"[VIEW] First plan keys: {list(weekly_plans[0].keys())}")
    
    response = make_response(render_template(
        'analysis_detail.html',
        analysis=analysis,
        report=report
    ))
    # Prevent caching to ensure fresh data
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@bot_bp.route('/analysis/<int:analysis_id>/delete', methods=['POST'])
@login_required
def delete_analysis(analysis_id):
    """Delete an analysis"""
    success, error = BotService.delete_analysis(analysis_id, current_user)
    
    if success:
        flash('Analysis deleted', 'success')
    else:
        flash(error, 'danger')
    
    return redirect(request.referrer or url_for('dashboard.index'))


@bot_bp.route('/status/<int:team_id>')
@login_required
def analysis_status(team_id):
    """Get status of latest analysis for a team (AJAX endpoint)"""
    team = Team.query.filter_by(id=team_id, user_id=current_user.id).first_or_404()
    
    latest_analysis = team.analyses.first()
    
    if not latest_analysis:
        return jsonify({'status': 'none'})
    
    return jsonify({
        'status': latest_analysis.status,
        'analysis_id': latest_analysis.id,
        'created_at': latest_analysis.created_at.isoformat(),
        'completed_at': latest_analysis.completed_at.isoformat() if latest_analysis.completed_at else None,
        'error_message': latest_analysis.error_message
    })

