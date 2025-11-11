"""
Dashboard routes
"""
from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user
from fpl_platform.services.team_service import TeamService

dashboard_bp = Blueprint('dashboard', __name__)


@dashboard_bp.route('/')
@login_required
def index():
    """Main dashboard"""
    teams = TeamService.get_user_teams(current_user)
    primary_team = TeamService.get_primary_team(current_user)
    
    return render_template(
        'dashboard.html',
        teams=teams,
        primary_team=primary_team
    )


@dashboard_bp.route('/teams')
@login_required
def teams():
    """Team management page"""
    teams = TeamService.get_user_teams(current_user)
    return render_template('teams.html', teams=teams)


@dashboard_bp.route('/teams/add', methods=['GET', 'POST'])
@login_required
def add_team():
    """Add a new team"""
    if request.method == 'POST':
        fpl_manager_id = request.form.get('fpl_manager_id')
        team_name = request.form.get('team_name')
        
        if not fpl_manager_id:
            flash('FPL Manager ID is required', 'danger')
        else:
            try:
                fpl_manager_id = int(fpl_manager_id)
                team, error = TeamService.create_team(
                    current_user,
                    fpl_manager_id,
                    team_name=team_name
                )
                
                if team:
                    flash('Team added successfully!', 'success')
                    return redirect(url_for('dashboard.teams'))
                else:
                    flash(error, 'danger')
            except ValueError:
                flash('Invalid FPL Manager ID', 'danger')
    
    return render_template('add_team.html')


@dashboard_bp.route('/teams/<int:team_id>')
@login_required
def team_detail(team_id):
    """Team detail page"""
    from fpl_platform.models.team import Team
    from fpl_platform.services.bot_service import BotService
    
    team = Team.query.filter_by(id=team_id, user_id=current_user.id).first_or_404()
    recent_analyses = BotService.get_team_analyses(team, limit=10)
    
    return render_template(
        'team_detail.html',
        team=team,
        recent_analyses=recent_analyses
    )


@dashboard_bp.route('/teams/<int:team_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_team(team_id):
    """Edit team settings"""
    from fpl_platform.models.team import Team
    
    team = Team.query.filter_by(id=team_id, user_id=current_user.id).first_or_404()
    
    if request.method == 'POST':
        # Update team settings
        team_name = request.form.get('team_name')
        risk_aversion = float(request.form.get('risk_aversion', 0.5))
        min_playing_chance = int(request.form.get('min_playing_chance', 75))
        analyze_wildcard = request.form.get('analyze_wildcard') == 'on'
        no_hits = request.form.get('no_hits') == 'on'
        auto_run_enabled = request.form.get('auto_run_enabled') == 'on'
        
        success, error = TeamService.update_team(
            team,
            team_name=team_name,
            risk_aversion=risk_aversion,
            min_playing_chance=min_playing_chance,
            analyze_wildcard=analyze_wildcard,
            no_hits=no_hits,
            auto_run_enabled=auto_run_enabled
        )
        
        if success:
            flash('Team settings updated!', 'success')
            return redirect(url_for('dashboard.team_detail', team_id=team.id))
        else:
            flash(error, 'danger')
    
    return render_template('edit_team.html', team=team)


@dashboard_bp.route('/teams/<int:team_id>/set-primary', methods=['POST'])
@login_required
def set_primary_team(team_id):
    """Set team as primary"""
    success, error = TeamService.set_primary_team(current_user, team_id)
    
    if success:
        flash('Primary team updated!', 'success')
    else:
        flash(error, 'danger')
    
    return redirect(url_for('dashboard.teams'))


@dashboard_bp.route('/teams/<int:team_id>/delete', methods=['POST'])
@login_required
def delete_team(team_id):
    """Delete team"""
    success, error = TeamService.delete_team(current_user, team_id)
    
    if success:
        flash('Team deleted successfully', 'success')
    else:
        flash(error, 'danger')
    
    return redirect(url_for('dashboard.teams'))

