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
    """Main dashboard with personalized content and FPL insights"""
    from fpl_platform.models import BotSubmission, SeasonEntry
    from fpl_platform.services.fpl_data_service import FPLDataService
    
    # Left side: User data
    teams = TeamService.get_user_teams(current_user)
    primary_team = TeamService.get_primary_team(current_user)
    
    # Get user's bots
    user_bots = BotSubmission.query.filter_by(
        user_id=current_user.id
    ).order_by(BotSubmission.created_at.desc()).limit(3).all()
    
    # Get user's competition entry
    season_entry = SeasonEntry.query.filter_by(
        user_id=current_user.id,
        is_active=True
    ).first()
    
    # Right side: FPL insights
    fpl_insights = _get_fpl_insights()
    
    return render_template(
        'dashboard.html',
        teams=teams,
        primary_team=primary_team,
        user_bots=user_bots,
        season_entry=season_entry,
        insights=fpl_insights
    )


def _get_fpl_insights():
    """Generate automated FPL insights"""
    from fpl_platform.services.fpl_data_service import FPLDataService
    
    insights = {
        'trending_players': [],
        'trending_teams': [],
        'form_alerts': [],
        'current_gameweek': 1,
        'data_available': False
    }
    
    try:
        # Fetch current season data
        season_data = FPLDataService.fetch_current_season_data()
        
        if not season_data:
            return insights
        
        insights['data_available'] = True
        insights['current_gameweek'] = season_data.get('current_gameweek', 1)
        
        players = season_data.get('players', {})
        teams_data = season_data.get('teams', {})
        
        # Get trending players (high form, recent points)
        player_list = []
        for pid, player in players.items():
            form = float(player.get('form', 0))
            if form > 0:  # Only players with positive form
                player_list.append({
                    'id': pid,
                    'name': player.get('name', ''),
                    'full_name': player.get('full_name', ''),
                    'team': teams_data.get(player.get('team', 0), {}).get('short_name', 'UNK'),
                    'position': player.get('position', ''),
                    'form': form,
                    'total_points': player.get('total_points', 0),
                    'cost': player.get('cost', 0),
                    'points_per_game': float(player.get('points_per_game', 0))
                })
        
        # Sort by form and get top performers
        player_list.sort(key=lambda x: x['form'], reverse=True)
        insights['trending_players'] = player_list[:10]
        
        # Get top performing teams (by average player form)
        team_performance = {}
        for pid, player in players.items():
            team_id = player.get('team')
            if team_id and team_id in teams_data:
                if team_id not in team_performance:
                    team_performance[team_id] = {
                        'name': teams_data[team_id].get('name', ''),
                        'short_name': teams_data[team_id].get('short_name', ''),
                        'forms': [],
                        'total_points': 0
                    }
                form = float(player.get('form', 0))
                if form > 0:
                    team_performance[team_id]['forms'].append(form)
                    team_performance[team_id]['total_points'] += player.get('total_points', 0)
        
        # Calculate average form per team
        team_list = []
        for team_id, data in team_performance.items():
            if data['forms']:
                avg_form = sum(data['forms']) / len(data['forms'])
                team_list.append({
                    'name': data['name'],
                    'short_name': data['short_name'],
                    'avg_form': avg_form,
                    'total_points': data['total_points'],
                    'player_count': len(data['forms'])
                })
        
        team_list.sort(key=lambda x: x['avg_form'], reverse=True)
        insights['trending_teams'] = team_list[:6]
        
        # Form alerts (players with very high recent form)
        high_form_players = [p for p in player_list if p['form'] >= 7.0]
        insights['form_alerts'] = high_form_players[:5]
        
    except Exception as e:
        print(f"Error fetching FPL insights: {e}")
    
    return insights


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

