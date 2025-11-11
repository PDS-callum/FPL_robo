"""
Competition routes for bot battles
"""
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app
from flask_login import login_required, current_user
from fpl_platform.services.bot_execution_service import BotExecutionService
from fpl_platform.models import BotSubmission, BacktestResult, SeasonEntry, User
from fpl_platform.database import db
import threading

competition_bp = Blueprint('competition', __name__)


@competition_bp.route('/')
@login_required
def index():
    """Competition home page"""
    # Get user's bot submissions
    submissions = BotSubmission.query.filter_by(user_id=current_user.id).order_by(
        BotSubmission.created_at.desc()
    ).all()
    
    # Get user's season entry if active
    season_entry = SeasonEntry.query.filter_by(user_id=current_user.id).first()
    
    # Get leaderboard (top 10)
    leaderboard = BotExecutionService.get_leaderboard(limit=10)
    
    return render_template(
        'competition/index.html',
        submissions=submissions,
        season_entry=season_entry,
        leaderboard=leaderboard
    )


@competition_bp.route('/submit', methods=['GET', 'POST'])
@login_required
def submit_bot():
    """Submit a new bot"""
    if request.method == 'GET':
        return render_template('competition/submit_bot.html')
    
    # Process submission
    bot_name = request.form.get('bot_name')
    bot_description = request.form.get('bot_description', '')
    
    # Get uploaded file
    if 'bot_file' not in request.files:
        flash('No file uploaded', 'danger')
        return redirect(url_for('competition.submit_bot'))
    
    file = request.files['bot_file']
    
    if file.filename == '':
        flash('No file selected', 'danger')
        return redirect(url_for('competition.submit_bot'))
    
    # Check file extension
    if not file.filename.endswith('.py'):
        flash('File must be a Python (.py) file', 'danger')
        return redirect(url_for('competition.submit_bot'))
    
    # Read file content
    try:
        bot_code = file.read().decode('utf-8')
    except Exception as e:
        flash(f'Error reading file: {str(e)}', 'danger')
        return redirect(url_for('competition.submit_bot'))
    
    if not bot_name or not bot_code:
        flash('Bot name and code are required', 'danger')
        return redirect(url_for('competition.submit_bot'))
    
    # Create submission
    submission, error = BotExecutionService.create_bot_submission(
        current_user.id,
        bot_name,
        bot_description,
        bot_code
    )
    
    if error:
        flash(f'Error creating submission: {error}', 'danger')
        return redirect(url_for('competition.submit_bot'))
    
    if not submission.is_validated:
        flash(f'Bot validation failed: {submission.validation_error}', 'warning')
    else:
        flash('Bot submitted successfully!', 'success')
    
    return redirect(url_for('competition.view_submission', submission_id=submission.id))


@competition_bp.route('/submission/<int:submission_id>')
@login_required
def view_submission(submission_id):
    """View bot submission details"""
    submission = BotSubmission.query.filter_by(
        id=submission_id,
        user_id=current_user.id
    ).first_or_404()
    
    # Get backtests for this submission
    backtests = BacktestResult.query.filter_by(
        bot_submission_id=submission_id
    ).order_by(BacktestResult.created_at.desc()).all()
    
    return render_template(
        'competition/submission_detail.html',
        submission=submission,
        backtests=backtests
    )


@competition_bp.route('/submission/<int:submission_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_submission(submission_id):
    """Edit bot submission"""
    submission = BotSubmission.query.filter_by(
        id=submission_id,
        user_id=current_user.id
    ).first_or_404()
    
    if request.method == 'GET':
        return render_template('competition/edit_bot.html', submission=submission)
    
    # Update submission
    submission.bot_name = request.form.get('bot_name', submission.bot_name)
    submission.bot_description = request.form.get('bot_description', submission.bot_description)
    
    # Check if a new file was uploaded
    new_code = None
    if 'bot_file' in request.files:
        file = request.files['bot_file']
        if file.filename != '':
            # Check file extension
            if not file.filename.endswith('.py'):
                flash('File must be a Python (.py) file', 'danger')
                return redirect(url_for('competition.edit_submission', submission_id=submission.id))
            
            # Read file content
            try:
                new_code = file.read().decode('utf-8')
            except Exception as e:
                flash(f'Error reading file: {str(e)}', 'danger')
                return redirect(url_for('competition.edit_submission', submission_id=submission.id))
    
    # Fallback to textarea if no file uploaded
    if not new_code:
        new_code = request.form.get('bot_code')
    
    if new_code and new_code != submission.bot_code:
        # Re-validate code
        is_valid, error_msg, _ = BotExecutionService.validate_bot_code(new_code)
        submission.bot_code = new_code
        submission.is_validated = is_valid
        submission.validation_error = error_msg
        
        if not is_valid:
            flash(f'Bot validation failed: {error_msg}', 'warning')
        else:
            flash('Bot updated and validated successfully!', 'success')
    
    db.session.commit()
    
    return redirect(url_for('competition.view_submission', submission_id=submission.id))


@competition_bp.route('/submission/<int:submission_id>/delete', methods=['POST'])
@login_required
def delete_submission(submission_id):
    """Delete bot submission"""
    submission = BotSubmission.query.filter_by(
        id=submission_id,
        user_id=current_user.id
    ).first_or_404()
    
    # Check if it's active in a season
    if submission.is_active:
        flash('Cannot delete active bot. Deactivate it first.', 'danger')
        return redirect(url_for('competition.view_submission', submission_id=submission.id))
    
    db.session.delete(submission)
    db.session.commit()
    
    flash('Bot submission deleted', 'success')
    return redirect(url_for('competition.index'))


@competition_bp.route('/submission/<int:submission_id>/backtest', methods=['POST'])
@login_required
def run_backtest(submission_id):
    """Run backtest for a bot submission"""
    submission = BotSubmission.query.filter_by(
        id=submission_id,
        user_id=current_user.id
    ).first_or_404()
    
    if not submission.is_validated:
        flash('Bot must be validated before backtesting', 'danger')
        return redirect(url_for('competition.view_submission', submission_id=submission.id))
    
    # Get backtest parameters
    season = request.form.get('season', '2023-24')
    use_current_season = request.form.get('use_current_season') == 'true'
    start_gw = int(request.form.get('start_gw', 1))
    end_gw = int(request.form.get('end_gw', 38))
    
    # Get app reference for threading
    app = current_app._get_current_object()
    
    # Run backtest in background
    def run_backtest_thread():
        with app.app_context():
            BotExecutionService.run_backtest(
                submission_id, 
                season, 
                start_gw, 
                end_gw,
                use_current_season=use_current_season
            )
    
    thread = threading.Thread(target=run_backtest_thread)
    thread.start()
    
    flash_msg = 'Backtest started! Using real data from current season.' if use_current_season else 'Backtest started! Check back in a few moments.'
    flash(flash_msg, 'info')
    return redirect(url_for('competition.view_submission', submission_id=submission.id))


@competition_bp.route('/backtest/<int:backtest_id>')
@login_required
def view_backtest(backtest_id):
    """View backtest results"""
    backtest = BacktestResult.query.filter_by(
        id=backtest_id,
        user_id=current_user.id
    ).first_or_404()
    
    submission = BotSubmission.query.get(backtest.bot_submission_id)
    
    return render_template(
        'competition/backtest_detail.html',
        backtest=backtest,
        submission=submission
    )


@competition_bp.route('/submission/<int:submission_id>/activate', methods=['POST'])
@login_required
def activate_bot(submission_id):
    """Activate bot for current season competition"""
    submission = BotSubmission.query.filter_by(
        id=submission_id,
        user_id=current_user.id
    ).first_or_404()
    
    if not submission.is_validated:
        flash('Bot must be validated before activation', 'danger')
        return redirect(url_for('competition.view_submission', submission_id=submission.id))
    
    season = request.form.get('season', '2024-25')
    
    entry, error = BotExecutionService.activate_bot_for_season(
        current_user.id,
        submission_id,
        season
    )
    
    if error:
        flash(f'Error activating bot: {error}', 'danger')
    else:
        flash('Bot activated for current season!', 'success')
    
    return redirect(url_for('competition.index'))


@competition_bp.route('/leaderboard')
@login_required
def leaderboard():
    """View full competition leaderboard"""
    season = request.args.get('season', '2024-25')
    page = request.args.get('page', 1, type=int)
    per_page = 50
    
    # Get all entries for season
    entries_query = SeasonEntry.query.filter_by(
        season=season,
        is_active=True
    ).order_by(SeasonEntry.total_points.desc())
    
    total_entries = entries_query.count()
    entries = entries_query.limit(per_page).offset((page - 1) * per_page).all()
    
    # Format leaderboard data
    leaderboard_data = []
    for rank, entry in enumerate(entries, start=(page - 1) * per_page + 1):
        user = User.query.get(entry.user_id)
        bot = BotSubmission.query.get(entry.bot_submission_id)
        
        leaderboard_data.append({
            'rank': rank,
            'entry': entry,
            'user': user,
            'bot': bot
        })
    
    total_pages = (total_entries + per_page - 1) // per_page
    
    return render_template(
        'competition/leaderboard.html',
        leaderboard=leaderboard_data,
        season=season,
        page=page,
        total_pages=total_pages,
        total_entries=total_entries
    )


@competition_bp.route('/my-entry')
@login_required
def my_entry():
    """View user's current season entry"""
    entry = SeasonEntry.query.filter_by(user_id=current_user.id).first()
    
    if not entry:
        flash('You do not have an active entry', 'info')
        return redirect(url_for('competition.index'))
    
    bot = BotSubmission.query.get(entry.bot_submission_id)
    
    return render_template(
        'competition/my_entry.html',
        entry=entry,
        bot=bot
    )


@competition_bp.route('/documentation')
@login_required
def documentation():
    """Bot development documentation"""
    return render_template('competition/documentation.html')


# API endpoints for AJAX requests
@competition_bp.route('/api/submission/<int:submission_id>/status')
@login_required
def get_submission_status(submission_id):
    """Get submission validation status"""
    submission = BotSubmission.query.filter_by(
        id=submission_id,
        user_id=current_user.id
    ).first_or_404()
    
    return jsonify({
        'is_validated': submission.is_validated,
        'validation_error': submission.validation_error,
        'is_active': submission.is_active
    })


@competition_bp.route('/api/backtest/<int:backtest_id>/status')
@login_required
def get_backtest_status(backtest_id):
    """Get backtest status"""
    backtest = BacktestResult.query.filter_by(
        id=backtest_id,
        user_id=current_user.id
    ).first_or_404()
    
    return jsonify({
        'status': backtest.status,
        'total_points': backtest.total_points,
        'completed_at': backtest.completed_at.isoformat() if backtest.completed_at else None
    })

