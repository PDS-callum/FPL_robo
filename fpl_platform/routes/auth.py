"""
Authentication routes
"""
from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user
from fpl_platform.services.auth_service import AuthService

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.index'))
    
    if request.method == 'POST':
        username_or_email = request.form.get('username_or_email')
        password = request.form.get('password')
        remember = request.form.get('remember', False) == 'on'
        
        success, error = AuthService.login(username_or_email, password, remember)
        
        if success:
            flash('Login successful!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page if next_page else url_for('dashboard.index'))
        else:
            flash(error, 'danger')
    
    return render_template('login.html')


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        
        # Validation
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
        elif len(password) < 6:
            flash('Password must be at least 6 characters', 'danger')
        else:
            user, error = AuthService.register_user(
                username=username,
                email=email,
                password=password,
                first_name=first_name,
                last_name=last_name
            )
            
            if user:
                flash('Registration successful! Please log in.', 'success')
                return redirect(url_for('auth.login'))
            else:
                flash(error, 'danger')
    
    return render_template('register.html')


@auth_bp.route('/logout')
@login_required
def logout():
    """Logout"""
    AuthService.logout()
    flash('You have been logged out.', 'info')
    return redirect(url_for('auth.login'))


@auth_bp.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """User profile page"""
    if request.method == 'POST':
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        email = request.form.get('email')
        
        success, error = AuthService.update_profile(
            current_user,
            first_name=first_name,
            last_name=last_name,
            email=email
        )
        
        if success:
            flash('Profile updated successfully!', 'success')
        else:
            flash(error, 'danger')
    
    # Get user's competition entry if any
    from fpl_platform.models import SeasonEntry, BotSubmission
    season_entry = SeasonEntry.query.filter_by(user_id=current_user.id).first()
    bot = None
    if season_entry:
        bot = BotSubmission.query.get(season_entry.bot_submission_id)
    
    return render_template('profile.html', season_entry=season_entry, bot=bot)


@auth_bp.route('/change-password', methods=['POST'])
@login_required
def change_password():
    """Change password"""
    old_password = request.form.get('old_password')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')
    
    if new_password != confirm_password:
        flash('New passwords do not match', 'danger')
    elif len(new_password) < 6:
        flash('Password must be at least 6 characters', 'danger')
    else:
        success, error = AuthService.change_password(
            current_user,
            old_password,
            new_password
        )
        
        if success:
            flash('Password changed successfully!', 'success')
        else:
            flash(error, 'danger')
    
    return redirect(url_for('auth.profile'))

