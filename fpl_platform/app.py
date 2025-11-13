"""
Main Flask application for FPL Vibes ✨
Where Strategy Meets Style
"""
from flask import Flask, render_template
from fpl_platform.config import config
from fpl_platform.database import init_db
from fpl_platform.routes import register_blueprints


def create_app(config_name='default'):
    """
    Create and configure Flask application
    
    Args:
        config_name: Configuration name (development, production, testing)
        
    Returns:
        Flask application instance
    """
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(config[config_name])
    
    # Initialize database
    init_db(app)
    
    # Register blueprints
    register_blueprints(app)
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return render_template('errors/404.html'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return render_template('errors/500.html'), 500
    
    # Index route (public landing page)
    @app.route('/welcome')
    def welcome():
        from flask_login import current_user
        if current_user.is_authenticated:
            from flask import redirect, url_for
            return redirect(url_for('dashboard.index'))
        return render_template('welcome.html')
    
    return app


if __name__ == '__main__':
    app = create_app('development')
    app.run(debug=True, host='127.0.0.1', port=5001)

