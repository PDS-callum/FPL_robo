"""
Platform routes
"""
from fpl_platform.routes.auth import auth_bp
from fpl_platform.routes.dashboard import dashboard_bp
from fpl_platform.routes.bot import bot_bp
from fpl_platform.routes.api import api_bp
from fpl_platform.routes.competition import competition_bp

__all__ = ['auth_bp', 'dashboard_bp', 'bot_bp', 'api_bp', 'competition_bp']


def register_blueprints(app):
    """Register all blueprints with the app"""
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(dashboard_bp, url_prefix='/')
    app.register_blueprint(bot_bp, url_prefix='/bot')
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(competition_bp, url_prefix='/competition')

