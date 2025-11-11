from setuptools import setup, find_packages

setup(
    name="fpl_platform",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'bot': ['ui/templates/*.html'],
        'fpl_platform': ['templates/*.html', 'static/css/*.css', 'static/js/*.js'],
    },
    install_requires=[
        "numpy>=1.19.5",
        "pandas>=1.2.0",
        "requests>=2.25.0",
        "python-dateutil>=2.8.2",
        "flask>=2.0.0",
        "Flask-SQLAlchemy>=3.0.0",
        "Flask-Migrate>=4.0.0",
        "Flask-Login>=0.6.0",
        "gunicorn>=20.1.0",
        "python-dotenv>=0.19.0"
    ],
    extras_require={
        "strategic": ["pulp>=2.7.0"]  # For advanced 5-GW strategic planning with MIP
    },
    entry_points={
        "console_scripts": [
            "fpl-bot=bot.main:main",
            "fpl-platform=fpl_platform.app:main",
        ],
    },
    python_requires=">=3.7",
    description="Fantasy Premier League Platform with AI-powered bot integration",
    author="Callum Waller",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
)