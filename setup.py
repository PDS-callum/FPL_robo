from setuptools import setup, find_packages

setup(
    name="fpl_bot",
    version="2.0.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'fpl_bot': ['ui/templates/*.html'],
    },
    install_requires=[
        "numpy>=1.19.5",
        "pandas>=1.2.0",
        "requests>=2.25.0",
        "python-dateutil>=2.8.2",
        "flask>=2.0.0"
    ],
    extras_require={
        "strategic": ["pulp>=2.7.0"]  # For advanced 5-GW strategic planning with MIP
    },
    entry_points={
        "console_scripts": [
            "fpl-bot=fpl_bot.main:main",
        ],
    },
    python_requires=">=3.7",
    description="Fantasy Premier League prediction and optimization bot",
    author="Callum Waller",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
)