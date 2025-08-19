from setuptools import setup, find_packages

setup(
    name="fpl_bot",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.19.5",
        "pandas>=1.2.0",
        "tensorflow>=2.4.0",
        "scikit-learn>=0.24.0",
        "requests>=2.25.0",
        "pulp>=2.4.0",
        "python-dateutil>=2.8.2"
    ],
    entry_points={
        "console_scripts": [
            "fpl-bot=fpl_bot.main:main",
        ],
    },
    python_requires=">=3.7",
    description="Fantasy Premier League prediction bot using CNN",
    author="Your Name",
)