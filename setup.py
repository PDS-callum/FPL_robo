from setuptools import setup, find_packages

setup(
    name="fpl_bot",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "tensorflow",
        "scikit-learn",
        "matplotlib",
        "requests",
        "pulp"
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