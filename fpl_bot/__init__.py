"""
FPL Bot - Fantasy Premier League prediction and management bot

A clean, modular FPL bot that predicts optimal transfers and team selections
based on current season data and manager history.
"""

__version__ = "2.0.0"
__author__ = "Callum Waller"

from .core.data_collector import DataCollector
from .core.manager_analyzer import ManagerAnalyzer
from .core.chip_manager import ChipManager
from .core.team_optimizer import TeamOptimizer
from .main import FPLBot

__all__ = [
    'DataCollector',
    'ManagerAnalyzer', 
    'ChipManager',
    'TeamOptimizer',
    'FPLBot'
]

