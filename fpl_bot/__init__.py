"""
FPL Bot - Fantasy Premier League prediction and management bot

A clean, modular FPL bot that predicts optimal transfers and team selections
based on current season data and manager history.
"""

__version__ = "2.0.0"
__author__ = "Callum Waller"

from .core.data_collector import DataCollector
from .core.manager_analyzer import ManagerAnalyzer
from .core.predictor import Predictor
from .core.transfer_optimizer import TransferOptimizer
from .core.chip_manager import ChipManager
from .main import FPLBot

__all__ = [
    'DataCollector',
    'ManagerAnalyzer', 
    'Predictor',
    'TransferOptimizer',
    'ChipManager',
    'FPLBot'
]

