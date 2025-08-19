"""
Constants and configuration values used across the FPL Bot codebase.
Centralizes commonly used mappings and constants to avoid duplication.
"""

# Position mappings
POSITION_MAP = {
    1: 'GK',
    2: 'DEF', 
    3: 'MID',
    4: 'FWD'
}

POSITION_NAMES = {
    1: 'Goalkeepers',
    2: 'Defenders', 
    3: 'Midfielders',
    4: 'Forwards'
}

# Formation requirements
FORMATION_REQUIREMENTS = {
    'GK': 1,   # Exactly 1 goalkeeper
    'DEF': (3, 5),  # 3-5 defenders  
    'MID': (3, 5),  # 3-5 midfielders
    'FWD': (1, 3)   # 1-3 forwards
}

# Team constraints
TEAM_SIZE = 15
PLAYING_TEAM_SIZE = 11
BUDGET_LIMIT = 100.0  # £100 million

# Data collection settings
AVAILABLE_SEASONS = [
    '2016-17', '2017-18', '2018-19', '2019-20', 
    '2020-21', '2021-22', '2022-23', '2023-24'
]

# Limited seasons for --all flag (to avoid too much data)
LIMITED_SEASONS = [s for s in AVAILABLE_SEASONS if s >= "2019-20" and s <= "2024-25"]

# Feature engineering parameters
ROLLING_WINDOWS = [3, 5, 10]
TARGET_COLUMNS = ['points_scored', 'minutes_played', 'goals_scored', 'assists']

# Model training defaults
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32

# File patterns
CSV_EXTENSION = '.csv'
JSON_EXTENSION = '.json'
PKL_EXTENSION = '.pkl'
