# FPL Robo

A Fantasy Premier League prediction bot using CNN modeling with comprehensive data processing and feature engineering.

## Installation

```bash
# Clone the repository
git clone https://github.com/PDS-callum/FPL_robo.git
cd fpl_robo

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

### 1. Data Processing

Process historical FPL data for machine learning:

```bash
# Basic data processing
python process_fpl_data.py

# Advanced feature engineering with validation
python process_fpl_data.py --advanced --validate --feature-importance

# Process specific seasons
python process_fpl_data.py --seasons 2023-24 2024-25 --advanced
```

### 2. Example Usage

```bash
# Run data processing examples
python examples/data_processing_example.py

# Collect FPL data from GitHub archive
python -m fpl_bot.main collect --seasons 2022-23 2023-24

# Process collected historical data for ML training
python -m fpl_bot.main process --seasons 2022-23 2023-24

# Collect and process data in one command
python -m fpl_bot.main collect-and-process --all

# Process data with specific targets
python -m fpl_bot.main process --targets points_scored goals_scored assists

# Train model with historical data
fpl-bot train-with-history --all --epochs 50 --batch_size 32

# Predict team for new season
fpl-bot predict --gameweek 1 --use-history --next-season

# Generate team analysis report and update README
python -m fpl_bot.main generate-readme
```

## Features

### Data Collection & Processing
- **Historical Data Collection**: Automated collection from FPL GitHub archive
- **Comprehensive Feature Engineering**: 50+ features including rolling statistics, momentum indicators, opponent analysis
- **Advanced Preprocessing**: Position-specific features, team form analysis, value metrics
- **Data Quality Validation**: Automated quality checks and outlier detection
- **ML-Ready Datasets**: Properly scaled and encoded data for multiple prediction tasks

### Machine Learning Pipeline
- **Multi-Target Prediction**: Points, minutes, goals, assists prediction
- **Feature Importance Analysis**: Identify the most predictive features
- **Time Series Handling**: Proper chronological data splitting
- **Model Training**: CNN models for player performance prediction

### Team Optimization
- **Budget Constraints**: Optimal team selection within £100M budget
- **Formation Requirements**: Automatic formation compliance (GK, DEF, MID, FWD)
- **Captain Selection**: Data-driven captain recommendations
- **Transfer Planning**: Weekly team optimization with transfer costs

## Data Processing Options

### Basic Processing
```bash
# Process all available data
python process_fpl_data.py

# Specific target variables
python process_fpl_data.py --targets points_scored minutes_played
```

### Advanced Processing
```bash
# Full feature engineering pipeline
python process_fpl_data.py --advanced --validate --feature-importance

# Custom data directory
python process_fpl_data.py --data-dir /path/to/data --advanced
```

### Team Composition Analysis
```bash
# Generate team analysis and update README
python -m fpl_bot.main generate-readme

# Print analysis without updating README
python -m fpl_bot.main generate-readme --print-only

# Specify custom data directory and README path
python -m fpl_bot.main generate-readme --data-dir custom/data --readme-path docs/README.md

# Legacy commands (if available)
python analyze_team_composition.py
python analysis_cli.py --mode quick
```

*The `generate-readme` command reads team prediction data from the season and automatically updates the README with fresh analysis including player loyalty, team distribution, formation usage, and gameweek summaries.*

## Project Structure

```
fpl_robo/
├── fpl_bot/
│   ├── utils/
│   │   ├── data_collection.py      # Data collection and basic processing
│   │   └── data_preprocessing.py   # Advanced feature engineering
│   └── main.py
├── data/
│   ├── historical/                 # Raw historical data by season
│   ├── processed/                  # Processed ML datasets
│   ├── features/                   # Feature files and importance analysis
│   └── models/                     # Trained model storage
├── examples/
│   └── data_processing_example.py  # Usage examples
├── process_fpl_data.py             # Main processing pipeline
├── DATA_PROCESSING_README.md       # Detailed processing documentation
└── requirements.txt
```

## Data Features

### Player Performance Features
- **Basic Stats**: Points, minutes, goals, assists, clean sheets
- **Advanced Metrics**: BPS, influence, creativity, threat
- **Historical Trends**: Rolling averages (3, 5, 10 gameweeks)
- **Form Indicators**: Points streaks, consistency metrics

### Team & Opponent Features  
- **Team Strength**: Attack/defense ratings for home/away
- **Fixture Analysis**: Opponent strength and difficulty ratings
- **Team Form**: Recent results and performance trends

### Value & Selection Features
- **Price Analysis**: Points per million, price categories
- **Ownership**: Selection percentages and differential identification
- **Position-Specific**: Expected points by player position

### Advanced Features (with --advanced flag)
- **Momentum Analysis**: Points/minutes streaks, form indicators
- **Rolling Statistics**: Comprehensive statistical windows
- **Opponent Strength**: Dynamic opponent analysis
- **Team Form**: Recent team performance metrics

## Machine Learning Targets

The system prepares data for multiple prediction tasks:

- **Points Prediction** (`points_scored`): Primary target for team selection
- **Minutes Prediction** (`minutes_played`): Rotation and starter analysis  
- **Goals Prediction** (`goals_scored`): Captain selection optimization
- **Assists Prediction** (`assists`): Creativity and playmaker identification

## Documentation

- **[Data Processing Guide](DATA_PROCESSING_README.md)**: Comprehensive processing documentation
- **[Examples](examples/)**: Usage examples and tutorials
- **Feature Engineering**: Advanced feature creation methods
- **Model Training**: CNN model development and optimization


# FPL Season Team Analysis Report

**Generated on:** 2025-08-22 19:13:44
**Last Updated:** 2025-08-14T10:07:31.186526
**Total Gameweeks Analyzed:** 1

## Season Summary

- **Average Team Cost:** £94.5M
- **Average Predicted Points:** 2479.7
- **Budget Utilization:** 0.0% (Latest GW)
- **Total Transfers Made:** 0

## Formation Usage

- **3-4-3:** 1 gameweeks (100.0%)

## Captain Choices

- **Mbeumo:** 1 gameweeks (100.0%)

## Most Loyal Players (Top 10)

- **Pickford:** 1/1 gameweeks (100.0%)
- **MilenkoviÄ‡:** 1/1 gameweeks (100.0%)
- **MuÃ±oz:** 1/1 gameweeks (100.0%)
- **Collins:** 1/1 gameweeks (100.0%)
- **Mbeumo:** 1/1 gameweeks (100.0%)
- **Cunha:** 1/1 gameweeks (100.0%)
- **Semenyo:** 1/1 gameweeks (100.0%)
- **J.Murphy:** 1/1 gameweeks (100.0%)
- **Wood:** 1/1 gameweeks (100.0%)
- **Wissa:** 1/1 gameweeks (100.0%)

## Team Distribution (Most Represented)

- **Nott'm Forest:** 3 player selections
- **Crystal Palace:** 2 player selections
- **Brentford:** 2 player selections
- **Man Utd:** 2 player selections
- **West Ham:** 2 player selections
- **Everton:** 1 player selections
- **Bournemouth:** 1 player selections
- **Newcastle:** 1 player selections
- **Fulham:** 1 player selections

## Position Analysis

### GKP
- **Pickford:** 1 appearances (100.0%)

### DEF
- **Milenković:** 1 appearances (100.0%)
- **Muñoz:** 1 appearances (100.0%)
- **Collins:** 1 appearances (100.0%)

### MID
- **Mbeumo:** 1 appearances (100.0%)
- **Cunha:** 1 appearances (100.0%)
- **Semenyo:** 1 appearances (100.0%)
- **J.Murphy:** 1 appearances (100.0%)

### FWD
- **Wood:** 1 appearances (100.0%)
- **Wissa:** 1 appearances (100.0%)
- **Bowen:** 1 appearances (100.0%)

# FPL Season Team Analysis Report

**Generated on:** 2025-08-22 15:56:44
**Last Updated:** 2025-08-19T21:21:28.640404
**Total Gameweeks Analyzed:** 1

## Season Summary

- **Average Team Cost:** £94.5M
- **Average Predicted Points:** 2479.7
- **Budget Utilization:** 94.5% (Latest GW)
- **Total Transfers Made:** 0

## Formation Usage

- **3-4-3:** 1 gameweeks (100.0%)

## Captain Choices

- **Mbeumo:** 1 gameweeks (100.0%)

## Most Loyal Players (Top 10)

- **Pickford:** 1/1 gameweeks (100.0%)
- **Milenković:** 1/1 gameweeks (100.0%)
- **Muñoz:** 1/1 gameweeks (100.0%)
- **Collins:** 1/1 gameweeks (100.0%)
- **Mbeumo:** 1/1 gameweeks (100.0%)
- **Cunha:** 1/1 gameweeks (100.0%)
- **Semenyo:** 1/1 gameweeks (100.0%)
- **J.Murphy:** 1/1 gameweeks (100.0%)
- **Wood:** 1/1 gameweeks (100.0%)
- **Wissa:** 1/1 gameweeks (100.0%)

## Team Distribution (Most Represented)

- **Nott'm Forest:** 3 player selections
- **Crystal Palace:** 2 player selections
- **Brentford:** 2 player selections
- **Man Utd:** 2 player selections
- **West Ham:** 2 player selections
- **Everton:** 1 player selections
- **Bournemouth:** 1 player selections
- **Newcastle:** 1 player selections
- **Fulham:** 1 player selections

## Position Analysis

### GKP
- **Pickford:** 1 appearances (100.0%)
- **Sels:** 1 appearances (100.0%)

### DEF
- **Milenković:** 1 appearances (100.0%)
- **Muñoz:** 1 appearances (100.0%)
- **Collins:** 1 appearances (100.0%)

### MID
- **Mbeumo:** 1 appearances (100.0%)
- **Cunha:** 1 appearances (100.0%)
- **Semenyo:** 1 appearances (100.0%)

### FWD
- **Wood:** 1 appearances (100.0%)
- **Wissa:** 1 appearances (100.0%)
- **Bowen:** 1 appearances (100.0%)

# FPL Season Team Composition Analysis

*Analysis generated on 2025-08-22 19:13:44*

## Season Overview

- **Total Gameweeks Analyzed:** 1
- **Average Team Cost:** £94.5m
- **Average Predicted Points:** 2479.7
- **Average Budget Utilization:** 0.0%

## Formation Analysis

### Most Used Formations

- **3-4-3:** 1 times (100.0%)

## Most Consistent Players

*Players selected in 50% or more gameweeks*

- **Pickford:** 1/1 gameweeks (100.0%)
- **MilenkoviÄ‡:** 1/1 gameweeks (100.0%)
- **MuÃ±oz:** 1/1 gameweeks (100.0%)
- **Collins:** 1/1 gameweeks (100.0%)
- **Mbeumo:** 1/1 gameweeks (100.0%)
- **Cunha:** 1/1 gameweeks (100.0%)
- **Semenyo:** 1/1 gameweeks (100.0%)
- **J.Murphy:** 1/1 gameweeks (100.0%)
- **Wood:** 1/1 gameweeks (100.0%)
- **Wissa:** 1/1 gameweeks (100.0%)
- **Bowen:** 1/1 gameweeks (100.0%)
- **GuÃ©hi:** 1/1 gameweeks (100.0%)
- **Wan-Bissaka:** 1/1 gameweeks (100.0%)
- **Iwobi:** 1/1 gameweeks (100.0%)
- **Sels:** 1/1 gameweeks (100.0%)

## Captaincy Analysis

### Most Captained Players

- **Mbeumo:** 1 times (100.0%)

### Most Vice-Captained Players

- **Wood:** 1 times (100.0%)

## Team Distribution Analysis

### Most Represented Teams

- **Nott'm Forest:** 3 player selections (avg 3.0 per GW)
- **Crystal Palace:** 2 player selections (avg 2.0 per GW)
- **Brentford:** 2 player selections (avg 2.0 per GW)
- **Man Utd:** 2 player selections (avg 2.0 per GW)
- **West Ham:** 2 player selections (avg 2.0 per GW)
- **Everton:** 1 player selections (avg 1.0 per GW)
- **Bournemouth:** 1 player selections (avg 1.0 per GW)
- **Newcastle:** 1 player selections (avg 1.0 per GW)
- **Fulham:** 1 player selections (avg 1.0 per GW)

## Position Distribution

- **DEF:** 5 selections (33.3%, avg 5.0 per GW)
- **MID:** 5 selections (33.3%, avg 5.0 per GW)
- **FWD:** 3 selections (20.0%, avg 3.0 per GW)
- **GKP:** 2 selections (13.3%, avg 2.0 per GW)

## Gameweek Summary

| GW | Formation | Cost | Predicted Pts | Teams Used | Captain | Vice-Captain |
|----|-----------|------|---------------|------------|---------|--------------|
| 1 | 3-4-3 | £94.5m | 2479.7 | 8 | Mbeumo | Wood |
