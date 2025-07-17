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
