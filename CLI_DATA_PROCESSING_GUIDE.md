# FPL Bot CLI Data Processing Guide

This guide explains how to use the new data processing functionality added to the FPL Bot CLI.

## Overview

The FPL Bot now includes comprehensive data processing capabilities that can:

1. **Collect** historical FPL data from the GitHub archive
2. **Process** the collected CSV data into ML-ready format
3. **Combine** both steps in a single command for convenience

## Available Commands

### 1. Collect Data Only

```bash
# Collect data for specific seasons
python -m fpl_bot.main collect --seasons 2023-24

# Collect data for multiple seasons
python -m fpl_bot.main collect --seasons 2022-23 2023-24

# Collect data for all available seasons (2019-20 to 2024-25)
python -m fpl_bot.main collect --all
```

**What this does:**
- Downloads CSV files from the FPL GitHub archive
- Saves data to `data/historical/<season>/` directories
- Creates files: `fixtures.csv`, `merged_gw.csv`, `players_raw.csv`, `teams.csv`

### 2. Process Data Only

```bash
# Process data for specific seasons (requires existing data)
python -m fpl_bot.main process --seasons 2023-24

# Process with custom target variables
python -m fpl_bot.main process --targets points_scored goals_scored assists

# Process all available data
python -m fpl_bot.main process
```

**What this does:**
- Reads CSV files from `data/historical/` directories
- Creates comprehensive player features (50+ features)
- Prepares ML-ready datasets for multiple prediction targets
- Saves processed data to `data/processed/` and `data/features/`

### 3. Collect and Process (Recommended)

```bash
# Collect and process specific seasons
python -m fpl_bot.main collect-and-process --seasons 2023-24

# Collect and process all available seasons
python -m fpl_bot.main collect-and-process --all

# Custom targets and seasons
python -m fpl_bot.main collect-and-process --seasons 2022-23 2023-24 --targets points_scored minutes_played
```

**What this does:**
- Combines both collection and processing steps
- Most convenient for getting started
- Ensures data consistency between collection and processing

## Data Structure

### Input Data (Historical CSV Files)
```
data/historical/
├── 2023-24/
│   ├── fixtures.csv      # Match fixtures and results
│   ├── merged_gw.csv     # Player performance by gameweek
│   ├── players_raw.csv   # Player information and season stats
│   └── teams.csv         # Team information and strength ratings
├── 2022-23/
│   └── ...
└── ...
```

### Output Data (Processed ML Files)
```
data/
├── processed/
│   ├── fpl_ml_dataset.csv    # Main ML-ready dataset
│   ├── label_encoders.pkl    # Categorical variable encoders
│   └── scalers.pkl           # Feature scalers
└── features/
    ├── features_points_scored.csv   # Features for points prediction
    ├── target_points_scored.csv     # Points target values
    ├── features_goals_scored.csv    # Features for goals prediction
    ├── target_goals_scored.csv      # Goals target values
    └── feature_names_*.json         # Feature name mappings
```

## Feature Engineering

The processing pipeline creates comprehensive features including:

### Player Performance Features
- **Basic Stats**: Points, minutes, goals, assists, clean sheets
- **Historical Trends**: Rolling averages (3, 5, 10 gameweeks)
- **Form Indicators**: Points streaks, consistency metrics
- **Season Totals**: Cumulative statistics

### Team & Opponent Features
- **Team Strength**: Attack/defense ratings for home/away
- **Fixture Analysis**: Opponent strength and difficulty ratings
- **Team Form**: Recent results and performance trends

### Value & Selection Features
- **Price Analysis**: Points per million, price categories
- **Position-Specific**: Expected points by player position

## Target Variables

The system prepares data for multiple prediction tasks:

- **`points_scored`**: Primary target for team selection
- **`minutes_played`**: Rotation and starter analysis
- **`goals_scored`**: Captain selection optimization
- **`assists`**: Creativity and playmaker identification

## Usage Examples

### Quick Start
```bash
# Get everything set up quickly
python -m fpl_bot.main collect-and-process --seasons 2023-24
```

### Historical Analysis
```bash
# Process multiple seasons for comprehensive training data
python -m fpl_bot.main collect-and-process --seasons 2021-22 2022-23 2023-24
```

### Custom Prediction Targets
```bash
# Focus on specific prediction tasks
python -m fpl_bot.main collect-and-process --all --targets points_scored goals_scored
```

## Integration with Existing Workflow

This new functionality integrates seamlessly with the existing FPL Bot pipeline:

1. **Data Collection**: Use the new CLI commands
2. **Model Training**: Use existing `train_model.py` or CNN models
3. **Team Prediction**: Use existing `predict_team.py`

## Error Handling

The system includes robust error handling for:
- Missing CSV files
- Malformed data
- Network issues during collection
- Encoding problems

## Performance

- **Collection**: ~1-2 minutes per season
- **Processing**: ~30 seconds to 2 minutes depending on data size
- **Storage**: ~10-50MB per season depending on detail level

## Next Steps

After processing data, you can:

1. Train ML models using the processed features
2. Analyze feature importance for player performance
3. Build custom prediction models
4. Integrate with team optimization algorithms

For more advanced usage, see the examples in the `examples/` directory.
