# FPL Data Processing Documentation

This document explains how to use the FPL data processing pipeline to transform raw Fantasy Premier League data into a format suitable for machine learning.

## Overview

The data processing system consists of several components:

1. **FPLDataProcessor** - Basic data processing and feature creation
2. **AdvancedFPLPreprocessor** - Advanced feature engineering
3. **process_fpl_data.py** - Main processing pipeline script

## Data Structure

### Input Data
The system expects data in the following structure:
```
data/
├── historical/
│   ├── 2023-24/
│   │   ├── fixtures.csv
│   │   ├── merged_gw.csv
│   │   ├── players_raw.csv
│   │   └── teams.csv
│   └── 2024-25/
│       └── ... (same structure)
└── raw/
    ├── bootstrap_static_*.json
    └── event_*_*.json
```

### Output Data
After processing, the following files are created:
```
data/
├── processed/
│   ├── fpl_ml_dataset.csv          # Basic processed dataset
│   ├── fpl_dataset_advanced.csv    # Advanced feature dataset
│   ├── label_encoders.pkl          # Saved encoders
│   ├── scalers.pkl                 # Saved scalers
│   └── data_quality_report.json    # Quality validation
└── features/
    ├── features_points_scored.csv   # Features for points prediction
    ├── target_points_scored.csv     # Target values for points
    ├── feature_names_points_scored.json
    └── feature_importance_*.csv     # Feature importance analysis
```

## Usage

### Basic Data Processing

```python
from fpl_bot.utils.data_collection import FPLDataProcessor

# Initialize processor
processor = FPLDataProcessor(data_dir="data")

# Process all available seasons
final_dataset, datasets = processor.process_all_data()

# Process specific seasons
final_dataset, datasets = processor.process_all_data(
    seasons=['2023-24', '2024-25'],
    target_columns=['points_scored', 'minutes_played', 'goals_scored', 'assists']
)
```

### Advanced Feature Engineering

```python
from fpl_bot.utils.data_preprocessing import AdvancedFPLPreprocessor

# Initialize advanced processor
advanced_processor = AdvancedFPLPreprocessor(data_dir="data")

# Create advanced features
advanced_dataset = advanced_processor.create_advanced_features(
    basic_dataset, teams_df, fixtures_df
)

# Analyze feature importance
importance_df = advanced_processor.feature_importance_analysis(
    X, y, feature_names, 'points_scored'
)
```

### Command Line Usage

```bash
# Basic processing
python process_fpl_data.py

# Process specific seasons with advanced features
python process_fpl_data.py --seasons 2023-24 2024-25 --advanced --validate

# Full processing with feature importance analysis
python process_fpl_data.py --advanced --validate --feature-importance

# Custom target variables
python process_fpl_data.py --targets points_scored minutes_played --advanced
```

### Command Line Options

- `--data-dir`: Specify data directory (default: 'data')
- `--seasons`: Specific seasons to process (e.g., 2023-24 2024-25)
- `--targets`: Target variables for ML (default: points_scored, minutes_played, goals_scored, assists)
- `--advanced`: Enable advanced feature engineering
- `--validate`: Run data quality validation
- `--feature-importance`: Analyze feature importance (requires --advanced)

## Features Created

### Basic Features
- **Player Info**: position, team, price, ownership percentage
- **Performance**: points, minutes, goals, assists, clean sheets
- **Historical Stats**: rolling averages (3, 5, 10 gameweeks)
- **Form Metrics**: recent form, season totals, consistency measures
- **Team Features**: team strengths, fixture difficulty
- **Value Metrics**: points per million, price categories

### Advanced Features
- **Rolling Statistics**: mean, std, max for multiple windows
- **Momentum Features**: points streaks, form indicators, starter status
- **Opponent Analysis**: opponent strength, fixture difficulty adjustments
- **Position-Specific**: expected points by position type
- **Team Form**: recent team results and performance
- **Value Analysis**: differential players, ownership categories

## Feature Categories

### Player Performance Features
- `points_scored`, `minutes_played`, `goals_scored`, `assists`
- `clean_sheets`, `goals_conceded`, `yellow_cards`, `red_cards`
- `saves`, `bonus`, `bps`, `influence`, `creativity`, `threat`

### Historical Features
- `avg_points_3gw`, `avg_points_5gw` - rolling averages
- `form_3gw`, `form_5gw` - recent form scores
- `total_points_season`, `games_played_season` - season aggregates
- `points_std`, `minutes_consistency` - consistency metrics

### Advanced Features
- `points_streak`, `blank_streak` - momentum indicators
- `is_in_form`, `is_regular_starter` - form flags
- `opponent_strength`, `fixture_difficulty_adjusted` - opponent analysis
- `expected_cs_points`, `expected_goal_points` - position-specific expectations
- `points_per_million`, `is_differential` - value metrics

### Team Features
- `team_form_points`, `team_recent_wins` - team performance
- `strength_*` - team strength ratings for attack/defense home/away

## Machine Learning Preparation

### Target Variables
The system can prepare datasets for multiple prediction tasks:
- **Points Prediction**: `points_scored` - primary target for team selection
- **Minutes Prediction**: `minutes_played` - for rotation risk assessment
- **Goals Prediction**: `goals_scored` - for captain selection
- **Assists Prediction**: `assists` - for creativity analysis

### Data Splitting
For time series data like FPL, use chronological splits:
```python
# Split by gameweek or season
train_data = dataset[dataset['gameweek'] <= 30]
test_data = dataset[dataset['gameweek'] > 30]
```

### Feature Selection
Use the feature importance analysis to select the most relevant features:
```python
# Load feature importance
importance_df = pd.read_csv('data/features/feature_importance_points_scored.csv')
top_features = importance_df.head(20)['feature'].tolist()
```

## Data Quality Validation

The system includes comprehensive data quality checks:

### Missing Values
- Identifies and reports missing values by column
- Provides strategies for handling missing data

### Outliers
- Detects outliers using IQR method
- Reports outlier counts for key numerical features

### Correlations
- Analyzes feature correlations with target variables
- Identifies highly correlated features for potential removal

### Data Types
- Validates appropriate data types for each column
- Ensures categorical variables are properly encoded

## Performance Considerations

### Memory Usage
- Large datasets may require chunked processing
- Consider using categorical data types for string columns
- Remove unnecessary columns after feature creation

### Processing Time
- Advanced feature engineering can be time-intensive
- Consider processing seasons separately for large datasets
- Use parallel processing where possible

### Storage
- Save intermediate results to avoid reprocessing
- Compress large datasets using efficient formats (parquet, feather)

## Troubleshooting

### Common Issues

1. **Missing Data Files**
   - Ensure historical data exists in correct directory structure
   - Check file formats and naming conventions

2. **Memory Errors**
   - Process smaller chunks of data
   - Reduce feature complexity or remove unnecessary columns

3. **Feature Engineering Failures**
   - Check for missing required columns
   - Validate data types before processing

4. **Encoding Errors**
   - Ensure consistent categorical values across seasons
   - Handle new categories in test data

### Debug Mode
Add debug prints and validation checks:
```python
# Enable debug mode
import logging
logging.basicConfig(level=logging.DEBUG)

# Validate data at each step
assert not df.empty, "DataFrame is empty"
assert 'points_scored' in df.columns, "Missing target column"
```

## Next Steps

After data processing, you can:
1. Train machine learning models for points prediction
2. Develop team optimization algorithms
3. Create performance dashboards and visualizations
4. Build automated team selection systems

For model training examples, see the machine learning documentation in the `models/` directory.
