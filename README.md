# FPL Robo

A Fantasy Premier League prediction bot using CNN modeling.

## Installation

```bash
# Clone the repository
git clone https://github.com/PDS-callum/FPL_robo.git
cd fpl_robo

# Install in development mode
pip install -e .
```

## Usage

```bash
# Collect FPL data from GitHub archive
fpl-bot collect --seasons 2022-23 2023-24

# Process data with enhanced features
fpl-bot process --all --lookback 3

# Train model with historical data
fpl-bot train-with-history --all --epochs 50 --batch_size 32

# Predict team for new season
fpl-bot predict --gameweek 1 --use-history --next-season --teams "Arsenal" "Man City" "Liverpool" "Chelsea" "Spurs" "Man Utd" "Newcastle" "Aston Villa" "West Ham" "Brighton" "Brentford" "Crystal Palace" "Fulham" "Wolves" "Bournemouth" "Everton" "Nott'm Forest" "Burnley" "Sheffield Utd" "Ipswich"
```

## Features

- Data collection from FPL GitHub history archive
- Enhanced data preprocessing and feature engineering
- Multi-season training data preparation
- CNN model for player points prediction
- Team optimization with budget and formation constraints
- Weekly team selection with captain recommendations

## Processing Options

Process data for training the model with enhanced features:

```bash
# Process all available historical seasons
fpl-bot process --all

# Process specific seasons
fpl-bot process --seasons 2021-22 2022-23 2023-24

# Change lookback period for sequence generation
fpl-bot process --lookback 5
```

## Project Structure
