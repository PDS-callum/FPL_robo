# FPL Bot

A Fantasy Premier League prediction bot using CNN modeling.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fpl_bot.git
cd fpl_bot

# Install in development mode
pip install -e .
```

## Usage

```bash
# Collect FPL data
fpl-bot collect

# Process data
fpl-bot process

# Train model
fpl-bot train --epochs 50 --batch_size 32

# Predict team
fpl-bot predict --gameweek 10 --budget 100.0

# Run all steps
fpl-bot all --gameweek 10 --budget 100.0 --epochs 50
```

## Features

- Data collection from FPL API
- Data preprocessing and feature engineering
- CNN model for player points prediction
- Team optimization with budget and formation constraints
- Weekly team selection with captain recommendations

## Project Structure
