# FPL Chip Functionality

This document describes the chip functionality implemented in the FPL bot, which allows the bot to intelligently decide when to use FPL chips and apply their effects to team selection and scoring.

## Overview

FPL chips are special abilities that managers can use to gain advantages:
- **Wildcard**: Complete team rebuild (2 per season, resets at GW19)
- **Free Hit**: One-week team optimization (2 per season, resets at GW19)  
- **Triple Captain**: Captain gets 3x points instead of 2x (2 per season, resets at GW19)
- **Bench Boost**: All 15 players score points (2 per season, resets at GW19)

## Implementation

### Core Components

1. **FPLChipManager** (`fpl_bot/utils/chip_manager.py`)
   - Manages chip state and availability
   - Determines when chips should be used
   - Records chip usage history

2. **Enhanced FPLTeamOptimizer** (`fpl_bot/utils/team_optimizer.py`)
   - New `optimize_team_with_chips()` method
   - Integrates chip decisions into team optimization
   - Applies chip effects to scoring

3. **Updated Prediction Pipeline**
   - `predict_team.py` now supports chip-enabled predictions
   - `iterative_season_manager.py` includes chip logic in gameweek predictions

### Chip Decision Logic

#### Wildcard
- **Timing**: Gameweeks 4-15 (early season optimization)
- **Trigger**: Team value 15% below optimal
- **Effect**: Allows unlimited transfers for complete team rebuild

#### Free Hit
- **Timing**: Blank gameweeks or double gameweeks
- **Trigger**: 5+ players from blank teams OR 3+ teams with double gameweeks
- **Effect**: One-week team optimization with unlimited transfers

#### Triple Captain
- **Timing**: When captain has excellent form/fixture
- **Trigger**: Captain predicted 8+ points OR double gameweek captain with 6+ points
- **Effect**: Captain gets 3x points instead of 2x

#### Bench Boost
- **Timing**: When bench players have strong fixtures
- **Trigger**: Bench predicted 15+ points OR 2+ bench teams with double gameweeks
- **Effect**: All 15 players score points

## Usage

### Basic Usage

```python
from fpl_bot.predict_team import predict_team_for_gameweek

# Make prediction with chip consideration
result = predict_team_for_gameweek(
    gameweek=5,
    budget=100.0,
    target='points_scored',
    data_dir='data'
)

# Check if a chip was used
if result.get('chip_used'):
    print(f"Chip used: {result['chip_used']}")
    print(f"Reason: {result['chip_config']['reason']}")
```

### Advanced Usage

```python
from fpl_bot.utils.chip_manager import FPLChipManager
from fpl_bot.utils.team_optimizer import FPLTeamOptimizer

# Initialize components
chip_manager = FPLChipManager("data")
optimizer = FPLTeamOptimizer(total_budget=100.0, data_dir="data")

# Check available chips
available_chips = chip_manager.get_available_chips(gameweek=5)
print(f"Available chips: {available_chips}")

# Get chip usage summary
summary = chip_manager.get_chip_usage_summary()
print(f"Chips used: {summary['chips_used']}")
```

## Configuration

### Chip State File

The chip manager maintains state in `chip_state.json`:

```json
{
  "chips_used": {
    "wildcard": 0,
    "free_hit": 0,
    "triple_captain": 0,
    "bench_boost": 0
  },
  "last_reset_gameweek": 0,
  "chip_usage_history": []
}
```

### Customizing Chip Logic

You can modify the chip decision thresholds in `FPLChipManager`:

```python
# Wildcard threshold (team value ratio)
if value_ratio < 0.85:  # 15% below optimal

# Triple Captain threshold
if captain_points >= 8.0:  # 8+ predicted points

# Bench Boost threshold  
if total_bench_points >= 15.0:  # 15+ bench points
```

## Integration Points

### Team Optimization

The chip functionality integrates with the existing team optimization pipeline:

1. **Normal Optimization**: Uses standard team selection
2. **Chip-Enabled Optimization**: Considers chip usage before team selection
3. **Chip Application**: Modifies team selection or scoring based on chip used

### Prediction Results

Chip information is included in prediction results:

```python
{
  "gameweek": 5,
  "total_predicted_points": 85.2,
  "chip_used": "triple_captain",
  "chip_config": {
    "reason": "high_captain_prediction",
    "captain_name": "Salah",
    "predicted_points": 8.5,
    "multiplier": 3
  },
  # ... other prediction data
}
```

## Testing

Run the test script to verify chip functionality:

```bash
python test_chips.py
```

This will test:
- Chip manager functionality
- Team optimizer with chips
- Full prediction pipeline with chips

## Monitoring

### Chip Usage Tracking

The system tracks chip usage throughout the season:

- **Current Usage**: Number of times each chip has been used
- **Usage History**: Detailed log of when and why chips were used
- **Automatic Reset**: Chips reset at gameweek 19

### Performance Impact

Chip usage affects predicted points:

- **Triple Captain**: +1x captain points (3x total instead of 2x)
- **Bench Boost**: +bench points (all 15 players score)
- **Wildcard/Free Hit**: Better team selection (unlimited transfers)

## Future Enhancements

Potential improvements to the chip system:

1. **Machine Learning**: Use ML to optimize chip timing
2. **Fixture Analysis**: Better blank/double gameweek detection
3. **Risk Assessment**: Consider variance in chip usage decisions
4. **Historical Analysis**: Learn from past chip usage patterns
5. **User Preferences**: Allow manual chip usage preferences

## Troubleshooting

### Common Issues

1. **No Chips Available**: Check if chips have been used up or if it's too early/late in season
2. **Chip Not Triggered**: Verify thresholds and fixture data availability
3. **State File Issues**: Delete `chip_state.json` to reset chip state

### Debug Information

Enable debug logging to see chip decision process:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will show detailed information about chip decision-making and team optimization.
