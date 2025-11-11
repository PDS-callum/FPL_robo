# Temporal Data Access - No Data Leakage Guarantee

## Overview

The FPL Competition Platform implements **strict temporal filtering** to ensure your bot only has access to data that would have been available at the time of each prediction. This prevents "look-ahead bias" or "data leakage" and makes backtests realistic.

## ✅ What Your Bot Can See

### When Predicting Gameweek N:

Your bot receives data through the `optimize(data)` method:

```python
data = {
    'current_team': [...],         # Your 15 players
    'available_players': [...],    # ALL FPL players (see below!)
    'gameweek': N,                 # Current gameweek number
    'bank': 0.5,                   # Money in bank
    'free_transfers': 1            # Free transfers available
}
```

### `current_team` - Your Squad (15 players)

Each player includes:
- `id`, `name`, `position`, `cost`, `team`
- `points_so_far`: Total points from GW1 to GW(N-1) **ONLY**
- `games_played`: Games played from GW1 to GW(N-1)
- `predicted_points`: Average points over last 5 completed games (up to GW(N-1))

### `available_players` - All FPL Players

**Critical:** This list contains ALL 600+ FPL players, not just your squad!

Each player includes the same fields as above, PLUS:
- `in_squad`: Boolean indicating if this player is in your current squad

**All stats are filtered to only include data up to GW(N-1)!**

## 🔒 Data Leakage Prevention

### Example: Predicting Gameweek 5

```
╔══════════════════════════════════════════════════════════╗
║  When your bot predicts GW5...                           ║
╠══════════════════════════════════════════════════════════╣
║  ✅ CAN SEE:  GW1, GW2, GW3, GW4 results                ║
║  ❌ CANNOT SEE: GW5, GW6, GW7... results                ║
╚══════════════════════════════════════════════════════════╝
```

**For each player:**
- `points_so_far` = Sum of points from GW1-4 **ONLY**
- `predicted_points` = Average of last 5 games (max: GW1-4)
- No access to GW5+ performance data

### Code Implementation

The temporal filtering happens here:

```python
# In bot_execution_service.py, line 313-361
for past_gw in range(start_gw, gw):  # STOPS BEFORE CURRENT GW!
    if past_gw in player_histories[pid]:
        gw_stats = player_histories[pid][past_gw]
        total_points_so_far += gw_stats['total_points']
```

**Key:** `range(start_gw, gw)` goes from 1 to (gw-1), never including current gw!

## 📊 Real-World Example

### Haaland in GW5 Backtest

**What your bot sees when predicting GW5:**

```python
{
    'id': 354,
    'name': 'Haaland',
    'position': 'FWD',
    'cost': 15.0,
    'team': 'MCI',
    'points_so_far': 52,        # Total from GW1-4 only
    'games_played': 4,           # Played GW1-4
    'predicted_points': 13.0,    # Average: (13+8+15+16)/4 = 13.0
    'in_squad': True
}
```

**What your bot CANNOT see:**
- ❌ His GW5 actual performance (unknown at prediction time)
- ❌ His final season total (would be ~250 points)
- ❌ Whether he'll get injured in GW8
- ❌ Any "future knowledge"

## 🎯 Why This Matters

### Without Temporal Filtering (Data Leakage):
```python
# Bot sees Haaland will score 250 points total
# Easy to just pick him! Not a real test.
predicted_points = 250  # Cheating!
```

### With Temporal Filtering (Realistic):
```python
# Bot only knows Haaland scored 52 in GW1-4
# Must use its own logic to predict GW5+
predicted_points = calculate_from_past_data()  # Fair test!
```

## 🛡️ Security Guarantees

### Bot Sandbox Restrictions:

1. **No File System Access**: Can't read external data files
2. **No Network Access**: Can't fetch live API data during prediction
3. **No Imports**: Restricted to safe built-ins only (no `os`, `sys`, `requests`, etc.)
4. **No Global State**: Each GW prediction runs independently

### Data Access Restrictions:

1. **Temporal Filtering**: Only past gameweek data (enforced at source)
2. **No Raw Histories**: Bot doesn't receive full season data dumps
3. **Pre-Computed Stats**: All stats calculated with GW(N-1) cutoff

## 📖 How to Use This in Your Bot

### Example: Smart Transfer Logic

```python
def optimize(self, data):
    available_players = data['available_players']
    current_team = data['current_team']
    gameweek = data['gameweek']
    
    # Find best-performing players based on PAST data only
    top_performers = sorted(
        available_players,
        key=lambda p: p['points_so_far'] / max(p['games_played'], 1),
        reverse=True
    )
    
    # Filter by form (last 5 games) - also past data only!
    in_form = [p for p in available_players 
               if p['predicted_points'] > 7.0]  # 7+ avg last 5 games
    
    # Your bot uses past data to predict future performance
    # This is a REAL test of your prediction algorithm!
    ...
```

## ✅ Verification Checklist

To verify no data leakage in your bot:

1. ☑ Only use `points_so_far` (not any "total_season_points")
2. ☑ Only use `predicted_points` (calculated from past games)
3. ☑ Never assume future knowledge (injuries, transfers, form changes)
4. ☑ Test locally with mock data that simulates progressive disclosure

## 🔬 Testing Your Bot

The platform's backtest runs like this:

```
GW1: Bot predicts with NO prior data → records result
GW2: Bot predicts with GW1 data → records result
GW3: Bot predicts with GW1-2 data → records result
...
GW11: Bot predicts with GW1-10 data → records result
```

Each prediction is independent and fair!

## 🎓 Summary

| What | Access Level | Notes |
|------|-------------|-------|
| **Past gameweeks (1 to N-1)** | ✅ Full access | Fair - would be available in real FPL |
| **Current gameweek (N)** | ❌ No access | Prevented - this is what you're predicting! |
| **Future gameweeks (N+1 onward)** | ❌ No access | Prevented - unknown at prediction time |
| **Player pool** | ✅ All 600+ players | But stats limited to past GWs only |
| **Static data (positions, teams)** | ✅ Full access | Fair - publicly known |
| **Live prices** | ✅ Current prices | Fair - would be visible in FPL |

---

**Your bot is tested fairly. May the best algorithm win!** 🏆

