# Data Security Audit: Bot Access Control

## Current Approach (What We're Fixing)

### Problem: Trust-Based System

Currently, the platform:
1. ✅ Computes temporal statistics correctly
2. ✅ Passes filtered data to bot
3. ❌ **But bot runs in same process with access to Python internals**

**Potential vulnerabilities:**
- Bot could theoretically access global variables
- Bot could inspect stack frames
- Bot could use `__import__` workarounds
- We're trusting the sandbox restrictions

## New Approach: Platform-Controlled Data (Recommended)

### Principle: "Don't Trust, Verify"

The platform should be the **ONLY entity** that:
1. Fetches player histories
2. Applies temporal filtering
3. Computes all statistics
4. Passes **ONLY** the final, safe data to the bot

**The bot should receive:**
- ✅ Pre-computed statistics only
- ✅ No access to raw history data
- ✅ No access to filtering logic
- ✅ No way to bypass temporal restrictions

## Current Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Platform (bot_execution_service.py)                         │
├─────────────────────────────────────────────────────────────┤
│ 1. Fetch player_histories (all GWs) ← RAW DATA              │
│ 2. Loop through gameweeks:                                  │
│    For GW N:                                                │
│      - Filter histories to GW1-(N-1) ← TEMPORAL FILTER      │
│      - Compute points_so_far                                │
│      - Compute predicted_points (form)                      │
│      - Create available_players_list                        │
│      - Pass to bot: {                                       │
│          'current_team': [...],                             │
│          'available_players': [...],  ← FILTERED DATA       │
│          'gameweek': N,                                     │
│          'bank': X,                                         │
│          'free_transfers': Y                                │
│        }                                                    │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ User Bot (advanced_ml_mip_bot.py)                          │
├─────────────────────────────────────────────────────────────┤
│ def optimize(self, team_data):                             │
│   current_team = team_data['current_team']                 │
│   available_players = team_data['available_players']       │
│                                                             │
│   # Bot receives ONLY pre-computed stats:                  │
│   # - points_so_far (GW1 to N-1 only)                      │
│   # - predicted_points (last 5 games avg)                  │
│   # - games_played, position, cost, etc.                   │
│                                                             │
│   # ✅ Bot cannot access player_histories                   │
│   # ✅ Bot cannot access raw gameweek data                  │
│   # ✅ Bot must trust platform's calculations               │
└─────────────────────────────────────────────────────────────┘
```

## What Bot Currently Receives (Per Player)

```python
# For each player in current_team and available_players:
{
    'id': 354,
    'name': 'Haaland',
    'full_name': 'Erling Haaland',
    'position': 'FWD',
    'cost': 15.0,
    'team': 'MCI',
    
    # TEMPORAL STATISTICS (Platform-computed, GW1 to N-1 only):
    'points_so_far': 52,        # ✅ Platform computes from histories
    'games_played': 4,           # ✅ Platform computes
    'predicted_points': 13.0,    # ✅ Platform computes (form)
    'in_squad': False            # ✅ Platform provides
}
```

**What bot does NOT receive:**
- ❌ `player_histories` dictionary
- ❌ Raw gameweek-by-gameweek data
- ❌ Access to `all_players` full data
- ❌ Future gameweek data

## Verification: Is Current Implementation Secure?

### ✅ Good Security Measures:

1. **Sandbox Namespace:**
   ```python
   safe_builtins = {
       'print': print, 'len': len, 'range': range, ...
       # No __import__, no eval, no exec
   }
   namespace = {'__builtins__': safe_builtins}
   exec(bot_code, namespace)
   ```

2. **Restricted Imports:**
   ```python
   RESTRICTED_IMPORTS = ['os', 'sys', 'subprocess', 'eval', 'exec', 
                         'compile', '__import__', 'open', 'file']
   ```

3. **Data Passed as Method Arguments:**
   ```python
   bot_instance.optimize(team_data)  # Only this data accessible
   ```

4. **No Global State:**
   - Each gameweek prediction is independent
   - Bot cannot store data between gameweeks (new call each time)

### ⚠️ Potential Vulnerabilities:

1. **Stack Frame Inspection (Theoretical):**
   ```python
   # Bot could theoretically try:
   import sys
   frame = sys._getframe(1)  # Access caller's frame
   player_histories = frame.f_locals['player_histories']  # ❌ Get raw data!
   ```
   **Mitigation:** `sys` is in restricted imports ✅

2. **Namespace Pollution:**
   ```python
   # If bot defines global variables:
   global_data = []
   
   def optimize(self, team_data):
       global_data.append(team_data)  # Store data
   ```
   **Impact:** Limited - can only store what's passed ✅

3. **Object Introspection:**
   ```python
   # Bot could try to inspect objects:
   def optimize(self, team_data):
       # Access hidden attributes?
       print(dir(team_data))
       print(team_data.__dict__)
   ```
   **Impact:** Can only see what's in team_data (already filtered) ✅

## Enhanced Security Recommendations

### 1. Deep Copy Data Before Passing to Bot

```python
import copy

# Before calling bot:
safe_team_data = copy.deepcopy(mock_team_data)
optimization = bot_instance.optimize(safe_team_data)

# This prevents bot from modifying original data structures
```

### 2. Freeze Data Structures (Read-Only)

```python
from types import MappingProxyType

# Make dictionaries read-only:
safe_team_data = {
    'current_team': tuple(current_squad),  # Tuple instead of list
    'available_players': tuple(available_players_list),
    'gameweek': gw,
    'bank': bank,
    'free_transfers': free_transfers
}
```

### 3. Data Schema Validation

```python
# Define exactly what bot should receive:
ALLOWED_PLAYER_FIELDS = {
    'id', 'name', 'position', 'cost', 'team',
    'points_so_far', 'games_played', 'predicted_points', 'in_squad'
}

def sanitize_player_data(player):
    """Only pass whitelisted fields to bot"""
    return {k: v for k, v in player.items() if k in ALLOWED_PLAYER_FIELDS}

available_players_list = [
    sanitize_player_data(player) for player in filtered_players
]
```

### 4. Compute ALL Statistics in Platform

```python
# Platform should compute everything the bot might need:
for player in available_players_list:
    # Basic stats (already doing this):
    player['points_so_far'] = ...
    player['predicted_points'] = ...  # form
    player['games_played'] = ...
    
    # Additional stats bot might need:
    player['goals_so_far'] = sum(gw['goals'] for gw in past_gws)
    player['assists_so_far'] = sum(gw['assists'] for gw in past_gws)
    player['clean_sheets_so_far'] = sum(gw['clean_sheets'] for gw in past_gws)
    player['minutes_so_far'] = sum(gw['minutes'] for gw in past_gws)
    player['bonus_so_far'] = sum(gw['bonus'] for gw in past_gws)
    player['form_last_3'] = avg(last_3_gws)
    player['form_last_5'] = avg(last_5_gws)
    
    # Value metrics:
    player['points_per_million'] = points_so_far / cost
    player['points_per_game'] = points_so_far / games_played
```

## Recommended Implementation

### New Data Structure:

```python
team_data = {
    'gameweek': 5,
    'bank': 0.5,
    'free_transfers': 1,
    
    'current_team': [
        {
            # Identity
            'id': 354,
            'name': 'Haaland',
            'position': 'FWD',
            'cost': 15.0,
            'team': 'MCI',
            
            # Cumulative stats (GW1-4 only for GW5 prediction)
            'points_so_far': 52,
            'games_played': 4,
            'minutes_so_far': 360,
            'goals_so_far': 8,
            'assists_so_far': 1,
            'bonus_so_far': 7,
            'clean_sheets_so_far': 1,
            
            # Form indicators (computed by platform)
            'form_last_3': 14.3,   # Average points last 3 games
            'form_last_5': 13.0,   # Average points last 5 games
            'form_trend': 'up',    # 'up', 'down', 'stable'
            
            # Value metrics
            'points_per_million': 3.47,  # 52 / 15.0
            'points_per_game': 13.0,     # 52 / 4
            
            # Context (read-only)
            'in_squad': True
        },
        # ... 14 more players
    ],
    
    'available_players': [
        # Same structure, 300 players
        # All with proper temporal filtering
    ]
}
```

### Benefits:

1. **Single Source of Truth:** Platform controls all data
2. **No Raw Access:** Bot never sees player_histories
3. **Rich Statistics:** Bot has all it needs for intelligent decisions
4. **Temporal Safety:** Platform guarantees no future data
5. **Auditable:** Easy to verify what bot receives

## Current Status: Already Pretty Secure! ✅

**Our current implementation is actually quite good:**

✅ Platform controls data fetching  
✅ Platform applies temporal filtering  
✅ Platform computes statistics  
✅ Bot receives only filtered data  
✅ Sandbox prevents dangerous operations  
✅ No way to access player_histories from bot  

**The data leakage we found was in the PLATFORM code** (using `total_points` for starting squad), **NOT a bot security issue!**

## Conclusion

**Current security is STRONG**, but we can enhance it with:

1. ✅ **Fix completed:** Remove `total_points` usage in starting squad selection
2. ✅ **Fix completed:** Remove `total_points` usage in top 300 selection
3. 🔄 **Optional enhancement:** Deep copy data before passing to bot
4. 🔄 **Optional enhancement:** Add more pre-computed statistics
5. 🔄 **Optional enhancement:** Whitelist allowed fields

**The platform IS the gatekeeper - bots only see what we give them!** 🔒


