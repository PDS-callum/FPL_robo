# How Clean Sheet Probability is Calculated

The bot calculates clean sheet probability for each team matchup using a multi-factor model that considers team defensive performance, opponent attacking strength, home/away advantage, and other factors.

## Step-by-Step Calculation Process

### Step 1: Build Team Statistics (`_build_team_stats`)

For each team, the bot calculates:

1. **Clean Sheet Rate**: 
   - Uses goalkeeper stats (most reliable)
   - `clean_sheet_rate = goalkeeper_clean_sheets / games_played`
   - Example: If Arsenal's GK has 6 clean sheets in 11 games → 6/11 = 54.5%
   - Capped between 10% and 60%

2. **Average Goals Conceded**:
   - `avg_goals_conceded = goalkeeper_goals_conceded / games_played`
   - Example: If Arsenal's GK conceded 12 goals in 11 games → 12/11 = 1.09 goals/game

3. **Average Goals Scored** (for opponent analysis):
   - `avg_goals_scored = total_team_goals / games_played`
   - Calculated separately for home and away

### Step 2: Calculate Clean Sheet Probability (`_predict_clean_sheet_probability`)

For a specific matchup (e.g., Arsenal vs Tottenham at home), the bot combines multiple factors:

#### Factor 1: Base Clean Sheet Rate
```
base_prob = team_cs_rate
```
- Example: Arsenal's base CS rate = 0.60 (60%)

#### Factor 2: Opponent Attack Strength
```
attack_ratio = opponent_avg_goals_scored / league_average_goals
```

Then applies penalties based on opponent strength:

- **Weak opponents** (attack_ratio < 1.0):
  - Small boost: `opponent_factor = (1.0 - attack_ratio) * 0.08`
  - Example: If opponent scores 0.9x average → +0.8% boost

- **Above-average opponents** (1.0 ≤ attack_ratio < 1.3):
  - Penalty: `opponent_factor = min(-0.10, -(attack_ratio - 1.0) * 0.30)`
  - Minimum -10% penalty
  - Example: Tottenham (1.05x average) → -1.5% penalty, but minimum -10%

- **Elite opponents** (attack_ratio ≥ 1.3):
  - Major penalty: `opponent_factor = -0.25 - (excess_attack * 0.25)`
  - Example: Man City (1.8x average) → -0.25 - (0.5 * 0.25) = -37.5%

#### Factor 3: Home/Away Advantage
```
if is_home:
    home_factor = 0.10  # +10% boost at home
    # Additional boost based on home defensive record
else:
    home_factor = -0.08  # -8% penalty away
    # Additional penalty based on away defensive record
```

#### Factor 4: Defensive Strength
```
defence_strength_factor = (league_avg_goals - team_avg_goals_conceded) * 0.15
```
- Teams that concede fewer goals get a boost
- Example: If Arsenal concedes 1.0 goals/game vs league avg 1.4 → +6% boost

#### Factor 5: Recent Form (currently neutral)
```
form_impact = (recent_form_factor - 1.0) * 0.05
```
- Currently set to 0 (neutral) but can be enhanced with recent form analysis

### Step 3: Combine All Factors
```
probability = base_prob + opponent_factor + home_factor + defence_strength_factor + form_impact
```

### Step 4: Apply Realistic Caps

The final probability is capped based on opponent strength:

- **Elite opponents** (attack_ratio ≥ 1.3): **Max 20%**
- **Strong opponents** (1.1 ≤ attack_ratio < 1.3): **Max 40%**
- **Above-average opponents** (1.0 ≤ attack_ratio < 1.1): **Max 50%**
- **Weak opponents** (attack_ratio < 1.0): **Max 70%**

Final bounds: **5% minimum, 70% maximum**

## Example: Arsenal vs Tottenham (Home)

Let's trace through the calculation:

1. **Base CS Rate**: Arsenal = 0.60 (60%)

2. **Opponent Factor**: 
   - Tottenham attack = 1.47 goals/game
   - League average = 1.4 goals/game
   - Attack ratio = 1.47 / 1.4 = 1.05
   - Since 1.0 ≤ 1.05 < 1.3: `opponent_factor = min(-0.10, -(1.05 - 1.0) * 0.30) = -0.10` (-10%)

3. **Home Factor**: `home_factor = 0.10` (+10%)

4. **Defensive Strength**: 
   - Arsenal concedes 1.09 goals/game
   - League average = 1.4 goals/game
   - `defence_strength_factor = (1.4 - 1.09) * 0.15 = +0.0465` (+4.65%)

5. **Form Impact**: `form_impact = 0` (neutral)

6. **Combined**: 
   ```
   probability = 0.60 - 0.10 + 0.10 + 0.0465 + 0
                = 0.6465 (64.65%)
   ```

7. **Apply Cap**: 
   - Attack ratio = 1.05 (above-average)
   - Cap at 50%: `probability = min(0.50, 0.6465) = 0.50` (50%)

**Final Result: 50% clean sheet probability**

## Why the Fix Was Needed

Before the fix:
- Arsenal vs Tottenham was getting **70% CS probability** (too high!)
- This was because:
  1. The opponent penalty was too weak for teams just above average (Tottenham at 1.05x)
  2. There was no cap for above-average teams
  3. The home advantage (+10%) was pushing it too high

After the fix:
- Arsenal vs Tottenham now gets **50% CS probability** (more realistic)
- The fix:
  1. Added minimum -10% penalty for any team above average
  2. Added 50% cap for above-average opponents (1.0-1.1 attack ratio)
  3. This prevents unrealistic CS rates in derby matches and top-6 fixtures

## Data Sources

The bot gets team statistics from:
- **Goalkeeper stats**: `goals_conceded`, `clean_sheets`, `starts` from FPL API
- **Team goals**: Sum of all players' `goals_scored` from FPL API
- **Games played**: Based on goalkeeper starts (most reliable indicator)

All stats are calculated from actual FPL data up to the current gameweek.

