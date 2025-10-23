# FPL Bot v2.0 - Strategic Fantasy Premier League Manager

A comprehensive Fantasy Premier League bot with **7-gameweek strategic planning**, **autonomous transfer execution**, **intelligent chip optimization**, and a beautiful **web dashboard**.

---

## Features

### Core Capabilities
- **🌐 Web Dashboard** - Beautiful, modern UI for viewing analysis (opens automatically in your browser)
- **7-Gameweek Strategic Planning** - Multi-period optimization with team evolution tracking
- **Autonomous Transfer Execution** - Automatically makes transfers on your FPL account
- **Intelligent Chip Timing** - Optimizes Triple Captain & Bench Boost based on future team composition
- **Fixture Run Analysis** - Identifies favorable fixture periods for quality teams
- **Transfer Banking Strategy** - Plans accumulation of up to 5 free transfers
- **Quality-Based Recommendations** - Prioritizes elite teams over mid-table fixture chasing

### Technical Features
- Multi-period team optimization
- Fixture difficulty calculation (60% position, 30% strength, 10% home/away)
- Player point projections adjusted for specific fixtures
- Chip usage constraints (one chip per GW)
- Transfer cost optimization
- Budget management across multiple weeks

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/FPL_robo.git
cd FPL_robo

# Install dependencies
pip install -e .

# Optional: For advanced MIP optimization (future)
pip install pulp
```

### Basic Usage

```bash
# Standard analysis with web dashboard (opens browser automatically)
python -m fpl_bot.main YOUR_MANAGER_ID

# Example
python -m fpl_bot.main 789800
```

**What happens**:
1. Bot starts web UI server on http://127.0.0.1:5000
2. Browser opens automatically to show dashboard
3. Terminal shows debug info only
4. Dashboard refreshes automatically with new data

**Terminal-only mode** (legacy):
```bash
python -m fpl_bot.main 789800 --no-ui --summary-only
```

### Autonomous Mode (Auto-Execute Transfers)

```powershell
# Set credentials
$env:FPL_EMAIL = "your@email.com"
$env:FPL_PASSWORD = "yourpassword"

# Run with auto-execute
python -m fpl_bot.main 789800 --auto-execute --summary-only --no-save
```

---

## Output Example

```
============================================================
FPL BOT ANALYSIS SUMMARY
============================================================

Manager: Bleep Blorp
Team: ctrl + alt + de-feet
Overall Rank: 4002964
Total Points: 374
Free Transfers Available: 0

CHIP DECISION: NO_CHIP (Confidence: 95%)
Reason: Save Triple Captain for GW12 (better opportunity)

ACTION PLAN:
1. Make 1 transfer(s) for 5.0 net points gain (OUT: Milenkovi, IN: Gabriel)
2. Captain Semenyo (8.5 predicted points)
3. Do not use any chip this gameweek

======================================================================
7-GAMEWEEK STRATEGIC PLAN (GW7-13)
======================================================================

WEEK-BY-WEEK PLAN:

>>> GAMEWEEK 7  (THIS WEEK)
>>> Transfers: None
>>> Expected Points: 56.0

    GAMEWEEK 8
    Transfers: 1 (OUT: Iwobi -> IN: Rice, +4.9 pts)
    Free Transfers Available: 1
    Expected Points: 60.0

    GAMEWEEK 9
    Transfers: 1 (OUT: J.Murphy -> IN: Bruno G., +4.0 pts)
    Free Transfers Available: 1
    Expected Points: 61.7

    GAMEWEEK 12
    Transfers: 1 (OUT: Wood -> IN: Haaland, +5.5 pts)
    Chip: TRIPLE CAPTAIN on Semenyo (2x9.0 = 18.0 pts)
    Expected Points: 77.1

CHIP TIMING RECOMMENDATIONS:

TRIPLE CAPTAIN: GW12 [RECOMMENDED]
  Captain: Semenyo
  Expected Points: 9.0 pts (2x = 18.0 pts total)
  Alternatives:
    - Gabriel: 6.1 pts
    - Rice: 5.6 pts

BENCH BOOST: GW13 [RECOMMENDED]
  Bench Total: 13.1 pts
  Bench Players:
    - Mbeumo (MID): 2.5 pts
    - Sels (GK): 2.2 pts
```

---

## Command Line Options

| Option | Description |
|--------|-------------|
| `manager_id` | Your FPL Manager ID (required) |
| `--no-ui` | Disable web UI (terminal only) |
| `--summary-only` | Show summary output (use with `--no-ui`) |
| `--no-save` | Don't save analysis results to file |
| `--port PORT` | Web UI port (default: 5000) |
| `--auto-execute` | Automatically execute recommended transfers |
| `--email EMAIL` | FPL account email (or use `FPL_EMAIL` env var) |
| `--password PASS` | FPL account password (or use `FPL_PASSWORD` env var) |
| `--no-ft-gain-last-gw` | Flag that wildcard/free hit was played last week (no FT gained) |
| `--no-hits` | Prevent optimizer from recommending point hits (only use free transfers) |
| `--budget-correction VALUE` | Adjust budget by amount in millions (e.g., -2.5) |
| `--risk VALUE` | Risk aversion: 0=aggressive, 1=conservative (default: 0.5) |
| `--min-playing-chance PCT` | Minimum % chance of playing to consider player (default: 75) |
| `--analyze-wildcard` | Analyze optimal wildcard timing (slower, ~5-10 min) |
| `--help` | Show all available options |

---

## Features Deep Dive

### 1. 7-Gameweek Strategic Planning

The bot automatically plans 7 gameweeks ahead, showing:
- Week-by-week transfer recommendations
- Free transfer accumulation (up to 5)
- Expected points for each week
- Team composition evolution
- Chip usage timing

**Why 7 weeks?** 
- Enough to identify fixture runs
- Plan chip usage strategically
- Accumulate free transfers for big moves
- See team evolution clearly

### 2. Intelligent Chip Optimization

Chips are analyzed based on **your actual planned team** for each future gameweek, not just current team.

**Triple Captain**:
- Finds best gameweek for your best captain
- Shows which player to captain
- Accounts for fixture difficulty
- Provides alternative options

**Bench Boost**:
- Identifies week with strongest bench
- Shows which bench players will score
- Considers team evolution
- Optimizes based on future transfers

**Key Constraint**: Only one chip per gameweek (enforced)

### 3. Fixture Difficulty Ratings

Calculated using weighted formula:
- **60% League Position** - Most important factor
- **30% FPL Strength Ratings** - Form and stats
- **10% Home/Away Advantage** - Venue impact

**Ratings**:
- Very Easy: ≤ 2.0
- Easy: ≤ 2.8
- Medium: ≤ 3.5
- Hard: ≤ 4.2
- Very Hard: > 4.2

**Example**:
- Arsenal (1st) vs Burnley (18th) = 1.8 (Very Easy) ✅
- Bournemouth (4th) vs Man City (5th) = 4.0 (Hard) ✅

### 4. Quality-Based Fixture Run Analysis

Identifies teams with 3+ consecutive easy fixtures, but **filters by quality**:

**Premium Targets (Recommended)**: Top 6 teams only
- Arsenal, Liverpool, Man City, Spurs, Bournemouth, Chelsea

**Value Targets**: 7th-10th place teams if VERY easy fixtures

**Avoid**: 11th+ place teams
- Newcastle, Wolves, Brighton (even with easy fixtures)

**Why?** Elite teams score more even with harder fixtures than mid-table teams with easy fixtures.

### 5. Transfer Banking Strategy

The bot plans to accumulate up to **5 free transfers**:

**When to Bank**:
- No beneficial transfers available (gain < 1.5 pts)
- Saving for expensive target (e.g., Salah, Haaland)
- Building toward fixture runs

**When to Use**:
- Beneficial transfer available (gain > 1.5 pts)
- Have accumulated 2+ FTs for multiple moves
- Fixture run starting soon

**Example**:
```
GW7: 0 FTs → Bank
GW8: 1 FT → Bank  
GW9: 2 FTs → Bank
GW10: 3 FTs → Use 1, bank 2
GW11: 3 FTs → Use 1, bank 2
GW12: 3 FTs → Make 3 transfers (bring in premiums)
```

### 6. Autonomous Transfer Execution

The bot can automatically make transfers on your FPL account.

**Safety Features**:
- Only executes if no transfers made this week
- Won't take bad points hits (requires net gain > 10)
- Authenticates securely via FPL API
- Shows transfer summary before executing
- Clear success/failure reporting

**Usage**:
```powershell
# Set credentials (one time)
$env:FPL_EMAIL = "your@email.com"
$env:FPL_PASSWORD = "yourpassword"

# Run with auto-execute
python -m fpl_bot.main 789800 --auto-execute --summary-only --no-save
```

**What Happens**:
1. Logs into your FPL account
2. Analyzes your team
3. Finds optimal transfers
4. Checks safety conditions
5. Executes transfers automatically
6. Reports success/failure

---

## How Strategic Planning Works

### Multi-Period Optimization Process

1. **Player Point Projection** (7 weeks)
   - Base prediction for each player
   - Adjusted for specific fixture difficulty
   - Position-specific adjustments (DEF benefit more from easy fixtures)

2. **Team Evolution Planning**
   - Simulates team composition for each GW
   - Plans when to use accumulated FTs
   - Tracks budget across weeks
   - Applies transfers when beneficial

3. **Chip Timing Optimization**
   - Analyzes each chip for each future GW
   - Uses **actual planned team** for that GW
   - Finds optimal timing across 7 weeks
   - Ensures one chip per GW constraint

4. **Fixture Run Detection**
   - Scans all teams for 3+ consecutive easy fixtures
   - Filters for quality (top 10 teams only)
   - Ranks by: Team Position + Fixture Ease + Length
   - Recommends premium targets only

### Decision Thresholds

**Transfer in Future GWs**:
- 1 transfer if gain ≥ 1.5 pts
- 2 transfers if avg gain ≥ 3.0 pts per transfer
- 3+ transfers if avg gain ≥ 3.5 pts per transfer

**Chip Usage**:
- Triple Captain: Recommend if benefit ≥ 8.0 pts
- Bench Boost: Recommend if benefit ≥ 8.0 pts

**Fixture Runs**:
- Premium Target: Top 6 teams with avg difficulty ≤ 2.8
- Value Target: 7-10th teams with avg difficulty < 2.0
- Avoid: 11th+ teams (regardless of fixtures)

---

## Architecture

### Core Modules

#### `fpl_bot/core/data_collector.py`
- Fetches data from FPL API
- Manages authentication for autonomous mode
- Executes transfers via API
- Calculates free transfer availability

#### `fpl_bot/core/manager_analyzer.py`
- Analyzes current team composition
- Performance history tracking
- Transfer history analysis

#### `fpl_bot/core/predictor.py`
- Predicts player performance
- Captain option recommendations
- Form and fixture analysis

#### `fpl_bot/core/transfer_optimizer.py`
- Single-week transfer optimization
- Budget constraint handling
- Net points calculation

#### `fpl_bot/core/chip_manager.py`
- Chip status tracking
- Fixture difficulty calculation
- Dynamic planning horizon (to GW19 or +10)

#### `fpl_bot/core/multi_period_planner.py` ⭐ NEW
- 7-gameweek strategic planning
- Team evolution simulation
- Multi-period chip optimization
- Quality-based fixture run detection
- Transfer banking strategy

#### `fpl_bot/main.py`
- Main bot orchestration
- CLI interface
- Report generation
- Display formatting

---

## Usage Scenarios

### Scenario 1: Weekly Analysis (Web Dashboard)

```bash
# Get recommendations with beautiful web UI
python -m fpl_bot.main 789800
```

**You Get**:
- 🌐 Web dashboard in your browser
- Current week transfer recommendations
- Captain choice
- Chip decision (use now or save for later)
- 7-week strategic plan
- Fixture run opportunities
- Terminal shows debug info only

### Scenario 2: Autonomous Weekly Management

```bash
# Set credentials once
$env:FPL_EMAIL = "your@email.com"
$env:FPL_PASSWORD = "yourpassword"

# Run every week automatically
python -m fpl_bot.main 789800 --auto-execute --summary-only --no-save
```

**Bot Will**:
- Analyze your team
- Execute beneficial transfers automatically
- Show 7-week strategic plan
- Report chip timing recommendations

### Scenario 3: Scheduled Automation

**Windows Task Scheduler**:
Create `run_fpl_bot.bat`:
```batch
@echo off
cd C:\path\to\FPL_robo
python -m fpl_bot.main 789800 --auto-execute --summary-only >> bot_log.txt 2>&1
```

Schedule to run **every Friday at 6 PM** before deadline.

**Linux/Mac Cron**:
```bash
# Add to crontab
0 18 * * 5 cd /path/to/FPL_robo && python -m fpl_bot.main 789800 --auto-execute --summary-only >> bot_log.txt 2>&1
```

---

## Understanding the Output

### Section 1: Team Summary
- Manager info and rank
- Current team value and budget
- Free transfers available
- Transfers made this week

### Section 2: Current Week Decisions
- **Transfer Decision**: What to do this week
- **Chip Decision**: Use chip now or save for later
- **Captain Decision**: Who to captain
- **Action Plan**: Summary of immediate actions

### Section 3: 7-Gameweek Strategic Plan

#### Week-by-Week Breakdown
Shows for each of next 7 gameweeks:
- Proposed transfers (if any)
- Free transfer status
- Chip usage (if planned)
- Expected points

#### Fixture Run Opportunities
Teams with 3+ easy consecutive fixtures:
- **Premium Targets**: Top 6 teams (recommended)
- **Value Targets**: 7-10th teams (budget options)
- **Avoid**: 11+ teams (marked clearly)

#### Chip Timing Recommendations
- **When** to use each chip
- **Who** to use it on (for TC)
- **Which players** will score (for BB)
- **Expected benefit** in points

#### Strategic Recommendations
- Immediate actions for this week
- Chip strategy for next 7 weeks
- Fixture opportunities to exploit

---

## Strategic Planning Explained

### How It Works

The bot simulates 7 future gameweeks:

1. **Projects player points** for each GW based on fixtures
2. **Plans team evolution** - when to make transfers
3. **Tracks free transfers** - shows accumulation to 5
4. **Optimizes chip timing** - uses actual future team composition
5. **Identifies opportunities** - fixture runs from quality teams

### Example Strategic Scenario

**Current Situation**: GW7, 0 FTs, have Semenyo

**Bot's 7-Week Plan**:
```
GW7: No transfers (already used FT)
GW8: Transfer Iwobi -> Rice (use 1 FT, +4.9 pts)
GW9: Transfer Murphy -> Bruno G. (use 1 FT, +4.0 pts)
GW10: Bank FT
GW11: Bank FT (now have 2 FTs)
GW12: Transfer Wood -> Haaland, USE TRIPLE CAPTAIN on Semenyo
GW13: USE BENCH BOOST (13.1 pts from bench)
```

**Result**: Team improves from 56 pts → 77 pts by GW12!

### Why This is Better Than Manual Planning

**Problem**: Hard to plan 7 weeks ahead manually
- Too many variables to track
- Don't know which fixtures are actually easy
- Hard to optimize chip timing
- Easy to waste free transfers

**Solution**: Bot automatically:
- ✅ Tracks all 743 players across 7 gameweeks
- ✅ Calculates fixture-adjusted points
- ✅ Finds optimal chip timing
- ✅ Shows when to use accumulated FTs
- ✅ Filters for quality teams only

---

## Strategic Principles

### 1. Team Quality > Fixture Difficulty

**Elite team with medium fixtures** > **Mid-table team with easy fixtures**

Example:
- ✅ Salah (Liverpool, 2nd) vs Chelsea (Hard) = Still elite
- ❌ Isak (Newcastle, 11th) vs Luton (Easy) = Not worth premium price

### 2. Fixture Run Targeting

Only target fixture runs from **top-6 teams**:
- Arsenal, Liverpool, Man City, Spurs, Bournemouth, Chelsea

**Don't target**:
- Newcastle (11th) even with 3 easy games
- Wolves (15th) even with 4 easy games
- Brighton (12th) even with 5 easy games

### 3. Transfer Banking for Big Moves

Save FTs when planning expensive transfers:

**Example**: Want to bring in Salah (£13m)?
```
GW8: Bank FT (1 total)
GW9: Bank FT (2 total)
GW10: Make 2 transfers to free up funds
GW11: Bring in Salah with saved FTs
```

### 4. Chip Timing Optimization

**Don't rush chips**:
- If GW7 captain scores 8.5 pts
- But GW12 captain scores 9.0 pts
- Save TC for GW12 (+1.0 pts extra)

### 5. Progressive Team Building

Make incremental improvements:
- Use 1 FT per week for small upgrades
- Build toward stronger team over 7 weeks
- Don't waste FTs, but don't hoard either

---

## Fixture Difficulty Guide

### How Difficulty is Calculated

```
Difficulty = (League Position × 60%) + (FPL Strength × 30%) + (Home/Away × 10%)
```

### Position-Based Difficulty

| Opponent Position | Base Difficulty |
|-------------------|-----------------|
| 1-6 (Top teams) | 4.0 - 5.0 (Hard/Very Hard) |
| 7-14 (Mid-table) | 2.5 - 4.0 (Medium) |
| 15-20 (Bottom) | 1.0 - 2.5 (Easy/Very Easy) |

### Adjustments

- **Home advantage**: -0.5 difficulty
- **Away disadvantage**: +0.5 difficulty
- **FPL Strength**: Fine-tunes based on form

### Examples

```
Arsenal (1st) vs Burnley (18th) Away = 1.8 (Very Easy)
Bournemouth (4th) vs Palace (6th) Away = 3.6 (Hard)
Newcastle (11th) vs Forest (17th) Home = 2.1 (Easy)
```

---

## Autonomous Mode

### How It Works

1. **Authentication**: Logs into FPL using your credentials
2. **Analysis**: Performs complete team analysis
3. **Validation**: Checks if transfers are beneficial
4. **Execution**: Makes transfers via FPL API
5. **Confirmation**: Reports success/failure

### Safety Checks

The bot **WON'T** execute transfers if:
- ❌ No credentials provided
- ❌ Authentication fails
- ❌ Transfers already made this week
- ❌ No beneficial transfers found
- ❌ Transfer cost exceeds benefit (net gain < 10 for hits)
- ❌ Budget constraints violated

### Security Best Practices

**Use Environment Variables** (Recommended):
```powershell
# Windows PowerShell
$env:FPL_EMAIL = "your@email.com"
$env:FPL_PASSWORD = "yourpassword"

# Then run without exposing credentials in command
python -m fpl_bot.main 789800 --auto-execute --summary-only
```

**Never**:
- ❌ Commit credentials to git
- ❌ Share your password
- ❌ Use --password in shared scripts

---

## Advanced Features

### Multi-Period Transfer Planning

The bot considers:
- **Current transfers**: Immediate impact
- **Future transfers**: Planned improvements
- **Budget evolution**: Saving for expensive players
- **FT accumulation**: Building to 5 FTs for major changes

### Chip Optimization with Team Evolution

**Traditional approach**:
```
"Your best captain this week scores 8.5 pts, use TC now"
```

**Strategic approach**:
```
Week 1: Captain scores 8.5 with current team
Week 2: Captain scores 8.6 after transfer
Week 3: Captain scores 9.0 after another transfer

Recommendation: Save TC for Week 3 when team is optimized
```

### Fixture Run Exploitation

When bot finds:
```
Liverpool [#2] has 4 easy fixtures (GW10-13)
Avg Difficulty: 1.95 - PREMIUM TARGET
```

**Strategic action**:
- Plan transfers in GW8-9 to bring in Liverpool players
- Use accumulated FTs to get Salah/TAA
- Time Triple Captain for their easiest fixture
- Maximize points during their favorable run

---

## API Usage

```python
from fpl_bot import FPLBot

# Initialize bot
bot = FPLBot()

# Run complete analysis
report = bot.run_analysis(manager_id=789800, save_results=False)

# Access strategic plan
strategic_plan = report['multi_period_plan']

# Get chip recommendations
chip_timing = strategic_plan['chip_plan']
triple_captain_gw = chip_timing['triple_captain']['best_gw']
captain_name = chip_timing['triple_captain']['details']['captain_name']

print(f"Use Triple Captain in GW{triple_captain_gw} on {captain_name}")

# Get transfer plan
team_evolution = strategic_plan['team_evolution']
for gw, week_plan in team_evolution.items():
    if week_plan['num_transfers'] > 0:
        for transfer in week_plan['transfers']:
            print(f"GW{gw}: {transfer['out_name']} -> {transfer['in_name']}")

# Print summary
bot.print_summary(report)
```

---

## Requirements

### Core Dependencies
```
Python 3.7+
numpy >= 1.19.5
pandas >= 1.2.0
requests >= 2.25.0
python-dateutil >= 2.8.2
flask >= 2.0.0
```

### Optional Dependencies
```
pulp >= 2.7.0  # For future MIP optimization
```

Install with:
```bash
pip install -e .                    # Core dependencies
pip install -e ".[strategic]"       # With pulp
```

---

## Configuration

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `FPL_EMAIL` | Your FPL account email (for autonomous mode) |
| `FPL_PASSWORD` | Your FPL account password (for autonomous mode) |

### Setting Environment Variables

**Windows PowerShell**:
```powershell
$env:FPL_EMAIL = "your@email.com"
$env:FPL_PASSWORD = "yourpassword"
```

**Windows CMD**:
```cmd
set FPL_EMAIL=your@email.com
set FPL_PASSWORD=yourpassword
```

**Linux/Mac**:
```bash
export FPL_EMAIL="your@email.com"
export FPL_PASSWORD="yourpassword"
```

---

## Troubleshooting

### "Free Transfers Available: 2" but I played wildcard last week

**Cause**: When you play a wildcard or free hit, you don't gain a free transfer the following week

**Solution**: Use the `--no-ft-gain-last-gw` flag to tell the bot:

```bash
python -m fpl_bot.main YOUR_ID --no-ft-gain-last-gw
```

This will reduce the calculated free transfers by 1 to account for the chip usage.

### "Free Transfers Available: 0" but I haven't made transfers

**Cause**: The bot detected a transfer earlier in the gameweek

**Check**: Log into FPL and verify your transfer history

**Solution**: If incorrect, this is an API data issue

### "Authentication failed"

**Causes**:
- Incorrect email/password
- Special characters in password
- FPL website issues

**Solutions**:
- Verify credentials
- Try escaping special characters
- Check FPL website is accessible

### "No beneficial transfers found"

**This is normal!** Means:
- Your team is already optimal
- No transfers improve expected points
- Bot is protecting you from bad moves

### "Save Triple Captain for GW12"

**This is strategic!** Means:
- Better opportunity coming in GW12
- Current week isn't optimal timing
- Trust the 7-week analysis

### Fixture runs show "AVOID"

**This is correct!** Means:
- Team is mid-table (11th+)
- Quality matters more than fixtures
- Stick with elite team players

---

## Tips for Best Results

### 1. Run Weekly Before Deadline
```bash
# Every Friday/Saturday
python -m fpl_bot.main YOUR_ID --summary-only --no-save
```

### 2. Review Strategic Plan
Don't just look at current week - check the 7-week plan for:
- When to use chips
- Future transfer targets
- FT banking opportunities

### 3. Trust the Quality Filter
If Newcastle has easy fixtures but is marked "AVOID":
- ✅ Trust the bot
- ❌ Don't chase mid-table fixture runs
- ✅ Keep elite team players

### 4. Plan for Fixture Runs
When bot shows:
```
Arsenal [#1] has 4 easy fixtures (GW10-13) - PREMIUM TARGET
```

Start planning NOW:
- Save transfers/budget
- Bring in Arsenal players before GW10
- Time chips for their best fixtures

### 5. Use Autonomous Mode Carefully
- ✅ Test without `--auto-execute` first
- ✅ Review recommendations
- ✅ Then enable autonomous mode
- ⚠️ Monitor execution logs

---

## Examples

### Example 1: Standard Weekly Analysis

```bash
python -m fpl_bot.main 789800 --summary-only --no-save
```

**Output**: Complete analysis with 7-week strategic plan

### Example 2: Autonomous Execution

```bash
python -m fpl_bot.main 789800 --auto-execute --summary-only --no-save
```

**Output**: Same analysis + automatic transfer execution

### Example 3: Save Results

```bash
python -m fpl_bot.main 789800
```

**Output**: Full JSON report saved to file

### Example 4: Python Script

```python
from fpl_bot import FPLBot

bot = FPLBot()
report = bot.run_analysis(789800, save_results=False)

# Access strategic insights
plan = report['multi_period_plan']
print(f"Best TC week: GW{plan['chip_plan']['triple_captain']['best_gw']}")
```

---

## Performance

| Operation | Time | API Calls |
|-----------|------|-----------|
| Standard Analysis | ~8-10s | 5-8 calls |
| + Auto-Execute | ~12s | 8-12 calls |
| 7-Week Planning | ~10s | 5-8 calls |

---

## Development

### Project Structure
```
FPL_robo/
├── fpl_bot/
│   ├── __init__.py
│   ├── main.py                          # CLI and orchestration
│   └── core/
│       ├── data_collector.py            # API and data fetching
│       ├── manager_analyzer.py          # Team analysis
│       ├── predictor.py                 # Point predictions
│       ├── transfer_optimizer.py        # Single-week transfers
│       ├── chip_manager.py              # Chip management
│       └── multi_period_planner.py      # 7-GW strategic planning
├── setup.py
├── README.md
└── LICENSE
```

### Adding New Features

**Add new prediction method**:
1. Edit `fpl_bot/core/predictor.py`
2. Add method to `Predictor` class
3. Use in predictions pipeline

**Enhance transfer optimization**:
1. Edit `fpl_bot/core/transfer_optimizer.py`
2. Modify `optimize_transfers` method
3. Test with different scenarios

**Improve strategic planning**:
1. Edit `fpl_bot/core/multi_period_planner.py`
2. Adjust `plan_gameweeks` method
3. Update thresholds as needed

---

## Changelog

### v2.1.0 (Current)
- ✅ **Web Dashboard UI** - Beautiful interface with auto-refresh
- ✅ 7-gameweek multi-period strategic planning (default)
- ✅ Chip optimization with team evolution
- ✅ Autonomous transfer execution
- ✅ Quality-based fixture run analysis
- ✅ Transfer banking strategy (up to 5 FTs)
- ✅ Improved fixture difficulty calculation
- ✅ Week-by-week breakdown display
- ✅ Player-specific chip recommendations
- ✅ Dynamic planning horizon (to GW19 or +7)
- ✅ Security via environment variables
- ✅ Clean separation of UI and debug output

### v2.0.0
- Complete rewrite with modular architecture
- Clean separation of concerns
- Improved prediction engine
- Enhanced transfer optimization
- Comprehensive chip management
- Better CLI interface

### v1.0.0
- Initial implementation (deprecated)

---

## FAQ

### Q: Do I need to install PuLP?

**A**: No, it's optional. Bot works great with heuristic optimization currently. PuLP is for future full MIP solver.

### Q: Will the bot make transfers without asking?

**A**: Only if you use `--auto-execute` flag. By default, it just shows recommendations.

### Q: How accurate are the predictions?

**A**: Predictions are fixture-adjusted estimates. Use as guidance, not guarantees. Check injuries/news before finalizing.

### Q: Can I trust the "AVOID" recommendations?

**A**: Yes! The bot uses data-driven quality filtering. Elite teams score more even with harder fixtures.

### Q: Should I always follow the 7-week plan?

**A**: Use as strategic guidance. Adjust for:
- Breaking news (injuries, suspensions)
- Form changes
- Price rises/falls
- Your own insights

### Q: Why does it recommend saving chips?

**A**: Strategic timing! Using TC in GW12 (9.0 pts) beats GW7 (8.5 pts). Small differences compound over a season.

### Q: What if I disagree with a transfer?

**A**: Don't use `--auto-execute`. Review recommendations and make transfers manually if you prefer different options.

---

## Security & Privacy

### Credentials
- Store in environment variables (not in code)
- Never commit to git
- Bot only uses for authentication
- Not stored permanently

### API Usage
- Minimal API calls
- Respects rate limits
- Proper error handling
- No data shared externally

### Autonomous Execution
- Only executes when explicitly enabled (`--auto-execute`)
- Clear logging of all actions
- Safety checks prevent bad decisions
- Can always disable and use manually

---

## Contributing

Contributions welcome! Areas for improvement:

1. **Full MIP Implementation** - Use PuLP for provably optimal solutions
2. **Price Change Predictions** - Factor in player price movements
3. **Formation Optimization** - Auto-select best starting XI
4. **Captain Auto-Setting** - Execute captain selection via API
5. **Wildcard Planning** - Optimal timing for wildcard chip
6. **Machine Learning** - Train models on historical data
7. **Double Gameweek Detection** - Identify and plan for DGWs
8. **Injury/Suspension Tracking** - Integrate with news APIs

### How to Contribute

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes
4. Test thoroughly
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push (`git push origin feature/amazing-feature`)
7. Open Pull Request

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Disclaimer

⚠️ **Use at your own risk!**

This bot:
- Makes real changes to your FPL team (in autonomous mode)
- Provides predictions, not guarantees
- Should be used as decision support, not blindly followed
- Is provided as-is with no warranties

For important gameweeks or cup matches, always review recommendations manually.

---

## Credits

**Author**: Callum Waller

**Data Sources**:
- [Fantasy Premier League API](https://fantasy.premierleague.com/api/)
- [FPL Data](https://www.fpl-data.co.uk/)

**Inspired by**: The FPL community's excellent optimization tools and strategies

---

## Support

Having issues or questions?

1. Check the [Troubleshooting](#troubleshooting) section
2. Review the example output above
3. Run with `--help` flag for command options
4. Check you're using Python 3.7+

---

## Quick Reference

### Standard Analysis (Web Dashboard)
```bash
python -m fpl_bot.main YOUR_MANAGER_ID
```

### Terminal Only Mode
```bash
python -m fpl_bot.main YOUR_MANAGER_ID --no-ui --summary-only
```

### Autonomous Mode
```bash
$env:FPL_EMAIL = "your@email.com"
$env:FPL_PASSWORD = "yourpassword"
python -m fpl_bot.main YOUR_MANAGER_ID --auto-execute
```

### Get Help
```bash
python -m fpl_bot.main --help
```

---

**Transform your FPL game with strategic 7-gameweek planning and a beautiful web dashboard!** 🚀🏆🌐
