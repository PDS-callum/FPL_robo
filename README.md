# FPL Platform - AI-Powered Fantasy Premier League Manager

A comprehensive web platform for managing Fantasy Premier League teams with integrated AI-powered bot, multi-user support, team management, and bot competitions.

---

## 🎯 Key Features

### Platform Features
- **👥 Multi-User System** - Secure user accounts with authentication
- **🏆 Multi-Team Management** - Manage multiple FPL teams from one account
- **🤖 AI Bot Integration** - Run advanced FPL analysis with one click
- **🏅 Bot Competition** - Create and compete with custom Python bots
- **📊 Analysis Dashboard** - Track analysis history and results
- **⚙️ Per-Team Settings** - Configure bot parameters for each team
- **📈 Historical Tracking** - View past analyses and compare strategies
- **🌐 Modern UI** - Beautiful Bootstrap-based responsive interface
- **🔌 REST API** - Programmatic access to all platform features

### Bot Engine (FPL Bot v2.0)
- **7-Gameweek Strategic Planning** - Multi-period optimization with team evolution
- **Intelligent Chip Timing** - Optimizes Triple Captain & Bench Boost usage
- **Fixture Analysis** - Quality-based fixture run detection
- **Transfer Banking** - Plans accumulation of up to 5 free transfers
- **Autonomous Execution** - Automatically makes transfers (optional)
- **Web Dashboard** - Beautiful UI with auto-refresh

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/FPL_robo.git
cd FPL_robo

# Install dependencies
pip install -r requirements.txt

# Launch platform
python run_platform.py
```

**Platform URL:** http://127.0.0.1:5001

### First Time Setup

1. **Register** - Create your account at `/auth/register`
2. **Login** - Sign in at `/auth/login`
3. **Add Team** - Enter your FPL Manager ID (find it in your FPL URL: `fantasy.premierleague.com/entry/[YOUR_ID]/event/1`)
4. **Run Analysis** - Click "Run Bot Analysis" on your team
5. **View Results** - See 7-week strategic plans and recommendations

---

## 📖 Platform Usage

### Dashboard
Your main hub showing team overviews, quick stats, and recent analyses.

### Managing Teams
- **Add Teams**: Enter FPL Manager ID and optional custom name
- **Edit Settings**: Configure bot parameters per team
- **Set Primary**: Choose your main team for dashboard display
- **View History**: Track all past analyses

### Bot Analysis
1. Navigate to your team
2. Click "Run Bot Analysis"
3. Wait 10-30 seconds for completion
4. Review recommendations:
   - Current week transfers
   - Captain suggestion
   - 7-week strategic plan
   - Chip timing recommendations
   - Fixture opportunities

### Bot Settings

Configure per-team:
- **Risk Aversion** (0-1): 0=aggressive, 0.5=balanced, 1=conservative
- **Min Playing Chance** (0-100%): Filter uncertain players (default: 75%)
- **Analyze Wildcard**: Enable wildcard timing analysis (slower)
- **Prevent Hits**: Only use free transfers, never take point hits

---

## 🏅 Bot Competition

### Overview
Create custom FPL bots in Python and compete against other users. Backtest on historical data and activate for live competition.

### Creating a Bot

Your bot must implement two methods:

```python
class MyFPLBot:
    def __init__(self):
        self.team = []
    
    def analyze(self, data):
        """Analyze current team and FPL data"""
        return {
            'status': 'ok',
            'message': 'Analysis complete',
            'insights': []
        }
    
    def optimize(self, data):
        """
        Optimize team selection and make transfers
        
        Parameters:
            data (dict): Contains:
                - current_team: List of 15 players in squad
                - available_players: All players (temporally filtered)
                - gameweek: Current gameweek number
                - bank: Available budget
                - free_transfers: Free transfers available
        
        Returns:
            dict: Transfer decisions and team lineup
        """
        return {
            'transfers': [],  # [{out: player_id, in: player_id}]
            'captain': player_id,
            'vice_captain': player_id,
            'lineup': [player_ids],  # 11 starting players
            'bench': [player_ids]   # 4 bench players
        }
```

### Data Security

**Temporal Filtering Guarantee:**
- Bots only receive data from completed gameweeks (GW 1 to GW-1)
- No future performance data is leaked
- Ensures fair competition based on prediction, not lookup
- Forces bots to use intelligence, not hindsight

**Allowed in Bot Code:**
- ✅ Basic Python (list, dict, loops, functions, classes)
- ✅ Math operations (sum, min, max, abs, round)
- ✅ Data structures and algorithms
- ✅ Your own logic and strategies

**Restricted for Security:**
- ❌ File system access (`open`, `file`)
- ❌ System operations (`os`, `subprocess`, `sys`)
- ❌ Dynamic code execution (`eval`, `exec`)
- ❌ Module imports (`__import__`, `import`)
- ❌ Network access (`requests`, `urllib`)

### Submitting Your Bot

1. Navigate to `/competition/`
2. Click "Submit New Bot"
3. Upload your `.py` file
4. Bot is validated automatically
5. Run backtest on historical seasons
6. Activate for current season competition

### Backtesting

Test your bot on previous seasons:
- Select season (e.g., 2023-24) or current season
- Choose gameweek range
- See detailed performance:
  - Total points scored
  - Team selected each week
  - Transfers made
  - Player stats (points, goals, assists, bonus)
  - Captain choices
- Compare against other bots

---

## 🔌 API Endpoints

All endpoints require authentication via session cookies.

### Teams
```bash
GET  /api/teams                    # List user's teams
GET  /api/teams/<id>               # Get team details
GET  /api/teams/<id>/analyses      # Get team analyses
```

### Analyses
```bash
GET  /api/analyses/<id>            # Get analysis details
GET  /api/analyses/<id>?full=true  # Include complete bot report
```

### Competition
```bash
GET  /api/competition/leaderboard  # Competition standings
GET  /api/competition/my-entry     # User's active entry
```

### Health Check
```bash
GET  /api/health                   # Platform status
```

---

## 🤖 Standalone Bot Usage

The bot can also run independently via CLI:

```bash
# Basic analysis
python -m bot.main YOUR_MANAGER_ID

# With options
python -m bot.main 789800 --verbose --analyze-wildcard --no-hits

# Autonomous mode (auto-execute transfers)
export FPL_EMAIL="your@email.com"
export FPL_PASSWORD="yourpassword"
python -m bot.main 789800 --auto-execute
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `manager_id` | Your FPL Manager ID (required) |
| `--no-ui` | Disable web UI (terminal only) |
| `--summary-only` | Show summary output only |
| `--no-save` | Don't save results to file |
| `--auto-execute` | Automatically execute transfers |
| `--no-hits` | Never recommend point hits |
| `--risk VALUE` | Risk aversion (0-1, default: 0.5) |
| `--min-playing-chance PCT` | Min % chance of playing (default: 75) |
| `--analyze-wildcard` | Analyze wildcard timing (~5-10 min) |

---

## 🏗️ Architecture

```
FPL_robo/
├── bot/                      # Bot engine (FPL Bot v2.0)
│   ├── core/                # Bot logic & optimization
│   │   ├── chip_manager.py
│   │   ├── data_collector.py
│   │   ├── manager_analyzer.py
│   │   ├── predictor.py
│   │   ├── team_optimizer.py
│   │   └── wildcard_optimizer.py
│   └── main.py              # CLI entry point
│
├── fpl_platform/            # Web platform
│   ├── models/             # Database models
│   │   ├── user.py         # User accounts
│   │   ├── team.py         # FPL teams
│   │   ├── analysis.py     # Bot analyses
│   │   └── competition.py  # Competition (BotSubmission, BacktestResult, SeasonEntry)
│   ├── routes/             # Web routes
│   │   ├── auth.py         # Authentication
│   │   ├── dashboard.py    # Dashboard pages
│   │   ├── bot.py          # Bot operations
│   │   ├── competition.py  # Competition features
│   │   └── api.py          # REST API
│   ├── services/           # Business logic
│   │   ├── auth_service.py
│   │   ├── bot_service.py
│   │   ├── bot_execution_service.py  # Competition bot execution & backtesting
│   │   ├── fpl_data_service.py       # FPL API integration
│   │   └── team_service.py
│   ├── templates/          # HTML templates
│   └── static/            # CSS, JS
│
├── run_platform.py         # Platform entry point
└── requirements.txt        # Dependencies
```

---

## 📊 Bot Features Deep Dive

### 7-Gameweek Strategic Planning

The bot plans 7 gameweeks ahead, showing:
- Week-by-week transfer recommendations
- Free transfer accumulation (up to 5)
- Expected points for each week
- Team composition evolution
- Optimal chip usage timing

### Fixture Difficulty Ratings

Calculated using weighted formula:
- **60% League Position** - Most important factor
- **30% FPL Strength Ratings** - Form and stats
- **10% Home/Away Advantage** - Venue impact

**Ratings:**
- Very Easy: ≤ 2.0
- Easy: ≤ 2.8
- Medium: ≤ 3.5
- Hard: ≤ 4.2
- Very Hard: > 4.2

### Quality-Based Recommendations

Elite team with medium fixtures > Mid-table team with easy fixtures

**Target:** Top 6 teams only
- Arsenal, Liverpool, Man City, Spurs, etc.

**Avoid:** Mid-table teams (11th+)
- Even with easy fixtures, quality matters more

### Chip Optimization

Chips analyzed based on **actual planned team** for each future gameweek:

**Triple Captain:**
- Finds best gameweek for your best captain
- Shows expected points impact
- Provides alternative options

**Bench Boost:**
- Identifies week with strongest bench
- Shows bench player point predictions
- Considers team evolution

**Constraint:** Only one chip per gameweek (enforced)

### Transfer Banking

Plans to accumulate up to 5 free transfers:
- Bank when no beneficial transfers available
- Use for expensive targets (Salah, Haaland)
- Build toward fixture runs

---

## ⚙️ Configuration

### Environment Variables

```bash
# Flask configuration
export FLASK_ENV=development
export SECRET_KEY=your-secret-key
export DATABASE_URL=sqlite:///fpl_platform.db

# Server configuration
export HOST=127.0.0.1
export PORT=5001

# FPL credentials (for autonomous mode)
export FPL_EMAIL=your@email.com
export FPL_PASSWORD=yourpassword
```

### Database

SQLite database with SQLAlchemy ORM:
- `users` - User accounts
- `teams` - FPL teams
- `analyses` - Bot analysis results
- `bot_submissions` - Competition bot code
- `backtest_results` - Historical backtest data
- `season_entries` - Active competition entries

---

## 🔧 Troubleshooting

### Common Issues

**Port Already in Use**
```bash
# Windows
netstat -ano | findstr :5001
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:5001 | xargs kill -9
```

**Database Errors**
```bash
# Reset database
rm instance/fpl_platform.db
python run_platform.py  # Will recreate
```

**Bot Analysis Stuck**
1. Refresh the page
2. Check terminal for errors
3. Restart platform

**Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

---

## 💡 Tips & Best Practices

### For Users
1. **Run Weekly** - Analyze before each deadline
2. **Review 7-Week Plan** - Don't just look at current week
3. **Trust Quality Filter** - Elite teams > fixture runs
4. **Use Multiple Teams** - Compare different strategies
5. **Track History** - Learn from past analyses

### For Bot Developers
1. **Start Simple** - Basic logic first, optimize later
2. **Test Thoroughly** - Use backtests before activating
3. **Consider Form** - Recent performance matters
4. **Watch Budget** - Track spending carefully
5. **Plan Ahead** - Use `gameweek` parameter to anticipate

---

## 📈 Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Standard Analysis | ~10-15s | 7-week plan included |
| Wildcard Analysis | ~5-10 min | Comprehensive optimization |
| Bot Backtest | ~30-60s | Per season, depends on GW range |
| Competition Bot Validation | <1s | Security checks |

---

## 🛠️ Development

### Running in Dev Mode

```bash
export FLASK_ENV=development
export FLASK_DEBUG=1
python run_platform.py
```

### Project Structure Principles
- **Models** - Database schema only
- **Services** - Business logic layer
- **Routes** - HTTP request handlers
- **Templates** - HTML presentation

### Adding Features

**New Route:**
1. Create in `fpl_platform/routes/`
2. Register blueprint

**New Model:**
1. Define in `fpl_platform/models/`
2. Use Flask-Migrate for schema changes

**New Service:**
1. Create in `fpl_platform/services/`
2. Import and use in routes

---

## 📜 Requirements

### Core Dependencies
- Python 3.7+
- Flask 2.0+
- SQLAlchemy
- Flask-Login
- Flask-Migrate
- requests
- numpy
- pandas

### Optional
- PuLP (for advanced MIP optimization)

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

1. **Machine Learning Models** - Train on historical data
2. **Price Change Predictions** - Factor in player prices
3. **Formation Optimization** - Auto-select best XI
4. **Double Gameweek Detection** - Identify and plan for DGWs
5. **Injury Tracking** - Integrate with news APIs
6. **Bot Marketplace** - Share and download community bots
7. **Performance Graphs** - Visual analytics for backtests
8. **Mobile App** - Native mobile experience

### How to Contribute
1. Fork repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

---

## 📄 License

MIT License - See LICENSE file for details.

---

## ⚠️ Disclaimer

**Use at your own risk!**

This platform:
- Makes real changes to your FPL team (in autonomous mode)
- Provides predictions, not guarantees
- Should be used as decision support, not blindly followed
- Is provided as-is with no warranties

Always review recommendations manually for important decisions.

---

## 💬 Support

Having issues?

1. Check Troubleshooting section above
2. Review command options with `--help`
3. Check Python version (3.7+ required)
4. Verify all dependencies installed
5. Check server logs for errors

---

## 🌟 Credits

**Author:** Callum Waller

**Data Sources:**
- [Fantasy Premier League API](https://fantasy.premierleague.com/api/)
- [FPL Community](https://www.reddit.com/r/FantasyPL/)

**Inspired by:** The FPL community's optimization tools and strategies

---

## 📚 Quick Reference

### Launch Platform
```bash
python run_platform.py
# Visit http://127.0.0.1:5001
```

### Standalone Bot
```bash
python -m bot.main YOUR_MANAGER_ID
```

### Autonomous Mode
```bash
export FPL_EMAIL="your@email.com"
export FPL_PASSWORD="yourpassword"
python -m bot.main YOUR_MANAGER_ID --auto-execute
```

---

**Transform your FPL game with AI-powered analysis and strategic planning!** 🚀🏆⚽
