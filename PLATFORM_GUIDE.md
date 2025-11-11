# FPL Platform - Complete Setup & Usage Guide

This guide walks you through setting up and using the FPL Platform.

---

## 📋 Table of Contents

1. [Installation](#installation)
2. [First Time Setup](#first-time-setup)
3. [Using the Platform](#using-the-platform)
4. [Bot Configuration](#bot-configuration)
5. [API Usage](#api-usage)
6. [Troubleshooting](#troubleshooting)
7. [Development](#development)

---

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Git

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/FPL_robo.git
cd FPL_robo
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### Step 4: Launch Platform

```bash
python run_platform.py
```

The platform will be available at: http://127.0.0.1:5001

---

## 🎯 First Time Setup

### 1. Register an Account

1. Open http://127.0.0.1:5001/welcome in your browser
2. Click "Register"
3. Fill in your details:
   - Username (unique)
   - Email
   - Password (minimum 6 characters)
4. Submit the form

### 2. Login

1. Go to http://127.0.0.1:5001/auth/login
2. Enter your username or email
3. Enter your password
4. Click "Login"

### 3. Add Your FPL Team

1. Click "My Teams" in the navigation
2. Click "Add Team"
3. Enter your FPL Manager ID:
   - Find it at: fantasy.premierleague.com/entry/**[YOUR_ID]**/event/1
   - Example: If URL is `/entry/789800/event/1`, your ID is `789800`
4. Optionally enter a custom team name
5. Click "Add Team"

### 4. Run Your First Analysis

1. From the dashboard, click "Run Bot Analysis" on your team
2. Or go to "My Teams" → Select team → "Analyze"
3. Click "Run Bot Analysis" button
4. Wait 10-30 seconds for analysis to complete
5. View your results!

---

## 💻 Using the Platform

### Dashboard

**URL:** `/`

Your main hub showing:
- Primary team overview
- Quick stats
- All your teams
- Quick access to analyses

### My Teams

**URL:** `/teams`

Manage all your FPL teams:
- View all teams
- Add new teams
- Edit team settings
- Delete teams
- Set primary team

### Team Detail

**URL:** `/teams/<team_id>`

Detailed view of a specific team:
- Team stats (rank, points, value)
- Recent analyses
- Quick access to run analysis

### Run Analysis

**URL:** `/bot/run/<team_id>`

Run bot analysis for a team:
- Shows current bot settings
- Recent analysis history
- Start new analysis
- Auto-refreshes when complete

### View Analysis

**URL:** `/bot/analysis/<analysis_id>`

View detailed results:
- Weekly plans (7 gameweeks)
- Transfer recommendations
- Captain suggestions
- Expected points
- Hit analysis

### Profile

**URL:** `/auth/profile`

Manage your account:
- Update personal information
- Change password
- View account stats

---

## ⚙️ Bot Configuration

Each team can have custom bot settings.

### Accessing Settings

1. Go to "My Teams"
2. Select a team
3. Click "Settings"

### Available Settings

#### Risk Aversion (0.0 - 1.0)

Controls bot's risk appetite:
- **0.0** - Aggressive: Favors high-ceiling players
- **0.5** - Balanced: Mix of safety and upside (default)
- **1.0** - Conservative: Favors consistent players

#### Minimum Playing Chance (0-100%)

Filter out uncertain players:
- **75%** - Default, excludes doubtful players
- **100%** - Only fully fit players
- **50%** - Include some rotation risks

#### Analyze Wildcard

Enable wildcard timing analysis:
- **Off** - Faster analysis (~10-15 seconds)
- **On** - Includes wildcard recommendations (~5-10 minutes)

**When to enable:**
- You have your wildcard available
- Planning multiple gameweeks ahead
- Major team restructuring needed

#### Prevent Point Hits

Force bot to only use free transfers:
- **Off** - Bot can recommend hits if beneficial
- **On** - Only use available free transfers

**When to enable:**
- You're risk-averse
- Small advantage isn't worth -4 points
- Accumulating free transfers

---

## 🔌 API Usage

The platform provides REST API endpoints for programmatic access.

### Authentication

Login via web interface first. API uses session cookies.

### Endpoints

#### Get All Teams

```bash
GET /api/teams
```

**Response:**
```json
{
  "teams": [
    {
      "id": 1,
      "fpl_manager_id": 789800,
      "team_name": "My Team",
      "overall_rank": 123456,
      "total_points": 450
    }
  ]
}
```

#### Get Team Details

```bash
GET /api/teams/<team_id>
```

#### Get Team Analyses

```bash
GET /api/teams/<team_id>/analyses?limit=10
```

**Response:**
```json
{
  "analyses": [
    {
      "id": 1,
      "gameweek": 10,
      "status": "completed",
      "expected_points": 65.5,
      "transfers_count": 1,
      "captain_name": "Haaland"
    }
  ]
}
```

#### Get Analysis Details

```bash
GET /api/analyses/<analysis_id>?full=true
```

Set `full=true` to include complete bot report.

#### Health Check

```bash
GET /api/health
```

### Example: Python Script

```python
import requests

# Base URL
BASE_URL = "http://127.0.0.1:5001/api"

# Create session (login first via web)
session = requests.Session()

# Get teams
response = session.get(f"{BASE_URL}/teams")
teams = response.json()["teams"]

# Get analyses for first team
team_id = teams[0]["id"]
response = session.get(f"{BASE_URL}/teams/{team_id}/analyses")
analyses = response.json()["analyses"]

# Get full analysis data
if analyses:
    analysis_id = analyses[0]["id"]
    response = session.get(f"{BASE_URL}/analyses/{analysis_id}?full=true")
    full_data = response.json()
    print(full_data)
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Import Errors

**Error:** `ModuleNotFoundError: No module named 'platform'`

**Solution:**
```bash
# Make sure you're in project root
cd FPL_robo

# Install in development mode
pip install -e .
```

#### 2. Database Errors

**Error:** `OperationalError: no such table: users`

**Solution:**
```bash
# Delete old database
rm fpl_platform.db

# Restart platform (will recreate database)
python run_platform.py
```

#### 3. Port Already in Use

**Error:** `OSError: [Errno 48] Address already in use`

**Solution:**
```bash
# Use different port
PORT=5002 python run_platform.py

# Or kill process using port 5001
# Windows:
netstat -ano | findstr :5001
taskkill /PID <PID> /F

# macOS/Linux:
lsof -ti:5001 | xargs kill -9
```

#### 4. Bot Analysis Stuck

**Symptom:** Analysis shows "running" indefinitely

**Solution:**
1. Refresh the page
2. Check terminal for errors
3. Restart the platform
4. Delete the stuck analysis:
   - Go to Team Details
   - Analysis will show as "failed"
   - You can delete it

#### 5. Login Issues

**Error:** "Invalid username or password"

**Solution:**
- Check username/email spelling
- Passwords are case-sensitive
- Try password reset (if implemented)
- Create new account if needed

---

## 👨‍💻 Development

### Project Structure

```
FPL_robo/
├── bot/                      # Bot engine
│   ├── __init__.py
│   ├── main.py              # Bot CLI entry point
│   ├── core/                # Core bot logic
│   │   ├── chip_manager.py
│   │   ├── data_collector.py
│   │   ├── manager_analyzer.py
│   │   ├── predictor.py
│   │   ├── team_optimizer.py
│   │   └── wildcard_optimizer.py
│   └── ui/                  # Legacy bot UI
│
├── platform/                # Web platform
│   ├── __init__.py
│   ├── app.py              # Flask app factory
│   ├── config.py           # Configuration
│   ├── database.py         # Database setup
│   ├── models/             # Database models
│   │   ├── user.py
│   │   ├── team.py
│   │   └── analysis.py
│   ├── routes/             # Flask routes
│   │   ├── auth.py         # Authentication
│   │   ├── dashboard.py    # Dashboard pages
│   │   ├── bot.py          # Bot operations
│   │   └── api.py          # REST API
│   ├── services/           # Business logic
│   │   ├── auth_service.py
│   │   ├── bot_service.py
│   │   └── team_service.py
│   ├── templates/          # HTML templates
│   │   ├── base.html
│   │   ├── dashboard.html
│   │   └── ...
│   └── static/             # Static files
│       ├── css/
│       └── js/
│
├── run_platform.py         # Platform entry point
├── requirements.txt        # Dependencies
└── README.md              # Documentation
```

### Running in Development Mode

```bash
# Set environment
export FLASK_ENV=development

# Run with debug mode
python run_platform.py
```

### Database Migrations

If you modify models:

```bash
# Initialize migrations (first time only)
flask db init

# Create migration
flask db migrate -m "Description of changes"

# Apply migration
flask db upgrade
```

### Adding New Features

#### Adding a Route

1. Create/edit file in `platform/routes/`
2. Add route function
3. Register blueprint in `platform/routes/__init__.py`

#### Adding a Database Model

1. Create model in `platform/models/`
2. Import in `platform/models/__init__.py`
3. Create migration
4. Apply migration

#### Adding a Service

1. Create service in `platform/services/`
2. Implement business logic
3. Use in routes/other services

### Testing

```bash
# Run tests (when implemented)
pytest

# With coverage
pytest --cov=platform
```

---

## 🌟 Tips & Best Practices

### For Users

1. **Run Analysis Weekly** - Before each gameweek deadline
2. **Check Multiple Teams** - Compare strategies across teams
3. **Save Important Analyses** - Reference past recommendations
4. **Adjust Settings** - Experiment with risk levels
5. **Use History** - Learn from past analyses

### For Developers

1. **Use Virtual Environment** - Keep dependencies isolated
2. **Follow Structure** - Keep code organized by layer
3. **Error Handling** - Always handle exceptions
4. **Database Sessions** - Commit/rollback properly
5. **Security** - Never commit secrets or passwords

---

## 📚 Additional Resources

- [FPL Official API](https://fantasy.premierleague.com/api/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Bootstrap Documentation](https://getbootstrap.com/docs/)

---

## 🤝 Support

For issues and questions:

1. Check this guide
2. Review main README.md
3. Check troubleshooting section
4. Create GitHub issue (if applicable)

---

## 📄 License

MIT License - See LICENSE file for details

---

**Happy FPL Managing! 🏆⚽**

