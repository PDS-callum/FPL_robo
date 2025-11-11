# FPL Bot Competition Guide

## Overview

The FPL Bot Competition is a new feature that allows users to create their own Fantasy Premier League bots and compete against other users. Your bot will be backtested on historical data and can be activated for the current season to compete in real-time.

## Features

### 1. Bot Submission
- Write your own FPL bot in Python
- Code validation ensures security and proper implementation
- Bot must implement required interface methods

### 2. Backtesting
- Test your bot on previous seasons (e.g., 2023-24, 2022-23)
- See how your bot would have performed historically
- Analyze performance metrics before going live

### 3. Season Competition
- Activate your bot for the current season
- Compete against other users' bots
- Track your ranking on the leaderboard

### 4. Profile Integration
- Your active bot's performance is displayed on your profile page
- Quick access to competition stats and leaderboard

## Getting Started

### 1. Navigate to Competition
- Click the "Competition" tab in the navigation bar
- Or go to http://127.0.0.1:5001/competition/

### 2. Create Your Bot

Create a Python file (`.py`) containing your bot class. Your bot must implement two methods:

```python
class MyFPLBot:
    def __init__(self):
        self.team = []
    
    def analyze(self, data):
        """
        Analyze current team and FPL data
        
        Parameters:
            data (dict): Contains current team, player stats, fixtures, etc.
        
        Returns:
            dict: Analysis results with insights
        """
        return {
            'status': 'ok',
            'message': 'Analysis complete',
            'insights': []
        }
    
    def optimize(self, data):
        """
        Optimize team selection and make transfers
        
        Parameters:
            data (dict): Contains player predictions, current team, budget, etc.
        
        Returns:
            dict: Transfer decisions and team lineup
        """
        return {
            'transfers': [],  # List of {out: player_id, in: player_id}
            'captain': player_id,
            'vice_captain': player_id,
            'lineup': [player_ids]  # 11 starting players
        }
```

### 3. Submit Your Bot
1. Click "Submit New Bot"
2. Give your bot a name and description
3. Upload your bot's Python file (.py)
4. Preview the file content to ensure it's correct
5. Submit for validation

**Pro Tips:**
- Download the sample bot template to get started quickly
- Write and test your code in your favorite IDE
- Use version control (Git) to track changes to your bot
- You can download your bot code from the submission page for editing

### 4. Run Backtest
1. View your bot submission
2. Click "Run Backtest"
3. Select season and gameweek range
4. Wait for results
5. Analyze performance metrics

### 5. Activate for Current Season
1. Ensure your bot is validated
2. Click "Activate for Season"
3. Confirm activation
4. Your bot will start competing!

## Security Restrictions

For security, the following are **not allowed** in bot code:

❌ File system access (`open`, `file`)  
❌ System operations (`os`, `subprocess`, `sys`)  
❌ Dynamic code execution (`eval`, `exec`, `compile`)  
❌ Module imports (`__import__`)  
❌ User input (`input`, `raw_input`)  

✅ Allowed:
- Basic Python built-ins (list, dict, str, int, float, etc.)
- Standard operations (print, len, range, sum, min, max, etc.)
- Data structures and algorithms

## Database Schema

### BotSubmission
- Stores user-submitted bot code
- Tracks validation status
- Links to backtests and season entries

### BacktestResult
- Stores historical performance data
- Includes detailed gameweek-by-gameweek results
- Performance metrics (avg points, best/worst GW, etc.)

### SeasonEntry
- Tracks active bot for current season
- Real-time performance data
- Gameweek history and statistics

## Routes

- `/competition/` - Competition home page
- `/competition/submit` - Submit a new bot
- `/competition/submission/<id>` - View bot details
- `/competition/submission/<id>/edit` - Edit bot
- `/competition/submission/<id>/backtest` - Run backtest
- `/competition/submission/<id>/activate` - Activate for season
- `/competition/backtest/<id>` - View backtest results
- `/competition/leaderboard` - Competition leaderboard
- `/competition/my-entry` - Your active entry details
- `/competition/documentation` - Bot development docs

## Testing

Run the test suite:

```bash
python test_competition.py
```

This will verify:
- ✅ Database tables exist
- ✅ Bot validation works
- ✅ Security restrictions are enforced
- ✅ Models function correctly
- ✅ Routes are registered

## Manual Testing

1. Start the server:
   ```bash
   python run_platform.py
   ```

2. Navigate to http://127.0.0.1:5001/competition/

3. Test the following:
   - Submit a new bot
   - Edit bot code
   - Run backtest
   - View backtest results
   - Activate bot for season
   - Check leaderboard
   - View profile page (should show active entry)

## Migration

To add competition tables to an existing database:

```bash
python migrate_db.py
```

This creates:
- `bot_submissions`
- `backtest_results`
- `season_entries`

## Future Enhancements

Potential improvements:
1. Real-time bot execution during gameweeks
2. More detailed performance analytics
3. Bot versioning and history
4. Community bot sharing
5. Advanced backtest options (different starting budgets, etc.)
6. Integration with actual FPL API for live competition
7. Bot marketplace/templates
8. Performance graphs and charts
9. Email notifications for bot performance
10. Bot vs bot head-to-head comparisons

## Troubleshooting

### Bot Validation Fails
- Ensure both `analyze` and `optimize` methods are implemented
- Check for syntax errors
- Verify no restricted imports are used

### Backtest Not Running
- Check bot is validated
- Ensure season data is available
- Check server logs for errors

### Can't Activate Bot
- Bot must be validated first
- Only one active bot per user
- Previous bot will be deactivated automatically

## Support

For issues or questions:
1. Check the documentation at `/competition/documentation`
2. Review error messages carefully
3. Test with the example bot template
4. Check server logs for detailed error information

---

**Happy Coding! May the best bot win! 🏆**

