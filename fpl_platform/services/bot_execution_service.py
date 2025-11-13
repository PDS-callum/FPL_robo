"""
Service for executing user-submitted bot code safely
"""
import sys
import io
import time
import json
import os
import traceback
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timedelta
from fpl_platform.database import db
from fpl_platform.models import BotSubmission, BacktestResult, SeasonEntry


class BotExecutionService:
    """
    Service to validate, execute, and backtest user-submitted bots
    """
    
    # Required methods that a bot class must implement
    REQUIRED_METHODS = ['analyze', 'optimize']
    
    # Restricted imports for security
    # CRITICAL: These restrictions prevent bots from accessing external data sources
    # and ensure the platform is the ONLY source of data for fair competition
    RESTRICTED_IMPORTS = [
        # System access
        'os', 'subprocess', 'sys', 'eval', 'exec', 'compile',
        '__import__', 'open', 'file', 'input', 'raw_input',
        # Network access - MUST BE BLOCKED to prevent external data sources
        'requests', 'urllib', 'urllib2', 'urllib3', 'http', 'httplib',
        'socket', 'socketserver', 'ftplib', 'telnetlib', 'smtplib',
        'poplib', 'imaplib', 'nntplib', 'ssl', 'asyncio',
        # Additional security
        'pickle', 'shelve', 'marshal', 'dbm', 'sqlite3',
        'multiprocessing', 'threading', 'thread'
    ]
    
    # Cache configuration
    CACHE_DIR = 'fpl_cache'
    CACHE_EXPIRY_HOURS = 24  # Current season cache expires after 24 hours
    
    @staticmethod
    def validate_bot_code(code):
        """
        Validate bot code for security and structure
        Returns (is_valid, error_message, bot_class_name)
        """
        try:
            # Check for restricted imports/functions with word boundaries
            import re
            for restricted in BotExecutionService.RESTRICTED_IMPORTS:
                # Use word boundaries to avoid false positives (e.g., 'os' in 'close')
                # Check for: import os, from os, os., os(, os[, or standalone os
                pattern = r'\b' + re.escape(restricted) + r'\b'
                if re.search(pattern, code):
                    return False, f"Restricted keyword '{restricted}' found in code", None
            
            # Try to compile the code
            try:
                compile(code, '<string>', 'exec')
            except SyntaxError as e:
                return False, f"Syntax error: {str(e)}", None
            
            # Execute code in restricted namespace to get the class
            import builtins
            namespace = {
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'map': map,
                    'filter': filter,
                    'sorted': sorted,
                    'sum': sum,
                    'min': min,
                    'max': max,
                    'abs': abs,
                    'round': round,
                    'list': list,
                    'dict': dict,
                    'set': set,
                    'tuple': tuple,
                    'str': str,
                    'int': int,
                    'float': float,
                    'bool': bool,
                    'True': True,
                    'False': False,
                    'None': None,
                    '__build_class__': builtins.__build_class__,
                    '__name__': '__main__',
                }
            }
            
            exec(code, namespace)
            
            # Find the bot class
            bot_class = None
            bot_class_name = None
            for name, obj in namespace.items():
                if isinstance(obj, type) and name not in ['__builtins__']:
                    bot_class = obj
                    bot_class_name = name
                    break
            
            if not bot_class:
                return False, "No class definition found in code", None
            
            # Check for required methods
            for method in BotExecutionService.REQUIRED_METHODS:
                if not hasattr(bot_class, method):
                    return False, f"Bot class must implement '{method}' method", None
            
            return True, None, bot_class_name
            
        except Exception as e:
            return False, f"Validation error: {str(e)}", None
    
    @staticmethod
    def _get_player_cache_path(season, is_current=False):
        """Get the cache file path for player histories"""
        os.makedirs(BotExecutionService.CACHE_DIR, exist_ok=True)
        cache_type = 'current' if is_current else 'historical'
        filename = f"player_histories_{season}_{cache_type}.json"
        return os.path.join(BotExecutionService.CACHE_DIR, filename)
    
    @staticmethod
    def _is_cache_valid(cache_path, is_current=False):
        """Check if cached data is still valid"""
        if not os.path.exists(cache_path):
            return False
        
        # Historical seasons never expire
        if not is_current:
            return True
        
        # Current season expires after CACHE_EXPIRY_HOURS
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        age = datetime.now() - cache_time
        return age < timedelta(hours=BotExecutionService.CACHE_EXPIRY_HOURS)
    
    @staticmethod
    def _load_cached_player_histories(season, is_current=False):
        """Load player histories from cache"""
        cache_path = BotExecutionService._get_player_cache_path(season, is_current)
        
        if not BotExecutionService._is_cache_valid(cache_path, is_current):
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Convert string keys back to integers (JSON converts int keys to strings)
                converted_data = {}
                for player_id_str, gw_data in data.items():
                    player_id = int(player_id_str)
                    # Also convert gameweek keys to integers
                    converted_gw_data = {}
                    if isinstance(gw_data, dict):
                        for gw_str, stats in gw_data.items():
                            converted_gw_data[int(gw_str)] = stats
                    converted_data[player_id] = converted_gw_data
                
                print(f"✓ Loaded player histories from cache ({season})")
                return converted_data
        except Exception as e:
            print(f"  Warning: Could not load cache: {e}")
            return None
    
    @staticmethod
    def _save_cached_player_histories(player_histories, season, is_current=False):
        """Save player histories to cache"""
        cache_path = BotExecutionService._get_player_cache_path(season, is_current)
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(player_histories, f)
                print(f"✓ Saved player histories to cache ({season})")
        except Exception as e:
            print(f"  Warning: Could not save cache: {e}")
    
    @staticmethod
    def create_bot_submission(user_id, bot_name, bot_description, bot_code):
        """
        Create a new bot submission
        Returns (submission, error_message)
        """
        try:
            # Validate the bot code
            is_valid, error_msg, bot_class_name = BotExecutionService.validate_bot_code(bot_code)
            
            # Create submission
            submission = BotSubmission(
                user_id=user_id,
                bot_name=bot_name,
                bot_description=bot_description,
                bot_code=bot_code,
                is_validated=is_valid,
                validation_error=error_msg
            )
            
            db.session.add(submission)
            db.session.commit()
            
            return submission, None
            
        except Exception as e:
            db.session.rollback()
            return None, str(e)
    
    @staticmethod
    def run_backtest(bot_submission_id, season="2023-24", start_gw=1, end_gw=38, use_current_season=False):
        """
        Run a backtest of a bot on historical data or current season
        
        Args:
            bot_submission_id: ID of the bot submission
            season: Season string (e.g., "2023-24" or "current")
            start_gw: Starting gameweek
            end_gw: Ending gameweek
            use_current_season: If True, fetch real data from FPL API
            
        Returns (backtest_result, error_message)
        """
        submission = BotSubmission.query.get(bot_submission_id)
        if not submission:
            return None, "Bot submission not found"
        
        if not submission.is_validated:
            return None, "Bot has not been validated"
        
        # If using current season, fetch real data from API
        if use_current_season or season.lower() == "current":
            from fpl_platform.services.fpl_data_service import FPLDataService
            season_data = FPLDataService.fetch_current_season_data()
            
            if not season_data:
                return None, "Failed to fetch current season data from FPL API"
            
            # Determine actual end gameweek (last completed GW)
            actual_end_gw = season_data['current_gameweek']
            if end_gw > actual_end_gw:
                end_gw = actual_end_gw
            
            season = f"2024-25 (Current)"
        else:
            season_data = None
        
        # Create backtest result record
        backtest = BacktestResult(
            bot_submission_id=bot_submission_id,
            user_id=submission.user_id,
            season=season,
            start_gameweek=start_gw,
            end_gameweek=end_gw,
            status='running'
        )
        db.session.add(backtest)
        db.session.commit()
        
        try:
            start_time = time.time()
            
            # Execute bot code in sandbox
            namespace = BotExecutionService._create_sandbox_namespace()
            exec(submission.bot_code, namespace)
            
            # Find and instantiate the bot class
            bot_class = None
            for name, obj in namespace.items():
                if isinstance(obj, type) and name not in ['__builtins__']:
                    bot_class = obj
                    break
            
            if not bot_class:
                raise Exception("Bot class not found after execution")
            
            # For now, we'll create mock backtest results
            # In a real implementation, you would:
            # 1. Load historical data for the specified season
            # 2. Simulate each gameweek
            # 3. Call bot.analyze() and bot.optimize() for each GW
            # 4. Track transfers, points, etc.
            
            gameweek_results = []
            total_points = 0
            transfers_made = 0
            hits_taken = 0
            
            # Instantiate the bot
            bot_instance = bot_class()
            
            # Initialize starting squad (GW1 team)
            if season_data:
                # Use real players from current season API
                all_players = season_data['players']
                teams_data = season_data['teams']
                
                # Create a starting squad using cost (market expectations)
                # CRITICAL: Cannot use total_points as that includes future performance data!
                # Cost represents market expectation of quality (expensive = expected to be good)
                # This is fair: no data leakage, just using pre-season market pricing
                sorted_players = sorted(
                    all_players.values(), 
                    key=lambda x: x['cost'],  # Sort by cost (expensive = premium players)
                    reverse=True
                )
                
                # Pick 15 players: 2 GKP, 5 DEF, 5 MID, 3 FWD
                # This simulates a typical pre-season squad (popular picks)
                starting_squad_ids = []
                position_counts = {'GKP': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
                position_targets = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
                
                for player in sorted_players:
                    pos = player['position']
                    if position_counts[pos] < position_targets[pos]:
                        starting_squad_ids.append(player['id'])
                        position_counts[pos] += 1
                        if len(starting_squad_ids) == 15:
                            break
                
                current_squad = [
                    {
                        'id': pid,
                        'name': all_players[pid]['name'],
                        'full_name': all_players[pid]['full_name'],
                        'position': all_players[pid]['position'],
                        'predicted_points': 5.0,  # Neutral prediction - bot must use own logic
                        'cost': all_players[pid]['cost'],
                        'team': teams_data[all_players[pid]['team']]['short_name'] if all_players[pid]['team'] in teams_data else 'UNK',
                        'points_so_far': 0  # Will be updated as season progresses
                    }
                    for pid in starting_squad_ids
                ]
            else:
                # Use default simulated squad
                player_names = [
                    'Raya', 'Saliba', 'Gabriel', 'Trippier', 'Saka', 'Odegaard', 
                    'Bruno Fernandes', 'Son', 'Haaland', 'Watkins', 'Isak',
                    'Steele', 'Ait-Nouri', 'Andreas', 'Archer'
                ]
                positions = ['GKP', 'DEF', 'DEF', 'DEF', 'MID', 'MID', 'MID', 'MID', 
                            'FWD', 'FWD', 'FWD', 'GKP', 'DEF', 'MID', 'FWD']
                
                current_squad = [
                    {
                        'id': i + 1, 
                        'name': player_names[i],
                        'position': positions[i],
                        'predicted_points': 5.0 + (i % 3), 
                        'cost': 5.0 + i * 0.5,
                        'team': ['ARS', 'ARS', 'ARS', 'NEW', 'ARS', 'ARS', 'MUN', 'TOT', 
                                'MCI', 'AVL', 'NEW', 'BRI', 'WOL', 'FUL', 'AVL'][i]
                    }
                    for i in range(15)
                ]
            
            bank = 0.5
            free_transfers = 1
            
            # Initialize chips (all available at start)
            chips_available = ['wildcard_1', 'bench_boost', 'free_hit', 'triple_captain']
            chips_used = []  # Track when chips are used: {chip: gameweek}
            
            # Cache player histories for current season (for real stats)
            # Fetch ALL players - let temporal filtering provide the security
            player_histories = {}
            if season_data:
                from fpl_platform.services.fpl_data_service import FPLDataService
                
                # Check if season string indicates current season
                is_current_season = 'current' in season.lower() or season == '2024-25'
                cache_season = season.replace(' (Current)', '').strip()
                
                # Try to load from cache first
                print(f"Checking cache for player histories ({cache_season})...")
                player_histories = BotExecutionService._load_cached_player_histories(
                    cache_season, 
                    is_current=is_current_season
                )
                
                # If cache miss or expired, fetch from API
                if player_histories is None:
                    player_histories = {}
                    
                    # Fetch histories for ALL available players
                    # This ensures complete player pool for all bots (fair competition)
                    # Temporal filtering prevents data leakage regardless of pool size
                    total_players = len(all_players)
                    print(f"Fetching player histories from FPL API for all {total_players} players...")
                    print(f"  (This may take a few minutes but will be cached for future backtests)")
                    
                    fetched_count = 0
                    for pid in all_players.keys():
                        try:
                            history = FPLDataService.fetch_player_history(pid)
                            if history and 'history' in history:
                                # Index by gameweek for quick lookup
                                player_histories[pid] = {
                                    gw_data['round']: gw_data 
                                    for gw_data in history['history']
                                }
                            else:
                                # Player has no history (hasn't played)
                                player_histories[pid] = {}
                            
                            fetched_count += 1
                            
                            # Progress reporting every 50 players
                            if fetched_count % 50 == 0:
                                print(f"  Progress: {fetched_count}/{total_players} players...")
                                
                        except Exception as e:
                            print(f"Could not fetch history for player {pid}: {e}")
                            player_histories[pid] = {}
                    
                    print(f"✓ Fetched histories for all {fetched_count} players successfully!")
                    
                    # Save to cache for next time
                    BotExecutionService._save_cached_player_histories(
                        player_histories,
                        cache_season,
                        is_current=is_current_season
                    )
                
                print(f"  (Complete player pool with temporal filtering)")
            
            # Simulate gameweeks (simplified)
            for gw in range(start_gw, end_gw + 1):
                # ============================================================
                # TEMPORAL FILTERING - CRITICAL SECURITY MECHANISM
                # ============================================================
                # Update ALL player data with stats from COMPLETED gameweeks only
                # This ensures NO DATA LEAKAGE - bot only sees past performance
                # Bot is predicting GW=gw, so it can ONLY see data from GW 1 to (gw-1)
                # ============================================================
                
                # First, update current squad players
                for player in current_squad:
                    pid = player['id']
                    
                    # Calculate cumulative stats from GW1 to (current_gw - 1)
                    if pid in player_histories:
                        total_points_so_far = 0
                        games_played = 0
                        
                        for past_gw in range(start_gw, gw):  # Only past gameweeks
                            if past_gw in player_histories[pid]:
                                gw_stats = player_histories[pid][past_gw]
                                total_points_so_far += gw_stats.get('total_points', 0)
                                if gw_stats.get('minutes', 0) > 0:
                                    games_played += 1
                        
                        # Update player with historical data only
                        player['points_so_far'] = total_points_so_far
                        player['games_played'] = games_played
                        
                        # Calculate form (last 5 games average) - only from past data
                        recent_gws = range(max(start_gw, gw - 5), gw)
                        recent_points = []
                        for recent_gw in recent_gws:
                            if recent_gw in player_histories[pid]:
                                recent_points.append(player_histories[pid][recent_gw].get('total_points', 0))
                        
                        player['predicted_points'] = sum(recent_points) / len(recent_points) if recent_points else 5.0
                    else:
                        # No history available - use neutral prediction
                        player['predicted_points'] = 5.0
                
                # Prepare available players pool with temporal filtering
                # Bot can see ALL players, but only with data up to (gw-1)
                available_players_list = []
                if season_data:
                    current_squad_ids = [p['id'] for p in current_squad]
                    
                    for pid, player_data in all_players.items():
                        # Calculate this player's stats from PAST gameweeks only (no leakage!)
                        total_points_so_far = 0
                        games_played = 0
                        recent_points = []
                        
                        if pid in player_histories:
                            # Only look at completed gameweeks (gw-1 and before)
                            for past_gw in range(start_gw, gw):  # STOPS BEFORE CURRENT GW
                                if past_gw in player_histories[pid]:
                                    gw_stats = player_histories[pid][past_gw]
                                    total_points_so_far += gw_stats.get('total_points', 0)
                                    if gw_stats.get('minutes', 0) > 0:
                                        games_played += 1
                            
                            # Form from last 5 completed games only
                            recent_gws = range(max(start_gw, gw - 5), gw)  # STOPS BEFORE CURRENT GW
                            for recent_gw in recent_gws:
                                if recent_gw in player_histories[pid]:
                                    recent_points.append(player_histories[pid][recent_gw].get('total_points', 0))
                        
                        form_score = sum(recent_points) / len(recent_points) if recent_points else 5.0
                        
                        # Add player to available pool with temporal data only
                        available_players_list.append({
                            'id': pid,
                            'name': player_data['name'],
                            'full_name': player_data.get('full_name', player_data['name']),
                            'position': player_data['position'],
                            'cost': player_data['cost'],
                            'team': teams_data[player_data['team']]['short_name'] if player_data['team'] in teams_data else 'UNK',
                            'points_so_far': total_points_so_far,  # Only up to (gw-1)
                            'games_played': games_played,
                            'predicted_points': form_score,  # Based only on past performance
                            'in_squad': pid in current_squad_ids
                        })
                
                # Get bot's team selection for this gameweek
                try:
                    # ============================================================
                    # DATA SANITIZATION - CRITICAL SECURITY MECHANISM
                    # ============================================================
                    # SECURITY: Platform is the ONLY source of data for bot
                    # Bot receives ONLY pre-computed, temporally-filtered statistics
                    # Bot has NO access to player_histories or raw API data
                    # All data below has been verified to contain ONLY past gameweek info
                    # ============================================================
                    
                    # Whitelist allowed fields (security hardening)
                    ALLOWED_FIELDS = {
                        'id', 'name', 'full_name', 'position', 'cost', 'team',
                        'points_so_far', 'games_played', 'predicted_points', 'in_squad'
                    }
                    
                    # Sanitize current team data
                    sanitized_current_team = [
                        {k: v for k, v in player.items() if k in ALLOWED_FIELDS}
                        for player in current_squad
                    ]
                    
                    # Sanitize available players data
                    sanitized_available_players = [
                        {k: v for k, v in player.items() if k in ALLOWED_FIELDS}
                        for player in available_players_list
                    ] if available_players_list else []
                    
                    # Create immutable team data dictionary
                    import copy
                    mock_team_data = copy.deepcopy({
                        'current_team': sanitized_current_team,
                        'available_players': sanitized_available_players,
                        'gameweek': gw,
                        'bank': float(bank),
                        'free_transfers': int(free_transfers),
                        'chips_available': list(chips_available)  # Available chips
                    })
                    
                    # Debug: Log data summary for first few gameweeks
                    if gw <= 3:
                        avg_predicted = sum(p['predicted_points'] for p in sanitized_current_team) / len(sanitized_current_team)
                        print(f"  GW{gw} Data Check: Current team avg predicted_points: {avg_predicted:.2f}")
                        if sanitized_available_players:
                            top_5 = sorted(sanitized_available_players, key=lambda x: x['predicted_points'], reverse=True)[:5]
                            print(f"  GW{gw} Top 5 available: {[(p['name'], p['predicted_points']) for p in top_5]}")
                    
                    # Call bot's optimize method with sanitized data
                    # Bot can ONLY access what's in this dictionary
                    optimization = bot_instance.optimize(mock_team_data)
                    
                    captain_id = optimization.get('captain', 1)
                    vice_captain_id = optimization.get('vice_captain', 2)
                    lineup_ids = optimization.get('lineup', list(range(1, 12)))
                    
                    # Check if bot wants to use a chip
                    chip_to_use = optimization.get('chip', None)
                    chip_used_this_week = None
                    
                    # Validate and process chip usage
                    if chip_to_use:
                        # Normalize chip name (handle different formats)
                        chip_mapping = {
                            'wildcard': 'wildcard_1',
                            'wildcard_1': 'wildcard_1',
                            'wildcard1': 'wildcard_1',
                            'bench_boost': 'bench_boost',
                            'benchboost': 'bench_boost',
                            'bb': 'bench_boost',
                            'free_hit': 'free_hit',
                            'freehit': 'free_hit',
                            'fh': 'free_hit',
                            'triple_captain': 'triple_captain',
                            'triplecaptain': 'triple_captain',
                            'tc': 'triple_captain'
                        }
                        
                        chip_normalized = chip_mapping.get(chip_to_use.lower(), None)
                        
                        # Check if chip is available
                        if chip_normalized and chip_normalized in chips_available:
                            chip_used_this_week = chip_normalized
                            chips_available.remove(chip_normalized)
                            chips_used.append({'chip': chip_normalized, 'gameweek': gw})
                            
                            # Add second wildcard after GW20 if first was used
                            if chip_normalized == 'wildcard_1' and gw >= 20 and 'wildcard_2' not in chips_available:
                                chips_available.append('wildcard_2')
                        elif chip_normalized:
                            print(f"  Warning: Bot tried to use {chip_normalized} but it's not available")
                    
                    # Handle chip-specific logic
                    free_transfers_for_transfers = free_transfers
                    if chip_used_this_week == 'wildcard_1' or chip_used_this_week == 'wildcard_2':
                        # Wildcard: unlimited free transfers
                        free_transfers_for_transfers = 99  # Effectively unlimited
                    elif chip_used_this_week == 'free_hit':
                        # Free hit: unlimited transfers but team reverts next week
                        free_transfers_for_transfers = 99
                        # Save team before free hit for restoration
                        team_before_free_hit = [p.copy() for p in current_squad]
                    
                    # Intelligent transfer system - make strategic transfers after GW1
                    transfers_this_week = []
                    num_transfers = 0
                    
                    # Only make transfers after GW2 and if we have free transfers (or chip active)
                    if gw > start_gw + 1 and (free_transfers_for_transfers > 0) and season_data:
                        # Identify underperforming players in squad (those not in lineup or low form)
                        bench_ids = [p['id'] for p in current_squad if p['id'] not in lineup_ids]
                        
                        # Find worst performing bench player(s)
                        bench_players = [p for p in current_squad if p['id'] in bench_ids]
                        bench_players.sort(key=lambda x: x.get('predicted_points', 0))
                        
                        # Make transfers when there's a clear upgrade available
                        # Only use 1 free transfer per week (unless chip active)
                        if True:  # Remove 3-week restriction
                            # Transfer out worst bench player
                            worst_player = bench_players[0] if bench_players else None
                            
                            if worst_player and free_transfers_for_transfers >= 1:
                                # Find a better replacement from real available players
                                position_needed = worst_player['position']
                                current_squad_ids = [p['id'] for p in current_squad]
                                available_budget = bank + worst_player['cost']
                                
                                # Get all players of same position not in squad
                                available_players = []
                                for pid, player_data in all_players.items():
                                    if (player_data['position'] == position_needed and 
                                        pid not in current_squad_ids and
                                        player_data['cost'] <= available_budget):
                                        
                                        # Calculate this player's form from history
                                        form_score = 5.0  # Default
                                        if pid in player_histories:
                                            recent_points = []
                                            for past_gw in range(max(start_gw, gw - 5), gw):
                                                if past_gw in player_histories[pid]:
                                                    recent_points.append(player_histories[pid][past_gw].get('total_points', 0))
                                            if recent_points:
                                                form_score = sum(recent_points) / len(recent_points)
                                        
                                        available_players.append({
                                            'id': pid,
                                            'name': player_data['name'],
                                            'cost': player_data['cost'],
                                            'form': form_score
                                        })
                                
                                # Sort by form and find best upgrade
                                available_players.sort(key=lambda x: x['form'], reverse=True)
                                
                                # Find a player significantly better than current (at least 1.0 pts improvement)
                                # This prevents constant tinkering for marginal gains
                                best_replacement = None
                                min_improvement = 1.0  # Minimum predicted points improvement to justify transfer
                                for candidate in available_players:
                                    improvement = candidate['form'] - worst_player.get('predicted_points', 0)
                                    if improvement >= min_improvement:
                                        best_replacement = candidate
                                        break
                                
                                if best_replacement:
                                    # Apply transfer with real player
                                    old_name = worst_player['name']
                                    new_player_id = best_replacement['id']
                                    new_player_name = best_replacement['name']
                                    new_cost = best_replacement['cost']
                                    
                                    for i, player in enumerate(current_squad):
                                        if player['id'] == worst_player['id']:
                                            current_squad[i] = {
                                                'id': new_player_id,
                                                'name': new_player_name,
                                                'position': position_needed,
                                                'predicted_points': best_replacement['form'],
                                                'cost': new_cost,
                                                'team': all_players[new_player_id]['team'],
                                                'points_so_far': 0,
                                                'games_played': 0
                                            }
                                            break
                                    
                                    transfers_this_week.append({
                                        'out': old_name,
                                        'in': new_player_name
                                    })
                                    num_transfers = 1
                                    
                                    # Adjust bank for transfer
                                    bank = bank + worst_player['cost'] - new_cost
                    
                    # Get player names for display
                    captain_name = next((p['name'] for p in current_squad if p['id'] == captain_id), 'Unknown')
                    vice_captain_name = next((p['name'] for p in current_squad if p['id'] == vice_captain_id), 'Unknown')
                    
                    # Enhanced lineup with detailed stats
                    lineup = []
                    bench = []
                    
                    for p in current_squad:
                        player_id = p.get('id')
                        position = p.get('position', 'MID')
                        
                        # Try to use real stats from cached player histories
                        real_stats = None
                        if player_id in player_histories and gw in player_histories[player_id]:
                            real_stats = player_histories[player_id][gw]
                        
                        if real_stats:
                            # Use actual FPL API stats
                            minutes = real_stats.get('minutes', 0)
                            goals = real_stats.get('goals_scored', 0)
                            assists = real_stats.get('assists', 0)
                            clean_sheet = real_stats.get('clean_sheets', 0)
                            saves = real_stats.get('saves', 0)
                            bonus = real_stats.get('bonus', 0)
                            points = real_stats.get('total_points', 0)
                        else:
                            # Fall back to simulated stats
                            base_points = p.get('predicted_points', 0)
                            
                            if position == 'GKP':
                                minutes = 90
                                goals = 0
                                assists = 0
                                clean_sheet = 1 if base_points > 4 else 0
                                saves = int(base_points * 2) if base_points > 0 else 0
                                points = int(2 + (4 * clean_sheet) + (saves // 3))  # GK scoring
                                bonus = 1 if points >= 6 else 0
                            elif position == 'DEF':
                                minutes = 90 if base_points > 3 else 60
                                goals = 1 if base_points > 10 else 0
                                assists = 1 if base_points > 8 and goals == 0 else 0
                                clean_sheet = 1 if base_points > 5 else 0
                                saves = 0
                                points = int(2 + (6 * goals) + (3 * assists) + (4 * clean_sheet))
                                bonus = int(base_points / 8) if points >= 6 else 0
                            elif position == 'MID':
                                minutes = 90 if base_points > 4 else 70
                                goals = 1 if base_points > 9 else 0
                                assists = 1 if base_points > 7 and goals == 0 else 0
                                clean_sheet = 1 if base_points > 6 else 0
                                saves = 0
                                points = int(2 + (5 * goals) + (3 * assists) + (1 * clean_sheet))
                                bonus = int(base_points / 7) if points >= 6 else 0
                            else:  # FWD
                                minutes = 90 if base_points > 5 else 75
                                goals = int(base_points / 5) if base_points > 4 else 0
                                assists = 1 if base_points > 7 and goals < 2 else 0
                                clean_sheet = 0
                                saves = 0
                                points = int(2 + (4 * goals) + (3 * assists))
                                bonus = int(base_points / 6) if points >= 8 else 0
                            
                            # Add bonus to simulated points
                            points = min(15, points + bonus)  # Cap at 15 to be realistic
                        
                        # Apply captain multiplier
                        if p['id'] == captain_id:
                            if chip_used_this_week == 'triple_captain':
                                points = points * 3  # Triple captain chip
                            else:
                                points = points * 2  # Normal captain
                        
                        player_stats = {
                            'id': p['id'],
                            'name': p['name'],
                            'position': p['position'],
                            'minutes': minutes,
                            'points': points,
                            'goals': goals,
                            'assists': assists,
                            'clean_sheet': clean_sheet,
                            'saves': saves,
                            'bonus': bonus
                        }
                        
                        if p['id'] in lineup_ids:
                            lineup.append(player_stats)
                        else:
                            bench.append(player_stats)
                    
                    # Calculate gameweek points from lineup
                    gw_points = sum(player['points'] for player in lineup)
                    
                    # Bench Boost: add bench points too
                    if chip_used_this_week == 'bench_boost':
                        bench_points = sum(player['points'] for player in bench)
                        gw_points += bench_points
                    
                    # Calculate hits (transfers beyond free transfers)
                    free_transfers_used = min(num_transfers, free_transfers)
                    hits_this_week = max(0, num_transfers - free_transfers_used)
                    
                    if hits_this_week > 0:
                        gw_points -= hits_this_week * 4  # Deduct points for hits
                    
                    hits_taken += hits_this_week
                    transfers_made += num_transfers
                    total_points += gw_points
                    
                    # Update free transfers for next week
                    if chip_used_this_week == 'free_hit':
                        # Free Hit: team reverts, free transfers unchanged
                        current_squad = team_before_free_hit
                        # Free transfers stay the same
                    elif chip_used_this_week in ['wildcard_1', 'wildcard_2']:
                        # Wildcard: reset to 1 free transfer
                        free_transfers = 1
                    elif num_transfers == 0:
                        free_transfers = min(2, free_transfers + 1)  # Accumulate up to 2
                    else:
                        free_transfers = 1  # Reset to 1 after making transfers
                    
                except Exception as e:
                    # If bot fails, use defaults
                    captain_name = current_squad[0]['name']
                    vice_captain_name = current_squad[1]['name']
                    lineup = [{'id': p['id'], 'name': p['name'], 'position': p['position']} 
                             for p in current_squad[:11]]
                    bench = [{'id': p['id'], 'name': p['name'], 'position': p['position']} 
                            for p in current_squad[11:]]
                    transfers_this_week = []
                    num_transfers = 0
                    hits_this_week = 0
                
                gameweek_results.append({
                    'gameweek': gw,
                    'points': gw_points,
                    'total_points': total_points,
                    'transfers': num_transfers,
                    'hits': hits_this_week,
                    'captain': captain_name,
                    'vice_captain': vice_captain_name,
                    'lineup': lineup,
                    'bench': bench,
                    'transfers_made': transfers_this_week,
                    'chip_used': chip_used_this_week  # Track chip usage
                })
            
            execution_time = time.time() - start_time
            
            # Estimate FPL rank based on total points and gameweeks played
            estimated_rank = _estimate_fpl_rank(total_points, end_gw - start_gw + 1, season)
            
            # Update backtest with results
            backtest.status = 'completed'
            backtest.total_points = total_points
            backtest.final_rank = estimated_rank
            backtest.team_value = 100.0 + (total_points / 100)
            backtest.transfers_made = transfers_made
            backtest.hits_taken = hits_taken
            backtest.chips_used = json.dumps(chips_used)  # Store chip usage
            backtest.execution_time_seconds = execution_time
            backtest.set_gameweek_results(gameweek_results)
            backtest.completed_at = datetime.utcnow()
            
            # Calculate metrics
            backtest.average_points_per_gw = total_points / (end_gw - start_gw + 1)
            backtest.best_gameweek = max(gameweek_results, key=lambda x: x['points'])['gameweek']
            backtest.best_gameweek_points = max(gameweek_results, key=lambda x: x['points'])['points']
            backtest.worst_gameweek = min(gameweek_results, key=lambda x: x['points'])['gameweek']
            backtest.worst_gameweek_points = min(gameweek_results, key=lambda x: x['points'])['points']
            
            db.session.commit()
            
            return backtest, None
            
        except Exception as e:
            backtest.status = 'failed'
            backtest.error_message = str(e)
            backtest.completed_at = datetime.utcnow()
            db.session.commit()
            return backtest, str(e)
    
    @staticmethod
    def activate_bot_for_season(user_id, bot_submission_id, season="2024-25", auto_backtest=True):
        """
        Activate a bot for the current season competition
        
        Args:
            user_id: User ID
            bot_submission_id: Bot submission ID
            season: Season string (e.g., "2024-25")
            auto_backtest: If True, automatically run backtest from GW1 to current GW
            
        Returns (season_entry, error_message)
        """
        submission = BotSubmission.query.get(bot_submission_id)
        if not submission:
            return None, "Bot submission not found"
        
        if submission.user_id != user_id:
            return None, "Unauthorized"
        
        if not submission.is_validated:
            return None, "Bot has not been validated"
        
        try:
            # Fetch current gameweek from FPL API
            from fpl_platform.services.fpl_data_service import FPLDataService
            season_data = FPLDataService.fetch_current_season_data()
            
            if season_data:
                current_gw = season_data['current_gameweek']
            else:
                # Fallback to GW1 if API fails
                current_gw = 1
                auto_backtest = False  # Can't backtest without API data
            
            # Check if user already has an active entry
            existing_entry = SeasonEntry.query.filter_by(user_id=user_id).first()
            if existing_entry:
                # Deactivate old bot
                old_bot = BotSubmission.query.get(existing_entry.bot_submission_id)
                if old_bot:
                    old_bot.is_active = False
                
                # Update to new bot
                existing_entry.bot_submission_id = bot_submission_id
                existing_entry.season = season
                existing_entry.entry_gameweek = current_gw
                existing_entry.current_gameweek = current_gw
                
                # Reset stats for new bot
                existing_entry.total_points = 0
                existing_entry.total_transfers = 0
                existing_entry.total_hits = 0
                existing_entry.team_value = 100.0
                existing_entry.bank = 0.0
                existing_entry.gameweek_history = None
                existing_entry.current_squad = None
            else:
                # Create new entry
                existing_entry = SeasonEntry(
                    user_id=user_id,
                    bot_submission_id=bot_submission_id,
                    season=season,
                    entry_gameweek=current_gw,
                    current_gameweek=current_gw,
                    total_points=0,
                    team_value=100.0,
                    bank=0.0
                )
                db.session.add(existing_entry)
            
            # Set bot as active
            submission.is_active = True
            
            # Commit the entry creation first
            db.session.commit()
            
            # Auto-backtest if enabled and we're past GW1
            if auto_backtest and current_gw > 1:
                print(f"\n🤖 Running automatic backtest from GW1 to GW{current_gw} for bot activation...")
                print(f"   This will populate your entry with historical performance data.\n")
                
                # Run backtest synchronously during activation
                backtest_result = BotExecutionService._run_season_backtest_for_entry(
                    existing_entry.id,
                    bot_submission_id,
                    start_gw=1,
                    end_gw=current_gw,
                    season_data=season_data
                )
                
                if backtest_result:
                    print(f"✓ Backtest complete! Your bot scored {existing_entry.total_points} points.")
                else:
                    print("⚠ Backtest encountered issues. Entry created with 0 points.")
            
            # Calculate initial rank based on leaderboard position
            all_entries = SeasonEntry.query.filter_by(
                season=season,
                is_active=True
            ).order_by(SeasonEntry.total_points.desc()).all()
            
            for rank, entry in enumerate(all_entries, start=1):
                entry.current_rank = rank
            
            db.session.commit()
            
            return existing_entry, None
            
        except Exception as e:
            db.session.rollback()
            return None, str(e)
    
    @staticmethod
    def get_leaderboard(season="2024-25", limit=100):
        """
        Get competition leaderboard
        Returns list of entries sorted by total points
        """
        entries = SeasonEntry.query.filter_by(
            season=season,
            is_active=True
        ).order_by(
            SeasonEntry.total_points.desc()
        ).limit(limit).all()
        
        # Update ranks in database for consistency
        all_entries = SeasonEntry.query.filter_by(
            season=season,
            is_active=True
        ).order_by(SeasonEntry.total_points.desc()).all()
        
        for rank, entry in enumerate(all_entries, start=1):
            if entry.current_rank != rank:
                entry.current_rank = rank
        
        db.session.commit()
        
        # Add rank and user/bot info
        leaderboard = []
        for rank, entry in enumerate(entries, start=1):
            entry_dict = entry.to_dict()
            entry_dict['rank'] = rank
            
            # Add user info
            from fpl_platform.models import User
            user = User.query.get(entry.user_id)
            entry_dict['username'] = user.username if user else "Unknown"
            
            # Add bot info
            bot = BotSubmission.query.get(entry.bot_submission_id)
            entry_dict['bot_name'] = bot.bot_name if bot else "Unknown"
            
            leaderboard.append(entry_dict)
        
        return leaderboard
    
    @staticmethod
    def _run_season_backtest_for_entry(entry_id, bot_submission_id, start_gw, end_gw, season_data):
        """
        Run backtest and directly update a SeasonEntry with results
        
        This is used during bot activation to populate the entry with historical data.
        Unlike run_backtest(), this updates the SeasonEntry directly instead of creating
        a BacktestResult record.
        
        Args:
            entry_id: SeasonEntry ID to update
            bot_submission_id: Bot submission ID
            start_gw: Starting gameweek
            end_gw: Ending gameweek (current gameweek)
            season_data: Current season data from FPL API
            
        Returns:
            True if successful, False otherwise
        """
        try:
            entry = SeasonEntry.query.get(entry_id)
            submission = BotSubmission.query.get(bot_submission_id)
            
            if not entry or not submission:
                return False
            
            # Execute bot code
            namespace = BotExecutionService._create_sandbox_namespace()
            exec(submission.bot_code, namespace)
            
            # Find bot class
            bot_class = None
            for name, obj in namespace.items():
                if isinstance(obj, type) and name not in ['__builtins__']:
                    bot_class = obj
                    break
            
            if not bot_class:
                return False
            
            bot_instance = bot_class()
            
            # Get players and teams from season data
            all_players = season_data['players']
            teams_data = season_data['teams']
            
            # Initialize starting squad (same logic as run_backtest)
            sorted_players = sorted(
                all_players.values(), 
                key=lambda x: x['cost'],
                reverse=True
            )
            
            starting_squad_ids = []
            position_counts = {'GKP': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
            position_targets = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
            
            for player in sorted_players:
                pos = player['position']
                if position_counts[pos] < position_targets[pos]:
                    starting_squad_ids.append(player['id'])
                    position_counts[pos] += 1
                    if len(starting_squad_ids) == 15:
                        break
            
            current_squad = [
                {
                    'id': pid,
                    'name': all_players[pid]['name'],
                    'full_name': all_players[pid]['full_name'],
                    'position': all_players[pid]['position'],
                    'predicted_points': 5.0,
                    'cost': all_players[pid]['cost'],
                    'team': teams_data[all_players[pid]['team']]['short_name'] if all_players[pid]['team'] in teams_data else 'UNK',
                    'points_so_far': 0
                }
                for pid in starting_squad_ids
            ]
            
            bank = 0.5
            free_transfers = 1
            
            # Initialize chips
            chips_available = ['wildcard_1', 'bench_boost', 'free_hit', 'triple_captain']
            chips_used = []
            
            # Fetch player histories for all players (with caching)
            from fpl_platform.services.fpl_data_service import FPLDataService
            
            # Try to load from cache first (always current season for activation)
            print(f"  Checking cache for player histories...")
            player_histories = BotExecutionService._load_cached_player_histories(
                '2024-25',
                is_current=True
            )
            
            # If cache miss, fetch from API
            if player_histories is None:
                player_histories = {}
                
                print(f"  Fetching player histories from FPL API for {len(all_players)} players...")
                print(f"    (This will be cached for future activations)")
                fetched_count = 0
                for pid in all_players.keys():
                    try:
                        history = FPLDataService.fetch_player_history(pid)
                        if history and 'history' in history:
                            player_histories[pid] = {
                                gw_data['round']: gw_data 
                                for gw_data in history['history']
                            }
                        else:
                            player_histories[pid] = {}
                        
                        fetched_count += 1
                        if fetched_count % 100 == 0:
                            print(f"    Progress: {fetched_count}/{len(all_players)} players...")
                            
                    except Exception as e:
                        player_histories[pid] = {}
                
                print(f"  ✓ Fetched player data for {fetched_count} players")
                
                # Save to cache
                BotExecutionService._save_cached_player_histories(
                    player_histories,
                    '2024-25',
                    is_current=True
                )
            
            # Simulate gameweeks
            gameweek_history = []
            total_points = 0
            total_transfers = 0
            total_hits = 0
            
            for gw in range(start_gw, end_gw + 1):
                # ============================================================
                # TEMPORAL FILTERING - CRITICAL SECURITY MECHANISM
                # ============================================================
                # Update player stats with temporal filtering (same logic as run_backtest)
                # Bot is predicting GW=gw, so it can ONLY see data from GW 1 to (gw-1)
                # ============================================================
                for player in current_squad:
                    pid = player['id']
                    if pid in player_histories:
                        total_points_so_far = 0
                        games_played = 0
                        
                        for past_gw in range(start_gw, gw):
                            if past_gw in player_histories[pid]:
                                gw_stats = player_histories[pid][past_gw]
                                total_points_so_far += gw_stats.get('total_points', 0)
                                if gw_stats.get('minutes', 0) > 0:
                                    games_played += 1
                        
                        player['points_so_far'] = total_points_so_far
                        player['games_played'] = games_played
                        
                        recent_gws = range(max(start_gw, gw - 5), gw)
                        recent_points = []
                        for recent_gw in recent_gws:
                            if recent_gw in player_histories[pid]:
                                recent_points.append(player_histories[pid][recent_gw].get('total_points', 0))
                        
                        player['predicted_points'] = sum(recent_points) / len(recent_points) if recent_points else 5.0
                    else:
                        player['predicted_points'] = 5.0
                
                # Prepare available players pool with temporal filtering
                # CONSISTENCY: Provide same data structure as regular backtest
                available_players_list = []
                current_squad_ids = [p['id'] for p in current_squad]
                
                for pid, player_data in all_players.items():
                    # Calculate this player's stats from PAST gameweeks only (no leakage!)
                    total_points_so_far = 0
                    games_played = 0
                    recent_points = []
                    
                    if pid in player_histories:
                        # Only look at completed gameweeks (gw-1 and before)
                        for past_gw in range(start_gw, gw):  # STOPS BEFORE CURRENT GW
                            if past_gw in player_histories[pid]:
                                gw_stats = player_histories[pid][past_gw]
                                total_points_so_far += gw_stats.get('total_points', 0)
                                if gw_stats.get('minutes', 0) > 0:
                                    games_played += 1
                        
                        # Form from last 5 completed games only
                        recent_gws = range(max(start_gw, gw - 5), gw)  # STOPS BEFORE CURRENT GW
                        for recent_gw in recent_gws:
                            if recent_gw in player_histories[pid]:
                                recent_points.append(player_histories[pid][recent_gw].get('total_points', 0))
                    
                    form_score = sum(recent_points) / len(recent_points) if recent_points else 5.0
                    
                    # Add player to available pool with temporal data only
                    available_players_list.append({
                        'id': pid,
                        'name': player_data['name'],
                        'full_name': player_data.get('full_name', player_data['name']),
                        'position': player_data['position'],
                        'cost': player_data['cost'],
                        'team': teams_data[player_data['team']]['short_name'] if player_data['team'] in teams_data else 'UNK',
                        'points_so_far': total_points_so_far,  # Only up to (gw-1)
                        'games_played': games_played,
                        'predicted_points': form_score,  # Based only on past performance
                        'in_squad': pid in current_squad_ids
                    })
                
                # Whitelist allowed fields (security hardening)
                ALLOWED_FIELDS = {
                    'id', 'name', 'full_name', 'position', 'cost', 'team',
                    'points_so_far', 'games_played', 'predicted_points', 'in_squad'
                }
                
                # Sanitize available players data
                sanitized_available_players = [
                    {k: v for k, v in player.items() if k in ALLOWED_FIELDS}
                    for player in available_players_list
                ]
                
                # Sanitize current team data
                sanitized_current_team = [
                    {k: v for k, v in player.items() if k in ALLOWED_FIELDS}
                    for player in current_squad
                ]
                
                # Prepare team data for bot
                import copy
                team_data = copy.deepcopy({
                    'current_team': sanitized_current_team,
                    'available_players': sanitized_available_players,  # Full player pool with temporal filtering
                    'gameweek': gw,
                    'bank': float(bank),
                    'free_transfers': int(free_transfers),
                    'chips_available': list(chips_available)
                })
                
                # Get bot's decisions
                try:
                    optimization = bot_instance.optimize(team_data)
                    captain_id = optimization.get('captain', current_squad[0]['id'])
                    lineup_ids = optimization.get('lineup', [p['id'] for p in current_squad[:11]])
                    
                    # Process chip usage
                    chip_to_use = optimization.get('chip', None)
                    chip_used_this_week = None
                    
                    if chip_to_use:
                        chip_mapping = {
                            'wildcard': 'wildcard_1', 'wildcard_1': 'wildcard_1', 'wildcard1': 'wildcard_1',
                            'bench_boost': 'bench_boost', 'benchboost': 'bench_boost', 'bb': 'bench_boost',
                            'free_hit': 'free_hit', 'freehit': 'free_hit', 'fh': 'free_hit',
                            'triple_captain': 'triple_captain', 'triplecaptain': 'triple_captain', 'tc': 'triple_captain'
                        }
                        chip_normalized = chip_mapping.get(chip_to_use.lower(), None)
                        
                        if chip_normalized and chip_normalized in chips_available:
                            chip_used_this_week = chip_normalized
                            chips_available.remove(chip_normalized)
                            chips_used.append({'chip': chip_normalized, 'gameweek': gw})
                            
                            if chip_normalized == 'wildcard_1' and gw >= 20 and 'wildcard_2' not in chips_available:
                                chips_available.append('wildcard_2')
                    
                except Exception:
                    captain_id = current_squad[0]['id']
                    lineup_ids = [p['id'] for p in current_squad[:11]]
                    chip_used_this_week = None
                
                # Calculate points for this gameweek
                gw_points = 0
                for player in current_squad:
                    if player['id'] in lineup_ids:
                        pid = player['id']
                        if pid in player_histories and gw in player_histories[pid]:
                            points = player_histories[pid][gw].get('total_points', 0)
                            if player['id'] == captain_id:
                                if chip_used_this_week == 'triple_captain':
                                    points *= 3
                                else:
                                    points *= 2
                            gw_points += points
                    elif chip_used_this_week == 'bench_boost':
                        # Bench boost: bench players also score
                        pid = player['id']
                        if pid in player_histories and gw in player_histories[pid]:
                            points = player_histories[pid][gw].get('total_points', 0)
                            gw_points += points
                
                total_points += gw_points
                
                gameweek_history.append({
                    'gameweek': gw,
                    'points': gw_points,
                    'total_points': total_points,
                    'transfers': 0,
                    'hits': 0,
                    'captain': next((p['name'] for p in current_squad if p['id'] == captain_id), 'Unknown'),
                    'chip_used': chip_used_this_week
                })
                
                # Update free transfers based on chip usage
                if chip_used_this_week in ['wildcard_1', 'wildcard_2']:
                    free_transfers = 1
                else:
                    free_transfers = min(2, free_transfers + 1)
                
                print(f"    GW{gw}: {gw_points} pts (Total: {total_points})")
            
            # Estimate FPL rank based on total points and gameweeks played
            num_gameweeks = end_gw - entry.entry_gameweek + 1
            estimated_rank = _estimate_fpl_rank(total_points, num_gameweeks, season)
            
            # Update the SeasonEntry with results
            entry.total_points = total_points
            entry.current_rank = estimated_rank
            entry.total_transfers = total_transfers
            entry.total_hits = total_hits
            entry.current_gameweek = end_gw
            entry.team_value = 100.0 + (total_points / 100)
            entry.chips_used = json.dumps(chips_used)
            entry.set_gameweek_history(gameweek_history)
            entry.set_current_squad(current_squad)
            entry.last_update = datetime.utcnow()
            
            db.session.commit()
            
            return True
            
        except Exception as e:
            print(f"  ✗ Error running season backtest: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    @staticmethod
    def _create_sandbox_namespace():
        """
        Create a restricted namespace for bot execution
        """
        # Provide safe built-ins
        import builtins
        safe_builtins = {
            'print': print,
            'len': len,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'sorted': sorted,
            'sum': sum,
            'min': min,
            'max': max,
            'abs': abs,
            'round': round,
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'True': True,
            'False': False,
            'None': None,
            '__build_class__': builtins.__build_class__,
            '__name__': '__main__',
        }
        
        return {'__builtins__': safe_builtins}


def _estimate_fpl_rank(total_points, gameweeks_played, season):
    """
    Estimate FPL rank based on total points and gameweeks played.
    
    Uses statistical models based on historical FPL data:
    - Average FPL manager scores ~60 points/GW
    - Top 10k average ~70 points/GW
    - Top 1k average ~75 points/GW
    - Overall rank 1 averages ~80+ points/GW
    
    Args:
        total_points: Total points scored
        gameweeks_played: Number of gameweeks completed
        season: Season string (for future enhancements)
        
    Returns:
        Estimated overall rank (integer)
    """
    if gameweeks_played == 0:
        return 5000000  # Default if no games played
    
    # Calculate points per gameweek
    ppg = total_points / gameweeks_played
    
    # FPL has approximately 10 million managers (varies by season)
    # Distribution is roughly log-normal
    total_managers = 10000000
    
    # Statistical model based on historical FPL rank distributions
    # These thresholds are based on actual FPL season data
    
    if ppg >= 80:
        # Elite tier - Top 100
        # Exponential at the very top
        percentile = max(0.00001, 1 - (ppg - 80) * 0.00002)
        rank = int(total_managers * percentile)
        return max(1, min(100, rank))
    
    elif ppg >= 75:
        # Exceptional - Top 1k
        # Linear interpolation between rank 100 and 1000
        position_in_range = (ppg - 75) / 5.0  # 0 to 1
        rank = 1000 - (position_in_range * 900)  # 1000 down to 100
        return int(rank)
    
    elif ppg >= 70:
        # Excellent - Top 10k
        # Linear interpolation between rank 1k and 10k
        position_in_range = (ppg - 70) / 5.0  # 0 to 1
        rank = 10000 - (position_in_range * 9000)  # 10k down to 1k
        return int(rank)
    
    elif ppg >= 65:
        # Very good - Top 100k
        # Linear interpolation between rank 10k and 100k
        position_in_range = (ppg - 65) / 5.0  # 0 to 1
        rank = 100000 - (position_in_range * 90000)  # 100k down to 10k
        return int(rank)
    
    elif ppg >= 60:
        # Good - Top 500k
        # Linear interpolation between rank 100k and 500k
        position_in_range = (ppg - 60) / 5.0  # 0 to 1
        rank = 500000 - (position_in_range * 400000)  # 500k down to 100k
        return int(rank)
    
    elif ppg >= 55:
        # Average - Top 1.5M
        # Linear interpolation between rank 500k and 1.5M
        position_in_range = (ppg - 55) / 5.0  # 0 to 1
        rank = 1500000 - (position_in_range * 1000000)  # 1.5M down to 500k
        return int(rank)
    
    elif ppg >= 50:
        # Below average - Top 3M
        # Linear interpolation between rank 1.5M and 3M
        position_in_range = (ppg - 50) / 5.0  # 0 to 1
        rank = 3000000 - (position_in_range * 1500000)  # 3M down to 1.5M
        return int(rank)
    
    elif ppg >= 45:
        # Poor - Top 5M
        # Linear interpolation between rank 3M and 5M
        position_in_range = (ppg - 45) / 5.0  # 0 to 1
        rank = 5000000 - (position_in_range * 2000000)  # 5M down to 3M
        return int(rank)
    
    else:
        # Very poor - Bottom half
        # Linear interpolation between rank 5M and 10M
        position_in_range = max(0, (ppg - 35) / 10.0)  # 0 to 1
        rank = 10000000 - (position_in_range * 5000000)  # 10M down to 5M
        return int(max(5000000, rank))

