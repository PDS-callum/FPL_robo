import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from .train_model import train_model, iterative_training_update
from .predict_team import predict_team_for_gameweek
from .utils.current_season_collector import FPLCurrentSeasonCollector
from .utils.team_optimizer import FPLTeamOptimizer
from .models.fpl_model import FPLPredictionModel
from .utils.performance_tracker import FPLPerformanceTracker
from .utils.weekly_dashboard import FPLWeeklyDashboard
from .utils.weekly_reports import FPLWeeklyReports
from .utils.smart_transfer_analyzer import FPLSmartTransferAnalyzer

ITERATIVE_DATA_PATH = Path("iterative_season_state.json")

class FPLIterativeSeasonManager:
    """
    Manages iterative training and prediction throughout a season.
    """
    """
    Manages iterative training and prediction throughout a season.
    
    This class handles:
    1. Training models on historical data
    2. Making predictions for each gameweek
    3. Updating models with new data as gameweeks complete
    4. Making transfer decisions based on prediction differences
    """
    
    def __init__(self, data_dir: str = "data", budget: float = 100.0, target: str = 'points_scored', enable_smart_transfers: bool = False):
        """
        Initialize the iterative season manager
        
        Parameters:
        -----------
        data_dir : str
            Data directory path
        budget : float
            Team budget in millions
        target : str
            Target variable for predictions ('points_scored', 'goals_scored', etc.)
        enable_smart_transfers : bool
            Whether to use smart transfer logic with fixture difficulty and form analysis
        """
        self.data_dir = data_dir
        self.budget = budget
        self.target = target
        self.enable_smart_transfers = enable_smart_transfers
        self.current_season_collector = FPLCurrentSeasonCollector(data_dir=data_dir)
        self.team_optimizer = FPLTeamOptimizer(total_budget=budget)
        
        # Initialize new analytics components
        self.performance_tracker = FPLPerformanceTracker(data_dir=data_dir)
        self.weekly_dashboard = FPLWeeklyDashboard(data_dir=data_dir)
        self.weekly_reports = FPLWeeklyReports(data_dir=data_dir)
        self.smart_transfer_analyzer = FPLSmartTransferAnalyzer(data_dir=data_dir)
        
        # Track available cash separately (for transfer calculations)
        self.available_cash = None  # Will be set when provided via command line
        
        # Track season progress
        self.season_state_file = os.path.join(data_dir, ITERATIVE_DATA_PATH)
        self.predictions_history = []
        self.teams_history = []
        self.transfers_history = []
        
        # Load existing state if available
        self.load_season_state()
    
    def load_season_state(self) -> None:
        """Load existing season state from disk"""
        try:
            if os.path.exists(self.season_state_file):
                with open(self.season_state_file, 'r') as f:
                    state = json.load(f)
                
                self.predictions_history = state.get('predictions_history', [])
                self.teams_history = state.get('teams_history', [])
                self.transfers_history = state.get('transfers_history', [])
                self.available_cash = state.get('available_cash', None)
                
                print(f"📂 Loaded season state: {len(self.teams_history)} teams, {len(self.transfers_history)} transfers")
                if self.available_cash is not None:
                    print(f"💰 Available cash: £{self.available_cash:.1f}m")
            else:
                print("📂 No existing season state found - starting fresh")
        except Exception as e:
            print(f"⚠️  Failed to load season state: {e}")
    
    def save_season_state(self) -> None:
        """Save current season state to disk"""
        try:
            def convert_to_json_serializable(obj):
                """Convert pandas/numpy objects to JSON serializable format"""
                if hasattr(obj, 'to_dict'):
                    return obj.to_dict()
                elif hasattr(obj, 'tolist'):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_json_serializable(item) for item in obj]
                else:
                    return obj
            
            state = {
                'predictions_history': convert_to_json_serializable(self.predictions_history),
                'teams_history': convert_to_json_serializable(self.teams_history),
                'transfers_history': convert_to_json_serializable(self.transfers_history),
                'last_updated': datetime.now().isoformat(),
                'target': self.target,
                'budget': self.budget,
                'available_cash': self.available_cash
            }
            
            os.makedirs(os.path.dirname(self.season_state_file), exist_ok=True)
            with open(self.season_state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            print(f"💾 Saved season state to {self.season_state_file}")
        except Exception as e:
            print(f"⚠️  Failed to save season state: {e}")
            print(f"   Attempting to save without problematic data...")
            # Fallback with minimal state
            try:
                state = {
                    'last_updated': datetime.now().isoformat(),
                    'target': str(self.target),
                    'budget': float(self.budget),
                    'available_cash': float(self.available_cash),
                    'teams_count': len(self.teams_history) if hasattr(self, 'teams_history') else 0
                }
                with open(self.season_state_file, 'w') as f:
                    json.dump(state, f, indent=2)
                print(f"💾 Saved minimal season state to {self.season_state_file}")
            except Exception as e2:
                print(f"⚠️  Failed to save even minimal state: {e2}")
    
    def update_budget(self, available_cash: float) -> None:
        """
        Update the available cash for transfers.
        
        This represents the cash in the bank that can be used for transfers,
        not the total team budget. When making transfers, the maximum that can
        be spent is: (selling price of outgoing player) + (available cash).
        
        Parameters:
        -----------
        available_cash : float
            The available cash in millions
        """
        old_cash = self.available_cash
        self.available_cash = available_cash
        
        if old_cash is not None:
            print(f"💰 Available cash updated from £{old_cash:.1f}m to £{available_cash:.1f}m")
        else:
            print(f"💰 Available cash set to £{available_cash:.1f}m")
        
        # Save the updated available cash to the state file
        try:
            self.save_season_state()
        except Exception as e:
            print(f"⚠️  Failed to save updated available cash: {e}")
    
    def resume_season(self) -> Optional[Dict[str, Any]]:
        """
        Resume the iterative season management from where it left off.
        
        This method loads the existing state and continues the season iteration
        from the next gameweek that needs to be processed.
        
        Returns:
        --------
        summary : dict
            Season summary statistics
        """
        print("🔄 Resuming FPL season management...")
        
        # Determine starting gameweek based on existing state
        if self.teams_history:
            # Find the last gameweek we predicted for
            last_predicted_gw = max([t['gameweek'] for t in self.teams_history])
            start_gameweek = last_predicted_gw + 1
            print(f"📈 Last prediction was for GW{last_predicted_gw}, resuming from GW{start_gameweek}")
        else:
            # No previous state, start from gameweek 1
            start_gameweek = 1
            print("📈 No previous predictions found, starting from GW1")
        
        # Run the season iteration starting from the appropriate gameweek
        return self.run_season_iteration(start_gameweek=start_gameweek)
    
    def get_current_gameweek_status(self) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
        """
        Get current gameweek and determine if it's been played
        
        Returns:
        --------
        current_gw : dict
            Current gameweek info (None if between seasons)
        next_unplayed_gw : int
            Next gameweek that hasn't been played (None if between seasons)
        """
        try:
            # Check if we're between seasons first
            if self.current_season_collector.is_season_complete():
                print("🔄 Between seasons - all fixtures completed")
                return None, None
            
            current_gw, bootstrap = self.current_season_collector.get_current_gameweek()
            
            # If current_gw is None (returned by collector when season complete), we're between seasons
            if current_gw is None:
                return None, None
            
            # Find the next unplayed gameweek
            next_unplayed_gw = None
            
            for event in bootstrap['events']:
                if not event['finished'] and not event['is_current']:
                    next_unplayed_gw = event['id']
                    break
              # If no unplayed gameweek found, check if current is finished
            if next_unplayed_gw is None and current_gw and current_gw['finished']:
                next_unplayed_gw = current_gw['id'] + 1
            elif next_unplayed_gw is None and current_gw and not current_gw['finished']:
                next_unplayed_gw = current_gw['id']
            
            return current_gw, next_unplayed_gw
            
        except Exception as e:
            print(f"❌ Failed to get gameweek status: {e}")
            return None, None
    
    def has_gameweek_been_played(self, gameweek: int) -> bool:
        """
        Check if a specific gameweek has been completed
        
        Parameters:
        -----------
        gameweek : int
            Gameweek number to check
            
        Returns:
        --------
        bool
            True if gameweek is finished
        """
        try:
            # Check if we're between seasons first
            if self.current_season_collector.is_season_complete():
                # If we're between seasons, consider all previous season gameweeks as played
                return gameweek <= 38
            
            _, bootstrap = self.current_season_collector.get_current_gameweek()
            
            # If bootstrap is None (from get_current_gameweek when season complete)
            if bootstrap is None:
                return gameweek <= 38
            
            for event in bootstrap['events']:
                if event['id'] == gameweek:
                    return event['finished']
            
            return False
            
        except Exception as e:
            print(f"⚠️  Failed to check gameweek status: {e}")
            return False
    
    def handle_between_seasons(self) -> Optional[Dict[str, Any]]:
        """
        Handle the case where we're between seasons (all 380 fixtures complete)
        Predicts a team for the first week of the next season and exits
        
        Returns:
        --------
        prediction_results : dict
            Prediction results for next season's GW1
        """
        print("="*60)
        print("BETWEEN SEASONS DETECTED")
        print("="*60)
        print("🏁 Previous season complete (380 fixtures played)")
        print("🔮 Making prediction for next season's Gameweek 1...")
        
        try:
            # Make a prediction for gameweek 1 of next season
            # We'll use the current API data as the base for next season
            prediction_results = self.make_gameweek_prediction(gameweek=1)
            
            if prediction_results:
                print("\n✅ Next season GW1 prediction complete!")
                print(f"💰 Team cost: £{prediction_results['total_cost']:.1f}m")
                print(f"📊 Predicted points: {prediction_results['total_predicted_points']:.1f}")
                print(f"👑 Captain: {prediction_results['captain']['name']}")
                print(f"⚡ Formation: {prediction_results['formation']}")
                
                # Save this as a special between-seasons prediction
                between_seasons_state = {
                    'between_seasons': True,
                    'prediction_date': datetime.now().isoformat(),
                    'next_season_gw1_prediction': prediction_results,
                    'message': 'Prediction made for next season GW1 while between seasons'
                }
                
                # Save to a special file
                between_seasons_file = os.path.join(self.data_dir, "between_seasons_prediction.json")
                with open(between_seasons_file, 'w') as f:
                    json.dump(between_seasons_state, f, indent=2)
                
                print(f"💾 Between-seasons prediction saved to {between_seasons_file}")
                print("\n🔄 System will exit until next season begins")
                
                return prediction_results
            else:
                print("❌ Failed to make between-seasons prediction")
                return None
                
        except Exception as e:
            print(f"❌ Error handling between-seasons scenario: {e}")
            return None

    def initial_model_training(self, historical_seasons: Optional[List[str]] = None, epochs: int = 100) -> Tuple[FPLPredictionModel, Dict[str, Any]]:
        """
        Train initial model on historical data
        
        Parameters:
        -----------
        historical_seasons : list, optional
            List of historical seasons to include
        epochs : int
            Number of training epochs
            
        Returns:
        --------
        model : FPLPredictionModel
            Trained model
        training_info : dict
            Training information
        """
        print("="*60)
        print("INITIAL MODEL TRAINING")
        print("="*60)
        
        # Train model on historical data (excluding current season for now)
        model, training_info = train_model(
            target=self.target,
            epochs=epochs,
            batch_size=32,
            include_current_season=False,  # Start without current season
            historical_seasons=historical_seasons,
            data_dir=self.data_dir,
            verbose=1
        )
        
        print(f"✅ Initial model training complete for {self.target}")
        return model, training_info
    
    def make_gameweek_prediction(self, gameweek: int, previous_team: Optional[Dict[str, Any]] = None, max_transfers: int = 1) -> Optional[Dict[str, Any]]:
        """
        Make team prediction for a specific gameweek, respecting transfer constraints and chip usage
        
        Parameters:
        -----------
        gameweek : int
            Gameweek to predict for
        previous_team : dict, optional
            Previous gameweek's team to build upon
        max_transfers : int
            Maximum number of transfers allowed (default: 1 free transfer)
            
        Returns:
        --------
        prediction_results : dict
            Team prediction results with transfer considerations and chip usage
        """
        print(f"\n🔮 Making prediction for gameweek {gameweek}...")
        
        try:
            # Get fixtures data for chip decisions
            fixtures_data = self._get_fixtures_data(gameweek)
            
            if previous_team is None:
                # First gameweek - select completely new team with chip consideration
                print("🆕 First gameweek - selecting optimal team from scratch")
                prediction_results = self._make_prediction_with_chips(
                    gameweek=gameweek,
                    budget=self.budget,
                    target=self.target,
                    fixtures_data=fixtures_data,
                    previous_team=None
                )
            else:
                # Subsequent gameweeks - apply smart transfers to previous team with chip consideration
                print(f"🔄 Building on previous team with up to {max_transfers} transfer(s)")
                prediction_results = self._make_transfer_optimized_prediction_with_chips(
                    gameweek, previous_team, max_transfers, fixtures_data
                )
            
            if prediction_results:
                # Record prediction for performance tracking
                self.performance_tracker.record_prediction(
                    gameweek=gameweek,
                    predicted_points=prediction_results['total_predicted_points'],
                    team_cost=prediction_results['total_cost'],
                    captain=prediction_results['captain']['name'],
                    formation=prediction_results['formation'],
                    target=self.target
                )
                
                # Add to history
                prediction_entry = {
                    'gameweek': gameweek,
                    'prediction_time': datetime.now().isoformat(),
                    'target': self.target,
                    'predicted_points': prediction_results['total_predicted_points'],
                    'team_cost': prediction_results['total_cost'],
                    'formation': prediction_results['formation'],
                    'captain': prediction_results['captain']['name'],
                    'vice_captain': prediction_results['vice_captain']['name']
                }
                
                self.predictions_history.append(prediction_entry)
                self.teams_history.append({
                    'gameweek': gameweek,
                    'team': prediction_results,
                    'timestamp': datetime.now().isoformat()
                })
                
                print(f"✅ Prediction complete for GW{gameweek}")
                print(f"💰 Team cost: £{prediction_results['total_cost']:.1f}m")
                print(f"📊 Predicted points: {prediction_results['total_predicted_points']:.1f}")
                
            return prediction_results
            
        except Exception as e:
            print(f"❌ Failed to make prediction for GW{gameweek}: {e}")
            return None
    
    def _get_fixtures_data(self, gameweek: int) -> pd.DataFrame:
        """Get fixtures data for a specific gameweek"""
        try:
            collector = FPLCurrentSeasonCollector(data_dir=self.data_dir)
            fixtures = collector.get_fixtures()
            if fixtures:
                fixtures_df = pd.DataFrame(fixtures)
                return fixtures_df[fixtures_df['event'] == gameweek]
            return pd.DataFrame()
        except Exception as e:
            print(f"⚠️ Could not load fixtures data: {e}")
            return pd.DataFrame()
    
    def _make_prediction_with_chips(self, gameweek: int, budget: float, target: str, fixtures_data: pd.DataFrame, previous_team: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Make prediction with chip consideration for first gameweek"""
        try:
            # Get all player predictions
            from .predict_team import predict_team_for_gameweek
            prediction_results = predict_team_for_gameweek(
                gameweek=gameweek,
                budget=budget,
                target=target,
                data_dir=self.data_dir,
                save_results=False
            )
            
            if not prediction_results:
                return None
            
            # Check for chip usage
            from .utils.chip_manager import FPLChipManager
            chip_manager = FPLChipManager(self.data_dir)
            
            # Convert team data to DataFrame for chip analysis
            if prediction_results.get('playing_xi') and prediction_results.get('bench'):
                team_players = prediction_results['playing_xi'] + prediction_results['bench']
                team_df = pd.DataFrame(team_players)
                
                # Get predictions for chip analysis
                predictions_df = self._get_predictions_for_chips(gameweek, target)
                
                if not predictions_df.empty:
                    chip_decision = chip_manager.should_use_chip(
                        gameweek, team_df, fixtures_data, predictions_df
                    )
                    
                    if chip_decision:
                        chip_name, chip_config = chip_decision
                        print(f"🎯 Using {chip_name.upper()} chip this gameweek!")
                        print(f"   Reason: {chip_config.get('reason', 'Unknown')}")
                        
                        # Record chip usage
                        chip_manager.record_chip_usage(gameweek, chip_name, chip_config)
                        
                        # Add chip info to results
                        prediction_results['chip_used'] = chip_name
                        prediction_results['chip_config'] = chip_config
                        
                        # Apply chip scoring modifications
                        if chip_name == 'triple_captain':
                            captain = prediction_results['captain']
                            captain_points = captain['predicted_points']
                            prediction_results['total_predicted_points'] += captain_points  # Add extra captain points
                            print(f"🔥 Triple Captain: {captain['name']} gets 3x points")
                        
                        elif chip_name == 'bench_boost':
                            bench_points = sum(p['predicted_points'] for p in prediction_results['bench'])
                            prediction_results['total_predicted_points'] += bench_points
                            print(f"🔥 Bench Boost: +{bench_points:.1f} points from bench")
            
            return prediction_results
            
        except Exception as e:
            print(f"❌ Error in chip-enabled prediction: {e}")
            return None
    
    def _make_transfer_optimized_prediction_with_chips(self, gameweek: int, previous_team: Dict[str, Any], max_transfers: int, fixtures_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Make transfer-optimized prediction with chip consideration"""
        try:
            # First, make the normal transfer-optimized prediction
            prediction_results = self._make_transfer_optimized_prediction(
                gameweek, previous_team, max_transfers
            )
            
            if not prediction_results:
                return None
            
            # Check for chip usage
            from .utils.chip_manager import FPLChipManager
            chip_manager = FPLChipManager(self.data_dir)
            
            # Convert team data to DataFrame for chip analysis
            if prediction_results.get('playing_xi') and prediction_results.get('bench'):
                team_players = prediction_results['playing_xi'] + prediction_results['bench']
                team_df = pd.DataFrame(team_players)
                
                # Get predictions for chip analysis
                predictions_df = self._get_predictions_for_chips(gameweek, self.target)
                
                if not predictions_df.empty:
                    chip_decision = chip_manager.should_use_chip(
                        gameweek, team_df, fixtures_data, predictions_df
                    )
                    
                    if chip_decision:
                        chip_name, chip_config = chip_decision
                        print(f"🎯 Using {chip_name.upper()} chip this gameweek!")
                        print(f"   Reason: {chip_config.get('reason', 'Unknown')}")
                        
                        # Record chip usage
                        chip_manager.record_chip_usage(gameweek, chip_name, chip_config)
                        
                        # Add chip info to results
                        prediction_results['chip_used'] = chip_name
                        prediction_results['chip_config'] = chip_config
                        
                        # Apply chip scoring modifications
                        if chip_name == 'triple_captain':
                            captain = prediction_results['captain']
                            captain_points = captain['predicted_points']
                            prediction_results['total_predicted_points'] += captain_points  # Add extra captain points
                            print(f"🔥 Triple Captain: {captain['name']} gets 3x points")
                        
                        elif chip_name == 'bench_boost':
                            bench_points = sum(p['predicted_points'] for p in prediction_results['bench'])
                            prediction_results['total_predicted_points'] += bench_points
                            print(f"🔥 Bench Boost: +{bench_points:.1f} points from bench")
            
            return prediction_results
            
        except Exception as e:
            print(f"❌ Error in chip-enabled transfer prediction: {e}")
            return None
    
    def _get_predictions_for_chips(self, gameweek: int, target: str) -> pd.DataFrame:
        """Get player predictions for chip analysis"""
        try:
            from .utils.current_season_collector import FPLCurrentSeasonCollector
            from .models.fpl_model import FPLPredictionModel
            import pickle
            import os
            
            # Get current players
            collector = FPLCurrentSeasonCollector(data_dir=self.data_dir)
            bootstrap = collector.get_bootstrap_static()
            players_df = pd.DataFrame(bootstrap['elements'])
            
            # Load model and make predictions
            scaler_path = os.path.join(self.data_dir, 'processed', 'scalers.pkl')
            scaler = None
            expected_features = 46
            
            try:
                with open(scaler_path, 'rb') as f:
                    scalers = pickle.load(f)
                scaler = scalers.get(f'{target}_scaler')
                expected_features = getattr(scaler, 'n_features_in_', 46) if scaler else 46
            except:
                pass
            
            # Create basic features for all players
            player_features = []
            for _, player in players_df.iterrows():
                features = {
                    'position': player.get('element_type', 0),
                    'team': player.get('team', 0),
                    'price': player.get('now_cost', 0) / 10,
                    'selected_by_percent': float(player.get('selected_by_percent', 0)),
                    'minutes_played': player.get('minutes', 0),
                    'goals_scored': player.get('goals_scored', 0),
                    'assists': player.get('assists', 0),
                    'clean_sheets': player.get('clean_sheets', 0),
                    'goals_conceded': player.get('goals_conceded', 0),
                    'yellow_cards': player.get('yellow_cards', 0),
                    'red_cards': player.get('red_cards', 0),
                    'saves': player.get('saves', 0),
                    'bonus': player.get('bonus', 0),
                    'bps': player.get('bps', 0),
                    'influence': float(player.get('influence', 0)),
                    'creativity': float(player.get('creativity', 0)),
                    'threat': float(player.get('threat', 0)),
                    'form': float(player.get('form', 0)),
                    'total_points': player.get('total_points', 0),
                    'points_per_game': float(player.get('points_per_game', 0))
                }
                player_features.append(features)
            
            features_df = pd.DataFrame(player_features)
            
            # Make predictions using the model
            model = FPLPredictionModel(n_features=expected_features)
            model.load(f'fpl_model_{target}.h5')
            
            # Prepare features for prediction - use only numeric columns
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            X_pred = features_df[numeric_cols].fillna(0).values
            
            # Pad or truncate to expected features
            if X_pred.shape[1] < expected_features:
                padding = np.zeros((X_pred.shape[0], expected_features - X_pred.shape[1]))
                X_pred = np.concatenate([X_pred, padding], axis=1)
            elif X_pred.shape[1] > expected_features:
                X_pred = X_pred[:, :expected_features]
            
            if scaler:
                X_pred = scaler.transform(X_pred)
            
            predictions = model.predict(X_pred).flatten()
            
            return pd.DataFrame({
                'id': players_df['id'],
                'predicted_points': predictions
            })
            
        except Exception as e:
            print(f"⚠️ Could not get predictions for chips: {e}")
            return pd.DataFrame()

    def _make_transfer_optimized_prediction(self, gameweek: int, previous_team: Dict[str, Any], max_transfers: int) -> Optional[Dict[str, Any]]:
        """
        Make a prediction by optimizing transfers from the previous team
        
        Parameters:
        -----------
        gameweek : int
            Gameweek to predict for
        previous_team : dict
            Previous team structure
        max_transfers : int
            Maximum number of transfers allowed
            
        Returns:
        --------
        prediction_results : dict
            Optimized team with transfers applied
        """
        from .utils.current_season_collector import FPLCurrentSeasonCollector
        from .models.fpl_model import FPLPredictionModel
        from .utils.data_collection import FPLDataProcessor
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        import pickle
        import os
        
        print(f"🔄 Optimizing transfers for GW{gameweek} (max {max_transfers} transfers)")
        
        # Step 1: Get current player predictions
        collector = FPLCurrentSeasonCollector(data_dir=self.data_dir)
        current_data = collector.collect_current_season_data()
        
        if not current_data:
            print("❌ Failed to collect current season data")
            return None
        
        # Step 2: Get predictions for all available players (not an optimal team)
        try:
            # Instead of getting a fresh optimal team, we need predictions for all players
            # so we can compare with our current team players
            from .predict_team import predict_team_for_gameweek
            from .utils.current_season_collector import FPLCurrentSeasonCollector
            from .models.fpl_model import FPLPredictionModel
            from .utils.data_collection import FPLDataProcessor
            from .utils.constants import POSITION_MAP
            import pandas as pd
            import numpy as np
            from sklearn.preprocessing import StandardScaler
            import pickle
            import os
            
            # Get all player predictions for this gameweek
            all_players_with_predictions, all_players_status = self._get_all_player_predictions(gameweek)
            
            if not all_players_with_predictions:
                print("❌ Failed to get player predictions")
                return None
            
            # Step 3: Apply transfer constraints to optimize from previous team
            return self._optimize_transfers_from_previous_team(
                all_players_with_predictions, previous_team, max_transfers, gameweek, all_players_status
            )
            
        except Exception as e:
            print(f"❌ Error in transfer optimization: {e}")
            return None

    def _get_all_player_predictions(self, gameweek: int) -> Optional[Dict[str, Any]]:
        """
        Get predictions for all available players for a given gameweek
        
        Parameters:
        -----------
        gameweek : int
            Gameweek to predict for
            
        Returns:
        --------
        players_dict : dict
            Dictionary mapping player names to their prediction data
        """
        try:
            # Use the existing predict_team_for_gameweek function to get all player predictions
            # but intercept the player data before team optimization
            from .predict_team import predict_team_for_gameweek
            from .utils.current_season_collector import FPLCurrentSeasonCollector
            from .models.fpl_model import FPLPredictionModel
            from .utils.constants import POSITION_MAP
            import pandas as pd
            import pickle
            import os
            
            # Get a fresh optimal team prediction (this includes all player predictions)
            fresh_prediction = predict_team_for_gameweek(
                gameweek=gameweek,
                budget=200.0,  # Use high budget to get all possible players considered
                target=self.target,
                data_dir=self.data_dir,
                save_results=False
            )
            
            if not fresh_prediction:
                print("❌ Failed to get fresh prediction")
                return None
            
            # Since we can't easily extract all player predictions from the existing function,
            # we'll work with what we have. Let's get current player data instead.
            collector = FPLCurrentSeasonCollector(data_dir=self.data_dir)
            current_data = collector.collect_current_season_data()
            
            if not current_data or 'bootstrap' not in current_data:
                print("❌ Failed to get current data")
                return None
            
            # Convert bootstrap players to a dict with basic info
            bootstrap = current_data['bootstrap']
            all_players_dict = {}
            all_players_status = {}  # Track status of ALL players (including injured) for team updating
            injured_count = 0
            low_availability_count = 0
            
            for player in bootstrap['elements']:
                # Check player availability/injury status
                chance_of_playing = player.get('chance_of_playing_next_round')
                player_status = player.get('status', 'a')  # 'a'=available, 'i'=injured, 'd'=doubtful, 'u'=unavailable
                
                # Always track status for existing team player updates
                all_players_status[player['web_name']] = {
                    'status': player_status,
                    'chance_of_playing': chance_of_playing,
                    'cost': player['now_cost'] / 10.0,
                    'id': player['id']
                }
                
                # Skip players who are clearly unavailable or injured for NEW selections
                if player_status == 'i':  # Injured
                    injured_count += 1
                    print(f"⚠️  Skipping injured player: {player['web_name']} (Status: {player_status})")
                    continue
                elif player_status in ['u', 's']:  # Unavailable (on loan, suspended, etc.)
                    print(f"⚠️  Skipping unavailable player: {player['web_name']} (Status: {player_status})")
                    continue
                elif chance_of_playing is not None and chance_of_playing < 25:  # Very low chance of playing
                    low_availability_count += 1
                    print(f"⚠️  Skipping low availability player: {player['web_name']} ({chance_of_playing}% chance of playing)")
                    continue
                
                # Simple heuristic: use total points + form as prediction
                form_str = str(player.get('form', '0'))
                form_val = float(form_str) if form_str.replace('.', '').isdigit() else 0.0
                
                predicted_pts = float(player.get('total_points', 0)) + form_val * 2
                
                # Apply availability penalty for doubtful players
                availability_penalty = 0
                if player_status == 'd':  # Doubtful
                    availability_penalty = predicted_pts * 0.15  # 15% penalty for doubtful players
                    print(f"📉 Applying availability penalty to {player['web_name']} (Status: doubtful)")
                elif chance_of_playing is not None and chance_of_playing < 75:  # Low-ish chance of playing
                    availability_penalty = predicted_pts * 0.1   # 10% penalty for players with <75% chance
                    print(f"📉 Applying availability penalty to {player['web_name']} ({chance_of_playing}% chance of playing)")
                
                player_data = {
                    'id': player['id'],
                    'name': player['web_name'],
                    'team': bootstrap['teams'][player['team'] - 1]['name'],
                    'position': POSITION_MAP.get(player['element_type'], 'Unknown'),
                    'cost': player['now_cost'] / 10.0,
                    'predicted_points': predicted_pts - availability_penalty,
                    'chance_of_playing': chance_of_playing,
                    'status': player_status,
                    'original_predicted_points': predicted_pts,  # Keep original for reference
                }
                all_players_dict[player['web_name']] = player_data
            
            print(f"✅ Generated predictions for {len(all_players_dict)} available players")
            print(f"📊 Excluded {injured_count} injured players and {low_availability_count} low availability players")
            return all_players_dict, all_players_status
            
        except Exception as e:
            print(f"❌ Error getting player predictions: {e}")
            return None, None

    def _optimize_transfers_from_previous_team(self, all_players_dict: Dict[str, Any], previous_team: Dict[str, Any], max_transfers: int, gameweek: int, all_players_status: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Optimize transfers by finding the best transfers to make from previous team
        
        Parameters:
        -----------
        all_players_dict : dict
            Dictionary with all players and their predictions
        previous_team : dict
            Previous gameweek's team
        max_transfers : int
            Maximum transfers allowed
        gameweek : int
            Current gameweek
            
        Returns:
        --------
        optimized_team : dict
            Team with optimal transfers applied
        """
        try:
            # Get all players from previous team (15 players total)
            prev_players = {}
            for player in previous_team['playing_xi'] + previous_team['bench']:
                prev_players[player['name']] = player
            
            print(f"📋 Previous team players: {list(prev_players.keys())}")
            
            # Update previous team players with fresh predictions if they exist
            updated_prev_players = {}
            injured_players = []
            unavailable_players = []
            
            for name, prev_player in prev_players.items():
                if name in all_players_dict:
                    # Update with fresh prediction and ensure consistent cost field
                    updated_player = prev_player.copy()
                    updated_player['predicted_points'] = all_players_dict[name]['predicted_points']
                    updated_player['chance_of_playing'] = all_players_dict[name].get('chance_of_playing')
                    updated_player['status'] = all_players_dict[name].get('status', 'a')
                    
                    # Check if player is injured or unavailable
                    if updated_player['status'] == 'i':  # Injured
                        injured_players.append(name)
                    elif updated_player['status'] in ['u', 's']:  # Unavailable or Suspended
                        unavailable_players.append(name)
                    elif updated_player['chance_of_playing'] is not None and updated_player['chance_of_playing'] < 25:
                        injured_players.append(name)  # Treat very low availability as injured
                    
                    # Ensure cost field is consistent (use 'cost' as standard)
                    if 'cost' not in updated_player and 'price' in updated_player:
                        updated_player['cost'] = updated_player['price']
                    elif 'cost' not in updated_player:
                        # Fallback to fresh cost data
                        updated_player['cost'] = all_players_dict[name]['cost']
                    updated_prev_players[name] = updated_player
                    
                    status_info = ""
                    if updated_player['status'] != 'a':
                        status_info = f" (Status: {updated_player['status']})"
                    elif updated_player['chance_of_playing'] is not None and updated_player['chance_of_playing'] < 75:
                        status_info = f" ({updated_player['chance_of_playing']}% chance)"
                    
                    print(f"✅ Updated {name}: {updated_player['predicted_points']:.1f} pts, £{updated_player['cost']:.1f}m{status_info}")
                elif name in all_players_status:
                    # Player is not available for selection (injured/unavailable) but we have status info
                    updated_player = prev_player.copy()
                    status_data = all_players_status[name]
                    updated_player['status'] = status_data['status']
                    updated_player['chance_of_playing'] = status_data['chance_of_playing']
                    
                    # Update cost if available
                    if 'cost' not in updated_player and 'cost' in status_data:
                        updated_player['cost'] = status_data['cost']
                    elif 'cost' not in updated_player and 'price' in updated_player:
                        updated_player['cost'] = updated_player['price']
                    elif 'cost' not in updated_player:
                        updated_player['cost'] = 5.0  # Emergency fallback
                    
                    # Check if player is injured or unavailable
                    if updated_player['status'] == 'i':  # Injured
                        injured_players.append(name)
                        print(f"🚑 Previous player {name} is INJURED (Status: {updated_player['status']}) - prioritizing for transfer")
                    elif updated_player['status'] in ['u', 's']:  # Unavailable or Suspended
                        unavailable_players.append(name)
                        status_desc = "SUSPENDED" if updated_player['status'] == 's' else "UNAVAILABLE"
                        print(f"❌ Previous player {name} is {status_desc} (Status: {updated_player['status']}) - must transfer out")
                    elif updated_player['chance_of_playing'] is not None and updated_player['chance_of_playing'] < 25:
                        injured_players.append(name)
                        print(f"🚑 Previous player {name} has low availability ({updated_player['chance_of_playing']}%) - prioritizing for transfer")
                    
                    updated_prev_players[name] = updated_player
                else:
                    # Player not found in current data (maybe transferred away from club)
                    # Ensure cost field exists
                    updated_player = prev_player.copy()
                    if 'cost' not in updated_player and 'price' in updated_player:
                        updated_player['cost'] = updated_player['price']
                    elif 'cost' not in updated_player:
                        # Emergency fallback - use a reasonable estimate
                        updated_player['cost'] = 5.0
                    print(f"⚠️  Previous player {name} not found in current data, using cached data")
                    updated_prev_players[name] = updated_player
            
            if injured_players:
                print(f"🚑 INJURED PLAYERS IN TEAM: {', '.join(injured_players)} - will prioritize for transfer")
            if unavailable_players:
                print(f"❌ UNAVAILABLE PLAYERS IN TEAM: {', '.join(unavailable_players)} - must transfer out")
            
            print(f"📊 Successfully updated {len(updated_prev_players)}/{len(prev_players)} players with fresh predictions")
            
            # Calculate current team total cost for budget validation
            try:
                current_team_cost = sum(p['cost'] for p in updated_prev_players.values())
                available_budget = self.budget
                budget_surplus = available_budget - current_team_cost
                
                print(f"💰 Current team cost: £{current_team_cost:.1f}m")
                print(f"💰 Available budget: £{available_budget:.1f}m")
                print(f"💰 Budget surplus: £{budget_surplus:.1f}m")
                
                if current_team_cost > available_budget + 0.1:  # Small tolerance for rounding
                    print(f"⚠️  WARNING: Current team cost (£{current_team_cost:.1f}m) exceeds available budget (£{available_budget:.1f}m)")
                    print("🔧 This might happen due to price changes. Use --budget argument to set correct available balance.")
                    print("🔧 Continuing with transfer optimization but transfers may be limited...")
                
            except KeyError as e:
                print(f"❌ Error calculating team cost: {e}")
                print("🔧 Some players missing cost information. This might cause transfer optimization issues.")
            
            # Find beneficial transfers
            transfer_candidates = []
            
            # Create a prioritized list of players to transfer out
            # Priority 1: Unavailable players (must transfer)
            # Priority 2: Injured players 
            # Priority 3: All other players
            prioritized_players = []
            
            # Add unavailable players first (highest priority)
            for player_name in unavailable_players:
                if player_name in updated_prev_players:
                    prioritized_players.append((updated_prev_players[player_name], 'unavailable'))
            
            # Add injured players second
            for player_name in injured_players:
                if player_name in updated_prev_players and player_name not in unavailable_players:
                    prioritized_players.append((updated_prev_players[player_name], 'injured'))
            
            # Add all other players
            for prev_name, prev_player in updated_prev_players.items():
                if prev_name not in injured_players and prev_name not in unavailable_players:
                    prioritized_players.append((prev_player, 'healthy'))
            
            for prev_player, priority_type in prioritized_players:
                # Find best replacement of same position
                best_replacement = None
                best_value = -999
                cheapest_affordable = None  # Fallback for injured players when budget is tight
                
                for candidate_name, candidate_player in all_players_dict.items():
                    if (candidate_name not in updated_prev_players and 
                        candidate_player['position'] == prev_player['position']):
                        
                        # Calculate transfer value
                        points_gain = candidate_player['predicted_points'] - prev_player['predicted_points']
                        cost_diff = candidate_player['cost'] - prev_player['cost']
                        
                        # Add priority bonus for injured/unavailable players
                        priority_bonus = 0
                        if priority_type == 'unavailable':
                            priority_bonus = 50  # Huge bonus - must transfer unavailable players
                        elif priority_type == 'injured':
                            priority_bonus = 20  # Large bonus for injured players
                        
                        # Apply smart transfer analysis if enabled
                        smart_analysis = None
                        if self.enable_smart_transfers:
                            try:
                                smart_analysis = self.smart_transfer_analyzer.analyze_transfer_opportunity(
                                    out_player=prev_player,
                                    in_player=candidate_player,
                                    gameweek=gameweek
                                )
                                # Use smart transfer rating to adjust the value
                                smart_rating = smart_analysis.get('overall_rating', 50) / 100.0  # Convert to 0-1 scale
                                points_gain = points_gain * smart_rating  # Scale points gain by smart rating
                            except Exception as e:
                                print(f"⚠️  Smart transfer analysis failed: {e}")
                                smart_analysis = None
                        
                        # Consider if we can afford the transfer
                        try:
                            # Calculate affordability using available cash model
                            # Available funds = outgoing player sale price + available cash in bank
                            if self.available_cash is not None:
                                # Use the provided available cash
                                available_funds = prev_player['cost'] + self.available_cash
                                budget_info = f"Sale: £{prev_player['cost']:.1f}m + Cash: £{self.available_cash:.1f}m = £{available_funds:.1f}m"
                            else:
                                # Fallback to old budget calculation if no available cash specified
                                current_team_cost = sum(p['cost'] for p in updated_prev_players.values())
                                available_funds = self.budget - current_team_cost + prev_player['cost']
                                budget_info = f"Budget remaining: £{available_funds:.1f}m"
                            
                            # Check if we can afford the incoming player
                            can_afford = candidate_player['cost'] <= available_funds
                            
                            if can_afford:
                                # Value = points gained + priority bonus minus transfer cost
                                transfer_cost = 4 if len(transfer_candidates) >= max_transfers else 0
                                net_value = points_gain + priority_bonus - transfer_cost
                                
                                if net_value > best_value:
                                    best_value = net_value
                                    best_replacement = {
                                        'out': prev_player,
                                        'in': candidate_player,
                                        'points_gain': points_gain,
                                        'priority_bonus': priority_bonus,
                                        'priority_type': priority_type,
                                        'cost_diff': cost_diff,
                                        'net_value': net_value,
                                        'transfer_cost': transfer_cost,
                                        'available_funds': available_funds,
                                        'budget_check': f"£{candidate_player['cost']:.1f}m <= £{available_funds:.1f}m (✅ Affordable)",
                                        'budget_info': budget_info,
                                        'smart_analysis': smart_analysis
                                    }
                                
                                # Track cheapest affordable option for injured players as fallback
                                if priority_type in ['injured', 'unavailable']:
                                    if cheapest_affordable is None or candidate_player['cost'] < cheapest_affordable['in']['cost']:
                                        cheapest_affordable = {
                                            'out': prev_player,
                                            'in': candidate_player,
                                            'points_gain': points_gain,
                                            'priority_bonus': priority_bonus,
                                            'priority_type': priority_type,
                                            'cost_diff': cost_diff,
                                            'net_value': net_value,
                                            'transfer_cost': transfer_cost,
                                            'available_funds': available_funds,
                                            'budget_check': f"£{candidate_player['cost']:.1f}m <= £{available_funds:.1f}m (✅ Affordable)",
                                            'budget_info': budget_info,
                                            'smart_analysis': smart_analysis
                                        }
                            else:
                                # For injured/unavailable players, show budget constraints even for lower point gains
                                if priority_type in ['injured', 'unavailable'] or points_gain > 3:
                                    print(f"  💸 Budget constraint: {prev_player['name']} → {candidate_player['name']}")
                                    print(f"     Points gain: +{points_gain:.1f}, Priority: {priority_type}, but £{candidate_player['cost']:.1f}m > £{available_funds:.1f}m available")
                                    print(f"     {budget_info}")
                        
                        except KeyError as e:
                            print(f"❌ Budget calculation error for {candidate_name}: {e}")
                            continue
                
                # Use best replacement if available, otherwise use cheapest affordable for injured players
                if best_replacement:
                    # Always add transfers for unavailable/injured players, even if negative value
                    # Only require positive value for healthy player transfers
                    if priority_type in ['unavailable', 'injured'] or best_replacement['net_value'] > 0:
                        transfer_candidates.append(best_replacement)
                elif cheapest_affordable and priority_type in ['injured', 'unavailable']:
                    # If no "good" replacement found but we have an affordable option for injured/unavailable player, use it
                    # BUT ONLY if it's the same position!
                    if cheapest_affordable['in']['position'] == prev_player['position']:
                        print(f"🚑 No positive-value replacement found for {priority_type} player {prev_player['name']}, using cheapest affordable option")
                        transfer_candidates.append(cheapest_affordable)
                    else:
                        print(f"❌ No valid same-position replacement found for {priority_type} player {prev_player['name']} ({prev_player['position']})")
                        print(f"   Cheapest affordable option {cheapest_affordable['in']['name']} is wrong position ({cheapest_affordable['in']['position']})")
            
            # Sort by net value and take the best transfers up to max_transfers
            transfer_candidates.sort(key=lambda x: x['net_value'], reverse=True)
            selected_transfers = transfer_candidates[:max_transfers]
            
            # Apply the selected transfers to create the new team
            new_team_players = updated_prev_players.copy()
            total_transfer_cost = 0
            
            print(f"\n💱 Applying {len(selected_transfers)} transfer(s):")
            for i, transfer in enumerate(selected_transfers):
                out_player = transfer['out']
                in_player = transfer['in']
                priority_type = transfer.get('priority_type', 'healthy')
                priority_bonus = transfer.get('priority_bonus', 0)
                
                # Remove old player and add new player
                del new_team_players[out_player['name']]
                new_team_players[in_player['name']] = in_player
                total_transfer_cost += transfer['transfer_cost']
                
                # Add emoji based on priority
                priority_emoji = ""
                if priority_type == 'unavailable':
                    priority_emoji = " ❌ UNAVAILABLE"
                elif priority_type == 'injured':
                    priority_emoji = " 🚑 INJURED"
                
                print(f"  {i+1}. OUT: {out_player['name']} ({out_player['team']}) - {out_player['predicted_points']:.1f} pts - £{out_player['cost']:.1f}m{priority_emoji}")
                print(f"     IN:  {in_player['name']} ({in_player['team']}) - {in_player['predicted_points']:.1f} pts - £{in_player['cost']:.1f}m")
                
                if priority_bonus > 0:
                    print(f"     Gain: {transfer['points_gain']:.1f} pts + {priority_bonus} priority bonus = {transfer['points_gain'] + priority_bonus:.1f} total (Cost: {transfer['transfer_cost']} pts)")
                else:
                    print(f"     Gain: {transfer['points_gain']:.1f} pts (Cost: {transfer['transfer_cost']} pts)")
                
                # Add smart transfer insights if available
                if self.enable_smart_transfers and 'smart_analysis' in transfer:
                    smart_analysis = transfer['smart_analysis']
                    if smart_analysis:
                        rating = smart_analysis.get('overall_rating', 0)
                        recommendation = smart_analysis.get('recommendation', 'HOLD')
                        print(f"     🧠 Smart Analysis: {rating:.1f}/100 - {recommendation}")
                        
                        # Show key insights
                        insights = smart_analysis.get('insights', [])
                        if insights:
                            for insight in insights[:2]:  # Show top 2 insights
                                print(f"     💡 {insight}")
                    
                print(f"     {transfer.get('budget_info', 'Budget info not available')}")
                print(f"     {transfer.get('budget_check', 'Budget check not available')}")
            
            # Update available cash after transfers
            if self.available_cash is not None:
                cash_spent = sum(t['cost_diff'] for t in selected_transfers)
                self.available_cash -= cash_spent
                print(f"\n💰 Cash spent on transfers: £{cash_spent:.1f}m")
                print(f"💰 Remaining available cash: £{self.available_cash:.1f}m")
                
                # Save updated cash amount
                self.save_season_state()
            
            # Validate final team
            try:
                final_team_cost = sum(p['cost'] for p in new_team_players.values())
                
                print(f"\n💰 Final team cost: £{final_team_cost:.1f}m")
                
                if self.available_cash is not None:
                    total_value = final_team_cost + self.available_cash
                    print(f"💰 Available cash: £{self.available_cash:.1f}m") 
                    print(f"💰 Total value: £{total_value:.1f}m")
                else:
                    budget_remaining = self.budget - final_team_cost
                    print(f"💰 Initial budget: £{self.budget:.1f}m") 
                    print(f"💰 Budget remaining: £{budget_remaining:.1f}m")
                    
                    if final_team_cost > self.budget + 0.1:  # Small tolerance for rounding
                        print(f"❌ ERROR: Final team exceeds budget by £{final_team_cost - self.budget:.1f}m!")
                        print("🔄 Reverting to previous team to avoid budget violation...")
                        print("💡 TIP: Use --budget argument to specify your actual available balance")
                        return self._create_team_from_players(list(updated_prev_players.values()), gameweek, 0, 0)
                
                print(f"✅ Budget validation passed!")
            except KeyError as e:
                print(f"⚠️  Warning: Could not validate final budget due to missing cost data: {e}")
            
            if not selected_transfers:
                print("💱 No beneficial transfers found - keeping current team")
                # Return the previous team with updated predictions
                return self._create_team_from_players(list(updated_prev_players.values()), gameweek, 0, 0)
            
            # Reorganize into playing XI and bench using the simpler team creation logic
            new_team_list = list(new_team_players.values())
            optimized_team = self._create_team_from_players(new_team_list, gameweek, len(selected_transfers), total_transfer_cost)
            
            # Adjust total predicted points for transfer costs (already handled in _create_team_from_players)
            if optimized_team:
                print(f"📊 Transfer penalty: -{total_transfer_cost} points (already included)")
                # Add transfer information to the team data
                optimized_team['transfers_made'] = len(selected_transfers)
                optimized_team['transfer_cost'] = total_transfer_cost
                optimized_team['transfer_details'] = selected_transfers
            
            return optimized_team
            
        except Exception as e:
            print(f"❌ Error optimizing transfers: {e}")
            return self._create_team_from_players(list(prev_players.values()), gameweek, 0, 0)  # Return previous team if optimization fails

    def _create_team_from_players(self, players: List[Dict[str, Any]], gameweek: int, transfers_made: int = 0, transfer_cost: int = 0) -> Optional[Dict[str, Any]]:
        """
        Create a team structure from a list of 15 players with updated predictions
        
        Parameters:
        -----------
        players : list
            List of 15 players
        gameweek : int
            Current gameweek
        transfers_made : int
            Number of transfers made
        transfer_cost : int
            Total transfer cost in points
            
        Returns:
        --------
        team_structure : dict
            Complete team structure
        """
        try:
            # Sort players by predicted points to select best XI
            players_sorted = sorted(players, key=lambda x: x.get('predicted_points', 0), reverse=True)
            
            # Separate injured/unavailable players to prioritize for benching
            available_players = []
            injured_unavailable_players = []
            
            for player in players:
                player_status = player.get('status', 'a')
                chance_of_playing = player.get('chance_of_playing')
                
                # Check if player is injured, unavailable, suspended, or has very low availability
                if (player_status in ['i', 'u', 's'] or 
                    (chance_of_playing is not None and chance_of_playing < 25)):
                    injured_unavailable_players.append(player)
                    if player_status == 'i':
                        print(f"🚑 {player['name']} is injured - will be benched if possible")
                    elif player_status == 'u':
                        print(f"❌ {player['name']} is unavailable - will be benched if possible")
                    elif player_status == 's':
                        print(f"🚫 {player['name']} is suspended - will be benched if possible")
                    else:
                        print(f"📉 {player['name']} has low availability ({chance_of_playing}%) - will be benched if possible")
                else:
                    available_players.append(player)
            
            # Use formation constraints to select playing XI from available players first
            from .utils.constants import POSITION_MAP
            
            # Count available players by position
            position_counts = {}
            for player in players:
                pos = player.get('position', 'Unknown')
                position_counts[pos] = position_counts.get(pos, 0) + 1
            
            # Select playing XI following FPL formation rules
            playing_xi = []
            bench = []
            
            # Debug: check positions of all players
            
            # Required positions for playing XI (exactly these amounts)
            required_xi = {'GK': 1, 'DEF': 3, 'MID': 3, 'FWD': 1}  # Minimum formation 3-3-1
            selected_by_position = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
            
            # Step 1: Select minimum required players for each position
            # Start with goalkeeper (exactly 1 required)
            # Handle multiple position formats: numeric (1), string ("GK", "GKP")
            def is_goalkeeper(player):
                pos = player.get('position')
                return pos == 1 or pos == 'GK' or pos == 'GKP'
            
            def is_defender(player):
                pos = player.get('position')
                return pos == 2 or pos == 'DEF'
            
            def is_midfielder(player):
                pos = player.get('position')
                return pos == 3 or pos == 'MID'
            
            def is_forward(player):
                pos = player.get('position')
                return pos == 4 or pos == 'FWD'
            
            # Step 1: Select goalkeeper (exactly 1 required)
            # Prioritize available goalkeepers over injured ones
            available_gks = [p for p in available_players if is_goalkeeper(p)]
            injured_gks = [p for p in injured_unavailable_players if is_goalkeeper(p)]
            all_gks = available_gks + injured_gks  # Available first, then injured as fallback
            
            if all_gks:
                best_gk = max(all_gks, key=lambda x: x.get('predicted_points', 0))
                playing_xi.append(best_gk)
                selected_by_position['GK'] = 1
                status_info = ""
                if best_gk in injured_unavailable_players:
                    status_info = f" 🚑 (Status: {best_gk.get('status', 'unknown')})"
                print(f"✅ Selected starting GK: {best_gk['name']} - {best_gk.get('predicted_points', 0):.1f} pts{status_info}")
            else:
                print("❌ No goalkeepers found!")
            
            # Step 2: Select minimum required defenders (3) - prioritize available players
            available_defs = [p for p in available_players if is_defender(p) and p not in playing_xi]
            injured_defs = [p for p in injured_unavailable_players if is_defender(p) and p not in playing_xi]
            all_def_players = available_defs + injured_defs  # Available first
            
            def_sorted = sorted(all_def_players, key=lambda x: x.get('predicted_points', 0), reverse=True)
            for i in range(min(3, len(def_sorted))):
                playing_xi.append(def_sorted[i])
                selected_by_position['DEF'] += 1
            
            # Step 3: Select minimum required midfielders (3) - prioritize available players  
            available_mids = [p for p in available_players if is_midfielder(p) and p not in playing_xi]
            injured_mids = [p for p in injured_unavailable_players if is_midfielder(p) and p not in playing_xi]
            all_mid_players = available_mids + injured_mids  # Available first
            
            mid_sorted = sorted(all_mid_players, key=lambda x: x.get('predicted_points', 0), reverse=True)
            for i in range(min(3, len(mid_sorted))):
                playing_xi.append(mid_sorted[i])
                selected_by_position['MID'] += 1
            
            # Step 4: Select minimum required forwards (1) - prioritize available players
            available_fwds = [p for p in available_players if is_forward(p) and p not in playing_xi]
            injured_fwds = [p for p in injured_unavailable_players if is_forward(p) and p not in playing_xi]
            all_fwd_players = available_fwds + injured_fwds  # Available first
            
            fwd_sorted = sorted(all_fwd_players, key=lambda x: x.get('predicted_points', 0), reverse=True)
            for i in range(min(1, len(fwd_sorted))):
                playing_xi.append(fwd_sorted[i])
                selected_by_position['FWD'] += 1
            
            # Step 5: Fill remaining spots (up to 11 total) with best available players
            # Maximum constraints: GK=1, DEF=5, MID=5, FWD=3
            max_positions = {'GK': 1, 'DEF': 5, 'MID': 5, 'FWD': 3}
            
            # Prioritize available players over injured ones for remaining spots
            remaining_available = [p for p in available_players if p not in playing_xi]
            remaining_injured = [p for p in injured_unavailable_players if p not in playing_xi]
            remaining_players = remaining_available + remaining_injured  # Available first
            
            remaining_sorted = sorted(remaining_players, key=lambda x: x.get('predicted_points', 0), reverse=True)
            
            for player in remaining_sorted:
                if len(playing_xi) >= 11:
                    break
                
                # Determine position and check if we can add more
                if is_goalkeeper(player) and selected_by_position['GK'] < max_positions['GK']:
                    playing_xi.append(player)
                    selected_by_position['GK'] += 1
                elif is_defender(player) and selected_by_position['DEF'] < max_positions['DEF']:
                    playing_xi.append(player)
                    selected_by_position['DEF'] += 1
                elif is_midfielder(player) and selected_by_position['MID'] < max_positions['MID']:
                    playing_xi.append(player)
                    selected_by_position['MID'] += 1
                elif is_forward(player) and selected_by_position['FWD'] < max_positions['FWD']:
                    playing_xi.append(player)
                    selected_by_position['FWD'] += 1
            
            # Remaining players go to bench
            for player in players:
                if player not in playing_xi:
                    bench.append(player)
            
            # Sort bench by predicted points
            bench.sort(key=lambda x: x.get('predicted_points', 0), reverse=True)
            
            # Post-processing: Swap injured players in playing XI with healthy players from bench
            if playing_xi and bench:
                swaps_made = []
                
                # Find injured/unavailable players in playing XI
                for i, xi_player in enumerate(playing_xi):
                    xi_status = xi_player.get('status', 'a')
                    xi_chance = xi_player.get('chance_of_playing')
                    
                    # Check if player is injured, unavailable, or has very low availability
                    if (xi_status in ['i', 'u'] or 
                        (xi_chance is not None and xi_chance < 25)):
                        
                        # Find a healthy replacement of the same position on the bench
                        for j, bench_player in enumerate(bench):
                            bench_status = bench_player.get('status', 'a')
                            bench_chance = bench_player.get('chance_of_playing')
                            
                            # Check if bench player is healthy and same position
                            if (bench_player.get('position') == xi_player.get('position') and
                                bench_status == 'a' and 
                                (bench_chance is None or bench_chance >= 75)):
                                
                                # Make the swap
                                playing_xi[i], bench[j] = bench_player, xi_player
                                
                                swap_info = f"Swapped {bench_player['name']} (healthy) for {xi_player['name']} "
                                if xi_status == 'i':
                                    swap_info += "(injured)"
                                elif xi_status == 'u': 
                                    swap_info += "(unavailable)"
                                else:
                                    swap_info += f"({xi_chance}% availability)"
                                    
                                swaps_made.append(swap_info)
                                print(f"🔄 {swap_info}")
                                break  # Found a replacement, stop looking
                
                if swaps_made:
                    print(f"✅ Made {len(swaps_made)} injury-related substitutions")
                    # Re-sort bench after swaps
                    bench.sort(key=lambda x: x.get('predicted_points', 0), reverse=True)
            
            # Use enhanced captain selection via team optimizer
            if len(playing_xi) >= 11:
                try:
                    # Create DataFrame for team optimizer
                    import pandas as pd
                    from .utils.team_optimizer import FPLTeamOptimizer
                    
                    # Create DataFrame from playing XI with field mapping for compatibility
                    xi_data = []
                    for player in playing_xi:
                        mapped_player = player.copy()
                        # Map field names for team optimizer compatibility
                        if 'name' in mapped_player and 'web_name' not in mapped_player:
                            mapped_player['web_name'] = mapped_player['name']
                        xi_data.append(mapped_player)
                    
                    xi_df = pd.DataFrame(xi_data)
                    
                    # Use team optimizer for enhanced captain selection
                    optimizer = FPLTeamOptimizer()
                    captain, vice_captain = optimizer._select_captain_and_vice_captain(xi_df)
                except Exception as e:
                    print(f"⚠️  Enhanced captain selection failed: {e}")
                    print("⚠️  Falling back to simple captain selection")
                    # Fallback to simple selection
                    playing_xi_sorted = sorted(playing_xi, key=lambda x: x.get('predicted_points', 0), reverse=True)
                    captain = playing_xi_sorted[0] if playing_xi_sorted else players[0]
                    vice_captain = playing_xi_sorted[1] if len(playing_xi_sorted) > 1 else captain
            else:
                # Fallback to simple selection if not enough players
                captain = playing_xi[0] if playing_xi else players[0]
                vice_captain = playing_xi[1] if len(playing_xi) > 1 else captain
            
            # Calculate formation string using the selected counts
            formation = f"{selected_by_position.get('DEF', 0)}-{selected_by_position.get('MID', 0)}-{selected_by_position.get('FWD', 0)}"
            
            print(f"📋 Formation: {formation} (GK: {selected_by_position.get('GK', 0)})")
            print(f"📊 Playing XI: {len(playing_xi)} players, Bench: {len(bench)} players")
            
            # Calculate totals - ensure we use consistent cost field
            total_cost = 0
            for p in players:
                if 'cost' in p:
                    total_cost += p['cost']
                elif 'price' in p:
                    total_cost += p['price']
                else:
                    print(f"⚠️  Warning: Player {p.get('name', 'Unknown')} missing cost information")
                    total_cost += 5.0  # Emergency fallback
            
            total_predicted_points = sum(p.get('predicted_points', 0) for p in playing_xi) - transfer_cost
            
            # Final budget validation
            if total_cost > self.budget + 0.1:  # Small tolerance for rounding
                print(f"❌ CRITICAL: Team cost £{total_cost:.1f}m exceeds budget £{self.budget:.1f}m by £{total_cost - self.budget:.1f}m!")
                print("🔧 This indicates a serious budget calculation error in transfer optimization")
                # Don't return None, but flag the issue clearly
            
            print(f"💰 Team cost validation: £{total_cost:.1f}m / £{self.budget:.1f}m ({'✅ OK' if total_cost <= self.budget + 0.1 else '❌ OVER BUDGET'})")
            
            return {
                'gameweek': gameweek,
                'playing_xi': playing_xi,
                'bench': bench,
                'captain': captain,
                'vice_captain': vice_captain,
                'formation': formation,
                'total_cost': total_cost,
                'total_predicted_points': total_predicted_points,
                'budget_remaining': self.budget - total_cost,
                'transfers_made': transfers_made,
                'transfer_cost': transfer_cost
            }
            
        except Exception as e:
            print(f"❌ Error creating team from players: {e}")
            return None

    def _organize_team_with_formation(self, players: List[Dict[str, Any]], gameweek: int) -> Optional[Dict[str, Any]]:
        """
        Organize 15 players into optimal formation and select captain/vice-captain
        
        Parameters:
        -----------
        players : list
            List of 15 players
        gameweek : int
            Current gameweek
            
        Returns:
        --------
        team_structure : dict
            Organized team with playing XI, bench, captain, etc.
        """
        try:
            from .utils.team_optimizer import FPLTeamOptimizer
            from .utils.constants import POSITION_MAP
            
            # Create a DataFrame from players
            players_df = pd.DataFrame(players)
            
            # Use team optimizer to select playing XI and captain/vice-captain
            optimizer = FPLTeamOptimizer()
            playing_xi, captain, vice_captain, formation_dict = optimizer.select_playing_xi(players_df)
            
            if playing_xi is None or len(playing_xi) != 11:
                print("❌ Failed to select valid playing XI")
                return None
            
            # Create bench (remaining players)
            playing_xi_ids = set(playing_xi['id'].tolist())
            bench_players = [p for p in players if p['id'] not in playing_xi_ids]
            
            # Sort bench by predicted points
            bench_players.sort(key=lambda x: x['predicted_points'], reverse=True)
            
            # Convert playing XI to list for consistency
            playing_xi_list = playing_xi.to_dict('records')
            
            # Calculate formation string from formation_dict
            formation = f"{formation_dict.get('DEF', 0)}-{formation_dict.get('MID', 0)}-{formation_dict.get('FWD', 0)}"
            
            # Calculate totals
            total_cost = sum(p['cost'] for p in players)
            total_predicted_points = sum(p['predicted_points'] for p in playing_xi_list)
            
            return {
                'gameweek': gameweek,
                'playing_xi': playing_xi_list,
                'bench': bench_players,
                'captain': captain,
                'vice_captain': vice_captain,
                'formation': formation,
                'total_cost': total_cost,
                'total_predicted_points': total_predicted_points,
                'budget_remaining': self.budget - total_cost
            }
            
        except Exception as e:
            print(f"❌ Error organizing team: {e}")
            return None
            print(f"❌ Failed to make prediction for GW{gameweek}: {e}")
            return None
    
    def calculate_transfer_suggestions(self, current_team: Dict[str, Any], new_team: Dict[str, Any], max_transfers: int = 1) -> List[Dict[str, Any]]:
        """
        Calculate suggested transfers between two teams
        
        Parameters:
        -----------
        current_team : dict
            Current team prediction results
        new_team : dict
            New team prediction results
        max_transfers : int
            Maximum number of transfers to suggest
            
        Returns:
        --------
        transfer_suggestions : list
            List of suggested transfers
        """
        if not current_team or not new_team:
            return []
        
        try:
            # Get player lists
            current_players = {p['name']: p for p in current_team['playing_xi']}
            current_bench = {p['name']: p for p in current_team['bench']}
            current_all = {**current_players, **current_bench}
            
            new_players = {p['name']: p for p in new_team['playing_xi']}
            new_bench = {p['name']: p for p in new_team['bench']}
            new_all = {**new_players, **new_bench}
            
            # Find players to transfer out (in current but not in new)
            players_out = [name for name in current_all.keys() if name not in new_all.keys()]
            
            # Find players to transfer in (in new but not in current)
            players_in = [name for name in new_all.keys() if name not in current_all.keys()]
            
            # Calculate transfer values (points gained per transfer)
            transfer_suggestions = []
            
            for out_name in players_out[:max_transfers]:
                for in_name in players_in[:max_transfers]:
                    if len(transfer_suggestions) >= max_transfers:
                        break
                    
                    out_player = current_all[out_name]
                    in_player = new_all[in_name]
                    
                    # Check position compatibility
                    if out_player['position'] == in_player['position']:
                        points_gain = in_player['predicted_points'] - out_player['predicted_points']
                        cost_diff = in_player['cost'] - out_player['cost']
                        
                        transfer_suggestions.append({
                            'transfer_out': out_player,
                            'transfer_in': in_player,
                            'points_gain': points_gain,
                            'cost_difference': cost_diff,
                            'value': points_gain / max(0.1, abs(cost_diff)) if cost_diff != 0 else points_gain * 10
                        })
            
            # Sort by value (points gain per cost)
            transfer_suggestions.sort(key=lambda x: x['value'], reverse=True)
            
            return transfer_suggestions[:max_transfers]
            
        except Exception as e:
            print(f"⚠️  Failed to calculate transfer suggestions: {e}")
            return []
    
    def update_model_with_gameweek(self, gameweek: int) -> Tuple[Optional[FPLPredictionModel], Optional[Dict[str, Any]]]:
        """
        Update model with completed gameweek data
        
        Parameters:
        -----------
        gameweek : int
            Completed gameweek to update with
            
        Returns:
        --------
        model : FPLPredictionModel
            Updated model
        update_info : dict
            Update information
        """
        print(f"\n🔄 Updating model with gameweek {gameweek} data...")
        
        try:
            # Update current season data first
            self.current_season_collector.update_training_data()
            
            # Perform iterative update
            model, update_info = iterative_training_update(
                gameweek=gameweek,
                target=self.target,
                data_dir=self.data_dir
            )
            
            print(f"✅ Model updated with GW{gameweek} data")
            return model, update_info
            
        except Exception as e:
            print(f"❌ Failed to update model with GW{gameweek}: {e}")
            return None, None
    
    def run_season_iteration(self, start_gameweek: int = 1, max_gameweeks: int = 38) -> Optional[Dict[str, Any]]:
        """
        Run the complete iterative season workflow
        
        Parameters:
        -----------
        start_gameweek : int
            Gameweek to start from
        max_gameweeks : int
            Maximum number of gameweeks to process
              Returns:
        --------
        summary : dict
            Season summary statistics
        """
        print("="*60)
        print("FPL ITERATIVE SEASON MANAGER")
        print("="*60)
        print(f"Target: {self.target}")
        print(f"Budget: £{self.budget}m")
        print(f"Starting from gameweek: {start_gameweek}")
        print("="*60)
        
        # Step 0: Check if we're between seasons (all 380 fixtures complete)
        print("\n🔍 Checking season status...")
        current_gw, next_unplayed_gw = self.get_current_gameweek_status()
        
        if current_gw is None and next_unplayed_gw is None:
            # We're between seasons - handle this case
            print("🔄 Between seasons detected!")
            between_seasons_result = self.handle_between_seasons()
            
            # Generate summary for between-seasons scenario
            summary = {
                'between_seasons': True,
                'message': 'Between seasons - predicted for next season GW1',
                'prediction_result': between_seasons_result,
                'total_gameweeks_predicted': 0,
                'total_transfers_made': 0,
                'target_model': self.target,
                'budget_used': self.budget
            }
            
            return summary
        
        # Step 1: Initial model training
        if not self.teams_history:  # Only train initially if no previous state
            print("\n🤖 Step 1: Initial model training...")
            model, training_info = self.initial_model_training()
            if not model:
                print("❌ Initial training failed!")
                return None
        else:
            print(f"\n📂 Resuming from existing state with {len(self.teams_history)} previous predictions")
        
        # Determine starting point
        if self.teams_history:
            last_predicted_gw = max([t['gameweek'] for t in self.teams_history])
            start_gameweek = max(start_gameweek, last_predicted_gw + 1)
            print(f"🔄 Resuming from gameweek {start_gameweek}")
        
        # Step 2: Iterative prediction and training loop
        current_gameweek = start_gameweek
        previous_team = None
        
        # Get previous team if resuming
        if self.teams_history:
            last_team_entry = max(self.teams_history, key=lambda x: x['gameweek'])
            previous_team = last_team_entry['team']
            print(f"📋 Using previous team from GW{last_team_entry['gameweek']} as starting point")
        
        while current_gameweek <= max_gameweeks:
            print(f"\n" + "="*40)
            print(f"PROCESSING GAMEWEEK {current_gameweek}")
            print("="*40)
            
            # Step 2.1: Make prediction for current gameweek with transfer constraints
            current_prediction = self.make_gameweek_prediction(
                gameweek=current_gameweek, 
                previous_team=previous_team,
                max_transfers=1  # Allow 1 free transfer per week
            )
            
            if not current_prediction:
                print(f"❌ Failed to predict GW{current_gameweek}, stopping")
                break
            
            # Step 2.1.5: Display weekly dashboard and generate reports
            print(f"\n📊 Generating weekly dashboard and reports for GW{current_gameweek}...")
            
            # Extract transfers made from the prediction
            transfers_made = []
            if current_prediction.get('transfers_made', 0) > 0:
                # Extract actual transfer details from the prediction
                transfer_details = current_prediction.get('transfer_details', [])
                transfers_made = []
                for transfer in transfer_details:
                    transfers_made.append({
                        'transfer_out': {
                            'name': transfer['out']['name'],
                            'position': transfer['out']['position'],
                            'team': transfer['out']['team'],
                            'cost': transfer['out']['cost'],
                            'predicted_points': transfer['out']['predicted_points']
                        },
                        'transfer_in': {
                            'name': transfer['in']['name'],
                            'position': transfer['in']['position'],
                            'team': transfer['in']['team'],
                            'cost': transfer['in']['cost'],
                            'predicted_points': transfer['in']['predicted_points']
                        },
                        'points_gain': transfer['points_gain'],
                        'cost_diff': transfer['cost_diff'],
                        'priority_type': transfer.get('priority_type', 'healthy'),
                        'smart_analysis': transfer.get('smart_analysis')
                    })
            
            # Display dashboard
            self.display_weekly_dashboard(
                gameweek=current_gameweek,
                team_prediction=current_prediction,
                previous_team=previous_team,
                transfers_made=transfers_made
            )
            
            # Generate reports
            reports = self.generate_weekly_reports(
                gameweek=current_gameweek,
                team_prediction=current_prediction,
                previous_team=previous_team,
                transfers_made=transfers_made
            )

            # Record any transfers made in the prediction process
            if hasattr(current_prediction, 'transfers_made') and current_prediction.get('transfers_made', 0) > 0:
                transfer_info = {
                    'gameweek': current_gameweek,
                    'transfers_made': current_prediction.get('transfers_made', 0),
                    'transfer_cost': current_prediction.get('transfer_cost', 0),
                    'timestamp': datetime.now().isoformat()
                }
                self.transfers_history.append(transfer_info)
                
                # Record transfers for performance tracking
                if transfers_made:
                    total_gain = sum(t.get('points_gain', 0) for t in transfers_made)
                    self.performance_tracker.record_transfer(
                        gameweek=current_gameweek,
                        transfers=transfers_made,
                        transfer_cost=current_prediction.get('transfer_cost', 0),
                        points_gain=total_gain
                    )
            
            # Step 2.2: Check if this gameweek has been played
            if self.has_gameweek_been_played(current_gameweek):
                print(f"✅ Gameweek {current_gameweek} has been completed")
                
                # Step 2.3: Update model with completed gameweek data
                model, update_info = self.update_model_with_gameweek(current_gameweek)
                
                if update_info:
                    print(f"📈 Model updated - Loss: {update_info.get('final_loss', 'N/A'):.4f}")
                
                # Move to next gameweek
                previous_team = current_prediction
                current_gameweek += 1
                
                # Save state after each completed gameweek
                self.save_season_state()
                
            else:
                print(f"⏸️  Gameweek {current_gameweek} has not been played yet")
                print("🏁 Stopping iterative process until gameweek is completed")
                
                # Save final state
                self.save_season_state()
                break
        
        # Step 3: Generate season summary
        summary = self.generate_season_summary()
        return summary
    
    def generate_season_summary(self) -> Dict[str, Any]:
        """
        Generate summary statistics for the season
        
        Returns:
        --------
        summary : dict
            Season summary statistics
        """
        if not self.predictions_history:
            return {'message': 'No predictions made yet'}
        
        summary = {
            'total_gameweeks_predicted': len(self.predictions_history),
            'total_transfers_made': len(self.transfers_history),
            'target_model': self.target,
            'budget_used': self.budget,
            'predictions': []
        }
        
        total_predicted_points = 0
        gameweeks_predicted = []
        
        for pred in self.predictions_history:
            total_predicted_points += pred['predicted_points']
            gameweeks_predicted.append(pred['gameweek'])
            
            summary['predictions'].append({
                'gameweek': pred['gameweek'],
                'predicted_points': pred['predicted_points'],
                'team_cost': pred['team_cost'],
                'captain': pred['captain'],
                'formation': pred['formation']
            })
        
        summary['average_predicted_points'] = total_predicted_points / len(self.predictions_history)
        summary['gameweeks_covered'] = f"{min(gameweeks_predicted)}-{max(gameweeks_predicted)}" if gameweeks_predicted else "None"
        summary['last_updated'] = datetime.now().isoformat()
        
        return summary
    
    def print_season_summary(self) -> None:
        """Print a formatted season summary"""
        summary = self.generate_season_summary()
        
        print("\n" + "="*60)
        print("SEASON SUMMARY")
        print("="*60)
        print(f"Target Model: {summary.get('target_model', 'N/A')}")
        print(f"Budget: £{summary.get('budget_used', 0)}m")
        print(f"Gameweeks Predicted: {summary.get('total_gameweeks_predicted', 0)}")
        print(f"Transfers Made: {summary.get('total_transfers_made', 0)}")
        print(f"Average Predicted Points: {summary.get('average_predicted_points', 0):.1f}")
        print(f"Gameweeks Covered: {summary.get('gameweeks_covered', 'None')}")
        
        if self.predictions_history:
            print(f"\nRecent Predictions:")
            for pred in self.predictions_history[-5:]:  # Last 5 predictions
                print(f"  GW{pred['gameweek']}: {pred['predicted_points']:.1f} pts, Captain: {pred['captain']}")
        
        if self.transfers_history:
            print(f"\nRecent Transfers:")
            for transfer in self.transfers_history[-3:]:  # Last 3 transfers
                gw = transfer['gameweek']
                if transfer['transfers']:
                    t = transfer['transfers'][0]
                    print(f"  GW{gw}: {t['transfer_out']['name']} → {t['transfer_in']['name']} ({t['points_gain']:.1f} pts)")
        
        print("="*60)
    
    def display_weekly_dashboard(self, gameweek: int, team_prediction: Dict[str, Any], 
                               previous_team: Optional[Dict[str, Any]] = None,
                               transfers_made: List[Dict[str, Any]] = None) -> None:
        """Display the weekly summary dashboard."""
        self.weekly_dashboard.print_weekly_summary(
            gameweek=gameweek,
            team_prediction=team_prediction,
            previous_team=previous_team,
            transfers_made=transfers_made
        )
    
    def generate_weekly_reports(self, gameweek: int, team_prediction: Dict[str, Any], 
                              previous_team: Optional[Dict[str, Any]] = None,
                              transfers_made: List[Dict[str, Any]] = None) -> Dict[str, str]:
        """Generate weekly reports (HTML and text)."""
        reports = {}
        
        # Generate HTML report
        try:
            html_report = self.weekly_reports.generate_html_report(
                gameweek=gameweek,
                team_prediction=team_prediction,
                previous_team=previous_team,
                transfers_made=transfers_made
            )
            reports['html'] = html_report
            print(f"📄 HTML report generated: {html_report}")
        except Exception as e:
            print(f"⚠️  Failed to generate HTML report: {e}")
        
        # Generate text report
        try:
            text_report = self.weekly_dashboard.save_weekly_summary(
                gameweek=gameweek,
                team_prediction=team_prediction,
                previous_team=previous_team,
                transfers_made=transfers_made
            )
            reports['text'] = text_report
            print(f"📄 Text report generated: {text_report}")
        except Exception as e:
            print(f"⚠️  Failed to generate text report: {e}")
        
        return reports
    
    def analyze_transfer_with_smart_logic(self, out_player: Dict[str, Any], 
                                        in_player: Dict[str, Any], 
                                        gameweek: int) -> Dict[str, Any]:
        """Analyze transfer with smart logic including fixture difficulty and form trends."""
        return self.smart_transfer_analyzer.analyze_transfer_opportunity(
            out_player=out_player,
            in_player=in_player,
            gameweek=gameweek
        )


def run_season_manager(
    data_dir: str = "data",
    target: str = 'points_scored',
    budget: float = 100.0,
    start_gameweek: int = 1,
    initial_training_epochs: int = 100
) -> Optional[Dict[str, Any]]:
    """
    Run the complete iterative season management system
    
    Parameters:
    -----------
    data_dir : str
        Data directory path
    target : str
        Target variable for predictions
    budget : float
        Team budget in millions
    start_gameweek : int
        Gameweek to start from
    initial_training_epochs : int
        Epochs for initial training
        
    Returns:
    --------
    summary : dict
        Season summary
    """
    manager = FPLIterativeSeasonManager(
        data_dir=data_dir,
        budget=budget,
        target=target
    )
    
    try:
        summary = manager.run_season_iteration(
            start_gameweek=start_gameweek,
            max_gameweeks=38
        )
        
        # Print summary
        manager.print_season_summary()
        
        return summary
        
    except KeyboardInterrupt:
        print("\n⏹️  Process interrupted by user")
        manager.save_season_state()
        manager.print_season_summary()
        return manager.generate_season_summary()
    
    except Exception as e:
        print(f"❌ Season management failed: {e}")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run FPL iterative season management')
    parser.add_argument('--data-dir', default='data', help='Data directory path')
    parser.add_argument('--target', default='points_scored', 
                       choices=['points_scored', 'goals_scored', 'assists', 'minutes_played'],
                       help='Target variable for predictions')
    parser.add_argument('--budget', type=float, default=100.0, help='Team budget in millions')
    parser.add_argument('--start-gameweek', type=int, default=1, help='Gameweek to start from')
    parser.add_argument('--initial-epochs', type=int, default=100, help='Epochs for initial training')
    
    args = parser.parse_args()
    
    summary = run_season_manager(
        data_dir=args.data_dir,
        target=args.target,
        budget=args.budget,
        start_gameweek=args.start_gameweek,
        initial_training_epochs=args.initial_epochs
    )
    
    if summary:
        print(f"\n✅ Season management completed successfully")
    else:
        print(f"\n❌ Season management failed")
