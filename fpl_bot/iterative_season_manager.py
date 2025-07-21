import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from .train_model import train_model, iterative_training_update
from .predict_team import predict_team_for_gameweek
from .utils.current_season_collector import FPLCurrentSeasonCollector
from .utils.team_optimizer import FPLTeamOptimizer

ITERATIVE_DATA_PATH = Path("iterative_season_state.json")

class FPLIterativeSeasonManager:
    """
    Manages iterative training and prediction throughout a season.
    
    This class handles:
    1. Training models on historical data
    2. Making predictions for each gameweek
    3. Updating models with new data as gameweeks complete
    4. Making transfer decisions based on prediction differences
    """
    
    def __init__(self, data_dir="data", budget=100.0, target='points_scored'):
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
        """
        self.data_dir = data_dir
        self.budget = budget
        self.target = target
        self.current_season_collector = FPLCurrentSeasonCollector(data_dir=data_dir)
        self.team_optimizer = FPLTeamOptimizer(total_budget=budget)
        
        # Track season progress
        self.season_state_file = os.path.join(data_dir, ITERATIVE_DATA_PATH)
        self.predictions_history = []
        self.teams_history = []
        self.transfers_history = []
        
        # Load existing state if available
        self.load_season_state()
    
    def load_season_state(self):
        """Load existing season state from disk"""
        try:
            if os.path.exists(self.season_state_file):
                with open(self.season_state_file, 'r') as f:
                    state = json.load(f)
                
                self.predictions_history = state.get('predictions_history', [])
                self.teams_history = state.get('teams_history', [])
                self.transfers_history = state.get('transfers_history', [])
                
                print(f"📂 Loaded season state: {len(self.teams_history)} teams, {len(self.transfers_history)} transfers")
            else:
                print("📂 No existing season state found - starting fresh")
        except Exception as e:
            print(f"⚠️  Failed to load season state: {e}")
    
    def save_season_state(self):
        """Save current season state to disk"""
        try:
            state = {
                'predictions_history': self.predictions_history,
                'teams_history': self.teams_history,
                'transfers_history': self.transfers_history,
                'last_updated': datetime.now().isoformat(),                'target': self.target,
                'budget': self.budget
            }
            
            os.makedirs(os.path.dirname(self.season_state_file), exist_ok=True)
            with open(self.season_state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            print(f"💾 Saved season state to {self.season_state_file}")
        except Exception as e:
            print(f"⚠️  Failed to save season state: {e}")
    
    def get_current_gameweek_status(self):
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
    
    def has_gameweek_been_played(self, gameweek):
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
    
    def handle_between_seasons(self):
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
            import traceback
            traceback.print_exc()
            return None

    def initial_model_training(self, historical_seasons=None, epochs=100):
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
    
    def make_gameweek_prediction(self, gameweek):
        """
        Make team prediction for a specific gameweek
        
        Parameters:
        -----------
        gameweek : int
            Gameweek to predict for
            
        Returns:
        --------
        prediction_results : dict
            Prediction results including team selection
        """
        print(f"\n🔮 Making prediction for gameweek {gameweek}...")
        
        try:
            prediction_results = predict_team_for_gameweek(
                gameweek=gameweek,
                budget=self.budget,
                target=self.target,
                data_dir=self.data_dir,
                save_results=False  # We'll manage saving ourselves
            )
            
            if prediction_results:
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
    
    def calculate_transfer_suggestions(self, current_team, new_team, max_transfers=1):
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
    
    def update_model_with_gameweek(self, gameweek):
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
    
    def run_season_iteration(self, start_gameweek=1, max_gameweeks=38):
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
        
        while current_gameweek <= max_gameweeks:
            print(f"\n" + "="*40)
            print(f"PROCESSING GAMEWEEK {current_gameweek}")
            print("="*40)
            
            # Step 2.1: Make prediction for current gameweek
            current_prediction = self.make_gameweek_prediction(current_gameweek)
            
            if not current_prediction:
                print(f"❌ Failed to predict GW{current_gameweek}, stopping")
                break
            
            # Step 2.2: Calculate transfers if we have a previous team
            if previous_team:
                transfer_suggestions = self.calculate_transfer_suggestions(
                    previous_team, current_prediction, max_transfers=1
                )
                
                if transfer_suggestions:
                    print(f"\n💱 Transfer suggestion for GW{current_gameweek}:")
                    for i, transfer in enumerate(transfer_suggestions, 1):
                        print(f"  {i}. OUT: {transfer['transfer_out']['name']} ({transfer['transfer_out']['team']})")
                        print(f"     IN:  {transfer['transfer_in']['name']} ({transfer['transfer_in']['team']})")
                        print(f"     Points gain: {transfer['points_gain']:.1f}")
                        print(f"     Cost diff: £{transfer['cost_difference']:.1f}m")
                    
                    # Record transfer
                    self.transfers_history.append({
                        'gameweek': current_gameweek,
                        'transfers': transfer_suggestions,
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    print(f"\n💱 No beneficial transfers found for GW{current_gameweek}")
            
            # Step 2.3: Check if this gameweek has been played
            if self.has_gameweek_been_played(current_gameweek):
                print(f"✅ Gameweek {current_gameweek} has been completed")
                
                # Step 2.4: Update model with completed gameweek data
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
    
    def generate_season_summary(self):
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
    
    def print_season_summary(self):
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


def run_season_manager(data_dir="data", target='points_scored', budget=100.0, 
                      start_gameweek=1, initial_training_epochs=100):
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
