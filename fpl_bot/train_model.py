import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .utils.data_collection import FPLDataProcessor
from .utils.current_season_collector import FPLCurrentSeasonCollector
from .models.fpl_model import FPLPredictionModel, train_model_with_processed_data
from datetime import datetime

def train_model(target='points_scored', epochs=100, batch_size=32, include_current_season=True, 
                historical_seasons=None, data_dir="data", verbose=1):
    """
    Train FPL prediction model with historical and current season data
    
    Parameters:
    -----------
    target : str
        Target variable to predict ('points_scored', 'goals_scored', etc.)
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    include_current_season : bool
        Whether to include current season data
    historical_seasons : list, optional
        List of historical seasons to include
    data_dir : str
        Data directory path
    verbose : int
        Verbosity level
        
    Returns:
    --------
    model : FPLPredictionModel
        Trained model
    training_info : dict
        Information about training process
    """
    print("="*60)
    print("FPL MODEL TRAINING")
    print("="*60)
    print(f"Target: {target}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Include Current Season: {include_current_season}")
    
    training_info = {
        'target': target,
        'epochs': epochs,
        'batch_size': batch_size,
        'include_current_season': include_current_season,
        'start_time': datetime.now(),
        'seasons_used': []
    }
    
    # Step 1: Update current season data if requested
    if include_current_season:
        print("\n📡 Updating current season data...")
        current_collector = FPLCurrentSeasonCollector(data_dir=data_dir)
        current_collector.update_training_data()
        training_info['seasons_used'].append('2024-25')  # Current season
    
    # Step 2: Process data including current season
    print("\n⚙️  Processing training data...")
    processor = FPLDataProcessor(data_dir=data_dir)
    
    # Determine seasons to process
    if historical_seasons is None:
        # Use all available seasons
        seasons_to_process = None
    else:
        seasons_to_process = historical_seasons
        training_info['seasons_used'].extend(historical_seasons)
    
    # Add current season to processing if included
    if include_current_season:
        if seasons_to_process is None:
            seasons_to_process = ['2024-25']
        else:
            seasons_to_process = list(seasons_to_process) + ['2024-25']
    
    # Process all data
    final_dataset, datasets = processor.process_all_data(
        seasons=seasons_to_process,
        target_columns=[target]
    )
    
    if final_dataset is None:
        raise ValueError("Failed to process training data")
    
    print(f"✅ Processed {len(final_dataset)} samples")
    training_info['samples_processed'] = len(final_dataset)
    
    # Step 3: Train model
    print(f"\n🤖 Training model for {target}...")
    model, history = train_model_with_processed_data(
        data_dir=data_dir,
        target=target,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Step 4: Save training information
    training_info['end_time'] = datetime.now()
    training_info['training_duration'] = training_info['end_time'] - training_info['start_time']
    training_info['final_loss'] = history.history['loss'][-1]
    training_info['final_val_loss'] = history.history['val_loss'][-1]
    training_info['final_mae'] = history.history['mae'][-1]
    training_info['final_val_mae'] = history.history['val_mae'][-1]
    
    # Save training info
    info_path = os.path.join(data_dir, 'models', f'training_info_{target}.json')
    os.makedirs(os.path.dirname(info_path), exist_ok=True)
    
    import json
    # Convert datetime objects to strings for JSON serialization
    info_to_save = training_info.copy()
    info_to_save['start_time'] = training_info['start_time'].isoformat()
    info_to_save['end_time'] = training_info['end_time'].isoformat()
    info_to_save['training_duration'] = str(training_info['training_duration'])
    
    with open(info_path, 'w') as f:
        json.dump(info_to_save, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"📊 Final Training Loss: {training_info['final_loss']:.4f}")
    print(f"📊 Final Validation Loss: {training_info['final_val_loss']:.4f}")
    print(f"📊 Final Training MAE: {training_info['final_mae']:.4f}")
    print(f"📊 Final Validation MAE: {training_info['final_val_mae']:.4f}")
    print(f"⏱️  Training Duration: {training_info['training_duration']}")
    print(f"💾 Model saved for target: {target}")
    
    return model, training_info

def iterative_training_update(gameweek=None, target='points_scored', data_dir="data"):
    """
    Update model training with new gameweek data (iterative learning)
    
    Parameters:
    -----------
    gameweek : int, optional
        Specific gameweek to update with (if None, uses latest available)
    target : str
        Target variable model to update
    data_dir : str
        Data directory path
        
    Returns:
    --------
    model : FPLPredictionModel
        Updated model
    update_info : dict
        Information about the update
    """
    print("="*60)
    print("ITERATIVE TRAINING UPDATE")
    print("="*60)
    
    update_info = {
        'target': target,
        'gameweek': gameweek,
        'update_time': datetime.now()
    }
    
    # Step 1: Collect latest current season data
    print("📡 Collecting latest gameweek data...")
    current_collector = FPLCurrentSeasonCollector(data_dir=data_dir)
    current_data = current_collector.collect_current_season_data()
    
    # Get current/latest gameweek
    if gameweek is None:
        current_gw, bootstrap = current_collector.get_current_gameweek()
        if current_gw:
            gameweek = current_gw['id']
            update_info['gameweek'] = gameweek
            print(f"📅 Updating with gameweek {gameweek}")
        else:
            print("⚠️  No current gameweek found")
            return None, update_info
    
    # Step 2: Convert and integrate new data
    print("⚙️  Converting new data to training format...")
    converted_data = current_collector.convert_to_training_format(current_data)
    
    # Step 3: Reprocess all data including new gameweek
    print("🔄 Reprocessing training data with updated gameweek...")
    processor = FPLDataProcessor(data_dir=data_dir)
    
    # Include current season in processing
    final_dataset, datasets = processor.process_all_data(
        seasons=['2024-25'],  # Focus on current season for iterative update
        target_columns=[target]
    )
    
    if final_dataset is None:
        print("❌ Failed to reprocess data")
        return None, update_info
    
    print(f"✅ Reprocessed {len(final_dataset)} samples")
    update_info['samples_reprocessed'] = len(final_dataset)
    
    # Step 4: Load existing model and retrain
    print(f"🤖 Updating model for {target}...")
    try:
        # Load existing model
        model = FPLPredictionModel()
        model.load(f'fpl_model_{target}.h5')
        print("📂 Loaded existing model")
        
        # Retrain with updated data (fewer epochs for iterative update)
        _, history = train_model_with_processed_data(
            data_dir=data_dir,
            target=target,
            epochs=20,  # Fewer epochs for iterative updates
            batch_size=32
        )
        
        update_info['update_successful'] = True
        update_info['final_loss'] = history.history['loss'][-1]
        update_info['final_val_loss'] = history.history['val_loss'][-1]
        
    except Exception as e:
        print(f"❌ Failed to update model: {e}")
        print("🔄 Training new model instead...")
        
        # If loading fails, train new model
        model, history = train_model_with_processed_data(
            data_dir=data_dir,
            target=target,
            epochs=50,
            batch_size=32
        )
        
        update_info['update_successful'] = False
        update_info['new_model_trained'] = True
        update_info['final_loss'] = history.history['loss'][-1]
        update_info['final_val_loss'] = history.history['val_loss'][-1]
    
    print("\n" + "="*60)
    print("ITERATIVE UPDATE COMPLETE!")
    print("="*60)
    print(f"📅 Updated with gameweek: {gameweek}")
    print(f"💾 Model updated for target: {target}")
    
    return model, update_info

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train FPL prediction model')
    parser.add_argument('--target', type=str, default='points_scored',
                       choices=['points_scored', 'goals_scored', 'assists', 'minutes_played'],
                       help='Target variable to predict')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--no-current-season', action='store_true', 
                       help='Exclude current season data from training')
    parser.add_argument('--seasons', nargs='+', help='Specific historical seasons to include')
    parser.add_argument('--data-dir', default='data', help='Data directory path')
    parser.add_argument('--iterative-update', action='store_true',
                       help='Perform iterative update instead of full training')
    parser.add_argument('--gameweek', type=int, help='Specific gameweek for iterative update')
    
    args = parser.parse_args()
    
    if args.iterative_update:
        # Perform iterative update
        model, info = iterative_training_update(
            gameweek=args.gameweek,
            target=args.target,
            data_dir=args.data_dir
        )
    else:
        # Perform full training
        model, info = train_model(
            target=args.target,
            epochs=args.epochs,
            batch_size=args.batch_size,
            include_current_season=not args.no_current_season,
            historical_seasons=args.seasons,
            data_dir=args.data_dir
        )
