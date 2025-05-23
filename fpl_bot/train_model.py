import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .utils.data_processing import FPLDataProcessor
from .models.cnn_model import FPLPredictionModel, train_model_with_data

def train_model(epochs=100, batch_size=32, lookback=3, n_features=14, cutoff_gw=None):
    """
    Train CNN model for FPL points prediction
    
    Parameters:
    -----------
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    lookback : int
        Number of gameweeks to use for input features
    n_features : int
        Number of features for the model
    cutoff_gw : int
        If provided, only use data up to this gameweek for training
        
    Returns:
    --------
    model : FPLPredictionModel
        Trained model
    """
    print("Starting model training...")
    
    # Initialize data processor with cutoff
    data_processor = FPLDataProcessor(cutoff_gw=cutoff_gw)
    
    # Prepare data with cutoff applied
    X, y, player_ids = data_processor.prepare_training_data(lookback=lookback)
    
    print(f"Training with {X.shape[0]} samples, {X.shape[1]} timesteps, and {X.shape[2]} features")
    
    # Train model
    model, history = train_model_with_data(X, y, lookback=lookback, n_features=n_features)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('data/plots', exist_ok=True)
    plt.savefig('data/plots/training_history.png')
    plt.close()
    
    print("Model training complete!")
    return model

def train_model_with_history(seasons=None, epochs=50, batch_size=32, lookback=3, n_features=14):
    """
    Train CNN model using historical FPL data from multiple seasons
    
    Parameters:
    -----------
    seasons : list
        List of seasons to use for training (e.g. ['2021-22', '2022-23'])
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    lookback : int
        Number of gameweeks to use for input features
    n_features : int
        Number of features for the model
        
    Returns:
    --------
    model : FPLPredictionModel
        Trained model
    """
    print("Starting model training with historical data...")
    if not seasons:
        print("No seasons specified, using defaults")
        from .utils.history_data_collector import FPLHistoricalDataCollector
        collector = FPLHistoricalDataCollector()
        seasons = collector.get_available_seasons()[-2:]  # Use last 2 seasons by default
    
    print(f"Using seasons: {', '.join(seasons)}")
    
    # Check if enhanced pre-processed data exists
    import os
    x_train_path = os.path.join("data", "processed", "X_train.npy")
    y_train_path = os.path.join("data", "processed", "y_train.npy")
    
    if os.path.exists(x_train_path) and os.path.exists(y_train_path):
        print("Found pre-processed enhanced training data. Using it for model training...")
        import numpy as np
        
        # Load the pre-processed data with allow_pickle=True to handle object arrays
        try:
            X = np.load(x_train_path, allow_pickle=True)
            y = np.load(y_train_path, allow_pickle=True)
            
            # Check if the data has the expected format
            if X.ndim == 3 and X.shape[1] >= lookback:
                print(f"Successfully loaded pre-processed data: {X.shape[0]} samples, {X.shape[1]} timesteps, {X.shape[2]} features")
                
                # If the timesteps are more than what we need, trim to match lookback
                if X.shape[1] > lookback:
                    print(f"Trimming timesteps from {X.shape[1]} to {lookback}")
                    X = X[:, -lookback:, :]
                
                # Use the loaded data for training
                from .models.cnn_model import train_model_with_data, FPLPredictionModel
                model, history = train_model_with_data(X, y, lookback=lookback, n_features=X.shape[2], 
                                                     epochs=epochs, batch_size=batch_size)
                
                # Save with historical type
                model.save(model_type="historical")
                
                # Plot training history
                import matplotlib.pyplot as plt
                
                plt.figure(figsize=(12, 5))
                
                plt.subplot(1, 2, 1)
                plt.plot(history.history['loss'], label='Training Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.title('Loss')
                plt.xlabel('Epoch')
                plt.legend()
                
                plt.subplot(1, 2, 2)
                plt.plot(history.history['mae'], label='Training MAE')
                plt.plot(history.history['val_mae'], label='Validation MAE')
                plt.title('Mean Absolute Error')
                plt.xlabel('Epoch')
                plt.legend()
                
                plt.tight_layout()
                
                # Save the plot with seasons in filename
                os.makedirs('data/plots', exist_ok=True)
                seasons_str = '_'.join(s.replace('-', '') for s in seasons)
                plt.savefig(f'data/plots/training_history_historical_{seasons_str}.png')
                plt.close()
                
                print("Model training with enhanced historical data complete!")
                return model
        except Exception as e:
            print(f"Error loading pre-processed data: {e}")
            print("Falling back to standard processing.")
    else:
        print("No pre-processed enhanced data found. Using standard data processing.")
    
    # Initialize multi-season data processor
    from .utils.data_processing import MultiSeasonDataProcessor
    data_processor = MultiSeasonDataProcessor(seasons=seasons, lookback=lookback)
    
    # Prepare data from historical seasons
    X, y, player_ids = data_processor.prepare_multi_season_training_data()
    
    # Check if we got valid data
    if X is None or y is None or len(X) == 0 or len(y) == 0:
        print("ERROR: Failed to process historical data")
        # Fallback to current season
        print("Falling back to current season data only")
        from .utils.data_processing import FPLDataProcessor
        current_processor = FPLDataProcessor(lookback=lookback)
        X, y, player_ids = current_processor.prepare_training_data(lookback=lookback)
    
    print(f"Training with {X.shape[0]} samples, {X.shape[1]} timesteps, and {X.shape[2]} features")
    
    # Train model using existing utility function
    from .models.cnn_model import train_model_with_data
    model, history = train_model_with_data(X, y, lookback=lookback, n_features=n_features,
                                         epochs=epochs, batch_size=batch_size)
    
    # Save with historical type
    model.save(model_type="historical")
    
    # Plot training history
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot with seasons in filename
    os.makedirs('data/plots', exist_ok=True)
    seasons_str = '_'.join(s.replace('-', '') for s in seasons)
    plt.savefig(f'data/plots/training_history_historical_{seasons_str}.png')
    plt.close()
    
    print("Model training with historical data complete!")
    return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CNN model for FPL points prediction')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    
    args = parser.parse_args()
    
    train_model(epochs=args.epochs, batch_size=args.batch_size)