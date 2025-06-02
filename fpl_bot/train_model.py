import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .utils.data_processing import FPLDataProcessor
from .models.cnn_model import FPLPredictionModel, train_model_with_data

def train_model(epochs=100, batch_size=32, lookback=3, n_features=14, cutoff_gw=None):
    """
    Train CNN model for FPL points prediction using enhanced preprocessing if available
    
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
    
    # Check for enhanced preprocessed data first
    enhanced_train_path = os.path.join("data", "processed", "enhanced_X_train.npy")
    enhanced_val_path = os.path.join("data", "processed", "enhanced_X_val.npy")
    enhanced_y_train_path = os.path.join("data", "processed", "enhanced_y_train.npy")
    enhanced_y_val_path = os.path.join("data", "processed", "enhanced_y_val.npy")
    
    if (os.path.exists(enhanced_train_path) and os.path.exists(enhanced_val_path) and 
        os.path.exists(enhanced_y_train_path) and os.path.exists(enhanced_y_val_path)):
        
        print("Found enhanced preprocessed data. Using it for model training...")
        import numpy as np
        
        try:
            # Load enhanced data
            X_train = np.load(enhanced_train_path, allow_pickle=True)
            X_val = np.load(enhanced_val_path, allow_pickle=True)
            y_train = np.load(enhanced_y_train_path, allow_pickle=True)
            y_val = np.load(enhanced_y_val_path, allow_pickle=True)
            
            print(f"Enhanced data loaded:")
            print(f"  Training: {X_train.shape[0]} samples, {X_train.shape[1]} timesteps, {X_train.shape[2]} features")
            print(f"  Validation: {X_val.shape[0]} samples")
            
            # Adjust parameters to match data
            if X_train.shape[1] != lookback:
                lookback = X_train.shape[1]
                print(f"Adjusted lookback to match data: {lookback}")
            
            n_features = X_train.shape[2]
            print(f"Using {n_features} features from enhanced preprocessing")
            
            # Train with enhanced data
            model, history = train_model_with_data(
                X_train, y_train, X_val, y_val,
                lookback=lookback, n_features=n_features,
                epochs=epochs, batch_size=batch_size
            )
            
            # Save as enhanced model
            model.save(model_type="enhanced_historical")
            
            # Enhanced plotting
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss (Enhanced)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(1, 3, 2)
            plt.plot(history.history['mae'], label='Training MAE')
            plt.plot(history.history['val_mae'], label='Validation MAE')
            plt.title('Mean Absolute Error (Enhanced)')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
            
            # Feature importance plot placeholder
            plt.subplot(1, 3, 3)
            feature_names_path = os.path.join("data", "processed", "enhanced_feature_names.npy")
            if os.path.exists(feature_names_path):
                feature_names = np.load(feature_names_path, allow_pickle=True)
                plt.text(0.1, 0.5, f'Enhanced Model\n{len(feature_names)} features\nLookback: {lookback}', 
                        transform=plt.gca().transAxes, fontsize=12)
            else:
                plt.text(0.1, 0.5, f'Enhanced Model\n{n_features} features\nLookback: {lookback}', 
                        transform=plt.gca().transAxes, fontsize=12)
            plt.title('Model Info')
            plt.axis('off')
            
            plt.tight_layout()
            
            # Save the plot
            os.makedirs('data/plots', exist_ok=True)
            plt.savefig('data/plots/training_history_enhanced.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("Enhanced model training complete!")
            return model
            
        except Exception as e:
            print(f"Error using enhanced data: {e}")
            print("Falling back to standard processing...")
    
    # Fallback to standard processing
    print("Using standard data processing pipeline...")
    
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
    Train CNN model using enhanced FPL data preprocessing
    
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
    print("Starting model training with enhanced data preprocessing...")
    if not seasons:
        print("No seasons specified, using defaults")
        from .utils.history_data_collector import FPLHistoricalDataCollector
        collector = FPLHistoricalDataCollector()
        seasons = collector.get_available_seasons()[-2:]  # Use last 2 seasons by default
    
    print(f"Using seasons: {', '.join(seasons)}")
    
    # Check if enhanced pre-processed data exists
    import os
    
    # Try loading enhanced preprocessed data first
    enhanced_train_path = os.path.join("data", "processed", "enhanced_X_train.npy")
    enhanced_val_path = os.path.join("data", "processed", "enhanced_X_val.npy")
    enhanced_y_train_path = os.path.join("data", "processed", "enhanced_y_train.npy")
    enhanced_y_val_path = os.path.join("data", "processed", "enhanced_y_val.npy")
    
    if (os.path.exists(enhanced_train_path) and os.path.exists(enhanced_val_path) and 
        os.path.exists(enhanced_y_train_path) and os.path.exists(enhanced_y_val_path)):
        
        print("Found enhanced preprocessed training data. Using it for model training...")
        import numpy as np
        
        try:
            # Load enhanced preprocessed data
            X_train = np.load(enhanced_train_path, allow_pickle=True)
            X_val = np.load(enhanced_val_path, allow_pickle=True)
            y_train = np.load(enhanced_y_train_path, allow_pickle=True)
            y_val = np.load(enhanced_y_val_path, allow_pickle=True)
            
            print(f"Successfully loaded enhanced data:")
            print(f"  Training: {X_train.shape[0]} samples, {X_train.shape[1]} timesteps, {X_train.shape[2]} features")
            print(f"  Validation: {X_val.shape[0]} samples, {X_val.shape[1]} timesteps, {X_val.shape[2]} features")
            
            # Adjust lookback if necessary
            if X_train.shape[1] != lookback:
                if X_train.shape[1] > lookback:
                    print(f"Trimming timesteps from {X_train.shape[1]} to {lookback}")
                    X_train = X_train[:, -lookback:, :]
                    X_val = X_val[:, -lookback:, :]
                else:
                    print(f"Warning: Data has {X_train.shape[1]} timesteps but model expects {lookback}")
                    lookback = X_train.shape[1]
            
            # Update n_features to match data
            n_features = X_train.shape[2]
            print(f"Using {n_features} features for model training")
            
            # Train model with enhanced data
            from .models.cnn_model import train_model_with_data
            model, history = train_model_with_data(
                X_train, y_train, 
                X_val=X_val, y_val=y_val,
                lookback=lookback, 
                n_features=n_features,
                epochs=epochs, 
                batch_size=batch_size
            )
            
            # Save with enhanced historical type
            model.save(model_type="enhanced_historical")
            
            # Load and display feature names if available
            feature_names_path = os.path.join("data", "processed", "enhanced_feature_names.npy")
            if os.path.exists(feature_names_path):
                feature_names = np.load(feature_names_path, allow_pickle=True)
                print(f"Model trained with {len(feature_names)} features:")
                print(f"  Top features: {list(feature_names[:10])}")
            
            # Plot enhanced training history
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss (Enhanced Data)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(1, 3, 2)
            plt.plot(history.history['mae'], label='Training MAE')
            plt.plot(history.history['val_mae'], label='Validation MAE')
            plt.title('Mean Absolute Error (Enhanced Data)')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
            
            # Add R² score if available
            if 'r2_score' in history.history:
                plt.subplot(1, 3, 3)
                plt.plot(history.history['r2_score'], label='Training R²')
                if 'val_r2_score' in history.history:
                    plt.plot(history.history['val_r2_score'], label='Validation R²')
                plt.title('R² Score (Enhanced Data)')
                plt.xlabel('Epoch')
                plt.ylabel('R² Score')
                plt.legend()
            
            plt.tight_layout()
            
            # Save the enhanced plot
            os.makedirs('data/plots', exist_ok=True)
            seasons_str = '_'.join(s.replace('-', '') for s in seasons)
            plt.savefig(f'data/plots/training_history_enhanced_{seasons_str}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("Enhanced model training complete!")
            print(f"Model saved as: enhanced_historical")
            print(f"Training plot saved as: training_history_enhanced_{seasons_str}.png")
            
            return model
            
        except Exception as e:
            print(f"Error loading enhanced preprocessed data: {e}")
            print("Falling back to standard processing.")
    
    # Fallback to standard processing if enhanced data not available
    print("Enhanced data not found. Checking for standard pre-processed data...")
    
    # Check for standard preprocessed data
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