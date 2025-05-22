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

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CNN model for FPL points prediction')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    
    args = parser.parse_args()
    
    train_model(epochs=args.epochs, batch_size=args.batch_size)