import os
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

class FPLPredictionModel:
    def __init__(self, n_features=45, models_dir="models/saved_models"):
        """
        Initialize FPL Prediction Model
        
        Parameters:
        -----------
        n_features : int
            Number of input features (based on our processed data)
        models_dir : str
            Directory to save/load models
        """
        self.n_features = n_features
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self.model = None
        self.is_trained = False
        
    def _build_model(self):
        """Build enhanced neural network model architecture for FPL prediction"""
        model = Sequential([
            # Input layer with more gradual size reduction
            Dense(128, activation='relu', input_shape=(self.n_features,)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Hidden layers with residual-like connections via better sizing
            Dense(96, activation='relu'),
            BatchNormalization(),
            Dropout(0.25),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(48, activation='relu'),
            BatchNormalization(),
            Dropout(0.15),
            
            Dense(32, activation='relu'),
            Dropout(0.1),
            
            Dense(16, activation='relu'),
            Dropout(0.05),
            
            # Output layer for points prediction
            Dense(1, activation='linear')
        ])
        
        # Use a more sophisticated optimizer setup
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            clipnorm=1.0  # Gradient clipping for stability
        )
        
        model.compile(
            optimizer=optimizer,
            loss='huber',  # More robust to outliers than MSE
            metrics=['mae', 'mse', 'mape']  # Added MAPE for percentage error
        )
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, verbose=1):
        """
        Train the model
        
        Parameters:
        -----------
        X_train, y_train : arrays
            Training data
        X_val, y_val : arrays, optional
            Validation data
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        verbose : int
            Verbosity level
        """
        # Build model if not already built
        if self.model is None:
            self.model = self._build_model()
        
        # Prepare validation data
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                patience=15, 
                restore_best_weights=True, 
                monitor='val_loss'
            ),
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=8, 
                min_lr=0.0001
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.models_dir, 'fpl_model.h5'),
                save_best_only=True,
                monitor='val_loss',
                mode='min'
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_trained = True
        return history
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        return self.model.predict(X, verbose=0)
    
    def save(self, filename='fpl_model.h5'):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        filepath = os.path.join(self.models_dir, filename)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        return filepath
    
    def load(self, filename='fpl_model.h5'):
        """Load a trained model"""
        filepath = os.path.join(self.models_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        self.model = load_model(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")
        return self.model
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        return self.model.evaluate(X_test, y_test, verbose=0)


def train_model_with_processed_data(data_dir="data", target='points_scored', epochs=100, batch_size=32):
    """
    Train model using processed FPL data
    
    Parameters:
    -----------
    data_dir : str
        Directory containing processed data
    target : str
        Target variable to predict
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    
    Returns:
    --------
    model : FPLPredictionModel
        Trained model
    history : tf.keras.History
        Training history
    """
    # Load processed features and targets
    features_path = os.path.join(data_dir, 'features', f'features_{target}.csv')
    target_path = os.path.join(data_dir, 'features', f'target_{target}.csv')
    
    if not os.path.exists(features_path) or not os.path.exists(target_path):
        raise FileNotFoundError(f"Processed data not found. Please run data processing first.")
    
    # Load data
    X = pd.read_csv(features_path)
    y = pd.read_csv(target_path).iloc[:, 0]  # First column contains target values
    
    print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    print(f"Target: {target}")
    
    # Handle missing values
    X = X.fillna(0)
    y = y.fillna(0)
    
    # Convert boolean columns to integers
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)
    
    # Convert to numpy arrays and ensure float32 dtype
    X = X.values.astype(np.float32)
    y = y.values.astype(np.float32)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train model
    model = FPLPredictionModel(n_features=X.shape[1])
    history = model.train(X_train, y_train, epochs=epochs, batch_size=batch_size)
    
    # Evaluate model
    test_loss, test_mae, test_mse = model.evaluate(X_test, y_test)
    print(f"\nFinal Test Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    
    # Save model
    model.save(f'fpl_model_{target}.h5')
    
    return model, history
