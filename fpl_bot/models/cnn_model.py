import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

class FPLPredictionModel:
    def __init__(self, lookback=3, n_features=14):
        self.lookback = lookback
        self.n_features = n_features
        self.model = None
        self._build_model()  # Build the model during initialization
        
    def _build_model(self):
        """Build CNN model architecture"""
        self.model = Sequential([
            # First convolutional layer
            Conv1D(filters=64, kernel_size=1, activation='relu', 
                   input_shape=(self.lookback, self.n_features)),
            MaxPooling1D(pool_size=2, padding='same'),
            
            # Second convolutional layer
            Conv1D(filters=128, kernel_size=1, activation='relu'),
            MaxPooling1D(pool_size=2, padding='same'),
            
            # Flatten and dense layers
            Flatten(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(1)  # Output layer for regression
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae']
        )
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        """Train the CNN model"""
        # Build model if it doesn't exist
        if self.model is None:
            self._build_model()
        
        if X_val is None or y_val is None:
            # Split training data if validation set not provided
            val_split = 0.2
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ModelCheckpoint(
                    filepath=os.path.join('data/models', 'fpl_cnn_model.h5'),
                    save_best_only=True,
                    monitor='val_loss'
                )
            ]
            
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=val_split,
                callbacks=callbacks,
                verbose=1
            )
        else:
            # Use provided validation set
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ModelCheckpoint(
                    filepath=os.path.join('data/models', 'fpl_cnn_model.h5'),
                    save_best_only=True,
                    monitor='val_loss'
                )
            ]
            
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
        
        return history
    
    def predict(self, X):
        """Make predictions using the trained model"""
        return self.model.predict(X)
    
    def save(self, model_type="standard"):
        """
        Save the model
        
        Parameters:
        -----------
        model_type : str
            Type of model to save: 'standard' or 'historical'
        """
        if self.model is None:
            raise ValueError("No model to save")
            
        # Create directory if it doesn't exist
        os.makedirs('data/models', exist_ok=True)
        
        # Determine file path based on model type
        if model_type == "historical":
            model_filepath = f'data/models/historical_model_lookback{self.lookback}.h5'
        else:
            model_filepath = f'data/models/model_lookback{self.lookback}.h5'
            
        self.model.save(model_filepath)
        print(f"Model saved to {model_filepath}")
        return model_filepath
    
    def load(self, model_type="standard"):
        """
        Load a saved model
        
        Parameters:
        -----------
        model_type : str
            Type of model to load: 'standard' or 'historical'
        """
        # Model file path based on type
        if model_type == "historical":
            model_filepath = f'data/models/historical_model_lookback{self.lookback}.h5'
        else:
            model_filepath = f'data/models/model_lookback{self.lookback}.h5'
            
        if not os.path.exists(model_filepath):
            raise FileNotFoundError(f"No saved model found at {model_filepath}")
            
        self.model = tf.keras.models.load_model(model_filepath)
        print(f"Model loaded from {model_filepath}")
        return self.model

def train_model_with_data(X, y, lookback=3, n_features=14, epochs=50, batch_size=32):
    """Utility function to train model with given data"""
    # Split data into train/validation sets
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    model = FPLPredictionModel(lookback=lookback, n_features=n_features)
    history = model.train(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)
    
    # Evaluate model
    val_loss, val_mae = model.model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation MAE: {val_mae:.4f}")
    
    # Save model
    model.save()
    
    return model, history