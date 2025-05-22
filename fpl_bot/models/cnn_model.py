import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

class FPLPredictionModel:
    def __init__(self, lookback=3, n_features=14, models_dir="models/saved_models"):
        self.lookback = lookback
        self.n_features = n_features
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self.model = self._build_model()
    
    def _build_model(self):
        """Build CNN model architecture"""
        model = Sequential([
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
        
        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        """Train the CNN model"""
        if X_val is None or y_val is None:
            # Split training data if validation set not provided
            val_split = 0.2
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ModelCheckpoint(
                    filepath=os.path.join(self.models_dir, 'fpl_cnn_model.h5'),
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
                    filepath=os.path.join(self.models_dir, 'fpl_cnn_model.h5'),
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
    
    def save(self, filename='fpl_cnn_model.h5'):
        """Save the trained model"""
        filepath = os.path.join(self.models_dir, filename)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        return filepath
    
    def load(self, filename='fpl_cnn_model.h5'):
        """Load a trained model"""
        filepath = os.path.join(self.models_dir, filename)
        self.model = load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model

def train_model_with_data(X, y, lookback=3, n_features=14):
    """Utility function to train model with given data"""
    # Split data into train/validation sets
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    model = FPLPredictionModel(lookback=lookback, n_features=n_features)
    history = model.train(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    val_loss, val_mae = model.model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation MAE: {val_mae:.4f}")
    
    # Save model
    model.save()
    
    return model, history