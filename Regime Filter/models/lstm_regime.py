"""
LSTM Sequence Model for Regime Prediction

Uses temporal patterns in market data to predict future regime states.
Captures regime transitions and momentum patterns that XGBoost may miss.

Architecture:
- Input: 20-day sequences of key features
- LSTM layers: 64 → 32 units with dropout
- Output: 5-class softmax (Bull/Bear/Stress/Choppy/Recovery)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
from pathlib import Path
import pickle

# Try to import PyTorch, fall back to TensorFlow/Keras if not available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Attempting TensorFlow/Keras...")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

if not PYTORCH_AVAILABLE and not TENSORFLOW_AVAILABLE:
    raise ImportError(
        "Neither PyTorch nor TensorFlow is installed. "
        "Install one with: pip install torch or pip install tensorflow"
    )


class LSTMRegimeModel:
    """
    LSTM-based regime prediction model

    Predicts market regime N days ahead using sequence patterns.
    Can use either PyTorch or TensorFlow/Keras backend.
    """

    # Regime mappings
    REGIME_MAPPING = {
        'Bull': 0,
        'Bear': 1,
        'Stress': 2,
        'Choppy': 3,
        'Recovery': 4
    }
    INVERSE_MAPPING = {v: k for k, v in REGIME_MAPPING.items()}

    # Key features for LSTM input
    # Only includes features that are ALWAYS available from core market data
    # (XU100 + USD/TRY). TCMB/external features are excluded because their
    # availability is unreliable, causing train/predict feature mismatches.
    DEFAULT_FEATURES = [
        # Price-based
        'return_20d',
        'realized_vol_20d',
        'max_drawdown_20d',
        'volume_ratio',

        # Risk indicators
        'usdtry_momentum_20d',

        # Additional core features (always available)
        'momentum_accel_20d',
        'return_60d',
        'usdtry_vol_20d',
        'price_to_ma_50',
    ]

    def __init__(self, sequence_length=20, forecast_horizon=5, hidden_size=64,
                 num_layers=2, dropout=0.2, learning_rate=0.001, backend='auto'):
        """
        Args:
            sequence_length: Number of days in input sequence (default: 20)
            forecast_horizon: Days ahead to predict (default: 5)
            hidden_size: LSTM hidden layer size (default: 64)
            num_layers: Number of LSTM layers (default: 2)
            dropout: Dropout rate for regularization (default: 0.2)
            learning_rate: Learning rate for optimizer (default: 0.001)
            backend: 'pytorch', 'tensorflow', or 'auto' (default: 'auto')
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate

        # Select backend
        if backend == 'auto':
            self.backend = 'pytorch' if PYTORCH_AVAILABLE else 'tensorflow'
        else:
            self.backend = backend

        if self.backend == 'pytorch' and not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch requested but not installed")
        if self.backend == 'tensorflow' and not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow requested but not installed")

        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.n_features = None
        self.n_classes = 5
        self.is_trained = False

        # Training history
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def _build_pytorch_model(self):
        """Build PyTorch LSTM model"""

        class LSTMClassifier(nn.Module):
            def __init__(self, n_features, hidden_size, num_layers, n_classes, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=n_features,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0
                )
                self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
                self.dropout = nn.Dropout(dropout)
                self.fc2 = nn.Linear(hidden_size // 2, n_classes)
                self.relu = nn.ReLU()

            def forward(self, x):
                # LSTM output
                lstm_out, (h_n, c_n) = self.lstm(x)
                # Use last hidden state
                last_hidden = h_n[-1]
                # Fully connected layers
                out = self.relu(self.fc1(last_hidden))
                out = self.dropout(out)
                out = self.fc2(out)
                return out

        return LSTMClassifier(
            self.n_features, self.hidden_size, self.num_layers,
            self.n_classes, self.dropout
        )

    def _build_tensorflow_model(self):
        """Build TensorFlow/Keras LSTM model"""
        model = keras.Sequential([
            layers.LSTM(self.hidden_size, return_sequences=True,
                       input_shape=(self.sequence_length, self.n_features)),
            layers.Dropout(self.dropout),
            layers.LSTM(self.hidden_size // 2, return_sequences=False),
            layers.Dropout(self.dropout),
            layers.Dense(self.hidden_size // 2, activation='relu'),
            layers.Dropout(self.dropout),
            layers.Dense(self.n_classes, activation='softmax')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def prepare_sequences(self, features, regimes, return_dates=False):
        """
        Prepare sequence data for LSTM training

        Args:
            features: DataFrame of features
            regimes: Series of regime labels
            return_dates: Whether to also return sequence end dates

        Returns:
            X: 3D array (samples, sequence_length, features)
            y: 1D array of target regime labels
        """
        print(f"\nPreparing sequences (length={self.sequence_length}, horizon={self.forecast_horizon})...")

        # Select available features
        available_features = [f for f in self.DEFAULT_FEATURES if f in features.columns]

        if len(available_features) < 5:
            # Fall back to any numeric features
            available_features = features.select_dtypes(include=[np.number]).columns.tolist()[:15]

        print(f"  Using {len(available_features)} features: {available_features[:5]}...")
        self.feature_names = available_features
        self.n_features = len(available_features)

        # Extract feature values
        X_raw = features[available_features].values

        # BUG-25 FIX: Create target regime N days ahead
        # The shift(-forecast_horizon) already moves regimes forward
        # So we should NOT add forecast_horizon again in the loop
        y_raw = regimes.shift(-self.forecast_horizon).map(self.REGIME_MAPPING).values

        # Create sequences
        X_sequences = []
        y_sequences = []
        sequence_dates = []

        # Loop creates sequences from day i to i+seq_len-1
        # Target should be the regime at day i+seq_len-1 (already shifted forward)
        for i in range(len(X_raw) - self.sequence_length - self.forecast_horizon + 1):
            # Get sequence of features
            seq = X_raw[i:i + self.sequence_length]

            # BUG-25 FIX: Target is at the END of the sequence (already shifted)
            # Don't add forecast_horizon again - y_raw is already shifted!
            target_idx = i + self.sequence_length - 1
            if target_idx < len(y_raw) and not np.isnan(y_raw[target_idx]):
                # Check for NaN in sequence
                if not np.isnan(seq).any():
                    X_sequences.append(seq)
                    y_sequences.append(y_raw[target_idx])
                    sequence_dates.append(features.index[target_idx])

        X = np.array(X_sequences)
        y = np.array(y_sequences)

        print(f"  Created {len(X)} sequences")
        print(f"  Sequence shape: {X.shape}")

        # Regime distribution
        if len(y) > 0:
            unique, counts = np.unique(y, return_counts=True)
            print(f"  Regime distribution:")
            for regime_id, count in zip(unique, counts):
                regime_name = self.INVERSE_MAPPING[int(regime_id)]
                pct = count / len(y) * 100
                print(f"    {regime_name}: {count} ({pct:.1f}%)")

        if return_dates:
            return X, y, pd.DatetimeIndex(sequence_dates)

        return X, y

    def train_test_split(self, X, y, train_ratio=0.8):
        """
        Split data into train and test sets (time-based, no shuffle)

        Args:
            X: Sequence data
            y: Target labels
            train_ratio: Fraction for training (default: 0.8)

        Returns:
            X_train, X_test, y_train, y_test
        """
        split_idx = int(len(X) * train_ratio)

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"\nTrain/Test split:")
        print(f"  Train: {len(X_train)} sequences")
        print(f"  Test:  {len(X_test)} sequences")

        return X_train, X_test, y_train, y_test

    def train_test_split_by_date(self, X, y, sequence_dates, train_end_date):
        """
        Split sequence data into train and test sets by date.

        Args:
            X: Sequence data
            y: Target labels
            sequence_dates: DatetimeIndex (one date per sequence)
            train_end_date: Last date included in training set

        Returns:
            X_train, X_test, y_train, y_test
        """
        if len(sequence_dates) != len(X):
            raise ValueError("sequence_dates length must match number of sequences")

        train_end_date = pd.to_datetime(train_end_date)
        sequence_dates = pd.DatetimeIndex(sequence_dates)

        train_mask = sequence_dates <= train_end_date
        test_mask = sequence_dates > train_end_date

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        print(f"\nTrain/Test split by date ({train_end_date.date()}):")
        print(f"  Train: {len(X_train)} sequences")
        print(f"  Test:  {len(X_test)} sequences")

        return X_train, X_test, y_train, y_test

    def _scale_sequences(self, X_train, X_test):
        """Scale features using StandardScaler"""
        # Fit scaler on training data only
        X_train_2d = X_train.reshape(-1, self.n_features)
        self.scaler.fit(X_train_2d)

        X_train_scaled = self._transform_sequences(X_train)
        X_test_scaled = self._transform_sequences(X_test)

        return X_train_scaled, X_test_scaled

    def _transform_sequences(self, X):
        """Transform pre-shaped sequences with fitted scaler."""
        if X.shape[0] == 0:
            return X.copy()

        n_samples = X.shape[0]
        X_2d = X.reshape(-1, self.n_features)
        X_scaled = self.scaler.transform(X_2d)
        return X_scaled.reshape(n_samples, self.sequence_length, self.n_features)

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=50, batch_size=32, early_stopping_patience=10):
        """
        Train the LSTM model

        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            early_stopping_patience: Epochs to wait for improvement

        Returns:
            Training history
        """
        print(f"\nTraining LSTM model ({self.backend} backend)...")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Early stopping patience: {early_stopping_patience}")

        # Build model
        if self.backend == 'pytorch':
            return self._train_pytorch(X_train, y_train, X_val, y_val,
                                       epochs, batch_size, early_stopping_patience)
        else:
            return self._train_tensorflow(X_train, y_train, X_val, y_val,
                                         epochs, batch_size, early_stopping_patience)

    def _train_pytorch(self, X_train, y_train, X_val, y_val,
                       epochs, batch_size, early_stopping_patience):
        """Train using PyTorch"""
        # Build model
        self.model = self._build_pytorch_model()

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.LongTensor(y_train.astype(int))

        if X_val is not None:
            X_val_t = torch.FloatTensor(X_val)
            y_val_t = torch.LongTensor(y_val.astype(int))

        # DataLoader
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        # Optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            avg_train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total

            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_t)
                    val_loss = criterion(val_outputs, y_val_t).item()
                    _, val_predicted = torch.max(val_outputs.data, 1)
                    val_acc = (val_predicted == y_val_t).sum().item() / len(y_val_t)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"\n  Early stopping at epoch {epoch + 1}")
                        self.model.load_state_dict(best_model_state)
                        break

                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch + 1}/{epochs}: "
                          f"train_loss={avg_train_loss:.4f}, train_acc={train_acc:.2%}, "
                          f"val_loss={val_loss:.4f}, val_acc={val_acc:.2%}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch + 1}/{epochs}: "
                          f"train_loss={avg_train_loss:.4f}, train_acc={train_acc:.2%}")

            self.history['train_loss'].append(avg_train_loss)
            self.history['train_acc'].append(train_acc)

        self.is_trained = True
        return self.history

    def _train_tensorflow(self, X_train, y_train, X_val, y_val,
                          epochs, batch_size, early_stopping_patience):
        """Train using TensorFlow/Keras"""
        # Build model
        self.model = self._build_tensorflow_model()

        # Callbacks
        callbacks = []
        if X_val is not None:
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True
            ))

        # Prepare validation data
        validation_data = (X_val, y_val) if X_val is not None else None

        # Train
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )

        # Store history
        self.history['train_loss'] = history.history['loss']
        self.history['train_acc'] = history.history['accuracy']
        if X_val is not None:
            self.history['val_loss'] = history.history['val_loss']
            self.history['val_acc'] = history.history['val_accuracy']

        self.is_trained = True
        return self.history

    def predict(self, X):
        """
        Make predictions on new data

        Args:
            X: Sequence data (3D array)

        Returns:
            predictions: Predicted regime labels (strings)
            probabilities: Class probabilities
            confidence: Max probability (confidence score)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        if self.backend == 'pytorch':
            self.model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X)
                outputs = self.model(X_t)
                probabilities = torch.softmax(outputs, dim=1).numpy()
                predictions_numeric = torch.argmax(outputs, dim=1).numpy()
        else:
            probabilities = self.model.predict(X, verbose=0)
            predictions_numeric = np.argmax(probabilities, axis=1)

        # Convert to labels
        predictions = [self.INVERSE_MAPPING[p] for p in predictions_numeric]
        confidence = probabilities.max(axis=1)

        return predictions, probabilities, confidence

    def predict_single(self, features_df, scaler=None):
        """
        Predict regime from a DataFrame of recent features

        Args:
            features_df: DataFrame with at least sequence_length rows
            scaler: Optional pre-fitted scaler

        Returns:
            prediction: Predicted regime label
            probabilities: Dict of regime probabilities
            confidence: Confidence score
        """
        if len(features_df) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} rows of data")

        # Select features
        available_features = [f for f in self.feature_names if f in features_df.columns]
        X_raw = features_df[available_features].tail(self.sequence_length).values

        # Scale
        if scaler is None:
            scaler = self.scaler

        X_scaled = scaler.transform(X_raw)
        X_seq = X_scaled.reshape(1, self.sequence_length, -1)

        # Predict
        predictions, probabilities, confidence = self.predict(X_seq)

        # Format probabilities
        prob_dict = {self.INVERSE_MAPPING[i]: float(probabilities[0][i])
                    for i in range(self.n_classes)}

        return predictions[0], prob_dict, float(confidence[0])

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance

        Args:
            X_test: Test sequences
            y_test: True labels

        Returns:
            metrics: Dictionary of performance metrics
        """
        print("\n" + "="*70)
        print("LSTM MODEL EVALUATION")
        print("="*70)

        predictions, probabilities, confidence = self.predict(X_test)

        # Convert predictions to numeric
        predictions_numeric = [self.REGIME_MAPPING[p] for p in predictions]

        # Overall accuracy
        accuracy = accuracy_score(y_test, predictions_numeric)
        print(f"\nOverall Accuracy: {accuracy:.2%}")
        print(f"Average Confidence: {confidence.mean():.2%}")

        # Per-regime metrics
        print("\nClassification Report:")
        print(classification_report(
            y_test, predictions_numeric,
            target_names=list(self.REGIME_MAPPING.keys()),
            zero_division=0
        ))

        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, predictions_numeric)
        cm_df = pd.DataFrame(
            cm,
            index=list(self.REGIME_MAPPING.keys()),
            columns=list(self.REGIME_MAPPING.keys())
        )
        print(cm_df)

        # Stress detection recall (critical!)
        stress_mask = y_test == 2  # Stress = 2
        if stress_mask.sum() > 0:
            stress_recall = (np.array(predictions_numeric)[stress_mask] == 2).sum() / stress_mask.sum()
            print(f"\nStress Detection Recall: {stress_recall:.2%}")
        else:
            stress_recall = 0.0

        return {
            'accuracy': accuracy,
            'avg_confidence': confidence.mean(),
            'stress_recall': stress_recall,
            'confusion_matrix': cm_df
        }

    def save(self, filepath):
        """Save model to disk"""
        filepath = Path(filepath)

        # Save model state
        if self.backend == 'pytorch':
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'config': {
                    'sequence_length': self.sequence_length,
                    'forecast_horizon': self.forecast_horizon,
                    'hidden_size': self.hidden_size,
                    'num_layers': self.num_layers,
                    'dropout': self.dropout,
                    'n_features': self.n_features,
                    'backend': self.backend
                }
            }, filepath)
        else:
            self.model.save(filepath.with_suffix('.keras'))
            # Save metadata separately
            with open(filepath.with_suffix('.meta'), 'wb') as f:
                pickle.dump({
                    'scaler': self.scaler,
                    'feature_names': self.feature_names,
                    'config': {
                        'sequence_length': self.sequence_length,
                        'forecast_horizon': self.forecast_horizon,
                        'hidden_size': self.hidden_size,
                        'num_layers': self.num_layers,
                        'dropout': self.dropout,
                        'n_features': self.n_features,
                        'backend': self.backend
                    }
                }, f)

        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """Load model from disk"""
        filepath = Path(filepath)

        if filepath.suffix == '.keras' or filepath.with_suffix('.keras').exists():
            # TensorFlow model
            model_path = filepath.with_suffix('.keras')
            meta_path = filepath.with_suffix('.meta')

            with open(meta_path, 'rb') as f:
                metadata = pickle.load(f)

            config = metadata['config']
            instance = cls(
                sequence_length=config['sequence_length'],
                forecast_horizon=config['forecast_horizon'],
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                dropout=config['dropout'],
                backend='tensorflow'
            )

            instance.model = keras.models.load_model(model_path)
            instance.scaler = metadata['scaler']
            instance.feature_names = metadata['feature_names']
            instance.n_features = config['n_features']
            instance.is_trained = True

        else:
            # PyTorch model
            checkpoint = torch.load(filepath, weights_only=False)
            config = checkpoint['config']

            instance = cls(
                sequence_length=config['sequence_length'],
                forecast_horizon=config['forecast_horizon'],
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                dropout=config['dropout'],
                backend='pytorch'
            )

            instance.n_features = config['n_features']
            instance.model = instance._build_pytorch_model()
            instance.model.load_state_dict(checkpoint['model_state_dict'])
            instance.scaler = checkpoint['scaler']
            instance.feature_names = checkpoint['feature_names']
            instance.is_trained = True

        print(f"Model loaded from {filepath}")
        return instance


if __name__ == "__main__":
    print("="*70)
    print("LSTM REGIME MODEL - DEMO")
    print("="*70)

    # This is a demo showing how to use the LSTM model
    # In practice, you would load actual features and regimes

    print("\nUsage:")
    print("  from models.lstm_regime import LSTMRegimeModel")
    print("  ")
    print("  # Initialize")
    print("  lstm = LSTMRegimeModel(sequence_length=20, forecast_horizon=5)")
    print("  ")
    print("  # Prepare data")
    print("  X, y = lstm.prepare_sequences(features, simplified_regimes)")
    print("  X_train, X_test, y_train, y_test = lstm.train_test_split(X, y)")
    print("  X_train_s, X_test_s = lstm._scale_sequences(X_train, X_test)")
    print("  ")
    print("  # Train")
    print("  lstm.train(X_train_s, y_train, X_test_s, y_test, epochs=50)")
    print("  ")
    print("  # Evaluate")
    print("  metrics = lstm.evaluate(X_test_s, y_test)")
    print("  ")
    print("  # Predict")
    print("  predictions, probs, confidence = lstm.predict(X_new)")

    print(f"\nBackend available: PyTorch={PYTORCH_AVAILABLE}, TensorFlow={TENSORFLOW_AVAILABLE}")
    print("="*70)
