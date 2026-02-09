"""
Ensemble Regime Model

Combines XGBoost, LSTM, and HMM models for robust regime prediction.
Uses weighted voting with dynamic weight adjustment based on rolling accuracy.

Ensemble Strategy:
- XGBoost (0.4): Best for tabular features, captures non-linear relationships
- LSTM (0.35): Captures temporal patterns and regime transitions
- HMM (0.25): Provides regime probabilities and transition dynamics
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
import pickle
import warnings
from datetime import datetime

# Import model components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from regime_models import PredictiveRegimeModel
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("PredictiveRegimeModel not available")

try:
    from regime_models import HMMRegimeClassifier
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    warnings.warn("HMMRegimeClassifier not available")

try:
    from models.lstm_regime import LSTMRegimeModel
    LSTM_AVAILABLE = True
except ImportError:
    try:
        from lstm_regime import LSTMRegimeModel
        LSTM_AVAILABLE = True
    except ImportError:
        LSTM_AVAILABLE = False
        warnings.warn("lstm_regime module not available")


class EnsembleRegimeModel:
    """
    Ensemble model combining XGBoost, LSTM, and HMM

    Features:
    - Soft voting using probability outputs from all models
    - Dynamic weight adjustment based on rolling accuracy
    - Disagreement indicator for uncertainty quantification
    - Model agreement metrics
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

    # Default model weights
    DEFAULT_WEIGHTS = {
        'xgboost': 0.40,
        'lstm': 0.35,
        'hmm': 0.25
    }

    def __init__(self, weights=None, dynamic_weights=True, forecast_horizon=0):
        """
        Args:
            weights: Dict of model weights (default: {'xgboost': 0.4, 'lstm': 0.35, 'hmm': 0.25})
            dynamic_weights: Whether to adjust weights based on rolling accuracy
            forecast_horizon: Days ahead to predict (0 = predict current regime)
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.dynamic_weights = dynamic_weights
        self.forecast_horizon = forecast_horizon

        # Models
        self.xgboost_model = None
        self.lstm_model = None
        self.hmm_model = None

        # Training state
        self.is_trained = False
        self.available_models = []

        # Rolling accuracy for dynamic weights
        self.rolling_accuracy = {
            'xgboost': [],
            'lstm': [],
            'hmm': []
        }

        # Model metadata
        self.train_date = None
        self.feature_names = None

    def _check_available_models(self):
        """Check which models are available"""
        self.available_models = []

        if XGBOOST_AVAILABLE:
            self.available_models.append('xgboost')
        if LSTM_AVAILABLE:
            self.available_models.append('lstm')
        if HMM_AVAILABLE:
            self.available_models.append('hmm')

        if len(self.available_models) == 0:
            raise ImportError(
                "No models available. Install required packages:\n"
                "  pip install xgboost  # for XGBoost\n"
                "  pip install torch    # for LSTM\n"
                "  pip install hmmlearn # for HMM"
            )

        print(f"Available models: {', '.join(self.available_models)}")

        # Normalize weights for available models
        total_weight = sum(self.weights[m] for m in self.available_models)
        for model in self.available_models:
            self.weights[model] /= total_weight

        return self.available_models

    def train(self, features, regimes, train_end_date='2020-12-31',
              xgboost_params=None, lstm_params=None, hmm_params=None):
        """
        Train all ensemble models

        Args:
            features: DataFrame of features
            regimes: Series of regime labels (simplified 5-state)
            train_end_date: Last date for training
            xgboost_params: Optional params for XGBoost model
            lstm_params: Optional params for LSTM model
            hmm_params: Optional params for HMM model

        Returns:
            Training results dict
        """
        print("="*70)
        print("TRAINING ENSEMBLE MODEL")
        print("="*70)
        print(f"Training end date: {train_end_date}")
        print(f"Forecast horizon: {self.forecast_horizon} days")

        self._check_available_models()
        self.train_date = datetime.now()
        self.feature_names = features.columns.tolist()

        results = {}

        # Train XGBoost
        if 'xgboost' in self.available_models:
            print("\n" + "-"*70)
            print("Training XGBoost Model")
            print("-"*70)
            try:
                results['xgboost'] = self._train_xgboost(
                    features, regimes, train_end_date, xgboost_params or {}
                )
            except Exception as e:
                print(f"XGBoost training failed: {e}")
                self.available_models.remove('xgboost')

        # Train LSTM
        if 'lstm' in self.available_models:
            print("\n" + "-"*70)
            print("Training LSTM Model")
            print("-"*70)
            try:
                results['lstm'] = self._train_lstm(
                    features, regimes, train_end_date, lstm_params or {}
                )
            except Exception as e:
                print(f"LSTM training failed: {e}")
                self.available_models.remove('lstm')

        # Train HMM
        if 'hmm' in self.available_models:
            print("\n" + "-"*70)
            print("Training HMM Model")
            print("-"*70)
            try:
                results['hmm'] = self._train_hmm(
                    features, hmm_params or {}, train_end_date
                )
            except Exception as e:
                print(f"HMM training failed: {e}")
                self.available_models.remove('hmm')

        # Re-normalize weights after any failures
        if len(self.available_models) > 0:
            total_weight = sum(self.weights[m] for m in self.available_models)
            for model in self.available_models:
                self.weights[model] /= total_weight

            self.is_trained = True
            print("\n" + "="*70)
            print("ENSEMBLE TRAINING COMPLETE")
            print("="*70)
            print(f"Active models: {', '.join(self.available_models)}")
            print(f"Final weights: {self.weights}")
        else:
            raise RuntimeError("All model training failed")

        return results

    def _train_xgboost(self, features, regimes, train_end_date, params):
        """Train XGBoost model"""
        self.xgboost_model = PredictiveRegimeModel(
            forecast_horizon=self.forecast_horizon,
            model_type='xgboost'
        )

        X, y = self.xgboost_model.prepare_data(features, regimes)
        X_train, X_test, y_train, y_test = self.xgboost_model.train_test_split(
            X, y, train_end_date=train_end_date
        )

        self.xgboost_model.train(X_train, y_train)
        predictions, confidence, _ = self.xgboost_model.predict(X_test)

        # Convert predictions to numeric
        pred_numeric = predictions.map(self.xgboost_model.regime_mapping)
        accuracy = accuracy_score(y_test, pred_numeric)

        print(f"  XGBoost Test Accuracy: {accuracy:.2%}")
        return {'accuracy': accuracy, 'test_samples': len(X_test)}

    def _train_lstm(self, features, regimes, train_end_date, params):
        """Train LSTM model"""
        sequence_length = params.get('sequence_length', 20)
        epochs = params.get('epochs', 50)
        batch_size = params.get('batch_size', 32)

        self.lstm_model = LSTMRegimeModel(
            sequence_length=sequence_length,
            forecast_horizon=self.forecast_horizon
        )

        # Prepare sequences
        X, y, seq_dates = self.lstm_model.prepare_sequences(
            features, regimes, return_dates=True
        )

        # Split data by date for consistency with other models.
        X_train, X_test, y_train, y_test = self.lstm_model.train_test_split_by_date(
            X, y, seq_dates, train_end_date
        )

        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError(
                f"Insufficient LSTM data after date split (train={len(X_train)}, test={len(X_test)}). "
                f"Adjust train_end_date or sequence settings."
            )

        # Validation split inside train set (avoid using test set for early stopping).
        val_size = max(1, int(len(X_train) * 0.15)) if len(X_train) >= 20 else 0
        if val_size > 0 and len(X_train) > val_size:
            X_train_core = X_train[:-val_size]
            y_train_core = y_train[:-val_size]
            X_val = X_train[-val_size:]
            y_val = y_train[-val_size:]
        else:
            X_train_core = X_train
            y_train_core = y_train
            X_val = None
            y_val = None

        # Scale
        X_train_scaled, X_test_scaled = self.lstm_model._scale_sequences(X_train_core, X_test)
        X_val_scaled = self.lstm_model._transform_sequences(X_val) if X_val is not None else None

        # Train
        self.lstm_model.train(
            X_train_scaled, y_train_core,
            X_val_scaled, y_val,
            epochs=epochs,
            batch_size=batch_size
        )

        # Evaluate
        predictions, _, confidence = self.lstm_model.predict(X_test_scaled)
        pred_numeric = [self.REGIME_MAPPING[p] for p in predictions]
        accuracy = accuracy_score(y_test, pred_numeric)

        print(f"  LSTM Test Accuracy: {accuracy:.2%}")
        return {'accuracy': accuracy, 'test_samples': len(X_test)}

    def _train_hmm(self, features, params, train_end_date=None):
        """Train HMM model"""
        n_regimes = params.get('n_regimes', 4)

        if train_end_date:
            features = features[features.index <= train_end_date]

        self.hmm_model = HMMRegimeClassifier(n_regimes=n_regimes)
        self.hmm_model.fit(features)

        # Predict and label
        predictions = self.hmm_model.predict(features)
        predictions = self.hmm_model.label_regimes(features, predictions)

        print(f"  HMM fitted with {n_regimes} regimes")
        return {'n_regimes': n_regimes}

    def predict(self, features, regimes=None, return_details=False):
        """
        Make ensemble predictions

        Args:
            features: DataFrame of features
            regimes: Optional current regimes (for XGBoost)
            return_details: Whether to return per-model predictions

        Returns:
            predictions: Ensemble predicted regimes
            confidence: Ensemble confidence scores
            disagreement: Model disagreement indicator
            details: (optional) Per-model predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Collect predictions from each model
        model_predictions = {}
        model_probabilities = {}

        # XGBoost predictions
        if 'xgboost' in self.available_models and self.xgboost_model is not None:
            try:
                # Prepare data
                X = features.copy()

                # Get features that model was trained on - ALL must be present
                trained_features = self.xgboost_model.feature_names
                missing = [f for f in trained_features if f not in X.columns]
                if missing:
                    raise ValueError(
                        f"Missing {len(missing)} features required by XGBoost: {missing[:5]}... "
                        f"Retrain the model with `python run_full_pipeline.py`"
                    )

                X = X[trained_features]

                # Handle NaN
                # Causal fill only: do not backfill with future values.
                X = X.ffill().fillna(0)

                # Predict
                X_scaled = self.xgboost_model.scaler.transform(X)
                pred_numeric = self.xgboost_model.model.predict(X_scaled)
                probs = self.xgboost_model.model.predict_proba(X_scaled)

                model_predictions['xgboost'] = pred_numeric
                model_probabilities['xgboost'] = probs
            except Exception as e:
                print(f"XGBoost prediction failed: {e}")

        # LSTM predictions
        if 'lstm' in self.available_models and self.lstm_model is not None:
            try:
                # Prepare sequences directly from features
                # Don't call prepare_sequences during inference - build sequences manually
                trained_features = self.lstm_model.feature_names
                missing = [f for f in trained_features if f not in features.columns]
                if missing:
                    raise ValueError(
                        f"Missing {len(missing)} features required by LSTM: {missing}. "
                        f"Retrain the model with `python run_full_pipeline.py`"
                    )

                X_raw = features[trained_features].values
                n_features = len(trained_features)
                seq_len = self.lstm_model.sequence_length

                # Create sequences for inference (no target needed)
                X_sequences = []
                valid_indices = []
                for i in range(len(X_raw) - seq_len + 1):
                    seq = X_raw[i:i + seq_len]
                    if not np.isnan(seq).any():
                        X_sequences.append(seq)
                        valid_indices.append(i + seq_len - 1)  # Index of last element in sequence

                if len(X_sequences) > 0:
                    X_seq = np.array(X_sequences)

                    # Scale using actual feature count from data
                    X_seq_2d = X_seq.reshape(-1, n_features)
                    X_seq_scaled = self.lstm_model.scaler.transform(X_seq_2d)
                    X_seq_scaled = X_seq_scaled.reshape(X_seq.shape)

                    # Predict
                    _, probs, _ = self.lstm_model.predict(X_seq_scaled)
                    pred_numeric = np.argmax(probs, axis=1)

                    # Map back to full features length
                    pred_padded = np.full(len(features), np.nan)
                    probs_padded = np.full((len(features), 5), np.nan)

                    for idx, valid_idx in enumerate(valid_indices):
                        pred_padded[valid_idx] = pred_numeric[idx]
                        probs_padded[valid_idx] = probs[idx]

                    model_predictions['lstm'] = pred_padded
                    model_probabilities['lstm'] = probs_padded
            except Exception as e:
                print(f"LSTM prediction failed: {e}")

        # HMM predictions
        if 'hmm' in self.available_models and self.hmm_model is not None:
            try:
                hmm_preds = self.hmm_model.predict(features)
                # Label the regimes if not already labeled
                if 'regime_name' not in hmm_preds.columns:
                    hmm_preds = self.hmm_model.label_regimes(features, hmm_preds)

                # Convert HMM regimes to standard 5-class
                hmm_to_standard = {
                    'Bull': 0, 'Bear': 1, 'Stress': 2, 'Sideways': 3, 'Choppy': 3
                }

                pred_numeric = hmm_preds['regime_name'].map(
                    lambda x: hmm_to_standard.get(x, 3)  # Default to Choppy
                ).values

                # Create probability matrix from HMM
                probs = np.zeros((len(features), 5))
                for i in range(min(4, self.hmm_model.n_regimes)):
                    col = f'prob_regime_{i}'
                    if col in hmm_preds.columns:
                        regime_name = self.hmm_model.regime_names.get(i, 'Sideways')
                        target_idx = hmm_to_standard.get(regime_name, 3)
                        probs[:, target_idx] = hmm_preds[col].values

                model_predictions['hmm'] = pred_numeric
                model_probabilities['hmm'] = probs
            except Exception as e:
                print(f"HMM prediction failed: {e}")

        # Ensemble combining
        if len(model_probabilities) == 0:
            raise RuntimeError("All model predictions failed")

        # Weighted average of probabilities
        ensemble_probs = np.zeros((len(features), 5))
        total_weight = 0

        for model_name, probs in model_probabilities.items():
            weight = self.weights.get(model_name, 0)
            # Handle NaN in probabilities
            mask = ~np.isnan(probs[:, 0])
            ensemble_probs[mask] += weight * probs[mask]
            total_weight += weight * mask.astype(float)

        # Normalize by total weight
        total_weight = np.maximum(total_weight, 1e-10)
        ensemble_probs = ensemble_probs / total_weight[:, np.newaxis]

        # Final predictions
        predictions_numeric = np.argmax(ensemble_probs, axis=1)
        predictions = [self.INVERSE_MAPPING[p] for p in predictions_numeric]
        confidence = ensemble_probs.max(axis=1)

        # Calculate disagreement
        disagreement = self._calculate_disagreement(model_predictions)

        # Create results
        results = pd.DataFrame(index=features.index)
        results['ensemble_prediction'] = predictions
        results['ensemble_confidence'] = confidence
        results['model_disagreement'] = disagreement

        for i, regime_name in self.INVERSE_MAPPING.items():
            results[f'prob_{regime_name}'] = ensemble_probs[:, i]

        if return_details:
            details = {
                'model_predictions': model_predictions,
                'model_probabilities': model_probabilities
            }
            return results, details

        return results

    def _calculate_disagreement(self, model_predictions):
        """
        Calculate disagreement between models

        Returns value between 0 (full agreement) and 1 (maximum disagreement)
        """
        if len(model_predictions) < 2:
            return np.zeros(len(list(model_predictions.values())[0]))

        # Get predictions as arrays
        preds = [p for p in model_predictions.values()]
        n_samples = len(preds[0])

        disagreement = np.zeros(n_samples)

        for i in range(n_samples):
            votes = [p[i] for p in preds if not np.isnan(p[i])]
            if len(votes) > 1:
                # Disagreement = 1 - (max_agreement / n_models)
                unique, counts = np.unique(votes, return_counts=True)
                max_agreement = counts.max() / len(votes)
                disagreement[i] = 1 - max_agreement

        return disagreement

    def predict_current(self, features, regimes=None):
        """
        Get current regime prediction (most recent)

        Returns:
            prediction: Current regime label
            confidence: Confidence score
            agreement: Dict of per-model predictions
            probabilities: Dict of regime probabilities
        """
        results, details = self.predict(features, regimes, return_details=True)

        # Get last prediction
        last_idx = results.index[-1]
        prediction = results.loc[last_idx, 'ensemble_prediction']
        confidence = results.loc[last_idx, 'ensemble_confidence']
        disagreement = results.loc[last_idx, 'model_disagreement']

        # Probabilities
        prob_cols = [c for c in results.columns if c.startswith('prob_')]
        probabilities = {c.replace('prob_', ''): results.loc[last_idx, c]
                        for c in prob_cols}

        # Per-model predictions
        agreement = {}
        for model_name, preds in details['model_predictions'].items():
            pred_val = preds[-1]
            if not np.isnan(pred_val):
                agreement[model_name] = self.INVERSE_MAPPING[int(pred_val)]

        return {
            'prediction': prediction,
            'confidence': confidence,
            'disagreement': disagreement,
            'probabilities': probabilities,
            'model_agreement': agreement,
            'date': last_idx
        }

    def update_weights(self, y_true, model_predictions, window=60):
        """
        Update weights based on recent model accuracy

        Args:
            y_true: True regime labels
            model_predictions: Dict of model predictions
            window: Rolling window for accuracy calculation
        """
        if not self.dynamic_weights:
            return

        for model_name, preds in model_predictions.items():
            # Calculate rolling accuracy
            correct = (preds == y_true).astype(float)
            rolling_acc = pd.Series(correct).rolling(window).mean().iloc[-1]

            if not np.isnan(rolling_acc):
                self.rolling_accuracy[model_name].append(rolling_acc)

        # Update weights based on recent accuracy
        if all(len(self.rolling_accuracy[m]) > 0 for m in self.available_models):
            recent_acc = {m: np.mean(self.rolling_accuracy[m][-10:])
                         for m in self.available_models}
            total_acc = sum(recent_acc.values())

            if total_acc > 0:
                for model in self.available_models:
                    self.weights[model] = recent_acc[model] / total_acc

            print(f"Updated weights: {self.weights}")

    def evaluate(self, features, regimes, test_start_date=None):
        """
        Evaluate ensemble model performance

        Args:
            features: DataFrame of features
            regimes: Series of true regime labels
            test_start_date: Start date for test period

        Returns:
            Evaluation metrics
        """
        print("\n" + "="*70)
        print("ENSEMBLE MODEL EVALUATION")
        print("="*70)

        # Filter to test period
        if test_start_date:
            features = features[features.index >= test_start_date]
            regimes = regimes[regimes.index >= test_start_date]

        # Get predictions
        results, details = self.predict(features, regimes, return_details=True)

        # Prepare true labels
        if self.forecast_horizon > 0:
            y_true = regimes.shift(-self.forecast_horizon).map(self.REGIME_MAPPING)
        else:
            y_true = regimes.map(self.REGIME_MAPPING)
        y_true = y_true.reindex(results.index)

        # Filter out NaN
        valid_mask = ~y_true.isna()
        y_true = y_true[valid_mask].values.astype(int)
        y_pred = results.loc[valid_mask.values, 'ensemble_prediction'].map(self.REGIME_MAPPING).values

        # Metrics
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\nEnsemble Accuracy: {accuracy:.2%}")
        print(f"Average Confidence: {results['ensemble_confidence'].mean():.2%}")
        print(f"Average Disagreement: {results['model_disagreement'].mean():.2%}")

        # Per-regime metrics
        print("\nClassification Report:")
        print(classification_report(
            y_true, y_pred,
            target_names=list(self.REGIME_MAPPING.keys()),
            zero_division=0
        ))

        # Compare individual models
        print("\nIndividual Model Accuracies:")
        for model_name, preds in details['model_predictions'].items():
            preds_arr = np.asarray(preds)[valid_mask.values]
            model_mask = ~np.isnan(preds_arr)
            if model_mask.any():
                model_y_true = y_true[model_mask]
                model_y_pred = preds_arr[model_mask].astype(int)
                model_acc = accuracy_score(model_y_true, model_y_pred)
                print(f"  {model_name}: {model_acc:.2%}")

        return {
            'accuracy': accuracy,
            'avg_confidence': results['ensemble_confidence'].mean(),
            'avg_disagreement': results['model_disagreement'].mean()
        }

    def save(self, filepath):
        """Save ensemble model to disk"""
        filepath = Path(filepath)
        filepath.mkdir(parents=True, exist_ok=True)

        # Save each model
        if self.xgboost_model:
            with open(filepath / 'xgboost_model.pkl', 'wb') as f:
                pickle.dump(self.xgboost_model, f)

        if self.lstm_model:
            self.lstm_model.save(filepath / 'lstm_model.pt')

        if self.hmm_model:
            with open(filepath / 'hmm_model.pkl', 'wb') as f:
                pickle.dump(self.hmm_model, f)

        # Save metadata
        metadata = {
            'weights': self.weights,
            'available_models': self.available_models,
            'dynamic_weights': self.dynamic_weights,
            'forecast_horizon': self.forecast_horizon,
            'train_date': self.train_date,
            'feature_names': self.feature_names,
            'rolling_accuracy': self.rolling_accuracy
        }

        with open(filepath / 'ensemble_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)

        print(f"Ensemble model saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """Load ensemble model from disk"""
        filepath = Path(filepath)

        # Load metadata
        with open(filepath / 'ensemble_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)

        instance = cls(
            weights=metadata['weights'],
            dynamic_weights=metadata['dynamic_weights'],
            forecast_horizon=metadata['forecast_horizon']
        )

        instance.available_models = metadata['available_models']
        instance.train_date = metadata['train_date']
        instance.feature_names = metadata['feature_names']
        instance.rolling_accuracy = metadata['rolling_accuracy']

        # Load models
        if (filepath / 'xgboost_model.pkl').exists():
            with open(filepath / 'xgboost_model.pkl', 'rb') as f:
                instance.xgboost_model = pickle.load(f)

        if (filepath / 'lstm_model.pt').exists():
            instance.lstm_model = LSTMRegimeModel.load(filepath / 'lstm_model.pt')

        if (filepath / 'hmm_model.pkl').exists():
            with open(filepath / 'hmm_model.pkl', 'rb') as f:
                instance.hmm_model = pickle.load(f)

        instance.is_trained = True
        print(f"Ensemble model loaded from {filepath}")

        return instance

    def get_summary(self):
        """Get summary of ensemble model"""
        summary = []
        summary.append("="*70)
        summary.append("ENSEMBLE MODEL SUMMARY")
        summary.append("="*70)

        summary.append(f"\nTrained: {self.train_date}")
        summary.append(f"Forecast Horizon: {self.forecast_horizon} days")
        summary.append(f"Dynamic Weights: {self.dynamic_weights}")

        summary.append(f"\nActive Models: {', '.join(self.available_models)}")
        summary.append("\nModel Weights:")
        for model, weight in self.weights.items():
            if model in self.available_models:
                summary.append(f"  {model}: {weight:.2%}")

        if any(self.rolling_accuracy.values()):
            summary.append("\nRecent Rolling Accuracy:")
            for model, accs in self.rolling_accuracy.items():
                if accs:
                    summary.append(f"  {model}: {np.mean(accs[-10:]):.2%}")

        return "\n".join(summary)


if __name__ == "__main__":
    print("="*70)
    print("ENSEMBLE REGIME MODEL - DEMO")
    print("="*70)

    print("\nUsage:")
    print("  from models.ensemble_regime import EnsembleRegimeModel")
    print("  ")
    print("  # Initialize")
    print("  ensemble = EnsembleRegimeModel(forecast_horizon=0)")
    print("  ")
    print("  # Train all models")
    print("  ensemble.train(features, simplified_regimes, train_end_date='2020-12-31')")
    print("  ")
    print("  # Predict")
    print("  results = ensemble.predict(features, regimes)")
    print("  ")
    print("  # Get current regime")
    print("  current = ensemble.predict_current(features, regimes)")
    print("  print(current['prediction'], current['confidence'])")
    print("  ")
    print("  # Evaluate")
    print("  metrics = ensemble.evaluate(features, regimes, test_start_date='2021-01-01')")
    print("  ")
    print("  # Save/Load")
    print("  ensemble.save('models/ensemble')")
    print("  loaded = EnsembleRegimeModel.load('models/ensemble')")

    print(f"\nAvailable: XGBoost={XGBOOST_AVAILABLE}, LSTM={LSTM_AVAILABLE}, HMM={HMM_AVAILABLE}")
    print("="*70)
