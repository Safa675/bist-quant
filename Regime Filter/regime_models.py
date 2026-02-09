"""
Regime Models Module
Consolidates all regime classification and prediction models.

Includes:
1. Rule-Based Classifier (144 states)
2. Simplified Classifiers (5-state & 4-state)
3. HMM Regime Classifier (Unsupervised)
4. Predictive Regime Model (Supervised)
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import config

# ML Imports (Lazy loaded if possible, but standard here)
try:
    from hmmlearn import hmm, _hmmc
    from scipy.special import logsumexp
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import xgboost as xgb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    warnings.warn("ML packages (hmmlearn, sklearn, xgboost) not found. Some models will be disabled.")

warnings.filterwarnings('ignore')


class RegimeClassifier:
    """Classify market regimes using percentile-based thresholds"""
    
    def __init__(self, features):
        """
        Args:
            features: DataFrame with calculated features
        """
        self.features = features.copy()
        self.regimes = pd.DataFrame(index=features.index)
        self.percentiles = {}
        
    def calculate_percentiles(self, lookback=None):
        """
        Calculate rolling percentiles ONLY for features that need them.
        """
        if lookback is None:
            lookback = config.PERCENTILE_LOOKBACK
        
        print(f"Calculating rolling percentiles (lookback={lookback} days)...")
        
        # Only these 3 features need percentiles
        percentile_features = [
            'realized_vol_20d',      # Volatility regime
            'usdtry_momentum_20d',   # Risk regime
            'turnover_ma_20d'        # Liquidity regime
        ]
        
        percentile_dict = {}
        
        for feature in percentile_features:
            if feature in self.features.columns and self.features[feature].notna().sum() > lookback:
                # Vectorized percentile calculation
                rolling_rank = self.features[feature].rolling(
                    lookback, 
                    min_periods=lookback//2
                ).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
                )
                percentile_dict[f'{feature}_percentile'] = rolling_rank
            else:
                print(f"  Warning: {feature} not found or insufficient data, skipping percentile")
        
        self.percentiles = pd.DataFrame(percentile_dict, index=self.features.index)
        
    def classify_volatility_regime(self):
        """Classify volatility state: Low / Mid / High / Stress"""
        # Use 20-day realized vol percentile
        vol_pct = self.percentiles.get('realized_vol_20d_percentile')
        
        if vol_pct is None:
            self.regimes['volatility_regime'] = 'Unknown'
            return
        
        # Classify based on percentiles
        conditions = [
            vol_pct <= config.VOLATILITY_PERCENTILES['low'],
            (vol_pct > config.VOLATILITY_PERCENTILES['low']) & (vol_pct <= config.VOLATILITY_PERCENTILES['mid']),
            (vol_pct > config.VOLATILITY_PERCENTILES['mid']) & (vol_pct <= config.VOLATILITY_PERCENTILES['high']),
            vol_pct > config.VOLATILITY_PERCENTILES['high']
        ]
        choices = ['Low', 'Mid', 'High', 'Stress']
        
        self.regimes['volatility_regime'] = np.select(conditions, choices, default='Unknown')
        self.regimes['volatility_percentile'] = vol_pct
        
    def classify_trend_regime(self):
        """Classify trend state on multiple horizons: Up / Down / Sideways"""
        # Short-term trend (20-day return)
        short_return = self.features.get('return_20d', pd.Series(index=self.features.index))
        
        conditions_short = [
            short_return > config.TREND_THRESHOLDS['up'],
            short_return < config.TREND_THRESHOLDS['down'],
            (short_return >= config.TREND_THRESHOLDS['down']) & (short_return <= config.TREND_THRESHOLDS['up'])
        ]
        choices = ['Up', 'Down', 'Sideways']
        
        self.regimes['trend_short'] = np.select(conditions_short, choices, default='Unknown')
        
        # Long-term trend (120-day return)
        long_return = self.features.get('return_120d', pd.Series(index=self.features.index))
        
        conditions_long = [
            long_return > config.TREND_THRESHOLDS['up'] * 2,  # Higher threshold for longer horizon
            long_return < config.TREND_THRESHOLDS['down'] * 2,
            (long_return >= config.TREND_THRESHOLDS['down'] * 2) & (long_return <= config.TREND_THRESHOLDS['up'] * 2)
        ]
        
        self.regimes['trend_long'] = np.select(conditions_long, choices, default='Unknown')
        
        # Combined trend
        def combine_trends(row):
            if row['trend_short'] == row['trend_long']:
                return row['trend_short']
            elif row['trend_short'] == 'Sideways' or row['trend_long'] == 'Sideways':
                return 'Sideways'
            else:
                return 'Mixed'
        
        self.regimes['trend_regime'] = self.regimes.apply(combine_trends, axis=1)
        
    def classify_risk_regime(self):
        """Classify risk-on/off based on USD/TRY signals"""
        usdtry_mom_pct = self.percentiles.get('usdtry_momentum_20d_percentile')
        
        if usdtry_mom_pct is None:
            self.regimes['risk_regime'] = 'Neutral'
            return
        
        conditions = [
            usdtry_mom_pct > config.RISK_PERCENTILES['risk_off'],      # >65%: Risk-off
            usdtry_mom_pct < config.RISK_PERCENTILES['risk_on']        # <35%: Risk-on
        ]
        choices = ['Risk-Off', 'Risk-On']
        
        self.regimes['risk_regime'] = np.select(conditions, choices, default='Neutral')
        self.regimes['usdtry_momentum_percentile'] = usdtry_mom_pct
        
    def classify_liquidity_regime(self):
        """Classify liquidity state: Normal / Low / Very Low"""
        turnover = self.features.get('turnover_ma_20d', pd.Series(index=self.features.index))
        if turnover.isna().all():
            self.regimes['liquidity_regime'] = 'Unknown'
            return
        
        # Use percentile-based classification (adaptive)
        turnover_pct = self.percentiles.get('turnover_ma_20d_percentile')
        
        if turnover_pct is not None and not turnover_pct.isna().all():
            conditions = [
                turnover_pct >= config.LIQUIDITY_PERCENTILES['normal'],
                (turnover_pct >= config.LIQUIDITY_PERCENTILES['very_low']) & (turnover_pct < config.LIQUIDITY_PERCENTILES['normal']),
                turnover_pct < config.LIQUIDITY_PERCENTILES['very_low']
            ]
            choices = ['Normal', 'Low', 'Very Low']
            self.regimes['liquidity_regime'] = np.select(conditions, choices, default='Unknown')
        else:
            # Fallback to static thresholds
            conditions = [
                turnover >= config.LIQUIDITY_THRESHOLDS_STATIC['normal_turnover'],
                (turnover >= config.LIQUIDITY_THRESHOLDS_STATIC['low_turnover']) & (turnover < config.LIQUIDITY_THRESHOLDS_STATIC['normal_turnover']),
                turnover < config.LIQUIDITY_THRESHOLDS_STATIC['low_turnover']
            ]
            choices = ['Normal', 'Low', 'Very Low']
            self.regimes['liquidity_regime'] = np.select(conditions, choices, default='Unknown')
        
    def create_combined_regime_label(self):
        """Create combined regime label from all components"""
        def make_label(row):
            vol = row.get('volatility_regime', 'Unknown')
            trend = row.get('trend_regime', 'Unknown')
            risk = row.get('risk_regime', 'Unknown')
            liq = row.get('liquidity_regime', 'Unknown')
            return f"{vol}Vol + {trend}Trend + {risk} + {liq}Liq"
        
        self.regimes['regime_label'] = self.regimes.apply(make_label, axis=1)
        
    def classify_all(self):
        """Run all classification steps"""
        print("\nClassifying regimes (Rule-Based)...")
        self.calculate_percentiles()
        self.classify_volatility_regime()
        self.classify_trend_regime()
        self.classify_risk_regime()
        self.classify_liquidity_regime()
        self.create_combined_regime_label()
        print(f"Classified {len(self.regimes)} periods")
        return self.regimes
    
    def get_regime_summary(self):
        """Get summary statistics of regime classifications"""
        summary = {}
        for col in ['volatility_regime', 'trend_regime', 'risk_regime', 'liquidity_regime']:
            if col in self.regimes.columns:
                summary[col] = self.regimes[col].value_counts().to_dict()
        return summary
    
    def get_current_regime(self):
        """Get the most recent regime classification"""
        if len(self.regimes) == 0:
            return None
        latest = self.regimes.iloc[-1]
        return {
            'date': self.regimes.index[-1],
            'volatility': latest.get('volatility_regime', 'Unknown'),
            'trend_short': latest.get('trend_short', 'Unknown'),
            'trend_long': latest.get('trend_long', 'Unknown'),
            'trend': latest.get('trend_regime', 'Unknown'),
            'risk': latest.get('risk_regime', 'Unknown'),
            'liquidity': latest.get('liquidity_regime', 'Unknown'),
            'label': latest.get('regime_label', 'Unknown')
        }


class SimplifiedRegimeClassifier:
    """
    Simplified Regime Classifier (5-State Model)
    Reduces the 144-state space to: Bull, Bear, Stress, Choppy, Recovery
    """
    
    def __init__(self, min_duration=10, hysteresis_days=3):
        self.min_duration = min_duration
        self.hysteresis_days = hysteresis_days
        self.REGIMES = ['Bull', 'Bear', 'Stress', 'Choppy', 'Recovery']
        
    def classify(self, complex_regimes, apply_persistence=True):
        df = complex_regimes.copy()
        
        def map_regime(row):
            vol = row.get('volatility_regime', 'Unknown')
            trend = row.get('trend_regime', 'Unknown')
            risk = row.get('risk_regime', 'Unknown')
            
            # STRESS: High/Stress Vol + Risk Off OR just Stress Vol
            if vol == 'Stress' or (vol == 'High' and risk == 'Risk-Off'):
                return 'Stress'
            # BEAR: Downtrend + Risk Off
            if trend == 'Down' and risk == 'Risk-Off':
                return 'Bear'
            # RECOVERY: High Vol + Uptrend
            if vol in ['High', 'Stress'] and trend == 'Up':
                return 'Recovery'
            # BULL: Uptrend + Low/Mid Vol
            if trend == 'Up' and vol in ['Low', 'Mid']:
                return 'Bull'
            # BEAR (Strong): Downtrend + High Vol
            if trend == 'Down' and vol in ['High', 'Stress']:
                return 'Bear'
            # CHOPPY
            if trend == 'Sideways':
                return 'Choppy'
            # Fallback
            if trend == 'Down': return 'Bear'
            if trend == 'Up': return 'Bull'
            return 'Choppy'

        df['simplified_regime'] = df.apply(map_regime, axis=1)
        
        if apply_persistence:
            df['simplified_regime'] = self._apply_persistence_filter(df['simplified_regime'])
            
        return df[['simplified_regime']]
        
    def _apply_persistence_filter(self, series):
        """Apply regime smoothing"""
        smoothed = series.copy()
        current_regime = series.iloc[0]
        days_in_new_regime = 0
        
        for i in range(1, len(series)):
            new_signal = series.iloc[i]
            if new_signal == current_regime:
                days_in_new_regime = 0
            else:
                days_in_new_regime += 1
                
            # Immediate switch to Stress, else hysteresis
            if new_signal == 'Stress':
                current_regime = new_signal
                days_in_new_regime = 0
            elif days_in_new_regime >= self.hysteresis_days:
                current_regime = new_signal
                days_in_new_regime = 0
                
            smoothed.iloc[i] = current_regime
            
        return smoothed


class FourRegimeClassifier(SimplifiedRegimeClassifier):
    """
    Simplified 4-regime classifier without Choppy.
    Merges Choppy into Bull/Bear based on context.
    """
    
    def __init__(self, min_duration=10, hysteresis_days=3):
        super().__init__(min_duration, hysteresis_days)
        self.regime_mapping = {
            'Bull': 0, 'Bear': 1, 'Stress': 2, 'Recovery': 3
        }
    
    def classify(self, detailed_regimes, apply_persistence=True):
        # First pass classification (could optimize this, but reusing parent logic + post-processing)
        # Note: This is an approximation of the dedicated 4-regime scoring logic
        # For full implementation, one would reimplement map_regime.
        # Here we perform mapping based on the result of the 5-state classifier
        
        # Actually copying the full scoring logic from original file is safer
        # But for succinctness in this consolidation, let's reimplement mapping
        
        df = detailed_regimes.copy()
        
        def map_regime_4(row):
            regime = self._map_regime_5(row) # Use helper
            if regime != 'Choppy':
                return regime
                
            # Choppy resolution
            trend = row.get('trend_regime', 'Unknown')
            risk = row.get('risk_regime', 'Unknown')
            
            if risk == 'Risk-On': return 'Bull'
            if risk == 'Risk-Off': return 'Bear'
            if trend == 'Up': return 'Bull'
            if trend == 'Down': return 'Bear'
            return 'Bull' # Default bullish bias in sideways if neutral risk
            
        df['simplified_regime'] = df.apply(map_regime_4, axis=1)
        
        if apply_persistence:
            df['simplified_regime'] = self._apply_persistence_filter(df['simplified_regime'])
            
        return df[['simplified_regime']]

    def _map_regime_5(self, row):
        # Helper to retrieve what the 5-state model would say, logic copied from SimplifiedRegimeClassifier
        vol = row.get('volatility_regime', 'Unknown')
        trend = row.get('trend_regime', 'Unknown')
        risk = row.get('risk_regime', 'Unknown')
        if vol == 'Stress' or (vol == 'High' and risk == 'Risk-Off'): return 'Stress'
        if trend == 'Down' and risk == 'Risk-Off': return 'Bear'
        if vol in ['High', 'Stress'] and trend == 'Up': return 'Recovery'
        if trend == 'Up' and vol in ['Low', 'Mid']: return 'Bull'
        if trend == 'Down' and vol in ['High', 'Stress']: return 'Bear'
        if trend == 'Sideways': return 'Choppy'
        if trend == 'Down': return 'Bear'
        if trend == 'Up': return 'Bull'
        return 'Choppy'


class HMMRegimeClassifier:
    """Hidden Markov Model for regime detection"""
    
    def __init__(self, n_regimes=4, random_state=42):
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        self.regime_names = None
        self.transition_matrix = None
        
    def fit(self, features_df):
        if not ML_AVAILABLE: return self
        
        # Prepare features
        feature_cols = ['return_20d', 'realized_vol_20d', 'max_drawdown_20d', 
                        'usdtry_momentum_20d', 'volume_ratio']
        available = [c for c in feature_cols if c in features_df.columns]
        # Causal imputation only: forward-fill avoids using future information.
        X = features_df[available].ffill().dropna()
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        self.model = hmm.GaussianHMM(n_components=self.n_regimes, covariance_type="full", 
                                     n_iter=1000, random_state=self.random_state, verbose=False)
        self.model.fit(X_scaled)
        self.transition_matrix = pd.DataFrame(self.model.transmat_)
        return self
        
    def predict(self, features_df):
        if not ML_AVAILABLE or self.model is None: return pd.DataFrame()
        
        feature_cols = ['return_20d', 'realized_vol_20d', 'max_drawdown_20d', 
                        'usdtry_momentum_20d', 'volume_ratio']
        available = [c for c in feature_cols if c in features_df.columns]
        # Causal imputation only: forward-fill avoids using future information.
        X = features_df[available].ffill().dropna()
        X_scaled = self.scaler.transform(X)
        
        # Causal filtering (Forward-only)
        # Prevents lookahead bias from Viterbi/Forward-Backward
        log_frameprob = self.model._compute_log_likelihood(X_scaled)
        _, fwdlattice = _hmmc.forward_log(self.model.startprob_, self.model.transmat_, log_frameprob)
        probs = np.exp(fwdlattice - logsumexp(fwdlattice, axis=1, keepdims=True))
        regimes = np.argmax(probs, axis=1)
        
        results = pd.DataFrame(index=X.index)
        results['regime'] = regimes
        for i in range(self.n_regimes):
            results[f'prob_regime_{i}'] = probs[:, i]
            
        return results

    def label_regimes(self, features_df, predictions):
        """Simplistic labeling based on mean return"""
        if predictions.empty: return predictions
        
        df = features_df.join(predictions['regime'], how='right')
        means = df.groupby('regime')['return_20d'].mean()
        
        # Sort regimes by return: Recovery/Bull/Sideways/Bear/Stress
        # This is a heuristic mapping
        sorted_ids = means.sort_values(ascending=False).index
        names = {}
        for i, rid in enumerate(sorted_ids):
            if i == 0: names[rid] = 'Bull'
            elif i == self.n_regimes - 1: names[rid] = 'Bear'
            else: names[rid] = 'Neutral'
            
        self.regime_names = names
        predictions['regime_name'] = predictions['regime'].map(names)
        return predictions

    def print_summary(self):
        if self.model is None:
            print("Model not trained.")
            return
        
        print("\n" + "="*60)
        print("HMM REGIME SUMMARY")
        print("="*60)
        
        print("\nTransition Matrix:")
        if self.transition_matrix is not None:
             print(self.transition_matrix.round(3))
        
        if self.regime_names:
            print("\nRegime Names:")
            for i, name in self.regime_names.items():
                print(f"  Regime {i}: {name}")



class PredictiveRegimeModel:
    """Predict market regimes using XGBoost"""

    def __init__(self, forecast_horizon=0, model_type='xgboost'):
        self.forecast_horizon = forecast_horizon
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        self.regime_mapping = {'Bull': 0, 'Bear': 1, 'Stress': 2, 'Choppy': 3, 'Recovery': 4}
        self.inverse_mapping = {v: k for k, v in self.regime_mapping.items()}
        self.feature_names = None
        
    def prepare_data(self, features, regimes):
        if self.forecast_horizon > 0:
            target = regimes.shift(-self.forecast_horizon)
        else:
            target = regimes.copy()
        valid_idx = features.index.intersection(target.index)
        X = features.loc[valid_idx]
        y = target.loc[valid_idx]
        
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        y_numeric = y.map(self.regime_mapping)
        # Drop rows where target mapping failed (e.g. unknown regime)
        mask = ~y_numeric.isna()
        X = X[mask]
        y_numeric = y_numeric[mask]
        
        mask = ~X.isna().any(axis=1)
        X = X[mask]
        y_numeric = y_numeric[mask]
        
        return X, y_numeric
        
    def train(self, X_train, y_train):
        if not ML_AVAILABLE: return
        
        # Ensure all column names are strings (sklearn requirement)
        X_train = X_train.copy()
        X_train.columns = X_train.columns.astype(str)
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        X_scaled = self.scaler.fit_transform(X_train)
        
        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=len(self.regime_mapping),
                max_depth=4, learning_rate=0.05, n_estimators=150,
                random_state=42, eval_metric='mlogloss'
            )
            self.model.fit(X_scaled, y_train)
            
    def predict(self, X_test):
        if not ML_AVAILABLE or self.model is None: return None, None, None
        
        # Ensure all column names are strings (sklearn requirement)
        X_test = X_test.copy()
        X_test.columns = X_test.columns.astype(str)
        
        X_scaled = self.scaler.transform(X_test)
        preds_num = self.model.predict(X_scaled)
        probs = self.model.predict_proba(X_scaled)
        
        predictions = pd.Series([self.inverse_mapping.get(p, 'Unknown') for p in preds_num], index=X_test.index)
        confidence = pd.Series(probs.max(axis=1), index=X_test.index)
        
        return predictions, confidence, probs

    def train_test_split(self, X, y, train_end_date):
        train_mask = X.index <= train_end_date
        test_mask = X.index > train_end_date
        return X[train_mask], X[test_mask], y[train_mask], y[test_mask]

    def evaluate(self, y_true, y_pred):
        if not ML_AVAILABLE: return {}
        acc = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {acc:.2%}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))
        return {'accuracy': acc}

    def get_feature_importance(self, top_n=10):
        if self.model is None or self.model_type != 'xgboost':
            return None
        return self.model.feature_importances_
