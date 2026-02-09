#!/usr/bin/env python3
"""
BIST Regime Filter - Full Pipeline Runner

This script runs the complete regime filter pipeline:
1. Fetch TCMB data (VIOP30 proxy, CDS, yield curve, inflation)
2. Load market data (XU100, USD/TRY)
3. Calculate features
4. Train all models (XGBoost, LSTM, HMM)
5. Create ensemble
6. Generate predictions and reports
7. Export results

Usage:
    python run_full_pipeline.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set up paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"

sys.path.insert(0, str(SCRIPT_DIR))


def _resolve_data_dir() -> Path:
    """Resolve the canonical data directory, preferring shared project data."""
    candidates = [
        PROJECT_ROOT / "data",
        SCRIPT_DIR / "data",
    ]

    for candidate in candidates:
        if (candidate / "xu100_prices.csv").exists():
            return candidate

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def _resolve_fetcher_dir(data_dir: Path) -> Path | None:
    """Resolve Fetcher-Scrapper location across possible layouts."""
    candidates = [
        data_dir / "Fetcher-Scrapper",
        PROJECT_ROOT / "data" / "Fetcher-Scrapper",
        SCRIPT_DIR / "data" / "Fetcher-Scrapper",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


DATA_DIR = _resolve_data_dir()
FETCHER_DIR = _resolve_fetcher_dir(DATA_DIR)

# Set TCMB API key from file
API_KEY_FILE = DATA_DIR / "EVDS API.txt"
if API_KEY_FILE.exists():
    with open(API_KEY_FILE, 'r') as f:
        api_key = f.read().strip()
        os.environ['TCMB_EVDS_API_KEY'] = api_key
        print(f"Loaded TCMB EVDS API key")

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Constants
TRAIN_END_DATE = '2023-12-31'


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)


def print_section(title):
    """Print formatted section header"""
    print(f"\n{'-'*60}")
    print(f" {title}")
    print(f"{'-'*60}")


def run_pipeline():
    """Run the complete regime filter pipeline"""

    start_time = datetime.now()

    print_header("BIST REGIME FILTER - FULL PIPELINE")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data directory: {DATA_DIR}")
    if FETCHER_DIR is not None:
        print(f"Fetcher directory: {FETCHER_DIR}")
    else:
        print("Fetcher directory: Not found (TCMB phase will use fallback mode)")
    print(f"Output directory: {OUTPUT_DIR}")

    # ================================================================
    # PHASE 1: FETCH TCMB DATA
    # ================================================================
    print_header("PHASE 1: FETCHING TCMB DATA")

    try:
        if FETCHER_DIR is None:
            raise FileNotFoundError("Fetcher-Scrapper directory not found")

        if str(FETCHER_DIR) not in sys.path:
            sys.path.append(str(FETCHER_DIR))
        from tcmb_data_fetcher import TCMBDataFetcher

        tcmb_fetcher = TCMBDataFetcher(
            api_key=os.environ.get('TCMB_EVDS_API_KEY'),
            cache_dir=DATA_DIR / "tcmb_cache"
        )

        tcmb_data = tcmb_fetcher.fetch_all(start_date='2013-01-01')

        # Save TCMB data
        tcmb_file = OUTPUT_DIR / "tcmb_indicators.csv"
        tcmb_data.to_csv(tcmb_file)
        print(f"\nTCMB data saved to: {tcmb_file}")
        print(f"Columns: {list(tcmb_data.columns)}")
        print(f"Date range: {tcmb_data.index[0]} to {tcmb_data.index[-1]}")

    except Exception as e:
        print(f"TCMB data fetch failed: {e}")
        print("Continuing without TCMB data...")
        tcmb_data = None

    # ================================================================
    # PHASE 2: LOAD MARKET DATA AND CALCULATE FEATURES
    # ================================================================
    print_header("PHASE 2: LOADING MARKET DATA & FEATURES")

    from regime_filter import RegimeFilter

    # Initialize regime filter
    rf = RegimeFilter(data_dir=str(DATA_DIR))

    print_section("Loading XU100 and USD/TRY data")
    rf.load_data(fetch_usdtry=True, load_stocks=False)

    print_section("Calculating technical features")
    rf.calculate_features()  # Now includes shift_for_prediction()

    # TCMB indicators are NOT merged into rf.features for ML training.
    # TCMB data availability is unreliable (API failures, missing columns),
    # which causes train/predict feature mismatches. The ML models only use
    # core features (XU100 + USD/TRY derived) that are always available.
    # TCMB data is saved separately for analysis/reporting purposes.
    if tcmb_data is not None:
        print_section("TCMB indicators available (saved separately, not used for ML)")
        print(f"  {len(tcmb_data.columns)} TCMB indicators: {list(tcmb_data.columns)}")

    # Save features (core features only — what ML models will use)
    features_file = OUTPUT_DIR / "all_features.csv"
    rf.features.to_csv(features_file)
    print(f"\nFeatures saved to: {features_file}")
    print(f"Total features: {len(rf.features.columns)}")

    # ================================================================
    # PHASE 3: CLASSIFY REGIMES (Rule-based)
    # ================================================================
    print_header("PHASE 3: RULE-BASED REGIME CLASSIFICATION")

    rf.classify_regimes()

    # Get simplified regimes
    from regime_models import SimplifiedRegimeClassifier
    simple_classifier = SimplifiedRegimeClassifier(min_duration=10, hysteresis_days=3)
    simplified_regimes_df = simple_classifier.classify(rf.regimes, apply_persistence=True)
    
    # CRITICAL FIX: Extract Series from DataFrame
    simplified_regimes = simplified_regimes_df['simplified_regime']

    # Print distribution
    print_section("Regime Distribution")
    regime_counts = simplified_regimes.value_counts()
    total = len(simplified_regimes)
    for regime, count in regime_counts.items():
        pct = count / total * 100
        print(f"  {regime:12s}: {count:5d} days ({pct:5.1f}%)")

    # Current regime
    current_regime = simplified_regimes.iloc[-1]
    current_date = simplified_regimes.index[-1]
    print(f"\nCurrent Regime: {current_regime} (as of {current_date.date()})")

    # ================================================================
    # PHASE 4: TRAIN HMM MODEL
    # ================================================================
    print_header("PHASE 4: HMM MODEL TRAINING")

    try:
        from regime_models import HMMRegimeClassifier

        hmm_model = HMMRegimeClassifier(n_regimes=4)
        
        # Filter for training (avoid look-ahead leakage in regime labeling)
        hmm_train_features = rf.features[rf.features.index <= TRAIN_END_DATE]
        hmm_model.fit(hmm_train_features)

        # Learn regime name mapping only from training period.
        hmm_train_predictions = hmm_model.predict(hmm_train_features)
        hmm_model.label_regimes(hmm_train_features, hmm_train_predictions)

        # Inference on full history with fixed train-derived mapping.
        hmm_predictions = hmm_model.predict(rf.features)
        if hmm_model.regime_names:
            hmm_predictions['regime_name'] = hmm_predictions['regime'].map(hmm_model.regime_names).fillna('Neutral')

        hmm_model.print_summary()

        # Save HMM model
        hmm_file = OUTPUT_DIR / "hmm_model.pkl"
        with open(hmm_file, 'wb') as f:
            pickle.dump(hmm_model, f)
        print(f"\nHMM model saved to: {hmm_file}")

    except Exception as e:
        print(f"HMM training failed: {e}")
        hmm_model = None

    # ================================================================
    # PHASE 5: TRAIN XGBOOST MODEL
    # ================================================================
    print_header("PHASE 5: XGBOOST MODEL TRAINING")

    try:
        from regime_models import PredictiveRegimeModel

        xgb_model = PredictiveRegimeModel(forecast_horizon=0, model_type='xgboost')

        X, y = xgb_model.prepare_data(rf.features, simplified_regimes)
        X_train, X_test, y_train, y_test = xgb_model.train_test_split(
            X, y, train_end_date=TRAIN_END_DATE
        )

        xgb_model.train(X_train, y_train)

        # Evaluate
        predictions, confidence, probs = xgb_model.predict(X_test)
        metrics = xgb_model.evaluate(y_test, predictions.map(xgb_model.regime_mapping))

        # Feature importance
        xgb_model.get_feature_importance(top_n=15)

        # Save XGBoost model
        xgb_file = OUTPUT_DIR / "xgboost_model.pkl"
        with open(xgb_file, 'wb') as f:
            pickle.dump(xgb_model, f)
        print(f"\nXGBoost model saved to: {xgb_file}")

    except Exception as e:
        print(f"XGBoost training failed: {e}")
        import traceback
        traceback.print_exc()
        xgb_model = None

    # ================================================================
    # PHASE 6: TRAIN LSTM MODEL
    # ================================================================
    print_header("PHASE 6: LSTM MODEL TRAINING")

    try:
        from models.lstm_regime import LSTMRegimeModel

        lstm_model = LSTMRegimeModel(
            sequence_length=20,
            forecast_horizon=0,
            hidden_size=64,
            num_layers=2,
            dropout=0.2
        )

        X_seq, y_seq, seq_dates = lstm_model.prepare_sequences(
            rf.features, simplified_regimes, return_dates=True
        )
        X_train_seq, X_test_seq, y_train_seq, y_test_seq = lstm_model.train_test_split_by_date(
            X_seq, y_seq, seq_dates, TRAIN_END_DATE
        )

        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            raise ValueError(
                f"Insufficient LSTM data after date split (train={len(X_train_seq)}, test={len(X_test_seq)}). "
                f"Adjust TRAIN_END_DATE or sequence settings."
            )

        # Keep validation inside training set (do not use test set for early stopping).
        val_size = max(1, int(len(X_train_seq) * 0.15)) if len(X_train_seq) >= 20 else 0
        if val_size > 0 and len(X_train_seq) > val_size:
            X_train_core = X_train_seq[:-val_size]
            y_train_core = y_train_seq[:-val_size]
            X_val_seq = X_train_seq[-val_size:]
            y_val_seq = y_train_seq[-val_size:]
        else:
            X_train_core = X_train_seq
            y_train_core = y_train_seq
            X_val_seq = None
            y_val_seq = None

        X_train_scaled, X_test_scaled = lstm_model._scale_sequences(X_train_core, X_test_seq)
        X_val_scaled = lstm_model._transform_sequences(X_val_seq) if X_val_seq is not None else None

        # Train with validation
        lstm_model.train(
            X_train_scaled, y_train_core,
            X_val_scaled, y_val_seq,
            epochs=30,
            batch_size=32,
            early_stopping_patience=5
        )

        # Evaluate
        lstm_metrics = lstm_model.evaluate(X_test_scaled, y_test_seq)

        # Save LSTM model
        lstm_file = OUTPUT_DIR / "lstm_model.pt"
        lstm_model.save(lstm_file)
        print(f"\nLSTM model saved to: {lstm_file}")

    except Exception as e:
        print(f"LSTM training failed: {e}")
        import traceback
        traceback.print_exc()
        lstm_model = None

    # ================================================================
    # PHASE 7: CREATE ENSEMBLE MODEL
    # ================================================================
    print_header("PHASE 7: ENSEMBLE MODEL")

    try:
        from models.ensemble_regime import EnsembleRegimeModel

        # Check which models are available
        available = []
        if xgb_model is not None:
            available.append('xgboost')
        if lstm_model is not None:
            available.append('lstm')
        if hmm_model is not None:
            available.append('hmm')

        print(f"Available models for ensemble: {available}")

        if len(available) >= 2:
            ensemble = EnsembleRegimeModel(forecast_horizon=0)

            # Manually assign trained models
            ensemble.xgboost_model = xgb_model
            ensemble.lstm_model = lstm_model
            ensemble.hmm_model = hmm_model
            ensemble.available_models = available
            ensemble.is_trained = True
            # Store only numeric feature names (exclude regime string columns)
            ensemble.feature_names = rf.features.select_dtypes(include=[np.number]).columns.tolist()
            ensemble.train_date = datetime.now()

            # Normalize weights for available models
            total_weight = sum(ensemble.weights[m] for m in available)
            for model in available:
                ensemble.weights[model] /= total_weight

            print(f"Ensemble weights: {ensemble.weights}")

            # Get current prediction
            current_pred = ensemble.predict_current(rf.features)

            print_section("Current Ensemble Prediction")
            print(f"  Regime: {current_pred['prediction']}")
            print(f"  Confidence: {current_pred['confidence']:.1%}")
            print(f"  Disagreement: {current_pred['disagreement']:.1%}")
            print(f"  Model Agreement: {current_pred['model_agreement']}")

            # Save ensemble
            ensemble_dir = OUTPUT_DIR / "ensemble_model"
            ensemble.save(ensemble_dir)
            print(f"\nEnsemble model saved to: {ensemble_dir}")

        else:
            print("Not enough models for ensemble (need at least 2)")
            ensemble = None

    except Exception as e:
        print(f"Ensemble creation failed: {e}")
        import traceback
        traceback.print_exc()
        ensemble = None

    # ================================================================
    # PHASE 8: GENERATE REPORTS
    # ================================================================
    print_header("PHASE 8: GENERATING REPORTS")

    # Export regimes
    rf.export_regimes(OUTPUT_DIR)

    # Export simplified regimes
    simplified_file = OUTPUT_DIR / "simplified_regimes.csv"
    simplified_regimes.to_csv(simplified_file, header=['regime'])
    print(f"Simplified regimes saved to: {simplified_file}")

    # Create summary report
    create_summary_report(
        rf, simplified_regimes, tcmb_data,
        xgb_model, lstm_model, hmm_model, ensemble,
        OUTPUT_DIR
    )

    # ================================================================
    # COMPLETE
    # ================================================================
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    print_header("PIPELINE COMPLETE")
    print(f"Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {elapsed:.1f} seconds")
    print(f"\nOutput files saved to: {OUTPUT_DIR}")

    # List output files
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        size = f.stat().st_size if f.is_file() else 0
        if f.is_dir():
            print(f"  [DIR] {f.name}/")
        else:
            print(f"  {f.name} ({size/1024:.1f} KB)")

    return {
        'regime_filter': rf,
        'simplified_regimes': simplified_regimes,
        'tcmb_data': tcmb_data,
        'xgb_model': xgb_model,
        'lstm_model': lstm_model,
        'hmm_model': hmm_model,
        'ensemble': ensemble
    }


def create_summary_report(rf, simplified_regimes, tcmb_data, xgb_model, lstm_model, hmm_model, ensemble, output_dir):
    """Create a comprehensive summary report"""

    report_file = output_dir / "pipeline_summary.txt"

    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BIST REGIME FILTER - PIPELINE SUMMARY REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")

        # Data Summary
        f.write("DATA SUMMARY\n")
        f.write("-"*40 + "\n")
        f.write(f"Date Range: {rf.features.index[0].date()} to {rf.features.index[-1].date()}\n")
        f.write(f"Total Trading Days: {len(rf.features)}\n")
        f.write(f"Technical Features: {len(rf.features.columns)}\n")
        if tcmb_data is not None:
            f.write(f"TCMB Indicators: {len(tcmb_data.columns)}\n")
        f.write("\n")

        # Current Regime
        f.write("CURRENT MARKET STATE\n")
        f.write("-"*40 + "\n")
        current = rf.get_current_regime()
        f.write(f"Date: {current['date'].date()}\n")
        f.write(f"Simplified Regime: {simplified_regimes.iloc[-1]}\n")  # Now a string, not a row
        f.write(f"Volatility: {current['volatility']}\n")
        f.write(f"Trend (Short): {current['trend_short']}\n")
        f.write(f"Trend (Long): {current['trend_long']}\n")
        f.write(f"Risk: {current['risk']}\n")
        f.write(f"Liquidity: {current['liquidity']}\n")
        f.write("\n")

        # Key Metrics
        f.write("KEY CURRENT METRICS\n")
        f.write("-"*40 + "\n")
        latest = rf.features.iloc[-1]
        key_metrics = [
            ('realized_vol_20d', 'Volatility (20d)', '{:.2%}'),
            ('return_20d', 'Return (20d)', '{:.2%}'),
            ('max_drawdown_20d', 'Max Drawdown (20d)', '{:.2%}'),
            ('usdtry_momentum_20d', 'USD/TRY Momentum', '{:.2%}'),
            ('volume_ratio', 'Volume Ratio', '{:.2f}'),
        ]
        for key, label, fmt in key_metrics:
            if key in latest.index:
                try:
                    f.write(f"{label}: {fmt.format(latest[key])}\n")
                except:
                    pass
        f.write("\n")

        # Regime Distribution
        f.write("REGIME DISTRIBUTION (Full History)\n")
        f.write("-"*40 + "\n")
        regime_counts = simplified_regimes.value_counts()
        total = len(simplified_regimes)
        for regime, count in regime_counts.items():
            pct = count / total * 100
            f.write(f"{regime:12s}: {count:5d} days ({pct:5.1f}%)\n")
        f.write("\n")

        # Model Performance
        f.write("MODEL SUMMARY\n")
        f.write("-"*40 + "\n")

        if xgb_model is not None:
            f.write("XGBoost: Trained\n")
        else:
            f.write("XGBoost: Not trained\n")

        if lstm_model is not None:
            f.write(f"LSTM: Trained (backend: {lstm_model.backend})\n")
        else:
            f.write("LSTM: Not trained\n")

        if hmm_model is not None:
            f.write(f"HMM: Trained ({hmm_model.n_regimes} regimes)\n")
        else:
            f.write("HMM: Not trained\n")

        if ensemble is not None:
            f.write(f"Ensemble: Active (models: {ensemble.available_models})\n")
            f.write(f"  Weights: {ensemble.weights}\n")
        else:
            f.write("Ensemble: Not created\n")
        f.write("\n")

        # Ensemble Prediction
        if ensemble is not None:
            try:
                pred = ensemble.predict_current(rf.features)
                f.write("ENSEMBLE PREDICTION\n")
                f.write("-"*40 + "\n")
                f.write(f"Predicted Regime: {pred['prediction']}\n")
                f.write(f"Confidence: {pred['confidence']:.1%}\n")
                f.write(f"Disagreement: {pred['disagreement']:.1%}\n")
                f.write("Model Agreement:\n")
                for model, prediction in pred['model_agreement'].items():
                    f.write(f"  {model}: {prediction}\n")
                f.write("\nProbabilities:\n")
                for regime, prob in sorted(pred['probabilities'].items(), key=lambda x: -x[1]):
                    f.write(f"  {regime}: {prob:.1%}\n")
            except Exception as e:
                f.write(f"Ensemble prediction error: {e}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

    print(f"Summary report saved to: {report_file}")


if __name__ == "__main__":
    results = run_pipeline()
