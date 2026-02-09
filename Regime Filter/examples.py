"""
Examples Module
Consolidates usage examples for the Regime Filter system.

Functions:
1. run_simple_example(): Basic usage demonstration
2. run_enhanced_example(): Advanced usage with HMM and backtesting
3. run_predictive_example(): Full predictive pipeline with ML
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path

# Import from consolidated modules
from market_data import DataLoader, FeatureEngine, LeadingIndicatorsFetcher, create_custom_indicators
from regime_models import (
    RegimeClassifier, SimplifiedRegimeClassifier, 
    HMMRegimeClassifier, PredictiveRegimeModel
)
from strategies import ThreeTierStrategy, DynamicAllocator, create_dynamic_strategy
from evaluation import RegimeBacktester

warnings.filterwarnings('ignore')

def run_simple_example():
    """
    SIMPLE EXAMPLE - How to Use the Regime Filter
    Shows: Current regime, position size, and trading recommendation.
    """
    print("="*80)
    print("BIST REGIME FILTER - SIMPLE EXAMPLE")
    print("="*80)
    
    # Step 1: Load data
    print("📊 Step 1: Loading market data...")
    loader = DataLoader()
    data = loader.load_all(fetch_usdtry=True)
    print(f"   ✓ Loaded {len(data)} days of data")
    
    # Step 2: Calculate features
    print("\n📈 Step 2: Calculating market features...")
    engine = FeatureEngine(data)
    features = engine.calculate_all_features()
    features_shifted = engine.shift_for_prediction(shift_days=1)
    
    # Step 3: Classify regime
    print("\n🎯 Step 3: Classifying market regime...")
    classifier = RegimeClassifier(features_shifted)
    detailed_regimes = classifier.classify_all()
    
    simple_classifier = SimplifiedRegimeClassifier(min_duration=10, hysteresis_days=3)
    simplified = simple_classifier.classify(detailed_regimes, apply_persistence=True)
    
    # Step 4: Get current regime
    current_regime = simplified['simplified_regime'].iloc[-1]
    current_date = simplified.index[-1]
    
    print("\n" + "="*80)
    print("CURRENT MARKET STATUS")
    print(f"\n📅 Date: {current_date.date()}")
    print(f"🏷️  Regime: {current_regime}")
    
    # Step 5: Get recommendation
    strategy = ThreeTierStrategy()
    position_size = strategy.get_position_size(current_regime)
    print(f"\n💰 RECOMMENDED POSITION SIZE: {position_size:.0%}")
    
    if current_regime == 'Bull':
        print("   ✓ Market is in BULL regime: Go FULL position (100%)")
    elif current_regime == 'Stress':
        print("   ⚠️  Market is in STRESS regime: Go to CASH (0%)")
    else:
        print(f"   → Market is in {current_regime} regime: Use partial position")


def run_enhanced_example():
    """
    Enhanced Regime Filter Example
    Demonstrates HMM discovery and backtesting validation.
    """
    print("\n" + "="*70)
    print(" "*15 + "ENHANCED BIST REGIME FILTER")
    print("="*70)
    
    # Load and process data
    loader = DataLoader()
    data = loader.load_all(fetch_usdtry=True)
    engine = FeatureEngine(data)
    features = engine.calculate_all_features()
    features_shifted = engine.shift_for_prediction(shift_days=1)
    
    # HMM
    print("\n[HMM Analysis]")
    try:
        hmm = HMMRegimeClassifier(n_regimes=4)
        hmm.fit(features_shifted)
        preds = hmm.predict(features_shifted)
        hmm.label_regimes(features_shifted, preds)
        print("  HMM model fitted and regimes labeled")
    except Exception as e:
        print(f"  HMM failed: {e}")
        
    # Rule Based
    print("\n[Rule-Based Classification]")
    classifier = RegimeClassifier(features_shifted)
    detailed = classifier.classify_all()
    
    simple = SimplifiedRegimeClassifier()
    simplified = simple.classify(detailed)
    
    # Backtest
    print("\n[Backtesting]")
    returns = data['XU100_Close'].pct_change()
    backtester = RegimeBacktester(data['XU100_Close'], simplified)
    
    bh = backtester.backtest_buy_and_hold()
    avoid_stress = backtester.backtest_regime_filter(['Stress'])
    rotation = backtester.backtest_regime_rotation({
        'Bull': 1.5, 'Recovery': 1.0, 'Choppy': 0.5, 'Bear': 0.2, 'Stress': 0.0
    })
    
    comparison = backtester.compare_strategies([bh, avoid_stress, rotation])
    print("\n" + comparison.to_string(index=False))


def run_predictive_example():
    """
    Predictive Regime System
    Full pipeline with leading indicators and predictive ML model.
    """
    print("="*70)
    print("PREDICTIVE REGIME FORECASTING SYSTEM")
    print("="*70)
    
    # 1. Load Data
    print("Loading data...")
    loader = DataLoader()
    data = loader.load_all(fetch_usdtry=True)
    
    # 2. Base Features & Regimes
    engine = FeatureEngine(data)
    features = engine.calculate_all_features()
    # No shift yet, we need aligned timestamps for combining with leading indicators first
    # Actually, base features should be calculated. Shifting for prediction happens later or we use shift_days=0 for 'current' state
    # FeatureEngine by default is 'current' day data.
    
    classifier = RegimeClassifier(features)
    detailed = classifier.classify_all()
    simple = SimplifiedRegimeClassifier()
    current_regimes = simple.classify(detailed)
    
    # 3. Leading Indicators
    print("Fetching leading indicators...")
    fetcher = LeadingIndicatorsFetcher()
    # Short date range for example speed, or full
    leading = fetcher.fetch_all(start_date=data.index[0].strftime('%Y-%m-%d'))
    custom = create_custom_indicators(data['XU100_Close'], data.get('USDTRY'), leading)
    
    # Combine
    # Use causal fill only to avoid injecting future information in examples.
    all_features = pd.concat([features, leading, custom], axis=1).ffill()
    
    # 4. Train Model
    print("Training predictive model...")
    model = PredictiveRegimeModel(forecast_horizon=5)
    X, y = model.prepare_data(all_features, current_regimes['simplified_regime'])
    
    train_end = '2022-12-31' # Example split
    X_train, X_test, y_train, y_test = model.train_test_split(X, y, train_end)
    
    model.train(X_train, y_train)
    preds, conf, _ = model.predict(X_test)
    # preds is already a Series of regime strings
    
    # 5. Dynamic Allocation
    print("Calculating dynamic allocation...")
    allocator = DynamicAllocator()
    volatility = data['XU100_Close'].pct_change().rolling(20).std() * np.sqrt(252)
    
    allocations, method_returns = create_dynamic_strategy(
        data['XU100_Close'].loc[X_test.index],
        preds, conf, volatility.loc[X_test.index],
        allocator
    )
    
    print(f"\n✅ Predictive Sharpe: {method_returns.mean()*252 / (method_returns.std()*np.sqrt(252)):.2f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == 'simple': run_simple_example()
        elif cmd == 'enhanced': run_enhanced_example()
        elif cmd == 'predictive': run_predictive_example()
        else: print(f"Unknown command: {cmd}")
    else:
        run_simple_example()
