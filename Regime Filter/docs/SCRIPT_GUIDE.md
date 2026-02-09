# BIST Regime Filter - Complete Script Guide

This guide explains what each script in the Regime Filter system does and how to use them.

---

## 📁 Directory Structure

```
Regime Filter/
├── Core Modules (Main functionality)
├── API (REST API & WebSocket server)
├── Models (Machine Learning models)
├── Alerts (Email notification system)
├── Utility Scripts (Examples, debugging, evaluation)
└── Configuration
```

---

## 🎯 Core Modules

### `regime_filter.py` - Main Orchestration Class
**Purpose**: The main entry point that ties everything together.

**What it does**:
- Loads market data (XU100 index, USD/TRY exchange rate)
- Calculates technical features (volatility, momentum, drawdowns)
- Classifies market regimes (Bull, Bear, Stress, Choppy, Recovery)
- Exports results to JSON and CSV

**How to use**:
```python
from regime_filter import RegimeFilter

# Initialize
rf = RegimeFilter()

# Run complete pipeline
regimes = rf.run_full_pipeline(fetch_usdtry=True)

# Get current regime
current = rf.get_current_regime()
print(f"Current regime: {current['label']}")

# Export results
rf.export_regimes()
```

**When to use**: This is your starting point for basic regime classification.

---

### `market_data.py` - Data Loading & Feature Engineering
**Purpose**: Handles all data loading and feature calculation.

**What it does**:
- **DataLoader**: Fetches XU100 prices, USD/TRY data, and individual stock prices
- **FeatureEngine**: Calculates 30+ features including:
  - Volatility metrics (realized vol, vol-of-vol)
  - Momentum indicators (20d, 60d, 120d, 252d)
  - Drawdown metrics
  - USD/TRY trend and volatility
  - Volume ratios
- **LeadingIndicatorsFetcher**: Gets global macro data (VIX, DXY, SPX) and Turkey-specific indicators

**How to use**:
```python
from market_data import DataLoader, FeatureEngine

# Load data
loader = DataLoader()
data = loader.load_all(fetch_usdtry=True)

# Calculate features
engine = FeatureEngine(data)
features = engine.calculate_all_features()

# Shift features to avoid look-ahead bias
features_shifted = engine.shift_for_prediction(shift_days=1)
```

**When to use**: When you need raw data or want to calculate custom features.

---

### `regime_models.py` - Regime Classification Models
**Purpose**: Contains all regime classification logic.

**What it does**:
- **RegimeClassifier**: Rule-based classifier using percentile thresholds
  - Classifies **volatility** (4 states) using 20d realized vol rolling percentile (252-day window):
    - **Low**: 0-25th percentile (below ~17% annualized)
    - **Mid**: 25-75th percentile (~17-28% annualized)
    - **High**: 75-92nd percentile (~28-40% annualized)
    - **Stress**: >92nd percentile (>40% annualized)
  - Classifies **trend** (4 states) by combining short-term (20d) and long-term (120d) returns:
    - **Up**: return > +1.5% (short) / > +3.0% (long)
    - **Down**: return < -1.5% (short) / < -3.0% (long)
    - **Sideways**: between thresholds, or one horizon sideways
    - **Mixed**: short and long horizons contradict (e.g. short Up + long Down)
  - Classifies **risk** (3 states) based on USD/TRY 20d momentum percentile:
    - **Risk-On**: below 35th percentile
    - **Neutral**: 35-65th percentile
    - **Risk-Off**: above 65th percentile
  - Classifies **liquidity** (3 states) based on 20d avg turnover percentile:
    - **Normal**: >=40th percentile
    - **Low**: 20-40th percentile
    - **Very Low**: <20th percentile
  - Combines into **4 × 4 × 3 × 3 = 144** possible states

- **SimplifiedRegimeClassifier**: Reduces 144 states to 5 regimes
  - Bull, Bear, Stress, Choppy, Recovery
  - Applies persistence filtering to reduce noise

- **HMMRegimeClassifier**: Unsupervised Hidden Markov Model
  - Discovers regimes from data patterns
  - No predefined rules

- **PredictiveRegimeModel**: XGBoost-based predictive model
  - Predicts future regime N days ahead
  - Uses all features including leading indicators

**How to use**:
```python
from regime_models import RegimeClassifier, SimplifiedRegimeClassifier

# Rule-based classification
classifier = RegimeClassifier(features)
detailed_regimes = classifier.classify_all()

# Simplify to 5 regimes
simple = SimplifiedRegimeClassifier()
regimes = simple.classify(detailed_regimes)

# Get current regime
current = regimes.iloc[-1]
print(f"Current regime: {current}")
```

**When to use**: When you want to classify regimes or experiment with different classification methods.

---

### `config.py` - Configuration Parameters
**Purpose**: Central configuration file for all system parameters.

**What it does**:
- Defines data paths (absolute paths that work from any directory)
- Sets feature calculation windows (volatility, momentum, drawdown)
- Configures regime classification thresholds (percentiles)
- API and scheduler settings
- Model paths and ensemble weights
- Alert configuration

**How to use**:
```python
import config

# Access configuration
print(f"Data directory: {config.DATA_DIR}")
print(f"Volatility windows: {config.VOLATILITY_WINDOWS}")
print(f"Output directory: {config.OUTPUT_DIR}")
```

**When to use**: When you need to adjust thresholds or paths. **This is the first place to look when tuning the system.**

---

## 🚀 Main Runner Scripts

### `run_full_pipeline.py` - Complete Pipeline Runner
**Purpose**: Runs the entire regime filter pipeline from start to finish.

**What it does**:
1. Fetches TCMB data (Turkish Central Bank indicators)
2. Loads market data (XU100, USD/TRY)
3. Calculates features
4. Trains/loads all models (XGBoost, LSTM, HMM)
5. Creates ensemble predictions
6. Generates comprehensive report
7. Saves all outputs

**How to use**:
```bash
# From terminal
cd "/home/safa/Documents/Markets/BIST/Regime Filter"
python run_full_pipeline.py
```

**When to use**: When you want to run everything at once and get a complete analysis. **This is the main script you'll run regularly.**

---

### `examples.py` - Usage Examples
**Purpose**: Demonstrates how to use the system with three examples.

**What it does**:
- **Simple Example**: Basic regime classification and position sizing
- **Enhanced Example**: HMM analysis and backtesting
- **Predictive Example**: Full ML pipeline with leading indicators

**How to use**:
```bash
# Run simple example
python examples.py simple

# Run enhanced example with HMM
python examples.py enhanced

# Run predictive example with ML
python examples.py predictive
```

**When to use**: When learning how to use the system or testing specific features.

---

## 🤖 Machine Learning Models

### `models/ensemble_regime.py` - Ensemble Model
**Purpose**: Combines XGBoost, LSTM, and HMM for robust predictions.

**What it does**:
- Trains three different models (XGBoost, LSTM, HMM)
- Combines predictions using weighted voting
- Dynamically adjusts weights based on rolling accuracy
- Provides confidence scores and model agreement metrics
- Can predict current regime or N days ahead

**How to use**:
```python
from models.ensemble_regime import EnsembleRegimeModel

# Initialize
ensemble = EnsembleRegimeModel(
    weights={'xgboost': 0.4, 'lstm': 0.35, 'hmm': 0.25},
    dynamic_weights=True
)

# Train
ensemble.train(features, regimes, train_end_date='2023-12-31')

# Predict
predictions, confidence, disagreement = ensemble.predict(features)

# Get current regime
current = ensemble.predict_current(features)
print(f"Regime: {current['prediction']}, Confidence: {current['confidence']:.2%}")

# Save/load
ensemble.save('outputs/ensemble_model')
loaded = EnsembleRegimeModel.load('outputs/ensemble_model')
```

**When to use**: When you want the most robust predictions using multiple models.

---

### `models/lstm_regime.py` - LSTM Sequence Model
**Purpose**: Uses temporal patterns to predict future regimes.

**What it does**:
- Takes sequences of 20 days of features
- Predicts regime N days ahead (default: 5 days)
- Captures regime transitions and momentum patterns
- Supports both PyTorch and TensorFlow backends
- Provides probability distributions over regimes

**How to use**:
```python
from models.lstm_regime import LSTMRegimeModel

# Initialize
lstm = LSTMRegimeModel(
    sequence_length=20,
    forecast_horizon=5,
    hidden_size=64,
    num_layers=2
)

# Prepare sequences
X, y = lstm.prepare_sequences(features, regimes)
X_train, X_test, y_train, y_test = lstm.train_test_split(X, y)

# Train
history = lstm.train(X_train, y_train, X_val=X_test, y_val=y_test, epochs=50)

# Predict
predictions, probabilities, confidence = lstm.predict(X_test)

# Evaluate
metrics = lstm.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.2%}")
```

**When to use**: When you want to predict future regimes or capture temporal patterns.

---

## 📊 Visualization & Analysis

### `dashboard.py` - Interactive Dashboard
**Purpose**: Creates interactive Plotly visualizations.

**What it does**:
- Creates comprehensive HTML dashboard with:
  - Price chart with regime backgrounds
  - Volatility metrics over time
  - Trend and momentum indicators
  - Risk-on/off indicators (USD/TRY)
  - Liquidity metrics
  - Regime distribution pie charts

**How to use**:
```python
from dashboard import RegimeDashboard

# Create dashboard
dashboard = RegimeDashboard(data, features, regimes)
dashboard.create_dashboard(output_file='outputs/regime_dashboard.html')

# Open outputs/regime_dashboard.html in browser
```

**When to use**: When you want to visualize regime classifications and market conditions.

---

### `evaluation.py` - Backtesting & Optimization
**Purpose**: Backtest trading strategies and optimize parameters.

**What it does**:
- **RegimeBacktester**: Tests regime-based strategies
  - Regime filter (avoid certain regimes)
  - Regime rotation (different allocations per regime)
  - Buy-and-hold benchmark
  - Calculates Sharpe ratio, max drawdown, etc.

- **ScoreOptimizer**: Grid search for optimal parameters
- **walk_forward_evaluation**: Out-of-sample testing

**How to use**:
```python
from evaluation import RegimeBacktester

# Initialize
backtester = RegimeBacktester(prices, regimes)

# Test avoiding stress regime
avoid_stress = backtester.backtest_regime_filter(['Stress'])

# Test regime rotation
rotation = backtester.backtest_regime_rotation({
    'Bull': 1.5, 'Recovery': 1.0, 'Choppy': 0.5, 
    'Bear': 0.2, 'Stress': 0.0
})

# Compare strategies
comparison = backtester.compare_strategies([avoid_stress, rotation])
print(comparison)
```

**When to use**: When you want to backtest trading strategies or optimize parameters.

---

### `strategies.py` - Trading Strategies
**Purpose**: Position sizing and allocation logic.

**What it does**:
- **DynamicAllocator**: Adaptive position sizing
  - Volatility targeting
  - Kelly criterion
  - Confidence-based allocation

- **ThreeTierStrategy**: Simple 3-tier position sizing
  - Aggressive (Bull): 100%
  - Neutral (Bear/Choppy/Recovery): 50%
  - Defensive (Stress): 0%

**How to use**:
```python
from strategies import ThreeTierStrategy

# Create strategy
strategy = ThreeTierStrategy(
    aggressive_weight=1.0,
    neutral_weight=0.5,
    defensive_weight=0.0
)

# Get position size
position = strategy.get_position_size('Bull')  # Returns 1.0

# Backtest
results = strategy.backtest(regimes, returns)
print(f"Sharpe: {results['sharpe_ratio']:.2f}")
```

**When to use**: When implementing trading strategies based on regimes.

---

## 🌐 API & Real-Time System

### `run_api.py` - API Server Entry Point
**Purpose**: Starts the FastAPI server.

**What it does**:
- Checks dependencies
- Loads environment variables
- Starts FastAPI server on port 8000
- Displays available endpoints

**How to use**:
```bash
# Start API server
python run_api.py

# Or with uvicorn directly
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**When to use**: When you want to run the regime filter as a web service.

---

### `api/main.py` - FastAPI Application
**Purpose**: REST API and WebSocket endpoints.

**What it does**:
- **REST Endpoints**:
  - `GET /regime/current` - Current regime prediction
  - `GET /regime/history` - Historical regimes
  - `GET /regime/prediction` - N-day ahead forecast
  - `GET /features/current` - Current feature values
  - `POST /regime/backtest` - Run backtest
  - `POST /regime/refresh` - Manual data refresh

- **WebSocket**:
  - `WS /ws/regime` - Real-time regime updates

- **Documentation**:
  - `/docs` - Swagger UI
  - `/redoc` - ReDoc

**How to use**:
```bash
# Start server
python run_api.py

# Access in browser
# http://localhost:8000/docs - API documentation
# http://localhost:8000/regime/current - Current regime

# Or use curl
curl http://localhost:8000/regime/current
```

**When to use**: When you want programmatic access to regime predictions.

---

### `api/scheduler.py` - Automated Updates
**Purpose**: Schedules daily regime updates.

**What it does**:
- Runs daily at 18:00 Istanbul time (after market close)
- Fetches new data
- Calculates features
- Runs ensemble prediction
- Broadcasts WebSocket updates
- Triggers email alerts on regime changes

**How to use**:
```python
from api.scheduler import get_scheduler

# Get scheduler
scheduler = get_scheduler()

# Add callbacks
scheduler.on_regime_change = lambda prev, new, date: print(f"Changed: {prev} -> {new}")

# Start
scheduler.start()

# Manual trigger
await scheduler.trigger_manual_update()
```

**When to use**: When running the API server - it handles automatic updates.

---

### `api/websocket.py` - WebSocket Manager
**Purpose**: Real-time updates via WebSocket.

**What it does**:
- Manages WebSocket connections
- Broadcasts regime updates to all connected clients
- Sends heartbeat messages
- Handles client disconnections

**When to use**: Automatically used by the API server for real-time updates.

---

### `api/models.py` - API Data Models
**Purpose**: Pydantic models for API request/response validation.

**What it does**:
- Defines data structures for API endpoints
- Validates input/output
- Generates API documentation

**When to use**: Automatically used by FastAPI for validation.

---

## 📧 Alert System

### `alerts/email_alerts.py` - Email Notifications
**Purpose**: Sends email alerts for important events.

**What it does**:
- Sends alerts when:
  - Regime changes (e.g., Bull → Stress)
  - High-confidence stress prediction
  - Model disagreement exceeds threshold
  - Volatility spike detected
- Rate limiting (max alerts per day)
- HTML formatted emails
- Async and sync sending

**How to use**:
```python
from alerts.email_alerts import send_regime_alert

# Send regime change alert
send_regime_alert(
    'regime_change',
    previous_regime='Bull',
    new_regime='Stress',
    confidence=0.85
)

# Send stress warning
send_regime_alert(
    'stress',
    confidence=0.92
)
```

**Configuration** (set environment variables):
```bash
export SMTP_HOST="smtp.gmail.com"
export SMTP_PORT="587"
export SMTP_USERNAME="your-email@gmail.com"
export SMTP_PASSWORD="your-app-password"
export ALERT_RECIPIENTS="recipient1@example.com,recipient2@example.com"
```

**When to use**: When you want email notifications for regime changes.

---

### `alerts/alert_config.py` - Alert Configuration
**Purpose**: Configuration for email alerts.

**What it does**:
- Loads SMTP settings from environment
- Defines alert thresholds
- Validates configuration

**When to use**: Automatically loaded by email alert system.

---

## 🔧 Utility Scripts

### `debug_ensemble.py` - Ensemble Debugger
**Purpose**: Comprehensive debugging for ensemble model.

**What it does**:
- Tests data loading and integrity
- Verifies model loading and state
- Checks prediction consistency
- Tests feature alignment
- Tests edge cases
- Memory and performance profiling
- Generates debugging report

**How to use**:
```bash
python debug_ensemble.py
```

**When to use**: When the ensemble model isn't working correctly.

---

### `evaluate_ensemble.py` - Ensemble Evaluation
**Purpose**: Evaluates ensemble model performance.

**What it does**:
- Compares ensemble vs random baseline
- Calculates accuracy, precision, recall, F1
- Confusion matrix analysis
- Per-regime performance metrics
- Generates markdown report

**How to use**:
```bash
python evaluate_ensemble.py
```

**When to use**: When you want to assess ensemble model quality.

---

## 📝 Quick Start Guide

### For Basic Regime Classification:
```bash
# 1. Run the simple example
python examples.py simple

# 2. Or run the full pipeline
python run_full_pipeline.py

# 3. Check outputs/regime_dashboard.html
```

### For Trading Strategy Development:
```python
from regime_filter import RegimeFilter
from strategies import ThreeTierStrategy
from evaluation import RegimeBacktester

# Get regimes
rf = RegimeFilter()
regimes = rf.run_full_pipeline()

# Create strategy
strategy = ThreeTierStrategy()

# Backtest
backtester = RegimeBacktester(rf.data['XU100_Close'], regimes)
results = strategy.backtest(regimes, rf.data['XU100_Close'].pct_change())
print(f"Sharpe: {results['sharpe_ratio']:.2f}")
```

### For API/Real-Time Usage:
```bash
# 1. Set up environment variables (optional)
export TCMB_EVDS_API_KEY="your-key"
export SMTP_USERNAME="your-email@gmail.com"
export SMTP_PASSWORD="your-password"

# 2. Start API server
python run_api.py

# 3. Access at http://localhost:8000/docs
```

---

## 🎓 Script Usage Summary

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `regime_filter.py` | Main orchestration | Starting point for basic usage |
| `run_full_pipeline.py` | Complete pipeline | Regular analysis runs |
| `examples.py` | Usage examples | Learning the system |
| `market_data.py` | Data & features | Custom feature engineering |
| `regime_models.py` | Classification | Experimenting with classifiers |
| `models/ensemble_regime.py` | ML ensemble | Robust predictions |
| `models/lstm_regime.py` | Sequence model | Future regime prediction |
| `dashboard.py` | Visualization | Visual analysis |
| `evaluation.py` | Backtesting | Strategy testing |
| `strategies.py` | Position sizing | Trading implementation |
| `run_api.py` | API server | Web service deployment |
| `api/main.py` | API endpoints | Programmatic access |
| `api/scheduler.py` | Auto updates | Production deployment |
| `alerts/email_alerts.py` | Notifications | Alert setup |
| `debug_ensemble.py` | Debugging | Troubleshooting |
| `evaluate_ensemble.py` | Model evaluation | Performance assessment |

---

## 💡 Common Workflows

### Daily Regime Check:
```bash
python examples.py simple
```

### Full Analysis with All Models:
```bash
python run_full_pipeline.py
```

### Backtest a Strategy:
```python
python -c "
from regime_filter import RegimeFilter
from evaluation import RegimeBacktester

rf = RegimeFilter()
regimes = rf.run_full_pipeline()
bt = RegimeBacktester(rf.data['XU100_Close'], regimes)
results = bt.backtest_regime_filter(['Stress'])
print(results)
"
```

### Run as Production Service:
```bash
# Set up alerts
export SMTP_USERNAME="your-email@gmail.com"
export SMTP_PASSWORD="your-password"
export ALERT_RECIPIENTS="alerts@example.com"

# Start API with scheduler
python run_api.py
```

---

## 🔍 Need Help?

- **Configuration issues**: Check `config.py`
- **Model not working**: Run `debug_ensemble.py`
- **Understanding features**: Check `market_data.py` docstrings
- **API documentation**: Visit `http://localhost:8000/docs` after starting server
- **Examples**: Run `python examples.py simple/enhanced/predictive`
