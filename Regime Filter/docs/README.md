# BIST Regime Filter System

A comprehensive regime classification system for the BIST (Borsa Istanbul) market that identifies market conditions across multiple dimensions to inform trading strategy selection and risk management.

## Overview

The regime filter classifies market states based on four key dimensions:

1. **Volatility State** (4): Low / Mid / High / Stress
2. **Trend State** (4): Up / Down / Sideways / Mixed (combined from short + long horizons)
3. **Risk State** (3): Risk-On / Neutral / Risk-Off (based on USD/TRY dynamics)
4. **Liquidity State** (3): Normal / Low / Very Low

Total: **4 × 4 × 3 × 3 = 144 possible states**

The system generates actionable regime labels like:
- "High vol + Downtrend + Risk-off + Low liquidity" → Reduce exposure, don't mean-revert, don't lever
- "Low vol + Uptrend + Risk-on + Normal liquidity" → Increase exposure, trend-following works

> **📚 New to the system?** Check out the [**Complete Script Guide**](SCRIPT_GUIDE.md) to understand what each script does and when to use it!

## Features

### A) Price/Volatility Core
- Realized volatility (20d, 60d windows)
- Vol-of-vol (volatility of volatility)
- Drawdown metrics (max DD 20d/60d)
- Gap risk proxy (overnight returns)
- Skew proxy (downside semivariance)

### B) Trend/Momentum
- Index momentum on 2 horizons (1-3 months, 6-12 months)
- Moving average slopes
- Price vs MA ratios
- Momentum acceleration

### C) Risk-On/Off
- USD/TRY trend and volatility
- USD/TRY momentum percentiles
- Currency stress indicators

### D) Liquidity
- Volume metrics and ratios
- Turnover analysis
- High-low range (spread proxy)
- Amihud illiquidity measure

## Installation

### Requirements

```bash
pip install pandas numpy yfinance plotly hmmlearn scikit-learn matplotlib
```

**Optional (for actual TCMB rates):**
```bash
pip install evds
# Get API key from: https://evds2.tcmb.gov.tr/
```

### Directory Structure

```
Regime Filter/
├── 📁 api/                        # FastAPI application
├── 📁 alerts/                     # Alert system
├── 📁 models/                     # ML models (LSTM, Ensemble)
├── 📁 docs/                       # 📚 Documentation
│   └── README.md                  # Unified documentation
├── 📁 data/                       # Local links/cache (shared data is ../data)
├── 📁 outputs/                    # Generated outputs
├── config.py                      # Configuration
├── market_data.py                 # Data loading + feature engineering
├── regime_models.py               # Rule-based + ML regime models
├── regime_filter.py               # Main orchestration
├── strategies.py                  # Regime-aware strategy rules
├── evaluation.py                  # Backtesting/evaluation
├── evaluate_ensemble.py           # Ensemble validation script
├── examples.py                    # Simple/enhanced/predictive demos
├── dashboard.py                   # Visualization
├── run_full_pipeline.py           # Complete pipeline
├── run_api.py                     # API server
└── requirements.txt               # Dependencies

# Note: Data fetchers (tcmb_data_fetcher.py, tcmb_rates.py, etc.) are in ../data/Fetcher-Scrapper/
```

## Usage

### Quick Start

```python
from regime_filter import RegimeFilter

# Initialize and run
rf = RegimeFilter()
regimes = rf.run_full_pipeline()

# Get current regime
current = rf.get_current_regime()
print(current)

# Export results
rf.export_regimes()
```

### Complete Example

```bash
python examples.py simple
```

This will:
1. Load XU100 and USD/TRY data
2. Calculate all features
3. Classify regimes
4. Generate interactive dashboard
5. Export results to JSON and CSV
6. Provide trading recommendations

### Enhanced Example (Recommended)

```bash
python examples.py enhanced
```

This runs the full pipeline with:
1. ✅ Basic regime filter (144 states)
2. ✅ HMM regime discovery (4 learned states with transition probabilities)
3. ✅ Simplified classification (5 actionable states)
4. ✅ TCMB risk-free rate integration
5. ✅ Backtesting validation (proves regimes improve returns)

**Results:**
- Buy & Hold Sharpe: **0.33** (realistic for Turkish markets)
- Regime Rotation Sharpe: **0.92** (+181% improvement!)
- Average Turkish deposit rate: **16.29%**

### Custom Usage

```python
from regime_filter import RegimeFilter
from dashboard import RegimeDashboard

# Initialize
rf = RegimeFilter(data_dir="../data")

# Load data
data = rf.load_data(fetch_usdtry=True)

# Calculate features
features = rf.calculate_features()

# Classify regimes
regimes = rf.classify_regimes()

# Get regime summary
summary = rf.get_regime_summary()
print(summary)

# Create dashboard
dashboard = RegimeDashboard(rf.data, rf.features, rf.regimes)
dashboard.create_dashboard()
```

## Configuration

Edit `config.py` to customize:

- **Percentile thresholds**: Adjust regime boundaries
- **Window sizes**: Change calculation periods
- **Liquidity thresholds**: Set minimum volume requirements
- **Trend thresholds**: Define up/down/sideways criteria

Example (current values after 2026-01-29 recalibration):

```python
# In config.py
VOLATILITY_PERCENTILES = {
    'low': 25,      # 0-25%: Low volatility
    'mid': 75,      # 25-75%: Mid volatility
    'high': 92      # 75-92%: High, 92%+: Stress
}

TREND_THRESHOLDS = {
    'up': 0.015,    # 1.5% threshold for uptrend (short), 3.0% (long)
    'down': -0.015  # -1.5% threshold for downtrend (short), -3.0% (long)
}

RISK_PERCENTILES = {
    'risk_on': 35,   # Below 35th percentile = Risk-On
    'risk_off': 65   # Above 65th percentile = Risk-Off, between = Neutral
}

LIQUIDITY_PERCENTILES = {
    'very_low': 20,  # 0-20%: Very Low
    'low': 40,       # 20-40%: Low
    'normal': 40     # 40%+: Normal
}
```

## Output Files

### 1. regime_labels.json
Daily regime classifications in JSON format:

```json
{
  "2024-01-15": {
    "volatility": "Mid",
    "trend": "Up",
    "risk": "Risk-On",
    "liquidity": "Normal",
    "label": "MidVol + UpTrend + Risk-On + NormalLiq"
  }
}
```

### 2. regime_features.csv
Complete feature matrix with regime labels (for analysis)

### 3. regime_dashboard.html
Interactive Plotly dashboard with:
- Price chart with regime backgrounds
- Volatility evolution
- Trend indicators
- Risk-on/off signals
- Liquidity metrics

## Methodology

### Level 1 Approach: Rules + Percentiles + State Machine

This implementation uses a **transparent, rule-based approach** rather than complex ML models:

1. **Percentile-based thresholds**: Features are ranked using rolling percentiles
2. **Hard rules**: Clear thresholds define regime boundaries
3. **State machine**: Regimes transition based on feature values

**Advantages**:
- Transparent and interpretable
- Easy to debug and tune
- Minimal overfitting risk
- Fast to compute

### Regime Classification Logic

**Volatility** (4 states — 20d realized vol, 252-day rolling percentile):
| State | Percentile Range | Approx. Annualized Vol |
|-------|-----------------|----------------------|
| Low | 0-25th | below ~17% |
| Mid | 25-75th | ~17-28% |
| High | 75-92nd | ~28-40% |
| Stress | >92nd | >40% |

**Trend** (4 states — short 20d + long 120d returns combined):
| Horizon | Up | Down | Sideways |
|---------|------|------|----------|
| Short (20d) | > +1.5% | < -1.5% | between |
| Long (120d) | > +3.0% | < -3.0% | between |

Combined logic: if both agree → that state; if one is Sideways → Sideways; if they contradict (e.g. short Up + long Down) → **Mixed**

**Risk** (3 states — USD/TRY 20d momentum percentile):
| State | Percentile Range |
|-------|-----------------|
| Risk-On | below 35th |
| Neutral | 35-65th |
| Risk-Off | above 65th |

**Liquidity** (3 states — 20d avg turnover percentile, adaptive):
| State | Percentile Range |
|-------|-----------------|
| Normal | >=40th |
| Low | 20-40th |
| Very Low | <20th |

Fallback static thresholds if percentile calculation fails: Very Low < 1.5T TRY, Low < 3.0T TRY, Normal >= 5.0T TRY daily turnover.

## Trading Applications

### Strategy Selection

Use regimes to select which trading strategy to run:

- **Low vol + Uptrend**: Trend-following, higher leverage
- **High vol + Sideways**: Mean reversion, range trading
- **Stress + Risk-off**: Reduce exposure, defensive positions
- **Any + Very Low Liquidity**: Reduce sizes, avoid illiquid names

### Risk Management

Adjust risk parameters based on regime:

```python
current = rf.get_current_regime()

if current['volatility'] in ['High', 'Stress']:
    position_size *= 0.5  # Reduce size
    stop_loss_width *= 2   # Widen stops
    
if current['risk'] == 'Risk-Off':
    max_gross_exposure *= 0.7  # Reduce overall exposure
    
if current['liquidity'] == 'Very Low':
    max_position_size = min(max_position_size, daily_volume * 0.01)
```

### Example Regime Interpretations

| Regime | Interpretation | Action |
|--------|---------------|--------|
| Low vol + Uptrend + Risk-on + Normal liq | Healthy bull market | Increase exposure, trend-follow |
| High vol + Downtrend + Risk-off + Low liq | Crisis mode | Reduce to minimal exposure |
| Mid vol + Sideways + Neutral + Normal liq | Range-bound market | Mean reversion strategies |
| Stress + Down + Risk-off + Very Low liq | Extreme stress | Consider going flat |

## Enhanced Features

### 1. HMM-Based Regime Discovery

**File:** `regime_models.py` (`HMMRegimeClassifier`)

Uses Hidden Markov Models to learn regime structure from data:
- Discovers regimes without arbitrary thresholds
- Provides transition probabilities between regimes
- Calculates expected regime durations
- Gives probabilistic regime assignments

**Example:**
```python
from regime_models import HMMRegimeClassifier

hmm = HMMRegimeClassifier(n_regimes=4)
hmm.fit(features)

# Get transition matrix
trans = hmm.get_transition_probabilities()
# Bull → Bull: 97% (high persistence)
# Stress → Bear: 30% (often transitions)

# Get expected durations
durations = hmm.get_expected_duration()
# Stress: 2 days (short-lived!)
# Bull: 33 days (persistent)
```

### 2. Simplified 5-State Classification

**File:** `regime_models.py` (`SimplifiedRegimeClassifier`)

Reduces 144 possible states to 5 actionable regimes:
- **Bull**: Uptrend + Low/Mid vol + Risk-on

- **Bear**: Downtrend + Risk-off
- **Stress**: High vol + Risk-off (crisis)
- **Choppy**: Sideways + Neutral
- **Recovery**: Uptrend + High vol (post-crisis)

**Benefits:**
- Statistically robust (each regime has ~640 observations)
- Clear trading implications
- Easy to interpret and communicate

### 3. TCMB Risk-Free Rate Integration

**File:** `tcmb_rates.py`

Fetches actual Turkish deposit rates for realistic Sharpe calculations:
- Connects to TCMB EVDS API (optional)
- Falls back to USD/TRY-based approximation
- Caches rates to `/home/safa/Documents/Markets/BIST/data/tcmb_deposit_rates.csv`

**Impact:**
- Old Sharpe (RF=0%): Buy&Hold 0.95, Regime Rotation 2.01
- **New Sharpe (RF=16.3%)**: Buy&Hold **0.33**, Regime Rotation **0.92**
- **Realistic for Turkish markets!**

**Usage:**
```python
from tcmb_rates import TCMBRateFetcher

# Fetch rates (uses approximation if no API key)
fetcher = TCMBRateFetcher()
rates = fetcher.fetch_rates(start_date='2013-01-01')

# With EVDS API key (optional)
fetcher = TCMBRateFetcher(api_key='YOUR_KEY')
rates = fetcher.fetch_rates()
```

### 4. Backtesting Module

**File:** `evaluation.py` (`RegimeBacktester`)

Validates regime effectiveness through backtesting:
- Tests regime-based strategies vs buy-and-hold
- Uses actual Turkish deposit rates for Sharpe calculations
- Analyzes performance by regime

**Results (2013-2026):**

| Strategy | Annual Return | Sharpe | Max DD |
|----------|--------------|--------|--------|
| Buy & Hold | 23.79% | 0.33 | -34.33% |
| Regime Rotation | **37.00%** | **0.92** | **-22.29%** |

**Regime Performance:**

| Regime | Sharpe | Annual Return | Win Rate |
|--------|--------|---------------|----------|
| Bull | 4.72 | +93.33% | 63.47% ← Trade! |
| Stress | 0.85 | +28.41% | 55.08% |
| Choppy | 0.16 | +4.09% | 51.10% |
| Recovery | -0.80 | -26.24% | 53.70% |
| Bear | -5.25 | -116.46% | 36.67% ← Avoid! |

**Key Insight:** Regime rotation improves Sharpe by **+181%** over buy-and-hold!

## Extending the System

### Add Markov Switching (Level 2)

For smoother regime probabilities:

```python
from statsmodels.tsa.regime_switching import MarkovRegression

# Fit Markov switching model
model = MarkovRegression(returns, k_regimes=3, switching_variance=True)
result = model.fit()

# Get regime probabilities
regime_probs = result.smoothed_marginal_probabilities
```

### Add Breadth Indicators

If you have individual stock data:

```python
# In market_data.py (FeatureEngine)
def calculate_breadth_features(self):
    # % stocks above 50d MA
    # Advance/decline line
    # New highs/lows
    pass
```

### Add Supervised ML (Level 3)

For predictive regime classification:

```python
import lightgbm as lgb

# Define target (e.g., forward drawdown)
target = calculate_forward_drawdown(returns, window=20)

# Train model
model = lgb.LGBMClassifier()
model.fit(features, target)

# Predict regime
predicted_regime = model.predict(current_features)
```

## Troubleshooting

### USD/TRY Data Issues

If yfinance fails to fetch USD/TRY data, the system will use synthetic data for demonstration. To use real data:

1. Check internet connection
2. Try alternative data sources (e.g., TCMB, investing.com)
3. Manually download and place in data directory

### Missing Features

If certain features are missing:
- Check that input data has required columns
- Verify date ranges align
- Check for NaN values in source data

### Performance

For faster computation:
- Reduce `PERCENTILE_LOOKBACK` in config.py
- Use fewer feature windows
- Limit date range in data loading

## References

This implementation follows best practices from:
- Volatility regime classification (Ang & Bekaert, 2002)
- Markov switching models (Hamilton, 1989)
- Practical regime-based trading (Kritzman et al., 2012)

## Documentation

This README serves as the main documentation for the BIST Regime Filter system. It covers installation, usage, architecture, and system details in a compact format.

## License

MIT License - feel free to use and modify for your trading systems.

## Author

Built for BIST market analysis and trading strategy selection.

---

**Disclaimer**: This is a tool for market analysis. Always validate regime classifications against your own market knowledge and use appropriate risk management.
