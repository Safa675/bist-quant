# `regime/` — Market Regime Classification

## Purpose

Classifies BIST market conditions into four named regimes (`Bull`, `Bear`, `Recovery`, `Stress`) using a 2-dimension model (trend × volatility). Regime labels are used by the backtester to apply defensive allocation (100% stocks in Bull/Recovery, 0% stocks in Bear/Stress — held in gold). Also provides macro feature engineering for research use.

## Files

```
regime/
├── simple_regime.py    # 2D (trend × vol) regime classifier for XU100
└── macro_features.py   # Macro indicator loader + risk index computation
```

---

## `simple_regime.py` — Regime Classifier

### Configuration

All parameters are in the module-level `CONFIG` dict (single source of truth):

| Parameter | Value | Rationale |
|---|---|---|
| `ma_window` | 50 | **Not 200** — shorter window, backtested to be optimal for BIST |
| `vol_window` | 63 | Rolling 63-day volatility |
| `vol_percentile_window` | 252 | 1-year rolling rank for vol percentile |
| `vol_stress_threshold` | 0.7 | Percentile above which = high-vol |
| `hysteresis_days` | 0 | Anti-whipsaw filter disabled (no-op) |

### Classification Logic

```
                 | Below MA (trend=0) | Above MA (trend=1) |
-----------------|--------------------|---------------------|
High vol (1)     |     STRESS         |     RECOVERY        |
Low vol (0)      |     BEAR           |     BULL            |
```

### Allocation Rules

| Regime | Stocks | Gold |
|---|---|---|
| `BULL` | 100% | 0% |
| `RECOVERY` | 100% | 0% |
| `BEAR` | 0% | 100% |
| `STRESS` | 0% | 100% |

### Usage

```python
from bist_quant.regime.simple_regime import SimpleRegimeClassifier, DataLoader

xu100 = DataLoader().load_xu100()
classifier = SimpleRegimeClassifier()
labels = classifier.classify(xu100)      # pd.Series of RegimeLabel
```

### `SimpleRegimeClassifier` methods
- `classify(prices)` — Full pipeline: features → classification → persistence filter → labels.
- `_calculate_features(prices)` — `above_ma` (0/1) + `vol_percentile` (rolling rank).
- `_classify_raw(features)` — Applies 2×2 mapping.
- `_apply_persistence(raw_regimes)` — Anti-whipsaw filter (no-op when `hysteresis_days=0`).

---

## `macro_features.py` — Macro Feature Engineering

Loads and computes macro indicators for regime augmentation research.

**Primary sources:** Static CSV files (`tcmb_indicators.csv`, `usdtry_data.csv`) + live borsapy supplement.

**Key outputs:**
- `compute_risk_index()` — Weighted composite: CDS percentile + VIX percentile + USDTRY momentum.
- `compute_usdtry_momentum()` — 20-day FX depreciation signal.
- `compute_cds_stress_flag()` — Rolling 90th percentile CDS stress indicator.

Provider properties are lazy-loaded: `economic_calendar`, `fixed_income`, `derivatives`, `fx_enhanced`. Failures in live provider calls fail silently — static CSVs remain the primary signal.

**`MacroConfig` dataclass** holds all tuning parameters: window sizes, risk index weights, CDS/VIX thresholds, allocation overrides.

---

## Local Rules for Contributors

1. **`ma_window = 50` is intentional and tuned.** Do not change it to 200 (the classic "golden cross" window). The 50-day window was selected after BIST-specific backtesting.
2. **All regime parameter changes require backtest regression.** Any change to `CONFIG` must be accompanied by a full backtesting run to measure impact on all strategies.
3. **`RegimeLabel` enum from `common/enums.py`** is the canonical regime type. Never use raw strings `"bull"` or `"bear"` — always use the enum.
4. **Macro features are research tools.** `macro_features.py` is not on the critical path for production backtests — it is for research and can fail gracefully. Do not add it to the main `Backtester` flow without careful testing.
5. **Static CSVs are primary.** Live provider calls supplement but do not replace them. Never make the `MacroFeatures` load depend on live API availability.
