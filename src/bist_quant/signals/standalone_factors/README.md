# `standalone_factors/` — Independent Factor Implementations

## Purpose

A self-contained library of 13 well-defined factor signal implementations, each with a common abstract base class, standardized normalization/selection framework, and complete reference documentation. These factors are designed to be used independently of the main `BUILDERS` registry.

## Files

```
standalone_factors/
├── base.py                          # FactorSignal ABC, shared enums, data contracts
├── FACTOR_REFERENCE.py              # Pure documentation (no runtime code)
├── size_signal.py                   # Size (market cap) factor
├── value_signal.py                  # Multi-metric value composite
├── profitability_signal.py          # Profitability / gross profit factor
├── investment_signal.py             # Capex/asset-growth investment factor
├── momentum_signal.py               # Price momentum factor
├── beta_signal.py                   # Market beta factor
├── quality_signal.py                # Quality composite factor
├── liquidity_signal.py              # AMIHUD illiquidity + volume factors
├── trading_intensity_signal.py      # Trading volume intensity factor
├── sentiment_signal.py              # Analyst sentiment signal
├── fundamental_momentum_signal.py   # Earnings revision momentum
├── carry_signal.py                  # Dividend / earnings yield carry
└── defensive_signal.py              # Low-beta + low-vol defensive factor
```

---

## Architecture

### Abstract Base Class (`base.py`)

All factors inherit from `FactorSignal(ABC)` which defines:

```python
class FactorSignal(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def compute_raw_signal(
        self,
        data: FactorData,
        params: FactorParams,
    ) -> tuple[pd.DataFrame, dict]:
        """Returns (raw_scores_df, metadata_dict)"""
        ...

    def compute_signal(
        self,
        data: FactorData,
        params: FactorParams,
    ) -> SignalOutput:
        """Public entry point: applies lag → winsorize → normalize → returns SignalOutput"""
        ...
```

**`compute_signal()` pipeline** (implemented in base, do not override):
1. Call `compute_raw_signal()` to get raw scores.
2. Apply `params.lag_days` shift.
3. Winsorize at `params.winsorize_pct` percentiles.
4. Normalize via `params.normalization_method`.
5. Return `SignalOutput(scores, raw_scores, metadata, component_scores)`.

---

## Data Contracts

### `FactorData` — Input Container

| Field | Type | Description |
|---|---|---|
| `close` | `pd.DataFrame` | (dates × tickers) close prices |
| `volume` | `pd.DataFrame` | (dates × tickers) volume |
| `fundamentals` | `pd.DataFrame \| None` | Flat normalized fundamentals |
| `fundamentals_parquet` | `Path \| None` | Path to consolidated parquet |
| `shares_outstanding` | `pd.DataFrame \| None` | Per-ticker shares outstanding |
| `market_cap` | `pd.DataFrame \| None` | (dates × tickers) market cap |
| `data_loader` | `DataLoader \| None` | Full DataLoader for advanced access |

### `FactorParams` — Tunable Parameters

| Field | Default | Description |
|---|---|---|
| `normalization_method` | `ZSCORE` | How to normalize across the cross-section |
| `winsorize_pct` | `0.05` | Winsorize at 5th/95th percentile |
| `min_observations` | `20` | Min non-NaN values for a valid signal date |
| `lookback` | `252` | Lookback days for price-based signals |
| `decay_halflife` | `None` | EWMA halflife for temporal decay |
| `lag_days` | `1` | Days to shift signal before return (anti-lookahead) |
| `custom` | `{}` | Factor-specific extra parameters |

### `NormalizationMethod` Enum

| Value | Description |
|---|---|
| `ZSCORE` | Standard z-score (default) |
| `RANK` | Percentile rank [0, 1] |
| `MINMAX` | Min-max scaling [0, 1] |
| `ROBUST_ZSCORE` | Median-based z-score |
| `RAW` | No normalization |

### `SelectionMethod` Enum

| Value | Description |
|---|---|
| `TOP_N` | Select top N stocks |
| `TOP_PCT` | Select top X% of universe |
| `THRESHOLD` | Select above a score threshold |
| `DECILE_LONG_SHORT` | Top decile long, bottom decile short |
| `QUINTILE` | Long top quintile only |

### `SignalOutput`

| Field | Description |
|---|---|
| `scores` | Final normalized signal DataFrame (dates × tickers) |
| `raw_scores` | Pre-normalization scores |
| `metadata` | Dict from `compute_raw_signal` |
| `component_scores` | Dict of sub-component DataFrames (for composites) |

---

## The 13 Factors

| Factor | File | Economic Rationale |
|---|---|---|
| Size | `size_signal.py` | Small-cap premium |
| Value | `value_signal.py` | Cheap stocks outperform expensive ones |
| Profitability | `profitability_signal.py` | High-margin firms outperform |
| Investment | `investment_signal.py` | Conservative capex firms outperform aggressive |
| Momentum | `momentum_signal.py` | Winners keep winning (12-1 month) |
| Beta | `beta_signal.py` | Low-beta premium (BAB factor) |
| Quality | `quality_signal.py` | High-quality firms command a premium |
| Liquidity | `liquidity_signal.py` | AMIHUD illiquidity commands a premium |
| Trading Intensity | `trading_intensity_signal.py` | Unusual volume predicts returns |
| Sentiment | `sentiment_signal.py` | Analyst upgrades predict near-term returns |
| Fundamental Momentum | `fundamental_momentum_signal.py` | Earnings revisions drive returns |
| Carry | `carry_signal.py` | High dividend/earnings yield outperforms |
| Defensive | `defensive_signal.py` | Low-beta + low-vol defensive positioning |

See `FACTOR_REFERENCE.py` for mathematical formulas and detailed economic intuition for each factor.

---

## Usage Example

```python
from bist_quant.signals.standalone_factors.momentum_signal import MomentumSignal
from bist_quant.signals.standalone_factors.base import FactorData, FactorParams, NormalizationMethod

signal = MomentumSignal()
data = FactorData(close=close_df, volume=volume_df)
params = FactorParams(
    normalization_method=NormalizationMethod.RANK,
    lag_days=1,
    lookback=252,
)
output = signal.compute_signal(data, params)
print(output.scores)  # (dates × tickers) DataFrame, NaN = exclude
```

---

## Local Rules for Contributors

1. **Override only `compute_raw_signal()`.** The `compute_signal()` method in `FactorSignal` handles lag, winsorization, and normalization — do not re-implement those in subclasses.
2. **Return `(raw_scores_df, metadata_dict)`** from `compute_raw_signal()`. The metadata dict should document what data sources and parameters were used.
3. **All normalization is cross-sectional (per row/date).** Do not apply time-series normalization within `compute_raw_signal`.
4. **`lag_days=1` default is intentional.** Do not change the default in `FactorParams` — all factors must lag by at least 1 day.
5. **`FACTOR_REFERENCE.py` is documentation.** Do not add executable code to that file. Update it whenever you add or change a factor's formula.
6. **`custom` dict for factor-specific params.** If your factor needs parameters not in `FactorParams`, add them to the `custom` dict — do not add new fields to `FactorParams` for single-factor use.
