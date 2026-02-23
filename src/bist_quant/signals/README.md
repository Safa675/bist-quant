# `signals/` — Signal & Factor Builder Layer

## Purpose

All alpha signal and factor implementations. Every signal returns a `(dates × tickers)` pandas DataFrame of float scores where higher = more attractive (unless individual signal docs say otherwise). `NaN` means the ticker is excluded on that date. The `signals/` package is the largest module in the library.

## Structure

```
signals/
├── protocol.py                   # SignalBuilder Protocol (duck-typing interface)
├── factory.py                    # Central registry (BUILDERS) + dispatch API
├── __init__.py                   # list_available_signals(), get_signal_module()
│
├── ── Config-Adapter Layers ──
├── momentum.py                   # Adapters: momentum, consistent_momentum, residual_momentum…
├── value.py                      # Adapters: value, investment, small_cap, asset_growth…
├── quality.py                    # Adapters: profitability, earnings_quality, fscore_reversal…
├── technical.py                  # Adapters: sma, donchian, xu100, macd, adx, supertrend…
├── composite.py                  # Multi-strategy combos + blending utilities
│
├── ── Low-Level Signal Implementations ──
├── factor_builders.py            # Core factor panel construction from financial data
├── factor_axes.py                # Combine raw factors into aggregate axes (quality/value/momentum)
├── axis_cache.py                 # Axis computation cache
├── orthogonalization.py          # Factor orthogonalization utilities
├── _context.py                   # Context extraction helpers (close_df, fundamentals, etc.)
├── debug_utils.py                # Signal debugging helpers
├── signal_exporter.py            # Export signal DataFrames to CSV
├── borsapy_indicators.py         # borsapy-native indicator wrappers
├── five_factor_pipeline.py       # Five-factor model pipeline
│
└── standalone_factors/           # Independent factor implementations (see sub-README)
    ├── base.py                   # FactorSignal ABC + NormalizationMethod + SelectionMethod
    ├── FACTOR_REFERENCE.py       # Pure reference documentation (no runtime code)
    └── *.py                      # 13 individual factor implementations
```

---

## The SignalBuilder Contract

Every signal builder **must** satisfy the `SignalBuilder` Protocol defined in `protocol.py`:

```python
def build_my_signal(
    dates: pd.DatetimeIndex,
    loader: DataLoader,
    config: dict,
    signal_params: dict | None = None,
) -> pd.DataFrame:  # shape: (dates × tickers), dtype: float64, NaN = exclude
    ...
```

**Rules enforced by `factory.py`:**
1. Return type must be a `pd.DataFrame`.
2. Output **must** pass `validate_signal_panel_schema()` from `common/utils.py`.
3. `signal_params` takes precedence over `config["parameters"]` (resolved by `_resolve_signal_params()`).

---

## Factory & Registry (`factory.py`)

`BUILDERS` is the merged dict from all category registries:
```
BUILDERS = {**momentum.BUILDERS, **value.BUILDERS, **quality.BUILDERS,
            **technical.BUILDERS, **composite.BUILDERS, ...}
```

**API:**
```python
from bist_quant.signals.factory import build_signal, get_available_signals

names = get_available_signals()   # sorted list of all registered signal names
df = build_signal("momentum", dates=dates, loader=loader, config=config)
```

If an unknown signal name is passed, `ValueError` is raised with the full list of available signals.

---

## Signal Categories

### Momentum Signals (`momentum.py`)
| Builder key | Description |
|---|---|
| `momentum` | 12-1 month price momentum |
| `consistent_momentum` | Multi-period momentum consistency |
| `residual_momentum` | Market-beta-adjusted momentum |
| `momentum_reversal_volatility` | Momentum + volatility reversal combo |
| `low_volatility` | Inverse trailing volatility |
| `trend_following` | MA-based trend signals |
| `sector_rotation` | Cross-sector relative momentum |
| `short_term_reversal` | 1-week mean reversion |

### Value Signals (`value.py`)
| Builder key | Description |
|---|---|
| `value` | Multi-metric value composite (P/E, P/B, EV/EBITDA, etc.) |
| `investment` | Capex and asset-growth-based investment factor |
| `small_cap` | Size (market cap) factor |
| `asset_growth` | Total asset growth signal |
| `dividend_rotation` | Dividend yield with seasonal rotation |
| `macro_hedge` | Macro-aware defensive positioning |
| `sovereign_risk` | Exposure-adjusted sovereign risk signal |

### Quality Signals (`quality.py`)
| Builder key | Description |
|---|---|
| `profitability` | Operating + gross profit margin composite |
| `earnings_quality` | Accruals and cash earnings quality |
| `fscore_reversal` | Piotroski F-score based reversal |
| `roa` | Return on assets |
| `accrual` | Total accrual signal |

### Technical Signals (`technical.py`)
| Builder key | Description |
|---|---|
| `sma` | Simple moving average crossover |
| `donchian` | Donchian channel breakout |
| `xu100` | Relative strength vs XU100 |
| `macd` | MACD histogram |
| `adx` | ADX directional strength |
| `supertrend` | Supertrend indicator |
| `ema` | EMA crossover |
| `obv` | On-balance volume |
| `atr` | ATR-normalized momentum |
| `parabolic_sar` | Parabolic SAR trend |
| `ichimoku` | Ichimoku cloud position |

### Composite Signals (`composite.py`)
Multi-strategy combinations using blending utilities:

| Function | Description |
|---|---|
| `weighted_sum(panels, weights)` | Weighted average of panel dict |
| `zscore_blend(panels, weights?)` | Cross-sectional z-score → weighted average |
| `rank_blend(panels, weights?)` | Percentile rank → weighted average |

Registered composites: `betting_against_beta`, `breakout_value`, `five_factor_rotation`, `momentum_asset_growth`, `pairs_trading`, `quality_momentum`, `quality_value`, `size_rotation_momentum`, `size_rotation_quality`, `size_rotation`, `small_cap_momentum`, `trend_value`.

---

## Low-Level Layer

### `factor_builders.py`
Builds raw factor panels directly from financial statement data (Parquet cache). Contains:
- Turkish IFRS field keys (`INCOME_SHEET`, `BALANCE_SHEET`, `CASH_FLOW_SHEET`)
- Row key tuples for ~20 financial line items — **ordered by preference** (first match wins)
- Core lookback constants: `VOLATILITY_LOOKBACK_DAYS = 63`, `BETA_LOOKBACK_DAYS = 252`, etc.

> ⚠️ Comment warning in the file: `BALANCE_SHEET` must **NOT** be changed to `"Finansal Durum Tablosu"` — this is a misnamed column in the source data.

### `factor_axes.py`
Combines raw factor panels into aggregate axes:
- `combine_quality_axis()`, `combine_value_axis()`, `combine_momentum_axis()`
- Bucketing: `N_BUCKETS = 5`, `BUCKET_LABELS = ("Q1_Low", …, "Q5_High")`
- Ensemble lookback windows: `(21, 63, 126, 252)` days (equal weighted)
- Uses `np.nansum` + `np.divide` with `where=valid_counts>0` for NaN-safe means

### `_context.py`
Shared helpers to extract `close_df`, `fundamentals`, `high_low_panels` from the `config` runtime context dict. All builder functions that need price or fundamental data call `require_context()` from here.

---

## Signal Export (`signal_exporter.py`)

```python
from bist_quant.signals.signal_exporter import SignalExporter

exporter = SignalExporter(signal_name="momentum")
path = exporter.export_factor_scores(scores_df)  # writes to outputs/signals/momentum/
```

---

## Local Rules for Contributors

1. **Every new signal must be registered in the appropriate `BUILDERS` dict** (`momentum.BUILDERS`, `value.BUILDERS`, etc.) and the `factory.py` will automatically pick it up.
2. **Output schema is strictly enforced.** Call `validate_signal_panel_schema(df, dates, tickers)` before returning from any builder.
3. **`higher score = more attractive`** is the convention. If your signal is "lower is better" (e.g. low P/E), negate it before returning.
4. **`NaN` means exclude.** Do not use `0.0` to exclude a ticker — use `np.nan`.
5. **Config-adapter pattern.** The category module (`momentum.py`, `value.py`, etc.) is a thin adapter that extracts parameters and calls the underlying `build_*_signal()` function. Keep these adapters thin — all logic goes in `*_signals.py` or `factor_builders.py`.
6. **Turkish column names in `factor_builders.py`.** When adding new financial row lookups, use verbatim Turkish labels from İş Yatırım and add them to the appropriate `FIELD_KEYS` tuple in order of preference.
7. **Lookback constants live in `factor_builders.py`** — do not hardcode lookback days anywhere else.
8. **Do not modify `factor_axes.py` constants** (`ENSEMBLE_LOOKBACK_WINDOWS`, `N_BUCKETS`) without re-running full backtests — they are tuned by historical analysis.
