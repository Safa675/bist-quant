# `common/` — Shared Infrastructure

## Purpose

The backbone of the library. Contains the data loading stack, backtesting engine, risk management, caching layers, configuration management, data path resolution, staleness handling, utility functions, and report generation. Most other modules depend on `common/`.

## Files

```
common/
├── data_paths.py          # Centralized, env-aware file path resolver (singleton)
├── data_loader.py         # Primary data loading class with per-dataset caches
├── data_manager.py        # High-level orchestrator: loads all datasets in one call
├── config_manager.py      # Strategy config loader (Python-first, YAML fallback)
├── backtester.py          # Event-loop backtesting engine
├── backtest_services.py   # Decomposed helper services used by Backtester
├── risk_manager.py        # Liquidity filter, vol-targeting, stop-loss, slippage
├── report_generator.py    # CSV artifact writer for backtest outputs
├── benchmarking.py        # Performance benchmark harness (synthetic data)
├── disk_cache.py          # Thread-safe TTL disk cache (Parquet or CSV.GZ + JSON sidecar)
├── panel_cache.py         # Session-scoped in-memory LRU cache for panel DataFrames
├── cache_config.py        # TTL configuration for all cache categories
├── staleness.py           # Per-ticker fundamental staleness computation + decay weights
├── utils.py               # Signal panel validation, TTM computation, Turkish locale helpers
├── portfolio_analytics.py # Multi-asset async analytics (stocks, funds, crypto, FX, US)
├── market_cap_utils.py    # Market-cap size bucket classification (large/mid/small cap)
├── enums.py               # Shared enumerations (RegimeLabel)
└── report_generator.py    # Parallel CSV backtest artifact writer
```

---

## Key Classes and Functions

### `enums.py`

| Symbol | Description |
|---|---|
| `RegimeLabel(str, Enum)` | `BULL`, `BEAR`, `RECOVERY`, `STRESS` — inherits `str` for JSON/CSV serialization |

`RegimeLabel.coerce(s)` does case-insensitive string-to-enum conversion. Always use this when loading from CSV/JSON.

---

### `data_paths.py` — Path Resolution Singleton

`DataPaths` is a dataclass that resolves all file paths via:
1. Environment variables (`BIST_DATA_DIR`, `BIST_REGIME_DIR`, `BIST_CACHE_DIR`)
2. Constructor arguments
3. Auto-detected repo root (walks parents looking for `pyproject.toml + data/`)

**All file paths are properties** — they auto-select Parquet over CSV when both exist.

```python
from bist_quant.common.data_paths import get_data_paths
paths = get_data_paths()          # singleton accessor
paths.prices_file                 # resolves to .parquet or .csv automatically
paths.reset_data_paths()          # call in tests to reset singleton
```

---

### `data_loader.py` — Primary Data Loading Class

`DataLoader` is the central dependency hub. It lazy-loads each dataset on first access and provides adapter facades as properties.

**Adapter properties (accessed as `loader.borsapy`, `loader.macro`, etc.):**

| Property | Type | Description |
|---|---|---|
| `loader.borsapy` | `BorsapyAdapter` | BIST price / fundamentals |
| `loader.macro` | `MacroAdapter` | Economic calendar + macro data |
| `loader.economic_calendar` | `EconomicCalendarProvider` | Structured event calendar |
| `loader.fixed_income` | `FixedIncomeProvider` | Bond yields, TCMB rate |
| `loader.derivatives` | `DerivativesProvider` | VIOP futures/options |
| `loader.fx_enhanced` | `FXEnhancedProvider` | Bank FX rates, gold/silver, intraday |

**Key data methods:**

| Method | Description |
|---|---|
| `load_prices()` | BIST OHLCV history (DataFrame) |
| `load_fundamentals()` | Consolidated fundamentals panel |
| `load_regime_predictions()` | Regime label Series |
| `load_xautry_prices()` | XAU/TRY gold prices |
| `load_xu100_prices()` | XU100 index prices |
| `build_close_panel()` | (dates × tickers) close price panel |
| `build_open_panel()` | (dates × tickers) open price panel |
| `build_volume_panel()` | (dates × tickers) volume panel |
| `risk_free_rate` | Live TCMB policy rate (cached) |

**Important environment variables:**

| Variable | Effect |
|---|---|
| `BIST_DATA_SOURCE` | `"borsapy"` (default) or `"local"` |
| `BIST_ENFORCE_FUNDAMENTAL_FRESHNESS` | `"1"` = enforce freshness gate |
| `BIST_ALLOW_STALE_FUNDAMENTALS` | `"1"` = bypass freshness gate |
| `BIST_MAX_MEDIAN_STALENESS_DAYS` | Override freshness threshold |

---

### `data_manager.py`

High-level wrapper over `DataLoader`:
- `DataManager.load_all()` — Loads prices (panel-first), fundamentals, regime series, FX in one call.
- `DataManager.build_runtime_context()` — Returns `dict[str, DataFrame]` for direct consumption by engines.
- `build_consolidated_prices_panel()` — Merges all per-ticker `.parquet` files; only rebuilds when staleness check fails.

---

### `config_manager.py`

Loads strategy configs from Python modules first, then YAML:
- `ConfigManager.load_signal_configs()` — Python-first, YAML fallback.
- `ConfigManager.deep_merge(base, overrides)` — Recursive dict merge.
- Module-level constants: `TOP_N`, `SLIPPAGE_BPS`, `DEFAULT_PORTFOLIO_OPTIONS`, `REGIME_ALLOCATIONS` — treat these as the **single source of truth** for all portfolio defaults.

---

### `backtester.py` — Event-Loop Backtest Engine

Runs day-by-day with monthly or quarterly rebalancing.

**Key parameters:**
- `signal_lag_days=1` (default) — enforces no lookahead. Setting `lag=0` emits a warning.
- `DEBUG` env var enables verbose per-day logging.

**Cash when regime filter is active:** redirected to XAU/TRY return (holds gold), not to zero return.

**Weight-sum check:** After every invested day, the weight vector must sum to ≤ 1 + 1e-6. Failure is a hard error.

---

### `backtest_services.py` — Decomposed Services

Service classes split out of `Backtester` for maintainability:

| Class | Responsibility |
|---|---|
| `DataPreparationService` | Align open prices, signals, regime series; identify rebalance days |
| `RebalancingSelectionService` | Select top-N on rebalance days with liquidity filter |
| `DailyReturnService` | Vectorized NumPy return lookup (pre-materialized matrix) |
| `TransactionCostModel` | Apply size-aware slippage on rebalance events |
| `BacktestMetricsService` | Compute Sharpe/Sortino/CAGR/drawdown |
| `BacktestPayloadAssembler` | Build final backward-compatible results dict |

**Extreme return neutralization:** Returns outside `[-50%, +100%]` are set to zero (split/corporate-action guard).

---

### `risk_manager.py`

Pure risk controls applied during backtesting:

| Method | Description |
|---|---|
| `filter_by_liquidity(tickers, volume, date)` | Remove below-quantile illiquid names |
| `inverse_downside_vol_weights(returns)` | 1/downside-vol position sizing |
| `apply_downside_vol_targeting(returns)` | Scale returns by rolling leverage for vol target |
| `apply_stop_loss(holdings, prices, entries)` | Remove drawdown-breached positions |
| `slippage_cost_bps(size, ticker)` | Size-aware tiered slippage (large/mid/small cap) |

All constants come from `config_manager` — never hardcode numbers here.

---

### `disk_cache.py`

Thread-safe, TTL-aware disk cache. Backed by Parquet (preferred) or CSV.GZ + JSON sidecar (`.meta.json`).

- `DiskCache(cache_dir, ttl)` — initialize.
- `get_dataframe(key)` / `set_dataframe(key, df)` — DataFrame operations.
- `get_json(key)` / `set_json(key, data)` — JSON operations.
- `is_valid(key)` — existence + non-expiry.
- `clear_expired()` — removes only TTL-expired entries.

**Threading:** All write operations wrapped in `threading.Lock`.

---

### `cache_config.py`

Centralized TTL configuration:

| Category | Default TTL |
|---|---|
| `prices`, `derivatives`, `us_stocks`, `commodities` | 4 hours |
| `fast_info`, `fx`, `crypto` | 15 minutes |
| `financials`, `dividends`, `financial_ratios` | 7 days |
| `index_components`, `gold`, `fixed_income` | 24 hours |
| `news`, `calendar`, `funds`, `macro` | 1 hour |

Override via `BIST_CACHE_TTL_<CATEGORY>` environment variables. Use `CacheTTL.from_env()` to load.

---

### `staleness.py`

Tracks how stale each ticker's fundamental data is and applies exponential decay:
- `FULL_WEIGHT_DAYS = 60` — full weight for data ≤ 60 days old.
- `DECAY_END_DAYS = 120` — linear decay to `MIN_WEIGHT = 0.1` by 120 days.
- `apply_staleness_decay(signal_panel, fund_path)` — multiplies signal scores by decay weights.

---

### `utils.py`

Essential shared utilities for signal construction:

| Function | Description |
|---|---|
| `validate_signal_panel_schema(df, dates, tickers)` | Full contract check: reindex, dtype cast, shape |
| `sum_ttm(df)` | Rolling 4-quarter TTM with gap detection |
| `apply_lag(series, ticker)` | Apply Turkish reporting lag (Q4: 70 days, Q1–Q3: 40 days) |
| `coerce_quarter_cols(df)` | Parse `YYYY/MM` column names to `DatetimeIndex` |
| `SignalDataError` | Raised for missing/invalid signal data |
| `normalize_ticker(s)` | Strip `.IS` suffix, uppercase |
| `debug_log(msg)` | Logs only when `DEBUG` env var is set |

---

### `market_cap_utils.py`

Splits the liquid universe into size buckets per date:
- Large cap: top `LARGE_CAP_PERCENTILE = 90` of market cap.
- Small cap: bottom `SMALL_CAP_PERCENTILE = 10`.
- Mid cap: everything else.
- Falls back to absolute head/tail counts when buckets are too small (< `MIN_BUCKET_NAMES = 10`).

---

## Local Rules for Contributors

1. **`validate_signal_panel_schema()` is mandatory** before any signal panel leaves `backtest_services.py`. Never skip it.
2. **No magic numbers.** All numeric constants must come from `config_manager` constants or `risk_manager` class-level constants.
3. **`signal_lag_days=1` is non-negotiable for production.** Do not remove the lag enforcement — it prevents lookahead bias.
4. **Port-to-frontend functions** (anything in `core_metrics.py`) must not use pandas or NumPy. The `common/` files except `core_metrics` may use pandas freely.
5. **Period column format is `"YYYY/MM"`** (string). Do not use other date formats for fundamentals quarter columns.
6. **Turkish reporting lags:** Q4 filings are delayed 70 days; Q1–Q3 are 40 days. These are hardcoded in `utils.apply_lag()` and must not be changed without backtesting regression.
7. **Always normalize DatetimeIndex** — strip timezone and floor to midnight via `_normalize_dt_index()` in `DataLoader`. Timezone-aware indices must never enter the system.
8. **Panel staleness check:** `build_consolidated_prices_panel()` requires the panel to be `> 100 KB` to be considered valid. Do not change this guard.
