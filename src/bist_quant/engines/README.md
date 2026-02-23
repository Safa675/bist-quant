# `engines/` — High-Level Engine Facades

## Purpose

Service-layer engines that bridge the Streamlit UI pages and the core library. Each engine wraps the complex library internals into a clean, page-oriented API with response caching, input validation, and structured error handling. Engines **do not** contain business logic — they orchestrate it.

## Files

```
engines/
├── factor_lab.py           # Factor catalog, factor backtests, result caching
├── signal_construction.py  # Multi-indicator technical signal builder + backtest
├── stock_filter.py         # Multi-dimensional stock screener engine
├── technical_scanner.py    # Expression-based technical condition scanner
├── types.py                # TypedDict response shapes for all engine APIs
└── errors.py               # Engine error hierarchy with machine-readable codes
```

---

## Error Hierarchy (`errors.py`)

```
QuantEngineError(RuntimeError)
├── QuantEngineValidationError   # code = "VALIDATION_ERROR" — bad inputs
├── QuantEngineDataError         # code = "DATA_ERROR"       — missing/malformed data
└── QuantEngineExecutionError    # code = "EXECUTION_ERROR"  — computation failure
```

All errors carry:
- `.code` — stable string constant for programmatic handling.
- `.user_message` — display-safe message (defaults to technical message if not overridden).

---

## Type Contracts (`types.py`)

All engine return values are `TypedDict` — no runtime enforcement, purely for static analysis and documentation.

| Type | Returned by |
|---|---|
| `FactorCatalogResult` | Factor catalog endpoint |
| `SignalConstructionSnapshotResult` | Signal construction snapshot |
| `SignalBacktestResult` | Full signal backtest |
| `StockScreenerResult` | Stock screener page |

---

## Engine Reference

### `factor_lab.py` — Factor Lab Engine

Serves the Factor Lab Streamlit page. Manages the factor catalog, parameter validation, and backtest execution.

**Key components:**
- `PARAM_SCHEMAS` — Dict mapping factor names (`"momentum"`, `"value"`, etc.) to UI parameter schema lists (each schema entry has `type`, `min`, `max`, `default`). Used to render dynamic UI controls.
- Module-level `_RESPONSE_CACHE` — Simple `dict[key, (timestamp, result)]` with a 600-second TTL.
- `PortfolioEngine` — Thin wrapper around `bist_quant.portfolio.PortfolioEngine` that preserves backward-compatibility via `__getattr__` passthrough.
- `FactorLabEngine` — Façade delegating everything to `CoreBackendService` via `__getattr__`.

**`_resolve_paths()`** — validates and returns `RuntimePaths`. Call this at the start of every engine method.

Uses `redirect_stdout` context manager to suppress verbose library output during backtest runs.

---

### `signal_construction.py` — Signal Construction Engine

Serves the Signal Construction Streamlit page.

**`DEFAULT_INDICATORS`** — 8 indicator default parameter sets:

| Key | Indicator |
|---|---|
| `rsi` | RSI (14) |
| `macd` | MACD (12, 26, 9) |
| `bollinger` | Bollinger Bands (20, 2) |
| `atr` | ATR (14) |
| `stochastic` | Stochastic (14, 3) |
| `adx` | ADX (14) |
| `supertrend` | Supertrend (7, 3.0) |
| `ta_consensus` | TA Consensus score |

**Environment variables:**
- `PRICE_CACHE_TTL_SEC` — price cache TTL (seconds)
- `INDEX_CACHE_TTL_SEC` — index cache TTL
- `DOWNLOAD_BATCH_SIZE` — tickers per borsapy batch

**Disk cache location:** `/tmp/bist-quant-signal-cache` on Vercel; `.cache/` locally.

**`_build_backtest_analytics_v2()`** — Computes yearly/monthly returns, drawdown, VaR/CVaR, benchmark correlation, and turnover analytics from raw backtest output.

---

### `stock_filter.py` — Stock Screener Engine

**`FILTER_FIELD_DEFS`** — 29 screener fields across groups:

| Group | Example fields |
|---|---|
| `valuation` | P/E, P/B, EV/EBITDA, dividend yield |
| `quality` | ROE, ROA, current ratio, debt/equity |
| `technical` | RSI, MACD signal, distance from 52W high |
| `momentum` | 1M/3M/6M/12M return |
| `liquidity` | Average daily volume, market cap |
| `growth` | Revenue growth, EPS growth |

**`TEMPLATE_PRESETS`** — 15 named filter presets (e.g. `small_cap`, `high_dividend`, `buy_recommendation`, `low_volatility`).

**`INDEX_OPTIONS`** — `XU030`, `XU050`, `XU100`, `XUTUM`, `CUSTOM`.

**`DATA_SOURCE_OPTIONS`** — `["local", "isyatirim", "hybrid"]`.

Technical indicators delegated to `TechnicalScannerEngine`.

---

### `technical_scanner.py` — Technical Condition Scanner

Expression-based scanner wrapping borsapy's `scan()` API.

**`PREDEFINED`** — 10 built-in scan templates:

| Name | Condition |
|---|---|
| `oversold_rsi` | RSI < 30 |
| `golden_cross` | SMA(50) crosses above SMA(200) |
| `death_cross` | SMA(50) crosses below SMA(200) |
| `breakout_52w` | Price at 52-week high |
| `volume_spike` | Volume 2× average |
| (5 more…) | |

**`TechnicalScannerEngine` methods:**
- `scan(universe, condition, interval)` — Single-condition filter.
- `scan_multi(universe, conditions, interval)` — Multi-condition AND filter (inner join).
- `predefined_scans()` — Returns a copy of `PREDEFINED`.

**Backward compatibility:** Tries both `bp.scan()` (function) and `bp.TechnicalScanner()` (class) call patterns for compatibility with multiple borsapy versions.

---

## Local Rules for Contributors

1. **Engines do not contain business logic.** Computation belongs in `common/`, `analytics/`, or `signals/`. Engines only orchestrate and cache.
2. **Always use `QuantEngineError` hierarchy.** Do not raise `ValueError` or `RuntimeError` directly from engines.
3. **Response caching uses `(timestamp, result)` tuples.** TTL is 600 seconds. Do not extend it to more than 3600 seconds — stale data will confuse users.
4. **`_resolve_paths()` at the top of every method.** Never hardcode file paths inside engine methods.
5. **`PARAM_SCHEMAS` must stay in sync with signal parameter APIs.** When adding a new parameter to a signal, update `factor_lab.py:PARAM_SCHEMAS` and the corresponding signal builder.
6. **`TypedDict` is documentation, not enforcement.** Do not add `isinstance` checks for engine return types — trust the type checker.
