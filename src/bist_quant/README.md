# `bist_quant` — Borsa İstanbul Quantitative Research Library

The root package of the `bist_quant` library. Version: **0.3.0**

## Purpose

A comprehensive Python library for quantitative finance research, backtesting, and portfolio optimization focused on Borsa Istanbul (BIST) markets. The package provides the full stack from raw data ingestion through signal construction, backtesting, analytics, realtime quotes, and API-ready services.

## Package Structure

```
src/bist_quant/
├── analytics/          # Pure-Python + pandas performance analytics and metrics
├── cli/                # Command-line tools (cache management, data validation)
├── clients/            # Data connectors (borsapy, FX, crypto, US stocks, TEFAS, macro)
├── common/             # Shared infrastructure: data loading, backtesting, caching, risk
├── configs/            # YAML strategy registry
├── data_pipeline/      # Fundamentals data ingestion pipeline (fetch → validate → merge)
├── engines/            # High-level Streamlit/API engine facades (FactorLab, Screener, etc.)
├── fetchers/           # One-off CLI scripts for seeding gold/index price files
├── jobs/               # Async job queue with lifecycle management and progress pub/sub
├── observability/      # Structured logging, in-memory metrics, crash telemetry
├── persistence/        # JSON-backed async job store and synchronous run store
├── realtime/           # Live quote fetching, streaming tick overlay, portfolio snapshot
├── regime/             # Market regime classification (trend × volatility) + macro features
├── security/           # Auth (API key / JWT), rate limiting, payload sanitization
├── services/           # Service layer: core portfolio, realtime market data, system ops
├── settings/           # Production settings from environment variables
├── signals/            # All signal/factor builders (momentum, value, quality, technical…)
│   └── standalone_factors/  # Independent factor implementations with abstract base
├── portfolio.py        # Core PortfolioEngine + PortfolioResult — main backtest entry point
└── runtime.py          # Runtime path resolution (project root, data dir, regime dir)
```

## Top-Level Exports

| Symbol | Source | Description |
|---|---|---|
| `PortfolioEngine` | `portfolio.py` | Main backtest runner |
| `PortfolioResult` | `portfolio.py` | Backtest output dataclass |
| `SignalResult` | `portfolio.py` | Signal output dataclass |
| `run_backtest` | `portfolio.py` | Convenience backtest function |
| `DataLoader` | `common/data_loader.py` | Primary data loading class |
| `ConfigManager` | `common/config_manager.py` | Strategy config loader |
| `DataPaths` | `common/data_paths.py` | Centralized path resolver |
| `Backtester` | `common/backtester.py` | Event-loop backtest engine |
| `RiskManager` | `common/risk_manager.py` | Liquidity, vol-targeting, stop-loss |
| `ReportGenerator` | `common/report_generator.py` | CSV artifact writer |

## Key Conventions

### Data Source Priority
- **borsapy** is always the primary data source for live and historical BIST data.
- **yfinance** is used only as a fallback in the `fetchers/` CLI scripts.
- `BIST_DATA_SOURCE` environment variable switches between `"borsapy"` (default) and `"local"`.

### Optional Dependencies
The following dependencies are guarded with `try/except` and degrade gracefully when absent:
`borsapy`, `pandera`, `prometheus_client`, `pyjwt`, `psutil`, `httpx`

### Python / Typing Conventions
- `from __future__ import annotations` used throughout (PEP 563 deferred evaluation).
- All config and path objects are `frozen=True` dataclasses.
- `TypedDict` used for engine return shapes (no runtime enforcement, purely for static analysis).
- Type stub file `py.typed` is present — the package is fully typed.

### Environment Variables (Selected)
| Variable | Purpose |
|---|---|
| `BIST_DATA_DIR` | Override default prices/fundamentals data directory |
| `BIST_REGIME_DIR` | Override regime outputs directory |
| `BIST_CACHE_DIR` | Override cache directory |
| `BIST_DATA_SOURCE` | `"borsapy"` (default) or `"local"` |
| `BIST_ENFORCE_FUNDAMENTAL_FRESHNESS` | Enable/disable freshness gate |
| `BIST_ALLOW_STALE_FUNDAMENTALS` | Override to allow stale data |
| `DEBUG` | Enable per-day verbose backtester logging |

### Signal Builder Contract
Every signal builder callable must conform to the `SignalBuilder` Protocol:
```python
def build_my_signal(
    dates: pd.DatetimeIndex,
    loader: DataLoader,
    config: dict,
    signal_params: dict | None = None,
) -> pd.DataFrame:  # shape: (dates × tickers), float scores, NaN = exclude
    ...
```
All builders are registered in category module `BUILDERS` dicts and centralized via `signals/factory.py`.

### Error Handling
- `SignalDataError` (from `common/utils.py`) for missing/invalid signal data.
- `QuantEngineError` hierarchy (from `engines/errors.py`) for engine-layer failures.
- `FundamentalsPipelineError` hierarchy (from `data_pipeline/errors.py`) for pipeline stages.
- All errors carry human-readable messages; engine errors also carry a `code` attribute.

### Logging
- All pipeline and service code uses `log_event(logger, event, **context)` which emits structured JSON lines.
- `DEBUG` env var activates verbose per-day logging in `Backtester` without any code change.

## Quick Start

```python
from bist_quant import PortfolioEngine, DataLoader, DataPaths

# Resolve paths (reads BIST_DATA_DIR env var or auto-detects repo root)
paths = DataPaths()

# Load all market data
loader = DataLoader(data_paths=paths)

# Run a strategy backtest
engine = PortfolioEngine(loader=loader)
result = engine.run_factor("momentum")
print(result.metrics)
```
