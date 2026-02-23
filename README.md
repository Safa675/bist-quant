# bist_quant

**Quantitative research and backtesting library for Borsa Istanbul (BIST).** Version 0.3.0.

Covers the full stack: raw data ingestion → fundamental pipeline → signal construction → event-loop backtesting → multi-asset analytics → realtime quotes → API-ready services.

---

## Installation

```bash
# Development install (recommended)
git clone https://github.com/Safa675/bist-quant.git
cd BIST
pip install -e ".[dev]"
```

Optional dependency groups:

```bash
pip install -e ".[full]"        # all features
pip install -e ".[api]"         # FastAPI server
pip install -e ".[borsapy]"     # borsapy market data
pip install -e ".[ml]"          # machine-learning extras
```

---

## Quick Start

### Run a Factor Backtest

```python
from bist_quant import PortfolioEngine, DataLoader, DataPaths

# Auto-detect data directory (or set BIST_DATA_DIR env var)
paths = DataPaths()
loader = DataLoader(data_paths=paths)

engine = PortfolioEngine(loader=loader)
result = engine.run_factor("momentum")

print(f"CAGR:         {result.metrics['cagr']:.1%}")
print(f"Sharpe:       {result.metrics['sharpe']:.2f}")
print(f"Max Drawdown: {result.metrics['max_drawdown']:.1%}")
```

### Build a Signal Manually

```python
from bist_quant.signals.factory import build_signal

signal_df = build_signal(
    name="momentum",           # see: get_available_signals()
    dates=loader.load_prices().index,
    loader=loader,
    config={},
    signal_params={"lookback_period": 252},
)
# Returns: (dates × tickers) DataFrame, float scores, NaN = exclude
```

### Discover All Signals

```python
from bist_quant.signals.factory import get_available_signals
print(get_available_signals())   # 40+ registered strategies
```

### Realtime Quotes

```python
from bist_quant.realtime.quotes import get_quote, get_quotes
quote = get_quote("THYAO")
quotes = get_quotes(["THYAO", "GARAN", "AKBNK"])
```

---

## Package Architecture

```
src/bist_quant/
├── portfolio.py          # PortfolioEngine — main backtest entry point
├── runtime.py            # Path resolution (project root, data dir)
├── analytics/            # Performance metrics (pure-Python + pandas)
├── clients/              # Data connectors: borsapy, FX, crypto, US stocks, TEFAS
├── common/               # Backtester, RiskManager, DataLoader, caching, staleness
├── configs/              # strategies.yaml — master strategy registry (40+ strategies)
├── data_pipeline/        # Fundamentals ingestion: fetch → validate → merge → freshness gate
├── engines/              # Streamlit/API engine facades (FactorLab, Screener, SignalBuilder)
├── fetchers/             # CLI scripts: seed gold prices and index history
├── jobs/                 # Async job queue with lifecycle management + progress pub/sub
├── observability/        # Structured JSON logging, metrics, crash telemetry
├── persistence/          # JSON-backed job store and run history store
├── realtime/             # Live quotes, streaming overlay, portfolio snapshot
├── regime/               # 2D regime classifier (trend × volatility) → Bull/Bear/Recovery/Stress
├── security/             # Auth (API key / JWT), rate limiting, payload sanitization
├── services/             # Service layer: CoreBackendService, RealtimeService, SystemService
├── settings/             # ProductionSettings — all config from environment variables
└── signals/              # 40+ signal/factor builders
    └── standalone_factors/   # 13 standalone factor implementations with ABC
```

---

## Module Reference

| Module | Key exports |
|---|---|
| `bist_quant` | `PortfolioEngine`, `DataLoader`, `DataPaths`, `Backtester`, `RiskManager` |
| `bist_quant.signals.factory` | `build_signal()`, `get_available_signals()` |
| `bist_quant.analytics` | `compute_performance_metrics()`, `PortfolioAnalytics`, `build_garch_volatility_forecast()` |
| `bist_quant.common.config_manager` | `ConfigManager`, `load_signal_configs()` |
| `bist_quant.common.data_paths` | `DataPaths`, `get_data_paths()` |
| `bist_quant.regime.simple_regime` | `SimpleRegimeClassifier` |
| `bist_quant.realtime.quotes` | `get_quote()`, `get_quote_streaming()` |
| `bist_quant.services.core_service` | `CoreBackendService` |
| `bist_quant.services.realtime_service` | `RealtimeService` |
| `bist_quant.jobs.manager` | `JobManager` |
| `bist_quant.security.auth` | `authenticate_request()` |

---

## Key Environment Variables

| Variable | Default | Description |
|---|---|---|
| `BIST_DATA_DIR` | auto-detected | Path to `data/` directory |
| `BIST_REGIME_DIR` | auto-detected | Path to regime outputs |
| `BIST_CACHE_DIR` | auto-detected | Disk cache directory |
| `BIST_DATA_SOURCE` | `borsapy` | `borsapy` or `local` |
| `BIST_ENFORCE_FUNDAMENTAL_FRESHNESS` | `0` | `1` = enforce freshness gate |
| `DEBUG` | unset | Set to any value for verbose backtester logs |
| `BIST_AUTH_MODE` | `none` | `none`, `api_key`, `jwt`, `either` |

---

## Data Sources

- **Primary:** [borsapy](https://github.com/borsapy/borsapy) — BIST prices, financials, FX, derivatives
- **Fallback:** [yfinance](https://github.com/ranaroussi/yfinance) — index seeding scripts only
- **MCP endpoint:** `https://borsamcp.fastmcp.app/mcp` — crypto, US stocks, TEFAS funds (async)

---

## Documentation

Full documentation lives in [`docs/`](docs/index.md):

- [Getting Started](docs/getting-started/installation.md)
- [Quick Start](docs/getting-started/quickstart.md)
- [Configuration](docs/getting-started/configuration.md)
- [Signals Guide](docs/user-guide/signals.md)
- [Backtesting Guide](docs/user-guide/backtesting.md)
- [Portfolio & Risk](docs/user-guide/portfolio.md)
- [Multi-Asset](docs/user-guide/multi-asset.md)
- [Examples](docs/examples/basic-backtest.md)

Source code documentation: every folder in [`src/bist_quant/`](src/bist_quant/) has a `README.md` with file-level API reference and local conventions.

---

## License

MIT — see [LICENSE](LICENSE).
