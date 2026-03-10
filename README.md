# bist_quant

**Quantitative research and backtesting library for Borsa Istanbul (BIST).** Version 0.3.0.

`bist_quant` is a Python library for factor research, signal construction, backtesting, portfolio analytics, and BIST-focused data workflows. It is designed to be usable from notebooks, scripts, dashboards, and external applications without requiring a repo checkout.

## Install

```bash
pip install bist-quant
```

Optional extras:

```bash
pip install "bist-quant[providers]" # httpx/yfinance/borsapy-backed providers
pip install "bist-quant[borsapy]"   # BIST market data integrations
pip install "bist-quant[ml]"        # machine-learning extras
pip install "bist-quant[full]"      # all optional research features
```

## Quick Start

```python
from bist_quant import DataLoader, DataPaths, PortfolioEngine

paths = DataPaths()
loader = DataLoader(data_paths=paths)
engine = PortfolioEngine(
    data_loader=loader,
    options={"use_regime_filter": False},
)

result = engine.run_factor("momentum")
print(f"Sharpe: {result['sharpe']:.2f}")
print(f"CAGR:   {result['cagr']:.1%}")
```

By default the library uses user-scoped directories:

- data: `~/.local/share/bist-quant/data`
- regime outputs: `~/.local/share/bist-quant/regime/simple_regime`
- cache: `~/.cache/bist-quant`

Override them with `BIST_DATA_DIR`, `BIST_REGIME_DIR`, and `BIST_CACHE_DIR` when needed.
If you are working from local parquet/CSV files without installing provider extras, set `BIST_DATA_SOURCE=local`.

## What Ships in the Library

- `portfolio.py` - `PortfolioEngine` and backtest entry points
- `signals/` - factor and signal builders
- `analytics/` - performance analytics and metrics
- `clients/` - optional data-provider integrations
- `common/` - data loading, backtester, risk, and shared helpers
- `configs/` - strategy registry and configuration loaders
- `regime/` - regime classification helpers
- `realtime/` - quote and market snapshot helpers

## Data Setup

The package does not ship datasets. Point it at an existing dataset directory or seed data into the default user-scoped location.

```bash
export BIST_DATA_DIR="$HOME/.local/share/bist-quant/data"
python -m bist_quant.fetchers.fetch_gold_prices
python -m bist_quant.fetchers.fetch_indices
python -m bist_quant.clients.update_prices
```

## Documentation

- [Installation](docs/getting-started/installation.md)
- [Quick Start](docs/getting-started/quickstart.md)
- [Configuration](docs/getting-started/configuration.md)
- [Signals Guide](docs/user-guide/signals.md)
- [Backtesting Guide](docs/user-guide/backtesting.md)
- [Portfolio & Risk](docs/user-guide/portfolio.md)
- [Multi-Asset Guide](docs/user-guide/multi-asset.md)

## Development

Contributor setup lives in `CONTRIBUTING.md`. App-stack deployment notes remain in `deploy/README.md`, but they are out of scope for the published library package.

## License

MIT - see `LICENSE`.
