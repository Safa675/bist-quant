# Installation

## Requirements

- Python 3.10, 3.11, or 3.12
- pip >= 21.0

## Install from PyPI

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

## Development Installation

```bash
git clone https://github.com/Safa675/bist-quant.git
cd BIST
pip install -e ".[dev]"
```

The `[dev]` group is for contributors. It includes testing, linting, and type-checking tools.

## What borsapy Provides

borsapy is the **primary data source** for this library. It provides:
- BIST historical OHLCV prices
- Turkish financial statements (via isyatirim.com.tr)
- FX rates (USD/TRY, EUR/TRY, XAU/TRY)
- Index composition (XU030, XU050, XU100, XUTUM)
- VIOP derivatives data
- TCMB / fixed income rates
- TradingView streaming quotes

## Data Directory Setup

By default the library uses user-scoped directories:

- data: `~/.local/share/bist-quant/data`
- regime outputs: `~/.local/share/bist-quant/regime/simple_regime`
- cache: `~/.cache/bist-quant`

Override them with environment variables when needed:

```bash
export BIST_DATA_DIR=/path/to/data
export BIST_REGIME_DIR=/path/to/regime/simple_regime
export BIST_CACHE_DIR=/path/to/.cache
```

Use this if your datasets live outside the default library directories.

## Seed Initial Data

The package does not ship market datasets. To populate the default data directory, seed baseline files after installation:

```bash
# Fetch 10 years of XAU/TRY and USD/TRY
python -m bist_quant.fetchers.fetch_gold_prices

# Fetch XU030, XU100, XUTUM index history
python -m bist_quant.fetchers.fetch_indices

# Incremental update of BIST prices
python -m bist_quant.clients.update_prices
```

## Verify Installation

```bash
python -c "import bist_quant; print(bist_quant.__version__)"
bist-quant --version
```

## Next Steps

- [Quick Start](quickstart.md)
- [Configuration](configuration.md)
