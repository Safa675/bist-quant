# Installation

## Requirements

- Python 3.10, 3.11, 3.12, or 3.13
- pip >= 21.0

## Development Installation (Recommended)

```bash
git clone https://github.com/Safa675/bist-quant.git
cd BIST
pip install -e ".[dev]"
```

The `[dev]` group includes testing tools (`pytest`, `hypothesis`), type checking, and linting.

## Optional Dependency Groups

```bash
pip install -e ".[full]"         # all optional features
pip install -e ".[api]"          # FastAPI server support
pip install -e ".[borsapy]"      # borsapy BIST market data client
pip install -e ".[ml]"           # machine learning extras
```

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

The library auto-detects the `data/` directory by searching parent folders for `pyproject.toml + data/`.
To override, set environment variables:

```bash
export BIST_DATA_DIR=/path/to/data
export BIST_REGIME_DIR=/path/to/outputs/regime
export BIST_CACHE_DIR=/path/to/.cache
```

Or create a `.env` file in the project root.

## Seed Initial Data

After installation, seed baseline price files:

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
python -c "import bist_quant; print(bist_quant.__version__)"  # 0.3.0
```

## Next Steps

- [Quick Start](quickstart.md)
- [Configuration](configuration.md)
