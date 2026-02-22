# Installation

## Requirements

- Python 3.10, 3.11, or 3.12
- pip >= 21.0

## Basic Installation

Install core package:

```bash
pip install bist-quant
```

Includes:
- Portfolio engine and backtesting
- Signal builders
- Core analytics and reporting
- CLI entry points

## Full Installation

Install all optional dependencies:

```bash
pip install bist-quant[full]
```

Adds:
- REST API support (FastAPI)
- Multi-asset loaders (crypto and US stocks)
- Borsapy integration
- Machine learning extras

## Optional Dependency Sets

```bash
# API support
pip install bist-quant[api]

# Multi-asset support
pip install bist-quant[multi-asset]

# Borsapy integration
pip install bist-quant[borsapy]

# Machine learning
pip install bist-quant[ml]

# Development tools
pip install bist-quant[dev]
```

## Development Installation

```bash
git clone https://github.com/Safa675/BIST.git
cd BIST
pip install -e ".[dev]"
```

## Verify Installation

```bash
python -c "import bist_quant; print(bist_quant.__version__)"
```

## Next Steps

- [Quick Start](quickstart.md)
- [Configuration](configuration.md)
