# `src/` — Source Code Root

This directory contains the `bist_quant` Python package — the core library for Borsa Istanbul quantitative research, backtesting, and portfolio optimization.

## Structure

```
src/
└── bist_quant/              # Main package (see bist_quant/README.md)
    ├── README.md            # Package overview, conventions, quick start
    ├── analytics/           # Performance analytics (README.md)
    ├── cli/                 # CLI utilities (README.md)
    ├── clients/             # Data connectors (README.md)
    ├── common/              # Shared infrastructure (README.md)
    ├── configs/             # Strategy YAML registry (README.md)
    ├── data_pipeline/       # Fundamentals ingestion pipeline (README.md)
    ├── engines/             # Engine facades for UI/API (README.md)
    ├── fetchers/            # Data seeding scripts (README.md)
    ├── jobs/                # Async job queue (README.md)
    ├── observability/       # Logging, metrics, telemetry (README.md)
    ├── persistence/         # JSON-backed stores (README.md)
    ├── realtime/            # Live market data (README.md)
    ├── regime/              # Regime classification (README.md)
    ├── security/            # Auth, rate limiting, sanitization (README.md)
    ├── services/            # Application service layer (README.md)
    ├── settings/            # Environment configuration (README.md)
    └── signals/             # Signal & factor builders (README.md)
        └── standalone_factors/  # Independent factor implementations (README.md)
```

## Installation

The package is installed as an editable install from the repository root:

```bash
pip install -e ".[dev]"
```

See [pyproject.toml](../../pyproject.toml) for dependency groups.

## Documentation Navigation

Each subfolder has a `README.md` explaining:
- The folder's purpose and scope
- Every file with its exports and key behavior
- Local conventions and rules that agents/contributors must follow

Start with [bist_quant/README.md](bist_quant/README.md) for the full package overview.
