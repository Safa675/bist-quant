# bist_quant

Quantitative trading library for BIST (Borsa Istanbul).

## Installation

```bash
# Core library (signals, backtesting, portfolio)
pip install bist-quant

# With API server support
pip install bist-quant[api]

# Full installation (all features)
pip install bist-quant[full]
```

## Quick Start

### As a Library

```python
from bist_quant.signals import MomentumSignal
from bist_quant.engines import BacktestEngine

signal = MomentumSignal(lookback=20)
engine = BacktestEngine(signals=[signal])
results = engine.run(start="2024-01-01", end="2024-12-31")
```

### As an API Server

```python
from bist_quant.api import create_app

app = create_app()

# Run with uvicorn
# uvicorn app:app --reload
```

### Using Services Directly

```python
from bist_quant.services import RealtimeService, SystemService
from bist_quant.jobs import JobManager, JobType

# Real-time quotes
realtime = RealtimeService()
quote = await realtime.get_quote("THYAO")

# System diagnostics
system = SystemService()
diagnostics = system.diagnostics_snapshot()

# Job management
jobs = JobManager()
job = await jobs.submit(JobType.BACKTEST, {"strategy": "momentum"})
```

## Modules

| Module | Description |
| --- | --- |
| `bist_quant.signals` | Signal generation (momentum, value, quality, technical) |
| `bist_quant.engines` | Backtest and optimization engines |
| `bist_quant.portfolio` | Portfolio construction and analytics |
| `bist_quant.services` | Business logic services (realtime, system) |
| `bist_quant.jobs` | Async job management |
| `bist_quant.security` | Rate limiting, authentication utilities |
| `bist_quant.config` | Configuration management |
| `bist_quant.observability` | Logging, metrics, telemetry |
| `bist_quant.api` | Optional FastAPI server factory |

## License

MIT
