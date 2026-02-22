# Configuration

BIST Quant supports both configuration files and runtime option overrides.

## Runtime Options

Pass options directly to `PortfolioEngine`:

```python
from bist_quant import PortfolioEngine

engine = PortfolioEngine(options={
    "top_n": 20,
    "weighting": "equal",
    "max_weight": 0.15,
    "transaction_cost": 0.001,
    "rebalance_frequency": "monthly",
})
```

## Configuration Manager

```python
from bist_quant import ConfigManager

manager = ConfigManager.from_default_paths()
all_signals = manager.load_signal_configs()
print(f"Loaded {len(all_signals)} signal configs")
```

## Environment Variables

Data and runtime behavior can also be configured through environment variables.
Use `.env.example` as a baseline for local setup.

## Validation

```python
from bist_quant import get_default_options

default_options = get_default_options()
print(default_options.keys())
```
