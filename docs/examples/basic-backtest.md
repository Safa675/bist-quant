# Basic Backtest Example

This example runs a momentum factor backtest with the current API.

## Setup

```python
from bist_quant import PortfolioEngine, DataLoader, DataPaths

paths  = DataPaths()                    # auto-detects data/ from project root
loader = DataLoader(data_paths=paths)
engine = PortfolioEngine(loader=loader)
```

## Run a Factor

```python
result = engine.run_factor("momentum")

print(f"CAGR:         {result.metrics['cagr']:.1%}")
print(f"Sharpe:       {result.metrics['sharpe']:.2f}")
print(f"Sortino:      {result.metrics['sortino']:.2f}")
print(f"Max Drawdown: {result.metrics['max_drawdown']:.1%}")
print(f"Calmar:       {result.metrics['calmar']:.2f}")
```

## Inspect Holdings

```python
# Daily returns
print(result.returns.describe())

# Holdings snapshot
print(result.positions.tail(20))

# Regime periods
print(result.regime_history.value_counts())
```

## Compare Multiple Strategies

```python
strategies = ["momentum", "value", "quality", "composite"]
results = {name: engine.run_factor(name) for name in strategies}

for name, res in results.items():
    print(f"{name:15s}  Sharpe={res.metrics['sharpe']:.2f}  CAGR={res.metrics['cagr']:.1%}")
```

## Custom Options

```python
from bist_quant import get_default_options

options = get_default_options()
options.update({
    "top_n": 10,
    "use_regime_filter": True,
    "use_vol_targeting": True,
    "target_downside_vol": 0.15,
    "use_slippage": True,
    "slippage_bps": 5.0,
    "rebalance_frequency": "quarterly",
})

result = engine.run_factor("momentum", portfolio_options=options)
print(result.metrics)
```
