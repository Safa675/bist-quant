# Basic Backtest Example

This example runs a momentum factor backtest with the current API.

## Setup

```python
from bist_quant import PortfolioEngine, DataLoader, DataPaths

paths  = DataPaths()
loader = DataLoader(data_paths=paths)
engine = PortfolioEngine(data_loader=loader, options={"use_regime_filter": False})
```

## Run a Factor

```python
result = engine.run_factor("momentum")

print(f"CAGR:         {result['cagr']:.1%}")
print(f"Sharpe:       {result['sharpe']:.2f}")
print(f"Sortino:      {result['sortino']:.2f}")
print(f"Max Drawdown: {result['max_drawdown']:.1%}")
print(f"Calmar:       {result['calmar']:.2f}")
```

## Inspect Holdings

```python
# Daily returns
print(result["returns"].describe())

# Holdings snapshot
print(result["holdings_history"].tail(20))

# Regime periods
print(result["regime_performance"].keys())
```

## Compare Multiple Strategies

```python
strategies = ["momentum", "value", "quality", "composite"]
results = {name: engine.run_factor(name) for name in strategies}

for name, res in results.items():
    print(f"{name:15s}  Sharpe={res['sharpe']:.2f}  CAGR={res['cagr']:.1%}")
```

## Custom Options

```python
from bist_quant import get_default_options

options = get_default_options()
options.update({
    "top_n": 10,
    "use_regime_filter": False,
    "use_vol_targeting": True,
    "target_downside_vol": 0.15,
    "use_slippage": True,
    "slippage_bps": 5.0,
    "rebalance_frequency": "quarterly",
})

result = engine.run_factor("momentum", override_config={"portfolio_options": options})
print({k: result[k] for k in ("cagr", "sharpe", "max_drawdown")})
```
