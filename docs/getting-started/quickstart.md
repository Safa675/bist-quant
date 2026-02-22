# Quick Start

This guide shows a minimal end-to-end backtest flow.

## 1. Build a Portfolio Engine

```python
from bist_quant import PortfolioEngine

engine = PortfolioEngine(options={
    "signal": "momentum",
    "lookback_period": 21,
    "holding_period": 5,
    "top_n": 10,
    "rebalance_frequency": "weekly",
})
```

## 2. Run a Backtest

```python
result = engine.run_backtest(
    signals=["momentum"],
    start_date="2023-01-01",
    end_date="2023-12-31",
)

print(f"Sharpe Ratio: {result.metrics.get('sharpe', 0.0):.2f}")
print(f"Total Return: {result.metrics.get('total_return', 0.0):.2%}")
print(f"Max Drawdown: {result.metrics.get('max_drawdown', 0.0):.2%}")
```

## 3. Inspect Results

```python
print(result.returns.tail())
print(result.positions.tail())
print(result.turnover.tail())
```

## 4. Use the Functional API

```python
from bist_quant import run_backtest

result = run_backtest(
    signals=["value"],
    start_date="2023-01-01",
    end_date="2023-12-31",
    top_n=15,
)
```

## Next Steps

- [Signals Guide](../user-guide/signals.md)
- [Backtesting Guide](../user-guide/backtesting.md)
- [Examples](../examples/basic-backtest.md)
