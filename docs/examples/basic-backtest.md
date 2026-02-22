# Basic Backtest Example

This example runs a momentum signal through the portfolio engine.

```python
from bist_quant import PortfolioEngine

engine = PortfolioEngine(options={
    "signal": "momentum",
    "lookback_period": 21,
    "top_n": 10,
    "rebalance_frequency": "weekly",
})

result = engine.run_backtest(
    signals=["momentum"],
    start_date="2023-01-01",
    end_date="2023-12-31",
)

print(result.metrics)
```

Notebook version: [examples/01_basic_backtest.ipynb](https://github.com/Safa675/BIST/blob/main/examples/01_basic_backtest.ipynb)
