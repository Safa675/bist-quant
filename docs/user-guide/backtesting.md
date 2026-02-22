# Backtesting

## Core Workflow

1. Configure a `PortfolioEngine`
2. Select one or more signals
3. Run `engine.run_backtest(...)`
4. Inspect `PortfolioResult`

## Example

```python
from bist_quant import PortfolioEngine

engine = PortfolioEngine(options={
    "rebalance_frequency": "monthly",
    "transaction_cost": 0.001,
    "top_n": 15,
})

result = engine.run_backtest(
    signals=["momentum", "value"],
    start_date="2022-01-01",
    end_date="2024-12-31",
)

print(result.metrics)
```

## Returned Artifacts

`PortfolioResult` includes:
- `returns`
- `positions`
- `turnover`
- `transaction_costs`
- `regime_history` (optional)
- `metrics`

## Practical Tips

- Start with a single signal for baseline behavior.
- Add transaction cost and position constraints early.
- Validate sensitivity to rebalance frequency and lookback windows.
