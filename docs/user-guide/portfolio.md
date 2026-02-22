# Portfolio Management

Portfolio construction behavior is controlled through `PortfolioEngine` options.

## Common Controls

- `top_n`: number of selected assets
- `weighting`: allocation method
- `max_weight`: per-position cap
- `rebalance_frequency`: daily, weekly, monthly, or quarterly
- `transaction_cost`: turnover penalty model input

## Example Configuration

```python
engine = PortfolioEngine(options={
    "top_n": 20,
    "weighting": "equal",
    "max_weight": 0.10,
    "rebalance_frequency": "weekly",
    "transaction_cost": 0.001,
})
```

## Multi-Signal Blending

```python
result = engine.run_backtest(
    signals=["momentum", "quality", "value"],
    start_date="2023-01-01",
    end_date="2023-12-31",
)
```
