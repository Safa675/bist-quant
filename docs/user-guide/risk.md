# Risk Management

Risk controls are available both in portfolio options and through `RiskManager`.

## Included Risk Metrics

- Volatility
- Max drawdown
- Value at Risk (VaR)
- Conditional Value at Risk (CVaR)
- Turnover and transaction cost diagnostics

## Example

```python
from bist_quant import RiskManager

risk_manager = RiskManager()
summary = risk_manager.calculate_risk_metrics(returns_series)
print(summary)
```

## Engine-Level Controls

Use portfolio options to apply guardrails during simulation:
- Position limits
- Liquidity filters
- Volatility targeting
- Stop-loss behavior
