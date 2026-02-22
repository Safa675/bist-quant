# BIST Quant

**BIST Quant** is a quantitative finance library for Python, designed for backtesting and analysis on BIST (Borsa Istanbul) and multi-asset markets.

## Features

- **74+ signal builders** for momentum, value, quality, volatility, and technical indicators
- **Backtesting engine** with vectorized execution and realistic constraints
- **Portfolio tools** for allocation and optimization workflows
- **Risk controls** including VaR, CVaR, max drawdown, and position sizing helpers
- **Multi-asset support** across BIST, crypto, US stocks, FX, commodities, and funds
- **Performance-focused stack** with NumPy and pandas
- **Extensible architecture** for custom signal and strategy research

## Installation

```bash
pip install bist-quant
```

For optional integrations:

```bash
pip install bist-quant[full]
```

## Quick Example

```python
from bist_quant import PortfolioEngine

engine = PortfolioEngine(options={
    "signal": "momentum",
    "lookback_period": 21,
    "top_n": 10,
})

result = engine.run_backtest(
    signals=["momentum"],
    start_date="2023-01-01",
    end_date="2023-12-31",
)

print(f"Sharpe Ratio: {result.metrics.get('sharpe', 0.0):.2f}")
print(f"Total Return: {result.metrics.get('total_return', 0.0):.2%}")
```

## Documentation

- [Getting Started](getting-started/installation.md)
- [User Guide](user-guide/signals.md)
- [API Reference](api-reference/core.md)
- [Examples](examples/basic-backtest.md)

## License

MIT License. See [LICENSE](https://github.com/Safa675/BIST/blob/main/LICENSE).
