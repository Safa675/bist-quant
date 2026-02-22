# Signals

BIST Quant provides a broad signal library covering momentum, value, quality,
technical, and composite strategies.

## Discover Available Signals

```python
from bist_quant.signals.factory import SignalFactory

signals = SignalFactory.get_available_signals()
print(f"Available signals: {len(signals)}")
print(signals[:10])
```

## Build a Signal via Factory

```python
from bist_quant.signals.factory import build_signal

signal_df = build_signal(
    name="momentum",
    dates=trading_dates,
    loader=data_loader,
    config={"signal_params": {"lookback_period": 21}},
)
```

## Signal Categories

- Momentum and trend-following
- Value and fundamental factors
- Quality and profitability
- Technical indicators
- Composite and blended strategies

## Integration in Backtests

Signals can be passed to `PortfolioEngine.run_backtest(...)` as:
- A list of signal names
- A dictionary of prebuilt signal DataFrames
- A single DataFrame
