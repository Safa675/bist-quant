# Quick Start

This guide walks through the core backtest flow from data loading to results.

## 1. Load Data

```python
from bist_quant import DataLoader, DataPaths

# DataPaths auto-detects the data/ directory from the project root
# Override with BIST_DATA_DIR env var if needed
paths = DataPaths()
loader = DataLoader(data_paths=paths)

# Access prices, fundamentals, and more
prices = loader.load_prices()           # full BIST OHLCV history
close = loader.build_close_panel()      # (dates × tickers) close prices
fundamentals = loader.load_fundamentals()  # consolidated financial statements
```

## 2. Run a Factor Backtest

```python
from bist_quant import PortfolioEngine

engine = PortfolioEngine(loader=loader)

# Run any registered strategy by name
result = engine.run_factor("momentum")

print(f"CAGR:         {result.metrics['cagr']:.1%}")
print(f"Sharpe:       {result.metrics['sharpe']:.2f}")
print(f"Max Drawdown: {result.metrics['max_drawdown']:.1%}")
print(f"Trades:       {len(result.positions)}")
```

## 3. Inspect Results

```python
# Daily return series
print(result.returns.tail())

# Holdings history (date, ticker, weight)
print(result.positions.tail())

# Regime periods
print(result.regime_history)
```

## 4. Available Strategies

```python
from bist_quant.signals.factory import get_available_signals
print(get_available_signals())  # 40+ registered signal names
```

Strategies are configured in `src/bist_quant/configs/strategies.yaml`.

## 5. Custom Portfolio Options

```python
from bist_quant import get_default_options

options = get_default_options()
options.update({
    "top_n": 15,
    "use_regime_filter": True,
    "use_vol_targeting": True,
    "target_downside_vol": 0.15,
    "use_slippage": True,
    "slippage_bps": 5.0,
    "use_stop_loss": False,
    "rebalance_frequency": "monthly",
})

result = engine.run_factor("momentum", portfolio_options=options)
```

## 6. Load a Single Signal

```python
from bist_quant.signals.factory import build_signal

signal_df = build_signal(
    name="value",
    dates=close.index,
    loader=loader,
    config={},
    signal_params={"metric_weights": {"pe": 0.4, "pb": 0.3, "ev_ebitda": 0.3}},
)
# (dates × tickers) DataFrame, higher = more undervalued
```

## Next Steps

- [Signals Guide](../user-guide/signals.md)
- [Backtesting Guide](../user-guide/backtesting.md)
- [Portfolio & Risk](../user-guide/portfolio.md)
- [Examples](../examples/basic-backtest.md)
