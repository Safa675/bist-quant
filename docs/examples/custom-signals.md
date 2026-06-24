# Custom Signal Example

This example shows how to implement a custom signal using the `SignalBuilder` Protocol,
then run it through the backtester.

## Implement a Custom Signal

```python
from __future__ import annotations

import pandas as pd
from bist_quant import DataLoader
from bist_quant.common.validators import validate_signal_panel_schema

def reversal_signal(
    dates: pd.DatetimeIndex,
    loader: DataLoader,
    config: dict,
    signal_params: dict | None = None,
) -> pd.DataFrame:
    """Short-term reversal: stocks with the worst 1-month return are scored highest."""
    params = signal_params or {}
    lookback = params.get("lookback_days", 21)

    close = loader.build_close_panel()
    ret_1m = close.pct_change(lookback)

    # Negate: lower recent return = higher score (more attractive)
    signal = -ret_1m
    signal = signal.reindex(index=dates)

    # Mandatory: validate before returning
    validate_signal_panel_schema(signal)
    return signal
```

## Register and Run

```python
from bist_quant import PortfolioEngine, DataLoader, DataPaths
from bist_quant.signals.factory import SignalFactory

# Register under a name
SignalFactory.register("my_reversal", reversal_signal)

loader = DataLoader(data_paths=DataPaths())
engine = PortfolioEngine(data_loader=loader, options={"use_regime_filter": False})
result = engine.run_factor("my_reversal")
print({k: result[k] for k in ("cagr", "sharpe", "max_drawdown")})
```

## Alternatively: Pass a Prebuilt Panel

```python
dates  = loader.build_close_panel().index
panel  = reversal_signal(dates, loader, config={})
result = engine.run_factor_from_panel(panel)
print({k: result[k] for k in ("cagr", "sharpe", "max_drawdown")})
```

## Subclass FactorSignal (Standalone Factor)

For more control (normalization, winsorization, selection helpers):

```python
from bist_quant.signals.standalone_factors import FactorData, FactorParams, FactorSignal

class MyReversal(FactorSignal):
    @property
    def name(self) -> str:
        return "my_reversal"

    @property
    def description(self) -> str:
        return "1-month reversal factor"

    def compute_raw_signal(self, data: FactorData, params: FactorParams):
        lookback = params.custom.get("lookback_days", 21)
        raw = -data.close.pct_change(lookback)
        return raw.reindex(index=data.dates), {}

# Usage
factor = MyReversal()
signal = factor.compute_signal(
    FactorData(close=close, dates=dates, tickers=close.columns),
    FactorParams(custom={"lookback_days": 21}),
).scores
```
