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
engine = PortfolioEngine(loader=loader)
result = engine.run_factor("my_reversal")
print(result.metrics)
```

## Alternatively: Pass a Prebuilt Panel

```python
dates  = loader.build_close_panel().index
panel  = reversal_signal(dates, loader, config={})
result = engine.run_factor_from_panel(panel)
print(result.metrics)
```

## Subclass BaseSignal (Standalone Factor)

For more control (automatic lagging, schema validation, output persistence):

```python
from bist_quant.signals.standalone_factors.base_signal import BaseSignal

class MyReversal(BaseSignal):
    name = "my_reversal"
    description = "1-month reversal factor"

    def compute_raw_signal(
        self,
        dates: pd.DatetimeIndex,
        loader: DataLoader,
        config: dict,
        custom: dict | None = None,
    ) -> pd.DataFrame:
        lookback = (custom or {}).get("lookback_days", 21)
        close = loader.build_close_panel()
        return -close.pct_change(lookback).reindex(index=dates)

# Usage
factor = MyReversal(lag_days=1)   # base class handles lag + validation auto
signal = factor.build(dates=dates, loader=loader, config={})
```
