# Custom Signals Example

You can pass a custom signal panel directly to `run_backtest`.

```python
import pandas as pd
from bist_quant import PortfolioEngine

# custom_signal: Date x Ticker DataFrame
custom_signal = pd.DataFrame(...)

engine = PortfolioEngine(options={"top_n": 15})
result = engine.run_backtest(
    signals=custom_signal,
    start_date="2023-01-01",
    end_date="2023-12-31",
)
```

Notebook version: [examples/02_custom_signals.ipynb](https://github.com/Safa675/BIST/blob/main/examples/02_custom_signals.ipynb)
