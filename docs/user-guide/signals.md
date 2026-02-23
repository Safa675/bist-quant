# Signals

BIST Quant ships 40+ signal builders covering momentum, value, quality, technical, and composite strategies.

## Discover Available Signals

```python
from bist_quant.signals.factory import get_available_signals

print(get_available_signals())
# ['momentum', 'value', 'quality', 'composite', 'reversal', 'low_vol', ...]
```

## Signal Contract

Every signal implements the `SignalBuilder` Protocol:

```python
from typing import Protocol
import pandas as pd
from bist_quant import DataLoader

class SignalBuilder(Protocol):
    def __call__(
        self,
        dates: pd.DatetimeIndex,
        loader: DataLoader,
        config: dict,
        signal_params: dict | None = None,
    ) -> pd.DataFrame: ...
```

**Return value rules:**
- Shape: `(dates × tickers)`, dtype `float64`
- Higher score = more attractive (system ranks descending)
- `NaN` = exclude that ticker on that date (not 0.0)
- DatetimeIndex must be timezone-naive, floored to midnight
- Validated by `validate_signal_panel_schema()` before output

## Build a Signal

```python
from bist_quant import DataLoader, DataPaths
from bist_quant.signals.factory import build_signal

loader = DataLoader(data_paths=DataPaths())
dates  = loader.build_close_panel().index

signal_df = build_signal(
    name="momentum",
    dates=dates,
    loader=loader,
    config={},
    signal_params={"skip_recent_days": 21},   # skip last month
)
print(signal_df.shape)   # (dates, tickers)
```

## Signal Categories

### Momentum
| Name | Description |
|---|---|
| `momentum` | 12-1 month total return momentum |
| `reversal` | Short-term 1-month reversal |
| `52w_high` | Proximity to 52-week high |
| `trend_following` | Moving-average crossover score |
| `residual_momentum` | Market-beta-adjusted momentum |

### Value
| Name | Description |
|---|---|
| `value` | P/E, P/B, EV/EBITDA composite |
| `earnings_yield` | Inverse P/E (E/P) |
| `book_to_price` | B/P (inverse P/B) |
| `ev_to_ebitda` | Enterprise value efficiency |

### Quality
| Name | Description |
|---|---|
| `quality` | ROE, ROA, gross margin composite |
| `roe_quality` | Return on equity |
| `profitability` | Operating leverage and margins |
| `earnings_stability` | Low earnings variance |

### Technical
| Name | Description |
|---|---|
| `rsi_factor` | RSI-based mean reversion |
| `volume_momentum` | Price × volume breakout |
| `low_volatility` | Realized vol inverse (Ang et al.) |
| `high_beta` | CAPM beta (inverse of low_vol) |

### Fundamentals-Based
| Name | Description |
|---|---|
| `fundamental_momentum` | Earnings revision factor |
| `accruals` | Operating accruals (Sloan) |
| `asset_growth` | Asset growth anomaly |
| `net_payout` | Buyback + dividend yield |

### Composite / Multi-Factor
| Name | Description |
|---|---|
| `composite` | Equal-weight blend of all factors |
| `qmj` | Quality Minus Junk (AQR style) |
| `ahp_composite` | AHP-weighted multi-factor |

## Using Signals in Backtests

```python
from bist_quant import PortfolioEngine

engine = PortfolioEngine(loader=loader)

# By registered name (runs full factor_builders.py pipeline)
result = engine.run_factor("momentum")

# Custom prebuilt panel
result = engine.run_factor_from_panel(signal_df, portfolio_options=options)
```

## Standalone Factor ABC

For standalone factor research, subclass `BaseSignal`:

```python
from bist_quant.signals.standalone_factors.base_signal import BaseSignal

class MyFactor(BaseSignal):
    name = "my_factor"
    description = "Custom factor"

    def compute_raw_signal(
        self,
        dates: pd.DatetimeIndex,
        loader: DataLoader,
        config: dict,
        custom: dict | None = None,
    ) -> pd.DataFrame:
        close = loader.build_close_panel()
        # ... return (dates x tickers) DataFrame, higher = better
```

The base class handles lagging (`lag_days=1` default), NaN fill, and schema validation.
