# Models Module

`Models/` is the portfolio research engine for BIST multi-factor strategies.

It loads data once, builds factor signals, applies portfolio/risk logic, runs backtests, and saves full diagnostics per factor.

## Folder Structure

```text
Models/
├── common/
│   ├── data_loader.py
│   └── utils.py
├── configs/
│   ├── momentum.py
│   ├── value.py
│   ├── profitability.py
│   ├── small_cap.py
│   ├── ... (19 factor configs)
│   └── xu100.py
├── signals/
│   ├── momentum_signals.py
│   ├── value_signals.py
│   ├── profitability_signals.py
│   ├── ... (one signal builder per factor)
│   └── xu100_signals.py
├── results/
│   ├── <factor_name>/
│   │   ├── equity_curve.csv
│   │   ├── returns.csv
│   │   ├── yearly_metrics.csv
│   │   ├── holdings.csv
│   │   ├── holdings_matrix.csv
│   │   ├── regime_performance.csv
│   │   └── summary.txt
│   ├── factor_correlation_matrix.csv
│   └── YEARLY RESULTS Year Model.txt
└── portfolio_engine.py
```

## Implemented Factors

Current factor configs in `Models/configs/`:

- `momentum`
- `value`
- `profitability`
- `investment`
- `small_cap`
- `small_cap_momentum`
- `sma`
- `donchian`
- `trend_value`
- `breakout_value`
- `quality_momentum`
- `quality_value`
- `size_rotation`
- `size_rotation_momentum`
- `size_rotation_quality`
- `currency_rotation`
- `dividend_rotation`
- `macro_hedge`
- `five_factor_rotation` (12-axis MWU-weighted multi-factor — see `signals/FIVE_FACTOR_ROTATION.md`)
- `xu100` (benchmark)

Legacy note:

- `Models/results/size/` exists from earlier runs. Current active config is `small_cap`.

## Engine Flow

1. `common/data_loader.py` loads/caches price, volume, fundamentals, regime predictions, XU100, and XAU/TRY.
2. `portfolio_engine.py` dynamically loads all configs from `Models/configs/`.
3. A factor-specific signal builder from `Models/signals/` creates daily cross-sectional scores.
4. Backtest logic applies:
- rebalance schedule (monthly or quarterly)
- liquidity filter
- optional inverse downside-vol weighting
- optional stop-loss and slippage
- optional regime allocation (blend/shift into XAU/TRY)
5. Results and metrics are written to `Models/results/<factor_name>/`.

## Run Commands

From repository root:

```bash
cd Models
python portfolio_engine.py --help
python portfolio_engine.py momentum
python portfolio_engine.py value
python portfolio_engine.py all
```

With custom global date bounds:

```bash
cd Models
python portfolio_engine.py momentum --start-date 2020-01-01 --end-date 2026-02-04
```

Note:

- Config-level timelines in `Models/configs/*.py` can override/default the period per factor.
- Typical ranges in current outputs are `2014-01-01` to `2026-02-04` or `2017-01-02` to `2026-02-04`.

## Config Example

Each factor config follows a structure like:

```python
SIGNAL_CONFIG = {
    "name": "momentum",
    "enabled": True,
    "rebalance_frequency": "monthly",  # or quarterly
    "timeline": {"start_date": "2014-01-01", "end_date": "2026-12-31"},
    "description": "12-1 momentum with downside volatility adjustment",
    "portfolio_options": {
        "use_regime_filter": True,
        "use_vol_targeting": False,
        "use_inverse_vol_sizing": True,
        "use_stop_loss": False,
        "use_liquidity_filter": True,
        "use_slippage": True,
        "slippage_bps": 5.0,
        "top_n": 20,
    },
}
```

## Results Artifacts

For each factor directory under `Models/results/<factor_name>/`:

- `returns.csv`: daily strategy return + benchmark returns (`XU100_Return`, `XAU_TRY_Return`)
- `equity_curve.csv`: cumulative equity series
- `yearly_metrics.csv`: annual return/sharpe/sortino + excess returns vs XU100 and XAU/TRY
- `holdings.csv`: daily holdings and weights
- `holdings_matrix.csv`: date x ticker weight matrix
- `regime_performance.csv`: performance breakdown by regime
- `summary.txt`: headline totals (CAGR, Sharpe, max drawdown, trade counts)

Cross-factor file:

- `Models/results/factor_correlation_matrix.csv`: return correlation matrix for all factors and benchmarks.

## Current Performance Snapshot

Based on current `summary.txt` files (latest dates in outputs end on `2026-02-04`):

| Factor | Total Return | CAGR | Max Drawdown | Sharpe |
| --- | ---: | ---: | ---: | ---: |
| `breakout_value` | 137728.04% | 120.03% | -45.13% | 2.79 |
| `small_cap_momentum` | 97218.70% | 111.83% | -52.85% | 2.27 |
| `size` | 82944.88% | 108.20% | -41.91% | 2.50 |
| `donchian` | 637520.34% | 104.16% | -44.63% | 2.49 |
| `value` | 36647.22% | 90.48% | -44.29% | 2.29 |
| `trend_value` | 38438.19% | 91.47% | -44.49% | 2.30 |
| `quality_momentum` | 19714.57% | 78.07% | -52.72% | 1.92 |
| `macro_hedge` | 13452.60% | 70.84% | -34.41% | 2.23 |
| `momentum` | 4318.81% | 36.16% | -53.55% | 1.19 |
| `xu100` | 1958.43% | 28.63% | -33.15% | 1.09 |

## Quick Analysis Code

```python
from pathlib import Path
import pandas as pd

base = Path("Models/results")

# Compare yearly metrics for two factors
value = pd.read_csv(base / "value" / "yearly_metrics.csv")
momentum = pd.read_csv(base / "momentum" / "yearly_metrics.csv")
print(value[["Year", "Return", "Excess_vs_XU100"]].tail())
print(momentum[["Year", "Return", "Excess_vs_XU100"]].tail())

# Inspect cross-factor correlation
corr = pd.read_csv(base / "factor_correlation_matrix.csv", index_col=0)
print(corr["XU100"].sort_values(ascending=False).head(10))
```

## Data Dependencies

The engine expects these main inputs under repository `data/`:

- `bist_prices_full.parquet` (or CSV fallback)
- `fundamental_data_consolidated.parquet` (or `fundamental_data/*.xlsx` fallback)
- `xu100_prices.csv`
- `xau_try_2013_2026.csv`
- `usdtry_data.csv` (for currency-aware signals)

And regime model artifacts under:

- `Regime Filter/outputs/ensemble_model/`
