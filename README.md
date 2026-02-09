# BIST Quant Research Repository

This repository contains a Borsa Istanbul (BIST) research stack with two main parts:

- `Models/`: multi-factor portfolio engine and signals
- `Regime Filter/`: market regime detection pipeline (XGBoost + LSTM + HMM + ensemble)

Both modules are wired through shared market/fundamental data in `data/`.

## Repository Structure

```text
BIST/
├── Models/
│   ├── common/
│   ├── configs/
│   ├── signals/
│   ├── results/
│   └── portfolio_engine.py
├── Regime Filter/
│   ├── api/
│   ├── alerts/
│   ├── data/
│   ├── docs/
│   ├── models/
│   ├── outputs/
│   ├── run_full_pipeline.py
│   └── run_api.py
├── data/
│   ├── Fetcher-Scrapper/
│   ├── fundamental_data/
│   ├── bist_prices_full.parquet
│   ├── xu100_prices.csv
│   ├── usdtry_data.csv
│   └── xau_try_2013_2026.csv
├── cache/
└── .gitignore
```

## Setup

```bash
cd /home/safa/Documents/Markets/BIST
python3 -m venv .venv
source .venv/bin/activate
pip install -r "Regime Filter/requirements.txt"
pip install pyarrow openpyxl
```

## Run The Regime Pipeline

```bash
cd "Regime Filter"
python run_full_pipeline.py
```

Outputs are written to `Regime Filter/outputs/`, including:

- `all_features.csv`
- `regime_features.csv`
- `simplified_regimes.csv`
- `ensemble_model/` (trained models)
- `pipeline_summary.txt`

Optional API server:

```bash
cd "Regime Filter"
python run_api.py
```

## Run Factor Backtests

```bash
cd Models
python portfolio_engine.py momentum
python portfolio_engine.py value
python portfolio_engine.py all
```

Backtest outputs are written to `Models/results/<factor_name>/`.

## Results Snapshot

Snapshot below is from current local result artifacts (`Models/results/*/summary.txt`) ending on `2026-02-04`.

| Factor | Total Return | CAGR | Max Drawdown | Sharpe |
| --- | ---: | ---: | ---: | ---: |
| `breakout_value` | 137728.04% | 120.03% | -45.13% | 2.79 |
| `small_cap_momentum` | 97218.70% | 111.83% | -52.85% | 2.27 |
| `size` | 82944.88% | 108.20% | -41.91% | 2.50 |
| `donchian` | 637520.34% | 104.16% | -44.63% | 2.49 |
| `trend_value` | 38438.19% | 91.47% | -44.49% | 2.30 |
| `value` | 36647.22% | 90.48% | -44.29% | 2.29 |
| `quality_momentum` | 19714.57% | 78.07% | -52.72% | 1.92 |
| `macro_hedge` | 13452.60% | 70.84% | -34.41% | 2.23 |
| `momentum` | 4318.81% | 36.16% | -53.55% | 1.19 |
| `xu100` (benchmark) | 1958.43% | 28.63% | -33.15% | 1.09 |

For complete details, inspect:

- `Models/results/factor_correlation_matrix.csv`
- `Models/results/<factor_name>/summary.txt`
- `Models/results/<factor_name>/yearly_metrics.csv`

## Quick Code Example

```python
import pandas as pd

# Example: yearly metrics for value factor
yearly = pd.read_csv("Models/results/value/yearly_metrics.csv")
print(yearly.tail(5))

# Example: compare model vs XU100 vs XAU/TRY daily returns
returns = pd.read_csv("Models/results/value/returns.csv", parse_dates=["date"])
print(returns[["date", "Return", "XU100_Return", "XAU_TRY_Return"]].tail())
```
