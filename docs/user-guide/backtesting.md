# Backtesting

The backtester runs an **event-loop simulation** day-by-day, rebalancing at `monthly` or `quarterly` boundaries.

## Core Components

| Module | Purpose |
|---|---|
| `PortfolioEngine` | Main entry point |
| `Backtester` | Event-loop state machine |
| `RiskManager` | Vol-targeting, stop-loss, slippage |
| `RegimeClassifier` | 2D market regime filter |
| `backtest_services.py` | Service-layer wrappers |

## Basic Example

```python
from bist_quant import PortfolioEngine, DataLoader, DataPaths

loader = DataLoader(data_paths=DataPaths())
engine = PortfolioEngine(data_loader=loader, options={"use_regime_filter": False})

result = engine.run_factor("momentum")
print({k: result[k] for k in ("cagr", "sharpe", "max_drawdown")})
```

## What the Event Loop Does

On each rebalance day:
1. **Check regime** (if `use_regime_filter=True`) — Bull/Recovery = stocks; Bear/Stress = gold/cash (XAU/TRY)
2. **Compute signal** for the current date using `signal_lag_days=1` offset (anti-lookahead)
3. **Apply liquidity filter** — remove tickers below `liquidity_quantile` by 30-day volume
4. **Rank and select** top-N tickers by signal score
5. **Size positions** — equal-weight, inverse-vol, or vol-targeted
6. **Apply slippage** — size-aware, tiered by market cap bucket
7. **Apply stop-loss** — per-position drawdown check if enabled
8. **Record** returns, holdings, turnover, regime

## Result Payload

```python
result["cagr"]               # float
result["sharpe"]             # float
result["returns"]            # pd.Series: daily portfolio returns
result["holdings_history"]   # pd.DataFrame: holdings history
result["equity"]             # pd.Series: cumulative NAV from 1.0
```

## Regime Filter

The regime classifier uses a 2D (MA-trend × realized-vol) grid.
**MA window = 50** (tuned for BIST) — not the classic 200.

```
Bull (trend up, vol low)      → 100% stocks
Recovery (trend up, vol high) → 100% stocks
Bear (trend down, vol low)    → 0% stocks → gold (XAU/TRY)
Stress (trend down, vol high) → 0% stocks → gold (XAU/TRY)
```

## Slippage Model

Slippage is tiered by market cap bucket:
- Large cap (BIST-30):   `slippage_bps` × 1.0
- Mid cap  (BIST-100):   `slippage_bps` × 1.5
- Small cap (rest):      `slippage_bps` × 2.5

Minimum recommended: `slippage_bps=5.0` for BIST.

## Walk-Forward Analysis

```python
from bist_quant.engines.analytics_engine import AnalyticsEngine

aengine = AnalyticsEngine(loader=loader)
wf = aengine.walk_forward_analysis(
    strategy="momentum",
    n_splits=5,
    train_pct=0.7,
)
print(wf)
```

## Anti-Lookahead Rules

- `signal_lag_days=1` is mandatory — signals computed on day T are first traded on day T+1
- Turkish fundamental reporting lags are baked in: **Q4 = 70 days**, **Q1-Q3 = 40 days**
- Never join closing prices and signals on the same row without lagging
