# Risk Management

Risk management is split between live controls during simulation (`RiskManager`) and post-hoc analytics (`analytics/` module).

## Live Risk Controls (during simulation)

### Volatility Targeting

Scales portfolio exposure so realized downside vol hits `target_downside_vol`:

```python
options["use_vol_targeting"] = True
options["target_downside_vol"] = 0.15   # 15% annualized downside vol
```

Leverage is capped at 1.0 (no shorting, no margin). Annualization uses **252** trading days.

### Stop-Loss

Per-position stop based on drawdown from entry price:

```python
options["use_stop_loss"] = True
options["stop_loss_threshold"] = 0.15   # exit if down 15% from entry
```

### Slippage

Tiered by market cap tier:

```python
options["use_slippage"] = True
options["slippage_bps"] = 5.0   # base rate — BIST minimum
```

### Position Limits

```python
options["max_position_weight"] = 0.20   # cap any single stock at 20%
options["liquidity_quantile"] = 0.20   # remove bottom 20% by volume
```

## Post-Hoc Analytics

`core_metrics.py` computes all performance/risk metrics. It has **no numpy or pandas dependency** — accepts `list[SeriesPoint]` (named tuple with `.date` and `.value`).

```python
from bist_quant.analytics.core_metrics import compute_metrics, to_series_points

points = to_series_points(result.returns)
summary = compute_metrics(points)

print(summary["cagr"])          # compound annual growth rate
print(summary["sharpe"])        # annualized Sharpe (252 days)
print(summary["sortino"])       # Sortino (downside only)
print(summary["max_drawdown"])  # max peak-to-trough drawdown
print(summary["calmar"])        # CAGR / |max_drawdown|
print(summary["var_95"])        # 5% historical VaR
print(summary["cvar_95"])       # 5% CVaR (expected shortfall)
```

## GARCH Volatility

```python
from bist_quant.analytics.garch_analyzer import GarchAnalyzer

analyzer = GarchAnalyzer()
vol_forecast = analyzer.fit_and_forecast(result.returns, horizon=10)
print(vol_forecast)   # 10-day forward vol forecast
```

## Kelly Sizing

```python
from bist_quant.analytics.kelly_analyzer import KellyAnalyzer

kelly = KellyAnalyzer()
full_kelly = kelly.compute(win_rate=0.55, avg_win=0.03, avg_loss=0.02)
half_kelly = full_kelly / 2
```

## Monte Carlo

```python
from bist_quant.analytics.monte_carlo_analyzer import MonteCarloAnalyzer

mc = MonteCarloAnalyzer()
results_df = mc.simulate(result.returns, n_simulations=1000, horizon=252)
print(results_df.describe())
```
