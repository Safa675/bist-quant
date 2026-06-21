# `analytics/` — Performance Analytics & Metrics

## Purpose

Provides all quantitative performance analytics for portfolios and strategies. Split into four focused modules covering pure-Python primitives (for frontend parity), NumPy/pandas-based standalone functions, a professional OO analytics class, and an advanced analytics layer with GARCH, walk-forward, factor construction, and compliance tooling.

## Files

```
analytics/
├── _shared.py               # Shared helpers (clamp, rounding, RNG noise, EMA, type aliases)
├── core_metrics.py          # Pure-Python (no NumPy) analytics primitives — frontend parity layer
├── portfolio_metrics.py     # pandas/NumPy standalone functions + PortfolioAnalytics class
├── advanced.py              # Thin re-export facade composing the split sub-modules below
├── professional/            # Professional trading analytics package (see sub-modules below)
│   ├── trading.py           # Crypto/forex/options/futures sizing, fund screening, spreads
│   ├── risk.py              # Strategy risk, stress tests, factor exposure, optimization
│   ├── reporting.py         # Sentiment, benchmarks, attribution, tax, market intelligence
│   ├── execution.py         # Iceberg/TWAP/VWAP, bracket orders, slippage simulation
│   └── compliance.py        # Compliance rules, position limits, alerting, escalation
├── volatility.py            # GARCH / EWMA volatility forecasting + proxy-asset construction
├── position_sizing.py       # Kelly, fixed-fractional, optimal-f, correlation-adjusted sizing
├── portfolio_construction.py # MPT / risk-parity / min-var / ERC / factor-based construction
├── parameter_optimization.py # Walk-forward + parameter-sensitivity heatmaps
├── regime_backtest.py       # MA strategy with regime filter backtesting
├── risk_metrics.py          # Strategy significance testing (t-stat + bootstrap p-value)
├── attribution.py           # Portfolio return attribution + risk decomposition
├── charting.py              # Volume profile, Renko, Point & Figure chart primitives
└── signals.py               # Indicator-based signal helpers
```

`advanced.py` re-exports the public API of `volatility`, `position_sizing`,
`portfolio_construction`, `parameter_optimization`, `regime_backtest`,
`risk_metrics`, `attribution`, `charting`, and `signals` — it is now a
composition facade, not the implementation home of those features.

---

### `core_metrics.py` — Pure-Python Analytics Engine

**No external dependencies.** All math is implemented from scratch for portability and deterministic behavior (matches a TypeScript twin).

**Key dataclasses (all are `@dataclass`):**

| Class | Description |
|---|---|
| `SeriesPoint` | Single `(date, value)` observation |
| `PerformanceMetrics` | Full metric bundle: CAGR, Sharpe, Sortino, VaR, CVaR, Calmar, Omega, max drawdown |
| `RollingMetricPoint` | Rolling 63-day metrics at a single date |
| `WalkForwardSplit` | One expanding-window OOS split result  |
| `MonteCarloPathPoint` / `MonteCarloSummary` | Bootstrap Monte Carlo fan output |
| `AllocationOptimizationResult` | Mean-variance optimization output |
| `RiskContributionResult` | Per-asset risk contribution |

**Key functions:**

| Function | Description |
|---|---|
| `compute_performance_metrics(returns, benchmark?)` | Primary full metrics computation |
| `build_rolling_metrics(equity_curve)` | Rolling 63-day / 126-day Sharpe and drawdown |
| `run_monte_carlo_bootstrap(returns, n_paths)` | Percentile bootstrap fan |
| `build_walk_forward_analysis(returns, splits)` | Expanding-window walk-forward |
| `optimize_mean_variance_allocation(series_list)` | Efficient frontier search |
| `compute_risk_contribution(weights, cov)` | Marginal risk contribution |
| `run_stress_scenarios(returns)` | Pre-defined stress scenarios |
| `curve_to_returns(equity)` / `returns_to_equity(returns)` | Series conversion |

**Critical conventions:**
- All functions accept/return `list[SeriesPoint]` — **not** DataFrames, for frontend compatibility.
- 252 trading-day annualization is used throughout.
- Bessel-corrected sample standard deviation everywhere.
- PRNG is `_Xorshift32` (seeded deterministic, matches JS implementation exactly).

---

### `portfolio_metrics.py` — Standalone Metric Functions + OO Class

Pandas/NumPy-based layer with richer API. Preferred for backend-only code paths.

**Standalone functions:**

| Function | Description |
|---|---|
| `calculate_sharpe_ratio(returns, risk_free_rate?)` | Annualized Sharpe |
| `calculate_sortino_ratio(returns, risk_free_rate?)` | Downside-deviation Sharpe |
| `calculate_max_drawdown(returns)` | Max peak-to-trough drawdown |
| `calculate_beta(returns, benchmark_returns)` | OLS market beta |
| `calculate_alpha(returns, benchmark_returns, rfr?)` | Jensen's alpha |
| `calculate_calmar_ratio(returns)` | CAGR / max drawdown |
| `calculate_information_ratio(returns, benchmark)` | Active return / tracking error |
| `calculate_var(returns, confidence?)` | Value-at-Risk |
| `calculate_cvar(returns, confidence?)` | Conditional VaR (expected shortfall) |
| `calculate_rolling_metrics(returns, window?)` | Rolling metric DataFrame |
| `get_default_risk_free_rate()` | Live Turkish policy rate via `FixedIncomeProvider` |

**OO class:**

`PortfolioAnalytics` wraps all the above with lazy metric caching. Constructors:
- `PortfolioAnalytics(returns)` — direct returns Series
- `PortfolioAnalytics.from_holdings(holdings, prices)` — compute returns from holdings
- `PortfolioAnalytics.from_equity_curve(equity)` — compute returns from an equity curve

**Rate coercion:** The `_coerce_rate_to_decimal()` helper automatically converts percent notation (e.g. `38.0`) to decimal (`0.38`) so callers can pass rates in either form.

---

### `advanced.py` — Advanced Quantitative Analytics

Thin composition facade. Re-exports the implementations living in the split
sub-modules (`volatility`, `position_sizing`, `portfolio_construction`,
`parameter_optimization`, `regime_backtest`, `risk_metrics`, `attribution`,
`charting`, `signals`) and adds one new composite:

| Function | Description |
|---|---|
| `build_backtest_integration_diagnostics(...)` | Cost-adjusted + MC + regime + significance combined report |

---

### `professional/` — Professional Trading & Compliance Analytics

Thin composition facade (`professional/__init__.py`) re-exporting five domain
sub-modules. Ported from TypeScript. No external scientific libraries — pure
math. Shared helpers `_compare()` and `_parse_date()` live in `_shared.py`.

| Sub-module | Example functions |
|---|---|
| `trading` | `build_crypto_trade_plan()`, `compute_forex_pip_value()`, `compute_option_greeks()`, `screen_funds()` |
| `risk` | `evaluate_strategy_risk()`, `run_portfolio_stress_test()`, `optimize_constrained_portfolio()`, `build_factor_exposure_model()` |
| `reporting` | `analyze_sentiment()`, `compare_benchmarks()`, `build_tax_report()`, `build_market_intelligence_snapshot()` |
| `execution` | `create_iceberg_slices()`, `build_twap_schedule()`, `build_vwap_schedule()`, `simulate_execution_with_slippage()` |
| `compliance` | `run_compliance_rule_engine()`, `monitor_position_limits()`, `evaluate_alert_conditions()`, `build_escalation_plan()` |

`run_performance_snapshot()` stays in `professional/__init__.py` because it
composes reporting + core metrics (same pattern as `advanced.py`).

---

## Local Rules for Contributors

1. **`core_metrics.py` is dependency-free.** Do not introduce NumPy or pandas imports into that file. It must remain pure Python.
2. **Do not duplicate statistical primitives.** If you need `mean()`, `std_dev()`, or `quantile()` in a new function, import from `core_metrics` — do not reimplement.
3. **`list[SeriesPoint]` interface.** Functions in `core_metrics` must accept and return `list[SeriesPoint]`. Functions in `portfolio_metrics` and `advanced` use pandas `Series`/`DataFrame`.
4. **Risk-free rate.** Always use `get_default_risk_free_rate()` from `portfolio_metrics` as the default; it resolves the live Turkish policy rate with a module-level cache.
5. **252-day annualization.** Use consistently. Do not use 260 or 365.
6. **Deterministic PRNG.** Any Monte Carlo in `core_metrics` must use `_Xorshift32(seed)`. In `advanced.py`, use `_make_deterministic_noise()`.
