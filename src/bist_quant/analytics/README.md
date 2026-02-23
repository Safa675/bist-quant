# `analytics/` — Performance Analytics & Metrics

## Purpose

Provides all quantitative performance analytics for portfolios and strategies. Split into four focused modules covering pure-Python primitives (for frontend parity), NumPy/pandas-based standalone functions, a professional OO analytics class, and an advanced analytics layer with GARCH, walk-forward, factor construction, and compliance tooling.

## Files

```
analytics/
├── core_metrics.py        # Pure-Python (no NumPy) analytics primitives
├── portfolio_metrics.py   # pandas/NumPy standalone functions + PortfolioAnalytics class
├── advanced.py            # GARCH, Kelly sizing, walk-forward, MPT, regime backtesting
└── professional.py        # Professional trading analytics (options, futures, compliance, tax)
```

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
- `PortfolioAnalytics.from_returns(returns)` — alias

**Rate coercion:** The `_coerce_rate_to_decimal()` helper automatically converts percent notation (e.g. `38.0`) to decimal (`0.38`) so callers can pass rates in either form.

---

### `advanced.py` — Advanced Quantitative Analytics

Built on `core_metrics` primitives (imports all math from there — **no duplicated math**).

**Key capabilities:**

| Function | Description |
|---|---|
| `build_garch_volatility_forecast(returns)` | EWMA/GARCH(1,1) with regime classification |
| `compute_kelly_fraction_percent(returns)` | Full Kelly criterion sizing |
| `compute_fixed_fractional_notional(...)` | Fixed-fractional position sizing |
| `compute_optimal_f(returns)` | Ralph Vince's optimal-f |
| `compute_correlation_adjusted_sizing(...)` | Penalize correlated concentration |
| `suggest_cross_asset_hedges(corr_matrix)` | Cross-asset hedge suggestions |
| `build_parameter_sensitivity_heatmap(...)` | MA fast/slow parameter grid search |
| `build_walk_forward_parameter_optimization(...)` | Walk-forward MA optimization |
| `build_portfolio_construction(...)` | MPT / risk-parity / min-var / ERC / factor-based construction |
| `run_regime_aware_backtest(...)` | MA strategy with regime filter |
| `test_strategy_significance(returns)` | t-stat + bootstrap p-value |
| `build_backtest_integration_diagnostics(...)` | Cost-adjusted + MC + regime + significance combined report |

---

### `professional.py` — Professional Trading & Compliance Analytics

Ported from TypeScript. No external scientific libraries — pure math.

**Key areas covered:**

| Area | Example functions |
|---|---|
| Trade planning | `build_crypto_trade_plan()`, `compute_forex_pip_value()`, `compute_option_greeks()` |
| Options | Black–Scholes Greeks, implied vol (internal implementation via `_normal_cdf`) |
| Futures | `compute_futures_margin()` |
| Fund screening | `screen_funds()` — multi-factor ETF/fund scoring |
| Risk monitoring | `evaluate_strategy_risk()`, `run_portfolio_stress_test()`, `monitor_liquidity_risk()`, `detect_concentration_risk()` |
| Portfolio optimization | `optimize_constrained_portfolio()` — randomized search with turnover constraint |
| Factor exposure | `build_factor_exposure_model()` — OLS regression on factor returns |
| Compliance | `check_compliance()`, `generate_alerts()`, `calculate_tax()` |

---

## Local Rules for Contributors

1. **`core_metrics.py` is dependency-free.** Do not introduce NumPy or pandas imports into that file. It must remain pure Python.
2. **Do not duplicate statistical primitives.** If you need `mean()`, `std_dev()`, or `quantile()` in a new function, import from `core_metrics` — do not reimplement.
3. **`list[SeriesPoint]` interface.** Functions in `core_metrics` must accept and return `list[SeriesPoint]`. Functions in `portfolio_metrics` and `advanced` use pandas `Series`/`DataFrame`.
4. **Risk-free rate.** Always use `get_default_risk_free_rate()` from `portfolio_metrics` as the default; it resolves the live Turkish policy rate with a module-level cache.
5. **252-day annualization.** Use consistently. Do not use 260 or 365.
6. **Deterministic PRNG.** Any Monte Carlo in `core_metrics` must use `_Xorshift32(seed)`. In `advanced.py`, use `_make_deterministic_noise()`.
