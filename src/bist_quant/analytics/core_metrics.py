"""
Core quantitative analytics metrics.

Ported from bist_quant.ai/src/lib/analytics/quant.ts.
Provides performance metrics, rolling analytics, Monte Carlo simulation,
walk-forward analysis, mean-variance optimization, risk contribution,
correlation analysis, stress scenarios, and transaction cost modeling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class SeriesPoint:
    """A single date-value observation."""

    date: str
    value: float


@dataclass
class PerformanceMetrics:
    """Core performance metrics for a return series."""

    observations: int = 0
    total_return: float = 0.0
    cagr: float = 0.0
    annualized_volatility: float = 0.0
    downside_deviation: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float | None = None
    var_95: float = 0.0
    cvar_95: float = 0.0
    beta: float | None = None
    alpha_annual: float | None = None


@dataclass
class RollingMetricPoint:
    """Rolling window metrics at a single date."""

    date: str
    rolling_return_63d: float | None = None
    rolling_sharpe_63d: float | None = None
    rolling_sortino_63d: float | None = None
    rolling_volatility_63d: float | None = None
    rolling_drawdown_126d: float | None = None


@dataclass
class WalkForwardSplit:
    """Results from one walk-forward split."""

    split: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_return: float
    test_return: float
    train_cagr: float
    test_cagr: float
    train_sharpe: float
    test_sharpe: float
    p_value: float | None = None


@dataclass
class MonteCarloPathPoint:
    """Percentile fan at one day horizon."""

    day: int
    p05: float
    p25: float
    p50: float
    p75: float
    p95: float


@dataclass
class MonteCarloSummary:
    """Summary of a Monte Carlo bootstrap simulation."""

    iterations: int = 0
    horizon_days: int = 0
    terminal_p05: float = 1.0
    terminal_p25: float = 1.0
    terminal_p50: float = 1.0
    terminal_p75: float = 1.0
    terminal_p95: float = 1.0
    expected_terminal: float = 1.0
    expected_cagr: float = 0.0
    probability_of_loss: float = 0.0
    max_terminal: float = 1.0
    min_terminal: float = 1.0
    paths: list[MonteCarloPathPoint] = field(default_factory=list)


@dataclass
class StressScenarioResult:
    """Result of a single stress scenario."""

    name: str
    shock_pct: float
    estimated_portfolio_impact_pct: float
    expected_shortfall_pct: float


@dataclass
class AllocationPoint:
    """One point on the efficient frontier."""

    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe: float


@dataclass
class AllocationOptimizationResult:
    """Result of mean-variance portfolio optimization."""

    assets: list[str]
    best_weights: dict[str, float]
    best_expected_return: float
    best_volatility: float
    best_sharpe: float
    frontier: list[AllocationPoint] = field(default_factory=list)


@dataclass
class RiskContributionResult:
    """Risk contribution analysis for portfolio assets."""

    contribution_pct: dict[str, float]
    marginal_contribution: dict[str, float]


@dataclass
class TransactionCostResult:
    """Result of applying transaction costs to a return series."""

    adjusted: list[SeriesPoint]
    total_cost_impact_pct: float
    per_rebalance_cost_pct: float
    rebalance_events: int


@dataclass
class DiversificationScore:
    """Diversification metrics from a correlation matrix."""

    average_pairwise_correlation: float
    diversification_ratio: float


# ---------------------------------------------------------------------------
# Math Utilities
# ---------------------------------------------------------------------------


def _safe_date(date: str) -> str:
    """Normalise a date string to YYYY-MM-DD where possible."""
    # Accept ISO-8601 or YYYY-MM-DD directly
    return date[:10] if len(date) >= 10 else date


def _sort_points(points: list[SeriesPoint]) -> list[SeriesPoint]:
    """Filter non-finite values and sort by date."""
    cleaned = [
        SeriesPoint(date=_safe_date(p.date), value=p.value)
        for p in points
        if math.isfinite(p.value)
    ]
    cleaned.sort(key=lambda p: p.date)
    return cleaned


def mean(values: list[float]) -> float:
    """Arithmetic mean."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def sample_std_dev(values: list[float]) -> float:
    """Sample standard deviation (Bessel-corrected)."""
    if len(values) < 2:
        return 0.0
    m = mean(values)
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(max(variance, 0.0))


def covariance(a: list[float], b: list[float]) -> float:
    """Sample covariance between two equal-length arrays."""
    if len(a) < 2 or len(b) < 2 or len(a) != len(b):
        return 0.0
    mean_a = mean(a)
    mean_b = mean(b)
    cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(len(a)))
    return cov / (len(a) - 1)


def correlation(a: list[float], b: list[float]) -> float:
    """Pearson correlation coefficient."""
    std_a = sample_std_dev(a)
    std_b = sample_std_dev(b)
    if std_a <= 0 or std_b <= 0:
        return 0.0
    return covariance(a, b) / (std_a * std_b)


def quantile(values: list[float], q: float) -> float:
    """Linear-interpolation quantile (matches JS behaviour)."""
    if not values:
        return 0.0
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def normal_cdf(x: float) -> float:
    """Cumulative distribution function of the standard normal."""
    sign = -1 if x < 0 else 1
    abs_x = abs(x) / math.sqrt(2)
    t = 1.0 / (1.0 + 0.3275911 * abs_x)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    erf = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * math.exp(-abs_x * abs_x))
    return 0.5 * (1.0 + sign * erf)


def _annualize_return(total_return_fraction: float, observations: int) -> float:
    """CAGR from total return fraction and observation count."""
    if observations <= 0:
        return 0.0
    if total_return_fraction <= -1:
        return -100.0
    return (math.pow(1 + total_return_fraction, 252 / observations) - 1) * 100


# ---------------------------------------------------------------------------
# Seeded RNG (xorshift32, matches JS implementation)
# ---------------------------------------------------------------------------


class _Xorshift32:
    """Deterministic PRNG matching the TypeScript xorshift implementation."""

    def __init__(self, seed: int = 42) -> None:
        self._state = seed & 0xFFFFFFFF
        if self._state == 0:
            self._state = 1

    def next(self) -> float:
        self._state ^= (self._state << 13) & 0xFFFFFFFF
        self._state ^= (self._state >> 17) & 0xFFFFFFFF
        self._state ^= (self._state << 5) & 0xFFFFFFFF
        self._state = self._state & 0xFFFFFFFF
        return (self._state % 1_000_000) / 1_000_000


# ---------------------------------------------------------------------------
# Core Series Functions
# ---------------------------------------------------------------------------


def curve_to_returns(curve: list[SeriesPoint]) -> list[SeriesPoint]:
    """Convert an equity curve to a daily return series."""
    ordered = _sort_points(curve)
    if len(ordered) < 2:
        return []
    output: list[SeriesPoint] = []
    for i in range(1, len(ordered)):
        prev = ordered[i - 1].value
        curr = ordered[i].value
        if not math.isfinite(prev) or not math.isfinite(curr) or prev <= 0:
            continue
        output.append(SeriesPoint(date=ordered[i].date, value=curr / prev - 1))
    return output


def returns_to_equity(
    returns: list[SeriesPoint],
    start_value: float = 1.0,
) -> list[SeriesPoint]:
    """Compound a return series into an equity curve."""
    running = start_value
    output: list[SeriesPoint] = []
    for row in _sort_points(returns):
        running *= 1 + row.value
        output.append(SeriesPoint(date=row.date, value=running))
    return output


def compute_max_drawdown(curve: list[SeriesPoint]) -> float:
    """Maximum drawdown of an equity curve (returned as negative %)."""
    ordered = _sort_points(curve)
    if not ordered:
        return 0.0
    peak = ordered[0].value
    min_drawdown = 0.0
    for point in ordered:
        peak = max(peak, point.value)
        if peak <= 0:
            continue
        drawdown = point.value / peak - 1
        min_drawdown = min(min_drawdown, drawdown)
    return min_drawdown * 100


# ---------------------------------------------------------------------------
# Series Alignment
# ---------------------------------------------------------------------------


def align_two_series(
    left: list[SeriesPoint],
    right: list[SeriesPoint],
) -> dict[str, Any]:
    """Align two series to common dates. Returns {dates, left, right}."""
    right_map: dict[str, float] = {}
    for row in _sort_points(right):
        right_map[row.date] = row.value

    dates: list[str] = []
    left_values: list[float] = []
    right_values: list[float] = []
    for row in _sort_points(left):
        match = right_map.get(row.date)
        if match is None:
            continue
        dates.append(row.date)
        left_values.append(row.value)
        right_values.append(match)
    return {"dates": dates, "left": left_values, "right": right_values}


def align_series_by_date(
    series_map: dict[str, list[SeriesPoint]],
) -> dict[str, Any]:
    """Align multiple named series to common dates. Returns {dates, values}."""
    keys = list(series_map.keys())
    if not keys:
        return {"dates": [], "values": {}}

    date_sets = [
        set(p.date for p in _sort_points(series_map[key])) for key in keys
    ]
    common_dates = sorted(d for d in date_sets[0] if all(d in s for s in date_sets))

    values: dict[str, list[float]] = {}
    for key in keys:
        date_to_value: dict[str, float] = {}
        for row in _sort_points(series_map[key]):
            date_to_value[row.date] = row.value
        values[key] = [date_to_value.get(d, 0.0) for d in common_dates]

    return {"dates": common_dates, "values": values}


# ---------------------------------------------------------------------------
# Performance Metrics
# ---------------------------------------------------------------------------


def compute_performance_metrics(
    returns: list[SeriesPoint],
    equity_curve: list[SeriesPoint] | None = None,
    benchmark_returns: list[SeriesPoint] | None = None,
) -> PerformanceMetrics:
    """Compute a comprehensive set of performance metrics from a return series."""
    ordered_returns = _sort_points(returns)
    values = [r.value for r in ordered_returns]
    observations = len(values)

    if not observations:
        return PerformanceMetrics()

    # Build or use equity curve
    curve = (
        _sort_points(equity_curve)
        if equity_curve and len(equity_curve) > 0
        else returns_to_equity(ordered_returns, 1.0)
    )

    if len(curve) > 1:
        total_return_fraction = curve[-1].value / curve[0].value - 1
    else:
        total_return_fraction = 1.0
        for v in values:
            total_return_fraction *= 1 + v
        total_return_fraction -= 1

    avg = mean(values)
    std = sample_std_dev(values)
    downside = [v for v in values if v < 0]
    downside_std = sample_std_dev(downside)

    annualized_vol = std * math.sqrt(252)
    downside_deviation = downside_std * math.sqrt(252)
    sharpe = (avg / std) * math.sqrt(252) if std > 0 else 0.0
    sortino = (avg / downside_std) * math.sqrt(252) if downside_std > 0 else 0.0

    max_dd = compute_max_drawdown(curve)
    cagr = _annualize_return(total_return_fraction, observations)
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0
    win_rate = (sum(1 for v in values if v > 0) / observations) * 100

    positive_sum = sum(v for v in values if v > 0)
    negative_sum_abs = abs(sum(v for v in values if v < 0))
    profit_factor = positive_sum / negative_sum_abs if negative_sum_abs > 0 else None

    var_95 = quantile(values, 0.05)
    cvar_set = [v for v in values if v <= var_95]
    cvar_95 = mean(cvar_set) if cvar_set else var_95

    beta: float | None = None
    alpha_annual: float | None = None
    if benchmark_returns and len(benchmark_returns) > 0:
        aligned = align_two_series(ordered_returns, benchmark_returns)
        if len(aligned["left"]) >= 2:
            bench_var = covariance(aligned["right"], aligned["right"])
            if bench_var > 0:
                beta = covariance(aligned["left"], aligned["right"]) / bench_var
            alpha_daily = mean(aligned["left"]) - (beta or 0) * mean(aligned["right"])
            alpha_annual = alpha_daily * 252 * 100

    return PerformanceMetrics(
        observations=observations,
        total_return=total_return_fraction * 100,
        cagr=cagr,
        annualized_volatility=annualized_vol * 100,
        downside_deviation=downside_deviation * 100,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=max_dd,
        win_rate=win_rate,
        profit_factor=profit_factor,
        var_95=var_95 * 100,
        cvar_95=cvar_95 * 100,
        beta=beta,
        alpha_annual=alpha_annual,
    )


# ---------------------------------------------------------------------------
# Rolling Metrics
# ---------------------------------------------------------------------------


def build_rolling_metrics(curve: list[SeriesPoint]) -> list[RollingMetricPoint]:
    """Build rolling 63-day and 126-day metrics from an equity curve."""
    ordered_curve = _sort_points(curve)
    if len(ordered_curve) < 25:
        return []
    rets = curve_to_returns(ordered_curve)
    output: list[RollingMetricPoint] = []

    for i in range(len(rets)):
        rolling63 = rets[max(0, i - 62) : i + 1]
        rolling126_curve = ordered_curve[max(0, i + 1 - 126) : i + 2]
        rolling_values = [r.value for r in rolling63]
        downside_values = [v for v in rolling_values if v < 0]
        std = sample_std_dev(rolling_values)
        downside_std = sample_std_dev(downside_values)

        rolling_return_fraction = 1.0
        for v in rolling_values:
            rolling_return_fraction *= 1 + v
        rolling_return_fraction -= 1

        rolling_sharpe = (mean(rolling_values) / std) * math.sqrt(252) if std > 0 else None
        rolling_sortino = (mean(rolling_values) / downside_std) * math.sqrt(252) if downside_std > 0 else None
        rolling_vol = std * math.sqrt(252) * 100 if std > 0 else None
        rolling_drawdown = compute_max_drawdown(rolling126_curve) if len(rolling126_curve) >= 20 else None

        output.append(
            RollingMetricPoint(
                date=rets[i].date,
                rolling_return_63d=round(rolling_return_fraction * 100, 2),
                rolling_sharpe_63d=round(rolling_sharpe, 3) if rolling_sharpe is not None else None,
                rolling_sortino_63d=round(rolling_sortino, 3) if rolling_sortino is not None else None,
                rolling_volatility_63d=round(rolling_vol, 2) if rolling_vol is not None else None,
                rolling_drawdown_126d=round(rolling_drawdown, 2) if rolling_drawdown is not None else None,
            )
        )
    return output


# ---------------------------------------------------------------------------
# Correlation Analysis
# ---------------------------------------------------------------------------


def build_correlation_matrix(
    series_map: dict[str, list[SeriesPoint]],
) -> dict[str, dict[str, float | None]]:
    """Build a pairwise correlation matrix from a named series map."""
    keys = list(series_map.keys())
    matrix: dict[str, dict[str, float | None]] = {}
    for left_key in keys:
        matrix[left_key] = {}
        for right_key in keys:
            if left_key == right_key:
                matrix[left_key][right_key] = 1.0
                continue
            aligned = align_two_series(series_map[left_key], series_map[right_key])
            if len(aligned["left"]) < 3:
                matrix[left_key][right_key] = None
                continue
            matrix[left_key][right_key] = round(
                correlation(aligned["left"], aligned["right"]), 4
            )
    return matrix


def build_rolling_correlation(
    left: list[SeriesPoint],
    right: list[SeriesPoint],
    window: int = 63,
) -> list[dict[str, Any]]:
    """Build rolling pairwise correlation between two series."""
    aligned = align_two_series(left, right)
    if len(aligned["left"]) < 3:
        return []
    output: list[dict[str, Any]] = []
    for i in range(len(aligned["left"])):
        start = max(0, i - window + 1)
        left_slice = aligned["left"][start : i + 1]
        right_slice = aligned["right"][start : i + 1]
        min_length = min(20, window)
        corr = (
            round(correlation(left_slice, right_slice), 4)
            if len(left_slice) >= min_length
            else None
        )
        output.append({"date": aligned["dates"][i], "correlation": corr})
    return output


# ---------------------------------------------------------------------------
# Walk-Forward Analysis
# ---------------------------------------------------------------------------


def _estimate_p_value_from_sharpe(
    sharpe: float,
    observations: int,
) -> float | None:
    """Estimate p-value from annualised Sharpe and observation count."""
    if not math.isfinite(sharpe) or observations < 3:
        return None
    t_score = abs(sharpe) * math.sqrt(observations / 252)
    p_value = 2 * (1 - normal_cdf(t_score))
    return max(0.0, min(1.0, p_value))


def build_walk_forward_analysis(
    returns: list[SeriesPoint],
    benchmark_returns: list[SeriesPoint] | None = None,
    splits: int = 5,
    train_ratio: float = 0.7,
) -> list[WalkForwardSplit]:
    """Run expanding-window walk-forward analysis on a return series."""
    ordered = _sort_points(returns)
    if len(ordered) < 60 or splits <= 0:
        return []

    safe_train_ratio = max(0.4, min(0.9, train_ratio))
    step = max(20, int((len(ordered) * (1 - safe_train_ratio)) / splits))
    output: list[WalkForwardSplit] = []

    for s in range(splits):
        train_end = int(len(ordered) * safe_train_ratio) + s * step
        test_end = min(len(ordered), train_end + step)
        train = ordered[:train_end]
        test = ordered[train_end:test_end]
        if len(train) < 40 or len(test) < 20:
            break

        train_metrics = compute_performance_metrics(train, None, benchmark_returns)
        test_metrics = compute_performance_metrics(test, None, benchmark_returns)
        output.append(
            WalkForwardSplit(
                split=s + 1,
                train_start=train[0].date,
                train_end=train[-1].date,
                test_start=test[0].date,
                test_end=test[-1].date,
                train_return=round(train_metrics.total_return, 2),
                test_return=round(test_metrics.total_return, 2),
                train_cagr=round(train_metrics.cagr, 2),
                test_cagr=round(test_metrics.cagr, 2),
                train_sharpe=round(train_metrics.sharpe, 3),
                test_sharpe=round(test_metrics.sharpe, 3),
                p_value=_estimate_p_value_from_sharpe(test_metrics.sharpe, len(test)),
            )
        )

    return output


# ---------------------------------------------------------------------------
# Monte Carlo Bootstrap
# ---------------------------------------------------------------------------


def run_monte_carlo_bootstrap(
    returns: list[SeriesPoint],
    iterations: int = 750,
    horizon_days: int = 252,
    seed: int = 42,
) -> MonteCarloSummary:
    """Run bootstrap Monte Carlo simulation from historical returns."""
    values = [r.value for r in _sort_points(returns)]
    if not values:
        return MonteCarloSummary()

    iters = max(100, min(10_000, int(iterations)))
    horizon = max(21, min(2520, int(horizon_days)))
    rng = _Xorshift32(seed)

    terminal_values: list[float] = []
    day_buckets: list[list[float]] = [[] for _ in range(horizon)]

    for _ in range(iters):
        equity = 1.0
        for day in range(horizon):
            idx = int(rng.next() * len(values))
            idx = min(len(values) - 1, idx)
            equity *= 1 + values[idx]
            day_buckets[day].append(equity)
        terminal_values.append(equity)

    expected_terminal = mean(terminal_values)
    expected_cagr = (math.pow(expected_terminal, 252 / horizon) - 1) * 100

    paths = [
        MonteCarloPathPoint(
            day=i + 1,
            p05=quantile(bucket, 0.05),
            p25=quantile(bucket, 0.25),
            p50=quantile(bucket, 0.50),
            p75=quantile(bucket, 0.75),
            p95=quantile(bucket, 0.95),
        )
        for i, bucket in enumerate(day_buckets)
    ]

    return MonteCarloSummary(
        iterations=iters,
        horizon_days=horizon,
        terminal_p05=quantile(terminal_values, 0.05),
        terminal_p25=quantile(terminal_values, 0.25),
        terminal_p50=quantile(terminal_values, 0.50),
        terminal_p75=quantile(terminal_values, 0.75),
        terminal_p95=quantile(terminal_values, 0.95),
        expected_terminal=expected_terminal,
        expected_cagr=round(expected_cagr, 2),
        probability_of_loss=(sum(1 for v in terminal_values if v < 1) / len(terminal_values)) * 100,
        max_terminal=max(terminal_values),
        min_terminal=min(terminal_values),
        paths=paths,
    )


# ---------------------------------------------------------------------------
# Stress Scenarios
# ---------------------------------------------------------------------------


def run_stress_scenarios(returns: list[SeriesPoint]) -> list[StressScenarioResult]:
    """Generate stress scenarios from historical returns."""
    ordered = [r.value for r in _sort_points(returns)]
    if not ordered:
        return []

    var_95 = quantile(ordered, 0.05)
    cvar_set = [v for v in ordered if v <= var_95]
    cvar_95 = mean(cvar_set) if cvar_set else var_95
    std = sample_std_dev(ordered)
    worst_day = min(ordered)

    if len(ordered) < 5:
        worst_week = worst_day
    else:
        worst_week = float("inf")
        for i in range(4, len(ordered)):
            cumulative = 1.0
            for v in ordered[i - 4 : i + 1]:
                cumulative *= 1 + v
            cumulative -= 1
            worst_week = min(worst_week, cumulative)

    return [
        StressScenarioResult(
            name="Historical Worst Day",
            shock_pct=round(worst_day * 100, 2),
            estimated_portfolio_impact_pct=round(worst_day * 100, 2),
            expected_shortfall_pct=round(cvar_95 * 100, 2),
        ),
        StressScenarioResult(
            name="Historical Worst 5D",
            shock_pct=round(worst_week * 100, 2),
            estimated_portfolio_impact_pct=round(worst_week * 100, 2),
            expected_shortfall_pct=round(cvar_95 * 100, 2),
        ),
        StressScenarioResult(
            name="Volatility Spike (2 sigma)",
            shock_pct=round(-2 * std * 100, 2),
            estimated_portfolio_impact_pct=round(-2 * std * 100, 2),
            expected_shortfall_pct=round(cvar_95 * 100, 2),
        ),
        StressScenarioResult(
            name="Tail Risk Shock (VaR95)",
            shock_pct=round(var_95 * 100, 2),
            estimated_portfolio_impact_pct=round(var_95 * 100, 2),
            expected_shortfall_pct=round(cvar_95 * 100, 2),
        ),
    ]


# ---------------------------------------------------------------------------
# Transaction Cost Modeling
# ---------------------------------------------------------------------------


def apply_transaction_costs(
    returns: list[SeriesPoint],
    *,
    slippage_bps: float = 0,
    spread_bps: float = 0,
    market_impact_bps: float = 0,
    tax_rate_pct: float = 0,
    rebalance_every_days: int = 21,
) -> TransactionCostResult:
    """Apply modelled transaction costs to a return series."""
    ordered = _sort_points(returns)
    if not ordered:
        return TransactionCostResult(adjusted=[], total_cost_impact_pct=0, per_rebalance_cost_pct=0, rebalance_events=0)

    slippage = slippage_bps / 10_000
    spread = spread_bps / 10_000
    impact = market_impact_bps / 10_000
    tax = tax_rate_pct / 100
    rebalance_days = max(1, int(rebalance_every_days))
    per_rebalance_cost = slippage + spread + impact + tax

    rebalance_events = 0
    total_cost_fraction = 0.0
    adjusted: list[SeriesPoint] = []
    for i, row in enumerate(ordered):
        adjusted_value = row.value
        if i % rebalance_days == 0:
            adjusted_value -= per_rebalance_cost
            rebalance_events += 1
            total_cost_fraction += per_rebalance_cost
        adjusted.append(SeriesPoint(date=row.date, value=adjusted_value))

    return TransactionCostResult(
        adjusted=adjusted,
        total_cost_impact_pct=total_cost_fraction * 100,
        per_rebalance_cost_pct=per_rebalance_cost * 100,
        rebalance_events=rebalance_events,
    )


# ---------------------------------------------------------------------------
# Mean-Variance Allocation Optimization
# ---------------------------------------------------------------------------


def _random_weights(asset_count: int, rng: _Xorshift32) -> list[float]:
    """Generate random portfolio weights via Dirichlet-like sampling."""
    samples = [-math.log(max(rng.next(), 0.000001)) for _ in range(asset_count)]
    total = sum(samples)
    return [s / total for s in samples]


def optimize_mean_variance_allocation(
    series_by_asset: dict[str, list[SeriesPoint]],
    *,
    iterations: int = 5000,
    seed: int = 42,
    max_frontier_points: int = 150,
) -> AllocationOptimizationResult:
    """Random-search mean-variance portfolio optimization with efficient frontier."""
    assets = list(series_by_asset.keys())
    if not assets:
        return AllocationOptimizationResult(
            assets=[], best_weights={}, best_expected_return=0, best_volatility=0, best_sharpe=0,
        )

    aligned = align_series_by_date(series_by_asset)
    if not aligned["dates"]:
        return AllocationOptimizationResult(
            assets=assets,
            best_weights={a: round(1 / len(assets), 4) for a in assets},
            best_expected_return=0,
            best_volatility=0,
            best_sharpe=0,
        )

    returns_by_asset = [aligned["values"][a] for a in assets]
    means = [mean(r) * 252 for r in returns_by_asset]
    cov_matrix = [
        [covariance(left, right) * 252 for right in returns_by_asset]
        for left in returns_by_asset
    ]

    iters = max(250, min(30_000, iterations))
    frontier_limit = max(30, min(500, max_frontier_points))
    rng = _Xorshift32(seed)

    best_sharpe = float("-inf")
    best_weights = _random_weights(len(assets), rng)
    best_return = 0.0
    best_vol = 0.0

    candidates: list[AllocationPoint] = []
    for _ in range(iters):
        weights = _random_weights(len(assets), rng)
        expected_return = sum(w * m for w, m in zip(weights, means))

        variance = 0.0
        for x in range(len(weights)):
            for y in range(len(weights)):
                variance += weights[x] * weights[y] * cov_matrix[x][y]
        vol = math.sqrt(max(variance, 0.0))
        sharpe = expected_return / vol if vol > 0 else 0.0

        weight_map = {a: round(weights[i], 4) for i, a in enumerate(assets)}
        candidates.append(
            AllocationPoint(
                weights=weight_map,
                expected_return=expected_return * 100,
                volatility=vol * 100,
                sharpe=sharpe,
            )
        )

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_weights = weights
            best_return = expected_return
            best_vol = vol

    # Build efficient frontier (ascending vol, rising returns)
    candidates.sort(key=lambda p: (p.volatility, -p.expected_return))
    frontier: list[AllocationPoint] = []
    for pt in candidates:
        if not frontier or pt.expected_return >= frontier[-1].expected_return:
            frontier.append(pt)
    frontier = frontier[:frontier_limit]

    return AllocationOptimizationResult(
        assets=assets,
        best_weights={a: round(best_weights[i], 4) for i, a in enumerate(assets)},
        best_expected_return=best_return * 100,
        best_volatility=best_vol * 100,
        best_sharpe=round(best_sharpe, 4),
        frontier=frontier,
    )


# ---------------------------------------------------------------------------
# Risk Contribution
# ---------------------------------------------------------------------------


def compute_risk_contribution(
    series_by_asset: dict[str, list[SeriesPoint]],
    weights: dict[str, float],
) -> RiskContributionResult:
    """Compute risk contribution and marginal risk for each asset in a portfolio."""
    assets = [a for a in series_by_asset.keys() if isinstance(weights.get(a), (int, float))]
    if not assets:
        return RiskContributionResult(contribution_pct={}, marginal_contribution={})

    filtered_map = {a: series_by_asset[a] for a in assets}
    aligned = align_series_by_date(filtered_map)
    if not aligned["dates"]:
        return RiskContributionResult(
            contribution_pct={a: 0 for a in assets},
            marginal_contribution={a: 0 for a in assets},
        )

    cov_matrix = [
        [covariance(aligned["values"][left], aligned["values"][right]) * 252 for right in assets]
        for left in assets
    ]
    weight_vector = [weights[a] for a in assets]
    marginal: list[float] = []

    for i in range(len(assets)):
        s = sum(cov_matrix[i][j] * weight_vector[j] for j in range(len(assets)))
        marginal.append(s)

    portfolio_variance = sum(marginal[i] * weight_vector[i] for i in range(len(assets)))
    contribution_raw = [
        (weight_vector[i] * marginal[i]) / portfolio_variance if portfolio_variance > 0 else 0
        for i in range(len(assets))
    ]

    return RiskContributionResult(
        contribution_pct={a: round(contribution_raw[i] * 100, 2) for i, a in enumerate(assets)},
        marginal_contribution={a: round(marginal[i], 6) for i, a in enumerate(assets)},
    )


# ---------------------------------------------------------------------------
# Combined / Utility Functions
# ---------------------------------------------------------------------------


def combine_return_series_equal_weight(
    series_map: dict[str, list[SeriesPoint]],
) -> list[SeriesPoint]:
    """Combine multiple return series using equal weights."""
    keys = list(series_map.keys())
    if not keys:
        return []
    aligned = align_series_by_date(series_map)
    if not aligned["dates"]:
        return []
    return [
        SeriesPoint(
            date=aligned["dates"][idx],
            value=sum(aligned["values"][k][idx] for k in keys) / len(keys),
        )
        for idx in range(len(aligned["dates"]))
    ]


def compute_diversification_score(
    correlation_matrix: dict[str, dict[str, float | None]],
) -> DiversificationScore:
    """Compute average pairwise correlation and diversification ratio."""
    keys = list(correlation_matrix.keys())
    if len(keys) < 2:
        return DiversificationScore(average_pairwise_correlation=0, diversification_ratio=1.0)

    values: list[float] = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            corr = (correlation_matrix.get(keys[i]) or {}).get(keys[j])
            if isinstance(corr, (int, float)) and math.isfinite(corr):
                values.append(corr)

    avg = mean(values) if values else 0.0
    diversification_ratio = 1.0 / max(0.05, 1 + avg)
    return DiversificationScore(
        average_pairwise_correlation=round(avg, 4),
        diversification_ratio=round(diversification_ratio, 4),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Data classes
    "SeriesPoint",
    "PerformanceMetrics",
    "RollingMetricPoint",
    "WalkForwardSplit",
    "MonteCarloPathPoint",
    "MonteCarloSummary",
    "StressScenarioResult",
    "AllocationPoint",
    "AllocationOptimizationResult",
    "RiskContributionResult",
    "TransactionCostResult",
    "DiversificationScore",
    # Math utilities
    "mean",
    "sample_std_dev",
    "covariance",
    "correlation",
    "quantile",
    "normal_cdf",
    # Core series functions
    "curve_to_returns",
    "returns_to_equity",
    "compute_max_drawdown",
    # Alignment
    "align_two_series",
    "align_series_by_date",
    # Performance metrics
    "compute_performance_metrics",
    # Rolling metrics
    "build_rolling_metrics",
    # Correlation
    "build_correlation_matrix",
    "build_rolling_correlation",
    # Walk-forward
    "build_walk_forward_analysis",
    # Monte Carlo
    "run_monte_carlo_bootstrap",
    # Stress
    "run_stress_scenarios",
    # Transaction costs
    "apply_transaction_costs",
    # Optimization
    "optimize_mean_variance_allocation",
    # Risk contribution
    "compute_risk_contribution",
    # Combination
    "combine_return_series_equal_weight",
    "compute_diversification_score",
]
