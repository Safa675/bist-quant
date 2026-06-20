"""Performance attribution breakdown and strategy significance testing."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .core_metrics import (
    SeriesPoint,
    _sort_points,
    _Xorshift32,
    align_series_by_date,
    correlation,
    covariance,
    mean,
    normal_cdf,
    sample_std_dev,
)
from ._shared import (
    _clamp,
    _to_fixed,
)


@dataclass
class SignificanceResult:
    t_stat: float
    p_value: float
    bootstrap_p_value: float
    is_significant_95: bool


@dataclass
class PerformanceAttributionBreakdown:
    asset_allocation_pct: float
    stock_selection_pct: float
    sector_rotation_pct: float
    currency_exposure_pct: float
    style_drift_score: float
    benchmark_relative_pct: float


def estimate_strategy_significance(
    strategy_returns: list[SeriesPoint],
    benchmark_returns: list[SeriesPoint] | None = None,
) -> SignificanceResult:
    """T-test + bootstrap significance of strategy excess returns."""
    ordered = _sort_points(strategy_returns)
    values = [r.value for r in ordered]

    if benchmark_returns and len(benchmark_returns) > 0:
        aligned = align_series_by_date({
            "strategy": ordered,
            "benchmark": _sort_points(benchmark_returns),
        })
        values = [
            (aligned["values"]["strategy"][i] or 0) - (aligned["values"]["benchmark"][i] or 0)
            for i in range(len(aligned["dates"]))
        ]

    if len(values) < 20:
        return SignificanceResult(t_stat=0, p_value=1, bootstrap_p_value=1, is_significant_95=False)

    avg = mean(values)
    std = sample_std_dev(values)
    t_stat = avg / (std / math.sqrt(len(values))) if std > 0 else 0.0
    p_value = _clamp(2 * (1 - normal_cdf(abs(t_stat))), 0, 1)

    rng = _Xorshift32(73)
    iterations = 600
    non_positive = 0
    for _ in range(iterations):
        boot_mean = 0.0
        for _ in range(len(values)):
            idx = int(rng.next() * len(values))
            idx = min(len(values) - 1, idx)
            boot_mean += values[idx]
        boot_mean /= len(values)
        if boot_mean <= 0:
            non_positive += 1
    bootstrap_p = non_positive / iterations

    return SignificanceResult(
        t_stat=_to_fixed(t_stat, 3),
        p_value=_to_fixed(p_value, 4),
        bootstrap_p_value=_to_fixed(bootstrap_p, 4),
        is_significant_95=p_value < 0.05 and bootstrap_p < 0.05,
    )


def compute_performance_attribution_breakdown(
    strategy_returns: list[SeriesPoint],
    benchmark_returns: list[SeriesPoint],
) -> PerformanceAttributionBreakdown:
    """Decompose strategy active return into allocation, selection, rotation, currency."""
    aligned = align_series_by_date({
        "strategy": _sort_points(strategy_returns),
        "benchmark": _sort_points(benchmark_returns),
    })
    if len(aligned["dates"]) < 20:
        return PerformanceAttributionBreakdown(0, 0, 0, 0, 0, 0)

    strategy = aligned["values"]["strategy"]
    benchmark = aligned["values"]["benchmark"]
    strat_total = 1.0
    for v in strategy:
        strat_total *= 1 + v
    strat_total -= 1
    bench_total = 1.0
    for v in benchmark:
        bench_total *= 1 + v
    bench_total -= 1

    active = strat_total - bench_total
    bench_var = covariance(benchmark, benchmark)
    beta_val = covariance(strategy, benchmark) / bench_var if bench_var > 0 else 1.0
    corr_val = correlation(strategy, benchmark)

    allocation = (beta_val - 1) * bench_total
    selection = active - allocation
    sector_rotation = selection * (0.28 + _clamp(1 - abs(corr_val), 0, 0.5))
    currency_exposure = active * (1 - abs(corr_val)) * 0.18

    rolling_beta: list[float] = []
    for i in range(62, len(strategy)):
        ss = strategy[i - 62 : i + 1]
        bs = benchmark[i - 62 : i + 1]
        vb = covariance(bs, bs)
        if vb <= 0:
            continue
        rolling_beta.append(covariance(ss, bs) / vb)
    style_drift = sample_std_dev(rolling_beta) * 100 if rolling_beta else abs(beta_val - 1) * 100

    return PerformanceAttributionBreakdown(
        asset_allocation_pct=_to_fixed(allocation * 100, 2),
        stock_selection_pct=_to_fixed((selection - sector_rotation) * 100, 2),
        sector_rotation_pct=_to_fixed(sector_rotation * 100, 2),
        currency_exposure_pct=_to_fixed(currency_exposure * 100, 2),
        style_drift_score=_to_fixed(style_drift, 2),
        benchmark_relative_pct=_to_fixed(active * 100, 2),
    )


__all__ = [
    "SignificanceResult",
    "PerformanceAttributionBreakdown",
    "estimate_strategy_significance",
    "compute_performance_attribution_breakdown",
]
