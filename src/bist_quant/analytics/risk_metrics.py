"""Monte Carlo robustness scoring and advanced risk metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .core_metrics import (
    MonteCarloSummary,
    SeriesPoint,
    _sort_points,
    compute_performance_metrics,
    mean,
    returns_to_equity,
    sample_std_dev,
)
from ._shared import (
    _clamp,
    _to_fixed,
)


@dataclass
class MonteCarloRobustnessSummary:
    robustness_score: float
    probability_of_loss: float
    terminal_tail_spread: float
    expected_cagr: float


@dataclass
class AdvancedRiskMetrics:
    cvar_95: float
    max_adverse_excursion: float
    expectancy_ratio: float | None
    profit_factor: float | None
    ulcer_index: float


def summarize_monte_carlo_robustness(
    summary: MonteCarloSummary,
) -> MonteCarloRobustnessSummary:
    """Score robustness from a Monte Carlo summary."""
    if not summary.iterations or not summary.horizon_days:
        return MonteCarloRobustnessSummary(0, 0, 0, 0)
    tail_spread = summary.terminal_p95 - summary.terminal_p05
    loss_penalty = _clamp(summary.probability_of_loss / 100, 0, 1)
    spread_penalty = _clamp(tail_spread / 2.5, 0, 1.5)
    reward = _clamp(summary.expected_cagr / 30, -1, 1.5)
    score = _clamp((1 - loss_penalty - spread_penalty * 0.45 + reward) * 100, 0, 100)
    return MonteCarloRobustnessSummary(
        robustness_score=_to_fixed(score, 2),
        probability_of_loss=_to_fixed(summary.probability_of_loss, 2),
        terminal_tail_spread=_to_fixed(tail_spread, 3),
        expected_cagr=_to_fixed(summary.expected_cagr, 2),
    )


def compute_advanced_risk_metrics(
    returns: list[SeriesPoint],
) -> AdvancedRiskMetrics:
    """CVaR, MAE, expectancy, profit factor, ulcer index."""
    ordered = _sort_points(returns)
    values = [r.value for r in ordered]
    metrics = compute_performance_metrics(ordered)
    if not values:
        return AdvancedRiskMetrics(0, 0, None, None, 0)

    equity = returns_to_equity(ordered, 1)
    peak = equity[0].value if equity else 1.0
    drawdowns: list[float] = []
    for row in equity:
        peak = max(peak, row.value)
        if peak <= 0:
            continue
        drawdowns.append(row.value / peak - 1)
    ulcer = math.sqrt(mean([dd * dd for dd in drawdowns])) * 100

    mae = 0.0
    for i in range(4, len(values)):
        cum = 1.0
        for v in values[i - 4 : i + 1]:
            cum *= 1 + v
        cum -= 1
        mae = min(mae, cum)

    positive = [v for v in values if v > 0]
    negative = [abs(v) for v in values if v < 0]
    avg_win = mean(positive) if positive else 0
    avg_loss = mean(negative) if negative else 0
    expectancy = avg_win / avg_loss if avg_loss > 0 else None

    return AdvancedRiskMetrics(
        cvar_95=_to_fixed(metrics.cvar_95, 2),
        max_adverse_excursion=_to_fixed(mae * 100, 2),
        expectancy_ratio=_to_fixed(expectancy, 3) if expectancy is not None else None,
        profit_factor=_to_fixed(metrics.profit_factor, 3) if metrics.profit_factor is not None else None,
        ulcer_index=_to_fixed(ulcer, 3),
    )


__all__ = [
    "MonteCarloRobustnessSummary",
    "AdvancedRiskMetrics",
    "summarize_monte_carlo_robustness",
    "compute_advanced_risk_metrics",
]
