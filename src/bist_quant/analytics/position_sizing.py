"""Position sizing, correlation-adjusted sizing, and hedge suggestions."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .core_metrics import (
    SeriesPoint,
    _sort_points,
    build_correlation_matrix,
    mean,
)
from ._shared import (
    _clamp,
    _normalize_weights,
    _to_fixed,
)


@dataclass
class CorrelationAdjustedSizingResult:
    adjusted_weights: dict[str, float]
    average_pairwise_correlation: float
    concentration_penalty: float


@dataclass
class HedgeSuggestion:
    primary: str
    hedge: str
    correlation: float
    hedge_ratio: float
    rationale: str


def compute_kelly_fraction_percent(
    win_rate_pct: float,
    win_loss_ratio: float,
    fractional_kelly: float = 0.5,
) -> float:
    """Kelly criterion position sizing (fractional)."""
    p = _clamp(win_rate_pct / 100, 0.01, 0.99)
    b = max(0.1, win_loss_ratio)
    full_kelly = p - (1 - p) / b
    return _to_fixed(_clamp(full_kelly * fractional_kelly * 100, 0, 100), 2)


def compute_fixed_fractional_notional(
    account_equity: float,
    risk_per_trade_pct: float,
    stop_distance_pct: float,
) -> float:
    """Fixed fractional position sizing."""
    if not math.isfinite(account_equity) or account_equity <= 0:
        return 0.0
    risk_budget = account_equity * (_clamp(risk_per_trade_pct, 0, 20) / 100)
    stop_fraction = max(0.001, _clamp(stop_distance_pct, 0.1, 100) / 100)
    return _to_fixed(risk_budget / stop_fraction, 2)


def compute_optimal_f(returns: list[SeriesPoint]) -> float:
    """Optimal-f position sizing via grid search on geometric growth."""
    values = [r.value for r in _sort_points(returns)]
    if len(values) < 20:
        return 0.0
    best_f = 0.0
    best_growth = float("-inf")
    f = 0.0
    while f <= 1.0:
        sum_log = 0.0
        valid = True
        for ret in values:
            gross = 1 + f * ret
            if gross <= 0:
                valid = False
                break
            sum_log += math.log(gross)
        if valid:
            growth = sum_log / len(values)
            if growth > best_growth:
                best_growth = growth
                best_f = f
        f += 0.01
    return _to_fixed(best_f, 3)


def compute_correlation_adjusted_sizing(
    series_by_asset: dict[str, list[SeriesPoint]],
    base_weights: dict[str, float],
) -> CorrelationAdjustedSizingResult:
    """Penalise concentrated correlated positions."""
    matrix = build_correlation_matrix(series_by_asset)
    assets = [a for a in base_weights if math.isfinite(base_weights[a]) and base_weights[a] > 0]
    if not assets:
        return CorrelationAdjustedSizingResult({}, 0, 0)

    adjusted_raw: dict[str, float] = {}
    pairwise: list[float] = []
    for asset in assets:
        corrs = []
        for peer in assets:
            if peer == asset:
                continue
            c = (matrix.get(asset) or {}).get(peer)
            if isinstance(c, (int, float)) and math.isfinite(c):
                corrs.append(c)
        avg_abs = mean([abs(c) for c in corrs]) if corrs else 0
        adjusted_raw[asset] = base_weights[asset] * (1 - 0.55 * avg_abs)
        pairwise.extend(corrs)

    normalized = _normalize_weights(adjusted_raw)
    avg_pw = mean(pairwise) if pairwise else 0
    return CorrelationAdjustedSizingResult(
        adjusted_weights=normalized,
        average_pairwise_correlation=_to_fixed(avg_pw, 4),
        concentration_penalty=_to_fixed(_clamp((abs(avg_pw) - 0.4) * 100, 0, 100), 2),
    )


def suggest_cross_asset_hedges(
    matrix: dict[str, dict[str, float | None]],
    max_suggestions: int = 5,
) -> list[HedgeSuggestion]:
    """Suggest hedges from a correlation matrix."""
    keys = list(matrix.keys())
    ideas: list[HedgeSuggestion] = []
    for li in range(len(keys)):
        for ri in range(li + 1, len(keys)):
            left, right = keys[li], keys[ri]
            corr = (matrix.get(left) or {}).get(right)
            if not isinstance(corr, (int, float)) or not math.isfinite(corr):
                continue
            if corr <= -0.25:
                ideas.append(HedgeSuggestion(
                    primary=left, hedge=right,
                    correlation=_to_fixed(corr, 3),
                    hedge_ratio=_to_fixed(_clamp(abs(corr), 0.2, 1.2), 3),
                    rationale="Negative correlation suggests offsetting directional risk.",
                ))
            elif corr >= 0.8:
                ideas.append(HedgeSuggestion(
                    primary=left, hedge=right,
                    correlation=_to_fixed(corr, 3),
                    hedge_ratio=_to_fixed(_clamp(corr * 0.5, 0.2, 0.8), 3),
                    rationale="Very high correlation indicates clustered risk; use partial overlay hedge.",
                ))
    ideas.sort(key=lambda h: abs(h.correlation), reverse=True)
    return ideas[: max(1, max_suggestions)]


__all__ = [
    "CorrelationAdjustedSizingResult",
    "HedgeSuggestion",
    "compute_kelly_fraction_percent",
    "compute_fixed_fractional_notional",
    "compute_optimal_f",
    "compute_correlation_adjusted_sizing",
    "suggest_cross_asset_hedges",
]
