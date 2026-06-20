"""Regime-aware backtesting (bull / bear / sideways classification)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .core_metrics import (
    SeriesPoint,
    _sort_points,
    mean,
    quantile,
    sample_std_dev,
)
from ._shared import (
    MarketRegime,
    _to_fixed,
)


@dataclass
class RegimeStatRow:
    regime: MarketRegime
    observations: int
    annual_return_pct: float
    annual_volatility_pct: float
    sharpe: float


@dataclass
class RegimeBacktestResult:
    regime_points: list[dict[str, Any]]
    regime_stats: list[RegimeStatRow]


def run_regime_aware_backtest(
    returns: list[SeriesPoint],
) -> RegimeBacktestResult:
    """Classify market regimes and compute per-regime statistics."""
    ordered = _sort_points(returns)
    values = [r.value for r in ordered]
    if len(ordered) < 40:
        return RegimeBacktestResult(
            regime_points=[],
            regime_stats=[
                RegimeStatRow(regime=r, observations=0, annual_return_pct=0, annual_volatility_pct=0, sharpe=0)
                for r in ("bull", "bear", "sideways")
            ],
        )

    rolling_vol: list[float] = []
    rolling_trend: list[float] = []
    for i in range(len(ordered)):
        start = max(0, i - 62)
        sl = values[start : i + 1]
        rolling_vol.append(sample_std_dev(sl) * math.sqrt(252))
        rolling_trend.append(mean(sl) * 252)

    high_vol = quantile(rolling_vol, 0.7)
    low_vol = quantile(rolling_vol, 0.3)
    regime_points: list[dict[str, Any]] = []
    buckets: dict[str, list[float]] = {"bull": [], "bear": [], "sideways": []}

    for i in range(len(ordered)):
        regime: MarketRegime = "sideways"
        if rolling_trend[i] > 0.05 and rolling_vol[i] <= high_vol:
            regime = "bull"
        elif rolling_trend[i] < -0.03 or rolling_vol[i] >= high_vol * 1.05:
            regime = "bear"
        elif rolling_vol[i] <= low_vol:
            regime = "sideways"
        regime_points.append({"date": ordered[i].date, "regime": regime})
        buckets[regime].append(values[i])

    stats: list[RegimeStatRow] = []
    for r in ("bull", "bear", "sideways"):
        rows = buckets[r]
        ar = mean(rows) * 252 * 100 if rows else 0
        av = sample_std_dev(rows) * math.sqrt(252) * 100 if len(rows) > 1 else 0
        stats.append(RegimeStatRow(
            regime=r,  # type: ignore[arg-type]
            observations=len(rows),
            annual_return_pct=_to_fixed(ar, 2),
            annual_volatility_pct=_to_fixed(av, 2),
            sharpe=_to_fixed(ar / av if av > 0 else 0, 3),
        ))

    return RegimeBacktestResult(regime_points=regime_points, regime_stats=stats)


__all__ = [
    "RegimeStatRow",
    "RegimeBacktestResult",
    "run_regime_aware_backtest",
]
