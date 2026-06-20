"""Multi-method portfolio construction (MPT / risk-parity / min-variance / ERC / factor)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .core_metrics import (
    SeriesPoint,
    align_series_by_date,
    compute_risk_contribution,
    mean,
    optimize_mean_variance_allocation,
    sample_std_dev,
)
from ._shared import (
    PortfolioConstructionMethod,
    _normalize_weights,
    _to_fixed,
)


@dataclass
class PortfolioConstructionResult:
    method: PortfolioConstructionMethod
    weights: dict[str, float]
    risk_contribution_pct: dict[str, float]
    expected_return_pct: float
    expected_volatility_pct: float
    expected_sharpe: float


def _annualized_moments(
    series_by_asset: dict[str, list[SeriesPoint]],
) -> dict[str, Any]:
    assets = list(series_by_asset.keys())
    aligned = align_series_by_date(series_by_asset)
    if not assets or not aligned["dates"]:
        return {"assets": [], "means": [], "vols": []}
    means = [mean(aligned["values"][a]) * 252 * 100 for a in assets]
    vols = [sample_std_dev(aligned["values"][a]) * math.sqrt(252) * 100 for a in assets]
    return {"assets": assets, "means": means, "vols": vols}


def _build_factor_based_weights(
    series_by_asset: dict[str, list[SeriesPoint]],
) -> dict[str, float]:
    assets = list(series_by_asset.keys())
    if not assets:
        return {}
    aligned = align_series_by_date(series_by_asset)
    if not aligned["dates"]:
        return _normalize_weights({a: 1 / len(assets) for a in assets})

    raw_scores: dict[str, float] = {}
    for asset in assets:
        vals = aligned["values"][asset]
        r63 = vals[-63:]
        r21 = vals[-21:]
        momentum = 1.0
        for v in r63:
            momentum *= 1 + v
        momentum -= 1
        value_factor = 1.0
        for v in r21:
            value_factor *= 1 + v
        value_factor = -(value_factor - 1)
        sd = sample_std_dev(vals)
        quality = mean(vals) / sd if sd > 0 else 0
        raw_scores[asset] = 0.45 * momentum + 0.25 * value_factor + 0.3 * quality

    min_score = min(raw_scores.values())
    shifted = {a: s - min_score + 1e-4 for a, s in raw_scores.items()}
    return _normalize_weights(shifted)


def construct_portfolio_weights(
    series_by_asset: dict[str, list[SeriesPoint]],
    method: PortfolioConstructionMethod,
) -> PortfolioConstructionResult:
    """Multi-method portfolio construction."""
    assets = list(series_by_asset.keys())
    if not assets:
        return PortfolioConstructionResult(method=method, weights={}, risk_contribution_pct={},
                                           expected_return_pct=0, expected_volatility_pct=0, expected_sharpe=0)

    equal = _normalize_weights({a: 1.0 for a in assets})
    moments = _annualized_moments(series_by_asset)
    weights = equal

    if method == "mpt":
        opt = optimize_mean_variance_allocation(series_by_asset, iterations=5000, seed=91, max_frontier_points=180)
        weights = opt.best_weights if opt.best_weights else equal
    elif method == "min_variance":
        opt = optimize_mean_variance_allocation(series_by_asset, iterations=5000, seed=92, max_frontier_points=180)
        if opt.frontier:
            lowest = min(opt.frontier, key=lambda p: p.volatility)
            weights = lowest.weights
        else:
            weights = equal
    elif method == "risk_parity":
        inv_vol = {a: 1 / max(moments["vols"][i], 0.01) for i, a in enumerate(moments["assets"])}
        weights = _normalize_weights(inv_vol)
    elif method == "equal_risk_contribution":
        inv_vol = {a: 1 / max(moments["vols"][i], 0.01) for i, a in enumerate(moments["assets"])}
        work = _normalize_weights(inv_vol)
        for _ in range(8):
            risk = compute_risk_contribution(series_by_asset, work)
            target = 100 / max(1, len(moments["assets"]))
            adjusted = {}
            for a in moments["assets"]:
                cc = max(0.001, risk.contribution_pct.get(a, 0.001))
                adjusted[a] = (work.get(a, 0)) * math.sqrt(target / cc)
            work = _normalize_weights(adjusted)
        weights = work
    elif method == "factor_based":
        weights = _build_factor_based_weights(series_by_asset)

    risk = compute_risk_contribution(series_by_asset, weights)
    exp_ret = sum((weights.get(a, 0)) * (moments["means"][i] if i < len(moments["means"]) else 0)
                  for i, a in enumerate(moments["assets"]))
    exp_vol = sum((weights.get(a, 0)) * (moments["vols"][i] if i < len(moments["vols"]) else 0)
                  for i, a in enumerate(moments["assets"]))
    sharpe = exp_ret / exp_vol if exp_vol > 0 else 0

    return PortfolioConstructionResult(
        method=method, weights=weights, risk_contribution_pct=risk.contribution_pct,
        expected_return_pct=_to_fixed(exp_ret, 2),
        expected_volatility_pct=_to_fixed(exp_vol, 2),
        expected_sharpe=_to_fixed(sharpe, 3),
    )


__all__ = [
    "PortfolioConstructionResult",
    "construct_portfolio_weights",
]
