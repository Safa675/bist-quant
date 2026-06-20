"""Parameter sensitivity heatmap and walk-forward optimization."""

from __future__ import annotations

from dataclasses import dataclass

from .core_metrics import (
    SeriesPoint,
    compute_performance_metrics,
    mean,
)
from ._shared import (
    PricePoint,
    _build_ma_strategy_returns,
    _to_fixed,
)


@dataclass
class ParameterHeatmapCell:
    fast: int
    slow: int
    sharpe: float
    cagr: float
    max_drawdown: float
    score: float


@dataclass
class ParameterHeatmapResult:
    cells: list[ParameterHeatmapCell]
    best: ParameterHeatmapCell | None


@dataclass
class WalkForwardOptimizationSplit:
    split: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    fast: int
    slow: int
    train_sharpe: float
    test_sharpe: float
    train_return: float
    test_return: float


@dataclass
class WalkForwardOptimizationResult:
    splits: list[WalkForwardOptimizationSplit]
    average_test_sharpe: float
    average_test_return: float


def build_parameter_sensitivity_heatmap(
    prices: list[PricePoint],
    fast_options: list[int],
    slow_options: list[int],
) -> ParameterHeatmapResult:
    """Parameter sweep over MA crossover and score each combination."""
    safe_fast = sorted(set(f for f in fast_options if isinstance(f, int) and f > 1))
    safe_slow = sorted(set(s for s in slow_options if isinstance(s, int) and s > 2))
    cells: list[ParameterHeatmapCell] = []

    for fast in safe_fast:
        for slow in safe_slow:
            if fast >= slow:
                continue
            strat_rets = _build_ma_strategy_returns(prices, fast, slow)
            if len(strat_rets) < 30:
                continue
            m = compute_performance_metrics(strat_rets)
            score = m.sharpe * 1.4 + m.cagr / 22 - abs(m.max_drawdown) / 35
            cells.append(ParameterHeatmapCell(
                fast=fast, slow=slow,
                sharpe=_to_fixed(m.sharpe, 3),
                cagr=_to_fixed(m.cagr, 2),
                max_drawdown=_to_fixed(m.max_drawdown, 2),
                score=_to_fixed(score, 3),
            ))

    best = max(cells, key=lambda c: c.score) if cells else None
    return ParameterHeatmapResult(cells=cells, best=best)


def run_walk_forward_parameter_optimization(
    prices: list[PricePoint],
    fast_options: list[int],
    slow_options: list[int],
    splits: int = 4,
) -> WalkForwardOptimizationResult:
    """Walk-forward optimization over MA parameters."""
    ordered = sorted(prices, key=lambda p: p.date)
    if len(ordered) < 120:
        return WalkForwardOptimizationResult(splits=[], average_test_sharpe=0, average_test_return=0)

    output: list[WalkForwardOptimizationSplit] = []
    split_count = max(2, min(8, int(splits)))
    train_base = int(len(ordered) * 0.6)
    step = max(20, int((len(ordered) - train_base) / split_count))

    for s in range(split_count):
        train_end = train_base + s * step
        test_end = min(len(ordered), train_end + step)
        if train_end < 70 or test_end - train_end < 20:
            break
        train_set = ordered[:train_end]
        test_set = ordered[max(0, train_end - 220) : test_end]
        heatmap = build_parameter_sensitivity_heatmap(train_set, fast_options, slow_options)
        if not heatmap.best:
            continue
        bf, bs = heatmap.best.fast, heatmap.best.slow
        train_rets = _build_ma_strategy_returns(train_set, bf, bs)
        all_test_rets = _build_ma_strategy_returns(test_set, bf, bs)
        test_start_date = ordered[train_end].date
        test_rets = [r for r in all_test_rets if r.date >= test_start_date]
        tm = compute_performance_metrics(train_rets)
        tsm = compute_performance_metrics(test_rets)
        output.append(WalkForwardOptimizationSplit(
            split=s + 1,
            train_start=train_set[0].date,
            train_end=train_set[-1].date,
            test_start=ordered[train_end].date,
            test_end=ordered[test_end - 1].date,
            fast=bf, slow=bs,
            train_sharpe=_to_fixed(tm.sharpe, 3),
            test_sharpe=_to_fixed(tsm.sharpe, 3),
            train_return=_to_fixed(tm.total_return, 2),
            test_return=_to_fixed(tsm.total_return, 2),
        ))

    return WalkForwardOptimizationResult(
        splits=output,
        average_test_sharpe=_to_fixed(mean([r.test_sharpe for r in output]), 3),
        average_test_return=_to_fixed(mean([r.test_return for r in output]), 2),
    )


__all__ = [
    "ParameterHeatmapCell",
    "ParameterHeatmapResult",
    "WalkForwardOptimizationSplit",
    "WalkForwardOptimizationResult",
    "build_parameter_sensitivity_heatmap",
    "run_walk_forward_parameter_optimization",
]
