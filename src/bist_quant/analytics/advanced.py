"""
Advanced quantitative analytics.

Ported from bist_quant.ai/src/lib/analytics/advanced.ts.
Provides GARCH volatility forecasting, position sizing, correlation-adjusted
sizing, hedge suggestions, parameter sensitivity heatmaps, walk-forward
parameter optimization, Monte Carlo robustness, advanced risk metrics,
portfolio construction (MPT / risk-parity / min-variance / ERC / factor-based),
regime-aware backtesting, strategy significance testing, volume profile,
Renko / Point & Figure charting, indicator combination signals, performance
attribution breakdown, and full backtest integration diagnostics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from .core_metrics import (
    MonteCarloSummary,
    PerformanceMetrics,
    SeriesPoint,
    _Xorshift32,
    _sort_points,
    align_series_by_date,
    apply_transaction_costs,
    build_correlation_matrix,
    compute_performance_metrics,
    compute_risk_contribution,
    correlation,
    covariance,
    mean,
    normal_cdf,
    optimize_mean_variance_allocation,
    quantile,
    returns_to_equity,
    run_monte_carlo_bootstrap,
    sample_std_dev,
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

VolatilityRegime = Literal["low", "normal", "high"]
PortfolioConstructionMethod = Literal[
    "mpt", "risk_parity", "min_variance", "equal_risk_contribution", "factor_based"
]
MarketRegime = Literal["bull", "bear", "sideways"]
ChartTimeframe = Literal["1D", "1W", "1M"]

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class PricePoint:
    date: str
    close: float
    volume: float | None = None


@dataclass
class VolatilityForecastPoint:
    date: str
    forecast_vol_pct: float
    realized_vol_pct: float
    regime: VolatilityRegime


@dataclass
class VolatilityForecastResult:
    window: int
    latest_forecast_vol_pct: float
    latest_realized_vol_pct: float
    latest_regime: VolatilityRegime
    adaptive_size_pct: float
    regime_distribution_pct: dict[str, float]
    series: list[VolatilityForecastPoint]


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


@dataclass
class PortfolioConstructionResult:
    method: PortfolioConstructionMethod
    weights: dict[str, float]
    risk_contribution_pct: dict[str, float]
    expected_return_pct: float
    expected_volatility_pct: float
    expected_sharpe: float


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


@dataclass
class SignificanceResult:
    t_stat: float
    p_value: float
    bootstrap_p_value: float
    is_significant_95: bool


@dataclass
class VolumeProfileBucket:
    bucket: str
    volume: float
    share_pct: float
    is_poc: bool


@dataclass
class RenkoBrick:
    index: int
    date: str
    price: float
    direction: Literal["up", "down"]


@dataclass
class PointFigureColumn:
    column: int
    type: Literal["X", "O"]
    start_date: str
    end_date: str
    boxes: int
    start_price: float
    end_price: float


@dataclass
class IndicatorCombinationSignal:
    trend_signal: Literal["bullish", "neutral", "bearish"]
    momentum_signal: Literal["overbought", "neutral", "oversold"]
    volatility_signal: Literal["compressed", "normal", "expanded"]
    rsi: float
    macd: float


@dataclass
class PerformanceAttributionBreakdown:
    asset_allocation_pct: float
    stock_selection_pct: float
    sector_rotation_pct: float
    currency_exposure_pct: float
    style_drift_score: float
    benchmark_relative_pct: float


@dataclass
class BacktestIntegrationDiagnostics:
    adjusted_metrics: PerformanceMetrics
    monte_carlo: MonteCarloSummary
    robustness: MonteCarloRobustnessSummary
    regime: RegimeBacktestResult
    significance: SignificanceResult
    total_cost_impact_pct: float


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float, hi: float) -> float:
    if not math.isfinite(value):
        return lo
    return max(lo, min(hi, value))


def _to_fixed(value: float, digits: int = 4) -> float:
    if not math.isfinite(value):
        return 0.0
    return round(value, digits)


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    entries = {k: v for k, v in weights.items() if math.isfinite(v) and v > 0}
    if not entries:
        return {}
    total = sum(entries.values())
    if total <= 0:
        return {}
    return {k: _to_fixed(v / total, 6) for k, v in entries.items()}


def _rolling_std(values: list[float], end_index: int, window: int) -> float:
    start = max(0, end_index - window + 1)
    return sample_std_dev(values[start : end_index + 1])


def _make_deterministic_noise(seed: int, index: int) -> float:
    raw = ((seed * 31 + index * 17) % 97) - 48
    return raw / 12_000


def _price_to_returns(points: list[PricePoint]) -> list[SeriesPoint]:
    ordered = sorted(points, key=lambda p: p.date)
    if len(ordered) < 2:
        return []
    returns: list[SeriesPoint] = []
    for i in range(1, len(ordered)):
        prev = ordered[i - 1].close
        cur = ordered[i].close
        if not math.isfinite(prev) or not math.isfinite(cur) or prev <= 0:
            continue
        returns.append(SeriesPoint(date=ordered[i].date, value=cur / prev - 1))
    return returns


def _build_ma_strategy_returns(
    points: list[PricePoint], fast: int, slow: int
) -> list[SeriesPoint]:
    ordered = sorted(points, key=lambda p: p.date)
    if len(ordered) < slow + 2:
        return []
    closes = [p.close for p in ordered]
    returns: list[SeriesPoint] = []
    prev_exposure = 0.0
    for i in range(1, len(ordered)):
        prev_c = closes[i - 1]
        cur_c = closes[i]
        if not math.isfinite(prev_c) or not math.isfinite(cur_c) or prev_c <= 0:
            continue
        spot = cur_c / prev_c - 1
        returns.append(SeriesPoint(date=ordered[i].date, value=spot * prev_exposure))
        if i + 1 < slow:
            prev_exposure = 0.0
            continue
        fast_slice = closes[max(0, i - fast + 1) : i + 1]
        slow_slice = closes[max(0, i - slow + 1) : i + 1]
        prev_exposure = 1.0 if mean(fast_slice) > mean(slow_slice) else 0.0
    return returns


def _ema(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    alpha = 2.0 / (period + 1)
    output: list[float] = []
    prev = values[0]
    for v in values:
        prev = alpha * v + (1 - alpha) * prev
        output.append(prev)
    return output


def _group_key_by_timeframe(date: str, timeframe: ChartTimeframe) -> str:
    if timeframe == "1D":
        return date
    try:
        parts = date[:10].split("-")
        y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
    except (ValueError, IndexError):
        return date
    if timeframe == "1W":
        import datetime
        dt = datetime.date(y, m, d)
        iso_year, iso_week, _ = dt.isocalendar()
        return f"{iso_year}-W{iso_week:02d}"
    return f"{y}-{m:02d}"


# ---------------------------------------------------------------------------
# Public Functions
# ---------------------------------------------------------------------------


def build_garch_volatility_forecast(
    returns: list[SeriesPoint],
    *,
    window: int = 50,
    target_vol_pct: float,
    base_allocation_pct: float,
    alpha: float = 0.08,
    beta: float = 0.9,
) -> VolatilityForecastResult:
    """EWMA/GARCH(1,1) volatility forecast with regime classification."""
    ordered = _sort_points(returns)
    values = [r.value for r in ordered]

    if not values:
        return VolatilityForecastResult(
            window=window,
            latest_forecast_vol_pct=0,
            latest_realized_vol_pct=0,
            latest_regime="normal",
            adaptive_size_pct=_clamp(base_allocation_pct, 0, 100),
            regime_distribution_pct={"low": 0, "normal": 100, "high": 0},
            series=[],
        )

    base_variance = max(sample_std_dev(values) ** 2, 1e-8)
    a = _clamp(alpha, 0.01, 0.2)
    b = _clamp(beta, 0.6, 0.98)
    omega = base_variance * max(1e-5, 1 - a - b)

    variance = base_variance
    forecast_vols: list[float] = []
    series: list[VolatilityForecastPoint] = []

    for i, pt in enumerate(ordered):
        ret = pt.value
        variance = omega + a * ret * ret + b * variance
        forecast = math.sqrt(max(variance, 0)) * math.sqrt(252) * 100
        realized = _rolling_std(values, i, window) * math.sqrt(252) * 100
        forecast_vols.append(forecast)
        series.append(VolatilityForecastPoint(
            date=pt.date,
            forecast_vol_pct=_to_fixed(forecast, 3),
            realized_vol_pct=_to_fixed(realized, 3),
            regime="normal",
        ))

    low_th = quantile(forecast_vols, 0.33)
    high_th = quantile(forecast_vols, 0.67)
    low_c = normal_c = high_c = 0

    for row in series:
        if row.forecast_vol_pct <= low_th:
            row.regime = "low"
            low_c += 1
        elif row.forecast_vol_pct >= high_th:
            row.regime = "high"
            high_c += 1
        else:
            normal_c += 1

    latest = series[-1] if series else VolatilityForecastPoint("", 0, 0, "normal")
    regime_mult = {"low": 1.1, "normal": 1.0, "high": 0.74}
    raw_adaptive = (
        (base_allocation_pct * target_vol_pct) / latest.forecast_vol_pct
        if latest.forecast_vol_pct > 0
        else base_allocation_pct
    )
    adaptive = raw_adaptive * regime_mult[latest.regime]
    total = max(1, len(series))

    return VolatilityForecastResult(
        window=window,
        latest_forecast_vol_pct=_to_fixed(latest.forecast_vol_pct, 2),
        latest_realized_vol_pct=_to_fixed(latest.realized_vol_pct, 2),
        latest_regime=latest.regime,
        adaptive_size_pct=_to_fixed(_clamp(adaptive, 0, 100), 2),
        regime_distribution_pct={
            "low": _to_fixed((low_c / total) * 100, 2),
            "normal": _to_fixed((normal_c / total) * 100, 2),
            "high": _to_fixed((high_c / total) * 100, 2),
        },
        series=series,
    )


def build_proxy_asset_return_series(
    base_returns: list[SeriesPoint],
    symbols: list[str],
) -> dict[str, list[SeriesPoint]]:
    """Create synthetic proxy return series for correlation analysis."""
    ordered = _sort_points(base_returns)
    if not ordered or not symbols:
        return {}
    safe_symbols = symbols[:12]
    output: dict[str, list[SeriesPoint]] = {}
    for sym_idx, symbol in enumerate(safe_symbols):
        seed = sum(ord(c) for c in symbol)
        lag = seed % 4
        scale = 0.76 + (seed % 35) / 100
        rows: list[SeriesPoint] = []
        for i, pt in enumerate(ordered):
            src_idx = max(0, i - lag)
            base = ordered[src_idx].value
            noise = _make_deterministic_noise(seed + sym_idx * 11, i)
            rows.append(SeriesPoint(
                date=pt.date,
                value=_clamp(base * scale + noise, -0.18, 0.18),
            ))
        output[symbol] = rows
    return output


# ---------------------------------------------------------------------------
# Position Sizing
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Correlation-Adjusted Sizing
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Hedge Suggestions
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Parameter Sensitivity Heatmap
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Walk-Forward Parameter Optimization
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Monte Carlo Robustness
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Advanced Risk Metrics
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Portfolio Construction
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Regime-Aware Backtest
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Strategy Significance
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Volume Profile / Renko / Point & Figure
# ---------------------------------------------------------------------------


def build_volume_profile(
    points: list[PricePoint],
    bucket_count: int = 12,
) -> list[VolumeProfileBucket]:
    """Build a volume profile from price data."""
    ordered = sorted(points, key=lambda p: p.date)
    if not ordered:
        return []
    closes = [p.close for p in ordered]
    min_price = min(closes)
    max_price = max(closes)
    buckets = max(6, min(30, int(bucket_count)))
    width = max(1e-6, (max_price - min_price) / buckets)
    totals = [0.0] * buckets

    for i, pt in enumerate(ordered):
        prev_close = ordered[i - 1].close if i > 0 else pt.close
        move = abs(pt.close / prev_close - 1) if prev_close > 0 else 0
        synthetic_volume = pt.volume if pt.volume is not None else 100_000 + move * 1_600_000
        bi = int(_clamp(math.floor((pt.close - min_price) / width), 0, buckets - 1))
        totals[bi] += synthetic_volume

    total_vol = sum(totals)
    poc = max(totals) if totals else 0
    result: list[VolumeProfileBucket] = []
    for i, vol in enumerate(totals):
        lo = min_price + width * i
        hi = lo + width
        result.append(VolumeProfileBucket(
            bucket=f"{_to_fixed(lo, 2)} - {_to_fixed(hi, 2)}",
            volume=_to_fixed(vol, 0),
            share_pct=_to_fixed((vol / total_vol) * 100, 2) if total_vol > 0 else 0,
            is_poc=vol == poc and vol > 0,
        ))
    return result


def build_renko_bricks(
    points: list[PricePoint],
    brick_size_pct: float = 1.2,
) -> list[RenkoBrick]:
    """Build Renko bricks from price data."""
    ordered = sorted(points, key=lambda p: p.date)
    if len(ordered) < 2:
        return []
    avg_price = mean([p.close for p in ordered])
    brick_size = max(0.01, avg_price * (_clamp(brick_size_pct, 0.1, 10) / 100))
    bricks: list[RenkoBrick] = []
    anchor = ordered[0].close
    guard = 0

    for i in range(1, len(ordered)):
        price = ordered[i].close
        diff = price - anchor
        while diff >= brick_size:
            anchor += brick_size
            bricks.append(RenkoBrick(index=len(bricks) + 1, date=ordered[i].date,
                                     price=_to_fixed(anchor, 3), direction="up"))
            diff = price - anchor
            guard += 1
            if guard > 50_000:
                break
        while diff <= -brick_size:
            anchor -= brick_size
            bricks.append(RenkoBrick(index=len(bricks) + 1, date=ordered[i].date,
                                     price=_to_fixed(anchor, 3), direction="down"))
            diff = price - anchor
            guard += 1
            if guard > 50_000:
                break
        if guard > 50_000:
            break
    return bricks[-320:]


def build_point_figure_columns(
    points: list[PricePoint],
    box_size_pct: float = 1.0,
    reversal: int = 3,
) -> list[PointFigureColumn]:
    """Build Point & Figure chart columns from price data."""
    ordered = sorted(points, key=lambda p: p.date)
    if len(ordered) < 2:
        return []
    avg_price = mean([p.close for p in ordered])
    box_size = max(0.01, avg_price * (_clamp(box_size_pct, 0.2, 8) / 100))
    reversal_boxes = max(2, min(6, int(reversal)))
    columns: list[PointFigureColumn] = []

    current: dict[str, Any] | None = None
    anchor = ordered[0].close

    for i in range(1, len(ordered)):
        price = ordered[i].close
        if current is None:
            if price >= anchor + box_size:
                boxes = max(1, int((price - anchor) / box_size))
                current = {"type": "X", "start_date": ordered[i - 1].date, "end_date": ordered[i].date,
                           "boxes": boxes, "start_price": anchor, "end_price": anchor + boxes * box_size}
                anchor = current["end_price"]
            elif price <= anchor - box_size:
                boxes = max(1, int((anchor - price) / box_size))
                current = {"type": "O", "start_date": ordered[i - 1].date, "end_date": ordered[i].date,
                           "boxes": boxes, "start_price": anchor, "end_price": anchor - boxes * box_size}
                anchor = current["end_price"]
            continue

        if current["type"] == "X":
            if price >= current["end_price"] + box_size:
                add = max(1, int((price - current["end_price"]) / box_size))
                current["boxes"] += add
                current["end_price"] += add * box_size
                current["end_date"] = ordered[i].date
                anchor = current["end_price"]
            elif price <= current["end_price"] - reversal_boxes * box_size:
                columns.append(PointFigureColumn(
                    column=len(columns) + 1, type=current["type"],
                    start_date=current["start_date"], end_date=current["end_date"],
                    boxes=current["boxes"],
                    start_price=_to_fixed(current["start_price"], 3),
                    end_price=_to_fixed(current["end_price"], 3),
                ))
                nb = max(reversal_boxes, int((current["end_price"] - price) / box_size))
                current = {"type": "O", "start_date": ordered[i].date, "end_date": ordered[i].date,
                           "boxes": nb, "start_price": current["end_price"],
                           "end_price": current["end_price"] - nb * box_size}
                anchor = current["end_price"]
        else:  # O column
            if price <= current["end_price"] - box_size:
                add = max(1, int((current["end_price"] - price) / box_size))
                current["boxes"] += add
                current["end_price"] -= add * box_size
                current["end_date"] = ordered[i].date
                anchor = current["end_price"]
            elif price >= current["end_price"] + reversal_boxes * box_size:
                columns.append(PointFigureColumn(
                    column=len(columns) + 1, type=current["type"],
                    start_date=current["start_date"], end_date=current["end_date"],
                    boxes=current["boxes"],
                    start_price=_to_fixed(current["start_price"], 3),
                    end_price=_to_fixed(current["end_price"], 3),
                ))
                nb = max(reversal_boxes, int((price - current["end_price"]) / box_size))
                current = {"type": "X", "start_date": ordered[i].date, "end_date": ordered[i].date,
                           "boxes": nb, "start_price": current["end_price"],
                           "end_price": current["end_price"] + nb * box_size}
                anchor = current["end_price"]

    if current:
        columns.append(PointFigureColumn(
            column=len(columns) + 1, type=current["type"],
            start_date=current["start_date"], end_date=current["end_date"],
            boxes=current["boxes"],
            start_price=_to_fixed(current["start_price"], 3),
            end_price=_to_fixed(current["end_price"], 3),
        ))
    return columns[-40:]


# ---------------------------------------------------------------------------
# Synchronized Timeframes
# ---------------------------------------------------------------------------


def build_synchronized_timeframes(
    points: list[PricePoint],
) -> dict[str, list[PricePoint]]:
    """Resample daily prices to weekly and monthly."""
    ordered = sorted(points, key=lambda p: p.date)
    output: dict[str, list[PricePoint]] = {"1D": ordered, "1W": [], "1M": []}

    for tf in ("1W", "1M"):
        grouped: dict[str, list[PricePoint]] = {}
        for row in ordered:
            key = _group_key_by_timeframe(row.date, tf)  # type: ignore[arg-type]
            grouped.setdefault(key, []).append(row)
        resampled: list[PricePoint] = []
        for _key in sorted(grouped):
            rows = grouped[_key]
            last = rows[-1]
            resampled.append(PricePoint(
                date=last.date,
                close=_to_fixed(last.close, 3),
                volume=_to_fixed(sum(r.volume or 0 for r in rows), 0),
            ))
        output[tf] = resampled
    return output


# ---------------------------------------------------------------------------
# Indicator Combination Signal
# ---------------------------------------------------------------------------


def build_indicator_combination_signal(
    points: list[PricePoint],
) -> IndicatorCombinationSignal:
    """RSI, MACD, SMA trend, and volatility regime signal."""
    ordered = sorted(points, key=lambda p: p.date)
    closes = [p.close for p in ordered]
    if len(closes) < 30:
        return IndicatorCombinationSignal("neutral", "neutral", "normal", 50, 0)

    sma20 = mean(closes[-20:])
    sma50 = mean(closes[-50:])
    trend: Literal["bullish", "neutral", "bearish"] = (
        "bullish" if sma20 > sma50 * 1.005 else "bearish" if sma20 < sma50 * 0.995 else "neutral"
    )

    gains = losses = 0.0
    for i in range(len(closes) - 14, len(closes)):
        delta = closes[i] - closes[i - 1]
        if delta > 0:
            gains += delta
        else:
            losses += abs(delta)
    rs = gains / losses if losses > 0 else 100
    rsi = 100 - 100 / (1 + rs)
    momentum: Literal["overbought", "neutral", "oversold"] = (
        "overbought" if rsi >= 70 else "oversold" if rsi <= 30 else "neutral"
    )

    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd_line = ema12[-1] - ema26[-1]

    recent_rets = [r.value for r in _price_to_returns(ordered)]
    vol_now = sample_std_dev(recent_rets[-21:])
    vol_base = sample_std_dev(recent_rets[-84:-21]) if len(recent_rets) > 84 else vol_now
    vol_signal: Literal["compressed", "normal", "expanded"] = (
        "expanded" if vol_now > vol_base * 1.25 else "compressed" if vol_now < vol_base * 0.8 else "normal"
    )

    return IndicatorCombinationSignal(
        trend_signal=trend,
        momentum_signal=momentum,
        volatility_signal=vol_signal,
        rsi=_to_fixed(rsi, 2),
        macd=_to_fixed(macd_line, 4),
    )


# ---------------------------------------------------------------------------
# Performance Attribution Breakdown
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Integration Diagnostics
# ---------------------------------------------------------------------------


def run_backtest_integration_diagnostics(
    returns: list[SeriesPoint],
    benchmark_returns: list[SeriesPoint] | None,
    *,
    slippage_bps: float,
    spread_bps: float,
    impact_bps: float,
    tax_pct: float,
    rebalance_days: int,
) -> BacktestIntegrationDiagnostics:
    """Full diagnostic pipeline: cost-adjust → metrics → MC → regime → significance."""
    adjusted = apply_transaction_costs(
        returns,
        slippage_bps=slippage_bps,
        spread_bps=spread_bps,
        market_impact_bps=impact_bps,
        tax_rate_pct=tax_pct,
        rebalance_every_days=rebalance_days,
    )
    eq = returns_to_equity(adjusted.adjusted, 1)
    adj_metrics = compute_performance_metrics(adjusted.adjusted, equity_curve=eq,
                                               benchmark_returns=benchmark_returns)
    mc = run_monte_carlo_bootstrap(adjusted.adjusted, iterations=700, horizon_days=252, seed=49)
    robust = summarize_monte_carlo_robustness(mc)
    regime = run_regime_aware_backtest(adjusted.adjusted)
    sig = estimate_strategy_significance(adjusted.adjusted, benchmark_returns)

    return BacktestIntegrationDiagnostics(
        adjusted_metrics=adj_metrics,
        monte_carlo=mc,
        robustness=robust,
        regime=regime,
        significance=sig,
        total_cost_impact_pct=_to_fixed(adjusted.total_cost_impact_pct, 2),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Data classes
    "PricePoint",
    "VolatilityForecastPoint",
    "VolatilityForecastResult",
    "CorrelationAdjustedSizingResult",
    "HedgeSuggestion",
    "ParameterHeatmapCell",
    "ParameterHeatmapResult",
    "WalkForwardOptimizationSplit",
    "WalkForwardOptimizationResult",
    "MonteCarloRobustnessSummary",
    "AdvancedRiskMetrics",
    "PortfolioConstructionResult",
    "RegimeStatRow",
    "RegimeBacktestResult",
    "SignificanceResult",
    "VolumeProfileBucket",
    "RenkoBrick",
    "PointFigureColumn",
    "IndicatorCombinationSignal",
    "PerformanceAttributionBreakdown",
    "BacktestIntegrationDiagnostics",
    # Functions
    "build_garch_volatility_forecast",
    "build_proxy_asset_return_series",
    "compute_kelly_fraction_percent",
    "compute_fixed_fractional_notional",
    "compute_optimal_f",
    "compute_correlation_adjusted_sizing",
    "suggest_cross_asset_hedges",
    "build_parameter_sensitivity_heatmap",
    "run_walk_forward_parameter_optimization",
    "summarize_monte_carlo_robustness",
    "compute_advanced_risk_metrics",
    "construct_portfolio_weights",
    "run_regime_aware_backtest",
    "estimate_strategy_significance",
    "build_volume_profile",
    "build_renko_bricks",
    "build_point_figure_columns",
    "build_synchronized_timeframes",
    "build_indicator_combination_signal",
    "compute_performance_attribution_breakdown",
    "run_backtest_integration_diagnostics",
]
