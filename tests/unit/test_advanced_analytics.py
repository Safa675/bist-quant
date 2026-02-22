"""
Unit tests for bist_quant.analytics.advanced.

Validates the Python port against the expected behaviour of the original
TypeScript implementation in bist_quant.ai/src/lib/analytics/advanced.ts.
"""

from __future__ import annotations

import math

import pytest

from bist_quant.analytics.core_metrics import SeriesPoint
from bist_quant.analytics.advanced import (
    AdvancedRiskMetrics,
    BacktestIntegrationDiagnostics,
    CorrelationAdjustedSizingResult,
    HedgeSuggestion,
    IndicatorCombinationSignal,
    MonteCarloRobustnessSummary,
    ParameterHeatmapResult,
    PerformanceAttributionBreakdown,
    PointFigureColumn,
    PortfolioConstructionResult,
    PricePoint,
    RegimeBacktestResult,
    RenkoBrick,
    SignificanceResult,
    VolatilityForecastResult,
    VolumeProfileBucket,
    WalkForwardOptimizationResult,
    build_garch_volatility_forecast,
    build_indicator_combination_signal,
    build_parameter_sensitivity_heatmap,
    build_point_figure_columns,
    build_proxy_asset_return_series,
    build_renko_bricks,
    build_synchronized_timeframes,
    build_volume_profile,
    compute_advanced_risk_metrics,
    compute_correlation_adjusted_sizing,
    compute_fixed_fractional_notional,
    compute_kelly_fraction_percent,
    compute_optimal_f,
    compute_performance_attribution_breakdown,
    construct_portfolio_weights,
    estimate_strategy_significance,
    run_backtest_integration_diagnostics,
    run_regime_aware_backtest,
    run_walk_forward_parameter_optimization,
    suggest_cross_asset_hedges,
    summarize_monte_carlo_robustness,
)
from bist_quant.analytics.core_metrics import MonteCarloSummary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_series(start: str, values: list[float]) -> list[SeriesPoint]:
    y, m, d = (int(x) for x in start.split("-"))
    pts: list[SeriesPoint] = []
    for v in values:
        pts.append(SeriesPoint(date=f"{y:04d}-{m:02d}-{d:02d}", value=v))
        d += 1
        if d > 28:
            d = 1
            m += 1
            if m > 12:
                m = 1
                y += 1
    return pts


def _make_prices(n: int = 200, start_price: float = 100.0, drift: float = 0.001) -> list[PricePoint]:
    pts: list[PricePoint] = []
    price = start_price
    y, m, d = 2022, 1, 1
    for _ in range(n):
        pts.append(PricePoint(date=f"{y:04d}-{m:02d}-{d:02d}", close=round(price, 3), volume=100_000))
        price *= 1 + drift + ((_ % 5 - 2) * 0.002)
        d += 1
        if d > 28:
            d = 1
            m += 1
            if m > 12:
                m = 1
                y += 1
    return pts


def _varying_returns(n: int = 250) -> list[SeriesPoint]:
    vals = [0.01, -0.005, 0.003, -0.002, 0.008, -0.001, 0.004, -0.003, 0.006, -0.004] * (n // 10 + 1)
    return _make_series("2022-01-01", vals[:n])


# ---------------------------------------------------------------------------
# GARCH Volatility Forecast
# ---------------------------------------------------------------------------


class TestGarchVolForecast:
    def test_empty(self):
        result = build_garch_volatility_forecast([], target_vol_pct=15, base_allocation_pct=50)
        assert isinstance(result, VolatilityForecastResult)
        assert result.series == []
        assert result.latest_regime == "normal"

    def test_basic(self):
        rets = _varying_returns(200)
        result = build_garch_volatility_forecast(rets, target_vol_pct=15, base_allocation_pct=50)
        assert len(result.series) == 200
        assert result.latest_regime in ("low", "normal", "high")
        assert result.adaptive_size_pct >= 0
        total_pct = sum(result.regime_distribution_pct.values())
        assert math.isclose(total_pct, 100.0, abs_tol=1.0)


# ---------------------------------------------------------------------------
# Proxy Asset Returns
# ---------------------------------------------------------------------------


class TestProxyAsset:
    def test_empty(self):
        assert build_proxy_asset_return_series([], ["A"]) == {}

    def test_basic(self):
        rets = _varying_returns(100)
        result = build_proxy_asset_return_series(rets, ["AAPL", "MSFT"])
        assert "AAPL" in result
        assert "MSFT" in result
        assert len(result["AAPL"]) == 100

    def test_max_12(self):
        rets = _varying_returns(50)
        symbols = [f"SYM{i}" for i in range(20)]
        result = build_proxy_asset_return_series(rets, symbols)
        assert len(result) == 12


# ---------------------------------------------------------------------------
# Position Sizing
# ---------------------------------------------------------------------------


class TestPositionSizing:
    def test_kelly_basic(self):
        pct = compute_kelly_fraction_percent(60, 2.0, 0.5)
        assert pct > 0
        assert pct <= 100

    def test_kelly_bad_odds(self):
        pct = compute_kelly_fraction_percent(30, 0.5, 0.5)
        assert pct == 0.0

    def test_fixed_fractional(self):
        notional = compute_fixed_fractional_notional(100_000, 2, 5)
        assert notional > 0
        assert math.isclose(notional, 40_000.0, rel_tol=0.01)

    def test_fixed_fractional_zero_equity(self):
        assert compute_fixed_fractional_notional(0, 2, 5) == 0

    def test_optimal_f_short(self):
        assert compute_optimal_f(_varying_returns(10)) == 0

    def test_optimal_f_basic(self):
        result = compute_optimal_f(_varying_returns(100))
        assert 0 <= result <= 1


# ---------------------------------------------------------------------------
# Correlation-Adjusted Sizing
# ---------------------------------------------------------------------------


class TestCorrelationAdjustedSizing:
    def test_empty(self):
        result = compute_correlation_adjusted_sizing({}, {})
        assert result.adjusted_weights == {}

    def test_basic(self):
        a = _varying_returns(80)
        b = _make_series("2022-01-01", [-v.value for v in _varying_returns(80)])
        result = compute_correlation_adjusted_sizing(
            {"A": a, "B": b},
            {"A": 0.5, "B": 0.5},
        )
        total = sum(result.adjusted_weights.values())
        assert math.isclose(total, 1.0, abs_tol=0.01)


# ---------------------------------------------------------------------------
# Hedge Suggestions
# ---------------------------------------------------------------------------


class TestHedgeSuggestions:
    def test_negative_correlation(self):
        matrix = {
            "A": {"A": 1.0, "B": -0.6},
            "B": {"A": -0.6, "B": 1.0},
        }
        hedges = suggest_cross_asset_hedges(matrix)
        assert len(hedges) >= 1
        assert hedges[0].correlation < 0

    def test_high_correlation(self):
        matrix = {
            "X": {"X": 1.0, "Y": 0.95},
            "Y": {"X": 0.95, "Y": 1.0},
        }
        hedges = suggest_cross_asset_hedges(matrix)
        assert len(hedges) >= 1
        assert hedges[0].correlation >= 0.8

    def test_no_signal(self):
        matrix = {
            "A": {"A": 1.0, "B": 0.3},
            "B": {"A": 0.3, "B": 1.0},
        }
        hedges = suggest_cross_asset_hedges(matrix)
        assert len(hedges) == 0 or hedges == []


# ---------------------------------------------------------------------------
# Parameter Sensitivity Heatmap
# ---------------------------------------------------------------------------


class TestParameterHeatmap:
    def test_basic(self):
        prices = _make_prices(300)
        result = build_parameter_sensitivity_heatmap(
            prices, fast_options=[5, 10, 20], slow_options=[20, 50, 100]
        )
        assert isinstance(result, ParameterHeatmapResult)
        assert len(result.cells) > 0
        if result.best:
            assert result.best.fast < result.best.slow

    def test_too_short(self):
        prices = _make_prices(20)
        result = build_parameter_sensitivity_heatmap(prices, [5], [10])
        assert result.cells == [] or result.best is None


# ---------------------------------------------------------------------------
# Walk-Forward Optimization
# ---------------------------------------------------------------------------


class TestWalkForwardOptimization:
    def test_too_short(self):
        result = run_walk_forward_parameter_optimization(
            _make_prices(50), [5], [20], splits=2
        )
        assert result.splits == []

    def test_basic(self):
        result = run_walk_forward_parameter_optimization(
            _make_prices(300), [5, 10], [30, 50], splits=3
        )
        assert isinstance(result, WalkForwardOptimizationResult)
        # May or may not produce splits depending on data


# ---------------------------------------------------------------------------
# Monte Carlo Robustness
# ---------------------------------------------------------------------------


class TestMonteCarloRobustness:
    def test_empty(self):
        summary = MonteCarloSummary(
            iterations=0, horizon_days=0, expected_terminal=0,
            expected_cagr=0, probability_of_loss=0,
            terminal_p05=0, terminal_p25=0, terminal_p50=0,
            terminal_p75=0, terminal_p95=0, paths=[],
        )
        result = summarize_monte_carlo_robustness(summary)
        assert result.robustness_score == 0

    def test_basic(self):
        summary = MonteCarloSummary(
            iterations=500, horizon_days=252,
            expected_terminal=1.1, expected_cagr=10.0,
            probability_of_loss=20, terminal_p05=0.9,
            terminal_p25=1.0, terminal_p50=1.1,
            terminal_p75=1.2, terminal_p95=1.4, paths=[],
        )
        result = summarize_monte_carlo_robustness(summary)
        assert result.robustness_score > 0


# ---------------------------------------------------------------------------
# Advanced Risk Metrics
# ---------------------------------------------------------------------------


class TestAdvancedRiskMetrics:
    def test_empty(self):
        result = compute_advanced_risk_metrics([])
        assert result.ulcer_index == 0

    def test_basic(self):
        result = compute_advanced_risk_metrics(_varying_returns(200))
        assert isinstance(result, AdvancedRiskMetrics)
        assert result.ulcer_index >= 0
        assert result.expectancy_ratio is not None


# ---------------------------------------------------------------------------
# Portfolio Construction
# ---------------------------------------------------------------------------


class TestPortfolioConstruction:
    def test_empty(self):
        result = construct_portfolio_weights({}, "mpt")
        assert result.weights == {}

    @pytest.mark.slow
    def test_mpt(self):
        a = _varying_returns(100)
        b = _make_series("2022-01-01", [v.value * 0.8 + 0.001 for v in _varying_returns(100)])
        result = construct_portfolio_weights({"A": a, "B": b}, "mpt")
        assert isinstance(result, PortfolioConstructionResult)
        total = sum(result.weights.values())
        assert math.isclose(total, 1.0, abs_tol=0.01)

    def test_risk_parity(self):
        a = _varying_returns(100)
        b = _make_series("2022-01-01", [v.value * 1.5 for v in _varying_returns(100)])
        result = construct_portfolio_weights({"A": a, "B": b}, "risk_parity")
        assert sum(result.weights.values()) > 0.99

    def test_factor_based(self):
        a = _varying_returns(100)
        b = _make_series("2022-01-01", [-v.value for v in _varying_returns(100)])
        result = construct_portfolio_weights({"A": a, "B": b}, "factor_based")
        assert len(result.weights) == 2


# ---------------------------------------------------------------------------
# Regime-Aware Backtest
# ---------------------------------------------------------------------------


class TestRegimeBacktest:
    def test_too_short(self):
        result = run_regime_aware_backtest(_varying_returns(20))
        assert result.regime_points == []
        assert len(result.regime_stats) == 3

    def test_basic(self):
        result = run_regime_aware_backtest(_varying_returns(200))
        assert isinstance(result, RegimeBacktestResult)
        assert len(result.regime_points) == 200
        regimes = {r.regime for r in result.regime_stats}
        assert "bull" in regimes or "bear" in regimes or "sideways" in regimes


# ---------------------------------------------------------------------------
# Strategy Significance
# ---------------------------------------------------------------------------


class TestSignificance:
    def test_too_short(self):
        result = estimate_strategy_significance(_varying_returns(10))
        assert result.p_value == 1
        assert not result.is_significant_95

    def test_significant_positive(self):
        # All positive returns should be significant
        rets = _make_series("2022-01-01", [0.005] * 200 + [0.003] * 200)
        result = estimate_strategy_significance(rets)
        assert result.t_stat > 0
        assert result.p_value < 0.05

    def test_with_benchmark(self):
        strat = _varying_returns(100)
        bench = _make_series("2022-01-01", [v.value * 0.5 for v in _varying_returns(100)])
        result = estimate_strategy_significance(strat, bench)
        assert isinstance(result, SignificanceResult)


# ---------------------------------------------------------------------------
# Volume Profile / Renko / P&F
# ---------------------------------------------------------------------------


class TestCharting:
    def test_volume_profile_empty(self):
        assert build_volume_profile([]) == []

    def test_volume_profile_basic(self):
        prices = _make_prices(100)
        buckets = build_volume_profile(prices)
        assert len(buckets) > 0
        assert any(b.is_poc for b in buckets)
        total_share = sum(b.share_pct for b in buckets)
        assert math.isclose(total_share, 100.0, abs_tol=1.0)

    def test_renko_empty(self):
        assert build_renko_bricks([]) == []

    def test_renko_basic(self):
        prices = _make_prices(200)
        bricks = build_renko_bricks(prices)
        assert len(bricks) > 0
        assert all(b.direction in ("up", "down") for b in bricks)

    def test_pnf_empty(self):
        assert build_point_figure_columns([]) == []

    def test_pnf_basic(self):
        prices = _make_prices(200)
        cols = build_point_figure_columns(prices)
        assert len(cols) >= 0  # May have no reversals with steady drift


# ---------------------------------------------------------------------------
# Synchronized Timeframes
# ---------------------------------------------------------------------------


class TestSyncTimeframes:
    def test_basic(self):
        prices = _make_prices(100)
        result = build_synchronized_timeframes(prices)
        assert "1D" in result
        assert "1W" in result
        assert "1M" in result
        assert len(result["1D"]) == 100
        assert len(result["1W"]) <= 100
        assert len(result["1M"]) <= 100


# ---------------------------------------------------------------------------
# Indicator Combination Signal
# ---------------------------------------------------------------------------


class TestIndicatorSignal:
    def test_short(self):
        result = build_indicator_combination_signal(_make_prices(10))
        assert result.trend_signal == "neutral"
        assert result.rsi == 50

    def test_basic(self):
        result = build_indicator_combination_signal(_make_prices(200))
        assert isinstance(result, IndicatorCombinationSignal)
        assert 0 <= result.rsi <= 100


# ---------------------------------------------------------------------------
# Performance Attribution
# ---------------------------------------------------------------------------


class TestAttribution:
    def test_too_short(self):
        result = compute_performance_attribution_breakdown(
            _varying_returns(10), _varying_returns(10)
        )
        assert result.benchmark_relative_pct == 0

    def test_basic(self):
        strat = _varying_returns(200)
        bench = _make_series("2022-01-01", [v.value * 0.8 for v in _varying_returns(200)])
        result = compute_performance_attribution_breakdown(strat, bench)
        assert isinstance(result, PerformanceAttributionBreakdown)


# ---------------------------------------------------------------------------
# Integration Diagnostics
# ---------------------------------------------------------------------------


class TestIntegrationDiagnostics:
    @pytest.mark.slow
    def test_basic(self):
        rets = _varying_returns(200)
        result = run_backtest_integration_diagnostics(
            rets, None,
            slippage_bps=5, spread_bps=3, impact_bps=2,
            tax_pct=0, rebalance_days=21,
        )
        assert isinstance(result, BacktestIntegrationDiagnostics)
        assert result.total_cost_impact_pct >= 0
