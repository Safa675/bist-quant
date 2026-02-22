"""
Unit tests for bist_quant.analytics.core_metrics.

Validates the Python port against the expected behaviour of the original
TypeScript implementation in bist_quant.ai/src/lib/analytics/quant.ts.
"""

from __future__ import annotations

import math

import pytest

from bist_quant.analytics.core_metrics import (
    AllocationOptimizationResult,
    DiversificationScore,
    MonteCarloSummary,
    PerformanceMetrics,
    RiskContributionResult,
    RollingMetricPoint,
    SeriesPoint,
    StressScenarioResult,
    TransactionCostResult,
    WalkForwardSplit,
    align_series_by_date,
    align_two_series,
    apply_transaction_costs,
    build_correlation_matrix,
    build_rolling_correlation,
    build_rolling_metrics,
    build_walk_forward_analysis,
    combine_return_series_equal_weight,
    compute_diversification_score,
    compute_max_drawdown,
    compute_performance_metrics,
    compute_risk_contribution,
    correlation,
    covariance,
    curve_to_returns,
    mean,
    normal_cdf,
    optimize_mean_variance_allocation,
    quantile,
    returns_to_equity,
    run_monte_carlo_bootstrap,
    run_stress_scenarios,
    sample_std_dev,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_points(start_date: str, values: list[float]) -> list[SeriesPoint]:
    """Create a list of SeriesPoints with sequential dates starting from start_date."""
    year, month, day = start_date.split("-")
    y, m, d = int(year), int(month), int(day)
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


def _daily_returns(n: int = 300, daily_ret: float = 0.0003) -> list[SeriesPoint]:
    """Generate n days of constant daily returns for testing."""
    return _make_points("2022-01-01", [daily_ret] * n)


def _equity_curve(n: int = 300, daily_ret: float = 0.0003) -> list[SeriesPoint]:
    """Generate an equity curve from constant daily returns."""
    pts: list[SeriesPoint] = []
    eq = 1.0
    year, month, day = 2022, 1, 1
    for _ in range(n):
        eq *= 1 + daily_ret
        pts.append(SeriesPoint(date=f"{year:04d}-{month:02d}-{day:02d}", value=eq))
        day += 1
        if day > 28:
            day = 1
            month += 1
            if month > 12:
                month = 1
                year += 1
    return pts


# ---------------------------------------------------------------------------
# Math Utilities
# ---------------------------------------------------------------------------


class TestMathUtils:
    def test_mean_empty(self):
        assert mean([]) == 0.0

    def test_mean_values(self):
        assert math.isclose(mean([1, 2, 3, 4, 5]), 3.0)

    def test_sample_std_dev_too_few(self):
        assert sample_std_dev([]) == 0.0
        assert sample_std_dev([5]) == 0.0

    def test_sample_std_dev(self):
        result = sample_std_dev([2, 4, 4, 4, 5, 5, 7, 9])
        # Sample std dev (Bessel-corrected) of this set is ~2.138
        assert math.isclose(result, 2.138, abs_tol=0.01)

    def test_covariance_mismatched_length(self):
        assert covariance([1, 2], [1, 2, 3]) == 0.0

    def test_covariance_identical(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        cov = covariance(vals, vals)
        std = sample_std_dev(vals)
        assert math.isclose(cov, std**2, rel_tol=0.01)

    def test_correlation_identical(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert math.isclose(correlation(vals, vals), 1.0, abs_tol=0.001)

    def test_correlation_opposite(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [5.0, 4.0, 3.0, 2.0, 1.0]
        assert math.isclose(correlation(a, b), -1.0, abs_tol=0.001)

    def test_quantile_edge_cases(self):
        vals = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert quantile(vals, 0.0) == 10.0
        assert quantile(vals, 1.0) == 50.0
        assert quantile([], 0.5) == 0.0
        assert math.isclose(quantile(vals, 0.5), 30.0)

    def test_normal_cdf_center(self):
        assert math.isclose(normal_cdf(0), 0.5, abs_tol=0.001)

    def test_normal_cdf_positive(self):
        # N(1.96) ≈ 0.975
        assert math.isclose(normal_cdf(1.96), 0.975, abs_tol=0.005)

    def test_normal_cdf_negative(self):
        # N(-1.96) ≈ 0.025
        assert math.isclose(normal_cdf(-1.96), 0.025, abs_tol=0.005)


# ---------------------------------------------------------------------------
# Series Functions
# ---------------------------------------------------------------------------


class TestSeriesFunctions:
    def test_curve_to_returns_empty(self):
        assert curve_to_returns([]) == []

    def test_curve_to_returns_single(self):
        assert curve_to_returns([SeriesPoint("2022-01-01", 100)]) == []

    def test_curve_to_returns_basic(self):
        curve = [
            SeriesPoint("2022-01-01", 100),
            SeriesPoint("2022-01-02", 110),
            SeriesPoint("2022-01-03", 99),
        ]
        rets = curve_to_returns(curve)
        assert len(rets) == 2
        assert math.isclose(rets[0].value, 0.1, abs_tol=0.001)
        assert math.isclose(rets[1].value, -0.1, abs_tol=0.001)

    def test_returns_to_equity(self):
        rets = [
            SeriesPoint("2022-01-01", 0.1),
            SeriesPoint("2022-01-02", -0.05),
            SeriesPoint("2022-01-03", 0.02),
        ]
        equity = returns_to_equity(rets, 1.0)
        assert len(equity) == 3
        assert math.isclose(equity[0].value, 1.1, abs_tol=0.001)
        assert math.isclose(equity[1].value, 1.045, abs_tol=0.001)

    def test_round_trip(self):
        """curve -> returns -> equity should approximately recover curve."""
        curve = _equity_curve(100, 0.001)
        rets = curve_to_returns(curve)
        recovered = returns_to_equity(rets, curve[0].value)
        # Last values should be close
        assert math.isclose(
            recovered[-1].value, curve[-1].value, rel_tol=0.001
        )

    def test_compute_max_drawdown_flat(self):
        curve = _make_points("2022-01-01", [100] * 10)
        assert compute_max_drawdown(curve) == 0.0

    def test_compute_max_drawdown_basic(self):
        curve = [
            SeriesPoint("2022-01-01", 100),
            SeriesPoint("2022-01-02", 120),
            SeriesPoint("2022-01-03", 90),
            SeriesPoint("2022-01-04", 110),
        ]
        dd = compute_max_drawdown(curve)
        # Max drawdown from 120 to 90 = -25%
        assert math.isclose(dd, -25.0, abs_tol=0.01)


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


class TestAlignment:
    def test_align_two_series_overlap(self):
        left = [SeriesPoint("2022-01-01", 1), SeriesPoint("2022-01-02", 2), SeriesPoint("2022-01-03", 3)]
        right = [SeriesPoint("2022-01-02", 20), SeriesPoint("2022-01-03", 30), SeriesPoint("2022-01-04", 40)]
        result = align_two_series(left, right)
        assert result["dates"] == ["2022-01-02", "2022-01-03"]
        assert result["left"] == [2, 3]
        assert result["right"] == [20, 30]

    def test_align_series_by_date_common(self):
        series_map = {
            "A": [SeriesPoint("2022-01-01", 1), SeriesPoint("2022-01-02", 2)],
            "B": [SeriesPoint("2022-01-02", 20), SeriesPoint("2022-01-03", 30)],
        }
        result = align_series_by_date(series_map)
        assert result["dates"] == ["2022-01-02"]
        assert result["values"]["A"] == [2]
        assert result["values"]["B"] == [20]

    def test_align_series_empty(self):
        result = align_series_by_date({})
        assert result["dates"] == []
        assert result["values"] == {}


# ---------------------------------------------------------------------------
# Performance Metrics
# ---------------------------------------------------------------------------


class TestPerformanceMetrics:
    def test_empty_returns(self):
        m = compute_performance_metrics([])
        assert m.observations == 0
        assert m.total_return == 0
        assert m.sharpe == 0

    def test_positive_returns(self):
        # Use varying returns (not constant) so std dev > 0
        values = [0.001, 0.002, 0.0005, 0.0015, 0.003] * 50
        rets = _make_points("2022-01-01", values)
        m = compute_performance_metrics(rets)
        assert m.observations == 250
        assert m.cagr > 0  # Positive CAGR
        assert m.sharpe > 0  # Positive Sharpe
        assert m.win_rate == 100.0  # All positive

    def test_mixed_returns(self):
        values = [0.01, -0.005, 0.003, -0.002, 0.008] * 50
        rets = _make_points("2022-01-01", values)
        m = compute_performance_metrics(rets)
        assert m.observations == 250
        assert m.win_rate > 0
        assert m.win_rate < 100
        assert m.var_95 < 0  # Negative VaR
        assert m.profit_factor is not None
        assert m.profit_factor > 0

    def test_benchmark_beta_alpha(self):
        # Use varying returns so covariance / variance are non-zero
        rets_vals = [0.001, -0.0005, 0.002, -0.001, 0.0015] * 50
        bench_vals = [0.0008, -0.0003, 0.0015, -0.0008, 0.001] * 50
        rets = _make_points("2022-01-01", rets_vals)
        bench = _make_points("2022-01-01", bench_vals)
        m = compute_performance_metrics(rets, benchmark_returns=bench)
        assert m.beta is not None
        assert m.alpha_annual is not None

    def test_with_equity_curve(self):
        curve = _equity_curve(100, 0.002)
        rets = curve_to_returns(curve)
        m = compute_performance_metrics(rets, equity_curve=curve)
        assert m.total_return > 0
        assert m.cagr > 0


# ---------------------------------------------------------------------------
# Rolling Metrics
# ---------------------------------------------------------------------------


class TestRollingMetrics:
    def test_too_short(self):
        curve = _equity_curve(20)
        assert build_rolling_metrics(curve) == []

    def test_basic(self):
        curve = _equity_curve(100, 0.001)
        result = build_rolling_metrics(curve)
        assert len(result) > 0
        assert all(isinstance(r, RollingMetricPoint) for r in result)
        # First few should have values
        non_null = [r for r in result if r.rolling_return_63d is not None]
        assert len(non_null) > 0


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


class TestCorrelation:
    def test_correlation_matrix_identical(self):
        # Use varying data so std dev > 0 and correlation is meaningful
        values = [0.01, -0.005, 0.003, -0.002, 0.008] * 20
        series = _make_points("2022-01-01", values)
        matrix = build_correlation_matrix({"A": series, "B": series})
        assert matrix["A"]["A"] == 1.0
        assert matrix["B"]["B"] == 1.0
        assert matrix["A"]["B"] is not None
        assert math.isclose(matrix["A"]["B"], 1.0, abs_tol=0.01)

    def test_correlation_matrix_too_short(self):
        short = [SeriesPoint("2022-01-01", 0.01)]
        matrix = build_correlation_matrix({"A": short, "B": short})
        assert matrix["A"]["B"] is None

    def test_rolling_correlation(self):
        a = _daily_returns(100, 0.001)
        b = _daily_returns(100, 0.002)
        result = build_rolling_correlation(a, b, window=20)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Walk-Forward
# ---------------------------------------------------------------------------


class TestWalkForward:
    def test_too_short(self):
        rets = _daily_returns(30)
        assert build_walk_forward_analysis(rets) == []

    def test_basic(self):
        rets = _daily_returns(300, 0.0005)
        splits = build_walk_forward_analysis(rets, splits=3)
        assert len(splits) > 0
        assert all(isinstance(s, WalkForwardSplit) for s in splits)
        for s in splits:
            assert s.split >= 1
            assert s.train_start < s.train_end
            assert s.test_start < s.test_end


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------


class TestMonteCarlo:
    def test_empty(self):
        result = run_monte_carlo_bootstrap([])
        assert result.iterations == 0
        assert result.paths == []

    def test_basic(self):
        rets = _daily_returns(252, 0.0003)
        result = run_monte_carlo_bootstrap(rets, iterations=200, horizon_days=126)
        assert result.iterations == 200
        assert result.horizon_days == 126
        assert result.expected_terminal > 0
        assert len(result.paths) == 126
        assert result.terminal_p05 <= result.terminal_p50 <= result.terminal_p95

    def test_deterministic(self):
        """Same seed gives same results."""
        rets = _daily_returns(100, 0.001)
        r1 = run_monte_carlo_bootstrap(rets, iterations=100, seed=123)
        r2 = run_monte_carlo_bootstrap(rets, iterations=100, seed=123)
        assert math.isclose(r1.expected_terminal, r2.expected_terminal)
        assert math.isclose(r1.probability_of_loss, r2.probability_of_loss)


# ---------------------------------------------------------------------------
# Stress Scenarios
# ---------------------------------------------------------------------------


class TestStressScenarios:
    def test_empty(self):
        assert run_stress_scenarios([]) == []

    def test_basic(self):
        values = [0.01, -0.02, 0.005, -0.03, 0.015, -0.01, 0.002] * 30
        rets = _make_points("2022-01-01", values)
        results = run_stress_scenarios(rets)
        assert len(results) == 4
        names = {r.name for r in results}
        assert "Historical Worst Day" in names
        assert "Tail Risk Shock (VaR95)" in names


# ---------------------------------------------------------------------------
# Transaction Costs
# ---------------------------------------------------------------------------


class TestTransactionCosts:
    def test_empty(self):
        result = apply_transaction_costs([])
        assert result.adjusted == []
        assert result.rebalance_events == 0

    def test_basic(self):
        rets = _daily_returns(100, 0.001)
        result = apply_transaction_costs(
            rets,
            slippage_bps=5,
            spread_bps=3,
            rebalance_every_days=21,
        )
        assert len(result.adjusted) == 100
        assert result.rebalance_events > 0
        assert result.total_cost_impact_pct > 0
        # Adjusted returns should be smaller
        original_sum = sum(r.value for r in rets)
        adjusted_sum = sum(r.value for r in result.adjusted)
        assert adjusted_sum < original_sum


# ---------------------------------------------------------------------------
# Mean-Variance Optimization
# ---------------------------------------------------------------------------


class TestMeanVarianceOptimization:
    def test_empty(self):
        result = optimize_mean_variance_allocation({})
        assert result.assets == []
        assert result.best_sharpe == 0

    def test_single_asset(self):
        rets = _daily_returns(100, 0.001)
        result = optimize_mean_variance_allocation({"A": rets})
        assert "A" in result.best_weights
        assert math.isclose(result.best_weights["A"], 1.0, abs_tol=0.01)

    def test_two_assets(self):
        a = _make_points("2022-01-01", [0.01, -0.005, 0.003, -0.002, 0.008] * 20)
        b = _make_points("2022-01-01", [-0.005, 0.01, -0.002, 0.003, 0.005] * 20)
        result = optimize_mean_variance_allocation({"A": a, "B": b}, iterations=500)
        assert len(result.assets) == 2
        total_weight = sum(result.best_weights.values())
        assert math.isclose(total_weight, 1.0, abs_tol=0.01)
        assert len(result.frontier) > 0

    def test_deterministic(self):
        a = _daily_returns(100, 0.001)
        b = _daily_returns(100, 0.0005)
        r1 = optimize_mean_variance_allocation({"A": a, "B": b}, iterations=200, seed=42)
        r2 = optimize_mean_variance_allocation({"A": a, "B": b}, iterations=200, seed=42)
        assert r1.best_weights == r2.best_weights


# ---------------------------------------------------------------------------
# Risk Contribution
# ---------------------------------------------------------------------------


class TestRiskContribution:
    def test_empty(self):
        result = compute_risk_contribution({}, {})
        assert result.contribution_pct == {}

    def test_basic(self):
        a = _make_points("2022-01-01", [0.01, -0.005, 0.003] * 30)
        b = _make_points("2022-01-01", [-0.003, 0.008, -0.001] * 30)
        result = compute_risk_contribution(
            {"A": a, "B": b},
            {"A": 0.6, "B": 0.4},
        )
        assert "A" in result.contribution_pct
        assert "B" in result.contribution_pct
        total = sum(result.contribution_pct.values())
        assert math.isclose(total, 100.0, abs_tol=1.0)


# ---------------------------------------------------------------------------
# Combination / Diversification
# ---------------------------------------------------------------------------


class TestCombination:
    def test_combine_equal_weight_empty(self):
        assert combine_return_series_equal_weight({}) == []

    def test_combine_equal_weight(self):
        a = _make_points("2022-01-01", [0.01, 0.02, 0.03])
        b = _make_points("2022-01-01", [0.03, 0.02, 0.01])
        combined = combine_return_series_equal_weight({"A": a, "B": b})
        assert len(combined) == 3
        assert math.isclose(combined[0].value, 0.02, abs_tol=0.001)
        assert math.isclose(combined[1].value, 0.02, abs_tol=0.001)

    def test_diversification_score_few_assets(self):
        result = compute_diversification_score({"A": {"A": 1.0}})
        assert result.average_pairwise_correlation == 0
        assert result.diversification_ratio == 1.0

    def test_diversification_score(self):
        matrix: dict[str, dict[str, float | None]] = {
            "A": {"A": 1.0, "B": 0.5, "C": 0.3},
            "B": {"A": 0.5, "B": 1.0, "C": 0.2},
            "C": {"A": 0.3, "B": 0.2, "C": 1.0},
        }
        result = compute_diversification_score(matrix)
        assert 0 < result.average_pairwise_correlation < 1
        assert result.diversification_ratio > 0
