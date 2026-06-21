"""Unit tests for analytics/portfolio_metrics.py.

Covers the standalone calculate_* functions, the PortfolioAnalytics class,
and the risk-free-rate resolution chain. This file previously had zero
direct unit tests despite being the most user-facing analytics module.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from bist_quant.analytics import portfolio_metrics as pm
from bist_quant.analytics.portfolio_metrics import (
    PortfolioAnalytics,
    calculate_alpha,
    calculate_beta,
    calculate_calmar_ratio,
    calculate_cvar,
    calculate_information_ratio,
    calculate_max_drawdown,
    calculate_rolling_metrics,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_var,
    get_default_risk_free_rate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_rf_cache():
    """Clear the module-level risk-free-rate cache before each test."""
    pm._RISK_FREE_RATE_CACHE = None
    yield
    pm._RISK_FREE_RATE_CACHE = None


@pytest.fixture
def positive_returns() -> pd.Series:
    """A deterministic returns series with positive drift (~25% annual)."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    returns = pd.Series(rng.normal(0.0009, 0.01, 252), index=dates)
    returns.name = "returns"
    return returns


@pytest.fixture
def mixed_returns() -> pd.Series:
    """A deterministic returns series with both gains and losses."""
    rng = np.random.default_rng(123)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    returns = pd.Series(rng.normal(0.0, 0.015, 252), index=dates)
    returns.name = "returns"
    return returns


@pytest.fixture
def benchmark_returns() -> pd.Series:
    """A benchmark series correlated with positive_returns."""
    rng = np.random.default_rng(99)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    returns = pd.Series(rng.normal(0.0003, 0.008, 252), index=dates)
    returns.name = "benchmark"
    return returns


# ---------------------------------------------------------------------------
# Risk-free rate resolution
# ---------------------------------------------------------------------------

class TestRiskFreeRate:
    """Tests for the risk-free-rate resolution chain."""

    def test_get_default_uses_provider(self) -> None:
        """get_default_risk_free_rate should resolve via FixedIncomeProvider."""
        with patch.object(pm, "FixedIncomeProvider") as mock_provider:
            mock_provider.return_value.get_risk_free_rate.return_value = 45.0
            rate = get_default_risk_free_rate()
        # 45.0 > 1.5 threshold → coerced to decimal 0.45
        assert rate == pytest.approx(0.45)

    def test_get_default_caches(self) -> None:
        """Second call should hit the cache, not the provider."""
        with patch.object(pm, "FixedIncomeProvider") as mock_provider:
            mock_provider.return_value.get_risk_free_rate.return_value = 30.0
            r1 = get_default_risk_free_rate()
            r2 = get_default_risk_free_rate()
        assert r1 == r2
        # Provider called only once due to caching
        assert mock_provider.return_value.get_risk_free_rate.call_count == 1

    def test_get_default_falls_back_on_error(self) -> None:
        """If provider raises, should fall back to RISK_FREE_RATE constant."""
        with patch.object(pm, "FixedIncomeProvider") as mock_provider:
            mock_provider.return_value.get_risk_free_rate.side_effect = RuntimeError("boom")
            rate = get_default_risk_free_rate()
        assert rate == pytest.approx(float(pm.RISK_FREE_RATE))

    def test_coerce_rate_percent_to_decimal(self) -> None:
        """Values above 1.5 are treated as percent and divided by 100."""
        assert pm._coerce_rate_to_decimal(45.0) == pytest.approx(0.45)
        assert pm._coerce_rate_to_decimal(28) == pytest.approx(0.28)

    def test_coerce_rate_decimal_passthrough(self) -> None:
        """Values at or below 1.5 pass through unchanged."""
        assert pm._coerce_rate_to_decimal(0.05) == pytest.approx(0.05)
        assert pm._coerce_rate_to_decimal(0.45) == pytest.approx(0.45)

    def test_coerce_rate_none_and_invalid(self) -> None:
        """None and non-finite values return None."""
        assert pm._coerce_rate_to_decimal(None) is None
        assert pm._coerce_rate_to_decimal(float("nan")) is None
        assert pm._coerce_rate_to_decimal("abc") is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Sharpe ratio
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    """Tests for calculate_sharpe_ratio."""

    def test_positive_drift_positive_sharpe(self, positive_returns) -> None:
        """Positive drift should yield a positive Sharpe ratio."""
        sharpe = calculate_sharpe_ratio(positive_returns, risk_free_rate=0.05)
        assert sharpe > 0

    def test_explicit_rf(self, positive_returns) -> None:
        """Higher risk-free rate should lower the Sharpe ratio."""
        low_rf = calculate_sharpe_ratio(positive_returns, risk_free_rate=0.01)
        high_rf = calculate_sharpe_ratio(positive_returns, risk_free_rate=0.40)
        assert low_rf > high_rf

    def test_constant_returns_returns_zero(self) -> None:
        """Zero-variance returns should return 0.0 (avoid division by zero)."""
        dates = pd.date_range("2023-01-01", periods=50, freq="B")
        const = pd.Series(0.001, index=dates)
        assert calculate_sharpe_ratio(const, risk_free_rate=0.05) == 0.0

    def test_empty_returns_zero(self) -> None:
        """Empty series should return 0.0."""
        assert calculate_sharpe_ratio(pd.Series(dtype=float), risk_free_rate=0.05) == 0.0

    def test_rf_percent_coerced(self, positive_returns) -> None:
        """Risk-free rate given as percent (e.g. 45.0) is coerced to 0.45."""
        sharpe_pct = calculate_sharpe_ratio(positive_returns, risk_free_rate=45.0)
        sharpe_dec = calculate_sharpe_ratio(positive_returns, risk_free_rate=0.45)
        assert sharpe_pct == pytest.approx(sharpe_dec)


# ---------------------------------------------------------------------------
# Sortino ratio
# ---------------------------------------------------------------------------

class TestSortinoRatio:
    """Tests for calculate_sortino_ratio."""

    def test_sortino_positive_for_positive_drift(self, positive_returns) -> None:
        assert calculate_sortino_ratio(positive_returns, risk_free_rate=0.05) > 0

    def test_sortino_geq_sharpe(self, mixed_returns) -> None:
        """Sortino (downside-only vol) should generally be >= Sharpe (total vol)."""
        sharpe = calculate_sharpe_ratio(mixed_returns, risk_free_rate=0.05)
        sortino = calculate_sortino_ratio(mixed_returns, risk_free_rate=0.05)
        # Downside deviation <= total std → Sortino >= Sharpe (for positive drift)
        assert sortino >= sharpe

    def test_sortino_all_positive_returns(self) -> None:
        """If there are fewer than 2 negative returns, Sortino returns 0.0."""
        dates = pd.date_range("2023-01-01", periods=50, freq="B")
        all_pos = pd.Series(np.full(50, 0.001), index=dates)
        assert calculate_sortino_ratio(all_pos, risk_free_rate=0.05) == 0.0


# ---------------------------------------------------------------------------
# Max drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    """Tests for calculate_max_drawdown."""

    def test_known_drawdown(self) -> None:
        """Prices: 100 → 120 → 90 → 110. Max DD = (90-120)/120 = -0.25."""
        dates = pd.date_range("2023-01-01", periods=4, freq="D")
        prices = pd.Series([100.0, 120.0, 90.0, 110.0], index=dates)
        dd, peak, trough = calculate_max_drawdown(prices=prices)
        assert dd == pytest.approx(-0.25)
        assert peak == dates[1]   # 120.0 at index 1
        assert trough == dates[2]  # 90.0 at index 2

    def test_no_drawdown(self) -> None:
        """Strictly increasing prices → drawdown is 0."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        prices = pd.Series(np.linspace(100, 110, 10), index=dates)
        dd, _, _ = calculate_max_drawdown(prices=prices)
        assert dd == pytest.approx(0.0)

    def test_from_returns(self) -> None:
        """Verify drawdown computed from returns matches prices input."""
        dates = pd.date_range("2023-01-01", periods=4, freq="D")
        prices = pd.Series([100.0, 120.0, 90.0, 110.0], index=dates)
        returns = prices.pct_change().dropna()
        dd_returns, _, _ = calculate_max_drawdown(returns=returns)
        dd_prices, _, _ = calculate_max_drawdown(prices=prices)
        assert dd_returns == pytest.approx(dd_prices, abs=1e-10)

    def test_no_input_returns_zero(self) -> None:
        """Neither returns nor prices → (0.0, None, None)."""
        dd, peak, trough = calculate_max_drawdown()
        assert dd == 0.0
        assert peak is None
        assert trough is None


# ---------------------------------------------------------------------------
# Beta / Alpha / Information Ratio
# ---------------------------------------------------------------------------

class TestBetaAlpha:
    """Tests for calculate_beta, calculate_alpha, calculate_information_ratio."""

    def test_beta_of_identical_series(self, positive_returns) -> None:
        """Beta of a series with itself should be ~1.0."""
        beta = calculate_beta(positive_returns, positive_returns)
        assert beta == pytest.approx(1.0, abs=1e-6)

    def test_beta_positive_correlation(self, positive_returns, benchmark_returns) -> None:
        beta = calculate_beta(positive_returns, benchmark_returns)
        assert isinstance(beta, float)

    def test_beta_too_short_returns_default(self) -> None:
        """Fewer than 30 aligned points → default beta of 1.0."""
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        port = pd.Series(np.random.default_rng(1).normal(0, 0.01, 20), index=dates)
        bench = pd.Series(np.random.default_rng(2).normal(0, 0.01, 20), index=dates)
        assert calculate_beta(port, bench) == 1.0

    def test_alpha_returns_float(self, positive_returns, benchmark_returns) -> None:
        alpha = calculate_alpha(positive_returns, benchmark_returns, risk_free_rate=0.05)
        assert isinstance(alpha, float)

    def test_information_ratio_returns_float(
        self, positive_returns, benchmark_returns,
    ) -> None:
        ir = calculate_information_ratio(positive_returns, benchmark_returns)
        assert isinstance(ir, float)

    def test_information_ratio_too_short(self) -> None:
        """Fewer than 30 aligned points → IR of 0.0."""
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        port = pd.Series(np.zeros(20), index=dates)
        bench = pd.Series(np.zeros(20), index=dates)
        assert calculate_information_ratio(port, bench) == 0.0


# ---------------------------------------------------------------------------
# Calmar / VaR / CVaR
# ---------------------------------------------------------------------------

class TestCalmarVarCvar:
    """Tests for calculate_calmar_ratio, calculate_var, calculate_cvar."""

    def test_calmar_positive_for_positive_drift(self, positive_returns) -> None:
        calmar = calculate_calmar_ratio(positive_returns)
        assert isinstance(calmar, float)

    def test_calmar_zero_drawdown(self) -> None:
        """No drawdown → Calmar returns 0.0."""
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        # Always-slightly-positive returns → strictly increasing equity
        returns = pd.Series(np.full(50, 0.001), index=dates)
        assert calculate_calmar_ratio(returns) == 0.0

    def test_var_historical_positive(self, mixed_returns) -> None:
        """Historical VaR should be a positive number (potential loss)."""
        var = calculate_var(mixed_returns, confidence=0.95, method="historical")
        assert var > 0

    def test_var_parametric_positive(self, mixed_returns) -> None:
        var = calculate_var(mixed_returns, confidence=0.95, method="parametric")
        assert var > 0

    def test_var_parametric_geq_at_higher_confidence(self, mixed_returns) -> None:
        """99% VaR should be at least as large as 95% VaR."""
        var_95 = calculate_var(mixed_returns, confidence=0.95, method="parametric")
        var_99 = calculate_var(mixed_returns, confidence=0.99, method="parametric")
        assert var_99 >= var_95

    def test_cvar_geq_var(self, mixed_returns) -> None:
        """CVaR (expected shortfall) should be >= VaR at the same confidence."""
        var = calculate_var(mixed_returns, confidence=0.95)
        cvar = calculate_cvar(mixed_returns, confidence=0.95)
        assert cvar >= var

    def test_var_empty_returns_zero(self) -> None:
        assert calculate_var(pd.Series(dtype=float)) == 0.0

    def test_cvar_empty_returns_zero(self) -> None:
        assert calculate_cvar(pd.Series(dtype=float)) == 0.0


# ---------------------------------------------------------------------------
# Rolling metrics
# ---------------------------------------------------------------------------

class TestRollingMetrics:
    """Tests for calculate_rolling_metrics."""

    def test_returns_dataframe(self, positive_returns) -> None:
        df = calculate_rolling_metrics(positive_returns, window=63, risk_free_rate=0.05)
        assert isinstance(df, pd.DataFrame)
        assert "volatility" in df.columns
        assert "sharpe" in df.columns
        assert "max_drawdown" in df.columns

    def test_with_benchmark_adds_beta(
        self, positive_returns, benchmark_returns,
    ) -> None:
        df = calculate_rolling_metrics(
            positive_returns, window=63,
            benchmark_returns=benchmark_returns, risk_free_rate=0.05,
        )
        assert "beta" in df.columns

    def test_no_benchmark_no_beta_column(self, positive_returns) -> None:
        df = calculate_rolling_metrics(positive_returns, window=63, risk_free_rate=0.05)
        assert "beta" not in df.columns

    def test_rolling_values_nan_before_window(self, positive_returns) -> None:
        """First window-1 rows should be NaN."""
        df = calculate_rolling_metrics(positive_returns, window=63, risk_free_rate=0.05)
        assert df["volatility"].iloc[:62].isna().all()
        assert df["volatility"].iloc[62:].notna().any()


# ---------------------------------------------------------------------------
# PortfolioAnalytics class
# ---------------------------------------------------------------------------

class TestPortfolioAnalyticsClass:
    """Tests for the PortfolioAnalytics OO wrapper."""

    def test_from_returns(self, positive_returns) -> None:
        pa = PortfolioAnalytics(positive_returns, risk_free_rate=0.05)
        assert len(pa.returns) == 252
        assert pa.risk_free_rate == pytest.approx(0.05)
        assert pa.name == "Portfolio"

    def test_sharpe_property_matches_function(self, positive_returns) -> None:
        """The .sharpe_ratio property should match the standalone function."""
        pa = PortfolioAnalytics(positive_returns, risk_free_rate=0.05)
        direct = calculate_sharpe_ratio(positive_returns, risk_free_rate=0.05)
        assert pa.sharpe_ratio == pytest.approx(direct)

    def test_sortino_property_matches_function(self, positive_returns) -> None:
        pa = PortfolioAnalytics(positive_returns, risk_free_rate=0.05)
        direct = calculate_sortino_ratio(positive_returns, risk_free_rate=0.05)
        assert pa.sortino_ratio == pytest.approx(direct)

    def test_max_drawdown_property_matches_function(self, positive_returns) -> None:
        pa = PortfolioAnalytics(positive_returns, risk_free_rate=0.05)
        direct_dd, _, _ = calculate_max_drawdown(returns=positive_returns)
        assert pa.max_drawdown == pytest.approx(direct_dd)

    def test_cache_is_used(self, positive_returns) -> None:
        """Accessing a property twice should hit the cache."""
        pa = PortfolioAnalytics(positive_returns, risk_free_rate=0.05)
        v1 = pa.sharpe_ratio
        v2 = pa.sharpe_ratio
        assert v1 == v2
        assert "sharpe" in pa._metrics_cache

    def test_beta_none_without_benchmark(self, positive_returns) -> None:
        pa = PortfolioAnalytics(positive_returns, risk_free_rate=0.05)
        assert pa.beta is None
        assert pa.alpha is None
        assert pa.information_ratio is None

    def test_beta_with_benchmark(
        self, positive_returns, benchmark_returns,
    ) -> None:
        pa = PortfolioAnalytics(
            positive_returns,
            benchmark_returns=benchmark_returns,
            risk_free_rate=0.05,
        )
        assert pa.beta is not None
        assert isinstance(pa.beta, float)

    def test_get_all_metrics_keys(self, positive_returns) -> None:
        pa = PortfolioAnalytics(positive_returns, risk_free_rate=0.05)
        metrics = pa.get_all_metrics()
        expected_keys = {
            "name", "start_date", "end_date", "trading_days",
            "cagr", "total_return", "volatility", "downside_volatility",
            "sharpe_ratio", "sortino_ratio", "calmar_ratio",
            "max_drawdown", "var_95", "cvar_95",
        }
        assert expected_keys.issubset(metrics.keys())

    def test_get_all_metrics_with_benchmark(
        self, positive_returns, benchmark_returns,
    ) -> None:
        pa = PortfolioAnalytics(
            positive_returns,
            benchmark_returns=benchmark_returns,
            risk_free_rate=0.05,
        )
        metrics = pa.get_all_metrics()
        assert "beta" in metrics
        assert "alpha" in metrics
        assert "information_ratio" in metrics

    def test_from_holdings(self) -> None:
        """from_holdings builds portfolio returns from a price panel."""
        dates = pd.date_range("2023-01-01", periods=60, freq="B")
        rng = np.random.default_rng(7)
        close_df = pd.DataFrame({
            "THYAO": 100 * np.cumprod(1 + rng.normal(0.001, 0.01, 60)),
            "GARAN": 50 * np.cumprod(1 + rng.normal(0.0005, 0.012, 60)),
        }, index=dates)
        pa = PortfolioAnalytics.from_holdings(
            holdings={"THYAO": 100, "GARAN": 200},
            close_df=close_df,
        )
        assert len(pa.returns) > 0
        assert pa.sharpe_ratio is not None

    def test_from_holdings_missing_symbol_raises(self) -> None:
        """A symbol not in close_df should raise ValueError."""
        dates = pd.date_range("2023-01-01", periods=10, freq="B")
        close_df = pd.DataFrame({"THYAO": np.linspace(100, 110, 10)}, index=dates)
        with pytest.raises(ValueError, match="No holdings found"):
            PortfolioAnalytics.from_holdings(
                holdings={"MISSING": 100}, close_df=close_df,
            )

    def test_from_equity_curve(self) -> None:
        """from_equity_curve derives returns from cumulative values."""
        dates = pd.date_range("2023-01-01", periods=60, freq="B")
        equity = pd.Series(
            np.linspace(100, 120, 60), index=dates, name="equity",
        )
        pa = PortfolioAnalytics.from_equity_curve(equity)
        assert len(pa.returns) == 59  # one less than equity (pct_change dropna)
        assert pa.returns.iloc[0] > 0  # upward sloping

    def test_from_equity_curve_with_benchmark(self) -> None:
        dates = pd.date_range("2023-01-01", periods=60, freq="B")
        equity = pd.Series(np.linspace(100, 120, 60), index=dates)
        bench = pd.Series(np.linspace(100, 115, 60), index=dates)
        pa = PortfolioAnalytics.from_equity_curve(equity, benchmark_curve=bench)
        assert pa.benchmark_returns is not None
        assert pa.beta is not None

    def test_get_rolling_metrics(self, positive_returns) -> None:
        pa = PortfolioAnalytics(positive_returns, risk_free_rate=0.05)
        df = pa.get_rolling_metrics(window=63)
        assert isinstance(df, pd.DataFrame)
        assert "volatility" in df.columns

    def test_risk_free_rate_percent_coerced(self, positive_returns) -> None:
        """Risk-free rate passed as percent (e.g. 28) is coerced to decimal."""
        pa = PortfolioAnalytics(positive_returns, risk_free_rate=28)
        assert pa.risk_free_rate == pytest.approx(0.28)
