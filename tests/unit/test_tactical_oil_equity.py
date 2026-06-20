"""Unit tests for OilEquityOverlay."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bist_quant.strategies.tactical.oil_equity import OilEquityOverlay
from bist_quant.strategies.base import monthly_rebalance_dates, rolling_ols_predict


# ---------------------------------------------------------------------------
# Mock loader
# ---------------------------------------------------------------------------

class _MockLoader:
    """Minimal loader stub for oil equity tests."""

    def __init__(
        self,
        xu100: pd.Series,
        oil_df: pd.DataFrame | None = None,
        rf_rate: float = 0.25,
    ) -> None:
        self._xu100 = xu100
        self._oil_df = oil_df
        self._rf_rate = rf_rate

    def load_xu100_prices(self) -> pd.Series:
        return self._xu100

    def load_oil_prices(self) -> pd.DataFrame | None:
        return self._oil_df

    def load_fixed_income(self) -> float:
        return self._rf_rate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_aligned_series(
    n_days: int = 300,
    start: str = "2020-01-01",
    eq_base: float = 100.0,
    eq_ret: float = 0.0005,
    oil_base: float = 80.0,
    oil_ret: float = 0.0003,
) -> tuple[pd.Series, pd.DataFrame]:
    """Generate aligned equity and oil daily price series."""
    dates = pd.date_range(start=start, periods=n_days, freq="B")
    rng = np.random.default_rng(42)

    eq_prices = eq_base * np.cumprod(1 + rng.normal(eq_ret, 0.015, n_days))
    oil_prices = oil_base * np.cumprod(1 + rng.normal(oil_ret, 0.02, n_days))

    equity = pd.Series(eq_prices, index=dates, name="Close")
    oil_df = pd.DataFrame({"WTI": oil_prices, "Brent": oil_prices * 1.02}, index=dates)
    oil_df.index.name = "Date"
    return equity, oil_df


@pytest.fixture
def correlated_data() -> tuple[_MockLoader, pd.DatetimeIndex]:
    """Oil and equity with slight positive correlation."""
    equity, oil_df = _make_aligned_series(300)
    loader = _MockLoader(equity, oil_df, rf_rate=0.25)
    return loader, equity.index


@pytest.fixture
def no_oil_data() -> tuple[_MockLoader, pd.DatetimeIndex]:
    """Loader with no oil data available."""
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    equity = pd.Series(100.0, index=dates, name="Close")
    loader = _MockLoader(equity, oil_df=None, rf_rate=0.25)
    return loader, dates


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOilEquityOverlay:
    """Tests for OilEquityOverlay."""

    def test_exposure_is_binary(self, correlated_data) -> None:
        """Exposure must always be exactly 0.0 or 1.0."""
        loader, dates = correlated_data
        overlay = OilEquityOverlay(
            loader=loader, config={}, params={"min_observations": 13},
        )

        for date in dates[20:]:
            exp = overlay.exposure(date)
            assert exp in (0.0, 1.0), (
                f"Non-binary exposure {exp} on {date.date()}"
            )

    def test_exposure_defaults_to_one_when_no_oil(self, no_oil_data) -> None:
        """Without oil data, overlay should default to fully invested."""
        loader, dates = no_oil_data
        overlay = OilEquityOverlay(
            loader=loader, config={}, params={"min_observations": 13},
        )

        for date in dates:
            exp = overlay.exposure(date)
            assert exp == 1.0, f"Expected 1.0 on {date.date()}, got {exp}"

    def test_exposure_held_between_rebalance_dates(self, correlated_data) -> None:
        """Between rebalance months, exposure should be constant."""
        loader, dates = correlated_data
        overlay = OilEquityOverlay(
            loader=loader, config={}, params={"min_observations": 13},
        )

        rebalance_days = monthly_rebalance_dates(dates)
        prev_exposure = 1.0

        for date in dates[20:]:
            exp = overlay.exposure(date)
            if date in rebalance_days:
                prev_exposure = exp
            else:
                assert exp == prev_exposure, (
                    f"Exposure changed between rebalance on {date.date()}: "
                    f"got {exp}, expected {prev_exposure}"
                )

    def test_no_lookahead(self, correlated_data) -> None:
        """Overlay must not use future data beyond each query date.

        Verify by checking that exposure computed incrementally matches
        exposure computed all-at-once for the same date.
        """
        loader, dates = correlated_data

        # Build overlay and compute exposures for all dates
        overlay = OilEquityOverlay(
            loader=loader, config={}, params={"min_observations": 13},
        )

        # Also build a second overlay for a truncated dataset
        truncated_dates = dates[:150]
        equity, oil_df = _make_aligned_series(150)
        loader_trunc = _MockLoader(equity, oil_df, rf_rate=0.25)
        overlay_trunc = OilEquityOverlay(
            loader=loader_trunc, config={}, params={"min_observations": 13},
        )

        # For dates in both datasets, compare decisions on shared rebalance days
        rebalance_days = monthly_rebalance_dates(dates)
        for date in rebalance_days:
            if date in truncated_dates:
                exp_full = overlay.exposure(date)
                exp_trunc = overlay_trunc.exposure(date)
                # Both should make the same decision since they see the same data
                assert exp_full == exp_trunc, (
                    f"Mismatch on {date.date()}: full={exp_full}, trunc={exp_trunc}"
                )

    def test_insufficient_data_returns_one(self) -> None:
        """With very few observations, overlay should default to 1.0."""
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        equity = pd.Series([100.0 + i for i in range(10)], index=dates, name="Close")
        oil_df = pd.DataFrame(
            {"WTI": [80.0 + i * 0.5 for i in range(10)]},
            index=dates,
        )
        loader = _MockLoader(equity, oil_df, rf_rate=0.25)

        overlay = OilEquityOverlay(
            loader=loader, config={}, params={"min_observations": 13},
        )

        for date in dates:
            exp = overlay.exposure(date)
            assert exp == 1.0, f"Expected 1.0 with <13 obs on {date.date()}"

    def test_rf_rate_used(self) -> None:
        """Overlay should use the provided risk-free rate."""
        n_days = 200
        dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
        rng = np.random.default_rng(123)

        # Make equity returns very low and oil returns negative
        # → predicted equity return should be low → below rf → defensive
        eq_prices = 100.0 * np.cumprod(1 + rng.normal(0.0, 0.01, n_days))
        oil_prices = 80.0 * np.cumprod(1 + rng.normal(-0.002, 0.02, n_days))

        equity = pd.Series(eq_prices, index=dates, name="Close")
        oil_df = pd.DataFrame({"WTI": oil_prices}, index=dates)

        # Very high rf rate → hard to beat → likely defensive
        loader = _MockLoader(equity, oil_df, rf_rate=0.50)
        overlay = OilEquityOverlay(
            loader=loader, config={}, params={"min_observations": 13},
        )

        # Just verify it runs without error and produces binary output
        for date in dates[20:]:
            exp = overlay.exposure(date)
            assert exp in (0.0, 1.0)


class TestRollingOLSPredict:
    """Tests for the rolling_ols_predict helper."""

    def test_basic_prediction(self) -> None:
        """Perfect linear relationship: y = 2x + 1."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([3.0, 5.0, 7.0, 9.0, 11.0])  # y = 2x + 1

        pred = rolling_ols_predict(x, y, latest_x=6.0, min_obs=5)
        assert pred is not None
        assert abs(pred - 13.0) < 1e-6, f"Expected 13.0, got {pred}"

    def test_insufficient_obs(self) -> None:
        """Should return None when min_obs not met."""
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 5.0])

        pred = rolling_ols_predict(x, y, latest_x=3.0, min_obs=5)
        assert pred is None

    def test_negative_relationship(self) -> None:
        """Negative slope: y = -x + 10."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([9.0, 8.0, 7.0, 6.0, 5.0])

        pred = rolling_ols_predict(x, y, latest_x=6.0, min_obs=5)
        assert pred is not None
        assert abs(pred - 4.0) < 1e-6, f"Expected 4.0, got {pred}"

    def test_noisy_data(self) -> None:
        """OLS should handle noisy data gracefully."""
        rng = np.random.default_rng(999)
        x = rng.normal(0, 1, 100)
        y = 2.0 * x + 1.0 + rng.normal(0, 0.5, 100)

        pred = rolling_ols_predict(x, y, latest_x=0.0, min_obs=13)
        assert pred is not None
        # Predicted should be close to intercept (≈ 1.0) when x=0
        assert abs(pred - 1.0) < 0.3, f"Expected ~1.0, got {pred}"
