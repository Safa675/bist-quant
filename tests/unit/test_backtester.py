"""Unit tests for Backtester."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bist_quant.common.backtester import (
    Backtester,
    identify_monthly_rebalance_days,
    identify_quarterly_rebalance_days,
)
from bist_quant.common.risk_manager import RiskManager


def _dummy_mcap_builder(close_df: pd.DataFrame, dates: pd.DatetimeIndex, loader) -> pd.DataFrame:
    return pd.DataFrame(1.0, index=dates, columns=close_df.columns)


class TestBacktester:
    """Tests for Backtester class."""

    def test_backtester_initialization(self, tmp_path) -> None:
        """Test Backtester can be initialized with explicit dependencies."""
        backtester = Backtester(
            loader=object(),
            data_dir=tmp_path,
            risk_manager=RiskManager(),
            build_size_market_cap_panel=_dummy_mcap_builder,
        )
        assert backtester is not None

    def test_backtester_with_prices(self, sample_prices_df: pd.DataFrame, tmp_path) -> None:
        """Test Backtester accepts prepared price panels via update_data."""
        prices = sample_prices_df.rename(
            columns={
                "date": "Date",
                "ticker": "Ticker",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
        prices["Date"] = pd.to_datetime(prices["Date"])

        close_df = prices.pivot_table(index="Date", columns="Ticker", values="Close", aggfunc="last")
        volume_df = prices.pivot_table(index="Date", columns="Ticker", values="Volume", aggfunc="last")

        regime_series = pd.Series("Bull", index=close_df.index)
        xu100_prices = pd.Series(
            np.linspace(100.0, 120.0, len(close_df)),
            index=close_df.index,
            name="Close",
        )

        risk_manager = RiskManager()
        backtester = Backtester(
            loader=object(),
            data_dir=tmp_path,
            risk_manager=risk_manager,
            build_size_market_cap_panel=_dummy_mcap_builder,
        )
        backtester.update_data(
            prices=prices,
            close_df=close_df,
            volume_df=volume_df,
            regime_series=regime_series,
            regime_allocations={},
            xu100_prices=xu100_prices,
            xautry_prices=None,
        )

        assert backtester.close_df is not None
        assert backtester.volume_df is not None

    def test_rebalance_day_helpers(self) -> None:
        """Test monthly and quarterly helper functions return non-empty schedules."""
        days = pd.date_range("2024-01-01", periods=260, freq="B")

        monthly = identify_monthly_rebalance_days(days)
        quarterly = identify_quarterly_rebalance_days(days)

        assert len(monthly) > 0
        assert len(quarterly) > 0
