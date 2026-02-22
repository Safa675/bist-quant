"""Unit tests for RiskManager."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bist_quant import RiskManager


class TestRiskManager:
    """Tests for RiskManager class."""

    def test_risk_manager_initialization(self) -> None:
        """Test RiskManager can be initialized."""
        rm = RiskManager()
        assert rm is not None

    def test_calculate_volatility_weights(self) -> None:
        """Test inverse downside-vol weighting produces normalized weights."""
        dates = pd.date_range("2023-01-01", periods=90, freq="B")
        close_df = pd.DataFrame(
            {
                "THYAO": np.linspace(10.0, 20.0, len(dates)),
                "GARAN": np.linspace(30.0, 40.0, len(dates)) * (1 + 0.01 * np.sin(np.arange(len(dates)))),
            },
            index=dates,
        )

        rm = RiskManager(close_df=close_df)
        weights = rm.inverse_downside_vol_weights(
            selected=["THYAO", "GARAN"],
            date=dates[-1],
            lookback=30,
            max_weight=0.8,
        )

        assert set(weights.index) == {"THYAO", "GARAN"}
        assert float(weights.sum()) == pytest.approx(1.0)

    def test_apply_downside_vol_targeting(self) -> None:
        """Test downside volatility targeting preserves series length."""
        returns = pd.Series(np.random.default_rng(42).normal(0.001, 0.02, 120))
        scaled = RiskManager.apply_downside_vol_targeting(returns, lookback=20)

        assert len(scaled) == len(returns)
        assert scaled.notna().all()

    def test_apply_stop_loss(self) -> None:
        """Test stop loss removes breached positions."""
        open_df = pd.DataFrame(
            {"THYAO": [8.0], "GARAN": [12.0]},
            index=[pd.Timestamp("2024-01-02")],
        )

        remaining = RiskManager.apply_stop_loss(
            current_holdings=["THYAO", "GARAN"],
            stopped_out=set(),
            entry_prices={"THYAO": 10.0, "GARAN": 10.0},
            open_df=open_df,
            date=pd.Timestamp("2024-01-02"),
            stop_loss_threshold=0.15,
        )

        assert "THYAO" not in remaining
        assert "GARAN" in remaining
