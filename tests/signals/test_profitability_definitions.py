"""Tests that assets-based and margin-based profitability are distinct factors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _synthetic_quarterly_frame() -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """One ticker: OpInc, GP, TotalAssets, Revenue by quarter."""
    quarters = pd.Index(["2023Q1", "2023Q2", "2023Q3", "2023Q4"], name="quarter")
    operating_income = pd.Series([100.0, 110.0, 120.0, 130.0], index=quarters)
    gross_profit = pd.Series([200.0, 210.0, 220.0, 230.0], index=quarters)
    total_assets = pd.Series([1000.0, 1020.0, 1040.0, 1060.0], index=quarters)
    revenue = pd.Series([500.0, 520.0, 540.0, 560.0], index=quarters)
    return operating_income, gross_profit, total_assets, revenue


class TestProfitabilityDefinitions:
    def test_assets_based_differs_from_margin_based(self) -> None:
        op, gp, ta, rev = _synthetic_quarterly_frame()

        assets_based = 0.5 * (op / ta) + 0.5 * (gp / ta)
        margin_based = 0.6 * (op / rev) + 0.4 * (gp / rev)

        assert list(assets_based.index) == list(margin_based.index)
        assert not (assets_based.round(8) == margin_based.round(8)).all()

    def test_margin_level_formula_uses_ttm_weights(self) -> None:
        op, gp, _ta, rev = _synthetic_quarterly_frame()
        op_ttm = op.cumsum()
        gp_ttm = gp.cumsum()
        rev_ttm = rev.cumsum()
        op_margin = op_ttm / rev_ttm.replace(0.0, np.nan)
        gp_margin = gp_ttm / rev_ttm.replace(0.0, np.nan)
        margin_level = 0.6 * op_margin + 0.4 * gp_margin
        assert float(margin_level.iloc[-1]) == pytest.approx(0.29245, rel=1e-3)
