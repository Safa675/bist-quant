"""Assert factor_builders panel functions delegate to core."""

from __future__ import annotations

import inspect

from bist_quant.signals import factor_builders as fb


def _assert_delegate(fn_name: str, core_subpath: str) -> None:
    source = inspect.getsource(getattr(fb, fn_name))
    assert "core.panels" in source or "core/" in source
    assert core_subpath in source
    assert source.count("\n") < 12


class TestFactorBuildersDelegate:
    def test_liquidity_delegates(self) -> None:
        _assert_delegate("build_liquidity_panels", "liquidity")

    def test_trading_intensity_delegates(self) -> None:
        _assert_delegate("build_trading_intensity_panels", "trading_intensity")

    def test_sentiment_delegates(self) -> None:
        _assert_delegate("build_sentiment_panels", "sentiment")

    def test_fundamental_momentum_delegates(self) -> None:
        _assert_delegate("build_fundamental_momentum_panels", "fundamental_momentum")

    def test_carry_delegates(self) -> None:
        _assert_delegate("build_carry_panels", "carry")

    def test_defensive_delegates(self) -> None:
        _assert_delegate("build_defensive_panels", "defensive")

    def test_vol_beta_delegates(self) -> None:
        _assert_delegate("build_volatility_beta_panels", "vol_beta")

    def test_profit_margin_delegates(self) -> None:
        _assert_delegate("_build_profitability_margin_panels", "profit_margin")
