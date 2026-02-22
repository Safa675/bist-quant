"""Unit tests for fixed income integrations."""

from __future__ import annotations

import pandas as pd
import pytest

from bist_quant.common.data_loader import DataLoader
from bist_quant.common.fixed_income_provider import (
    DEFAULT_FALLBACK_RISK_FREE_RATE,
    FixedIncomeProvider,
)


class _DummyTCMB:
    policy_rate = 38.0
    overnight = {"borrowing": 36.5, "lending": 41.0}
    late_liquidity = {"borrowing": 40.0, "lending": 44.0}
    rates = pd.DataFrame(
        [
            {"type": "overnight", "borrowing": 36.5, "lending": 41.0},
            {"type": "late_liquidity", "borrowing": 40.0, "lending": 44.0},
        ]
    )

    def history(self, rate_type: str = "policy", period: str = "1y") -> pd.DataFrame:
        del rate_type, period
        return pd.DataFrame(
            {
                "Date": ["2025-12-29", "2025-12-30", "2025-12-31"],
                "policy_rate": [36.0, 37.0, 38.0],
            }
        )


class _DummyBorsapy:
    TCMB = _DummyTCMB

    @staticmethod
    def bonds() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "maturity": ["2Y", "5Y", "10Y"],
                "yield": [26.42, 27.15, 28.03],
                "change": [0.1, 0.2, 0.3],
            }
        )

    @staticmethod
    def risk_free_rate() -> float:
        return 28.03

    @staticmethod
    def eurobonds() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "currency": ["USD", "USD", "EUR"],
                "ask_yield": [7.2, 7.8, 4.0],
            }
        )


def test_fixed_income_provider_bond_yields_and_curve() -> None:
    provider = FixedIncomeProvider(borsapy_module=_DummyBorsapy)

    yields = provider.get_bond_yields()
    assert yields["2Y"] == pytest.approx(26.42)
    assert yields["10Y"] == pytest.approx(28.03)

    curve = provider.get_yield_curve()
    assert list(curve["maturity"]) == ["2Y", "5Y", "10Y"]
    assert curve["yield"].iloc[-1] == pytest.approx(28.03)


def test_fixed_income_provider_risk_free_rate_decimal() -> None:
    provider = FixedIncomeProvider(borsapy_module=_DummyBorsapy)
    assert provider.get_risk_free_rate() == pytest.approx(0.2803)


def test_fixed_income_provider_tcmb_and_spread() -> None:
    provider = FixedIncomeProvider(borsapy_module=_DummyBorsapy)

    rates = provider.get_tcmb_rates()
    assert rates["policy_rate"] == pytest.approx(38.0)
    assert rates["overnight"]["borrowing"] == pytest.approx(36.5)

    history = provider.get_tcmb_history()
    assert not history.empty
    assert "rate" in history.columns
    assert history["rate"].iloc[-1] == pytest.approx(38.0)

    spread = provider.get_spread_index()
    assert spread == pytest.approx(7.5)


def test_data_loader_risk_free_rate_csv_fallback(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "Date": ["2025-12-31"],
            "deposit_rate": [32.0],
        }
    ).to_csv(data_dir / "tcmb_deposit_rates.csv", index=False)

    loader = DataLoader(data_dir=data_dir)

    class _BrokenProvider:
        @staticmethod
        def get_risk_free_rate() -> float:
            return float("nan")

    loader._fixed_income_provider = _BrokenProvider()  # type: ignore[assignment]
    assert loader.risk_free_rate == pytest.approx(0.32)


def test_data_loader_risk_free_rate_default_fallback(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(data_dir=data_dir)

    class _BrokenProvider:
        @staticmethod
        def get_risk_free_rate() -> float:
            raise RuntimeError("no live rate")

    loader._fixed_income_provider = _BrokenProvider()  # type: ignore[assignment]
    assert loader.risk_free_rate == pytest.approx(DEFAULT_FALLBACK_RISK_FREE_RATE)

