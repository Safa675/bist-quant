"""Unit tests for derivatives integrations."""

from __future__ import annotations

from typing import Any

import pytest

from bist_quant.clients.derivatives_provider import DerivativesProvider


_CONTRACTS: list[dict[str, Any]] = [
    {
        "symbol": "XU030D0326",
        "base_symbol": "XU030D",
        "contract_type": "futures",
        "expiry": "2026-03-31",
        "last_price": 10120.0,
        "volume": 4_500.0,
        "open_interest": 12_000.0,
    },
    {
        "symbol": "XU030D0426",
        "base_symbol": "XU030D",
        "contract_type": "futures",
        "expiry": "2026-04-30",
        "last_price": 10180.0,
        "volume": 2_200.0,
        "open_interest": 9_500.0,
    },
    {
        "symbol": "XU030D0326C",
        "base_symbol": "XU030D",
        "contract_type": "option",
        "option_type": "call",
        "expiry": "2026-03-31",
        "strike": 10250.0,
        "last_price": 130.0,
        "volume": 1200.0,
    },
    {
        "symbol": "XU030D0326P",
        "base_symbol": "XU030D",
        "contract_type": "option",
        "option_type": "put",
        "expiry": "2026-03-31",
        "strike": 9750.0,
        "last_price": 95.0,
        "volume": 900.0,
    },
    {
        "symbol": "USDTRYD0326",
        "base_symbol": "USDTRYD",
        "contract_type": "futures",
        "expiry": "2026-03-31",
        "last_price": 37.25,
        "volume": 8_000.0,
    },
]


class _DummyVIOP:
    def contracts(self):  # noqa: ANN201
        return _CONTRACTS

    def futures(self):  # noqa: ANN201
        return [row for row in _CONTRACTS if row["contract_type"] == "futures"]

    def options(self):  # noqa: ANN201
        return [row for row in _CONTRACTS if row["contract_type"] == "option"]


class _DummyIndex:
    def __init__(self, symbol: str) -> None:
        self.info = {"last": 10000.0 if symbol == "XU030" else None}


class _DummyBP:
    VIOP = _DummyVIOP

    @staticmethod
    def viop_contracts(base_symbol: str | None = None):  # noqa: ANN205
        if base_symbol is None:
            return _CONTRACTS
        normalized = str(base_symbol).upper()
        return [row for row in _CONTRACTS if str(row.get("base_symbol", "")).upper() == normalized]

    @staticmethod
    def index(symbol: str) -> _DummyIndex:
        return _DummyIndex(symbol)


def test_derivatives_provider_futures_and_options_filters() -> None:
    provider = DerivativesProvider(borsapy_module=_DummyBP())

    futures = provider.get_futures()
    options = provider.get_options()

    assert set(futures["symbol"]) == {"XU030D0326", "XU030D0426", "USDTRYD0326"}
    assert set(options["symbol"]) == {"XU030D0326C", "XU030D0326P"}


def test_derivatives_provider_contract_lookup_and_basis() -> None:
    provider = DerivativesProvider(borsapy_module=_DummyBP())

    contracts = provider.get_contracts("XU030D")
    assert {row["symbol"] for row in contracts} == {
        "XU030D0326",
        "XU030D0426",
        "XU030D0326C",
        "XU030D0326P",
    }

    basis = provider.get_futures_basis("XU030D")
    assert basis["contract"] == "XU030D0326"
    assert basis["spot_symbol"] == "XU030"
    assert basis["spot_price"] == pytest.approx(10000.0)
    assert basis["futures_price"] == pytest.approx(10120.0)
    assert basis["basis"] == pytest.approx(120.0)
    assert basis["basis_pct"] == pytest.approx(1.2)
    assert basis["regime"] == "contango"


def test_derivatives_provider_put_call_ratio_and_index_premium() -> None:
    provider = DerivativesProvider(borsapy_module=_DummyBP())

    put_call = provider.get_put_call_ratio()
    premium = provider.get_index_futures_premium()

    assert put_call == pytest.approx(0.75)
    assert premium["premium_points"] == pytest.approx(120.0)
    assert premium["premium_pct"] == pytest.approx(1.2)
    assert premium["is_premium"] is True

