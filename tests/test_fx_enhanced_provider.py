"""Unit tests for enhanced FX integrations."""

from __future__ import annotations

import asyncio

import pandas as pd

from bist_quant.clients.fx_commodities_client import FXCommoditiesClient
from bist_quant.clients.fx_enhanced_provider import FXEnhancedProvider


class _DummyFX:
    def __init__(self, asset: str) -> None:
        self.asset = asset

    @property
    def bank_rates(self) -> pd.DataFrame:
        if self.asset != "USD":
            raise ValueError("unsupported currency")
        return pd.DataFrame(
            [
                {
                    "bank": "akbank",
                    "bank_name": "Akbank",
                    "currency": "USD",
                    "buy": 34.10,
                    "sell": 34.55,
                    "spread": 1.32,
                },
                {
                    "bank": "garanti",
                    "bank_name": "Garanti",
                    "currency": "USD",
                    "buy": 34.20,
                    "sell": 34.50,
                    "spread": 0.88,
                },
            ]
        )

    @property
    def institution_rates(self) -> pd.DataFrame:
        if self.asset != "gram-altin":
            raise ValueError("unsupported asset")
        return pd.DataFrame(
            [
                {
                    "institution": "kapalicarsi",
                    "institution_name": "Kapalicarsi",
                    "asset": "gram-altin",
                    "buy": 3005.0,
                    "sell": 3015.0,
                },
                {
                    "institution": "akbank",
                    "institution_name": "Akbank",
                    "asset": "gram-altin",
                    "buy": 3003.0,
                    "sell": 3018.0,
                    "spread": 0.50,
                },
            ]
        )

    def history(
        self,
        period: str = "1mo",
        interval: str = "1d",
        start=None,  # noqa: ANN001
        end=None,  # noqa: ANN001
    ) -> pd.DataFrame:
        del start, end
        if self.asset not in {"USD", "EUR", "GBP", "XAU", "XAG", "BRENT", "WTI"}:
            raise ValueError("unsupported for intraday")
        if interval not in {"1h", "30m", "15m", "5m", "1m", "4h"}:
            raise ValueError("unsupported interval")
        index = pd.date_range("2026-02-20 10:00:00", periods=6, freq="h")
        return pd.DataFrame(
            {
                "Open": [34.00, 34.05, 34.10, 34.12, 34.15, 34.20],
                "High": [34.08, 34.12, 34.17, 34.20, 34.23, 34.30],
                "Low": [33.98, 34.02, 34.05, 34.08, 34.12, 34.18],
                "Close": [34.04, 34.10, 34.14, 34.18, 34.21, 34.26],
                "Volume": [100, 130, 140, 120, 115, 150],
            },
            index=index,
        )


class _DummyBP:
    FX = _DummyFX


def test_fx_enhanced_provider_bank_and_institution_rates() -> None:
    provider = FXEnhancedProvider(borsapy_module=_DummyBP())

    bank = provider.get_bank_rates(currency="USD")
    institution = provider.get_institution_rates(asset="gram-altin")

    assert not bank.empty
    assert set(["bank", "bank_name", "currency", "buy", "sell", "spread", "mid"]).issubset(bank.columns)
    assert set(bank["currency"]) == {"USD"}
    assert not institution.empty
    assert set(["institution", "institution_name", "asset", "buy", "sell", "spread", "mid"]).issubset(
        institution.columns
    )
    assert set(institution["asset"]) == {"gram-altin"}


def test_fx_enhanced_provider_intraday_normalization() -> None:
    provider = FXEnhancedProvider(borsapy_module=_DummyBP())
    intraday = provider.get_intraday(currency="USD", interval="1h", period="5d")

    assert not intraday.empty
    assert list(intraday.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "asset",
        "interval",
        "period",
        "source",
    ]
    assert intraday["asset"].iloc[-1] == "USD"
    assert intraday["interval"].iloc[-1] == "1h"
    assert intraday["source"].iloc[-1] == "tradingview"


def test_fx_enhanced_provider_carry_spread_summary() -> None:
    provider = FXEnhancedProvider(borsapy_module=_DummyBP())
    summary = provider.get_carry_spread()

    assert summary["currency"] == "USD"
    assert summary["bank_count"] == 2
    assert summary["avg_spread_pct"] is not None
    assert summary["best_bank"] is not None


def test_fx_commodities_client_merges_bank_rates_with_mcp_rows() -> None:
    provider = FXEnhancedProvider(borsapy_module=_DummyBP())
    client = FXCommoditiesClient(enhanced_provider=provider, include_bank_rates=True)

    async def _fake_call(tool: str, params: dict) -> dict:  # noqa: ANN202
        del tool, params
        return {
            "data": {
                "rates": [
                    {
                        "pair": "USD/TRY",
                        "bid": 34.00,
                        "ask": 34.10,
                        "last_price": 34.05,
                        "change_percent": 0.4,
                    }
                ]
            }
        }

    client._call_mcp_async = _fake_call  # type: ignore[assignment]

    async def _run() -> pd.DataFrame:
        try:
            return await client.get_fx_rates(["USD/TRY"])
        finally:
            await client.close()

    merged = asyncio.run(_run())

    assert not merged.empty
    assert "source" in merged.columns
    assert set(merged["source"]) >= {"mcp", "bank_rates"}
    assert (merged["pair"] == "USD/TRY").all()
