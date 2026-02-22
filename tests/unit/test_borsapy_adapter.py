"""Unit tests for BorsapyAdapter İş Yatırım screener extensions."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from bist_quant.clients.borsapy_adapter import BorsapyAdapter


def _adapter_with_client(tmp_path: Path, client: object) -> BorsapyAdapter:
    loader = SimpleNamespace(data_dir=tmp_path)
    adapter = BorsapyAdapter(loader=loader)
    adapter._client = client  # noqa: SLF001 - unit-test injection
    return adapter


def test_screen_stocks_isyatirim_delegates_to_client(tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    class FakeClient:
        def screen_stocks(self, template=None, filters=None):  # noqa: ANN001
            calls["template"] = template
            calls["filters"] = dict(filters or {})
            return pd.DataFrame({"symbol": ["THYAO"], "forward_pe": [8.5]})

    adapter = _adapter_with_client(tmp_path, FakeClient())
    result = adapter.screen_stocks_isyatirim(template="high_upside", pe_max=12, index="XU100")

    assert not result.empty
    assert list(result["symbol"]) == ["THYAO"]
    assert calls == {
        "template": "high_upside",
        "filters": {"pe_max": 12, "index": "XU100"},
    }


def test_get_analyst_data_normalizes_targets_and_recommendation(tmp_path: Path) -> None:
    class FakeClient:
        def get_analyst_targets(self, symbol: str) -> dict[str, object]:
            assert symbol == "THYAO"
            return {
                "price_targets": {
                    "target_price": 120.0,
                    "high": 140.0,
                    "low": 100.0,
                    "count": 11,
                },
                "recommendations": pd.DataFrame([{"recommendation": "buy", "score": 1.8}]),
                "last_price": 100.0,
                "forward_pe": 9.7,
            }

    adapter = _adapter_with_client(tmp_path, FakeClient())
    data = adapter.get_analyst_data("THYAO.IS")

    assert data["symbol"] == "THYAO"
    assert data["analyst_target_price"] == 120.0
    assert data["analyst_target_high"] == 140.0
    assert data["analyst_target_low"] == 100.0
    assert data["analyst_target_count"] == 11
    assert data["recommendation"] == "AL"
    assert data["recommendation_score"] == 1.8
    assert data["upside_potential"] == pytest.approx(20.0)
    assert data["forward_pe"] == 9.7


def test_get_foreign_ownership_extracts_ratios_and_changes(tmp_path: Path) -> None:
    ticker = SimpleNamespace(
        foreign_ownership={
            "foreign_ratio": 42.4,
            "foreign_change_1w": 0.6,
            "foreign_change_1m": 1.1,
        }
    )

    class FakeClient:
        def get_ticker(self, symbol: str):  # noqa: ANN001
            assert symbol == "GARAN"
            return ticker

        def get_fast_info(self, symbol: str) -> dict[str, float]:
            assert symbol == "GARAN"
            return {"float_ratio": 33.5}

    adapter = _adapter_with_client(tmp_path, FakeClient())
    data = adapter.get_foreign_ownership("GARAN.IS")

    assert data["symbol"] == "GARAN"
    assert data["foreign_ratio"] == 42.4
    assert data["foreign_change_1w"] == 0.6
    assert data["foreign_change_1m"] == 1.1
    assert data["float_ratio"] == 33.5


def test_get_screener_criteria_uses_client_method(tmp_path: Path) -> None:
    class FakeClient:
        def get_screener_criteria(self) -> dict[str, dict[str, str]]:
            return {
                "forward_pe": {"name": "Forward P/E", "type": "number"},
                "foreign_ratio": {"name": "Foreign Ownership", "type": "number"},
            }

    adapter = _adapter_with_client(tmp_path, FakeClient())
    criteria = adapter.get_screener_criteria()

    assert len(criteria) == 2
    assert {item["key"] for item in criteria} == {"forward_pe", "foreign_ratio"}
