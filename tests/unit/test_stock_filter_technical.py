"""Unit tests for stock_filter technical scan integration."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from bist_quant.engines import stock_filter


def _sample_screen_frame() -> tuple[pd.DataFrame, str, dict[str, str]]:
    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "name": ["AAA", "BBB", "CCC"],
            "market_cap_usd": [100.0, 200.0, 150.0],
            "upside_potential": [10.0, 20.0, 5.0],
            "rsi_14": [45.0, 31.0, 60.0],
            "return_1m": [1.0, 3.0, -2.0],
            "recommendation": ["AL", "TUT", "SAT"],
            "sector": ["Tech", "Tech", "Bank"],
        }
    )
    return frame, "2026-02-20", {"AAA": "Tech", "BBB": "Tech", "CCC": "Bank"}


def test_run_response_applies_technical_scan(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(stock_filter, "_resolve_paths", lambda runtime_paths: runtime_paths)
    monkeypatch.setattr(
        stock_filter,
        "_load_local_screen_frame",
        lambda runtime_paths, force_refresh=False: _sample_screen_frame(),
    )

    def fake_scan(self, universe, condition, interval):  # noqa: ANN001
        assert universe == ["AAA", "BBB", "CCC"]
        assert condition == "rsi < 30"
        assert interval == "1d"
        return pd.DataFrame(
            {
                "symbol": ["BBB"],
                "rsi": [25.0],
            }
        )

    monkeypatch.setattr(stock_filter.TechnicalScannerEngine, "scan", fake_scan)

    payload = {
        "technical_condition": "rsi < 30",
        "limit": 20,
    }
    response = stock_filter._run_response(
        payload,
        runtime_paths=SimpleNamespace(data_dir=tmp_path),
    )

    assert response["meta"]["technical_conditions"] == ["rsi < 30"]
    assert response["meta"]["returned_rows"] == 1
    assert response["rows"][0]["symbol"] == "BBB"
    assert "rsi" in {col["key"] for col in response["columns"]}
    assert response["rows"][0]["rsi"] == 25.0


def test_run_response_rejects_unknown_technical_template(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(stock_filter, "_resolve_paths", lambda runtime_paths: runtime_paths)
    monkeypatch.setattr(
        stock_filter,
        "_load_local_screen_frame",
        lambda runtime_paths, force_refresh=False: _sample_screen_frame(),
    )

    with pytest.raises(ValueError):
        stock_filter._run_response(
            {"technical_scan_name": "not_a_real_scan"},
            runtime_paths=SimpleNamespace(data_dir=tmp_path),
        )
