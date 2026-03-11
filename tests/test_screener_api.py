"""Contract tests for screener endpoints."""

from __future__ import annotations

import pandas as pd
import pytest


def test_screener_sparklines_returns_symbol_series(monkeypatch) -> None:
    fastapi = pytest.importorskip("fastapi", reason="fastapi not installed", exc_type=ImportError)
    del fastapi
    bist_quant_api = pytest.importorskip(
        "bist_quant.api", reason="bist_quant.api not available", exc_type=ImportError
    )

    from fastapi.testclient import TestClient

    import bist_quant.engines.stock_filter as stock_filter

    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    close_df = pd.DataFrame(
        {
            "THYAO": [100.0 + i for i in range(len(dates))],
            "GARAN": [50.0 + i * 0.5 for i in range(len(dates))],
        },
        index=dates,
    )
    monkeypatch.setitem(stock_filter._SCREEN_CACHE, "close_df", close_df)

    app = bist_quant_api.create_app()
    client = TestClient(app)

    response = client.post(
        "/api/screener/sparklines",
        json={"symbols": ["thyao", "GARAN.IS"], "points": 30},
    )
    assert response.status_code == 200
    payload = response.json()
    assert sorted(payload.keys()) == ["GARAN", "THYAO"]
    assert len(payload["THYAO"]) == 30
    assert len(payload["GARAN"]) == 30
    assert all(isinstance(x, (int, float)) for x in payload["THYAO"])


def test_screener_sparklines_rejects_empty_symbols() -> None:
    fastapi = pytest.importorskip("fastapi", reason="fastapi not installed", exc_type=ImportError)
    del fastapi
    bist_quant_api = pytest.importorskip(
        "bist_quant.api", reason="bist_quant.api not available", exc_type=ImportError
    )

    from fastapi.testclient import TestClient

    app = bist_quant_api.create_app()
    client = TestClient(app)

    response = client.post("/api/screener/sparklines", json={"symbols": []})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "symbols" in str(detail)
