"""Contract tests for dashboard API routes."""

from __future__ import annotations

import pytest


def test_dashboard_overview_contract() -> None:
    fastapi = pytest.importorskip("fastapi", reason="fastapi not installed", exc_type=ImportError)
    del fastapi
    bist_quant_api = pytest.importorskip(
        "bist_quant.api", reason="bist_quant.api not available", exc_type=ImportError
    )

    from fastapi.testclient import TestClient

    app = bist_quant_api.create_app()
    client = TestClient(app)

    response = client.get("/api/dashboard/overview", params={"lookback": 126})
    assert response.status_code == 200

    payload = response.json()
    assert "kpi" in payload
    assert "regime" in payload
    assert "timeline" in payload
    assert "macro" in payload
    assert "lookback" in payload
    assert payload["lookback"] == 126


def test_dashboard_routes_registered() -> None:
    bist_quant_api = pytest.importorskip(
        "bist_quant.api", reason="bist_quant.api not available", exc_type=ImportError
    )
    app = bist_quant_api.create_app()
    paths = {route.path for route in app.routes}

    assert "/api/dashboard/overview" in paths
    assert "/api/dashboard/regime-history" in paths
    assert "/api/dashboard/macro" in paths
