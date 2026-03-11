"""Backend contract coverage for every frontend-consumed API endpoint."""

from __future__ import annotations

from pathlib import Path

import pytest


FRONTEND_CONSUMED_PATHS = {
    "/api/analytics/benchmark/xu100",
    "/api/analytics/run",
    "/api/backtest/run",
    "/api/compliance/activity-anomalies",
    "/api/compliance/check",
    "/api/compliance/position-limits",
    "/api/compliance/rules",
    "/api/health/live",
    "/api/jobs",
    "/api/meta/signals",
    "/api/meta/system",
    "/api/professional/crypto-sizing",
    "/api/professional/greeks",
    "/api/professional/pip-value",
    "/api/professional/stress",
    "/api/screener/metadata",
    "/api/screener/run",
    "/api/screener/sparklines",
    "/api/signal-construction/backtest",
    "/api/signal-construction/five-factor",
    "/api/signal-construction/orthogonalization",
    "/api/signal-construction/snapshot",
}


def _build_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    fastapi = pytest.importorskip("fastapi", reason="fastapi not installed", exc_type=ImportError)
    del fastapi
    bist_quant_api = pytest.importorskip(
        "bist_quant.api", reason="bist_quant.api not available", exc_type=ImportError
    )

    import bist_quant.api.main as api_main
    from bist_quant.api.jobs import JobManager
    from fastapi.testclient import TestClient

    monkeypatch.setattr(
        api_main,
        "job_manager",
        JobManager(max_workers=2, store_path=tmp_path / "api_jobs_frontend_contracts.json"),
    )

    app = bist_quant_api.create_app()
    return TestClient(app)


def test_frontend_consumed_routes_registered(monkeypatch, tmp_path) -> None:
    client = _build_client(monkeypatch, tmp_path)
    paths = {route.path for route in client.app.routes}

    dynamic_paths = {"/api/factors/{name}", "/api/jobs/{job_id}", "/api/jobs/{job_id}/retry"}

    assert FRONTEND_CONSUMED_PATHS.issubset(paths)
    assert dynamic_paths.issubset(paths)


def test_health_live_contract(monkeypatch, tmp_path) -> None:
    client = _build_client(monkeypatch, tmp_path)
    response = client.get("/api/health/live")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True


def test_meta_signals_contract(monkeypatch, tmp_path) -> None:
    import bist_quant.services.core_service as core_service

    monkeypatch.setattr(
        core_service.CoreBackendService,
        "list_available_signals",
        lambda self: ["momentum", "quality", "value"],
    )
    client = _build_client(monkeypatch, tmp_path)

    response = client.get("/api/meta/signals")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 3
    assert payload["signals"] == ["momentum", "quality", "value"]


def test_meta_system_contract(monkeypatch, tmp_path) -> None:
    import bist_quant.services.system_service as system_service

    monkeypatch.setattr(
        system_service.SystemService,
        "diagnostics_snapshot",
        lambda self: {"system": "ok", "sources": {"prices": True}},
    )
    client = _build_client(monkeypatch, tmp_path)

    response = client.get("/api/meta/system")
    assert response.status_code == 200
    payload = response.json()
    assert payload["system"] == "ok"


def test_screener_metadata_contract(monkeypatch, tmp_path) -> None:
    import bist_quant.engines.stock_filter as stock_filter

    monkeypatch.setattr(
        stock_filter,
        "get_stock_filter_metadata",
        lambda: {
            "indexes": ["XU100"],
            "templates": ["quality", "value"],
            "recommendation_labels": ["AL", "SAT"],
        },
    )
    client = _build_client(monkeypatch, tmp_path)

    response = client.get("/api/screener/metadata")
    assert response.status_code == 200
    payload = response.json()
    assert payload["indexes"] == ["XU100"]


def test_screener_run_contract_and_validation(monkeypatch, tmp_path) -> None:
    import bist_quant.engines.stock_filter as stock_filter

    monkeypatch.setattr(
        stock_filter,
        "run_stock_filter",
        lambda _: {
            "count": 2,
            "rows": [
                {"symbol": "AKBNK", "pe": 6.1, "roe": 0.22, "sector": "Banks"},
                {"symbol": "THYAO", "pe": 7.4, "roe": 0.18, "sector": "Industrials"},
            ],
        },
    )
    client = _build_client(monkeypatch, tmp_path)

    ok = client.post("/api/screener/run", json={"index": "XU100", "limit": 20})
    assert ok.status_code == 200
    assert ok.json()["count"] == 2

    bad = client.post("/api/screener/run", json={"index": "XU100", "limit": 0})
    assert bad.status_code == 422


def test_professional_greeks_contract_and_validation(monkeypatch, tmp_path) -> None:
    client = _build_client(monkeypatch, tmp_path)

    ok = client.post(
        "/api/professional/greeks",
        json={
            "option_type": "call",
            "spot": 100,
            "strike": 100,
            "time_years": 0.25,
            "volatility": 0.2,
            "risk_free_rate": 0.05,
        },
    )
    assert ok.status_code == 200
    payload = ok.json()
    assert "delta" in payload
    assert "theoretical_price" in payload

    bad = client.post(
        "/api/professional/greeks",
        json={
            "option_type": "call",
            "spot": -100,
            "strike": 100,
            "time_years": 0.25,
            "volatility": 0.2,
            "risk_free_rate": 0.05,
        },
    )
    assert bad.status_code == 422


def test_professional_stress_contract_and_validation(monkeypatch, tmp_path) -> None:
    client = _build_client(monkeypatch, tmp_path)

    ok = client.post(
        "/api/professional/stress",
        json={
            "portfolio_value": 1_000_000,
            "shocks": [
                {"factor": "Equity", "shock_pct": -20, "beta": 1.0},
                {"factor": "FX", "shock_pct": 10, "beta": 0.4},
            ],
        },
    )
    assert ok.status_code == 200
    payload = ok.json()
    assert "scenario_loss_pct" in payload
    assert "by_factor" in payload

    bad = client.post("/api/professional/stress", json={"portfolio_value": 1_000_000, "shocks": []})
    assert bad.status_code == 422


def test_professional_crypto_sizing_contract_and_validation(monkeypatch, tmp_path) -> None:
    client = _build_client(monkeypatch, tmp_path)

    ok = client.post(
        "/api/professional/crypto-sizing",
        json={
            "pair": "BTCUSDT",
            "side": "long",
            "entry_price": 60_000,
            "equity": 10_000,
            "risk_pct": 2,
            "leverage": 5,
            "stop_distance_pct": 3,
        },
    )
    assert ok.status_code == 200
    payload = ok.json()
    assert "quantity" in payload
    assert "notional" in payload

    bad = client.post(
        "/api/professional/crypto-sizing",
        json={
            "pair": "BTCUSDT",
            "side": "long",
            "entry_price": 60_000,
            "equity": 10_000,
            "risk_pct": 0,
            "leverage": 5,
            "stop_distance_pct": 3,
        },
    )
    assert bad.status_code == 422


def test_compliance_rules_contract(monkeypatch, tmp_path) -> None:
    client = _build_client(monkeypatch, tmp_path)

    response = client.get("/api/compliance/rules")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["rules"], list)
    assert payload["rules"]


def test_compliance_check_contract_and_validation(monkeypatch, tmp_path) -> None:
    import bist_quant.analytics.professional as professional

    monkeypatch.setattr(professional, "run_compliance_rule_engine", lambda record, rules: [])
    client = _build_client(monkeypatch, tmp_path)

    ok = client.post(
        "/api/compliance/check",
        json={
            "transaction": {
                "id": "tx-1",
                "timestamp": "2026-01-02T10:00:00Z",
                "user_id": "usr-1",
                "order_id": "ord-1",
                "symbol": "THYAO",
                "side": "buy",
                "quantity": 100,
                "price": 10,
                "venue": "XIST",
                "strategy_id": "strat-1",
            },
            "rules": [],
        },
    )
    assert ok.status_code == 200
    payload = ok.json()
    assert payload["passed"] is True
    assert payload["hits"] == []

    bad = client.post("/api/compliance/check", json={"rules": []})
    assert bad.status_code == 422


def test_compliance_position_limits_contract_and_validation(monkeypatch, tmp_path) -> None:
    import bist_quant.analytics.professional as professional

    monkeypatch.setattr(
        professional,
        "monitor_position_limits",
        lambda positions: [
            {"symbol": "THYAO", "value": 1_200_000, "limit": 1_000_000, "breach": True}
        ],
    )
    client = _build_client(monkeypatch, tmp_path)

    ok = client.post(
        "/api/compliance/position-limits",
        json={"positions": [{"symbol": "THYAO", "value": 1_200_000, "limit": 1_000_000}]},
    )
    assert ok.status_code == 200
    payload = ok.json()
    assert payload["breach_count"] == 1
    assert payload["total_checked"] == 1

    bad = client.post("/api/compliance/position-limits", json={"positions": []})
    assert bad.status_code == 422
