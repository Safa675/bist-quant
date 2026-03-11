"""Phase 4 API contract tests for parity surfaces."""

from __future__ import annotations

from pathlib import Path

import pytest


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
        JobManager(max_workers=2, store_path=tmp_path / "api_jobs_phase4.json"),
    )
    app = bist_quant_api.create_app()
    return TestClient(app), api_main


def test_backtest_accepts_extended_risk_fields(monkeypatch, tmp_path) -> None:
    client, api_main = _build_client(monkeypatch, tmp_path)

    captured = {}

    def _fake_run(req):
        captured["use_regime_filter"] = req.use_regime_filter
        captured["use_liquidity_filter"] = req.use_liquidity_filter
        captured["use_slippage"] = req.use_slippage
        captured["slippage_bps"] = req.slippage_bps
        captured["use_stop_loss"] = req.use_stop_loss
        captured["stop_loss_threshold"] = req.stop_loss_threshold
        captured["use_vol_targeting"] = req.use_vol_targeting
        captured["target_downside_vol"] = req.target_downside_vol
        captured["benchmark"] = req.benchmark
        return {"metrics": {"cagr": 0.1, "sharpe": 1.2}}

    monkeypatch.setattr(api_main, "_run_backtest_request", _fake_run)

    response = client.post(
        "/api/backtest/run",
        json={
            "factor_name": "momentum",
            "start_date": "2020-01-01",
            "end_date": "2021-01-01",
            "top_n": 20,
            "use_regime_filter": True,
            "use_liquidity_filter": True,
            "use_slippage": True,
            "slippage_bps": 7,
            "use_stop_loss": True,
            "stop_loss_threshold": 0.12,
            "use_vol_targeting": True,
            "target_downside_vol": 0.18,
            "benchmark": "XU100",
        },
    )
    assert response.status_code == 200
    assert captured["use_regime_filter"] is True
    assert captured["use_liquidity_filter"] is True
    assert captured["use_slippage"] is True
    assert captured["slippage_bps"] == 7
    assert captured["use_stop_loss"] is True
    assert captured["stop_loss_threshold"] == pytest.approx(0.12)
    assert captured["use_vol_targeting"] is True
    assert captured["target_downside_vol"] == pytest.approx(0.18)
    assert captured["benchmark"] == "XU100"


def test_analytics_benchmark_endpoint_contract(monkeypatch, tmp_path) -> None:
    client, _ = _build_client(monkeypatch, tmp_path)
    response = client.get("/api/analytics/benchmark/xu100")
    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "XU100"
    assert isinstance(payload["curve"], list)


def test_analytics_run_returns_deep_sections(monkeypatch, tmp_path) -> None:
    client, _ = _build_client(monkeypatch, tmp_path)
    response = client.post(
        "/api/analytics/run",
        json={
            "equity_curve": [
                {"date": "2024-01-02", "value": 100},
                {"date": "2024-01-03", "value": 101},
                {"date": "2024-01-04", "value": 102},
                {"date": "2024-01-05", "value": 101},
                {"date": "2024-01-08", "value": 103},
                {"date": "2024-01-09", "value": 104},
                {"date": "2024-01-10", "value": 103},
                {"date": "2024-01-11", "value": 106},
                {"date": "2024-01-12", "value": 107},
                {"date": "2024-01-15", "value": 108},
            ],
            "methods": [
                "performance",
                "rolling",
                "walk_forward",
                "monte_carlo",
                "attribution",
                "risk",
                "stress",
                "transaction_costs",
            ],
            "walk_forward_splits": 3,
            "train_ratio": 0.7,
            "include_benchmark": False,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert "performance" in payload
    assert "rolling" in payload
    assert "walk_forward" in payload
    assert "monte_carlo" in payload
    assert "risk" in payload
    assert "stress" in payload
    assert "transaction_costs" in payload


def test_compliance_activity_anomalies_endpoint(monkeypatch, tmp_path) -> None:
    client, _ = _build_client(monkeypatch, tmp_path)

    import bist_quant.analytics.professional as professional

    monkeypatch.setattr(
        professional,
        "detect_user_activity_anomalies",
        lambda rows: [{"user_id": "USR-001", "actions_per_hour": 20.0, "z_score": 2.5}],
    )

    response = client.post(
        "/api/compliance/activity-anomalies",
        json={"events": [{"user_id": "USR-001"}, {"user_id": "USR-001"}]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["anomaly_count"] == 1
    assert payload["anomalies"][0]["user_id"] == "USR-001"


def test_professional_pip_value_endpoint(monkeypatch, tmp_path) -> None:
    client, _ = _build_client(monkeypatch, tmp_path)
    response = client.post(
        "/api/professional/pip-value",
        json={"pair": "EURUSD", "lot_size": 100000, "account_conversion_rate": 1.0},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["pair"] == "EURUSD"
    assert "pip_size" in payload
    assert "pip_value_quote" in payload
    assert "pip_value_account" in payload


def test_signal_construction_endpoints(monkeypatch, tmp_path) -> None:
    client, _ = _build_client(monkeypatch, tmp_path)

    import bist_quant.engines.signal_construction as signal_engine
    import bist_quant.services.core_service as core_service

    monkeypatch.setattr(
        signal_engine,
        "run_signal_snapshot",
        lambda payload: {
            "meta": {"universe": payload.get("universe", "XU100")},
            "signals": [{"symbol": "THYAO", "action": "BUY", "combined_score": 0.8}],
            "indicator_summaries": [
                {"name": "rsi", "buy_count": 1, "hold_count": 0, "sell_count": 0}
            ],
        },
    )
    monkeypatch.setattr(
        signal_engine,
        "run_signal_backtest",
        lambda payload: {
            "meta": {"universe": payload.get("universe", "XU100")},
            "metrics": {"cagr": 0.1, "sharpe": 1.2},
            "signals": [],
            "current_holdings": ["THYAO"],
            "equity_curve": [{"date": "2024-01-02", "value": 1.0}],
            "benchmark_curve": [{"date": "2024-01-02", "value": 1.0}],
            "analytics_v2": {},
        },
    )
    monkeypatch.setattr(
        core_service.CoreBackendService,
        "run_backtest",
        lambda self, **kwargs: {
            "metrics": {"cagr": 0.11, "sharpe": 1.4},
            "equity_curve": [{"date": "2024-01-02", "value": 1.0}],
            "kwargs": kwargs,
        },
    )

    snap = client.post("/api/signal-construction/snapshot", json={"universe": "XU100"})
    assert snap.status_code == 200
    assert snap.json()["meta"]["universe"] == "XU100"

    bt = client.post("/api/signal-construction/backtest", json={"universe": "XU100"})
    assert bt.status_code == 200
    assert bt.json()["metrics"]["sharpe"] == 1.2

    ff = client.post(
        "/api/signal-construction/five-factor",
        json={"factor_name": "five_factor_rotation", "top_n": 20},
    )
    assert ff.status_code == 200
    assert ff.json()["metrics"]["sharpe"] == 1.4

    orth = client.post(
        "/api/signal-construction/orthogonalization",
        json={"enabled": True, "axes": ["momentum", "value"], "min_overlap": 20},
    )
    assert orth.status_code == 200
    assert orth.json()["status"] == "configured"
    assert orth.json()["enabled"] is True
