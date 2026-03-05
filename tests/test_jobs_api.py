"""Tests for async jobs and backtest route wiring."""

from __future__ import annotations

import time
from pathlib import Path

import pytest


def _build_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")
    del fastapi
    bist_quant_api = pytest.importorskip("bist_quant.api", reason="bist_quant.api not available")

    import bist_quant.api.main as api_main
    from bist_quant.api.jobs import JobManager
    from fastapi.testclient import TestClient

    monkeypatch.setattr(
        api_main,
        "job_manager",
        JobManager(max_workers=2, store_path=tmp_path / "api_jobs_test.json"),
    )
    app = bist_quant_api.create_app()
    return TestClient(app), api_main


def _assert_structured_error(response, *, status_code: int, code: str) -> dict:
    assert response.status_code == status_code
    body = response.json()
    assert isinstance(body.get("detail"), dict)
    detail = body["detail"]
    assert detail["code"] == code
    assert isinstance(detail.get("detail"), str)
    return detail


def test_create_and_poll_backtest_job(monkeypatch, tmp_path) -> None:
    client, api_main = _build_client(monkeypatch, tmp_path)

    def _fake_run(payload):
        del payload
        return {
            "metrics": {
                "cagr": 0.1,
                "sharpe": 1.2,
                "max_drawdown": -0.2,
                "annualized_volatility": 0.25,
            }
        }

    monkeypatch.setattr(api_main, "_run_backtest_request", _fake_run)

    created = client.post(
        "/api/jobs",
        json={
            "kind": "backtest",
            "request": {
                "factor_name": "momentum",
                "start_date": "2020-01-01",
                "end_date": "2021-01-01",
                "top_n": 20,
            },
        },
    )
    assert created.status_code == 200
    job_id = created.json()["id"]

    final_payload = None
    for _ in range(20):
        resp = client.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200
        payload = resp.json()
        if payload["status"] == "completed":
            final_payload = payload
            break
        time.sleep(0.05)

    assert final_payload is not None
    metrics = final_payload["result"]["metrics"]
    assert metrics["cagr"] == 0.1
    assert metrics["sharpe"] == 1.2


def test_backtest_run_async_mode(monkeypatch, tmp_path) -> None:
    client, api_main = _build_client(monkeypatch, tmp_path)

    monkeypatch.setattr(
        api_main, "_run_backtest_request", lambda payload: {"metrics": {"cagr": 0.12}}
    )

    response = client.post(
        "/api/backtest/run?async_job=true",
        json={
            "factor_name": "momentum",
            "start_date": "2020-01-01",
            "end_date": "2021-01-01",
            "top_n": 20,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["kind"] == "backtest"
    assert payload["status"] in {"queued", "running", "completed"}


def test_retry_job_endpoint(monkeypatch, tmp_path) -> None:
    client, api_main = _build_client(monkeypatch, tmp_path)

    monkeypatch.setattr(
        api_main,
        "_run_backtest_request",
        lambda payload: {"metrics": {"cagr": 0.2}, "factor_name": payload.factor_name},
    )

    created = client.post(
        "/api/jobs",
        json={
            "kind": "backtest",
            "request": {
                "factor_name": "momentum",
                "start_date": "2020-01-01",
                "end_date": "2021-01-01",
                "top_n": 20,
            },
        },
    )
    assert created.status_code == 200
    original_id = created.json()["id"]

    retry_resp = client.post(f"/api/jobs/{original_id}/retry")
    assert retry_resp.status_code == 200
    retry_payload = retry_resp.json()
    assert retry_payload["id"] != original_id
    assert retry_payload["kind"] == "backtest"


def test_create_job_rejects_unsupported_kind_with_structured_400(monkeypatch, tmp_path) -> None:
    client, _ = _build_client(monkeypatch, tmp_path)
    response = client.post("/api/jobs", json={"kind": "nope", "request": {}})
    detail = _assert_structured_error(response, status_code=400, code="unsupported_job_kind")
    assert "Unsupported job kind" in detail["detail"]
    assert "Use one of:" in detail["hint"]


def test_create_job_rejects_missing_kind_with_structured_422(monkeypatch, tmp_path) -> None:
    client, _ = _build_client(monkeypatch, tmp_path)
    response = client.post("/api/jobs", json={"request": {}})
    detail = _assert_structured_error(response, status_code=422, code="request_validation_error")
    assert isinstance(detail["errors"], list)
    assert detail["errors"]


def test_create_job_rejects_non_object_request_with_structured_422(monkeypatch, tmp_path) -> None:
    client, _ = _build_client(monkeypatch, tmp_path)
    response = client.post("/api/jobs", json={"kind": "backtest", "request": []})
    detail = _assert_structured_error(response, status_code=422, code="request_validation_error")
    assert isinstance(detail["errors"], list)
    assert detail["errors"]


def test_create_job_rejects_invalid_backtest_payload_with_structured_422(monkeypatch, tmp_path) -> None:
    client, _ = _build_client(monkeypatch, tmp_path)
    response = client.post(
        "/api/jobs",
        json={
            "kind": "backtest",
            "request": {
                "factor_name": "",
                "start_date": "2020-01-01",
                "end_date": "2021-01-01",
                "top_n": 20,
            },
        },
    )
    detail = _assert_structured_error(response, status_code=422, code="job_validation_error")
    assert "Invalid request payload" in detail["detail"]
    assert isinstance(detail["errors"], list)
    assert detail["errors"]


def test_create_job_rejects_invalid_analytics_payload_with_structured_422(monkeypatch, tmp_path) -> None:
    client, _ = _build_client(monkeypatch, tmp_path)
    response = client.post(
        "/api/jobs",
        json={
            "kind": "analytics",
            "request": {
                "methods": ["performance"],
            },
        },
    )
    detail = _assert_structured_error(response, status_code=422, code="job_validation_error")
    assert "analytics" in detail["detail"]
    assert isinstance(detail["errors"], list)
    assert detail["errors"]


def test_create_job_rejects_optimize_without_parameter_space(monkeypatch, tmp_path) -> None:
    client, _ = _build_client(monkeypatch, tmp_path)
    response = client.post(
        "/api/jobs",
        json={
            "kind": "optimize",
            "request": {
                "signal": "momentum",
                "params": {"start_date": "2020-01-01", "end_date": "2021-01-01"},
                "method": "grid",
                "parameter_space": [],
            },
        },
    )
    detail = _assert_structured_error(response, status_code=422, code="job_validation_error")
    assert "parameter_space" in detail["detail"]


def test_get_job_rejects_unknown_id_with_structured_404(monkeypatch, tmp_path) -> None:
    client, _ = _build_client(monkeypatch, tmp_path)
    detail = _assert_structured_error(
        client.get("/api/jobs/not-a-real-id"),
        status_code=404,
        code="job_not_found",
    )
    assert "not found" in detail["detail"].lower()


def test_cancel_job_rejects_unknown_id_with_structured_404(monkeypatch, tmp_path) -> None:
    client, _ = _build_client(monkeypatch, tmp_path)
    detail = _assert_structured_error(
        client.delete("/api/jobs/not-a-real-id"),
        status_code=404,
        code="job_not_found",
    )
    assert "not found" in detail["detail"].lower()


def test_retry_job_rejects_unknown_id_with_structured_404(monkeypatch, tmp_path) -> None:
    client, _ = _build_client(monkeypatch, tmp_path)
    detail = _assert_structured_error(
        client.post("/api/jobs/not-a-real-id/retry"),
        status_code=404,
        code="job_not_found",
    )
    assert "not found" in detail["detail"].lower()


def test_retry_job_rejects_missing_request_payload_with_structured_400(monkeypatch, tmp_path) -> None:
    client, api_main = _build_client(monkeypatch, tmp_path)

    monkeypatch.setattr(api_main, "_run_backtest_request", lambda payload: {"ok": True})
    created = client.post(
        "/api/jobs",
        json={
            "kind": "backtest",
            "request": {"factor_name": "momentum", "start_date": "2020-01-01", "end_date": "2021-01-01"},
        },
    )
    assert created.status_code == 200
    job_id = created.json()["id"]

    record = api_main.job_manager.get(job_id)
    assert record is not None
    record.request = {}

    detail = _assert_structured_error(
        client.post(f"/api/jobs/{job_id}/retry"),
        status_code=400,
        code="job_request_missing",
    )
    assert "payload missing" in detail["detail"].lower()


def test_retry_job_rejects_unsupported_stored_kind_with_structured_400(monkeypatch, tmp_path) -> None:
    client, api_main = _build_client(monkeypatch, tmp_path)

    monkeypatch.setattr(api_main, "_run_backtest_request", lambda payload: {"ok": True})
    created = client.post(
        "/api/jobs",
        json={
            "kind": "backtest",
            "request": {"factor_name": "momentum", "start_date": "2020-01-01", "end_date": "2021-01-01"},
        },
    )
    assert created.status_code == 200
    job_id = created.json()["id"]

    record = api_main.job_manager.get(job_id)
    assert record is not None
    record.kind = "legacy"

    detail = _assert_structured_error(
        client.post(f"/api/jobs/{job_id}/retry"),
        status_code=400,
        code="retry_unsupported_job_kind",
    )
    assert "unsupported job kind" in detail["detail"].lower()


def test_invalid_payload_errors_are_deterministic(monkeypatch, tmp_path) -> None:
    client, _ = _build_client(monkeypatch, tmp_path)

    payload = {
        "kind": "backtest",
        "request": {"factor_name": "", "start_date": "2020-01-01", "end_date": "2021-01-01"},
    }
    first = client.post("/api/jobs", json=payload)
    second = client.post("/api/jobs", json=payload)

    first_detail = _assert_structured_error(first, status_code=422, code="job_validation_error")
    second_detail = _assert_structured_error(second, status_code=422, code="job_validation_error")
    assert first_detail["detail"] == second_detail["detail"]
    assert first_detail.get("hint") == second_detail.get("hint")
