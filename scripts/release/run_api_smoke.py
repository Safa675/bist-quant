#!/usr/bin/env python3
"""API smoke checks for release-readiness workflows."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from time import sleep, time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


BASE_URL = os.environ.get("SMOKE_API_URL", "http://127.0.0.1:8001").rstrip("/")
TIMEOUT_SECONDS = float(os.environ.get("SMOKE_HTTP_TIMEOUT", "30"))
JOB_TIMEOUT_SECONDS = float(os.environ.get("SMOKE_JOB_TIMEOUT", "360"))


@dataclass
class SmokeError(Exception):
    message: str



def _http_json(method: str, path: str, payload: Any | None = None) -> tuple[int, Any]:
    body = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")

    req = Request(f"{BASE_URL}{path}", data=body, headers=headers, method=method.upper())
    try:
        with urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw) if raw else {}
            return resp.status, data
    except HTTPError as exc:
        raw = exc.read().decode("utf-8")
        data = json.loads(raw) if raw else {"detail": raw}
        return exc.code, data



def _require(status: int, expected: int, context: str, payload: Any) -> None:
    if status != expected:
        raise SmokeError(f"{context} -> expected {expected}, got {status}: {payload}")


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _screener_stats(payload: Any, *, context: str) -> tuple[int, int, set[str]]:
    if not isinstance(payload, dict):
        raise SmokeError(f"{context} -> expected dict payload, got: {payload}")

    rows_raw = payload.get("rows")
    if not isinstance(rows_raw, list):
        rows_raw = payload.get("results")
    if not isinstance(rows_raw, list):
        rows_raw = []

    rows = [row for row in rows_raw if isinstance(row, dict)]
    row_count = len(rows)

    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    returned_rows = _as_int(meta.get("returned_rows")) or _as_int(payload.get("count")) or row_count
    total_matches = _as_int(meta.get("total_matches")) or _as_int(payload.get("count")) or row_count

    if returned_rows != row_count:
        raise SmokeError(
            f"{context} -> KPI mismatch (returned_rows={returned_rows}, actual_rows={row_count})"
        )

    symbols = {
        str(row.get("symbol")).strip().upper()
        for row in rows
        if isinstance(row.get("symbol"), str) and str(row.get("symbol")).strip()
    }
    return total_matches, row_count, symbols


def _wait_for_job(job_id: str) -> dict[str, Any]:
    deadline = time() + JOB_TIMEOUT_SECONDS
    last_payload: dict[str, Any] = {}
    while time() < deadline:
        status, payload = _http_json("GET", f"/api/jobs/{job_id}")
        _require(status, 200, f"poll job {job_id}", payload)
        if not isinstance(payload, dict):
            raise SmokeError(f"poll job {job_id} -> non-dict payload: {payload}")
        last_payload = payload
        state = str(payload.get("status", "")).lower()
        if state in {"completed", "failed", "cancelled"}:
            return payload
        sleep(2)

    raise SmokeError(f"job {job_id} did not reach terminal state within timeout: {last_payload}")



def main() -> int:
    print(f"[smoke] base_url={BASE_URL}")

    status, health = _http_json("GET", "/api/health/live")
    _require(status, 200, "health", health)
    print("[smoke] health/live PASS")

    status, meta = _http_json("GET", "/api/meta/signals")
    _require(status, 200, "meta/signals", meta)
    signals = meta.get("signals") if isinstance(meta, dict) else None
    if not isinstance(signals, list) or len(signals) < 2:
        raise SmokeError(f"meta/signals -> expected at least 2 signals, got: {meta}")
    print(f"[smoke] meta/signals PASS count={len(signals)}")

    signal_a = str(signals[0])
    signal_b = str(signals[1])

    status, analytics = _http_json(
        "POST",
        "/api/analytics/run",
        {
            "equity_curve": [
                {"date": "2024-01-02", "value": 100},
                {"date": "2024-01-03", "value": 101},
                {"date": "2024-01-04", "value": 102},
                {"date": "2024-01-05", "value": 101},
                {"date": "2024-01-08", "value": 103},
                {"date": "2024-01-09", "value": 104},
            ],
            "methods": ["performance", "rolling", "risk"],
            "include_benchmark": False,
        },
    )
    _require(status, 200, "analytics/run", analytics)
    if not isinstance(analytics, dict) or "performance" not in analytics:
        raise SmokeError(f"analytics/run -> missing performance payload: {analytics}")
    print("[smoke] analytics/run PASS")

    status, screener_base = _http_json(
        "POST",
        "/api/screener/run",
        {"index": "XU100", "limit": 60, "sort_by": "pe", "sort_desc": False},
    )
    _require(status, 200, "screener/run baseline", screener_base)
    base_total, base_rows, base_symbols = _screener_stats(screener_base, context="screener baseline")

    filtered_payloads = [
        {"filters": {"pe": {"min": 50}}},
        {"filters": {"pe": {"max": 8}}},
    ]
    changed = False
    change_reason = ""
    changed_total = base_total
    changed_rows = base_rows

    for filter_payload in filtered_payloads:
        status, screener_filtered = _http_json(
            "POST",
            "/api/screener/run",
            {
                "index": "XU100",
                "limit": 60,
                "sort_by": "pe",
                "sort_desc": False,
                **filter_payload,
            },
        )
        _require(status, 200, "screener/run filtered", screener_filtered)
        filtered_total, filtered_rows, filtered_symbols = _screener_stats(
            screener_filtered,
            context=f"screener filtered {filter_payload}",
        )

        if filtered_rows != base_rows:
            changed = True
            change_reason = "row_count"
        elif filtered_total != base_total:
            changed = True
            change_reason = "total_matches"
        elif filtered_symbols != base_symbols:
            changed = True
            change_reason = "symbol_set"

        if changed:
            changed_total = filtered_total
            changed_rows = filtered_rows
            break

    if not changed:
        raise SmokeError(
            "screener filters did not materially change output "
            f"(base_total={base_total}, base_rows={base_rows})"
        )

    print(
        "[smoke] screener PASS "
        f"base_total={base_total} base_rows={base_rows} "
        f"filtered_total={changed_total} filtered_rows={changed_rows} "
        f"change={change_reason}"
    )

    status, backtest_job = _http_json(
        "POST",
        "/api/jobs",
        {
            "kind": "backtest",
            "request": {
                "factor_name": signal_a,
                "start_date": "2021-01-01",
                "end_date": "2022-12-31",
                "top_n": 15,
                "rebalance_frequency": "monthly",
            },
        },
    )
    _require(status, 200, "jobs/backtest create", backtest_job)
    backtest_id = str(backtest_job.get("id"))
    backtest_terminal = _wait_for_job(backtest_id)
    if backtest_terminal.get("status") != "completed":
        raise SmokeError(f"backtest job did not complete: {backtest_terminal}")
    print(f"[smoke] backtest PASS job_id={backtest_id}")

    status, combine_job = _http_json(
        "POST",
        "/api/jobs",
        {
            "kind": "factor_combine",
            "request": {
                "signals": [{"name": signal_a, "weight": 0.5}, {"name": signal_b, "weight": 0.5}],
                "method": "equal",
                "start_date": "2021-01-01",
                "end_date": "2022-12-31",
            },
        },
    )
    _require(status, 200, "jobs/factor_combine create", combine_job)
    combine_id = str(combine_job.get("id"))
    combine_terminal = _wait_for_job(combine_id)
    if combine_terminal.get("status") != "completed":
        raise SmokeError(f"factor_combine job did not complete: {combine_terminal}")
    print(f"[smoke] factor_combine PASS job_id={combine_id}")

    status, optimize_job = _http_json(
        "POST",
        "/api/jobs",
        {
            "kind": "optimize",
            "request": {
                "signal": signal_a,
                "method": "grid",
                "max_trials": 6,
                "train_ratio": 0.7,
                "parameter_space": [
                    {"key": "top_n", "type": "int", "min": 10, "max": 20, "step": 5}
                ],
                "params": {
                    "start_date": "2021-01-01",
                    "end_date": "2022-12-31",
                    "rebalance_frequency": "monthly",
                    "top_n": 15,
                },
            },
        },
    )
    _require(status, 200, "jobs/optimize create", optimize_job)
    optimize_id = str(optimize_job.get("id"))
    optimize_terminal = _wait_for_job(optimize_id)
    if optimize_terminal.get("status") != "completed":
        raise SmokeError(f"optimize job did not complete: {optimize_terminal}")
    print(f"[smoke] optimize PASS job_id={optimize_id}")

    status, sig_backtest = _http_json(
        "POST",
        "/api/signal-construction/backtest",
        {
            "universe": "XU100",
            "period": "6mo",
            "interval": "1d",
            "top_n": 20,
            "indicators": {
                "rsi": {"enabled": True, "params": {"period": 14, "oversold": 30, "overbought": 70}},
                "macd": {"enabled": True, "params": {"fast": 12, "slow": 26, "signal": 9}},
            },
        },
    )
    _require(status, 200, "signal-construction/backtest", sig_backtest)
    if not isinstance(sig_backtest, dict) or "metrics" not in sig_backtest:
        raise SmokeError(f"signal-construction/backtest -> missing metrics: {sig_backtest}")
    print("[smoke] signal-construction PASS")

    rules = [
        {
            "id": "max_qty",
            "field": "quantity",
            "comparator": ">",
            "threshold": 1000,
            "message": "Quantity too large",
            "severity": "warning",
        }
    ]

    status, compliance_pass = _http_json(
        "POST",
        "/api/compliance/check",
        {
            "transaction": {
                "id": "tx-pass",
                "timestamp": "2026-01-03T10:00:00Z",
                "user_id": "usr-1",
                "order_id": "ord-1",
                "symbol": "THYAO",
                "side": "buy",
                "quantity": 10,
                "price": 100,
                "venue": "XIST",
                "strategy_id": "s1",
            },
            "rules": rules,
        },
    )
    _require(status, 200, "compliance/check PASS case", compliance_pass)
    if not bool(compliance_pass.get("passed", False)):
        raise SmokeError(f"compliance PASS case failed unexpectedly: {compliance_pass}")

    status, compliance_fail = _http_json(
        "POST",
        "/api/compliance/check",
        {
            "transaction": {
                "id": "tx-fail",
                "timestamp": "2026-01-03T10:00:00Z",
                "user_id": "usr-1",
                "order_id": "ord-2",
                "symbol": "THYAO",
                "side": "buy",
                "quantity": 5000,
                "price": 100,
                "venue": "XIST",
                "strategy_id": "s1",
            },
            "rules": rules,
        },
    )
    _require(status, 200, "compliance/check FAIL case", compliance_fail)
    if bool(compliance_fail.get("passed", True)):
        raise SmokeError(f"compliance FAIL case did not fail: {compliance_fail}")
    print("[smoke] compliance PASS/FAIL scenarios PASS")

    print("[smoke] API smoke suite completed successfully")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SmokeError as exc:
        print(f"[smoke] FAIL: {exc.message}")
        raise SystemExit(1)
