"""Unit tests for API JobManager persistence behavior."""

from __future__ import annotations

import time
from pathlib import Path

from bist_quant.api.jobs import JobManager


def test_job_manager_persists_records(tmp_path: Path) -> None:
    store_path = tmp_path / "api_jobs.json"
    manager = JobManager(max_workers=1, store_path=store_path)

    record = manager.create(
        kind="backtest",
        fn=lambda: {"metrics": {"cagr": 0.11}},
        meta={"factor_name": "momentum"},
        request={"factor_name": "momentum", "start_date": "2020-01-01", "end_date": "2021-01-01"},
    )

    for _ in range(40):
        loaded = manager.get(record.id)
        if loaded is not None and loaded.status == "completed":
            break
        time.sleep(0.03)

    assert store_path.exists()

    manager_reloaded = JobManager(max_workers=1, store_path=store_path)
    reloaded = manager_reloaded.get(record.id)
    assert reloaded is not None
    assert reloaded.kind == "backtest"
    assert reloaded.meta.get("factor_name") == "momentum"
    assert reloaded.request.get("factor_name") == "momentum"
