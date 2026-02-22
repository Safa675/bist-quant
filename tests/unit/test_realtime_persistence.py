"""Unit tests for bist_quant.realtime and bist_quant.persistence."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Mock borsapy before any bist_quant import to prevent network hangs
if "borsapy" not in sys.modules:
    sys.modules["borsapy"] = MagicMock()

from bist_quant.realtime.quotes import (
    normalize_change_pct,
    normalize_symbol,
    to_float,
)
from bist_quant.realtime.ticks import (
    fallback_realtime_ticks,
    normalize_realtime_symbols,
)
from bist_quant.persistence import RunStore, RunStoreError


# ---------------------------------------------------------------------------
# quotes.py
# ---------------------------------------------------------------------------


class TestNormalizeSymbol:
    def test_basic(self):
        assert normalize_symbol("thyao") == "THYAO"

    def test_with_exchange(self):
        assert normalize_symbol("thyao.e") == "THYAO"

    def test_empty(self):
        assert normalize_symbol("") == ""

    def test_whitespace(self):
        assert normalize_symbol("  akbnk  ") == "AKBNK"


class TestToFloat:
    def test_valid(self):
        assert to_float("3.14") == 3.14

    def test_int(self):
        assert to_float(42) == 42.0

    def test_none(self):
        assert to_float(None) is None

    def test_nan(self):
        assert to_float(float("nan")) is None

    def test_inf(self):
        assert to_float(float("inf")) is None

    def test_garbage(self):
        assert to_float("not_a_number") is None


class TestNormalizeChangePct:
    def test_both_match(self):
        result = normalize_change_pct(5.0, 105.0, 100.0)
        assert result == 5.0

    def test_provided_only(self):
        result = normalize_change_pct(3.5, None, None)
        assert result == 3.5

    def test_computed_only(self):
        result = normalize_change_pct(None, 110.0, 100.0)
        assert math.isclose(result, 10.0, rel_tol=0.001)

    def test_decimal_correction(self):
        # When provided is in decimal form (0.05 instead of 5.0)
        result = normalize_change_pct(0.05, 105.0, 100.0)
        assert result == 5.0

    def test_all_none(self):
        assert normalize_change_pct(None, None, None) is None

    def test_zero_prev_close(self):
        assert normalize_change_pct(None, 100.0, 0.0) is None


# ---------------------------------------------------------------------------
# ticks.py
# ---------------------------------------------------------------------------


class TestNormalizeRealtimeSymbols:
    def test_string(self):
        assert normalize_realtime_symbols("THYAO,AKBNK") == ["THYAO", "AKBNK"]

    def test_list(self):
        assert normalize_realtime_symbols(["thyao", "garan"]) == ["THYAO", "GARAN"]

    def test_deduplicate(self):
        assert normalize_realtime_symbols("THYAO,THYAO,AKBNK") == ["THYAO", "AKBNK"]

    def test_empty(self):
        assert normalize_realtime_symbols([]) == ["XU100"]

    def test_max_20(self):
        syms = [f"SYM{i}" for i in range(30)]
        result = normalize_realtime_symbols(syms)
        assert len(result) == 20


class TestFallbackTicks:
    def test_basic(self):
        ticks = fallback_realtime_ticks(["THYAO", "AKBNK"], {})
        assert len(ticks) == 2
        assert ticks[0]["symbol"] == "THYAO"
        assert "price" in ticks[0]
        assert "change_pct" in ticks[0]
        assert "volume" in ticks[0]

    def test_with_previous(self):
        prev = {"THYAO": {"price": 50.0, "volume": 500000.0}}
        ticks = fallback_realtime_ticks(["THYAO"], prev)
        assert len(ticks) == 1
        # Price should be close to 50 (small drift)
        assert 45.0 < ticks[0]["price"] < 55.0


# ---------------------------------------------------------------------------
# RunStore
# ---------------------------------------------------------------------------


class TestRunStore:
    @pytest.fixture
    def store(self, tmp_path):
        return RunStore(
            store_path=tmp_path / "runs.json",
            artifacts_dir=tmp_path / "artifacts",
        )

    def test_create_run(self, store):
        run = store.create_or_update_run(
            run_id="test_run_1",
            kind="backtest",
            request_payload={"strategy": "momentum"},
            status="running",
        )
        assert run["id"] == "test_run_1"
        assert run["kind"] == "backtest"
        assert run["status"] == "running"
        assert run["started_at"] is not None

    def test_update_run(self, store):
        store.create_or_update_run(
            run_id="r1", kind="backtest",
            request_payload={"x": 1}, status="running",
        )
        updated = store.create_or_update_run(
            run_id="r1", kind="backtest",
            request_payload={"x": 1}, status="succeeded",
        )
        assert updated["status"] == "succeeded"
        assert updated["finished_at"] is not None

    def test_list_runs(self, store):
        for i in range(5):
            store.create_or_update_run(
                run_id=f"run_{i}", kind="backtest",
                request_payload={}, status="succeeded",
            )
        result = store.list_runs(limit=3)
        assert result["total"] == 5
        assert len(result["runs"]) == 3

    def test_list_runs_filter_kind(self, store):
        store.create_or_update_run(run_id="a", kind="backtest", request_payload={}, status="running")
        store.create_or_update_run(run_id="b", kind="factor_lab", request_payload={}, status="running")
        result = store.list_runs(kind="backtest")
        assert result["total"] == 1
        assert result["runs"][0]["kind"] == "backtest"

    def test_get_run(self, store):
        store.create_or_update_run(run_id="x1", kind="bt", request_payload={}, status="running")
        assert store.get_run("x1") is not None
        assert store.get_run("nonexistent") is None

    def test_find_by_meta(self, store):
        store.create_or_update_run(
            run_id="m1", kind="bt", request_payload={},
            status="running", meta={"job_id": "j123"},
        )
        found = store.find_run_by_meta(key="job_id", value="j123")
        assert found is not None
        assert found["id"] == "m1"

    def test_save_and_read_artifact(self, store):
        store.create_or_update_run(run_id="a1", kind="bt", request_payload={}, status="succeeded")
        artifact = store.save_artifact(kind="bt", run_id="a1", payload={"data": [1, 2, 3]})
        assert "id" in artifact
        assert artifact["size_bytes"] > 0

        result = store.read_artifact(artifact["id"])
        assert result is not None
        path, data = result
        parsed = json.loads(data)
        assert parsed["data"] == [1, 2, 3]

    def test_read_nonexistent_artifact(self, store):
        assert store.read_artifact("does_not_exist") is None

    def test_auto_generate_run_id(self, store):
        run = store.create_or_update_run(
            run_id=None, kind="backtest",
            request_payload={}, status="queued",
        )
        assert run["id"].startswith("backtest_")

    def test_merge_meta(self, store):
        store.create_or_update_run(
            run_id="mm1", kind="bt", request_payload={},
            status="running", meta={"a": 1},
        )
        updated = store.create_or_update_run(
            run_id="mm1", kind="bt", request_payload={},
            status="running", meta={"b": 2},
        )
        assert updated["meta"]["a"] == 1
        assert updated["meta"]["b"] == 2
