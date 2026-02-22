"""Unit tests for stock_filter data_source routing and hybrid enrichment."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bist_quant.engines import stock_filter


def _local_frame() -> tuple[pd.DataFrame, str, dict[str, str]]:
    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "name": ["AAA", "BBB"],
            "market_cap_usd": [100.0, 120.0],
            "market_cap": [3_000.0, 3_500.0],
            "pe": [12.0, 14.0],
            "forward_pe": [np.nan, np.nan],
            "pb": [1.2, 1.5],
            "dividend_yield": [3.2, 2.4],
            "upside_potential": [8.0, 5.0],
            "analyst_target_price": [np.nan, np.nan],
            "roe": [18.0, 15.0],
            "net_margin": [10.0, 9.0],
            "foreign_ratio": [np.nan, np.nan],
            "foreign_change_1w": [np.nan, np.nan],
            "foreign_change_1m": [np.nan, np.nan],
            "float_ratio": [np.nan, np.nan],
            "return_1m": [1.1, -0.3],
            "rsi_14": [48.0, 52.0],
            "macd_hist": [0.1, -0.1],
            "atr_14_pct": [2.1, 2.4],
            "recommendation": ["TUT", "TUT"],
            "sector": ["Tech", "Bank"],
        }
    )
    return frame, "2026-02-20", {"AAA": "Tech", "BBB": "Bank"}


def _isy_frame() -> tuple[pd.DataFrame, str, dict[str, str]]:
    frame = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "name": ["AAA"],
            "forward_pe": [8.2],
            "foreign_ratio": [41.0],
            "foreign_change_1w": [0.5],
            "foreign_change_1m": [1.0],
            "analyst_target_price": [150.0],
            "upside_potential": [25.0],
            "recommendation": ["AL"],
            "sector": ["Tech"],
        }
    )
    return frame, "2026-02-21", {"AAA": "Tech"}


def test_run_response_hybrid_merges_isyatirim_and_enrichment(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(stock_filter, "_resolve_paths", lambda runtime_paths: runtime_paths)
    monkeypatch.setattr(stock_filter, "_load_local_screen_frame", lambda runtime_paths, force_refresh=False: _local_frame())

    class FakeAdapter:
        pass

    fake_adapter = FakeAdapter()
    monkeypatch.setattr(
        stock_filter,
        "_load_isyatirim_screen_frame",
        lambda runtime_paths, template=None, screen_filters=None: (*_isy_frame(), fake_adapter),
    )
    monkeypatch.setattr(
        stock_filter,
        "_build_isyatirim_enrichment_frame",
        lambda adapter, symbols: pd.DataFrame(
            {
                "symbol": ["BBB"],
                "forward_pe": [7.9],
                "foreign_ratio": [38.0],
                "foreign_change_1w": [0.4],
                "foreign_change_1m": [0.9],
                "analyst_target_price": [180.0],
                "upside_potential": [30.0],
                "recommendation": ["AL"],
            }
        ),
    )

    response = stock_filter._run_response(
        {
            "data_source": "hybrid",
            "sort_by": "symbol",
            "sort_desc": False,
            "limit": 10,
            "fields": [
                "symbol",
                "forward_pe",
                "foreign_ratio",
                "recommendation",
                "upside_potential",
            ],
        },
        runtime_paths=SimpleNamespace(data_dir=tmp_path),
    )

    rows = {row["symbol"]: row for row in response["rows"]}
    assert response["meta"]["data_source"] == "hybrid"
    assert response["meta"]["isyatirim_enrichment"] is True
    assert rows["AAA"]["forward_pe"] == 8.2
    assert rows["BBB"]["forward_pe"] == 7.9
    assert rows["BBB"]["foreign_ratio"] == 38.0
    assert rows["BBB"]["recommendation"] == "AL"
    assert rows["BBB"]["upside_potential"] == 30.0


def test_run_response_uses_isyatirim_source_without_local_loader(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(stock_filter, "_resolve_paths", lambda runtime_paths: runtime_paths)

    def _fail_local(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("local loader should not be called for isyatirim mode")

    monkeypatch.setattr(stock_filter, "_load_local_screen_frame", _fail_local)
    monkeypatch.setattr(
        stock_filter,
        "_load_isyatirim_screen_frame",
        lambda runtime_paths, template=None, screen_filters=None: (*_isy_frame(), None),
    )

    response = stock_filter._run_response(
        {
            "data_source": "isyatirim",
            "sort_by": "symbol",
            "sort_desc": False,
            "limit": 10,
        },
        runtime_paths=SimpleNamespace(data_dir=tmp_path),
    )

    assert response["meta"]["data_source"] == "isyatirim"
    assert response["meta"]["returned_rows"] == 1
    assert response["rows"][0]["symbol"] == "AAA"


def test_run_response_rejects_unknown_data_source() -> None:
    with pytest.raises(ValueError):
        stock_filter._run_response({"data_source": "unknown-source"})
