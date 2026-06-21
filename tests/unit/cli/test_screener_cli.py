"""Unit tests for screener CLI."""

from __future__ import annotations

import json
from argparse import Namespace

import pytest

from bist_quant.cli.screener_cli import cmd_screener_metadata, cmd_screener_run, cmd_scan


def test_cmd_screener_metadata_json(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        "bist_quant.cli.screener_cli.get_screener_metadata",
        lambda: {"indexes": ["XU100"], "templates": ["high_dividend"], "filters": [], "data_sources": ["local"]},
    )
    cmd_screener_metadata(Namespace(json=True))
    payload = json.loads(capsys.readouterr().out)
    assert payload["indexes"] == ["XU100"]


def test_cmd_screener_run_json(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        "bist_quant.cli.screener_cli.run_screener",
        lambda payload: {"meta": {"returned_rows": 1}, "rows": [{"symbol": "THYAO"}], "columns": [], "applied_filters": [], "chart": {}},
    )
    cmd_screener_run(
        Namespace(
            data_source="local",
            limit=5,
            sort_by="symbol",
            ascending=False,
            index="XU100",
            template="",
            sector="",
            recommendation="",
            symbols="",
            technical_scan="",
            refresh_cache=False,
            json=True,
        )
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["rows"][0]["symbol"] == "THYAO"


def test_cmd_scan_json(capsys, monkeypatch) -> None:
    import pandas as pd

    class FakeScanner:
        def scan(self, universe, condition, interval, template=None):  # noqa: ANN001
            return pd.DataFrame({"symbol": ["THYAO"], "rsi": [28.0]})

    monkeypatch.setattr("bist_quant.cli.screener_cli.TechnicalScanner", FakeScanner)
    cmd_scan(
        Namespace(
            universe="XU100",
            symbols="",
            condition="rsi < 30",
            template="",
            conditions="",
            interval="1d",
            json=True,
        )
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["symbol"] == "THYAO"


def test_cmd_screener_run_exits_on_screening_error(capsys, monkeypatch) -> None:
    from bist_quant.screening.errors import ScreeningValidationError

    monkeypatch.setattr(
        "bist_quant.cli.screener_cli.run_screener",
        lambda payload: (_ for _ in ()).throw(ScreeningValidationError("bad", user_message="bad payload")),
    )
    with pytest.raises(SystemExit) as exc:
        cmd_screener_run(
            Namespace(
                data_source="local",
                limit=5,
                sort_by="symbol",
                ascending=False,
                index="",
                template="",
                sector="",
                recommendation="",
                symbols="",
                technical_scan="",
                refresh_cache=False,
                json=False,
            )
        )
    assert exc.value.code == 1
