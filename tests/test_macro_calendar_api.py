"""Tests for macro calendar service and API route wiring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import bist_quant.common.data_loader as data_loader_module
from bist_quant.api import create_app
from bist_quant.services import SystemService


def test_system_service_get_macro_calendar_serializes_events(monkeypatch, tmp_path: Path) -> None:
    class _DummyCalendar:
        @staticmethod
        def get_events(
            period: str = "1w",
            country: str | list[str] | None = None,
            importance: str | None = None,
        ) -> pd.DataFrame:
            del period, country, importance
            return pd.DataFrame(
                [
                    {
                        "Date": pd.Timestamp("2026-02-21"),
                        "Time": "10:00",
                        "Country": "TR",
                        "Importance": "high",
                        "Event": "CBRT Decision",
                        "Forecast": pd.NA,
                        "Previous": "42.5",
                    }
                ]
            )

    class _DummyLoader:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            self.economic_calendar = _DummyCalendar()

    monkeypatch.setattr(data_loader_module, "DataLoader", _DummyLoader)

    project_root = tmp_path / "project"
    (project_root / "data").mkdir(parents=True, exist_ok=True)
    service = SystemService(project_root=project_root, app_data_dir=str(tmp_path / "app"))

    payload = service.get_macro_calendar(period="1w", country=["TR"], importance="high")

    assert payload["count"] == 1
    assert payload["country"] == ["TR"]
    assert payload["importance"] == "high"
    assert payload["events"][0]["Country"] == "TR"
    assert str(payload["events"][0]["Date"]).startswith("2026-02-21")


def test_api_macro_calendar_route_registered() -> None:
    app = create_app()
    paths = {route.path for route in app.routes}
    assert "/api/macro/calendar" in paths
