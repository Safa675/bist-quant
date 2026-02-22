"""Unit tests for external consensus factor axis integration."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from bist_quant.engines import factor_lab


def test_available_factor_names_include_external_consensus() -> None:
    engine = SimpleNamespace(signal_configs={"momentum": {}, "value": {}})
    names = factor_lab._available_factor_names(engine)
    assert "external_consensus" in names
    assert "momentum" in names


def test_build_external_consensus_panel_uses_ta_builder(monkeypatch) -> None:
    dates = pd.date_range("2025-01-01", periods=2, freq="D")
    close_df = pd.DataFrame({"THYAO": [100.0, 101.0], "GARAN": [50.0, 51.0]}, index=dates)
    engine = SimpleNamespace(close_df=close_df)
    captured: dict[str, object] = {}

    def fake_build_signal_panel(self, symbols, dates, interval="1d", fillna_value=0.0):  # noqa: ANN001
        captured["symbols"] = list(symbols)
        captured["interval"] = interval
        captured["fillna_value"] = fillna_value
        return pd.DataFrame({"THYAO": [0.5, 0.5], "GARAN": [-0.5, -0.5]}, index=dates)

    monkeypatch.setattr(factor_lab.TAConsensusSignals, "build_signal_panel", fake_build_signal_panel)

    panel = factor_lab._build_external_consensus_panel(
        engine=engine,
        signal_params={"interval": "1W", "batch_size": 50, "fillna_value": 0.0},
    )

    assert panel.shape == (2, 2)
    assert captured["symbols"] == ["THYAO", "GARAN"]
    assert captured["interval"] == "1W"
    assert (panel["THYAO"] == 0.5).all()
    assert (panel["GARAN"] == -0.5).all()

