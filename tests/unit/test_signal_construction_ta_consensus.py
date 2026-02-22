"""Unit tests for TA consensus integration in signal construction engine."""

from __future__ import annotations

import pandas as pd

from bist_quant.engines import signal_construction


def test_build_signal_panel_for_ta_consensus(monkeypatch) -> None:
    dates = pd.date_range("2025-01-01", periods=3, freq="D")
    close_df = pd.DataFrame({"THYAO": [1.0, 2.0, 3.0], "GARAN": [1.0, 2.0, 3.0]}, index=dates)
    high_df = close_df.copy()
    low_df = close_df.copy()

    def fake_build_consensus_panel(self, symbols, interval="1d"):  # noqa: ANN001
        assert symbols == ["THYAO", "GARAN"]
        assert interval == "1d"
        return pd.DataFrame(
            {
                "symbol": ["THYAO", "GARAN"],
                "consensus_score": [1.0, -0.5],
            }
        )

    monkeypatch.setattr(
        signal_construction.TAConsensusSignals,
        "build_consensus_panel",
        fake_build_consensus_panel,
    )

    values_panel, signal_panel = signal_construction._build_signal_panel_for_indicator(
        "ta_consensus",
        params={"interval": "1d", "batch_size": 10},
        close_df=close_df,
        high_df=high_df,
        low_df=low_df,
    )

    assert values_panel.shape == close_df.shape
    assert signal_panel.shape == close_df.shape
    assert (values_panel["THYAO"] == 1.0).all()
    assert (values_panel["GARAN"] == -0.5).all()
    assert (signal_panel["THYAO"] == 1).all()
    assert (signal_panel["GARAN"] == -1).all()


def test_get_signal_metadata_contains_ta_consensus() -> None:
    metadata = signal_construction.get_signal_metadata()
    keys = [row["key"] for row in metadata["indicators"]]
    assert "ta_consensus" in keys

