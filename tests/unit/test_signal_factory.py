"""Unit tests for signal factory helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from bist_quant.signals import factory


class TestSignalFactory:
    """Tests for signal factory functionality."""

    def test_get_available_signals(self) -> None:
        """Test available signals are discoverable."""
        signals = factory.get_available_signals()
        assert isinstance(signals, list)
        assert len(signals) > 0
        assert "ta_consensus" in signals
        assert "sovereign_risk" in signals

    def test_unknown_signal_raises(self) -> None:
        """Test unknown signal name raises clear error."""
        with pytest.raises(ValueError):
            factory.build_signal(
                name="nonexistent_signal_xyz",
                dates=pd.date_range("2024-01-01", periods=3, freq="D"),
                loader=object(),
                config={},
            )

    def test_build_signal_uses_merged_params(self, monkeypatch) -> None:
        """Test builder receives merged legacy and signal params."""
        captured = {}

        def dummy_builder(dates, loader, config, signal_params):
            captured["params"] = dict(signal_params)
            return pd.DataFrame({"THYAO": [1.0] * len(dates)}, index=dates)

        monkeypatch.setitem(factory.BUILDERS, "dummy_signal", dummy_builder)

        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        result = factory.build_signal(
            name="dummy_signal",
            dates=dates,
            loader=object(),
            config={"parameters": {"window": 10}, "signal_params": {"threshold": 0.2}},
        )

        assert isinstance(result, pd.DataFrame)
        assert captured["params"]["window"] == 10
        assert captured["params"]["threshold"] == 0.2
