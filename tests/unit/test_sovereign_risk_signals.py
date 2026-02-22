"""Unit tests for sovereign risk signal overlays."""

from __future__ import annotations

import pandas as pd

from bist_quant.signals import sovereign_risk_signals as srs


class _DummyMacroFeatures:
    def __init__(self, stress: pd.Series) -> None:
        self._stress = stress
        self._loaded = True

    def load(self) -> None:  # pragma: no cover - should not be called in tests
        raise AssertionError("load() should not be called for preloaded dummy macro")

    def compute_eurobond_stress(self) -> pd.Series:
        return self._stress


def test_sovereign_risk_signal_panel_applies_stress_multiplier() -> None:
    dates = pd.date_range("2026-02-16", periods=4, freq="D")
    close_df = pd.DataFrame(
        {
            "THYAO": [300.0, 302.0, 301.0, 305.0],
            "GARAN": [80.0, 81.0, 80.5, 82.0],
        },
        index=dates,
    )
    stress = pd.Series([False, False, True, True], index=dates, name="eurobond_stress")
    builder = srs.SovereignRiskSignals(
        macro_features=_DummyMacroFeatures(stress),
        calm_multiplier=1.0,
        stress_multiplier=0.3,
    )

    panel = builder.build_signal_panel(close_df=close_df, dates=dates)

    assert panel.shape == close_df.shape
    assert (panel.loc[dates[:2], "THYAO"] == 1.0).all()
    assert (panel.loc[dates[2:], "GARAN"] == 0.3).all()


def test_build_sovereign_risk_from_config_uses_runtime_context(monkeypatch) -> None:
    captured: dict[str, object] = {}
    dates = pd.date_range("2026-03-01", periods=2, freq="D")
    close_df = pd.DataFrame(
        {"THYAO": [1.0, 2.0], "GARAN": [3.0, 4.0]},
        index=dates,
    )

    def _fake_build(
        close_df: pd.DataFrame,
        dates: pd.DatetimeIndex,
        data_loader,
        *,
        calm_multiplier: float,
        stress_multiplier: float,
    ) -> pd.DataFrame:
        captured["cols"] = list(close_df.columns)
        captured["dates"] = dates
        captured["loader"] = data_loader
        captured["calm_multiplier"] = calm_multiplier
        captured["stress_multiplier"] = stress_multiplier
        return pd.DataFrame(1.0, index=dates, columns=close_df.columns)

    monkeypatch.setattr(srs, "build_sovereign_risk_signals", _fake_build)

    loader = object()
    out = srs.build_sovereign_risk_from_config(
        dates=dates,
        loader=loader,
        config={"_runtime_context": {"close_df": close_df}},
        signal_params={"calm_multiplier": 0.9, "stress_multiplier": 0.25},
    )

    assert captured["cols"] == ["THYAO", "GARAN"]
    assert captured["dates"].equals(dates)
    assert captured["loader"] is loader
    assert captured["calm_multiplier"] == 0.9
    assert captured["stress_multiplier"] == 0.25
    assert out.shape == (2, 2)
