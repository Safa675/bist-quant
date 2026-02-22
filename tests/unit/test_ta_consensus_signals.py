"""Unit tests for TradingView TA consensus signal integration."""

from __future__ import annotations

import pandas as pd

from bist_quant.signals.ta_consensus_signals import (
    TAConsensusSignals,
    build_ta_consensus_from_config,
)


class _DummyTicker:
    def __init__(self, symbol: str, payloads: dict[str, dict], calls: list[tuple[str, str]]) -> None:
        self._symbol = symbol
        self._payloads = payloads
        self._calls = calls

    def ta_signals(self, interval: str = "1d") -> dict:
        self._calls.append((self._symbol, interval))
        payload = self._payloads.get(self._symbol)
        if payload is None:
            raise RuntimeError(f"missing payload for {self._symbol}")
        return payload


class _DummyBorsapy:
    def __init__(self, payloads: dict[str, dict], calls: list[tuple[str, str]]) -> None:
        self._payloads = payloads
        self._calls = calls

    def Ticker(self, symbol: str) -> _DummyTicker:  # noqa: N802 - external API shape
        return _DummyTicker(symbol=symbol, payloads=self._payloads, calls=self._calls)


def test_build_consensus_panel_maps_scores_and_counts() -> None:
    calls: list[tuple[str, str]] = []
    payloads = {
        "THYAO": {
            "summary": {"recommendation": "STRONG_BUY", "buy": 21, "sell": 2, "neutral": 5},
            "oscillators": {"recommendation": "BUY"},
            "moving_averages": {"recommendation": "STRONG_BUY"},
        },
        "GARAN": {
            "summary": {"recommendation": "SELL", "buy": 4, "sell": 16, "neutral": 8},
            "oscillators": {"recommendation": "NEUTRAL"},
            "moving_averages": {"recommendation": "SELL"},
        },
    }
    builder = TAConsensusSignals(borsapy_module=_DummyBorsapy(payloads, calls))

    panel = builder.build_consensus_panel(["THYAO.IS", "GARAN"], interval="1d")

    assert list(panel["symbol"]) == ["THYAO", "GARAN"]
    assert panel.loc[panel["symbol"] == "THYAO", "consensus_score"].iloc[0] == 1.0
    assert panel.loc[panel["symbol"] == "GARAN", "consensus_score"].iloc[0] == -0.5
    assert panel.loc[panel["symbol"] == "THYAO", "buy_count"].iloc[0] == 21
    assert panel.loc[panel["symbol"] == "GARAN", "sell_count"].iloc[0] == 16
    assert calls == [("THYAO", "1d"), ("GARAN", "1d")]


def test_build_oscillator_panel_collects_indicator_votes() -> None:
    calls: list[tuple[str, str]] = []
    payloads = {
        "THYAO": {
            "summary": {"recommendation": "BUY"},
            "oscillators": {"compute": {"RSI": "BUY", "MACD": "STRONG_SELL"}},
            "moving_averages": {},
        },
    }
    builder = TAConsensusSignals(borsapy_module=_DummyBorsapy(payloads, calls))

    panel = builder.build_oscillator_panel(["THYAO"], interval="1W")

    assert list(panel.index) == ["THYAO"]
    assert panel.loc["THYAO", "RSI"] == "BUY"
    assert panel.loc["THYAO", "MACD"] == "STRONG_SELL"
    assert calls == [("THYAO", "1W")]


def test_build_signal_panel_broadcasts_consensus_and_fills_missing() -> None:
    calls: list[tuple[str, str]] = []
    payloads = {
        "THYAO": {
            "summary": {"recommendation": "BUY", "buy": 10, "sell": 6, "neutral": 12},
            "oscillators": {},
            "moving_averages": {},
        },
    }
    dates = pd.date_range("2025-01-01", periods=3, freq="D")
    builder = TAConsensusSignals(borsapy_module=_DummyBorsapy(payloads, calls))

    panel = builder.build_signal_panel(["THYAO", "GARAN"], dates, interval="1d", fillna_value=0.0)

    assert panel.shape == (3, 2)
    assert list(panel.columns) == ["THYAO", "GARAN"]
    assert (panel["THYAO"] == 0.5).all()
    assert (panel["GARAN"] == 0.0).all()


def test_build_consensus_panel_applies_rate_limiting(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    payloads = {
        symbol: {
            "summary": {"recommendation": "NEUTRAL", "buy": 0, "sell": 0, "neutral": 28},
            "oscillators": {},
            "moving_averages": {},
        }
        for symbol in ["AAA", "BBB", "CCC", "DDD", "EEE"]
    }
    sleep_calls: list[float] = []
    monkeypatch.setattr(TAConsensusSignals, "_sleep", staticmethod(lambda seconds: sleep_calls.append(seconds)))

    builder = TAConsensusSignals(
        borsapy_module=_DummyBorsapy(payloads, calls),
        batch_size=2,
        request_sleep_seconds=0.1,
        batch_pause_seconds=0.5,
    )
    builder.build_consensus_panel(["AAA", "BBB", "CCC", "DDD", "EEE"])

    assert sleep_calls.count(0.1) == 4  # between requests
    assert sleep_calls.count(0.5) == 2  # between batches


def test_build_ta_consensus_from_config_uses_runtime_symbols(monkeypatch) -> None:
    captured: dict[str, object] = {}
    dates = pd.date_range("2025-03-01", periods=2, freq="D")
    close_df = pd.DataFrame(
        {"THYAO": [100.0, 101.0], "GARAN": [50.0, 51.0]},
        index=dates,
    )

    def fake_build_signal_panel(self, symbols, dates, interval="1d", fillna_value=0.0):  # noqa: ANN001
        captured["symbols"] = list(symbols)
        captured["interval"] = interval
        captured["fillna_value"] = fillna_value
        return pd.DataFrame({"THYAO": [1.0, 1.0], "GARAN": [0.0, 0.0]}, index=dates)

    monkeypatch.setattr(TAConsensusSignals, "build_signal_panel", fake_build_signal_panel)

    out = build_ta_consensus_from_config(
        dates=dates,
        loader=object(),
        config={"_runtime_context": {"close_df": close_df}},
        signal_params={"interval": "1W", "batch_size": 10, "fillna_value": 0.0},
    )

    assert captured["symbols"] == ["THYAO", "GARAN"]
    assert captured["interval"] == "1W"
    assert captured["fillna_value"] == 0.0
    assert list(out.columns) == ["THYAO", "GARAN"]
    assert out.shape == (2, 2)

