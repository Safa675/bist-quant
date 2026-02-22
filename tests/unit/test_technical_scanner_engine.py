"""Unit tests for TechnicalScannerEngine."""

from __future__ import annotations

import pandas as pd

from bist_quant.engines.technical_scanner import TechnicalScannerEngine


class _DummyBorsapy:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def scan(self, universe, condition, interval):  # noqa: ANN001
        self.calls.append(
            {
                "universe": universe,
                "condition": condition,
                "interval": interval,
            }
        )
        if condition == "rsi < 30":
            return pd.DataFrame(
                {
                    "Ticker": ["thyao.is", "garan"],
                    "rsi": [24.5, 28.0],
                }
            )
        if condition == "macd crosses_above signal":
            return pd.DataFrame(
                {
                    "symbol": ["THYAO", "AKBNK"],
                    "macd": [1.2, 0.8],
                }
            )
        return pd.DataFrame()


def test_scan_normalizes_symbol() -> None:
    engine = TechnicalScannerEngine(borsapy_module=_DummyBorsapy())
    result = engine.scan(universe="XU100", condition="rsi < 30", interval="1d")

    assert not result.empty
    assert list(result["symbol"]) == ["THYAO", "GARAN"]
    assert "rsi" in result.columns


def test_scan_multi_intersection() -> None:
    engine = TechnicalScannerEngine(borsapy_module=_DummyBorsapy())
    result = engine.scan_multi(
        universe="XU100",
        conditions=["rsi < 30", "macd crosses_above signal"],
        interval="1d",
    )

    assert not result.empty
    assert list(result["symbol"]) == ["THYAO"]
    assert int(result["matched_conditions"].iloc[0]) == 2


def test_predefined_scans_returns_copy() -> None:
    engine = TechnicalScannerEngine(borsapy_module=_DummyBorsapy())
    presets = engine.predefined_scans()
    presets["oversold"] = "mutated"

    assert engine.PREDEFINED["oversold"] == "rsi < 30"


def test_scan_without_supported_backend_returns_empty() -> None:
    engine = TechnicalScannerEngine(borsapy_module=object())
    result = engine.scan(universe="XU100", condition="rsi < 30", interval="1d")

    assert isinstance(result, pd.DataFrame)
    assert "symbol" in result.columns
    assert result.empty
