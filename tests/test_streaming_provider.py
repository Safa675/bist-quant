from __future__ import annotations

from typing import Any

import pytest

from bist_quant.realtime.quotes import get_quote_streaming
from bist_quant.realtime.streaming import StreamingProvider
from bist_quant.realtime.ticks import fetch_realtime_ticks


class _FakeTradingViewStream:
    last_instance: "_FakeTradingViewStream | None" = None

    def __init__(self, auth_token: str | None = None) -> None:
        self.auth_token = auth_token
        self.is_connected = False
        self.connect_timeout: float | None = None
        self.disconnected = False
        self.subscribed: set[str] = set()
        self.subscribe_calls: list[tuple[str, str]] = []
        self.chart_subscriptions: dict[str, set[str]] = {}
        self.quote_callbacks: dict[str, list[Any]] = {}
        _FakeTradingViewStream.last_instance = self

    def connect(self, timeout: float = 10.0) -> bool:
        self.is_connected = True
        self.connect_timeout = timeout
        return True

    def disconnect(self) -> None:
        self.disconnected = True
        self.is_connected = False

    def subscribe(self, symbol: str, exchange: str = "BIST") -> None:
        self.subscribe_calls.append((symbol, exchange))
        _ = exchange
        self.subscribed.add(symbol)

    def subscribe_chart(self, symbol: str, interval: str = "1m", exchange: str = "BIST") -> None:
        _ = exchange
        self.chart_subscriptions.setdefault(symbol, set()).add(interval)

    def get_quote(self, symbol: str) -> dict[str, Any] | None:
        if symbol not in self.subscribed:
            return None
        return {
            "symbol": symbol,
            "last": 101.25,
            "change_percent": 1.5,
            "change": 1.5,
            "open": 100.0,
            "high": 102.0,
            "low": 99.5,
            "prev_close": 99.75,
            "volume": 1_250_000,
            "market_cap_basic": 123_000_000_000,
            "_updated": 1_736_600_000,
        }

    def get_all_quotes(self) -> dict[str, dict[str, Any]]:
        return {symbol: self.get_quote(symbol) for symbol in self.subscribed if self.get_quote(symbol)}

    def get_candle(self, symbol: str, interval: str) -> dict[str, Any] | None:
        if symbol not in self.subscribed:
            return None
        return {"time": 1_736_600_000, "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 10_000}

    def on_quote(self, symbol: str, callback) -> None:
        self.quote_callbacks.setdefault(symbol, []).append(callback)


class _FakeBP:
    TradingViewStream = _FakeTradingViewStream


def test_streaming_provider_connect_subscribe_and_get_quote() -> None:
    provider = StreamingProvider(auth_config={"auth_token": "token-abc", "connect_timeout": 2}, bp=_FakeBP())
    provider.subscribe(["thyao"])

    quote = provider.get_quote("THYAO")
    assert quote["symbol"] == "THYAO"
    assert quote["last_price"] == pytest.approx(101.25)
    assert quote["change_pct"] == pytest.approx(1.5)
    assert quote["volume"] == pytest.approx(1_250_000)

    stream = _FakeTradingViewStream.last_instance
    assert stream is not None
    assert stream.auth_token == "token-abc"
    assert stream.connect_timeout == pytest.approx(2.0)
    assert "THYAO" in stream.subscribed


def test_streaming_provider_callbacks_and_context_manager() -> None:
    events: list[tuple[str, dict[str, Any]]] = []

    def _on_quote(symbol: str, quote: dict[str, Any]) -> None:
        events.append((symbol, quote))

    with StreamingProvider(bp=_FakeBP()) as provider:
        provider.on_quote("THYAO", _on_quote)
        provider.subscribe(["THYAO"])

    stream = _FakeTradingViewStream.last_instance
    assert stream is not None
    assert stream.disconnected is True
    assert "THYAO" in stream.quote_callbacks
    assert len(stream.quote_callbacks["THYAO"]) == 1
    assert events == []


def test_streaming_provider_viop_subscription_and_quote() -> None:
    provider = StreamingProvider(bp=_FakeBP())

    provider.subscribe_viop(["xu030d0326"])
    quote = provider.get_viop_quote("XU030D0326")

    stream = _FakeTradingViewStream.last_instance
    assert stream is not None
    assert "XU030D0326" in stream.subscribed
    assert ("XU030D0326", "VIOP") in stream.subscribe_calls
    assert quote["symbol"] == "XU030D0326"
    assert quote["last_price"] == pytest.approx(101.25)


def test_get_quote_streaming_falls_back_to_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    class _NoDataProvider:
        def subscribe(self, symbols: list[str]) -> None:
            _ = symbols

        def get_quote(self, symbol: str) -> dict[str, Any]:
            return {"symbol": symbol, "error": "no streamed data"}

    monkeypatch.setattr("bist_quant.realtime.quotes._get_streaming_provider", lambda auth_config=None: _NoDataProvider())
    monkeypatch.setattr(
        "bist_quant.realtime.quotes._quote_payload",
        lambda symbol: {"symbol": symbol, "last_price": 88.5, "timestamp": "fallback"},
    )

    payload = get_quote_streaming("THYAO")
    assert payload["symbol"] == "THYAO"
    assert payload["last_price"] == pytest.approx(88.5)


def test_fetch_realtime_ticks_prefers_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    def _streamed(symbols: list[str], auth_config=None) -> dict[str, Any]:
        _ = auth_config
        return {
            "quotes": {
                symbol: {
                    "symbol": symbol,
                    "last_price": 100.0 + idx,
                    "change_pct": 1.0,
                    "volume": 500_000.0,
                    "timestamp": "2026-02-21T12:00:00+00:00",
                }
                for idx, symbol in enumerate(symbols)
            },
            "count": len(symbols),
            "timestamp": "2026-02-21T12:00:00+00:00",
        }

    # Patch in module namespace used by fetch_realtime_ticks.
    import bist_quant.realtime.ticks as ticks_module

    monkeypatch.setattr(ticks_module, "get_quotes_streaming", _streamed)
    ticks = fetch_realtime_ticks(["THYAO", "AKBNK"], streaming=True)
    assert len(ticks) == 2
    assert ticks[0]["symbol"] == "THYAO"
    assert ticks[1]["symbol"] == "AKBNK"
