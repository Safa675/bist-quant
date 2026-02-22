"""
WebSocket-based streaming provider using borsapy TradingViewStream.

This provider wraps borsapy's persistent stream API with:
- lazy initialization (no hard dependency at import time)
- exception-safe methods (no exceptions leak to callers)
- normalized quote payloads compatible with realtime quote/tick callers
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _normalize_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper().split(".")[0]


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        parsed = float(value)
        if not math.isfinite(parsed):
            return None
        return parsed
    except Exception:
        return None


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _auth_token_from_config(auth_config: dict[str, Any]) -> str | None:
    for key in ("auth_token", "token", "tradingview_auth_token"):
        value = auth_config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


class StreamingProvider:
    """Manages a persistent TradingViewStream connection."""

    def __init__(
        self,
        auth_config: dict[str, Any] | None = None,
        bp: Any | None = None,
    ) -> None:
        self._auth_config = dict(auth_config or {})
        self._bp = bp
        self._stream: Any | None = None
        self._callbacks: dict[str, list[Callable[[str, dict[str, Any]], None]]] = {}
        self._attached_callbacks: set[tuple[str, int]] = set()

    def _load_bp(self) -> Any | None:
        if self._bp is not None:
            return self._bp
        try:
            import borsapy as bp

            self._bp = bp
        except Exception as exc:
            logger.warning("borsapy import failed for streaming provider: %s", exc)
            return None
        return self._bp

    def _connect_timeout(self) -> float:
        timeout = self._auth_config.get("connect_timeout", 10.0)
        try:
            return max(1.0, float(timeout))
        except Exception:
            return 10.0

    def _ensure_stream(self) -> Any | None:
        if self._stream is not None:
            return self._stream

        bp = self._load_bp()
        if bp is None:
            return None

        stream_cls = getattr(bp, "TradingViewStream", None)
        if stream_cls is None:
            logger.warning("borsapy.TradingViewStream is unavailable")
            return None

        token = _auth_token_from_config(self._auth_config)
        try:
            if token:
                self._stream = stream_cls(auth_token=token)
            else:
                self._stream = stream_cls()
        except TypeError:
            # Compatibility fallback for implementations using positional auth token.
            try:
                self._stream = stream_cls(token)
            except Exception as exc:
                logger.warning("TradingViewStream initialization failed: %s", exc)
                self._stream = None
                return None
        except Exception as exc:
            logger.warning("TradingViewStream initialization failed: %s", exc)
            self._stream = None
            return None

        self._attached_callbacks.clear()
        return self._stream

    def _is_connected(self) -> bool:
        stream = self._stream
        if stream is None:
            return False
        state = getattr(stream, "is_connected", False)
        if callable(state):
            try:
                return bool(state())
            except Exception:
                return False
        return bool(state)

    def _normalize_timestamp(self, raw: Any) -> str:
        if isinstance(raw, (int, float)):
            try:
                return datetime.fromtimestamp(float(raw), tz=timezone.utc).isoformat()
            except Exception:
                return _utc_iso_now()
        if isinstance(raw, str) and raw.strip():
            return raw
        return _utc_iso_now()

    def _normalize_quote(self, symbol: str, payload: dict[str, Any]) -> dict[str, Any]:
        last_price = _to_float(payload.get("last_price", payload.get("last")))
        prev_close = _to_float(payload.get("prev_close", payload.get("previous_close")))
        change = _to_float(payload.get("change"))
        if change is None and last_price is not None and prev_close not in (None, 0.0):
            change = last_price - prev_close

        change_pct = _to_float(payload.get("change_pct", payload.get("change_percent")))
        if change_pct is None and change is not None and prev_close not in (None, 0.0):
            change_pct = (change / prev_close) * 100.0

        high = _to_float(payload.get("high", payload.get("day_high")))
        low = _to_float(payload.get("low", payload.get("day_low")))
        opened = _to_float(payload.get("open", payload.get("open_price")))
        volume = _to_float(payload.get("volume"))
        bid = _to_float(payload.get("bid"))
        ask = _to_float(payload.get("ask"))
        market_cap = _to_float(payload.get("market_cap"))
        if market_cap is None:
            market_cap = _to_float(payload.get("market_cap_basic"))

        timestamp = self._normalize_timestamp(payload.get("timestamp", payload.get("_updated")))

        normalized = {
            "symbol": symbol,
            "last_price": last_price,
            "change": change,
            "change_pct": change_pct,
            "volume": volume,
            "bid": bid,
            "ask": ask,
            "high": high,
            "low": low,
            "open": opened,
            "prev_close": prev_close,
            "market_cap": market_cap,
            "timestamp": timestamp,
        }

        if "exchange" in payload:
            normalized["exchange"] = payload.get("exchange")
        if "currency" in payload:
            normalized["currency"] = payload.get("currency")
        if "description" in payload:
            normalized["description"] = payload.get("description")

        return normalized

    def _attach_callback(self, symbol: str, callback: Callable[[str, dict[str, Any]], None]) -> None:
        stream = self._stream
        if stream is None or not self._is_connected():
            return

        on_quote = getattr(stream, "on_quote", None)
        if not callable(on_quote):
            return

        key = (symbol, id(callback))
        if key in self._attached_callbacks:
            return

        try:
            on_quote(symbol, callback)
            self._attached_callbacks.add(key)
        except Exception as exc:
            logger.warning("Failed to attach stream callback for %s: %s", symbol, exc)

    def _attach_all_callbacks(self) -> None:
        for symbol, callbacks in self._callbacks.items():
            for callback in callbacks:
                self._attach_callback(symbol, callback)

    def _subscribe_symbol(
        self,
        symbol: str,
        exchange_candidates: list[str] | None = None,
    ) -> bool:
        stream = self._stream
        if stream is None or not self._is_connected():
            return False

        subscribe = getattr(stream, "subscribe", None)
        if not callable(subscribe):
            logger.warning("TradingView stream subscribe() is unavailable")
            return False

        exchanges = [str(item).strip() for item in (exchange_candidates or []) if str(item).strip()]
        attempts: list[tuple[str, str | None]] = []
        attempts.extend((symbol, exchange) for exchange in exchanges)
        attempts.append((symbol, None))
        # Backward compatible default market path if caller did not pass exchanges.
        if not exchanges:
            attempts.append((symbol, "BIST"))

        for ticker, exchange in attempts:
            try:
                if exchange is None:
                    subscribe(ticker)
                else:
                    subscribe(ticker, exchange)
                return True
            except TypeError:
                if exchange is None:
                    continue
                try:
                    subscribe(ticker)
                    return True
                except Exception:
                    continue
            except Exception:
                continue
        return False

    def connect(self) -> None:
        """Establish WebSocket connection, optionally with TradingView auth."""
        stream = self._ensure_stream()
        if stream is None:
            return
        if self._is_connected():
            return

        try:
            timeout = self._connect_timeout()
            try:
                stream.connect(timeout=timeout)
            except TypeError:
                stream.connect()
        except Exception as exc:
            logger.warning("TradingView stream connection failed: %s", exc)
            return

        self._attach_all_callbacks()

    def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to quote updates for given symbols."""
        if not symbols:
            return

        self.connect()
        stream = self._stream
        if stream is None or not self._is_connected():
            return

        seen: set[str] = set()
        for raw_symbol in symbols:
            symbol = _normalize_symbol(raw_symbol)
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            ok = self._subscribe_symbol(symbol, exchange_candidates=["BIST"])
            if not ok:
                logger.warning("Failed to subscribe stream symbol %s", symbol)
                continue

            for callback in self._callbacks.get(symbol, []):
                self._attach_callback(symbol, callback)

    def subscribe_viop(
        self,
        symbols: list[str],
        exchange_candidates: list[str] | None = None,
    ) -> None:
        """Subscribe to VIOP quote updates for derivative contracts."""
        if not symbols:
            return

        self.connect()
        stream = self._stream
        if stream is None or not self._is_connected():
            return

        seen: set[str] = set()
        candidates = exchange_candidates or ["VIOP", "BISTVIOP", "BIST"]
        for raw_symbol in symbols:
            symbol = _normalize_symbol(raw_symbol)
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            ok = self._subscribe_symbol(symbol, exchange_candidates=candidates)
            if not ok:
                logger.warning("Failed to subscribe VIOP stream symbol %s", symbol)
                continue
            for callback in self._callbacks.get(symbol, []):
                self._attach_callback(symbol, callback)

    def subscribe_chart(self, symbol: str, interval: str = "1m") -> None:
        """Subscribe to OHLCV candle streaming."""
        normalized = _normalize_symbol(symbol)
        if not normalized:
            return

        self.connect()
        stream = self._stream
        if stream is None or not self._is_connected():
            return

        subscribe_chart = getattr(stream, "subscribe_chart", None)
        if not callable(subscribe_chart):
            logger.warning("TradingView stream subscribe_chart() is unavailable")
            return

        try:
            subscribe_chart(normalized, str(interval or "1m").lower())
        except TypeError:
            try:
                subscribe_chart(normalized, str(interval or "1m").lower(), "BIST")
            except Exception as exc:
                logger.warning("Failed to subscribe chart stream %s/%s: %s", normalized, interval, exc)
        except Exception as exc:
            logger.warning("Failed to subscribe chart stream %s/%s: %s", normalized, interval, exc)

    def get_quote(self, symbol: str) -> dict[str, Any]:
        """Get cached quote (<1ms) — replaces _quote_payload() in quotes.py."""
        normalized = _normalize_symbol(symbol)
        if not normalized:
            return {"symbol": normalized, "error": "Invalid symbol", "timestamp": _utc_iso_now()}

        self.connect()
        stream = self._stream
        if stream is None or not self._is_connected():
            return {"symbol": normalized, "error": "Streaming unavailable", "timestamp": _utc_iso_now()}

        get_quote = getattr(stream, "get_quote", None)
        if not callable(get_quote):
            return {
                "symbol": normalized,
                "error": "Stream get_quote is unavailable",
                "timestamp": _utc_iso_now(),
            }

        try:
            raw_quote = get_quote(normalized)
        except Exception as exc:
            logger.warning("Stream quote fetch failed for %s: %s", normalized, exc)
            return {"symbol": normalized, "error": str(exc), "timestamp": _utc_iso_now()}

        if not isinstance(raw_quote, dict) or not raw_quote:
            return {
                "symbol": normalized,
                "error": "No stream quote available",
                "timestamp": _utc_iso_now(),
            }

        return self._normalize_quote(normalized, raw_quote)

    def get_viop_quote(self, symbol: str) -> dict[str, Any]:
        """Get latest quote for a VIOP contract."""
        normalized = _normalize_symbol(symbol)
        if not normalized:
            return {"symbol": normalized, "error": "Invalid symbol", "timestamp": _utc_iso_now()}
        self.subscribe_viop([normalized])
        return self.get_quote(normalized)

    def get_quotes(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """Batch cached quotes — replaces get_quotes() in quotes.py."""
        normalized: list[str] = []
        seen: set[str] = set()
        for raw_symbol in symbols:
            symbol = _normalize_symbol(raw_symbol)
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            normalized.append(symbol)

        if not normalized:
            return {}

        self.subscribe(normalized)
        stream = self._stream
        if stream is None or not self._is_connected():
            return {
                symbol: {"symbol": symbol, "error": "Streaming unavailable", "timestamp": _utc_iso_now()}
                for symbol in normalized
            }

        get_quote = getattr(stream, "get_quote", None)
        get_all_quotes = getattr(stream, "get_all_quotes", None)

        all_quotes: dict[str, Any] = {}
        if callable(get_all_quotes):
            try:
                payload = get_all_quotes()
                if isinstance(payload, dict):
                    all_quotes = payload
            except Exception as exc:
                logger.warning("Stream batch quote fetch failed: %s", exc)

        result: dict[str, dict[str, Any]] = {}
        for symbol in normalized:
            raw_quote = all_quotes.get(symbol)
            if (not isinstance(raw_quote, dict) or not raw_quote) and callable(get_quote):
                try:
                    raw_quote = get_quote(symbol)
                except Exception:
                    raw_quote = None

            if isinstance(raw_quote, dict) and raw_quote:
                result[symbol] = self._normalize_quote(symbol, raw_quote)
            else:
                result[symbol] = {
                    "symbol": symbol,
                    "error": "No stream quote available",
                    "timestamp": _utc_iso_now(),
                }
        return result

    def get_viop_quotes(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """Batch latest quotes for VIOP contracts."""
        self.subscribe_viop(symbols)
        return self.get_quotes(symbols)

    def get_candle(self, symbol: str, interval: str) -> dict[str, Any] | None:
        """Get latest candle for a subscribed chart."""
        normalized = _normalize_symbol(symbol)
        if not normalized:
            return None

        self.connect()
        stream = self._stream
        if stream is None or not self._is_connected():
            return None

        get_candle = getattr(stream, "get_candle", None)
        if not callable(get_candle):
            return None

        try:
            candle = get_candle(normalized, str(interval or "1m").lower())
        except Exception as exc:
            logger.warning("Stream candle fetch failed for %s/%s: %s", normalized, interval, exc)
            return None

        return candle if isinstance(candle, dict) else None

    def on_quote(self, symbol: str, callback: Callable[[str, dict[str, Any]], None]) -> None:
        """Register callback for real-time quote updates."""
        normalized = _normalize_symbol(symbol)
        if not normalized:
            return

        callbacks = self._callbacks.setdefault(normalized, [])
        if callback not in callbacks:
            callbacks.append(callback)

        self.connect()
        self._attach_callback(normalized, callback)

    def disconnect(self) -> None:
        """Clean shutdown."""
        stream = self._stream
        if stream is None:
            return

        disconnect = getattr(stream, "disconnect", None)
        if callable(disconnect):
            try:
                disconnect()
            except Exception as exc:
                logger.warning("TradingView stream disconnect failed: %s", exc)

        self._stream = None
        self._attached_callbacks.clear()

    def __enter__(self) -> "StreamingProvider":
        self.connect()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.disconnect()
