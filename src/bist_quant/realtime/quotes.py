"""
Realtime quote functions for BIST market data.

Migrated from bist_quant.ai/api/realtime_api.py.
Uses borsapy for live data from Borsa Istanbul.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

from .streaming import StreamingProvider

logger = logging.getLogger(__name__)

_STREAMING_PROVIDER: StreamingProvider | None = None
_STREAMING_PROVIDER_FINGERPRINT: tuple[tuple[str, str], ...] | None = None


def normalize_symbol(symbol: str) -> str:
    """Normalize a ticker symbol (strip, uppercase, remove exchange suffix)."""
    return str(symbol or "").strip().upper().split(".")[0]


def to_float(value: Any) -> float | None:
    """Safely parse a value to float, returning None on failure or non-finite."""
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


def _auth_config_fingerprint(auth_config: dict[str, Any] | None) -> tuple[tuple[str, str], ...]:
    if not auth_config:
        return ()
    return tuple(sorted((str(k), repr(v)) for k, v in auth_config.items()))


def _load_default_streaming_auth_config() -> dict[str, Any] | None:
    try:
        from bist_quant.settings.settings import load_production_settings

        settings = load_production_settings()
        config: dict[str, Any] = dict(settings.tradingview_auth_config)
        config["connect_timeout"] = float(settings.tradingview_connect_timeout_seconds)
        return config
    except Exception:
        return None


def _resolve_streaming_auth_config(auth_config: dict[str, Any] | None) -> dict[str, Any] | None:
    if auth_config is not None:
        return auth_config
    return _load_default_streaming_auth_config()


def _get_streaming_provider(auth_config: dict[str, Any] | None = None) -> StreamingProvider | None:
    global _STREAMING_PROVIDER
    global _STREAMING_PROVIDER_FINGERPRINT

    resolved_auth = _resolve_streaming_auth_config(auth_config)
    fingerprint = _auth_config_fingerprint(resolved_auth)

    if _STREAMING_PROVIDER is not None and _STREAMING_PROVIDER_FINGERPRINT == fingerprint:
        return _STREAMING_PROVIDER

    if _STREAMING_PROVIDER is not None:
        try:
            _STREAMING_PROVIDER.disconnect()
        except Exception:
            pass

    try:
        _STREAMING_PROVIDER = StreamingProvider(auth_config=resolved_auth)
        _STREAMING_PROVIDER_FINGERPRINT = fingerprint
    except Exception as exc:
        logger.warning("Failed to initialize streaming provider: %s", exc)
        _STREAMING_PROVIDER = None
        _STREAMING_PROVIDER_FINGERPRINT = None
    return _STREAMING_PROVIDER


def normalize_change_pct(
    raw_change_pct: Any,
    value: float | None,
    prev_close: float | None,
) -> float | None:
    """Normalize percentage change, reconciling provided vs computed values."""
    computed = None
    if value is not None and prev_close not in (None, 0.0):
        computed = ((value - prev_close) / prev_close) * 100.0

    provided = to_float(raw_change_pct)
    if provided is None:
        return computed
    if computed is None:
        return provided

    if abs(provided - computed) <= 0.15:
        return provided
    if abs((provided * 100.0) - computed) <= 0.15:
        return provided * 100.0
    return computed


def _quote_payload(symbol: str) -> dict[str, Any]:
    """Fetch a single quote payload using borsapy."""
    import borsapy as bp

    symbol = normalize_symbol(symbol)
    if not symbol:
        return {"symbol": symbol, "error": "Invalid symbol"}

    try:
        ticker = bp.Ticker(symbol)
        fast_info = dict(ticker.fast_info)

        last_price = to_float(fast_info.get("last_price"))
        prev_close = to_float(fast_info.get("previous_close"))
        change = None
        change_pct = None
        if last_price is not None and prev_close not in (None, 0.0):
            change = last_price - prev_close
            change_pct = (change / prev_close) * 100.0

        return {
            "symbol": symbol,
            "last_price": last_price,
            "change": change,
            "change_pct": change_pct,
            "volume": to_float(fast_info.get("volume")),
            "bid": None,
            "ask": None,
            "high": to_float(fast_info.get("day_high")),
            "low": to_float(fast_info.get("day_low")),
            "open": to_float(fast_info.get("open")),
            "prev_close": prev_close,
            "market_cap": to_float(fast_info.get("market_cap")),
            "timestamp": _utc_iso_now(),
        }
    except Exception as exc:
        return {"symbol": symbol, "error": str(exc), "timestamp": _utc_iso_now()}


def get_quote(symbol: str) -> dict[str, Any]:
    """Get a single realtime quote."""
    return _quote_payload(symbol)


def get_quote_streaming(
    symbol: str,
    auth_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Get a single realtime quote via persistent stream with polling fallback."""
    normalized = normalize_symbol(symbol)
    if not normalized:
        return {"symbol": normalized, "error": "Invalid symbol", "timestamp": _utc_iso_now()}

    provider = _get_streaming_provider(auth_config=auth_config)
    if provider is None:
        return _quote_payload(normalized)

    try:
        provider.subscribe([normalized])
        streamed = provider.get_quote(normalized)
    except Exception as exc:
        logger.warning("Streaming quote failed for %s: %s", normalized, exc)
        streamed = {}

    if isinstance(streamed, dict) and streamed.get("last_price") is not None and "error" not in streamed:
        return streamed
    return _quote_payload(normalized)


def get_quotes(symbols: list[str]) -> dict[str, Any]:
    """Get multiple realtime quotes."""
    normalized = [normalize_symbol(s) for s in symbols]
    normalized = [s for s in normalized if s]
    quotes = {symbol: _quote_payload(symbol) for symbol in normalized}
    return {
        "quotes": quotes,
        "count": len(quotes),
        "timestamp": _utc_iso_now(),
    }


def get_quotes_streaming(
    symbols: list[str],
    auth_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Get multiple realtime quotes via stream with per-symbol polling fallback."""
    normalized = [normalize_symbol(s) for s in symbols]
    normalized = [s for s in normalized if s]
    if not normalized:
        return {"quotes": {}, "count": 0, "timestamp": _utc_iso_now()}

    provider = _get_streaming_provider(auth_config=auth_config)
    if provider is None:
        return get_quotes(normalized)

    streamed_quotes: dict[str, dict[str, Any]] = {}
    try:
        provider.subscribe(normalized)
        streamed_quotes = provider.get_quotes(normalized)
    except Exception as exc:
        logger.warning("Streaming batch quote fetch failed: %s", exc)

    merged: dict[str, dict[str, Any]] = {}
    for symbol in normalized:
        payload = streamed_quotes.get(symbol)
        if isinstance(payload, dict) and payload.get("last_price") is not None and "error" not in payload:
            merged[symbol] = payload
        else:
            merged[symbol] = _quote_payload(symbol)

    return {
        "quotes": merged,
        "count": len(merged),
        "timestamp": _utc_iso_now(),
    }


def get_index(index_name: str) -> dict[str, Any]:
    """Get index data (e.g. XU100, XU030)."""
    import borsapy as bp

    index_name = normalize_symbol(index_name) or "XU100"
    try:
        idx = bp.index(index_name)
        info = idx.info if isinstance(idx.info, dict) else {}
        value = to_float(info.get("last"))
        prev_close = to_float(info.get("close"))
        change_pct = normalize_change_pct(info.get("change_percent"), value, prev_close)
        return {
            "index": index_name,
            "value": value,
            "change_pct": change_pct,
            "prev_close": prev_close,
            "timestamp": _utc_iso_now(),
        }
    except Exception as exc:
        return {"index": index_name, "error": str(exc), "timestamp": _utc_iso_now()}


def _get_usdtry_fallback() -> dict[str, Any]:
    import borsapy as bp

    try:
        fx = bp.FX("USD")
        info = fx.info if isinstance(fx.info, dict) else {}
        rate = to_float(info.get("last"))
        open_rate = to_float(info.get("open"))
        change_pct = None
        if rate is not None and open_rate not in (None, 0.0):
            change_pct = ((rate - open_rate) / open_rate) * 100.0
        return {"rate": rate, "change_pct": change_pct, "timestamp": _utc_iso_now()}
    except Exception as exc:
        return {"error": str(exc), "timestamp": _utc_iso_now()}


def _index_from_stream_payload(index_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    value = to_float(payload.get("last_price"))
    prev_close = to_float(payload.get("prev_close"))
    change_pct = normalize_change_pct(payload.get("change_pct"), value, prev_close)
    if value is None:
        return get_index(index_name)
    return {
        "index": index_name,
        "value": value,
        "change_pct": change_pct,
        "prev_close": prev_close,
        "timestamp": payload.get("timestamp", _utc_iso_now()),
    }


def get_market(
    streaming: bool = True,
    auth_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Get market summary: XU100, XU030, and USDTRY."""
    summary: dict[str, Any] = {"timestamp": _utc_iso_now()}

    streamed_quotes: dict[str, dict[str, Any]] = {}
    if streaming:
        try:
            streamed = get_quotes_streaming(["XU100", "XU030", "USDTRY"], auth_config=auth_config)
            quotes = streamed.get("quotes")
            if isinstance(quotes, dict):
                streamed_quotes = quotes
        except Exception as exc:
            logger.warning("Streaming market summary failed: %s", exc)

    xu100_payload = streamed_quotes.get("XU100", {})
    if isinstance(xu100_payload, dict) and xu100_payload.get("last_price") is not None:
        summary["xu100"] = _index_from_stream_payload("XU100", xu100_payload)
    else:
        summary["xu100"] = get_index("XU100")

    xu030_payload = streamed_quotes.get("XU030", {})
    if isinstance(xu030_payload, dict) and xu030_payload.get("last_price") is not None:
        summary["xu030"] = _index_from_stream_payload("XU030", xu030_payload)
    else:
        summary["xu030"] = get_index("XU030")

    usdtry_payload = streamed_quotes.get("USDTRY", {})
    streamed_rate = to_float(usdtry_payload.get("last_price")) if isinstance(usdtry_payload, dict) else None
    if streamed_rate is not None:
        summary["usdtry"] = {
            "rate": streamed_rate,
            "change_pct": to_float(usdtry_payload.get("change_pct")),
            "timestamp": usdtry_payload.get("timestamp", _utc_iso_now()),
        }
    else:
        summary["usdtry"] = _get_usdtry_fallback()
    return summary
