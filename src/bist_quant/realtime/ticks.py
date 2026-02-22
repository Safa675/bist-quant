"""
Tick extraction and realtime data fetching.

Migrated from bist_quant.ai/api/index.py (lines 922-1027):
  - _normalize_realtime_symbols()
  - _extract_tick_payload()
  - _fetch_realtime_ticks()
  - _fallback_realtime_ticks()
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timezone
from typing import Any

from .quotes import get_quotes_streaming, to_float

logger = logging.getLogger(__name__)


def normalize_realtime_symbols(raw: Any) -> list[str]:
    """Parse and deduplicate symbol input, capped at 20 symbols."""
    if isinstance(raw, str):
        items = raw.split(",")
    elif isinstance(raw, list):
        items = raw
    else:
        items = []
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        symbol = str(item).strip().upper().split(".")[0]
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
        if len(out) >= 20:
            break
    return out or ["XU100"]


def extract_tick_payload(raw_prices: Any, symbols: list[str]) -> list[dict[str, Any]]:
    """Extract tick data from a pandas DataFrame (borsapy download result).

    Handles both single-ticker and multi-ticker MultiIndex DataFrames.
    """
    try:
        import pandas as pd
    except ImportError:
        return []

    if not isinstance(raw_prices, pd.DataFrame) or raw_prices.empty:
        return []

    frame = raw_prices
    if isinstance(frame.columns, pd.MultiIndex):
        lvl0 = frame.columns.get_level_values(0)
        if {"Open", "High", "Low", "Close"}.issubset(set(lvl0)):
            frame = frame.swaplevel(0, 1, axis=1).sort_index(axis=1)

    ticks: list[dict[str, Any]] = []
    now_iso = pd.Timestamp.utcnow().isoformat()
    for symbol in symbols:
        try:
            if isinstance(frame.columns, pd.MultiIndex):
                if symbol not in set(frame.columns.get_level_values(0)):
                    continue
                sub = frame[symbol]
            else:
                sub = frame
            if not isinstance(sub, pd.DataFrame) or "Close" not in sub.columns:
                continue

            close = pd.to_numeric(sub["Close"], errors="coerce").dropna()
            if close.empty:
                continue
            latest = float(close.iloc[-1])
            prev = float(close.iloc[-2]) if len(close) > 1 else latest
            volume_series = pd.to_numeric(sub.get("Volume"), errors="coerce").dropna()
            volume = float(volume_series.iloc[-1]) if not volume_series.empty else 0.0
            change_pct = ((latest / prev - 1.0) * 100.0) if prev else 0.0
            ticks.append(
                {
                    "symbol": symbol,
                    "price": round(latest, 6),
                    "change_pct": round(change_pct, 6),
                    "volume": max(0.0, round(volume, 3)),
                    "timestamp": now_iso,
                }
            )
        except Exception:
            continue
    return ticks


def _ticks_from_stream_quotes(
    symbols: list[str],
    quote_map: dict[str, dict[str, Any]],
    default_timestamp: str,
) -> list[dict[str, Any]]:
    ticks: list[dict[str, Any]] = []
    for symbol in symbols:
        payload = quote_map.get(symbol)
        if not isinstance(payload, dict) or payload.get("error"):
            continue

        price = to_float(payload.get("last_price"))
        if price is None:
            continue

        change_pct = to_float(payload.get("change_pct"))
        volume = to_float(payload.get("volume"))
        timestamp = payload.get("timestamp")
        if not isinstance(timestamp, str) or not timestamp.strip():
            timestamp = default_timestamp

        ticks.append(
            {
                "symbol": symbol,
                "price": round(price, 6),
                "change_pct": round(change_pct or 0.0, 6),
                "volume": max(0.0, round((volume or 0.0), 3)),
                "timestamp": timestamp,
            }
        )
    return ticks


def fetch_realtime_ticks(
    symbols: list[str],
    streaming: bool = True,
    auth_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Fetch realtime ticks via stream first, then fall back to borsapy download."""
    normalized = normalize_realtime_symbols(symbols)
    if streaming:
        try:
            streamed = get_quotes_streaming(normalized, auth_config=auth_config)
            quote_map = streamed.get("quotes", {}) if isinstance(streamed, dict) else {}
            now_iso = datetime.now(timezone.utc).isoformat()
            ticks = _ticks_from_stream_quotes(
                normalized,
                quote_map if isinstance(quote_map, dict) else {},
                default_timestamp=now_iso,
            )
            if ticks:
                return ticks
        except Exception as exc:
            logger.warning("Streaming tick fetch failed: %s", exc)

    import borsapy as bp

    raw = bp.download(
        normalized,
        period="5d",
        interval="1d",
        group_by="ticker",
        progress=False,
    )
    ticks = extract_tick_payload(raw, normalized)
    if not ticks:
        raise RuntimeError("upstream quote source returned no ticks")
    return ticks


def fallback_realtime_ticks(
    symbols: list[str],
    previous: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate synthetic fallback ticks when upstream is unavailable."""
    try:
        import pandas as pd

        now_iso = pd.Timestamp.utcnow().isoformat()
    except ImportError:
        from datetime import datetime, timezone

        now_iso = datetime.now(timezone.utc).isoformat()

    ticks: list[dict[str, Any]] = []
    for idx, symbol in enumerate(symbols):
        prev = previous.get(symbol)
        prev_price = (
            float(prev.get("price", 100.0 + (idx * 7.0)))
            if isinstance(prev, dict)
            else 100.0 + (idx * 7.0)
        )
        prev_volume = (
            float(prev.get("volume", 400000.0 + idx * 15000.0))
            if isinstance(prev, dict)
            else 400000.0 + idx * 15000.0
        )
        drift = (random.random() - 0.5) * 0.012
        price = max(0.1, prev_price * (1.0 + drift))
        change_pct = ((price / prev_price - 1.0) * 100.0) if prev_price else 0.0
        volume = max(0.0, prev_volume + random.uniform(-9000.0, 16000.0))
        ticks.append(
            {
                "symbol": symbol,
                "price": round(price, 6),
                "change_pct": round(change_pct, 6),
                "volume": round(volume, 3),
                "timestamp": now_iso,
            }
        )
    return ticks
