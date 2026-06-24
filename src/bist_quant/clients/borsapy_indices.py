"""Helper module for BIST indices components and indicator calculations via borsapy."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import borsapy as bp

logger = logging.getLogger(__name__)


def get_index(client: Any, index: str = "XU100") -> bp.Index:
    """Get index object."""
    index = index.upper()
    if index not in client._index_cache:
        client._index_cache[index] = bp.Index(index)
    return client._index_cache[index]


def get_index_components(client: Any, index: str = "XU100") -> list[str]:
    """Get index constituent symbols."""
    index = index.upper()
    if index not in client._index_components_cache:
        # Try disk cache
        if client._disk_cache is not None:
            cached = client._disk_cache.get_json("index_components", index)
            if isinstance(cached, list) and cached:
                client._index_components_cache[index] = cached
                return cached

        try:
            idx = get_index(client, index)
            components = list(idx.component_symbols or [])
            client._index_components_cache[index] = components
            # Persist to disk cache
            if client._disk_cache is not None and components:
                client._disk_cache.set_json("index_components", index, components)
        except Exception as e:
            logger.info(f"  Warning: Failed to get components for {index}: {e}")
            client._index_components_cache[index] = []

    return client._index_components_cache[index]


def get_all_indices(client: Any) -> list[dict[str, Any]]:
    """Get all available BIST indices."""
    try:
        return bp.all_indices()
    except Exception:
        return []


def get_history_with_indicators(
    client: Any,
    symbol: str,
    indicators: list[str] = None,
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Get OHLCV history with built-in technical indicators."""
    ticker = client.get_ticker(symbol)
    try:
        if indicators:
            return ticker.history_with_indicators(
                period=period, interval=interval, indicators=indicators
            )
        else:
            return ticker.history_with_indicators(period=period, interval=interval)
    except Exception as e:
        logger.info(f"  Warning: Failed to get indicators for {symbol}: {e}")
        return client.get_history(symbol, period=period, interval=interval)


def calculate_rsi(
    prices: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Calculate RSI indicator."""
    try:
        return bp.calculate_rsi(prices, period=period)
    except Exception:
        return pd.Series(dtype=float)


def calculate_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Calculate MACD indicator."""
    try:
        return bp.calculate_macd(prices, fast=fast, slow=slow, signal=signal)
    except Exception:
        return pd.DataFrame()


def calculate_supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """Calculate Supertrend indicator."""
    try:
        return bp.calculate_supertrend(high, low, close, period, multiplier)
    except Exception:
        return pd.DataFrame()
