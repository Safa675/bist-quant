"""Shared helpers for the realtime sub-package.

``streaming.py`` and ``quotes.py`` previously each defined their own copies
of symbol normalisation, safe float parsing and UTC-timestamp formatting.
This module centralises those primitives so both modules stay in sync.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any


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


def utc_iso_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


__all__ = ["normalize_symbol", "to_float", "utc_iso_now"]
