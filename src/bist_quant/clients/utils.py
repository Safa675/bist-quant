"""Utility functions for clients and providers in the bist-quant package.

Consolidates parsing, dynamic loading, and helper routines shared across providers.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any
import pandas as pd

logger = logging.getLogger(__name__)


def to_float(value: Any) -> float | None:
    """Parse a value to float, handling Turkish formatting and non-numeric characters."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None

    text = str(value).strip()
    if not text:
        return None
    text = text.replace("%", "")

    if "," in text:
        if "." in text:
            if text.find(".") < text.find(","):
                text = text.replace(".", "").replace(",", ".")
            else:
                text = text.replace(",", "")
        else:
            text = text.replace(",", ".")

    text = re.sub(r"[^0-9eE.\-+]", "", text)
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def as_frame(payload: Any) -> pd.DataFrame:
    """Normalize arbitrary payloads (DataFrames, dicts, lists, tuples) into a DataFrame."""
    if isinstance(payload, pd.DataFrame):
        return payload.copy()
    if isinstance(payload, (list, tuple)):
        if not payload:
            return pd.DataFrame()
        try:
            return pd.DataFrame(payload)
        except Exception:
            return pd.DataFrame()
    if isinstance(payload, dict):
        if not payload:
            return pd.DataFrame()
        # If all values are scalar/non-containers, make single-row frame.
        if all(not isinstance(v, (list, tuple, dict, pd.Series, pd.DataFrame)) for v in payload.values()):
            return pd.DataFrame([payload])
        try:
            return pd.DataFrame(payload)
        except Exception:
            return pd.DataFrame([payload])
    return pd.DataFrame()


def pick_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    """Case-insensitive matching of column names against list of candidate names."""
    if frame is None or frame.empty:
        return None
    lookup = {str(col).strip().lower(): col for col in frame.columns}
    for candidate in candidates:
        hit = lookup.get(candidate.lower())
        if hit is not None:
            return str(hit)
    return None


def call_if_callable(obj: Any, *args: Any, **kwargs: Any) -> Any:
    """Safely execute callable, handling TypeError for unexpected arguments."""
    if not callable(obj):
        return None
    try:
        return obj(*args, **kwargs)
    except TypeError:
        try:
            return obj(*args)
        except Exception:
            return None
    except Exception:
        return None


def get_borsapy_module(provider_name: str, import_attempted: bool) -> tuple[Any | None, bool]:
    """Dynamically attempt to import the borsapy module."""
    if import_attempted:
        return None, True
    try:
        import borsapy as bp  # type: ignore[import-not-found]
        return bp, True
    except Exception as exc:
        logger.info(f"  {provider_name}: borsapy unavailable: {exc}")
        return None, True
