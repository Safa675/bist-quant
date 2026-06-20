"""Shared low-level panel-validation helpers.

The five public validation/alignment functions live in ``common/utils.py``
(because they are deeply intertwined with ``debug_log`` /
``raise_signal_data_error`` / ``SignalDataError`` defined there). This module
holds only the three recurring atomic checks that were previously
copy-pasted 3–4× across those functions:

- ``_check_datetime_index``    — NaT / duplicate / monotonic-increasing
- ``_check_no_duplicate_columns`` — duplicate ticker columns
- ``_check_no_object_cols``    — object-dtype rejection

Each helper accepts an ``error_fn(msg)`` callable so callers can raise either
``ValueError`` / ``TypeError`` (the ``validate_numeric_panel`` family) or
``SignalDataError`` (via ``raise_signal_data_error`` for
``validate_signal_panel_schema``) without duplicating the check logic.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

# A function that raises an exception with a formatted message.
_ErrorFn = Callable[[str], None]


def _check_datetime_index(idx: pd.DatetimeIndex, label: str, error_fn: _ErrorFn) -> None:
    """Validate a DatetimeIndex has no NaT, no duplicates, and is monotonic."""
    if idx.hasnans:
        error_fn(f"{label}: index contains NaT values")
    if idx.has_duplicates:
        error_fn(f"{label}: index contains duplicate dates")
    if not idx.is_monotonic_increasing:
        error_fn(f"{label}: index must be monotonic increasing")


def _check_no_duplicate_columns(df: pd.DataFrame, label: str, error_fn: _ErrorFn) -> None:
    """Validate a DataFrame has no duplicate ticker columns."""
    if df.columns.has_duplicates:
        dup_cols = df.columns[df.columns.duplicated()].unique().tolist()[:5]
        error_fn(f"{label}: duplicate ticker columns found (sample={dup_cols})")


def _check_no_object_cols(df: pd.DataFrame, label: str, error_fn: _ErrorFn) -> None:
    """Validate a DataFrame has no object-dtype columns."""
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if object_cols:
        error_fn(f"{label}: object dtype columns are not allowed (sample={object_cols[:5]})")


__all__ = [
    "_check_datetime_index",
    "_check_no_duplicate_columns",
    "_check_no_object_cols",
]
