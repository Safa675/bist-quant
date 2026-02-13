"""
Weekly (day-of-week) seasonality signal construction.

Signal logic:
- 1.0 on dates where dayofweek is in target_days
- 0.0 otherwise
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


# Defaults from XU100 seasonality analysis (2013-2025, top 2 weekdays by mean return):
# Tuesday (1), Monday (0)
DEFAULT_TARGET_DAYS: tuple[int, ...] = (1, 0)


def build_weekly_signals(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
    target_days: Sequence[int] | None = None,
) -> pd.DataFrame:
    """
    Build day-of-week seasonality long-only signals.

    Args:
        close_df: Close prices DataFrame (dates x tickers)
        dates: Target dates for signals
        data_loader: Unused, kept for interface consistency
        target_days: Weekday integers (0=Mon .. 6=Sun) to hold long

    Returns:
        DataFrame (dates x tickers) with 1.0 on target weekdays, 0.0 otherwise.
    """
    del data_loader

    selected_days = DEFAULT_TARGET_DAYS if target_days is None else tuple(target_days)
    if len(selected_days) == 0:
        selected_days = DEFAULT_TARGET_DAYS

    date_index = pd.DatetimeIndex(dates)
    day_mask = pd.Index(date_index.dayofweek).isin(selected_days)

    base_signal = day_mask.astype(float)[:, None]
    signal_matrix = np.repeat(base_signal, close_df.shape[1], axis=1)
    result = pd.DataFrame(signal_matrix, index=date_index, columns=close_df.columns, dtype=float)

    active_days = int(day_mask.sum())
    print(
        f"\nBuilding weekly seasonality signals..."
        f" active weekdays={list(selected_days)}, active days={active_days}/{len(date_index)}"
    )
    print(f"  Signal shape: {result.shape[0]} days x {result.shape[1]} tickers")

    return result
