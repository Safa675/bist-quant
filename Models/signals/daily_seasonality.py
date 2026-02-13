"""
Daily (calendar-day) seasonality signal construction.

Signal logic:
- 1.0 on dates matching target (month, day) pairs
- 0.0 otherwise
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


# Defaults from XU100 seasonality analysis (top 10 calendar days by mean daily return)
DEFAULT_TARGET_DAYS: tuple[tuple[int, int], ...] = (
    (2, 16),
    (11, 2),
    (4, 5),
    (2, 15),
    (3, 2),
    (10, 18),
    (11, 15),
    (7, 20),
    (1, 17),
    (7, 15),
)


def build_daily_seasonality_signals(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
    target_days: Sequence[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """
    Build daily seasonality long-only signals.

    Args:
        close_df: Close prices DataFrame (dates x tickers)
        dates: Target dates for signals
        data_loader: Unused, kept for interface consistency
        target_days: Month-day tuples to hold long, e.g. [(1, 2), (5, 15)]

    Returns:
        DataFrame (dates x tickers) with 1.0 on target days, 0.0 otherwise.
    """
    del data_loader

    selected_days = DEFAULT_TARGET_DAYS if target_days is None else tuple(target_days)
    if len(selected_days) == 0:
        selected_days = DEFAULT_TARGET_DAYS

    target_array = np.asarray(selected_days, dtype=int)
    if target_array.ndim != 2 or target_array.shape[1] != 2:
        raise ValueError("target_days must be a sequence of (month, day) tuples.")

    date_index = pd.DatetimeIndex(dates)
    date_pairs = pd.MultiIndex.from_arrays([date_index.month, date_index.day])
    target_pairs = pd.MultiIndex.from_arrays([target_array[:, 0], target_array[:, 1]])
    day_mask = date_pairs.isin(target_pairs)

    base_signal = day_mask.astype(float)[:, None]
    signal_matrix = np.repeat(base_signal, close_df.shape[1], axis=1)
    result = pd.DataFrame(signal_matrix, index=date_index, columns=close_df.columns, dtype=float)

    active_days = int(day_mask.sum())
    print(
        f"\nBuilding daily seasonality signals..."
        f" active day-patterns={len(target_pairs)}, active days={active_days}/{len(date_index)}"
    )
    print(f"  Signal shape: {result.shape[0]} days x {result.shape[1]} tickers")

    return result
