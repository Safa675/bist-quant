"""
Monthly seasonality signal construction.

Signal logic:
- 1.0 on dates where month is in target_months
- 0.0 (or configured inactive_value) otherwise
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


# Defaults from XU100 seasonality analysis (2013-2025, top 3 months by mean daily return):
# November, July, January
DEFAULT_TARGET_MONTHS: tuple[int, ...] = (11, 7, 1)


def build_monthly_signals(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
    target_months: Sequence[int] | None = None,
    inactive_value: float = 0.0,
) -> pd.DataFrame:
    """
    Build monthly seasonality long-only signals.

    Args:
        close_df: Close prices DataFrame (dates x tickers)
        dates: Target dates for signals
        data_loader: Unused, kept for interface consistency
        target_months: Months to hold long (1-12)
        inactive_value: Value for non-target months (default 0.0)

    Returns:
        DataFrame (dates x tickers) with 1.0 on target months, inactive_value otherwise.
    """
    del data_loader

    selected_months = DEFAULT_TARGET_MONTHS if target_months is None else tuple(target_months)
    if len(selected_months) == 0:
        selected_months = DEFAULT_TARGET_MONTHS

    date_index = pd.DatetimeIndex(dates)
    month_mask = pd.Index(date_index.month).isin(selected_months)

    base_signal = np.where(month_mask[:, None], 1.0, inactive_value)
    signal_matrix = np.repeat(base_signal, close_df.shape[1], axis=1)
    result = pd.DataFrame(signal_matrix, index=date_index, columns=close_df.columns, dtype=float)

    active_days = int(month_mask.sum())
    print(
        f"\nBuilding monthly seasonality signals..."
        f" active months={list(selected_months)}, active days={active_days}/{len(date_index)}"
    )
    print(f"  Signal shape: {result.shape[0]} days x {result.shape[1]} tickers")

    return result
