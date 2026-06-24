"""Low volatility primitives (BUILDERS low_volatility path)."""

from __future__ import annotations

import numpy as np
import pandas as pd

VOLATILITY_LOOKBACK = 252
WEEKLY_VOLATILITY = True
MIN_DOWNSIDE_NEGATIVE_OBS = 2


def calculate_daily_volatility(
    close_df: pd.DataFrame,
    lookback: int = VOLATILITY_LOOKBACK,
) -> pd.DataFrame:
    daily_returns = close_df.pct_change()
    return daily_returns.rolling(lookback, min_periods=lookback // 2).std()


def calculate_weekly_volatility(
    close_df: pd.DataFrame,
    lookback: int = VOLATILITY_LOOKBACK,
) -> pd.DataFrame:
    weekly_returns = close_df / close_df.shift(5) - 1.0
    weeks_lookback = lookback // 5
    return weekly_returns.rolling(weeks_lookback, min_periods=weeks_lookback // 2).std()


def calculate_downside_volatility(
    close_df: pd.DataFrame,
    lookback: int = VOLATILITY_LOOKBACK,
) -> pd.DataFrame:
    daily_returns = close_df.pct_change()
    min_periods = lookback // 2
    negative_only = daily_returns.where(daily_returns < 0.0)
    total_counts = daily_returns.rolling(lookback, min_periods=min_periods).count()
    negative_counts = negative_only.rolling(lookback, min_periods=1).count()
    downside_vol = negative_only.rolling(lookback, min_periods=1).std()
    return downside_vol.where(
        (total_counts >= min_periods) & (negative_counts > MIN_DOWNSIDE_NEGATIVE_OBS)
    )


def calculate_low_volatility_scores(
    close_df: pd.DataFrame,
    *,
    use_weekly: bool = WEEKLY_VOLATILITY,
    use_downside: bool = False,
    lookback: int = VOLATILITY_LOOKBACK,
) -> pd.DataFrame:
    if use_downside:
        volatility = calculate_downside_volatility(close_df, lookback)
    elif use_weekly:
        volatility = calculate_weekly_volatility(close_df, lookback)
    else:
        volatility = calculate_daily_volatility(close_df, lookback)
    return -volatility.replace([np.inf, -np.inf], np.nan)


def calculate_inverted_annualized_volatility(
    close_df: pd.DataFrame,
    *,
    lookback: int = 63,
    min_obs: int | None = None,
) -> pd.DataFrame:
    """Standalone LowVolatilitySignal path: inverted annualized daily vol."""
    if min_obs is None:
        min_obs = max(int(lookback * 0.5), 21)
    daily_returns = close_df.pct_change()
    volatility = daily_returns.rolling(lookback, min_periods=min_obs).std() * np.sqrt(252)
    return -volatility


__all__ = [
    "VOLATILITY_LOOKBACK",
    "WEEKLY_VOLATILITY",
    "calculate_daily_volatility",
    "calculate_weekly_volatility",
    "calculate_downside_volatility",
    "calculate_low_volatility_scores",
    "calculate_inverted_annualized_volatility",
]
