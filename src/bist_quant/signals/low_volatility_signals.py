"""Low Volatility Factor Signal Construction."""

from __future__ import annotations

import logging

import pandas as pd

from bist_quant.signals.core.low_volatility import (
    VOLATILITY_LOOKBACK,
    WEEKLY_VOLATILITY,
    calculate_low_volatility_scores,
)

logger = logging.getLogger(__name__)


def calculate_daily_volatility(close_df, lookback: int = VOLATILITY_LOOKBACK):
    from bist_quant.signals.core.low_volatility import calculate_daily_volatility as _fn
    return _fn(close_df, lookback)


def calculate_weekly_volatility(close_df, lookback: int = VOLATILITY_LOOKBACK):
    from bist_quant.signals.core.low_volatility import calculate_weekly_volatility as _fn
    return _fn(close_df, lookback)


def calculate_downside_volatility(close_df, lookback: int = VOLATILITY_LOOKBACK):
    from bist_quant.signals.core.low_volatility import calculate_downside_volatility as _fn
    return _fn(close_df, lookback)


def build_low_volatility_signals(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
) -> pd.DataFrame:
    logger.info("\n🔧 Building low volatility signals...")
    logger.info(f"  Volatility Lookback: {VOLATILITY_LOOKBACK} days")
    logger.info(f"  Using Weekly Returns: {WEEKLY_VOLATILITY}")

    low_vol_scores = calculate_low_volatility_scores(close_df)
    result = low_vol_scores.reindex(dates)

    valid_scores = result.dropna(how="all")
    if not valid_scores.empty:
        latest = valid_scores.iloc[-1].dropna()
        if len(latest) > 0:
            latest_vol = -latest
            logger.info(
                f"  Latest volatility range: {latest_vol.min()*100:.2f}% to {latest_vol.max()*100:.2f}%"
            )
            logger.info(f"  Median volatility: {latest_vol.median()*100:.2f}%")

    logger.info(f"  ✅ Low volatility signals: {result.shape[0]} days × {result.shape[1]} tickers")
    return result
