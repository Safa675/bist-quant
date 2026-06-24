"""
Momentum Signal Construction

Calculates risk-adjusted momentum scores based on:
- 12-1 Momentum: Past 12 months return excluding the last month
- Downside Volatility: Rolling standard deviation of negative returns
- Risk-Adjusted Score: (12-1 Return) / Downside Volatility
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from bist_quant.common.utils import validate_signal_panel_schema
from bist_quant.signals.core.momentum import (
    DOWNSIDE_VOL_LOOKBACK,
    MOMENTUM_LOOKBACK,
    MOMENTUM_SKIP,
    compute_price_momentum,
    compute_prod_downside_volatility,
    compute_risk_adjusted_momentum,
)

logger = logging.getLogger(__name__)


def calculate_12_minus_1_momentum(
    close_df: pd.DataFrame,
    lookback: int = MOMENTUM_LOOKBACK,
    skip: int = MOMENTUM_SKIP,
) -> pd.DataFrame:
    return compute_price_momentum(close_df, lookback=lookback, skip=skip, mode="prod")


def calculate_downside_volatility(
    close_df: pd.DataFrame,
    lookback: int = DOWNSIDE_VOL_LOOKBACK,
    skip: int = MOMENTUM_SKIP,
) -> pd.DataFrame:
    return compute_prod_downside_volatility(close_df, lookback=lookback, skip=skip)


def calculate_momentum_scores(
    close_df: pd.DataFrame,
    lookback: int = MOMENTUM_LOOKBACK,
    skip: int = MOMENTUM_SKIP,
    vol_lookback: int = DOWNSIDE_VOL_LOOKBACK,
) -> pd.DataFrame:
    return compute_risk_adjusted_momentum(
        close_df,
        lookback=lookback,
        skip=skip,
        vol_lookback=vol_lookback,
        mode="prod",
    )


def build_momentum_signals(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
    lookback: int = MOMENTUM_LOOKBACK,
    skip: int = MOMENTUM_SKIP,
    vol_lookback: int = DOWNSIDE_VOL_LOOKBACK,
) -> pd.DataFrame:
    logger.info("\n🔧 Building momentum signals...")
    logger.info(f"  Momentum: {lookback} days lookback, {skip} days skip")
    logger.info(f"  Downside Vol: {vol_lookback} days lookback")

    momentum_scores = calculate_momentum_scores(
        close_df,
        lookback=lookback,
        skip=skip,
        vol_lookback=vol_lookback,
    )

    result = momentum_scores.reindex(dates)
    result = validate_signal_panel_schema(
        result,
        dates=dates,
        tickers=close_df.columns,
        signal_name="momentum",
        context="final score panel",
        dtype=np.float32,
    )

    valid_scores = result.dropna(how="all")
    if not valid_scores.empty:
        latest = valid_scores.iloc[-1].dropna()
        if len(latest) > 0:
            logger.info(f"  Latest scores - Mean: {latest.mean():.2f}, Std: {latest.std():.2f}")
            logger.info(f"  Latest scores - Min: {latest.min():.2f}, Max: {latest.max():.2f}")

    logger.info(f"  ✅ Momentum signals: {result.shape[0]} days × {result.shape[1]} tickers")
    return result
