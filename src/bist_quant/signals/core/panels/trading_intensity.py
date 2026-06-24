"""Canonical panel builder (migrated from factor_builders)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from bist_quant.common.utils import (
    align_numeric_panel,
    apply_lag,
    coerce_quarter_cols,
    get_consolidated_sheet,
    pick_row,
    pick_row_from_sheet,
    sum_ttm,
    validate_reference_axes,
)
from bist_quant.signals.core.panels._helpers import (
    finalize_builder_outputs,
    load_shares_panel,
    build_metric_panel,
    panel_constants,
)
from bist_quant.signals.fundamental_keys import (
    DIVIDENDS_PAID_KEYS,
    GROSS_PROFIT_KEYS,
    NET_INCOME_KEYS,
    OPERATING_INCOME_KEYS,
    REVENUE_KEYS,
)

logger = logging.getLogger(__name__)

_C = panel_constants()

def build_trading_intensity_panels(
    close: pd.DataFrame,
    volume_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
    data_loader=None,
) -> Dict[str, pd.DataFrame]:
    """
    Build Trading Intensity panels: relative volume, volume trend, turnover velocity.

    Trading intensity measures how actively a stock is being traded relative to:
    1. Its own historical volume (relative volume)
    2. Recent vs older trading activity (volume trend)
    3. How fast shares change hands (turnover velocity using shares outstanding)

    This is conceptually different from liquidity:
    - Liquidity = ease of trading without price impact (Amihud, spreads)
    - Trading Intensity = level of trading activity / attention
    """
    dates, tickers = validate_reference_axes(dates, tickers, "build_trading_intensity_panels")
    del close  # Not needed for this panel family.

    logger.info("  Building trading intensity panels...")

    volume = align_numeric_panel(volume_df, "volume_df", dates, tickers)

    # Relative Volume = volume / avg volume (252-day baseline)
    # High relative volume indicates unusual trading activity
    avg_volume = volume.rolling(252, min_periods=63).mean()
    relative_volume = (volume / avg_volume.replace(0, np.nan)).rolling(
        _C["TURNOVER_LOOKBACK_DAYS"], min_periods=21
    ).mean()
    relative_volume = relative_volume.replace([np.inf, -np.inf], np.nan)

    # Volume Trend = short-term avg / long-term avg - 1
    # Positive trend means increasing trading activity
    short_vol = volume.rolling(21, min_periods=10).mean()
    long_vol = volume.rolling(126, min_periods=42).mean()
    volume_trend = (short_vol / long_vol.replace(0, np.nan) - 1.0).replace([np.inf, -np.inf], np.nan)

    # Turnover Velocity = real turnover rate (volume / shares outstanding)
    # How fast the float is turning over
    turnover_velocity = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)

    shares_aligned, has_shares = load_shares_panel(data_loader, dates, tickers)
    if has_shares:
        # Daily turnover rate
        daily_turnover = volume / shares_aligned.replace(0, np.nan)
        # Smooth and annualize (252 trading days)
        turnover_velocity = daily_turnover.rolling(21, min_periods=10).mean() * 252
        turnover_velocity = turnover_velocity.replace([np.inf, -np.inf], np.nan)
        logger.info("    Turnover velocity computed using shares outstanding")
    else:
        logger.warning("    ⚠️  Shares outstanding not available - turnover velocity will be empty")

    logger.info(f"    Trading intensity panels: RelVol {relative_volume.notna().sum().sum()}, "
          f"VolTrend {volume_trend.notna().sum().sum()}, "
          f"TurnoverVel {turnover_velocity.notna().sum().sum()} data points")

    return finalize_builder_outputs("build_trading_intensity_panels", {
            "trading_intensity_relative_volume": relative_volume,
            "trading_intensity_volume_trend": volume_trend,
            "trading_intensity_turnover_velocity": turnover_velocity,
        },
        "trading_intensity",
        dates,
        tickers,
    )


