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

def build_liquidity_panels(
    close: pd.DataFrame,
    volume_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
    data_loader=None,
) -> Dict[str, pd.DataFrame]:
    """
    Build Liquidity factor panels: Amihud illiquidity, real turnover, bid-ask spread proxy.

    Real turnover = volume / shares_outstanding (from SERMAYE column in isyatirim data)
    This is distinct from trading intensity which uses relative volume measures.
    """
    dates, tickers = validate_reference_axes(dates, tickers, "build_liquidity_panels")
    close = align_numeric_panel(close, "close", dates, tickers)

    logger.info("  Building liquidity factor panels...")

    daily_returns = close.pct_change().abs()
    volume = align_numeric_panel(volume_df, "volume_df", dates, tickers)

    # Amihud illiquidity = |return| / dollar volume (lower = more liquid)
    dollar_volume = close * volume
    amihud_daily = daily_returns / dollar_volume.replace(0, np.nan)
    amihud_panel = amihud_daily.rolling(_C["AMIHUD_LOOKBACK_DAYS"], min_periods=10).mean()
    # Log transform to reduce skewness
    amihud_panel = np.log1p(amihud_panel * 1e6).replace([np.inf, -np.inf], np.nan)

    # Real Turnover = volume / shares_outstanding
    turnover_panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)

    shares_aligned, has_shares = load_shares_panel(data_loader, dates, tickers)
    if has_shares:
        # Real turnover = daily volume / shares outstanding
        real_turnover = volume / shares_aligned.replace(0, np.nan)
        # Smooth with 63-day rolling mean
        turnover_panel = real_turnover.rolling(_C["TURNOVER_LOOKBACK_DAYS"], min_periods=21).mean()
        turnover_panel = turnover_panel.replace([np.inf, -np.inf], np.nan)
        logger.info("    Real turnover computed using shares outstanding")
    else:
        logger.warning("    ⚠️  Shares outstanding not available - turnover panel will be empty")

    # Bid-ask spread proxy: high-low range relative to close
    # This captures the spread cost component of liquidity
    # Note: requires high/low data which may not be in close df
    spread_proxy_panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)

    logger.info(f"    Liquidity panels: Amihud {amihud_panel.notna().sum().sum()}, "
          f"Turnover {turnover_panel.notna().sum().sum()} data points")

    return finalize_builder_outputs("build_liquidity_panels", {
            "liquidity_amihud": amihud_panel,
            "liquidity_turnover": turnover_panel,
            "liquidity_spread_proxy": spread_proxy_panel,
        },
        "liquidity",
        dates,
        tickers,
    )


