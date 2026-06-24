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

def build_realized_volatility_panel(
    close: pd.DataFrame,
    dates: pd.DatetimeIndex,
    lookback: int = _C["VOLATILITY_LOOKBACK_DAYS"],
) -> pd.DataFrame:
    """Build rolling realized volatility panel (annualized)."""
    dates, tickers = validate_reference_axes(dates, close.columns, "build_realized_volatility_panel")
    close = align_numeric_panel(close, "close", dates, tickers)

    daily_returns = close.pct_change()
    min_obs = max(int(lookback * _C["MIN_ROLLING_OBS_RATIO"]), 21)
    vol = daily_returns.rolling(lookback, min_periods=min_obs).std() * np.sqrt(252)
    return align_numeric_panel(vol, "realized_volatility", dates, tickers)



def build_market_beta_panel(
    close: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
    lookback: int = _C["BETA_LOOKBACK_DAYS"],
) -> pd.DataFrame:
    """Build rolling market beta panel.

    Note: This uses a loop over tickers which can be slow for large universes.
    Consider vectorization if performance is critical.
    """
    del data_loader  # unused
    dates, tickers = validate_reference_axes(dates, close.columns, "build_market_beta_panel")
    close = align_numeric_panel(close, "close", dates, tickers)

    daily_returns = close.pct_change()
    min_obs = max(int(lookback * _C["MIN_ROLLING_OBS_RATIO"]), _C["BETA_MIN_OBS"])

    # Equal-weighted market return as benchmark
    market_return = daily_returns.mean(axis=1)
    market_var = market_return.rolling(lookback, min_periods=min_obs).var()

    beta_panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)

    for ticker in tickers:
        stock_ret = daily_returns[ticker]
        cov = stock_ret.rolling(lookback, min_periods=min_obs).cov(market_return)
        beta = cov / market_var.replace(0.0, np.nan)
        beta_panel[ticker] = beta.reindex(dates)

    beta_panel = beta_panel.clip(lower=-2.0, upper=5.0)
    return align_numeric_panel(beta_panel, "market_beta", dates, tickers)



def build_volatility_beta_panels(
    close: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build realized volatility and market beta panels for risk axis."""
    dates, tickers = validate_reference_axes(dates, close.columns, "build_volatility_beta_panels")
    close = align_numeric_panel(close, "close", dates, tickers)

    logger.info("  Building volatility panel...")
    vol_panel = build_realized_volatility_panel(close, dates)

    logger.info("  Building market beta panel...")
    beta_panel = build_market_beta_panel(close, dates, data_loader)

    return vol_panel, beta_panel
