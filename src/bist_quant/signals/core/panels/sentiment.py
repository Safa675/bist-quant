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

def build_sentiment_panels(
    close: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
) -> Dict[str, pd.DataFrame]:
    """
    Build Sentiment/Price Action panels: 52-week high proximity, price acceleration, reversal.
    """
    dates, tickers = validate_reference_axes(dates, tickers, "build_sentiment_panels")
    close = align_numeric_panel(close, "close", dates, tickers)

    logger.info("  Building sentiment/price action panels...")

    # 52-week high proximity: current price / 52-week high
    rolling_high = close.rolling(252, min_periods=126).max()
    high_proximity = close / rolling_high.replace(0, np.nan)
    high_proximity = high_proximity.replace([np.inf, -np.inf], np.nan)

    # Price acceleration: short-term momentum - medium-term momentum
    mom_fast = close.pct_change(_C["PRICE_ACCELERATION_FAST"])
    mom_slow = close.pct_change(_C["PRICE_ACCELERATION_SLOW"])
    price_acceleration = mom_fast - mom_slow

    # Short-term reversal: negative of very short-term return (mean reversion)
    reversal = -close.pct_change(_C["REVERSAL_LOOKBACK_DAYS"])

    logger.info(f"    Sentiment panels: 52wHigh {high_proximity.notna().sum().sum()}, "
          f"Accel {price_acceleration.notna().sum().sum()}, "
          f"Reversal {reversal.notna().sum().sum()} data points")

    return finalize_builder_outputs("build_sentiment_panels", {
            "sentiment_52w_high_pct": high_proximity,
            "sentiment_price_acceleration": price_acceleration,
            "sentiment_reversal": reversal,
        },
        "sentiment",
        dates,
        tickers,
    )


