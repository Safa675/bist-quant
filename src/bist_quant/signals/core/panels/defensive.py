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

def build_defensive_panels(
    fundamentals: Dict,
    close: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
    data_loader,
) -> Dict[str, pd.DataFrame]:
    """
    Build Defensive/Cyclical panels: earnings stability, beta to market.
    """
    del fundamentals  # Unused in current implementation.
    dates, tickers = validate_reference_axes(dates, tickers, "build_defensive_panels")
    close = align_numeric_panel(close, "close", dates, tickers)

    logger.info("  Building defensive factor panels...")
    fundamentals_parquet = data_loader.load_fundamentals_parquet() if data_loader is not None else None

    stability_panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)

    if fundamentals_parquet is not None:
        count = 0
        success_count = 0
        for ticker in tickers:
            try:
                inc = get_consolidated_sheet(fundamentals_parquet, ticker, _C["INCOME_SHEET"])
                if inc.empty:
                    continue

                net_income_row = pick_row_from_sheet(inc, NET_INCOME_KEYS)
                if net_income_row is not None:
                    ni = coerce_quarter_cols(net_income_row)
                    if not ni.empty:
                        ni_ttm = sum_ttm(ni)

                        # Rolling earnings stability (inverse of coefficient of variation)
                        rolling_mean = ni_ttm.rolling(_C["EARNINGS_STABILITY_QUARTERS"], min_periods=4).mean()
                        rolling_std = ni_ttm.rolling(_C["EARNINGS_STABILITY_QUARTERS"], min_periods=4).std()
                        cv = rolling_std / rolling_mean.abs().replace(0, np.nan)
                        stability = 1.0 / cv.replace(0, np.nan)
                        stability = stability.replace([np.inf, -np.inf], np.nan).clip(-10, 10)

                        if not stability.dropna().empty:
                            stability_panel[ticker] = apply_lag(stability, dates)
                            success_count += 1

            except KeyError:
                # Ticker not found
                continue
            except (ValueError, TypeError):
                # Data conversion issues
                continue

            count += 1
            if count % 50 == 0:
                logger.info(f"    Defensive progress: {count}/{len(tickers)} ({success_count} with data)")

        logger.info(f"    Earnings stability built: {success_count}/{len(tickers)} tickers with data")

    # Beta to market
    from bist_quant.signals.core.panels.vol_beta import build_market_beta_panel

    beta_panel = build_market_beta_panel(close, dates, data_loader)

    return finalize_builder_outputs("build_defensive_panels", {
            "defensive_earnings_stability": stability_panel,
            "defensive_beta_to_market": beta_panel,
        },
        "defensive",
        dates,
        tickers,
    )


