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

def build_carry_panels(
    fundamentals: Dict,
    close: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
    data_loader,
) -> Dict[str, pd.DataFrame]:
    """
    Build Carry factor panels: dividend yield.

    Note: In Turkey, buyback data is rarely available, so we focus on dividends.
    """
    del fundamentals  # Unused in current implementation.
    dates, tickers = validate_reference_axes(dates, tickers, "build_carry_panels")
    close = align_numeric_panel(close, "close", dates, tickers)

    logger.info("  Building carry factor panels...")

    div_yield_panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)

    # Try to load dividend yield from fundamental metrics first
    try:
        metrics_df = data_loader.load_fundamental_metrics()
        if not metrics_df.empty and "dividend_yield" in metrics_df.columns:
            logger.info("    Loading dividend yield from fundamental metrics...")
            div_yield_panel = build_metric_panel(metrics_df, "dividend_yield", dates, tickers)
    except Exception as e:
        logger.warning(f"    ⚠️  Could not load dividend metrics: {e}")

    # If we got data from metrics, we're done
    metrics_count = div_yield_panel.notna().sum().sum()
    if metrics_count > 1000:
        logger.info(f"    Carry panels from metrics: {metrics_count} data points")
        return finalize_builder_outputs("build_carry_panels", {
                "carry_dividend_yield": div_yield_panel,
                "carry_shareholder_yield": div_yield_panel.copy(),  # Same as div yield for Turkey
            },
            "carry",
            dates,
            tickers,
        )

    # Otherwise try to build from cash flow statements
    logger.info("    Building dividend yield from cash flow statements + shares outstanding...")
    fundamentals_parquet = data_loader.load_fundamentals_parquet() if data_loader is not None else None

    if fundamentals_parquet is None:
        logger.warning("    ⚠️  No fundamentals parquet - carry panels will be empty")
        return finalize_builder_outputs("build_carry_panels", {
                "carry_dividend_yield": div_yield_panel,
                "carry_shareholder_yield": div_yield_panel.copy(),
            },
            "carry",
            dates,
            tickers,
        )

    count = 0
    success_count = 0
    for ticker in tickers:
        # Skip if already have data
        if div_yield_panel[ticker].notna().sum() > 100:
            success_count += 1
            continue

        try:
            cf = get_consolidated_sheet(fundamentals_parquet, ticker, _C["CASH_FLOW_SHEET"])
            if cf.empty:
                continue

            div_row = pick_row_from_sheet(cf, DIVIDENDS_PAID_KEYS)
            if div_row is not None:
                divs = coerce_quarter_cols(div_row).abs()  # dividends paid are usually negative
                if not divs.empty:
                    div_ttm = sum_ttm(divs)
                    if div_ttm.empty:
                        continue

                    # Get shares outstanding (check method exists first)
                    shares = None
                    if data_loader and hasattr(data_loader, "load_shares_outstanding"):
                        shares = data_loader.load_shares_outstanding(ticker)
                    if shares is None or shares.empty:
                        continue

                    shares = shares.sort_index()
                    shares = shares[~shares.index.duplicated(keep="last")]

                    # Get price for this ticker
                    if ticker not in close.columns:
                        continue
                    price = close[ticker].reindex(dates)

                    # Apply lag to div_ttm to get it on daily dates
                    div_ttm_daily = apply_lag(div_ttm, dates)
                    if div_ttm_daily.empty or div_ttm_daily.isna().all():
                        continue

                    # Align shares to dates (using ffill() instead of deprecated method param)
                    shares_daily = shares.reindex(dates).ffill()

                    # Market cap = price * shares
                    mcap = price * shares_daily

                    # Dividend yield = TTM dividends / market cap
                    div_yield = div_ttm_daily / mcap.replace(0, np.nan)
                    div_yield = div_yield.replace([np.inf, -np.inf], np.nan)

                    # Clip extreme values (yield > 50% is suspicious)
                    div_yield = div_yield.clip(upper=0.5)

                    if div_yield.notna().sum() > 50:
                        div_yield_panel[ticker] = div_yield
                        success_count += 1

        except KeyError:
            # Ticker not found
            continue
        except (ValueError, TypeError):
            # Data conversion issues
            continue

        count += 1
        if count % 50 == 0:
            logger.info(f"    Carry progress: {count}/{len(tickers)} ({success_count} with data)")

    logger.info(f"    Carry panels built: {success_count}/{len(tickers)} tickers with data")

    return finalize_builder_outputs("build_carry_panels", {
            "carry_dividend_yield": div_yield_panel,
            "carry_shareholder_yield": div_yield_panel.copy(),
        },
        "carry",
        dates,
        tickers,
    )


