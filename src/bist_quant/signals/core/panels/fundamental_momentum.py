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

def build_fundamental_momentum_panels(
    fundamentals: Dict,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
    data_loader,
) -> Dict[str, pd.DataFrame]:
    """
    Build Fundamental Momentum panels: margin change, sales growth acceleration.
    """
    del fundamentals  # Unused in current implementation.
    dates, tickers = validate_reference_axes(dates, tickers, "build_fundamental_momentum_panels")

    logger.info("  Building fundamental momentum panels...")
    fundamentals_parquet = data_loader.load_fundamentals_parquet() if data_loader is not None else None

    margin_change_panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)
    sales_accel_panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)

    if fundamentals_parquet is None:
        logger.warning("    ⚠️  No fundamentals parquet - fundmom panels will be empty")
        return finalize_builder_outputs("build_fundamental_momentum_panels", {
                "fundmom_margin_change": margin_change_panel,
                "fundmom_sales_accel": sales_accel_panel,
            },
            "fundmom",
            dates,
            tickers,
        )

    count = 0
    success_count = 0
    for ticker in tickers:
        try:
            inc = get_consolidated_sheet(fundamentals_parquet, ticker, _C["INCOME_SHEET"])
            if inc.empty:
                continue

            revenue_row = pick_row_from_sheet(inc, REVENUE_KEYS)
            op_income_row = pick_row_from_sheet(inc, OPERATING_INCOME_KEYS)

            if revenue_row is not None and op_income_row is not None:
                rev = coerce_quarter_cols(revenue_row)
                op = coerce_quarter_cols(op_income_row)

                if not rev.empty and not op.empty:
                    rev_ttm = sum_ttm(rev)
                    op_ttm = sum_ttm(op)

                    # Operating margin and its YoY change
                    op_margin = op_ttm / rev_ttm.replace(0, np.nan)
                    margin_change = op_margin.diff(_C["MARGIN_CHANGE_QUARTERS"])
                    margin_change = margin_change.replace([np.inf, -np.inf], np.nan)
                    if not margin_change.dropna().empty:
                        margin_change_panel[ticker] = apply_lag(margin_change, dates)

                    # Sales growth and its acceleration
                    sales_growth = rev_ttm.pct_change(_C["SALES_ACCEL_QUARTERS"])
                    sales_accel = sales_growth.diff(_C["SALES_ACCEL_QUARTERS"])
                    sales_accel = sales_accel.replace([np.inf, -np.inf], np.nan)
                    if not sales_accel.dropna().empty:
                        sales_accel_panel[ticker] = apply_lag(sales_accel, dates)

                    success_count += 1

        except KeyError:
            # Ticker not found in fundamentals
            continue
        except (ValueError, TypeError):
            # Data conversion issues
            continue

        count += 1
        if count % 50 == 0:
            logger.info(f"    FundMom progress: {count}/{len(tickers)} ({success_count} with data)")

    logger.info(f"    FundMom panels built: {success_count}/{len(tickers)} tickers with data")

    return finalize_builder_outputs("build_fundamental_momentum_panels", {
            "fundmom_margin_change": margin_change_panel,
            "fundmom_sales_accel": sales_accel_panel,
        },
        "fundmom",
        dates,
        tickers,
    )


