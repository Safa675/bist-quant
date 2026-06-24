"""Profit margin panels (profit_margin axis — not BUILDERS profitability)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from bist_quant.common.utils import (
    apply_lag,
    coerce_quarter_cols,
    get_consolidated_sheet,
    pick_row,
    pick_row_from_sheet,
    sum_ttm,
)
from bist_quant.signals.core.panels._helpers import panel_constants
from bist_quant.signals.fundamental_keys import (
    GROSS_PROFIT_KEYS,
    OPERATING_INCOME_KEYS,
    REVENUE_KEYS,
)

logger = logging.getLogger(__name__)
_C = panel_constants()


def calculate_margin_level_growth_for_ticker(
    xlsx_path: Path | None,
    ticker: str,
    fundamentals_parquet: pd.DataFrame | None = None,
) -> tuple[pd.Series | None, pd.Series | None]:
    """Return profitability margin level and YoY margin growth series for one ticker."""
    income_sheet = _C["INCOME_SHEET"]
    if fundamentals_parquet is not None:
        inc = get_consolidated_sheet(fundamentals_parquet, ticker, income_sheet)
        if inc.empty:
            return None, None
        rev_row = pick_row_from_sheet(inc, REVENUE_KEYS)
        op_row = pick_row_from_sheet(inc, OPERATING_INCOME_KEYS)
        gp_row = pick_row_from_sheet(inc, GROSS_PROFIT_KEYS)
    else:
        if xlsx_path is None:
            return None, None
        try:
            inc = pd.read_excel(xlsx_path, sheet_name=income_sheet)
        except Exception:
            return None, None
        rev_row = pick_row(inc, REVENUE_KEYS)
        op_row = pick_row(inc, OPERATING_INCOME_KEYS)
        gp_row = pick_row(inc, GROSS_PROFIT_KEYS)

    if rev_row is None or op_row is None or gp_row is None:
        return None, None

    rev = coerce_quarter_cols(rev_row)
    op = coerce_quarter_cols(op_row)
    gp = coerce_quarter_cols(gp_row)
    if rev.empty or op.empty or gp.empty:
        return None, None

    rev_ttm = sum_ttm(rev)
    op_ttm = sum_ttm(op)
    gp_ttm = sum_ttm(gp)
    if rev_ttm.empty or op_ttm.empty or gp_ttm.empty:
        return None, None

    combined = pd.concat([rev_ttm, op_ttm, gp_ttm], axis=1, join="inner").dropna()
    if combined.empty:
        return None, None
    combined.columns = ["RevenueTTM", "OperatingIncomeTTM", "GrossProfitTTM"]

    revenue = combined["RevenueTTM"].replace(0.0, np.nan)
    op_margin = combined["OperatingIncomeTTM"] / revenue
    gp_margin = combined["GrossProfitTTM"] / revenue
    margin_level = (0.60 * op_margin + 0.40 * gp_margin).replace([np.inf, -np.inf], np.nan).dropna()
    if margin_level.empty:
        return None, None

    margin_growth = margin_level.diff(4).replace([np.inf, -np.inf], np.nan).dropna()
    return margin_level.sort_index(), margin_growth.sort_index() if not margin_growth.empty else None


def build_profitability_margin_panels(
    fundamentals: Dict,
    dates: pd.DatetimeIndex,
    data_loader,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build lagged margin-level and margin-growth panels."""
    fundamentals_parquet = data_loader.load_fundamentals_parquet() if data_loader is not None else None
    level_panel: Dict[str, pd.Series] = {}
    growth_panel: Dict[str, pd.Series] = {}

    count = 0
    for ticker, fund_data in fundamentals.items():
        xlsx_path = fund_data.get("path") if isinstance(fund_data, dict) else None
        margin_level, margin_growth = calculate_margin_level_growth_for_ticker(
            xlsx_path=xlsx_path,
            ticker=ticker,
            fundamentals_parquet=fundamentals_parquet,
        )

        if margin_level is not None and not margin_level.empty:
            lagged_level = apply_lag(margin_level, dates)
            if not lagged_level.empty:
                level_panel[ticker] = lagged_level

        if margin_growth is not None and not margin_growth.empty:
            lagged_growth = apply_lag(margin_growth, dates)
            if not lagged_growth.empty:
                growth_panel[ticker] = lagged_growth

        count += 1
        if count % 100 == 0:
            logger.info(f"  Profitability margin progress: {count}/{len(fundamentals)}")

    return pd.DataFrame(level_panel, index=dates), pd.DataFrame(growth_panel, index=dates)


__all__ = [
    "calculate_margin_level_growth_for_ticker",
    "build_profitability_margin_panels",
]
