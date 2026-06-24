"""Investment factor primitives: reinvestment metrics and conservative profile."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from bist_quant.signals.core.constants import (
    BALANCE_SHEET,
    CASH_FLOW_SHEET,
    INCOME_SHEET,
)


def squash_metric(
    panel: pd.DataFrame,
    scale: float,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
) -> pd.DataFrame:
    clipped = panel
    if lower is not None or upper is not None:
        clipped = clipped.clip(lower=lower, upper=upper)
    return np.tanh(clipped / scale)


def build_conservative_profile(
    debt_panel: pd.DataFrame,
    cash_panel: pd.DataFrame,
    current_panel: pd.DataFrame,
    payout_panel: pd.DataFrame,
    *,
    debt_weight: float = -0.55,
    cash_weight: float = 0.30,
    current_weight: float = 0.30,
    payout_weight: float = 0.20,
) -> pd.DataFrame:
    """Conservative investment axis side (five_factor_rotation)."""
    debt_score = squash_metric(debt_panel, scale=1.5, lower=-2.0, upper=6.0).fillna(0.0)
    cash_score = squash_metric(cash_panel, scale=1.0, lower=0.0, upper=8.0).fillna(0.0)
    current_score = squash_metric(current_panel, scale=2.0, lower=0.0, upper=12.0).fillna(0.0)
    payout_score = squash_metric(payout_panel, scale=0.5, lower=-1.0, upper=2.0).fillna(0.0)

    profile = (
        debt_weight * debt_score
        + cash_weight * cash_score
        + current_weight * current_score
        + payout_weight * payout_score
    )
    valid = debt_panel.notna() | cash_panel.notna() | current_panel.notna() | payout_panel.notna()
    return profile.where(valid)


def calculate_investment_metrics_for_ticker(
    xlsx_path: Path | None,
    ticker: str,
    data_loader,
    fundamentals_parquet: pd.DataFrame | None = None,
) -> Dict:
    """Per-ticker reinvestment inputs (shared with investment_signals)."""
    from bist_quant.common.utils import (
        coerce_quarter_cols,
        get_consolidated_sheet,
        pick_row,
        pick_row_from_sheet,
        sum_ttm,
    )
    from bist_quant.signals.fundamental_keys import (
        DIVIDENDS_PAID_KEYS,
        RD_KEYS,
        REVENUE_KEYS,
        TOTAL_ASSETS_KEYS,
    )

    if fundamentals_parquet is not None:
        inc = get_consolidated_sheet(fundamentals_parquet, ticker, INCOME_SHEET)
        bs = get_consolidated_sheet(fundamentals_parquet, ticker, BALANCE_SHEET)
        cf = get_consolidated_sheet(fundamentals_parquet, ticker, CASH_FLOW_SHEET)
        if inc.empty and bs.empty and cf.empty:
            return {}
        rev_row = pick_row_from_sheet(inc, REVENUE_KEYS)
        rd_row = pick_row_from_sheet(inc, RD_KEYS)
        assets_row = pick_row_from_sheet(bs, TOTAL_ASSETS_KEYS)
        div_row = pick_row_from_sheet(cf, DIVIDENDS_PAID_KEYS) if not cf.empty else None
    else:
        if xlsx_path is None:
            return {}
        try:
            inc = pd.read_excel(xlsx_path, sheet_name=INCOME_SHEET)
            bs = pd.read_excel(xlsx_path, sheet_name=BALANCE_SHEET)
            try:
                cf = pd.read_excel(xlsx_path, sheet_name=CASH_FLOW_SHEET)
            except Exception:
                cf = None
        except Exception:
            return {}
        rev_row = pick_row(inc, REVENUE_KEYS)
        rd_row = pick_row(inc, RD_KEYS)
        assets_row = pick_row(bs, TOTAL_ASSETS_KEYS)
        div_row = pick_row(cf, DIVIDENDS_PAID_KEYS) if cf is not None else None

    rev = coerce_quarter_cols(rev_row) if rev_row is not None else pd.Series(dtype=float)
    rd = coerce_quarter_cols(rd_row) if rd_row is not None else pd.Series(dtype=float)
    assets = coerce_quarter_cols(assets_row) if assets_row is not None else pd.Series(dtype=float)
    div = coerce_quarter_cols(div_row) if div_row is not None else pd.Series(dtype=float)

    return {
        "revenue_ttm": sum_ttm(rev),
        "rd_ttm": sum_ttm(rd),
        "dividends_ttm": sum_ttm(div),
        "total_assets": assets,
        "shares_outstanding": data_loader.load_shares_outstanding(ticker),
    }


__all__ = [
    "INCOME_SHEET",
    "BALANCE_SHEET",
    "CASH_FLOW_SHEET",
    "squash_metric",
    "build_conservative_profile",
    "calculate_investment_metrics_for_ticker",
]
