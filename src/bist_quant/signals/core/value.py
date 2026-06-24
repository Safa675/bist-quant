"""Value factor metric calculators (shared across BUILDERS and standalone)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from bist_quant.common.utils import (
    coerce_quarter_cols,
    get_consolidated_sheet,
    pick_row,
    pick_row_from_sheet,
    sum_ttm,
)
from bist_quant.signals.core.constants import (
    BALANCE_SHEET,
    CASH_FLOW_SHEET,
    INCOME_SHEET,
)
from bist_quant.signals.fundamental_keys import (
    CAPEX_KEYS,
    CASH_KEYS,
    EBITDA_KEYS,
    NET_INCOME_KEYS,
    OPERATING_CF_KEYS as OPERATING_CASH_FLOW_KEYS,
    REVENUE_KEYS,
    TOTAL_DEBT_KEYS,
)


def calculate_value_metrics_for_ticker(
    xlsx_path: Path | None,
    ticker: str,
    data_loader,
    fundamentals_parquet: pd.DataFrame | None = None,
) -> Dict:
    """Calculate value metrics for a single ticker."""
    if fundamentals_parquet is not None:
        inc = get_consolidated_sheet(fundamentals_parquet, ticker, INCOME_SHEET)
        bs = get_consolidated_sheet(fundamentals_parquet, ticker, BALANCE_SHEET)
        cf = get_consolidated_sheet(fundamentals_parquet, ticker, CASH_FLOW_SHEET)
        if inc.empty and bs.empty and cf.empty:
            return {}
        ni_row = pick_row_from_sheet(inc, NET_INCOME_KEYS)
        rev_row = pick_row_from_sheet(inc, REVENUE_KEYS)
        ebitda_row = pick_row_from_sheet(inc, EBITDA_KEYS)
        ocf_row = pick_row_from_sheet(cf, OPERATING_CASH_FLOW_KEYS)
        capex_row = pick_row_from_sheet(cf, CAPEX_KEYS)
        debt_row = pick_row_from_sheet(bs, TOTAL_DEBT_KEYS)
        cash_row = pick_row_from_sheet(bs, CASH_KEYS)
    else:
        if xlsx_path is None:
            return {}
        try:
            inc = pd.read_excel(xlsx_path, sheet_name=INCOME_SHEET)
            bs = pd.read_excel(xlsx_path, sheet_name=BALANCE_SHEET)
            cf = pd.read_excel(xlsx_path, sheet_name=CASH_FLOW_SHEET)
        except Exception:
            return {}
        ni_row = pick_row(inc, NET_INCOME_KEYS)
        rev_row = pick_row(inc, REVENUE_KEYS)
        ebitda_row = pick_row(inc, EBITDA_KEYS)
        ocf_row = pick_row(cf, OPERATING_CASH_FLOW_KEYS)
        capex_row = pick_row(cf, CAPEX_KEYS)
        debt_row = pick_row(bs, TOTAL_DEBT_KEYS)
        cash_row = pick_row(bs, CASH_KEYS)

    ni = coerce_quarter_cols(ni_row) if ni_row is not None else pd.Series(dtype=float)
    rev = coerce_quarter_cols(rev_row) if rev_row is not None else pd.Series(dtype=float)
    ebitda = coerce_quarter_cols(ebitda_row) if ebitda_row is not None else pd.Series(dtype=float)
    ocf = coerce_quarter_cols(ocf_row) if ocf_row is not None else pd.Series(dtype=float)
    capex = coerce_quarter_cols(capex_row) if capex_row is not None else pd.Series(dtype=float)
    debt = coerce_quarter_cols(debt_row) if debt_row is not None else pd.Series(dtype=float)
    cash = coerce_quarter_cols(cash_row) if cash_row is not None else pd.Series(dtype=float)

    ni_ttm = sum_ttm(ni)
    rev_ttm = sum_ttm(rev)
    ebitda_ttm = sum_ttm(ebitda)
    ocf_ttm = sum_ttm(ocf)
    capex_ttm = sum_ttm(capex)
    capex_aligned = (
        capex_ttm.reindex(ocf_ttm.index, method="ffill").fillna(0)
        if not capex_ttm.empty
        else pd.Series(0.0, index=ocf_ttm.index)
    )
    fcf_ttm = ocf_ttm - capex_aligned

    return {
        "net_income_ttm": ni_ttm,
        "revenue_ttm": rev_ttm,
        "ebitda_ttm": ebitda_ttm,
        "ocf_ttm": ocf_ttm,
        "fcf_ttm": fcf_ttm,
        "debt": debt,
        "cash": cash,
        "shares_outstanding": data_loader.load_shares_outstanding(ticker),
    }


__all__ = [
    "INCOME_SHEET",
    "BALANCE_SHEET",
    "CASH_FLOW_SHEET",
    "calculate_value_metrics_for_ticker",
]
