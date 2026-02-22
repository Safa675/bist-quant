"""
Profitability Signal Construction

Calculates profitability scores based on:
- Operating Income / Total Assets (50%)
- Gross Profit / Total Assets (50%)
"""

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# Add parent to path
from bist_quant.common.utils import (
    apply_lag,
    apply_staleness_weighting,
    coerce_quarter_cols,
    get_consolidated_sheet,
    pick_row,
    pick_row_from_sheet,
    validate_signal_panel_schema,
)

logger = logging.getLogger(__name__)


# Fundamental data keys
INCOME_SHEET = "Gelir Tablosu (Çeyreklik)"
BALANCE_SHEET = "Bilanço"

OPERATING_INCOME_KEYS = (
    "Faaliyet Karı (Zararı)",
    "Finansman Geliri (Gideri) Öncesi Faaliyet Karı (Zararı)",
    "FAALİYET KARI (ZARARI)",
)
GROSS_PROFIT_KEYS = (
    "Brüt Kar (Zarar)",
    "Ticari Faaliyetlerden Brüt Kar (Zarar)",
    "TİCARİ FAALİYETLERDEN BRÜT KAR (ZARAR)",
)
TOTAL_ASSETS_KEYS = (
    "Toplam Varlıklar",
    "Toplam Aktifler",
    "TOPLAM VARLIKLAR",
)


def calculate_profitability_for_ticker(
    xlsx_path: Path | None,
    ticker: str,
    fundamentals_parquet: pd.DataFrame | None = None,
    operating_income_weight: float = 0.5,
    gross_profit_weight: float = 0.5,
) -> pd.Series | None:
    """Calculate profitability ratio for a single ticker"""
    if fundamentals_parquet is not None:
        inc = get_consolidated_sheet(fundamentals_parquet, ticker, INCOME_SHEET)
        bs = get_consolidated_sheet(fundamentals_parquet, ticker, BALANCE_SHEET)
        if inc.empty and bs.empty:
            return None
        op_row = pick_row_from_sheet(inc, OPERATING_INCOME_KEYS)
        gp_row = pick_row_from_sheet(inc, GROSS_PROFIT_KEYS)
        ta_row = pick_row_from_sheet(bs, TOTAL_ASSETS_KEYS)
    else:
        if xlsx_path is None:
            return None
        try:
            inc = pd.read_excel(xlsx_path, sheet_name=INCOME_SHEET)
            bs = pd.read_excel(xlsx_path, sheet_name=BALANCE_SHEET)
        except Exception:
            return None

        op_row = pick_row(inc, OPERATING_INCOME_KEYS)
        gp_row = pick_row(inc, GROSS_PROFIT_KEYS)
        ta_row = pick_row(bs, TOTAL_ASSETS_KEYS)
    if op_row is None or gp_row is None or ta_row is None:
        return None

    op = coerce_quarter_cols(op_row)
    gp = coerce_quarter_cols(gp_row)
    ta = coerce_quarter_cols(ta_row)
    if op.empty or gp.empty or ta.empty:
        return None

    combined = pd.concat([op, gp, ta], axis=1, join="inner")
    combined.columns = ["OperatingIncome", "GrossProfit", "TotalAssets"]
    combined = combined.dropna()
    if combined.empty:
        return None

    total_weight = operating_income_weight + gross_profit_weight
    if total_weight <= 0:
        operating_income_weight = 0.5
        gross_profit_weight = 0.5
        total_weight = 1.0

    op_weight = operating_income_weight / total_weight
    gp_weight = gross_profit_weight / total_weight

    # Profitability = weighted sum of operating and gross profitability
    ratio = op_weight * (combined["OperatingIncome"] / combined["TotalAssets"]) + gp_weight * (
        combined["GrossProfit"] / combined["TotalAssets"]
    )
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    if ratio.empty:
        return None
    return ratio.sort_index()


def build_profitability_signals(
    fundamentals: Dict,
    dates: pd.DatetimeIndex,
    data_loader=None,
    operating_income_weight: float = 0.5,
    gross_profit_weight: float = 0.5,
) -> pd.DataFrame:
    """
    Build profitability signal panel with proper lag
    
    Returns:
        DataFrame (dates x tickers) with profitability scores
    """
    logger.info("\n🔧 Building profitability signals...")
    logger.info(
        f"  Component weights: operating_income={operating_income_weight:.2f}, "
        f"gross_profit={gross_profit_weight:.2f}"
    )
    
    fundamentals_parquet = data_loader.load_fundamentals_parquet() if data_loader is not None else None

    panel = {}
    count = 0
    
    for ticker, fund_data in fundamentals.items():
        xlsx_path = fund_data['path']
        prof_series = calculate_profitability_for_ticker(
            xlsx_path,
            ticker,
            fundamentals_parquet,
            operating_income_weight=operating_income_weight,
            gross_profit_weight=gross_profit_weight,
        )
        
        if prof_series is not None:
            # Apply lag
            lagged = apply_lag(prof_series, dates)
            if not lagged.empty:
                panel[ticker] = lagged
                count += 1
                if count % 50 == 0:
                    logger.info(f"  Processed {count} tickers...")
    
    result = pd.DataFrame(panel, index=dates)

    # Apply staleness-based down-weighting (Part D)
    result = apply_staleness_weighting(result)

    result = validate_signal_panel_schema(
        result,
        dates=dates,
        tickers=pd.Index(sorted(fundamentals.keys())),
        signal_name="profitability",
        context="final score panel",
        dtype=np.float32,
    )
    logger.info(f"  ✅ Profitability signals: {result.shape[0]} days × {result.shape[1]} tickers")
    return result
