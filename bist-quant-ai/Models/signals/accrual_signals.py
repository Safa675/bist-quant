"""
Accrual Anomaly Signal Construction

The accrual anomaly exploits the fact that companies with high accruals 
(non-cash earnings) tend to underperform, while low accrual companies outperform.

Accruals Formula (Sloan 1996):
BS_ACC = (ΔCA - ΔCash) - (ΔCL - ΔSTD - ΔITP) - Dep

Where:
- ΔCA = annual change in current assets
- ΔCash = change in cash and cash equivalents
- ΔCL = change in current liabilities
- ΔSTD = change in short-term debt (current debt)
- ΔITP = change in income taxes payable
- Dep = annual depreciation and amortization expense

Lower accruals = Higher quality earnings = Better signal score
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.utils import (
    pick_row,
    coerce_quarter_cols,
    apply_lag,
    get_consolidated_sheet,
    pick_row_from_sheet,
)


# Fundamental data keys
BALANCE_SHEET = "Bilanço"
INCOME_SHEET = "Gelir Tablosu (Çeyreklik)"

# Balance Sheet Keys
CURRENT_ASSETS_KEYS = (
    "Dönen Varlıklar",
    "Toplam Dönen Varlıklar",
)
CASH_KEYS = (
    "Nakit ve Nakit Benzerleri",
    "Nakit ve Nakit Benzeri Varlıklar",
)
CURRENT_LIABILITIES_KEYS = (
    "Kısa Vadeli Yükümlülükler",
    "Toplam Kısa Vadeli Yükümlülükler",
)
CURRENT_DEBT_KEYS = (
    "Finansal Borçlar",
    "Kısa Vadeli Finansal Borçlar",
    "Kısa Vadeli Borçlanmalar",
)
INCOME_TAX_PAYABLE_KEYS = (
    "Dönem Karı Vergi Yükümlülüğü",
    "Cari Dönem Vergisi ile İlgili Yükümlülükler",
    "Kısa Vadeli Borç Karşılıkları",
)
TOTAL_ASSETS_KEYS = (
    "Toplam Varlıklar",
    "Toplam Aktifler",
)

# Income Statement Keys
DEPRECIATION_KEYS = (
    "Amortisman ve İtfa Giderleri",
    "Amortisman ve İtfa Gideri",
    "Amortisman Giderleri",
)


def calculate_accruals_for_ticker(
    xlsx_path: Path | None,
    ticker: str,
    fundamentals_parquet: pd.DataFrame | None = None,
) -> pd.Series | None:
    """Calculate accruals for a single ticker"""
    
    # Load data
    if fundamentals_parquet is not None:
        bs = get_consolidated_sheet(fundamentals_parquet, ticker, BALANCE_SHEET)
        inc = get_consolidated_sheet(fundamentals_parquet, ticker, INCOME_SHEET)
        if bs.empty and inc.empty:
            return None
        
        ca_row = pick_row_from_sheet(bs, CURRENT_ASSETS_KEYS)
        cash_row = pick_row_from_sheet(bs, CASH_KEYS)
        cl_row = pick_row_from_sheet(bs, CURRENT_LIABILITIES_KEYS)
        cd_row = pick_row_from_sheet(bs, CURRENT_DEBT_KEYS)
        itp_row = pick_row_from_sheet(bs, INCOME_TAX_PAYABLE_KEYS)
        ta_row = pick_row_from_sheet(bs, TOTAL_ASSETS_KEYS)
        dep_row = pick_row_from_sheet(inc, DEPRECIATION_KEYS)
    else:
        if xlsx_path is None:
            return None
        try:
            bs = pd.read_excel(xlsx_path, sheet_name=BALANCE_SHEET)
            inc = pd.read_excel(xlsx_path, sheet_name=INCOME_SHEET)
        except Exception:
            return None
        
        ca_row = pick_row(bs, CURRENT_ASSETS_KEYS)
        cash_row = pick_row(bs, CASH_KEYS)
        cl_row = pick_row(bs, CURRENT_LIABILITIES_KEYS)
        cd_row = pick_row(bs, CURRENT_DEBT_KEYS)
        itp_row = pick_row(bs, INCOME_TAX_PAYABLE_KEYS)
        ta_row = pick_row(bs, TOTAL_ASSETS_KEYS)
        dep_row = pick_row(inc, DEPRECIATION_KEYS)
    
    # Check if we have minimum required data
    if ca_row is None or cash_row is None or cl_row is None or ta_row is None:
        return None
    
    # Convert to series
    ca = coerce_quarter_cols(ca_row)
    cash = coerce_quarter_cols(cash_row)
    cl = coerce_quarter_cols(cl_row)
    ta = coerce_quarter_cols(ta_row)
    
    # Optional fields - use zeros if not available
    if cd_row is not None:
        cd = coerce_quarter_cols(cd_row)
    else:
        cd = pd.Series(0, index=ca.index)
    
    if itp_row is not None:
        itp = coerce_quarter_cols(itp_row)
    else:
        itp = pd.Series(0, index=ca.index)
    
    if dep_row is not None:
        dep = coerce_quarter_cols(dep_row)
    else:
        dep = pd.Series(0, index=ca.index)
    
    if ca.empty or cash.empty or cl.empty or ta.empty:
        return None
    
    # Align all series
    combined = pd.concat([ca, cash, cl, cd, itp, dep, ta], axis=1, join='inner')
    combined.columns = ['CA', 'Cash', 'CL', 'CD', 'ITP', 'Dep', 'TA']
    combined = combined.dropna()
    
    if len(combined) < 2:  # Need at least 2 periods to calculate changes
        return None
    
    # Calculate changes (current - previous)
    delta_ca = combined['CA'].diff()
    delta_cash = combined['Cash'].diff()
    delta_cl = combined['CL'].diff()
    delta_cd = combined['CD'].diff()
    delta_itp = combined['ITP'].diff()
    
    # Use current period depreciation
    dep = combined['Dep']
    
    # Use average total assets (current + previous) / 2
    avg_ta = (combined['TA'] + combined['TA'].shift(1)) / 2
    
    # Calculate accruals: BS_ACC = (ΔCA - ΔCash) - (ΔCL - ΔSTD - ΔITP) - Dep
    accruals = (
        (delta_ca - delta_cash) - 
        (delta_cl - delta_cd - delta_itp) - 
        dep
    ) / avg_ta
    
    # Remove infinities and NaNs
    accruals = accruals.replace([np.inf, -np.inf], np.nan).dropna()
    
    if accruals.empty:
        return None
    
    # Invert: Lower accruals = Higher score (we want low accruals)
    # Multiply by -1 so that low accruals get high scores
    quality_score = -accruals
    
    return quality_score.sort_index()


def build_accrual_signals(
    fundamentals: Dict,
    dates: pd.DatetimeIndex,
    data_loader=None,
) -> pd.DataFrame:
    """
    Build accrual quality signal panel with proper lag
    
    Returns:
        DataFrame (dates x tickers) with accrual quality scores
        Higher score = Lower accruals = Better quality
    """
    print("\n🔧 Building accrual quality signals...")
    print("  Formula: BS_ACC = (ΔCA - ΔCash) - (ΔCL - ΔSTD - ΔITP) - Dep")
    print("  Lower accruals = Higher quality earnings")
    
    fundamentals_parquet = data_loader.load_fundamentals_parquet() if data_loader is not None else None
    
    panel = {}
    count = 0
    
    for ticker, fund_data in fundamentals.items():
        xlsx_path = fund_data['path']
        accrual_series = calculate_accruals_for_ticker(
            xlsx_path,
            ticker,
            fundamentals_parquet,
        )
        
        if accrual_series is not None:
            # Apply lag
            lagged = apply_lag(accrual_series, dates)
            if not lagged.empty:
                panel[ticker] = lagged
                count += 1
                if count % 50 == 0:
                    print(f"  Processed {count} tickers...")
    
    result = pd.DataFrame(panel, index=dates)
    
    # Summary stats
    if not result.empty:
        latest = result.iloc[-1].dropna()
        if len(latest) > 0:
            print(f"  Latest scores - Mean: {latest.mean():.4f}, Std: {latest.std():.4f}")
            print(f"  Latest scores - Min: {latest.min():.4f}, Max: {latest.max():.4f}")
            
            # Show top 5 quality stocks (highest scores = lowest accruals)
            top_5 = latest.nlargest(5)
            print(f"  Top 5 quality stocks (lowest accruals): {', '.join(top_5.index.tolist())}")
    
    print(f"  ✅ Accrual signals: {result.shape[0]} days × {result.shape[1]} tickers")
    return result
