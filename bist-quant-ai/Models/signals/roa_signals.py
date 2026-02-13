"""
ROA (Return on Assets) Effect Signal Construction

Companies with high ROA (profitability relative to assets) tend to outperform.
This is a simple but powerful profitability metric.

Formula:
ROA = Net Income / Total Assets

Higher ROA = Higher score = Better profitability
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
INCOME_SHEET = "Gelir Tablosu (Çeyreklik)"
BALANCE_SHEET = "Bilanço"

NET_INCOME_KEYS = (
    "Dönem Karı (Zararı)",
    "Net Dönem Karı (Zararı)",
    "Sürdürülen Faaliyetler Dönem Karı (Zararı)",
)

TOTAL_ASSETS_KEYS = (
    "Toplam Varlıklar",
    "Toplam Aktifler",
)


def calculate_roa_for_ticker(
    xlsx_path: Path | None,
    ticker: str,
    fundamentals_parquet: pd.DataFrame | None = None,
) -> pd.Series | None:
    """Calculate ROA for a single ticker"""
    
    # Load data
    if fundamentals_parquet is not None:
        inc = get_consolidated_sheet(fundamentals_parquet, ticker, INCOME_SHEET)
        bs = get_consolidated_sheet(fundamentals_parquet, ticker, BALANCE_SHEET)
        if inc.empty and bs.empty:
            return None
        
        ni_row = pick_row_from_sheet(inc, NET_INCOME_KEYS)
        ta_row = pick_row_from_sheet(bs, TOTAL_ASSETS_KEYS)
    else:
        if xlsx_path is None:
            return None
        try:
            inc = pd.read_excel(xlsx_path, sheet_name=INCOME_SHEET)
            bs = pd.read_excel(xlsx_path, sheet_name=BALANCE_SHEET)
        except Exception:
            return None
        
        ni_row = pick_row(inc, NET_INCOME_KEYS)
        ta_row = pick_row(bs, TOTAL_ASSETS_KEYS)
    
    if ni_row is None or ta_row is None:
        return None
    
    # Convert to series
    ni = coerce_quarter_cols(ni_row)
    ta = coerce_quarter_cols(ta_row)
    
    if ni.empty or ta.empty:
        return None
    
    # Align series
    combined = pd.concat([ni, ta], axis=1, join='inner')
    combined.columns = ['NetIncome', 'TotalAssets']
    combined = combined.dropna()
    
    if combined.empty:
        return None
    
    # Calculate ROA = Net Income / Total Assets
    roa = combined['NetIncome'] / combined['TotalAssets']
    
    # Remove infinities and NaNs
    roa = roa.replace([np.inf, -np.inf], np.nan).dropna()
    
    if roa.empty:
        return None
    
    return roa.sort_index()


def build_roa_signals(
    fundamentals: Dict,
    dates: pd.DatetimeIndex,
    data_loader=None,
) -> pd.DataFrame:
    """
    Build ROA signal panel with proper lag
    
    Returns:
        DataFrame (dates x tickers) with ROA scores
        Higher ROA = Better profitability
    """
    print("\n🔧 Building ROA signals...")
    print("  Formula: ROA = Net Income / Total Assets")
    print("  Higher ROA = Better profitability")
    
    fundamentals_parquet = data_loader.load_fundamentals_parquet() if data_loader is not None else None
    
    panel = {}
    count = 0
    
    for ticker, fund_data in fundamentals.items():
        xlsx_path = fund_data['path']
        roa_series = calculate_roa_for_ticker(
            xlsx_path,
            ticker,
            fundamentals_parquet,
        )
        
        if roa_series is not None:
            # Apply lag
            lagged = apply_lag(roa_series, dates)
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
            print(f"  Latest ROA - Mean: {latest.mean():.4f}, Std: {latest.std():.4f}")
            print(f"  Latest ROA - Min: {latest.min():.4f}, Max: {latest.max():.4f}")
            
            # Show top 5 profitable stocks (highest ROA)
            top_5 = latest.nlargest(5)
            print(f"  Top 5 profitable stocks (highest ROA): {', '.join(top_5.index.tolist())}")
    
    print(f"  ✅ ROA signals: {result.shape[0]} days × {result.shape[1]} tickers")
    return result
