"""
Asset Growth Effect Signal Construction

Companies that grow assets too quickly tend to underperform.
This captures the "empire building" phenomenon where management
over-expands, leading to poor capital allocation and subsequent underperformance.

Formula:
Asset Growth = (Total Assets_t - Total Assets_t-1) / Total Assets_t-1

Lower asset growth = Better signal score (conservative companies)
Higher asset growth = Worse signal score (over-expanding companies)
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

TOTAL_ASSETS_KEYS = (
    "Toplam Varlıklar",
    "Toplam Aktifler",
)


def calculate_asset_growth_for_ticker(
    xlsx_path: Path | None,
    ticker: str,
    fundamentals_parquet: pd.DataFrame | None = None,
) -> pd.Series | None:
    """Calculate asset growth for a single ticker"""
    
    # Load data
    if fundamentals_parquet is not None:
        bs = get_consolidated_sheet(fundamentals_parquet, ticker, BALANCE_SHEET)
        if bs.empty:
            return None
        ta_row = pick_row_from_sheet(bs, TOTAL_ASSETS_KEYS)
    else:
        if xlsx_path is None:
            return None
        try:
            bs = pd.read_excel(xlsx_path, sheet_name=BALANCE_SHEET)
        except Exception:
            return None
        ta_row = pick_row(bs, TOTAL_ASSETS_KEYS)
    
    if ta_row is None:
        return None
    
    # Convert to series
    ta = coerce_quarter_cols(ta_row)
    
    if ta.empty or len(ta) < 2:
        return None
    
    # Calculate YoY asset growth
    # pct_change() gives us (current - previous) / previous
    asset_growth = ta.pct_change()
    
    # Remove infinities and NaNs
    asset_growth = asset_growth.replace([np.inf, -np.inf], np.nan).dropna()
    
    if asset_growth.empty:
        return None
    
    # Invert: Lower asset growth = Higher score (we want conservative companies)
    # Multiply by -1 so that low growth gets high scores
    quality_score = -asset_growth
    
    return quality_score.sort_index()


def build_asset_growth_signals(
    fundamentals: Dict,
    dates: pd.DatetimeIndex,
    data_loader=None,
) -> pd.DataFrame:
    """
    Build asset growth signal panel with proper lag
    
    Returns:
        DataFrame (dates x tickers) with asset growth quality scores
        Higher score = Lower asset growth = Better quality (conservative management)
    """
    print("\n🔧 Building asset growth signals...")
    print("  Formula: Asset Growth = ΔTotal Assets / Total Assets")
    print("  Lower asset growth = Conservative management = Higher quality")
    
    fundamentals_parquet = data_loader.load_fundamentals_parquet() if data_loader is not None else None
    
    panel = {}
    count = 0
    
    for ticker, fund_data in fundamentals.items():
        xlsx_path = fund_data['path']
        growth_series = calculate_asset_growth_for_ticker(
            xlsx_path,
            ticker,
            fundamentals_parquet,
        )
        
        if growth_series is not None:
            # Apply lag
            lagged = apply_lag(growth_series, dates)
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
            
            # Show top 5 quality stocks (highest scores = lowest asset growth)
            top_5 = latest.nlargest(5)
            print(f"  Top 5 conservative stocks (lowest asset growth): {', '.join(top_5.index.tolist())}")
    
    print(f"  ✅ Asset growth signals: {result.shape[0]} days × {result.shape[1]} tickers")
    return result
