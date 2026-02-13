"""
Investment Signal Construction

Calculates composite investment scores based on 3 DIRECT metrics (higher = better):
1. R&D Intensity = R&D / Sales (higher = more innovative)
2. Net Payout Yield = Dividends / Market Cap (higher = better)
3. Investment Efficiency = ΔRevenue / ΔTotal Assets (higher = better ROI)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.utils import (
    pick_row,
    coerce_quarter_cols,
    sum_ttm,
    get_consolidated_sheet,
    pick_row_from_sheet,
    apply_lag,
)


# Fundamental data keys
INCOME_SHEET = "Gelir Tablosu (Çeyreklik)"
BALANCE_SHEET = "Bilanço"
CASH_FLOW_SHEET = "Nakit Akış (Çeyreklik)"

REVENUE_KEYS = (
    "Satış Gelirleri",
    "Toplam Hasılat",
    "Hasılat",
    "Net Satışlar",
)
RD_KEYS = (
    "Araştırma ve Geliştirme Giderleri (-)",
    "Araştırma ve Geliştirme Giderleri",
)
DIVIDENDS_PAID_KEYS = (
    "Ödenen Temettüler",
)
TOTAL_ASSETS_KEYS = (
    "Toplam Varlıklar",
    "Toplam Aktifler",
)


def calculate_investment_metrics_for_ticker(
    xlsx_path: Path | None,
    ticker: str,
    data_loader,
    fundamentals_parquet: pd.DataFrame | None = None,
) -> Dict:
    """Calculate investment metrics for a single ticker"""
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
    
    rev_ttm = sum_ttm(rev)
    rd_ttm = sum_ttm(rd)
    div_ttm = sum_ttm(div)
    
    shares = data_loader.load_shares_outstanding(ticker)
    
    return {
        'revenue_ttm': rev_ttm,
        'rd_ttm': rd_ttm,
        'dividends_ttm': div_ttm,
        'total_assets': assets,
        'shares_outstanding': shares,
    }


def build_investment_signals(
    fundamentals: Dict,
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader,
) -> pd.DataFrame:
    """
    Build composite investment signal panel
    
    Returns:
        DataFrame (dates x tickers) with investment scores
    """
    print("\n🔧 Building investment signals...")
    
    panels = {
        'rd_intensity': {},
        'payout_yield': {},
        'inv_efficiency': {},
    }
    
    fundamentals_parquet = data_loader.load_fundamentals_parquet()

    count = 0
    for ticker, fund_data in fundamentals.items():
        if ticker not in close_df.columns:
            continue
        
        xlsx_path = fund_data['path']
        metrics = calculate_investment_metrics_for_ticker(
            xlsx_path,
            ticker,
            data_loader,
            fundamentals_parquet,
        )
        
        if not metrics:
            continue
        
        price_series = close_df[ticker].dropna()
        shares = metrics.get('shares_outstanding', pd.Series(dtype=float))
        
        # Calculate market cap - SKIP if no shares data (Bug #1 fix)
        if shares.empty:
            # Cannot calculate proper market cap without shares data
            # Skip this ticker to avoid using price as market cap
            continue
        
        # Remove duplicates before reindexing
        shares = shares[~shares.index.duplicated(keep='last')]
        shares_aligned = shares.reindex(price_series.index).ffill()
        market_cap = price_series * shares_aligned
        
        rev_ttm = metrics.get('revenue_ttm', pd.Series(dtype=float))
        rd_ttm = metrics.get('rd_ttm', pd.Series(dtype=float))
        div_ttm = metrics.get('dividends_ttm', pd.Series(dtype=float))
        assets = metrics.get('total_assets', pd.Series(dtype=float))
        
        # 1. R&D Intensity - Apply reporting lag
        if not rd_ttm.empty and not rev_ttm.empty:
            ratio = abs(rd_ttm) / rev_ttm.reindex(rd_ttm.index).ffill()
            ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
            if not ratio.empty:
                lagged_ratio = apply_lag(ratio, dates)
                if not lagged_ratio.empty:
                    panels['rd_intensity'][ticker] = lagged_ratio
        
        # 2. Payout Yield - Apply reporting lag
        if not div_ttm.empty:
            ratio = abs(div_ttm) / market_cap.reindex(div_ttm.index).ffill()
            ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
            if not ratio.empty:
                lagged_ratio = apply_lag(ratio, dates)
                if not lagged_ratio.empty:
                    panels['payout_yield'][ticker] = lagged_ratio
        
        # 3. Investment Efficiency (YoY) - Apply reporting lag
        # Bug #3 fix: Clip extreme values to prevent outliers
        if not rev_ttm.empty and not assets.empty:
            rev_yoy = rev_ttm.diff(periods=4)
            assets_yoy = assets.diff(periods=4)
            # Add floor to denominator to avoid extreme ratios
            assets_yoy_floor = assets_yoy.abs().clip(lower=assets.rolling(4).mean() * 0.01)
            ratio = rev_yoy / assets_yoy_floor.reindex(rev_yoy.index).ffill()
            ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
            # Clip to reasonable bounds to prevent extreme outliers
            ratio = ratio.clip(lower=-5, upper=5)
            if not ratio.empty:
                lagged_ratio = apply_lag(ratio, dates)
                if not lagged_ratio.empty:
                    panels['inv_efficiency'][ticker] = lagged_ratio
        
        count += 1
        if count % 50 == 0:
            print(f"  Processed {count} tickers...")
    
    # Cross-sectional z-score normalization (Bug #2 fix)
    # Normalize each ratio type before combining to prevent scale bias
    print("  Normalizing ratios (z-score per date)...")
    normalized_panels = {}
    for panel_name, panel_dict in panels.items():
        if panel_dict:
            df = pd.DataFrame(panel_dict, index=dates)
            # Z-score: (x - mean) / std, computed cross-sectionally per date
            row_mean = df.mean(axis=1)
            row_std = df.std(axis=1)
            # Avoid division by zero
            row_std = row_std.replace(0, np.nan)
            df_zscore = df.sub(row_mean, axis=0).div(row_std, axis=0)
            normalized_panels[panel_name] = df_zscore
    
    # Combine into composite score
    print("  Combining into composite investment score...")
    composite_panel = {}
    
    for ticker in close_df.columns:
        scores_list = []
        for panel_name, panel_df in normalized_panels.items():
            if ticker in panel_df.columns:
                scores_list.append(panel_df[ticker])
        
        if scores_list:
            # Average across all available normalized ratios
            composite = pd.concat(scores_list, axis=1).mean(axis=1)
            composite_panel[ticker] = composite
    
    result = pd.DataFrame(composite_panel, index=dates)
    print(f"  ✅ Investment signals: {result.shape[0]} days × {result.shape[1]} tickers")
    return result
