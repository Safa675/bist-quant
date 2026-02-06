"""
Quality Momentum Signal Construction

Combines momentum with profitability to identify high-quality trending stocks.

Logic:
- Momentum identifies stocks with positive price trends (6-month return)
- Profitability filters for fundamentally sound companies (ROE, ROA, Operating Margin)
- Combined signal avoids "junk rallies" - only profitable companies with momentum

This addresses a key weakness of pure momentum strategies: they can chase
unprofitable companies that are rallying on hype rather than fundamentals.

Scoring:
- 60% weight on momentum (6-month return rank)
- 40% weight on profitability (ROE/ROA composite rank)
- Both components must be above median (50th percentile) to get a non-zero score
"""

import pandas as pd
import numpy as np
from pathlib import Path
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

OPERATING_INCOME_KEYS = (
    "Faaliyet Karı (Zararı)",
    "Finansman Geliri (Gideri) Öncesi Faaliyet Karı (Zararı)",
)
GROSS_PROFIT_KEYS = (
    "Brüt Kar (Zarar)",
    "Ticari Faaliyetlerden Brüt Kar (Zarar)",
)
TOTAL_ASSETS_KEYS = (
    "Toplam Varlıklar",
    "    Toplam Varlıklar",
    "Toplam Aktifler",
)
TOTAL_EQUITY_KEYS = (
    "Toplam Özkaynaklar",
    "    Toplam Özkaynaklar",
    "Ana Ortaklığa Ait Özkaynaklar",
)


def calculate_profitability_for_ticker(
    xlsx_path: Path | None,
    ticker: str,
    fundamentals_parquet: pd.DataFrame | None = None,
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
        eq_row = pick_row_from_sheet(bs, TOTAL_EQUITY_KEYS)
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
        eq_row = pick_row(bs, TOTAL_EQUITY_KEYS)
    
    if op_row is None or gp_row is None or ta_row is None or eq_row is None:
        return None

    op = coerce_quarter_cols(op_row)
    gp = coerce_quarter_cols(gp_row)
    ta = coerce_quarter_cols(ta_row)
    eq = coerce_quarter_cols(eq_row)
    
    if op.empty or gp.empty or ta.empty or eq.empty:
        return None

    combined = pd.concat([op, gp, ta, eq], axis=1, join="inner")
    combined.columns = ["OperatingIncome", "GrossProfit", "TotalAssets", "TotalEquity"]
    combined = combined.dropna()
    if combined.empty:
        return None

    # Calculate composite profitability score
    # ROA = Operating Income / Total Assets
    # ROE = Operating Income / Total Equity  
    # Operating Margin = Operating Income / Gross Profit
    roa = combined["OperatingIncome"] / combined["TotalAssets"]
    roe = combined["OperatingIncome"] / combined["TotalEquity"]
    op_margin = combined["OperatingIncome"] / combined["GrossProfit"]
    
    # Composite: 40% ROA, 40% ROE, 20% Operating Margin
    profitability = 0.4 * roa + 0.4 * roe + 0.2 * op_margin
    profitability = profitability.replace([np.inf, -np.inf], np.nan).dropna()
    
    if profitability.empty:
        return None
    
    return profitability.sort_index()


def build_quality_momentum_signals(
    close_df: pd.DataFrame,
    fundamentals: dict,
    dates: pd.DatetimeIndex,
    data_loader=None,
) -> pd.DataFrame:
    """
    Build quality momentum signal panel
    
    Combines 6-month momentum with profitability metrics to identify
    high-quality trending stocks.
    
    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        fundamentals: Dict of fundamental data
        dates: DatetimeIndex to align signals to
        data_loader: DataLoader instance
    
    Returns:
        DataFrame (dates x tickers) with quality momentum scores (0-100)
    """
    print("\n🔧 Building quality momentum signals...")
    print("  Momentum lookback: 126 days (6 months)")
    print("  Profitability: ROA (40%) + ROE (40%) + Operating Margin (20%)")
    print("  Weighting: Momentum (60%) + Profitability (40%)")
    
    # 1. Calculate 6-month momentum
    print("  Calculating momentum...")
    momentum_6m = close_df.pct_change(126)
    
    # Rank momentum (0-100 scale)
    momentum_rank = momentum_6m.rank(axis=1, pct=True) * 100
    
    # 2. Calculate profitability for all tickers
    print("  Calculating profitability...")
    fundamentals_parquet = data_loader.load_fundamentals_parquet() if data_loader is not None else None
    
    profitability_panel = {}
    count = 0
    
    for ticker, fund_data in fundamentals.items():
        xlsx_path = fund_data['path']
        prof_series = calculate_profitability_for_ticker(
            xlsx_path,
            ticker,
            fundamentals_parquet,
        )
        
        if prof_series is not None:
            # Apply lag
            lagged = apply_lag(prof_series, dates)
            if not lagged.empty:
                profitability_panel[ticker] = lagged
                count += 1
                if count % 50 == 0:
                    print(f"    Processed {count} tickers...")
    
    profitability_df = pd.DataFrame(profitability_panel, index=dates)
    print(f"  ✅ Profitability: {profitability_df.shape[0]} days × {profitability_df.shape[1]} tickers")
    
    # Rank profitability (0-100 scale)
    profitability_rank = profitability_df.rank(axis=1, pct=True) * 100
    
    # 3. Combine momentum and profitability
    print("  Combining signals...")
    
    # Align columns
    common_tickers = momentum_rank.columns.intersection(profitability_rank.columns)
    momentum_rank = momentum_rank[common_tickers]
    profitability_rank = profitability_rank[common_tickers]
    
    # Combined score: 60% momentum, 40% profitability
    combined_score = 0.6 * momentum_rank + 0.4 * profitability_rank
    
    # Filter: Both momentum AND profitability must be above median (50)
    # This ensures we only select stocks that are good on BOTH dimensions
    quality_filter = (momentum_rank > 50) & (profitability_rank > 50)
    combined_score = combined_score.where(quality_filter, 0)
    
    # Reindex to all tickers in close_df
    result = pd.DataFrame(0.0, index=dates, columns=close_df.columns)
    for ticker in common_tickers:
        if ticker in result.columns:
            result[ticker] = combined_score[ticker]
    
    # Fill NaN with 0
    result = result.fillna(0.0)
    
    # Summary stats
    valid_scores = result[result > 0].stack()
    if len(valid_scores) > 0:
        print(f"  Valid scores - Mean: {valid_scores.mean():.1f}, Std: {valid_scores.std():.1f}")
        print(f"  Valid scores - Min: {valid_scores.min():.1f}, Max: {valid_scores.max():.1f}")
        
        # Show top quality momentum stocks
        latest = result.iloc[-1]
        top_5 = latest.nlargest(5)
        if len(top_5[top_5 > 0]) > 0:
            print(f"  Top 5 quality momentum stocks: {', '.join(top_5[top_5 > 0].index.tolist())}")
    
    print(f"  ✅ Quality momentum signals: {result.shape[0]} days × {result.shape[1]} tickers")
    
    return result
