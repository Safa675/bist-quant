"""
Quality Value Signal Construction

Combines value metrics with profitability to identify undervalued, profitable companies.

Logic:
- Value metrics identify cheap stocks (low P/B, P/E, EV/EBITDA)
- Profitability filters for fundamentally sound companies (ROE, ROA, Operating Margin)
- Combined signal avoids "value traps" - cheap stocks that are cheap for a reason

This addresses a key weakness of pure value strategies: they can select
unprofitable companies that are cheap because they're failing.

Scoring:
- 50% weight on value (P/B, P/E composite rank, inverted so low is good)
- 50% weight on profitability (ROE/ROA composite rank)
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

# Value metrics
NET_INCOME_KEYS = (
    "Dönem Karı (Zararı)",
    "Net Dönem Karı (Zararı)",
    "    Dönem Karı (Zararı)",
)
BOOK_VALUE_KEYS = (
    "Toplam Özkaynaklar",
    "    Toplam Özkaynaklar",
    "Ana Ortaklığa Ait Özkaynaklar",
)

# Profitability metrics
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
)


def calculate_value_for_ticker(
    xlsx_path: Path | None,
    ticker: str,
    close_df: pd.DataFrame,
    data_loader,
    fundamentals_parquet: pd.DataFrame | None = None,
) -> pd.Series | None:
    """Calculate value metrics for a single ticker"""
    if fundamentals_parquet is not None:
        inc = get_consolidated_sheet(fundamentals_parquet, ticker, INCOME_SHEET)
        bs = get_consolidated_sheet(fundamentals_parquet, ticker, BALANCE_SHEET)
        if inc.empty and bs.empty:
            return None
        ni_row = pick_row_from_sheet(inc, NET_INCOME_KEYS)
        bv_row = pick_row_from_sheet(bs, BOOK_VALUE_KEYS)
    else:
        if xlsx_path is None:
            return None
        try:
            inc = pd.read_excel(xlsx_path, sheet_name=INCOME_SHEET)
            bs = pd.read_excel(xlsx_path, sheet_name=BALANCE_SHEET)
        except Exception:
            return None
        ni_row = pick_row(inc, NET_INCOME_KEYS)
        bv_row = pick_row(bs, BOOK_VALUE_KEYS)
    
    if ni_row is None or bv_row is None:
        return None
    
    ni = coerce_quarter_cols(ni_row)
    bv = coerce_quarter_cols(bv_row)
    
    if ni.empty or bv.empty:
        return None
    
    # Get shares outstanding
    shares = data_loader.load_shares_outstanding(ticker) if data_loader else None
    if shares is None or shares.empty:
        return None
    
    # Align dates
    combined = pd.concat([ni, bv], axis=1, join="inner")
    combined.columns = ["NetIncome", "BookValue"]
    combined = combined.dropna()
    
    if combined.empty:
        return None
    
    # Get prices aligned to fundamental dates
    prices_aligned = close_df[ticker].reindex(combined.index, method='ffill')
    shares_aligned = shares.reindex(combined.index, method='ffill')
    
    # Calculate market cap
    market_cap = prices_aligned * shares_aligned
    
    # Calculate P/B and P/E
    pb = market_cap / combined["BookValue"]
    
    # For P/E, only use positive earnings (negative earnings = loss-making company)
    # Filter out negative earnings to avoid misleading "cheap" P/E ratios
    pe = market_cap / combined["NetIncome"]
    pe_mask = combined["NetIncome"] > 0
    pe = pe.where(pe_mask, np.nan)
    
    # Composite value score (lower is better, so we'll invert it)
    # If P/E is NaN (negative earnings), use only P/B
    # If both are valid, equal weight P/B and P/E
    value_score = pd.Series(index=pb.index, dtype=float)
    
    # Where both P/B and P/E are valid
    both_valid = pb.notna() & pe.notna()
    value_score[both_valid] = 0.5 * pb[both_valid] + 0.5 * pe[both_valid]
    
    # Where only P/B is valid (negative earnings)
    only_pb_valid = pb.notna() & pe.isna()
    value_score[only_pb_valid] = pb[only_pb_valid]
    
    # Clean up inf values and NaN
    value_score = value_score.replace([np.inf, -np.inf], np.nan).dropna()
    
    if value_score.empty:
        return None
    
    return value_score.sort_index()


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
        bv_row = pick_row_from_sheet(bs, BOOK_VALUE_KEYS)
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
        bv_row = pick_row(bs, BOOK_VALUE_KEYS)
    
    if op_row is None or gp_row is None or ta_row is None or bv_row is None:
        return None
    
    op = coerce_quarter_cols(op_row)
    gp = coerce_quarter_cols(gp_row)
    ta = coerce_quarter_cols(ta_row)
    bv = coerce_quarter_cols(bv_row)
    
    if op.empty or gp.empty or ta.empty or bv.empty:
        return None
    
    combined = pd.concat([op, gp, ta, bv], axis=1, join="inner")
    combined.columns = ["OperatingIncome", "GrossProfit", "TotalAssets", "BookValue"]
    combined = combined.dropna()
    
    if combined.empty:
        return None
    
    # Calculate ROA, ROE, Operating Margin
    roa = combined["OperatingIncome"] / combined["TotalAssets"]
    roe = combined["OperatingIncome"] / combined["BookValue"]
    op_margin = combined["OperatingIncome"] / combined["GrossProfit"]
    
    # Composite: 40% ROA, 40% ROE, 20% Operating Margin
    profitability = 0.4 * roa + 0.4 * roe + 0.2 * op_margin
    profitability = profitability.replace([np.inf, -np.inf], np.nan).dropna()
    
    if profitability.empty:
        return None
    
    return profitability.sort_index()


def build_quality_value_signals(
    close_df: pd.DataFrame,
    fundamentals: dict,
    dates: pd.DatetimeIndex,
    data_loader=None,
) -> pd.DataFrame:
    """
    Build quality value signal panel
    
    Combines value metrics (P/B, P/E) with profitability metrics to identify
    undervalued, profitable companies and avoid value traps.
    
    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        fundamentals: Dict of fundamental data
        dates: DatetimeIndex to align signals to
        data_loader: DataLoader instance
    
    Returns:
        DataFrame (dates x tickers) with quality value scores (0-100)
    """
    print("\n🔧 Building quality value signals...")
    print("  Value metrics: P/B (50%) + P/E (50%)")
    print("  Profitability: ROA (40%) + ROE (40%) + Operating Margin (20%)")
    print("  Weighting: Value (50%) + Profitability (50%)")
    
    fundamentals_parquet = data_loader.load_fundamentals_parquet() if data_loader is not None else None
    
    # 1. Calculate value metrics
    print("  Calculating value metrics...")
    value_panel = {}
    count = 0
    
    for ticker, fund_data in fundamentals.items():
        xlsx_path = fund_data['path']
        value_series = calculate_value_for_ticker(
            xlsx_path,
            ticker,
            close_df,
            data_loader,
            fundamentals_parquet,
        )
        
        if value_series is not None:
            lagged = apply_lag(value_series, dates)
            if not lagged.empty:
                value_panel[ticker] = lagged
                count += 1
                if count % 50 == 0:
                    print(f"    Processed {count} tickers...")
    
    value_df = pd.DataFrame(value_panel, index=dates)
    print(f"  ✅ Value metrics: {value_df.shape[0]} days × {value_df.shape[1]} tickers")
    
    # Rank value (INVERTED - lower values are better, so we invert the rank)
    value_rank = (1 - value_df.rank(axis=1, pct=True)) * 100
    
    # 2. Calculate profitability
    print("  Calculating profitability...")
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
            lagged = apply_lag(prof_series, dates)
            if not lagged.empty:
                profitability_panel[ticker] = lagged
                count += 1
                if count % 50 == 0:
                    print(f"    Processed {count} tickers...")
    
    profitability_df = pd.DataFrame(profitability_panel, index=dates)
    print(f"  ✅ Profitability: {profitability_df.shape[0]} days × {profitability_df.shape[1]} tickers")
    
    # Rank profitability (higher is better)
    profitability_rank = profitability_df.rank(axis=1, pct=True) * 100
    
    # 3. Combine value and profitability
    print("  Combining signals...")
    
    # Align columns
    common_tickers = value_rank.columns.intersection(profitability_rank.columns)
    value_rank = value_rank[common_tickers]
    profitability_rank = profitability_rank[common_tickers]
    
    # Combined score: 50% value, 50% profitability
    combined_score = 0.5 * value_rank + 0.5 * profitability_rank
    
    # Filter: Both value AND profitability must be above median (50)
    # This ensures we only select stocks that are cheap AND profitable
    quality_filter = (value_rank > 50) & (profitability_rank > 50)
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
        
        # Show top quality value stocks
        latest = result.iloc[-1]
        top_5 = latest.nlargest(5)
        if len(top_5[top_5 > 0]) > 0:
            print(f"  Top 5 quality value stocks: {', '.join(top_5[top_5 > 0].index.tolist())}")
    
    print(f"  ✅ Quality value signals: {result.shape[0]} days × {result.shape[1]} tickers")
    
    return result
