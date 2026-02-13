"""
Small Cap Momentum Signal Construction

Combines small market cap with momentum to identify early-stage breakouts.

Logic:
- Size filter identifies small cap stocks (bottom 30% by market cap)
- Momentum identifies stocks with positive price trends (3-month and 6-month returns)
- Combined signal captures "early discovery" plays - small caps breaking out

Scoring:
- 30% weight on size (inverted - smaller is better)
- 40% weight on 6-month momentum
- 30% weight on 3-month momentum (for recent acceleration)
- Size must be in bottom 50% (small/mid caps) to get a non-zero score
- Momentum must be positive to get a non-zero score
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def calculate_market_cap(close_df: pd.DataFrame, data_loader=None) -> pd.DataFrame:
    """Calculate market cap = close * shares outstanding using DataLoader caches."""
    print("  📊 Loading shares outstanding...")

    if data_loader is None:
        print("  ⚠️  No data_loader provided")
        return pd.DataFrame()

    dates = close_df.index
    tickers = close_df.columns

    # Fast path: prebuilt consolidated shares panel (CSV -> DataLoader cache).
    if hasattr(data_loader, "load_shares_outstanding_panel"):
        try:
            shares_panel = data_loader.load_shares_outstanding_panel()
            if shares_panel is not None and not shares_panel.empty:
                shares = shares_panel.copy()
                shares.index = pd.to_datetime(shares.index, errors="coerce")
                shares = shares.sort_index()
                shares.columns = pd.Index([str(c).split(".")[0].upper() for c in shares.columns])
                shares = shares.reindex(index=dates, columns=tickers).ffill()
                market_cap = close_df.reindex(index=dates, columns=tickers).astype(float) * shares.astype(float)
                non_na = int(market_cap.notna().sum().sum())
                if non_na > 0:
                    print(f"  ✅ Loaded shares panel for {shares.shape[1]} tickers")
                    return market_cap
        except Exception as e:
            print(f"  ⚠️  Failed consolidated shares panel path: {e}")

    # Slow fallback: ticker-by-ticker shares series.
    if not hasattr(data_loader, "load_shares_outstanding"):
        print("  ⚠️  data_loader has no shares API")
        return pd.DataFrame()

    market_cap = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)
    success = 0

    for idx, ticker in enumerate(tickers, start=1):
        try:
            shares = data_loader.load_shares_outstanding(ticker)
            if shares is None or shares.empty:
                continue
            shares = shares.sort_index()
            shares = shares[~shares.index.duplicated(keep="last")]
            shares = shares.reindex(dates, method="ffill")
            market_cap[ticker] = close_df[ticker].reindex(dates) * shares
            success += 1
            if idx % 100 == 0:
                print(f"  Shares fallback progress: {idx}/{len(tickers)} ({success} tickers)")
        except Exception:
            continue

    if int(market_cap.notna().sum().sum()) == 0:
        return pd.DataFrame()

    print(f"  ✅ Loaded shares fallback for {success} tickers")
    return market_cap


def build_small_cap_momentum_signals(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
) -> pd.DataFrame:
    """
    Build small cap momentum signal panel
    
    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        dates: DatetimeIndex to align signals to
        data_loader: DataLoader instance for shares-outstanding data
    
    Returns:
        DataFrame (dates x tickers) with small cap momentum scores (0-100)
    """
    print("\n🔧 Building small cap momentum signals...")
    print("  Size filter: Bottom 50% by market cap (small/mid caps)")
    print("  Momentum: 6-month (40%) + 3-month (30%)")
    print("  Weighting: Size (30%) + 6M Momentum (40%) + 3M Momentum (30%)")
    
    # 1. Calculate market cap
    print("  Calculating market cap...")
    market_cap_df = calculate_market_cap(close_df, data_loader)
    
    if market_cap_df.empty:
        print("  ⚠️  No market cap data available")
        print("  Returning zero scores for all stocks")
        return pd.DataFrame(0.0, index=dates, columns=close_df.columns)
    
    print(f"  ✅ Market cap: {market_cap_df.shape[0]} days × {market_cap_df.shape[1]} tickers")
    
    # Rank market cap (INVERTED - smaller is better)
    size_rank = (1 - market_cap_df.rank(axis=1, pct=True)) * 100
    
    # 2. Calculate momentum
    print("  Calculating momentum...")
    
    # 6-month momentum
    momentum_6m = close_df.pct_change(126)
    momentum_6m_rank = momentum_6m.rank(axis=1, pct=True) * 100
    
    # 3-month momentum
    momentum_3m = close_df.pct_change(63)
    momentum_3m_rank = momentum_3m.rank(axis=1, pct=True) * 100
    
    # 3. Combine
    print("  Combining signals...")
    
    # Align columns
    common_tickers = size_rank.columns.intersection(momentum_6m_rank.columns).intersection(momentum_3m_rank.columns)
    size_rank = size_rank[common_tickers]
    momentum_6m_rank = momentum_6m_rank[common_tickers]
    momentum_3m_rank = momentum_3m_rank[common_tickers]
    
    # Combined score
    combined_score = (
        0.30 * size_rank +
        0.40 * momentum_6m_rank +
        0.30 * momentum_3m_rank
    )
    
    # Filters
    size_filter = size_rank > 50
    momentum_filter = (momentum_6m_rank > 50) & (momentum_3m_rank > 50)
    quality_filter = size_filter & momentum_filter
    combined_score = combined_score.where(quality_filter, 0)
    
    # Reindex to all tickers
    result = pd.DataFrame(0.0, index=dates, columns=close_df.columns)
    for ticker in common_tickers:
        if ticker in result.columns:
            result[ticker] = combined_score[ticker]
    
    result = result.fillna(0.0)
    
    # Summary
    valid_scores = result[result > 0].stack()
    if len(valid_scores) > 0:
        print(f"  Valid scores - Mean: {valid_scores.mean():.1f}, Std: {valid_scores.std():.1f}")
        print(f"  Valid scores - Min: {valid_scores.min():.1f}, Max: {valid_scores.max():.1f}")
        
        latest = result.iloc[-1]
        top_5 = latest.nlargest(5)
        if len(top_5[top_5 > 0]) > 0:
            print(f"  Top 5 small cap momentum stocks: {', '.join(top_5[top_5 > 0].index.tolist())}")
            
            if not market_cap_df.empty:
                latest_mcap = market_cap_df.iloc[-1]
                for ticker in top_5[top_5 > 0].index:
                    if ticker in latest_mcap.index:
                        mcap_b = latest_mcap[ticker] / 1e9
                        print(f"    {ticker}: {mcap_b:.2f}B TRY market cap")
    
    print(f"  ✅ Small cap momentum signals: {result.shape[0]} days × {result.shape[1]} tickers")
    
    return result
