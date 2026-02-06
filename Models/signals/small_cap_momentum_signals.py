"""
Small Cap Momentum Signal Construction

Combines small market cap with momentum to identify early-stage breakouts.

Logic:
- Size filter identifies small cap stocks (bottom 30% by market cap)
- Momentum identifies stocks with positive price trends (3-month and 6-month returns)
- Combined signal captures "early discovery" plays - small caps breaking out

This addresses a key opportunity: small caps that start trending often have
more upside potential than large caps, as they're being discovered by the market.

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


def calculate_market_cap(
    close_df: pd.DataFrame,
    data_loader,
) -> pd.DataFrame:
    """Calculate market cap for all tickers"""
    market_cap_panel = {}
    
    for ticker in close_df.columns:
        shares = data_loader.load_shares_outstanding(ticker) if data_loader else None
        if shares is None or shares.empty:
            continue
        
        # Align shares to price dates
        shares_aligned = shares.reindex(close_df.index, method='ffill')
        
        # Market cap = price * shares
        market_cap = close_df[ticker] * shares_aligned
        market_cap_panel[ticker] = market_cap
    
    return pd.DataFrame(market_cap_panel, index=close_df.index)


def build_small_cap_momentum_signals(
    close_df: pd.DataFrame,
    fundamentals: dict,
    dates: pd.DatetimeIndex,
    data_loader=None,
) -> pd.DataFrame:
    """
    Build small cap momentum signal panel
    
    Combines small market cap with momentum to identify early-stage breakouts
    and "discovery" plays.
    
    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        fundamentals: Dict of fundamental data (not used but kept for interface consistency)
        dates: DatetimeIndex to align signals to
        data_loader: DataLoader instance
    
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
    
    # 3-month momentum (for recent acceleration)
    momentum_3m = close_df.pct_change(63)
    momentum_3m_rank = momentum_3m.rank(axis=1, pct=True) * 100
    
    # 3. Combine size and momentum
    print("  Combining signals...")
    
    # Align columns
    common_tickers = size_rank.columns.intersection(momentum_6m_rank.columns).intersection(momentum_3m_rank.columns)
    size_rank = size_rank[common_tickers]
    momentum_6m_rank = momentum_6m_rank[common_tickers]
    momentum_3m_rank = momentum_3m_rank[common_tickers]
    
    # Combined score: 30% size, 40% 6M momentum, 30% 3M momentum
    combined_score = (
        0.30 * size_rank +
        0.40 * momentum_6m_rank +
        0.30 * momentum_3m_rank
    )
    
    # Filter 1: Size must be in bottom 50% (small/mid caps only)
    size_filter = size_rank > 50
    
    # Filter 2: Both momentum measures must be positive (above 50th percentile)
    momentum_filter = (momentum_6m_rank > 50) & (momentum_3m_rank > 50)
    
    # Combined filter
    quality_filter = size_filter & momentum_filter
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
        
        # Show top small cap momentum stocks
        latest = result.iloc[-1]
        top_5 = latest.nlargest(5)
        if len(top_5[top_5 > 0]) > 0:
            print(f"  Top 5 small cap momentum stocks: {', '.join(top_5[top_5 > 0].index.tolist())}")
            
            # Show their market caps for context
            if not market_cap_df.empty:
                latest_mcap = market_cap_df.iloc[-1]
                for ticker in top_5[top_5 > 0].index:
                    if ticker in latest_mcap.index:
                        mcap_b = latest_mcap[ticker] / 1e9  # Convert to billions
                        print(f"    {ticker}: {mcap_b:.2f}B TRY market cap")
    
    print(f"  ✅ Small cap momentum signals: {result.shape[0]} days × {result.shape[1]} tickers")
    
    return result
