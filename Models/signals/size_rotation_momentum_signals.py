"""
Size Rotation Momentum Signal Construction

Combines size rotation regime detection with pure momentum.

Logic:
- Detect size regime: Are small caps or large caps outperforming?
- Apply 6-month momentum within the favored size bucket
- When small caps leading: Pick highest momentum small caps
- When large caps leading: Pick highest momentum large caps

This is a more aggressive version of size_rotation that focuses purely
on momentum within the winning size segment.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import size rotation helpers
from signals.size_rotation_signals import (
    calculate_size_regime,
    XU100_TICKERS,
    RELATIVE_PERF_LOOKBACK,
    SWITCH_THRESHOLD,
)


# ============================================================================
# PARAMETERS
# ============================================================================

MOMENTUM_LOOKBACK = 126  # 6 months
MOMENTUM_SKIP = 21  # Skip most recent month (avoid reversal)


# ============================================================================
# MOMENTUM CALCULATION
# ============================================================================

def calculate_momentum(
    close_df: pd.DataFrame,
    lookback: int = MOMENTUM_LOOKBACK,
    skip: int = MOMENTUM_SKIP,
) -> pd.DataFrame:
    """
    Calculate 12-1 style momentum (skip recent month).

    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        lookback: Total lookback period
        skip: Days to skip (most recent)

    Returns:
        DataFrame of momentum scores
    """
    # Shift to skip recent days, then calculate return over lookback
    shifted = close_df.shift(skip)
    momentum = shifted.pct_change(lookback - skip)
    return momentum


# ============================================================================
# SIZE ROTATION MOMENTUM SIGNAL
# ============================================================================

def build_size_rotation_momentum_signals(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
) -> pd.DataFrame:
    """
    Build size rotation momentum signal.

    Combines size regime detection with momentum:
    - In small_cap regime: Only consider small cap momentum
    - In large_cap regime: Only consider large cap momentum
    - In neutral: Consider all stocks

    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        dates: DatetimeIndex to align signals to
        data_loader: DataLoader instance (optional)

    Returns:
        DataFrame (dates x tickers) with rotation momentum scores
    """
    print("\n🔧 Building size rotation momentum signals...")
    print(f"  Size regime lookback: {RELATIVE_PERF_LOOKBACK} days")
    print(f"  Momentum lookback: {MOMENTUM_LOOKBACK} days (skip {MOMENTUM_SKIP})")
    print(f"  Switch threshold: ±{SWITCH_THRESHOLD}")

    # Calculate size regime
    print("  Calculating size regime...")
    size_regime = calculate_size_regime(close_df)

    # Show current regime
    latest_regime = size_regime['regime'].iloc[-1] if not size_regime.empty else 'unknown'
    latest_z = size_regime['z_score'].iloc[-1] if not size_regime.empty else 0
    print(f"  Current regime: {latest_regime.upper()} (z-score: {latest_z:.2f})")

    # Calculate momentum
    print("  Calculating momentum...")
    momentum = calculate_momentum(close_df)

    # Classify tickers
    large_cap_tickers = set(t for t in XU100_TICKERS if t in close_df.columns)
    small_cap_tickers = set(t for t in close_df.columns if t not in large_cap_tickers)

    print(f"  Large caps: {len(large_cap_tickers)}, Small caps: {len(small_cap_tickers)}")

    # Build rotation-aware momentum scores
    print("  Building rotation-aware scores...")
    scores = pd.DataFrame(index=close_df.index, columns=close_df.columns, dtype=float)

    for date in close_df.index:
        if date not in size_regime.index or date not in momentum.index:
            continue

        regime = size_regime.loc[date, 'regime']
        mom_today = momentum.loc[date]

        if pd.isna(regime):
            regime = 'neutral'

        if regime == 'small_cap':
            # Only score small caps, zero out large caps
            small_mom = mom_today[mom_today.index.isin(small_cap_tickers)].dropna()
            if len(small_mom) > 0:
                # Rank small caps by momentum (0-100)
                ranks = small_mom.rank(pct=True) * 100
                for ticker, rank in ranks.items():
                    scores.loc[date, ticker] = rank
            # Large caps get 0
            for ticker in large_cap_tickers:
                scores.loc[date, ticker] = 0

        elif regime == 'large_cap':
            # Only score large caps, zero out small caps
            large_mom = mom_today[mom_today.index.isin(large_cap_tickers)].dropna()
            if len(large_mom) > 0:
                ranks = large_mom.rank(pct=True) * 100
                for ticker, rank in ranks.items():
                    scores.loc[date, ticker] = rank
            # Small caps get 0
            for ticker in small_cap_tickers:
                scores.loc[date, ticker] = 0

        else:  # neutral
            # Score all stocks by momentum
            all_mom = mom_today.dropna()
            if len(all_mom) > 0:
                ranks = all_mom.rank(pct=True) * 100
                for ticker, rank in ranks.items():
                    scores.loc[date, ticker] = rank

    # Reindex to requested dates
    result = scores.reindex(dates)
    result = result.fillna(0)

    # Summary
    latest = result.iloc[-1]
    nonzero = latest[latest > 0]
    if len(nonzero) > 0:
        print(f"  Latest non-zero scores: {len(nonzero)} tickers")
        top_5 = nonzero.nlargest(5)
        print(f"  Top 5: {', '.join(top_5.index.tolist())}")

    print(f"  ✅ Size rotation momentum signals: {result.shape[0]} days × {result.shape[1]} tickers")

    return result
