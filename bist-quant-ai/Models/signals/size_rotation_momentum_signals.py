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
    build_market_cap_panel,
    build_liquidity_panel,
    get_size_buckets_for_date,
    RELATIVE_PERF_LOOKBACK,
    SWITCH_THRESHOLD,
    SIZE_LIQUIDITY_QUANTILE,
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

    # Build dynamic size inputs
    print("  Building market-cap panel (SERMAYE × price)...")
    market_cap_df = build_market_cap_panel(close_df, close_df.index, data_loader)
    print("  Loading liquidity panel...")
    liquidity_df = build_liquidity_panel(close_df, close_df.index, data_loader)

    # Calculate size regime
    print("  Calculating size regime...")
    size_regime = calculate_size_regime(
        close_df,
        market_cap_df=market_cap_df,
        liquidity_df=liquidity_df,
        liquidity_quantile=SIZE_LIQUIDITY_QUANTILE,
    )

    # Show current regime
    latest_regime = size_regime['regime'].iloc[-1] if not size_regime.empty else 'unknown'
    latest_z = size_regime['z_score'].iloc[-1] if not size_regime.empty else 0
    print(f"  Current regime: {latest_regime.upper()} (z-score: {latest_z:.2f})")

    # Calculate momentum
    print("  Calculating momentum...")
    momentum = calculate_momentum(close_df)

    # Show latest bucket counts
    latest_date = close_df.index[-1]
    latest_mcap = market_cap_df.loc[latest_date] if latest_date in market_cap_df.index else pd.Series(dtype=float)
    latest_liq = liquidity_df.loc[latest_date] if latest_date in liquidity_df.index else pd.Series(dtype=float)
    liquid_latest, small_latest, large_latest = get_size_buckets_for_date(
        latest_mcap,
        latest_liq,
        liquidity_quantile=SIZE_LIQUIDITY_QUANTILE,
    )
    print(f"  Latest liquid universe: {len(liquid_latest)}")
    print(f"  Latest large caps (top 10%): {len(large_latest)}, small caps (bottom 10%): {len(small_latest)}")

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

        mcaps = market_cap_df.loc[date] if date in market_cap_df.index else pd.Series(dtype=float)
        liq = liquidity_df.loc[date] if date in liquidity_df.index else pd.Series(dtype=float)
        liquid_universe, small_caps, large_caps = get_size_buckets_for_date(
            mcaps,
            liq,
            liquidity_quantile=SIZE_LIQUIDITY_QUANTILE,
        )
        if not liquid_universe:
            continue

        if regime == 'small_cap':
            # Only score liquid small caps, zero out everything else
            small_mom = mom_today.reindex(list(small_caps)).dropna()
            if len(small_mom) > 0:
                # Rank small caps by momentum (0-100)
                ranks = small_mom.rank(pct=True) * 100
                for ticker, rank in ranks.items():
                    scores.loc[date, ticker] = rank

        elif regime == 'large_cap':
            # Only score liquid large caps, zero out everything else
            large_mom = mom_today.reindex(list(large_caps)).dropna()
            if len(large_mom) > 0:
                ranks = large_mom.rank(pct=True) * 100
                for ticker, rank in ranks.items():
                    scores.loc[date, ticker] = rank

        else:  # neutral
            # Score all liquid names by momentum
            all_mom = mom_today.reindex(list(liquid_universe)).dropna()
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
