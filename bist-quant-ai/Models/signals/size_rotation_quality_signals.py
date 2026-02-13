"""
Size Rotation Quality Signal Construction

Combines size rotation regime with quality momentum (momentum + profitability).

Logic:
- Detect size regime: Are small caps or large caps outperforming?
- Within the favored size bucket, apply quality momentum:
  - Momentum: 6-month price return
  - Profitability: ROA, ROE, Operating Margin composite
  - Both must be above median to qualify

This is the most sophisticated size rotation signal:
- Adapts to size regime (avoids fighting the tape)
- Uses momentum (follows trends)
- Requires profitability (avoids junk rallies)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import helpers
from signals.size_rotation_signals import (
    calculate_size_regime,
    build_market_cap_panel,
    build_liquidity_panel,
    get_size_buckets_for_date,
    RELATIVE_PERF_LOOKBACK,
    SWITCH_THRESHOLD,
    SIZE_LIQUIDITY_QUANTILE,
)
from signals.quality_momentum_signals import calculate_profitability_for_ticker
from common.utils import apply_lag


# ============================================================================
# PARAMETERS
# ============================================================================

MOMENTUM_LOOKBACK = 126  # 6 months
MOMENTUM_WEIGHT = 0.6
PROFITABILITY_WEIGHT = 0.4
MIN_PERCENTILE = 50  # Both momentum and profitability must be above median


# ============================================================================
# SIZE ROTATION QUALITY SIGNAL
# ============================================================================

def build_size_rotation_quality_signals(
    close_df: pd.DataFrame,
    fundamentals: dict,
    dates: pd.DatetimeIndex,
    data_loader=None,
) -> pd.DataFrame:
    """
    Build size rotation quality signal.

    Combines:
    1. Size regime detection (small vs large cap leadership)
    2. Momentum within the favored size segment
    3. Profitability filter (ROA, ROE, Operating Margin)

    Only stocks that pass BOTH momentum AND profitability filters get scored.

    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        fundamentals: Dict of fundamental data
        dates: DatetimeIndex to align signals to
        data_loader: DataLoader instance

    Returns:
        DataFrame (dates x tickers) with quality rotation scores
    """
    print("\n🔧 Building size rotation quality signals...")
    print(f"  Size regime lookback: {RELATIVE_PERF_LOOKBACK} days")
    print(f"  Momentum lookback: {MOMENTUM_LOOKBACK} days")
    print(f"  Weights: Momentum {MOMENTUM_WEIGHT*100:.0f}%, Profitability {PROFITABILITY_WEIGHT*100:.0f}%")
    print(f"  Min percentile filter: {MIN_PERCENTILE}")

    # 1. Build dynamic size inputs and calculate regime
    print("  Building market-cap panel (SERMAYE × price)...")
    market_cap_df = build_market_cap_panel(close_df, close_df.index, data_loader)
    print("  Loading liquidity panel...")
    liquidity_df = build_liquidity_panel(close_df, close_df.index, data_loader)

    # 2. Calculate size regime
    print("  Calculating size regime...")
    size_regime = calculate_size_regime(
        close_df,
        market_cap_df=market_cap_df,
        liquidity_df=liquidity_df,
        liquidity_quantile=SIZE_LIQUIDITY_QUANTILE,
    )

    latest_regime = size_regime['regime'].iloc[-1] if not size_regime.empty else 'unknown'
    latest_z = size_regime['z_score'].iloc[-1] if not size_regime.empty else 0
    print(f"  Current regime: {latest_regime.upper()} (z-score: {latest_z:.2f})")

    # 3. Calculate momentum
    print("  Calculating momentum...")
    momentum = close_df.pct_change(MOMENTUM_LOOKBACK)
    momentum_rank = momentum.rank(axis=1, pct=True) * 100

    # 4. Calculate profitability
    print("  Calculating profitability...")
    fundamentals_parquet = data_loader.load_fundamentals_parquet() if data_loader else None

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
                if count % 100 == 0:
                    print(f"    Processed {count} tickers...")

    profitability_df = pd.DataFrame(profitability_panel, index=dates)
    print(f"  ✅ Profitability: {profitability_df.shape[1]} tickers")

    # Rank profitability
    profitability_rank = profitability_df.rank(axis=1, pct=True) * 100

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

    # 5. Build rotation-aware quality scores
    print("  Building rotation-aware quality scores...")
    scores = pd.DataFrame(0.0, index=dates, columns=close_df.columns)

    for date in dates:
        if date not in size_regime.index:
            continue

        regime = size_regime.loc[date, 'regime']
        if pd.isna(regime):
            regime = 'neutral'

        # Get ranks for this date
        mom_ranks = momentum_rank.loc[date] if date in momentum_rank.index else pd.Series()
        prof_ranks = profitability_rank.loc[date] if date in profitability_rank.index else pd.Series()

        mcaps = market_cap_df.loc[date] if date in market_cap_df.index else pd.Series(dtype=float)
        liq = liquidity_df.loc[date] if date in liquidity_df.index else pd.Series(dtype=float)
        liquid_universe, small_caps, large_caps = get_size_buckets_for_date(
            mcaps,
            liq,
            liquidity_quantile=SIZE_LIQUIDITY_QUANTILE,
        )
        if not liquid_universe:
            continue

        # Determine which tickers to consider based on regime
        if regime == 'small_cap':
            eligible_tickers = small_caps
        elif regime == 'large_cap':
            eligible_tickers = large_caps
        else:  # neutral
            eligible_tickers = liquid_universe

        for ticker in eligible_tickers:
            mom_r = mom_ranks.get(ticker, np.nan)
            prof_r = prof_ranks.get(ticker, np.nan)

            if pd.isna(mom_r) or pd.isna(prof_r):
                continue

            # Quality filter: both must be above median
            if mom_r < MIN_PERCENTILE or prof_r < MIN_PERCENTILE:
                continue

            # Combined score
            combined = MOMENTUM_WEIGHT * mom_r + PROFITABILITY_WEIGHT * prof_r
            scores.loc[date, ticker] = combined

    # Fill remaining NaN
    scores = scores.fillna(0.0)

    # Summary
    latest = scores.iloc[-1]
    nonzero = latest[latest > 0]
    if len(nonzero) > 0:
        print(f"  Latest qualifying stocks: {len(nonzero)}")
        top_5 = nonzero.nlargest(5)
        print(f"  Top 5: {', '.join(top_5.index.tolist())}")

        # Show regime composition
        large_in_top = sum(1 for t in top_5.index if t in large_latest)
        print(f"  Top 5 composition: {large_in_top} large caps, {5-large_in_top} small caps")

    print(f"  ✅ Size rotation quality signals: {scores.shape[0]} days × {scores.shape[1]} tickers")

    return scores
