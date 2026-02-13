"""
Pairs Trading Signal Construction

Statistical arbitrage strategy based on price cointegration:
- Find pairs of stocks with high historical correlation
- Trade mean reversion when prices diverge

Strategy:
- Formation period: Find pairs with smallest sum of squared deviations
- Trading: Go long the underperformer, short the outperformer when spread diverges

Based on Quantpedia strategy: Pairs Trading with Stocks
Signal: Z-score of spread for identified pairs
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# PAIRS TRADING PARAMETERS
# ============================================================================

FORMATION_PERIOD = 252  # 12 months for pair formation
TRADING_PERIOD = 126    # 6 months trading period
TOP_PAIRS = 20          # Number of top pairs to track
ENTRY_ZSCORE = 2.0      # Enter when spread diverges by 2 std
EXIT_ZSCORE = 0.5       # Exit when spread reverts to 0.5 std
MIN_CORRELATION = 0.7   # Minimum correlation for pair candidates


# ============================================================================
# PAIR FINDING FUNCTIONS
# ============================================================================

def normalize_prices(close_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize prices to start at 1 for comparison.

    Args:
        close_df: DataFrame of close prices

    Returns:
        Normalized price DataFrame
    """
    first_valid = close_df.apply(lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else np.nan)
    return close_df / first_valid


def calculate_distance(series1: pd.Series, series2: pd.Series) -> float:
    """
    Calculate sum of squared deviations between two normalized price series.

    Args:
        series1: First normalized price series
        series2: Second normalized price series

    Returns:
        Sum of squared deviations (lower = more similar)
    """
    common = series1.dropna().index.intersection(series2.dropna().index)
    if len(common) < 20:
        return np.inf

    diff = series1.loc[common] - series2.loc[common]
    return (diff ** 2).sum()


def find_pairs(
    close_df: pd.DataFrame,
    lookback: int = FORMATION_PERIOD,
    top_n: int = TOP_PAIRS,
    min_corr: float = MIN_CORRELATION,
) -> List[Tuple[str, str, float]]:
    """
    Find the top pairs with smallest distance measure.

    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        lookback: Formation period lookback
        top_n: Number of top pairs to return
        min_corr: Minimum correlation threshold

    Returns:
        List of (ticker1, ticker2, distance) tuples
    """
    # Get recent data for pair formation
    recent_df = close_df.iloc[-lookback:].dropna(axis=1, how='any')

    if recent_df.shape[1] < 10:
        return []

    # Normalize prices
    normalized = normalize_prices(recent_df)

    # Pre-filter by correlation (much faster than computing all distances)
    corr_matrix = recent_df.pct_change().corr()

    # Find candidate pairs with high correlation
    candidates = []
    tickers = list(normalized.columns)

    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            if i >= j:
                continue
            if t1 in corr_matrix.columns and t2 in corr_matrix.columns:
                corr = corr_matrix.loc[t1, t2]
                if corr >= min_corr:
                    candidates.append((t1, t2, corr))

    # Calculate distances for candidates only
    pairs_with_distance = []
    for t1, t2, corr in candidates:
        dist = calculate_distance(normalized[t1], normalized[t2])
        if dist != np.inf:
            pairs_with_distance.append((t1, t2, dist))

    # Sort by distance and return top pairs
    pairs_with_distance.sort(key=lambda x: x[2])

    return pairs_with_distance[:top_n]


# ============================================================================
# SPREAD AND Z-SCORE CALCULATION
# ============================================================================

def calculate_spread_zscore(
    close_df: pd.DataFrame,
    pairs: List[Tuple[str, str, float]],
    lookback: int = FORMATION_PERIOD,
) -> Dict[Tuple[str, str], pd.Series]:
    """
    Calculate z-score of spread for each pair.

    Spread = normalized_price_1 - normalized_price_2
    Z-score = (spread - rolling_mean) / rolling_std

    Args:
        close_df: DataFrame of close prices
        pairs: List of (ticker1, ticker2, distance) tuples
        lookback: Lookback for mean/std calculation

    Returns:
        Dict mapping pair to z-score Series
    """
    result = {}

    for t1, t2, _ in pairs:
        if t1 not in close_df.columns or t2 not in close_df.columns:
            continue

        # Calculate normalized spread
        norm1 = close_df[t1] / close_df[t1].iloc[0]
        norm2 = close_df[t2] / close_df[t2].iloc[0]
        spread = norm1 - norm2

        # Calculate rolling z-score
        rolling_mean = spread.rolling(lookback, min_periods=lookback // 2).mean()
        rolling_std = spread.rolling(lookback, min_periods=lookback // 2).std()

        zscore = (spread - rolling_mean) / rolling_std.replace(0, np.nan)
        result[(t1, t2)] = zscore

    return result


# ============================================================================
# SIGNAL BUILDER
# ============================================================================

def build_pairs_trading_signals(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
) -> pd.DataFrame:
    """
    Build pairs trading signal panel.

    For each stock, the signal is based on its role in pairs:
    - If stock is the "underperformer" in a diverged pair: positive signal (long)
    - If stock is the "outperformer" in a diverged pair: negative signal (short)
    - Stocks not in active pairs: neutral

    Note: This signal is less suitable for long-only portfolios.
    Consider using as a supplement or for hedging.

    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        dates: DatetimeIndex to align signals to
        data_loader: Optional DataLoader instance

    Returns:
        DataFrame (dates x tickers) with pairs trading scores
    """
    print("\n🔧 Building pairs trading signals...")
    print(f"  Formation Period: {FORMATION_PERIOD} days")
    print(f"  Top Pairs: {TOP_PAIRS}")
    print(f"  Entry Z-score: ±{ENTRY_ZSCORE}")
    print(f"  Min Correlation: {MIN_CORRELATION}")

    # Initialize result
    result = pd.DataFrame(0.0, index=dates, columns=close_df.columns)

    # Find pairs using full history
    pairs = find_pairs(close_df, FORMATION_PERIOD, TOP_PAIRS, MIN_CORRELATION)
    print(f"  Found {len(pairs)} pairs")

    if len(pairs) == 0:
        print("  ⚠️ No pairs found, returning neutral signals")
        return result

    # Show top 5 pairs
    for i, (t1, t2, dist) in enumerate(pairs[:5]):
        print(f"    Pair {i+1}: {t1} - {t2} (distance: {dist:.4f})")

    # Calculate z-scores for all pairs
    zscore_dict = calculate_spread_zscore(close_df, pairs, FORMATION_PERIOD)

    # Build signal for each stock based on pair membership
    for (t1, t2), zscore in zscore_dict.items():
        zscore_aligned = zscore.reindex(dates).fillna(0)

        # When z-score > threshold: t1 overperformed, t2 underperformed
        # Signal: short t1 (negative), long t2 (positive)
        long_t2_mask = zscore_aligned > ENTRY_ZSCORE
        long_t1_mask = zscore_aligned < -ENTRY_ZSCORE

        # Add to stock signals
        if t1 in result.columns:
            # t1 gets negative signal when it's overperforming (short candidate)
            result.loc[long_t2_mask, t1] -= zscore_aligned[long_t2_mask].abs() / ENTRY_ZSCORE
            # t1 gets positive signal when it's underperforming (long candidate)
            result.loc[long_t1_mask, t1] += zscore_aligned[long_t1_mask].abs() / ENTRY_ZSCORE

        if t2 in result.columns:
            # t2 gets positive signal when it's underperforming (long candidate)
            result.loc[long_t2_mask, t2] += zscore_aligned[long_t2_mask].abs() / ENTRY_ZSCORE
            # t2 gets negative signal when it's overperforming (short candidate)
            result.loc[long_t1_mask, t2] -= zscore_aligned[long_t1_mask].abs() / ENTRY_ZSCORE

    # Normalize to reasonable range
    result = result.clip(-3, 3)

    # Summary
    valid_scores = result.dropna(how='all')
    if not valid_scores.empty:
        latest = valid_scores.iloc[-1]
        n_active = (latest.abs() > 0.1).sum()
        print(f"  Stocks in active pairs (latest): {n_active}")

    print(f"  ✅ Pairs trading signals: {result.shape[0]} days × {result.shape[1]} tickers")

    return result
