"""
Size Rotation Signal Construction

Dynamically switches between small caps and large caps based on relative performance.

Logic:
- Calculate rolling relative performance: Small Cap Index vs Large Cap Index
- When small caps outperforming (positive momentum spread) -> Favor small caps
- When large caps outperforming (negative momentum spread) -> Favor large caps
- Combines size rotation regime with momentum within each size bucket

This signal adapts to market regimes:
- Risk-on periods: Small caps lead, signal favors small cap momentum
- Risk-off periods: Large caps lead, signal favors large cap quality/momentum

The key insight: Don't fight the tape. If large caps are leading,
ride that wave instead of hoping small caps will catch up.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# SIZE ROTATION PARAMETERS
# ============================================================================

# Lookback for relative performance calculation
RELATIVE_PERF_LOOKBACK = 63  # ~3 months

# Threshold for switching (z-score of relative performance)
SWITCH_THRESHOLD = 0.5  # Switch when z-score exceeds this

# Momentum lookback within size bucket
MOMENTUM_LOOKBACK = 126  # 6 months

# Size percentile cutoffs
LARGE_CAP_PERCENTILE = 80  # Top 20% by market cap = large caps
SMALL_CAP_PERCENTILE = 50  # Bottom 50% by market cap = small caps

# XU100 constituents (major large caps) - used as proxy for large cap universe
XU100_TICKERS = [
    'THYAO', 'GARAN', 'AKBNK', 'SISE', 'KCHOL', 'TUPRS', 'SAHOL', 'EREGL',
    'ASELS', 'BIMAS', 'TCELL', 'PGSUS', 'KOZAL', 'KOZAA', 'SASA', 'TAVHL',
    'FROTO', 'TOASO', 'ARCLK', 'TTKOM', 'DOHOL', 'VESTL', 'PETKM', 'EKGYO',
    'HEKTS', 'GUBRF', 'ISCTR', 'VAKBN', 'YKBNK', 'TSKB', 'ENKAI', 'OTKAR',
    'AEFES', 'CCOLA', 'ULKER', 'BIZIM', 'MGROS', 'SOKM', 'KORDS', 'BRISA',
    'CIMSA', 'AKCNS', 'GOLTS', 'KARSN', 'TTRAK', 'GESAN', 'OYAKC', 'BUCIM',
    'ISGYO', 'EMLAK', 'KLRHO', 'MAVI', 'EGEEN', 'VESBE', 'ALARK', 'AGHOL',
    'AKSEN', 'AKSA', 'ALBRK', 'ANHYT', 'ANSGR', 'ASUZU', 'AYDEM', 'BAGFS',
    'BANVT', 'BERA', 'BFREN', 'BIENY', 'BJKAS', 'BRSAN', 'BRYAT', 'CANTE',
    'CEMTS', 'DEVA', 'DOAS', 'ECILC', 'ECZYT', 'ENJSA', 'ESEN', 'EUPWR',
    'FENER', 'FLAP', 'GLYHO', 'GSDHO', 'HALKB', 'IPEKE', 'ISMEN', 'KARTN',
    'KERVT', 'KONTR', 'LOGO', 'MPARK', 'NETAS', 'ODAS', 'PAPIL', 'PSGYO',
    'QUAGR', 'SAHOL', 'SELEC', 'SKBNK', 'SMRTG', 'SNGYO', 'SODSN', 'TABGD',
    'TKFEN', 'TMSN', 'TRILC', 'TSGYO', 'TURSG', 'VERUS', 'YATAS', 'YEOTK'
]


# ============================================================================
# SIZE REGIME DETECTION
# ============================================================================

def calculate_size_indices(
    close_df: pd.DataFrame,
    market_cap_df: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate synthetic small cap and large cap indices.

    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        market_cap_df: DataFrame of market caps (Date x Ticker)

    Returns:
        Tuple of (small_cap_index, large_cap_index) as pd.Series
    """
    # For each date, classify stocks into size buckets
    small_cap_returns = []
    large_cap_returns = []

    daily_returns = close_df.pct_change()

    for date in close_df.index:
        if date not in market_cap_df.index:
            continue

        mcaps = market_cap_df.loc[date].dropna()
        if len(mcaps) < 20:
            continue

        # Classify by market cap percentile
        large_threshold = mcaps.quantile(LARGE_CAP_PERCENTILE / 100)
        small_threshold = mcaps.quantile(SMALL_CAP_PERCENTILE / 100)

        large_caps = mcaps[mcaps >= large_threshold].index
        small_caps = mcaps[mcaps <= small_threshold].index

        # Get returns for this date
        if date in daily_returns.index:
            rets = daily_returns.loc[date]

            # Equal-weighted returns for each bucket
            large_ret = rets[rets.index.isin(large_caps)].mean()
            small_ret = rets[rets.index.isin(small_caps)].mean()

            small_cap_returns.append({'date': date, 'return': small_ret})
            large_cap_returns.append({'date': date, 'return': large_ret})

    # Build index series
    if not small_cap_returns:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    small_df = pd.DataFrame(small_cap_returns).set_index('date')
    large_df = pd.DataFrame(large_cap_returns).set_index('date')

    # Cumulative returns (index level)
    small_index = (1 + small_df['return'].fillna(0)).cumprod()
    large_index = (1 + large_df['return'].fillna(0)).cumprod()

    return small_index, large_index


def calculate_size_regime(
    close_df: pd.DataFrame,
    market_cap_df: pd.DataFrame = None,
    lookback: int = RELATIVE_PERF_LOOKBACK,
) -> pd.DataFrame:
    """
    Calculate size rotation regime based on relative performance.

    Uses XU100 as large cap proxy if market_cap_df not available.

    Returns:
        DataFrame with columns:
        - 'small_perf': Rolling small cap performance
        - 'large_perf': Rolling large cap performance
        - 'spread': small - large (positive = small caps winning)
        - 'z_score': Standardized spread
        - 'regime': 'small_cap', 'large_cap', or 'neutral'
    """
    result = pd.DataFrame(index=close_df.index)

    # Calculate returns
    daily_returns = close_df.pct_change()

    # Use XU100 tickers as large cap proxy
    large_cap_tickers = [t for t in XU100_TICKERS if t in close_df.columns]
    small_cap_tickers = [t for t in close_df.columns if t not in large_cap_tickers]

    # Equal-weighted daily returns for each bucket
    large_ret = daily_returns[large_cap_tickers].mean(axis=1)
    small_ret = daily_returns[small_cap_tickers].mean(axis=1)

    # Rolling cumulative returns
    result['large_perf'] = large_ret.rolling(lookback, min_periods=lookback//2).sum()
    result['small_perf'] = small_ret.rolling(lookback, min_periods=lookback//2).sum()

    # Spread: positive = small caps outperforming
    result['spread'] = result['small_perf'] - result['large_perf']

    # Z-score of spread
    spread_mean = result['spread'].rolling(252, min_periods=63).mean()
    spread_std = result['spread'].rolling(252, min_periods=63).std()
    result['z_score'] = (result['spread'] - spread_mean) / spread_std

    # Regime classification
    result['regime'] = 'neutral'
    result.loc[result['z_score'] > SWITCH_THRESHOLD, 'regime'] = 'small_cap'
    result.loc[result['z_score'] < -SWITCH_THRESHOLD, 'regime'] = 'large_cap'

    return result


# ============================================================================
# SIZE ROTATION SIGNAL
# ============================================================================

def calculate_size_rotation_scores(
    close_df: pd.DataFrame,
    size_regime: pd.DataFrame,
    momentum_lookback: int = MOMENTUM_LOOKBACK,
) -> pd.DataFrame:
    """
    Calculate size rotation signal scores.

    In small_cap regime: Score small caps by momentum, penalize large caps
    In large_cap regime: Score large caps by momentum, penalize small caps
    In neutral: Blend both

    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        size_regime: DataFrame from calculate_size_regime()
        momentum_lookback: Lookback for momentum calculation

    Returns:
        DataFrame (dates x tickers) with rotation-adjusted scores (0-100)
    """
    # Calculate momentum for all stocks
    momentum = close_df.pct_change(momentum_lookback)
    momentum_rank = momentum.rank(axis=1, pct=True) * 100

    # Classify tickers
    large_cap_tickers = set(t for t in XU100_TICKERS if t in close_df.columns)

    # Initialize scores
    scores = pd.DataFrame(0.0, index=close_df.index, columns=close_df.columns)

    for date in close_df.index:
        if date not in size_regime.index or date not in momentum_rank.index:
            continue

        regime = size_regime.loc[date, 'regime']
        z_score = size_regime.loc[date, 'z_score']
        mom_ranks = momentum_rank.loc[date]

        if pd.isna(regime):
            regime = 'neutral'

        for ticker in close_df.columns:
            if pd.isna(mom_ranks.get(ticker)):
                continue

            mom_rank = mom_ranks[ticker]
            is_large_cap = ticker in large_cap_tickers

            if regime == 'small_cap':
                # Favor small caps with momentum
                if is_large_cap:
                    # Penalize large caps (but still consider top momentum large caps)
                    score = mom_rank * 0.5  # 50% weight
                else:
                    # Boost small caps
                    score = mom_rank * 1.2  # 120% weight (capped at 100)

            elif regime == 'large_cap':
                # Favor large caps with momentum
                if is_large_cap:
                    # Boost large caps
                    score = mom_rank * 1.2
                else:
                    # Penalize small caps
                    score = mom_rank * 0.5

            else:  # neutral
                # Equal treatment
                score = mom_rank

            scores.loc[date, ticker] = min(score, 100)  # Cap at 100

    return scores


# ============================================================================
# SIGNAL BUILDER (MAIN INTERFACE)
# ============================================================================

def build_size_rotation_signals(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
) -> pd.DataFrame:
    """
    Build size rotation signal panel.

    This signal dynamically tilts towards small or large caps based on
    which size segment is showing relative strength.

    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        dates: DatetimeIndex to align signals to
        data_loader: DataLoader instance (optional)

    Returns:
        DataFrame (dates x tickers) with size rotation scores
    """
    print("\n🔧 Building size rotation signals...")
    print(f"  Relative performance lookback: {RELATIVE_PERF_LOOKBACK} days")
    print(f"  Switch threshold (z-score): ±{SWITCH_THRESHOLD}")
    print(f"  Momentum lookback: {MOMENTUM_LOOKBACK} days")

    # Calculate size regime
    print("  Calculating size regime...")
    size_regime = calculate_size_regime(close_df)

    # Show current regime
    latest_regime = size_regime['regime'].iloc[-1] if not size_regime.empty else 'unknown'
    latest_z = size_regime['z_score'].iloc[-1] if not size_regime.empty else 0
    print(f"  Current regime: {latest_regime.upper()} (z-score: {latest_z:.2f})")

    # Regime distribution
    if not size_regime.empty:
        regime_counts = size_regime['regime'].value_counts()
        print(f"  Historical regime distribution:")
        for regime, count in regime_counts.items():
            pct = count / len(size_regime) * 100
            print(f"    {regime}: {count} days ({pct:.1f}%)")

    # Calculate rotation scores
    print("  Calculating rotation-adjusted scores...")
    scores = calculate_size_rotation_scores(close_df, size_regime)

    # Reindex to requested dates
    result = scores.reindex(dates)

    # Fill NaN with neutral score
    result = result.fillna(50.0)

    # Summary stats
    latest = result.iloc[-1].dropna()
    if len(latest) > 0:
        print(f"  Latest scores - Mean: {latest.mean():.1f}, Std: {latest.std():.1f}")

        # Show top picks by regime
        if latest_regime == 'large_cap':
            large_cap_scores = latest[[t for t in latest.index if t in XU100_TICKERS]]
            top_5 = large_cap_scores.nlargest(5)
            print(f"  Top 5 large caps (favored regime): {', '.join(top_5.index.tolist())}")
        elif latest_regime == 'small_cap':
            small_cap_scores = latest[[t for t in latest.index if t not in XU100_TICKERS]]
            top_5 = small_cap_scores.nlargest(5)
            print(f"  Top 5 small caps (favored regime): {', '.join(top_5.index.tolist())}")
        else:
            top_5 = latest.nlargest(5)
            print(f"  Top 5 overall (neutral regime): {', '.join(top_5.index.tolist())}")

    print(f"  ✅ Size rotation signals: {result.shape[0]} days × {result.shape[1]} tickers")

    return result
