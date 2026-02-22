"""
Size Rotation Signal Construction

Dynamically switches between small caps and large caps based on relative performance.

Logic:
- Calculate rolling relative performance: Small Cap Index vs Large Cap Index
- When small caps outperforming (positive momentum spread) -> Favor small caps
- When large caps outperforming (negative momentum spread) -> Favor large caps
- Combines size rotation regime with momentum within each size bucket

Size buckets are dynamic and data-driven:
- Big caps: top 10% by market cap
- Small caps: bottom 10% by market cap
- Liquidity filter is applied before bucketing to avoid microcaps

This signal adapts to market regimes:
- Risk-on periods: Small caps lead, signal favors small cap momentum
- Risk-off periods: Large caps lead, signal favors large cap quality/momentum

The key insight: Don't fight the tape. If large caps are leading,
ride that wave instead of hoping small caps will catch up.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

# Add parent to path
from bist_quant.common.market_cap_utils import (
    SIZE_LIQUIDITY_QUANTILE,
    get_size_buckets_for_date,
)
from bist_quant.common.utils import (
    assert_has_cross_section,
    assert_panel_not_constant,
    raise_signal_data_error,
)

logger = logging.getLogger(__name__)

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
LARGE_CAP_PERCENTILE = 90  # Top 10% by market cap = large caps
SMALL_CAP_PERCENTILE = 10  # Bottom 10% by market cap = small caps

# Minimum required names in each bucket
MIN_BUCKET_NAMES = 10


# ============================================================================
# SIZE BUCKET HELPERS
# ============================================================================

def build_market_cap_panel(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
) -> pd.DataFrame:
    """Build Date x Ticker market-cap panel using vectorized shares x price."""
    dates = pd.DatetimeIndex(dates)
    close_aligned = close_df.reindex(index=dates)
    panel = pd.DataFrame(np.nan, index=dates, columns=close_aligned.columns, dtype=float)
    if data_loader is None:
        return panel

    panel_cache = getattr(data_loader, "panel_cache", None)
    cache_key = None
    if panel_cache is not None:
        cache_key = panel_cache.make_key(
            "market_cap",
            start=dates[0] if len(dates) else None,
            end=dates[-1] if len(dates) else None,
            rows=int(len(dates)),
            tickers=tuple(str(col) for col in close_aligned.columns),
        )
        cached_panel = panel_cache.get(cache_key)
        if isinstance(cached_panel, pd.DataFrame):
            return cached_panel

    shares_panel = data_loader.load_shares_outstanding_panel()
    missing_tickers = list(close_aligned.columns)

    if shares_panel is not None and not shares_panel.empty:
        shares_aligned = shares_panel.reindex(index=dates, columns=close_aligned.columns).ffill()
        panel_values = np.multiply(
            close_aligned.to_numpy(dtype=np.float64, copy=False),
            shares_aligned.to_numpy(dtype=np.float64, copy=False),
        )
        panel = pd.DataFrame(panel_values, index=dates, columns=close_aligned.columns)

        covered_tickers = shares_aligned.notna().any(axis=0)
        missing_tickers = covered_tickers.index[~covered_tickers].tolist()

    # Fallback path for tickers absent from consolidated shares sources.
    if missing_tickers:
        logger.info(f"  Market-cap fallback for {len(missing_tickers)} tickers (missing consolidated shares)...")
        for idx, ticker in enumerate(missing_tickers, start=1):
            shares = data_loader.load_shares_outstanding(ticker)
            if shares is None or shares.empty:
                continue

            shares = shares.sort_index()
            shares = shares[~shares.index.duplicated(keep="last")]
            shares = shares.reindex(dates, method="ffill")
            panel[ticker] = close_aligned[ticker] * shares

            if idx % 100 == 0 or idx == len(missing_tickers):
                logger.info(f"  Market-cap fallback progress: {idx}/{len(missing_tickers)}")

    if panel_cache is not None and cache_key is not None:
        panel_cache.set(cache_key, panel)

    return panel


def build_liquidity_panel(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
) -> pd.DataFrame:
    """Get Date x Ticker rolling-ADV panel from data loader cache/build."""
    if data_loader is None:
        return pd.DataFrame(np.nan, index=dates, columns=close_df.columns)

    panel_cache = getattr(data_loader, "panel_cache", None)
    cache_key = None
    if panel_cache is not None:
        cache_key = panel_cache.make_key(
            "liquidity",
            start=dates[0] if len(dates) else None,
            end=dates[-1] if len(dates) else None,
            rows=int(len(dates)),
            tickers=tuple(str(col) for col in close_df.columns),
        )
        cached_panel = panel_cache.get(cache_key)
        if isinstance(cached_panel, pd.DataFrame):
            return cached_panel

    volume_df = getattr(data_loader, "_volume_df", None)
    if volume_df is None:
        try:
            prices_file = data_loader.data_dir / "bist_prices_full.csv"
            prices = data_loader.load_prices(prices_file)
            volume_df = data_loader.build_volume_panel(prices)
        except Exception:
            return pd.DataFrame(np.nan, index=dates, columns=close_df.columns)

    aligned = volume_df.reindex(index=dates, columns=close_df.columns)
    if panel_cache is not None and cache_key is not None:
        panel_cache.set(cache_key, aligned)
    return aligned


# ============================================================================
# SIZE REGIME DETECTION
# ============================================================================

def calculate_size_indices(
    close_df: pd.DataFrame,
    market_cap_df: pd.DataFrame,
    liquidity_df: pd.DataFrame | None = None,
    liquidity_quantile: float = SIZE_LIQUIDITY_QUANTILE,
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate synthetic small cap and large cap indices.

    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        market_cap_df: DataFrame of market caps (Date x Ticker)
        liquidity_df: DataFrame of rolling ADV/liquidity (Date x Ticker)
        liquidity_quantile: Bottom-liquidity cutoff to exclude microcaps

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

        mcaps = market_cap_df.loc[date]
        liq = liquidity_df.loc[date] if liquidity_df is not None and date in liquidity_df.index else pd.Series(dtype=float)
        _, small_caps, large_caps = get_size_buckets_for_date(
            mcaps,
            liq,
            liquidity_quantile=liquidity_quantile,
        )
        if len(small_caps) < MIN_BUCKET_NAMES or len(large_caps) < MIN_BUCKET_NAMES:
            continue

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
    liquidity_df: pd.DataFrame | None = None,
    lookback: int = RELATIVE_PERF_LOOKBACK,
    liquidity_quantile: float = SIZE_LIQUIDITY_QUANTILE,
) -> pd.DataFrame:
    """
    Calculate size rotation regime based on relative performance.

    Uses dynamic market-cap deciles within the liquid universe.

    Returns:
        DataFrame with columns:
        - 'small_perf': Rolling small cap performance
        - 'large_perf': Rolling large cap performance
        - 'spread': small - large (positive = small caps winning)
        - 'z_score': Standardized spread
        - 'regime': 'small_cap', 'large_cap', or 'neutral'
    """
    result = pd.DataFrame(index=close_df.index)
    result['large_perf'] = np.nan
    result['small_perf'] = np.nan
    result['spread'] = np.nan
    result['z_score'] = np.nan
    result['regime'] = np.nan

    if market_cap_df is None:
        return result

    # Calculate returns
    daily_returns = close_df.pct_change()

    small_daily = pd.Series(np.nan, index=close_df.index, dtype=float)
    large_daily = pd.Series(np.nan, index=close_df.index, dtype=float)

    for date in close_df.index:
        if date not in market_cap_df.index or date not in daily_returns.index:
            continue

        mcaps = market_cap_df.loc[date]
        liq = liquidity_df.loc[date] if liquidity_df is not None and date in liquidity_df.index else pd.Series(dtype=float)
        _, small_caps, large_caps = get_size_buckets_for_date(
            mcaps,
            liq,
            liquidity_quantile=liquidity_quantile,
        )
        if len(small_caps) < MIN_BUCKET_NAMES or len(large_caps) < MIN_BUCKET_NAMES:
            continue

        rets = daily_returns.loc[date]
        small_ret = rets.reindex(list(small_caps)).dropna().mean()
        large_ret = rets.reindex(list(large_caps)).dropna().mean()

        small_daily.loc[date] = small_ret
        large_daily.loc[date] = large_ret

    # Rolling cumulative returns
    result['large_perf'] = large_daily.rolling(lookback, min_periods=lookback//2).sum()
    result['small_perf'] = small_daily.rolling(lookback, min_periods=lookback//2).sum()

    # Spread: positive = small caps outperforming
    result['spread'] = result['small_perf'] - result['large_perf']

    # Z-score of spread
    spread_mean = result['spread'].rolling(252, min_periods=63).mean()
    spread_std = result['spread'].rolling(252, min_periods=63).std()
    result['z_score'] = (result['spread'] - spread_mean) / spread_std

    # Regime classification
    valid_z = result['z_score'].notna()
    result.loc[valid_z, 'regime'] = 'neutral'
    result.loc[result['z_score'] > SWITCH_THRESHOLD, 'regime'] = 'small_cap'
    result.loc[result['z_score'] < -SWITCH_THRESHOLD, 'regime'] = 'large_cap'

    return result


# ============================================================================
# SIZE ROTATION SIGNAL
# ============================================================================

def calculate_size_rotation_scores(
    close_df: pd.DataFrame,
    size_regime: pd.DataFrame,
    market_cap_df: pd.DataFrame,
    liquidity_df: pd.DataFrame | None = None,
    momentum_lookback: int = MOMENTUM_LOOKBACK,
    liquidity_quantile: float = SIZE_LIQUIDITY_QUANTILE,
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

    # Initialize scores with NaN to avoid silently treating missing data as neutral.
    scores = pd.DataFrame(np.nan, index=close_df.index, columns=close_df.columns, dtype=float)

    for date in close_df.index:
        if date not in size_regime.index or date not in momentum_rank.index:
            continue

        regime = size_regime.loc[date, 'regime']
        mom_ranks = momentum_rank.loc[date]

        if pd.isna(regime):
            continue

        mcaps = market_cap_df.loc[date] if date in market_cap_df.index else pd.Series(dtype=float)
        liq = liquidity_df.loc[date] if liquidity_df is not None and date in liquidity_df.index else pd.Series(dtype=float)
        liquid_universe, small_caps, large_caps = get_size_buckets_for_date(
            mcaps,
            liq,
            liquidity_quantile=liquidity_quantile,
        )
        if not liquid_universe:
            continue

        for ticker in liquid_universe:
            if pd.isna(mom_ranks.get(ticker)):
                continue

            mom_rank = mom_ranks[ticker]
            is_large_cap = ticker in large_caps
            is_small_cap = ticker in small_caps

            if regime == 'small_cap':
                # Favor small caps with momentum, penalize large caps, discount mid-caps
                if is_small_cap:
                    score = mom_rank * 1.2  # 120% weight (capped at 100)
                elif is_large_cap:
                    # Penalize large caps (but still consider top momentum large caps)
                    score = mom_rank * 0.5  # 50% weight
                else:
                    score = mom_rank * 0.8

            elif regime == 'large_cap':
                # Favor large caps with momentum, penalize small caps, discount mid-caps
                if is_large_cap:
                    # Boost large caps
                    score = mom_rank * 1.2
                elif is_small_cap:
                    # Penalize small caps
                    score = mom_rank * 0.5
                else:
                    score = mom_rank * 0.8

            else:  # neutral
                # Equal treatment inside liquid universe
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
    logger.info("\n🔧 Building size rotation signals...")
    logger.info(f"  Relative performance lookback: {RELATIVE_PERF_LOOKBACK} days")
    logger.info(f"  Switch threshold (z-score): ±{SWITCH_THRESHOLD}")
    logger.info(f"  Momentum lookback: {MOMENTUM_LOOKBACK} days")

    if data_loader is None:
        raise_signal_data_error(
            "size_rotation",
            "no data_loader provided; market-cap inputs are required",
        )

    # Build market-cap and liquidity panels
    logger.info("  Building market-cap panel (SERMAYE × price)...")
    market_cap_df = build_market_cap_panel(close_df, close_df.index, data_loader)
    assert_has_cross_section(
        market_cap_df,
        "size_rotation",
        "market-cap panel",
        min_valid_tickers=2 * MIN_BUCKET_NAMES,
    )
    logger.info("  Loading liquidity panel...")
    liquidity_df = build_liquidity_panel(close_df, close_df.index, data_loader)

    # Calculate size regime
    logger.info("  Calculating size regime...")
    size_regime = calculate_size_regime(
        close_df,
        market_cap_df=market_cap_df,
        liquidity_df=liquidity_df,
        liquidity_quantile=SIZE_LIQUIDITY_QUANTILE,
    )

    valid_regime_days = int(size_regime["regime"].notna().sum())
    if valid_regime_days < 20:
        raise_signal_data_error(
            "size_rotation",
            f"insufficient regime coverage: {valid_regime_days} valid regime days (< 20)",
        )

    # Show current regime
    latest_regime = size_regime['regime'].iloc[-1] if not size_regime.empty else 'unknown'
    if pd.isna(latest_regime):
        latest_regime = "unknown"
    latest_z = size_regime['z_score'].iloc[-1] if not size_regime.empty else 0
    logger.info(f"  Current regime: {latest_regime.upper()} (z-score: {latest_z:.2f})")

    # Regime distribution
    if not size_regime.empty:
        regime_counts = size_regime['regime'].value_counts()
        logger.info("  Historical regime distribution:")
        for regime, count in regime_counts.items():
            pct = count / len(size_regime) * 100
            logger.info(f"    {regime}: {count} days ({pct:.1f}%)")

    # Calculate rotation scores
    logger.info("  Calculating rotation-adjusted scores...")
    scores = calculate_size_rotation_scores(
        close_df,
        size_regime,
        market_cap_df=market_cap_df,
        liquidity_df=liquidity_df,
        liquidity_quantile=SIZE_LIQUIDITY_QUANTILE,
    )

    # Reindex to requested dates
    result = scores.reindex(dates)

    assert_has_cross_section(
        result,
        "size_rotation",
        "final score panel",
        min_valid_tickers=5,
    )
    assert_panel_not_constant(result, "size_rotation", "final score panel")
    latest_valid = int(result.iloc[-1].notna().sum()) if len(result.index) else 0
    if latest_valid < 5:
        raise_signal_data_error(
            "size_rotation",
            f"latest date has insufficient coverage: {latest_valid} valid names (< 5)",
        )

    # Summary stats
    latest = result.iloc[-1].dropna()
    if len(latest) > 0:
        logger.info(f"  Latest scores - Mean: {latest.mean():.1f}, Std: {latest.std():.1f}")

        # Show top picks by regime from current dynamic buckets
        mcap_latest = market_cap_df.loc[latest.name] if latest.name in market_cap_df.index else pd.Series(dtype=float)
        liq_latest = liquidity_df.loc[latest.name] if latest.name in liquidity_df.index else pd.Series(dtype=float)
        _, small_caps_latest, large_caps_latest = get_size_buckets_for_date(
            mcap_latest,
            liq_latest,
            liquidity_quantile=SIZE_LIQUIDITY_QUANTILE,
        )

        if latest_regime == 'large_cap' and large_caps_latest:
            large_cap_scores = latest.reindex(list(large_caps_latest)).dropna()
            top_5 = large_cap_scores.nlargest(5)
            logger.info(f"  Top 5 large caps (favored regime): {', '.join(top_5.index.tolist())}")
        elif latest_regime == 'small_cap' and small_caps_latest:
            small_cap_scores = latest.reindex(list(small_caps_latest)).dropna()
            top_5 = small_cap_scores.nlargest(5)
            logger.info(f"  Top 5 small caps (favored regime): {', '.join(top_5.index.tolist())}")
        else:
            top_5 = latest.nlargest(5)
            logger.info(f"  Top 5 overall (neutral regime): {', '.join(top_5.index.tolist())}")

    logger.info(f"  ✅ Size rotation signals: {result.shape[0]} days × {result.shape[1]} tickers")

    return result
