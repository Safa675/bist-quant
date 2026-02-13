"""
Factor Axis Construction and Screening

Functions to:
1. Combine raw panels into factor axes
2. Screen factors by quintile spread
3. Compute axis scores using quintile bucket selection
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# PARAMETERS
# ============================================================================

# Quintile buckets
N_BUCKETS = 5
BUCKET_LABELS = ("Q1_Low", "Q2_LowMid", "Q3_Center", "Q4_HighMid", "Q5_High")

# Multi-lookback ensemble (structural horizons: 1mo / 1q / 6mo / 1yr)
ENSEMBLE_LOOKBACK_WINDOWS = (21, 63, 126, 252)
ENSEMBLE_LOOKBACK_WEIGHTS = (0.25, 0.25, 0.25, 0.25)  # equal weight — no tuning
MIN_ROLLING_OBS_RATIO = 0.5


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def cross_sectional_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score by date."""
    panel = panel.astype(float)
    row_mean = panel.mean(axis=1)
    row_std = panel.std(axis=1).replace(0.0, np.nan)
    return panel.sub(row_mean, axis=0).div(row_std, axis=0)


def cross_sectional_rank(panel: pd.DataFrame, higher_is_better: bool = True) -> pd.DataFrame:
    """Cross-sectional percentile rank by date, scaled to 0-100.

    Uses 'average' method for ties, producing ranks in (0, 100].
    The lowest ranked stock gets a small positive value, not 0.
    """
    ranks = panel.rank(axis=1, pct=True, method="average")
    if not higher_is_better:
        ranks = 1.0 - ranks
    return (ranks * 100.0).where(panel.notna())


def rolling_cumulative_return(daily_returns: pd.Series, lookback: int, min_obs: int | None = None) -> pd.Series:
    """Compute rolling compounded returns using log-sum for numerical stability.

    Missing days are excluded from the calculation (not treated as 0% return).
    Uses log1p/expm1 for numerical stability with small returns.
    """
    if min_obs is None:
        min_obs = max(int(lookback * MIN_ROLLING_OBS_RATIO), 10)
    # Clip extreme negative returns to prevent log of negative numbers
    clipped = daily_returns.clip(lower=-0.99)
    log_growth = np.log1p(clipped)
    # Use min_periods to handle missing data properly (NaNs don't contribute)
    roll_log_sum = log_growth.rolling(lookback, min_periods=min_obs).sum()
    return np.expm1(roll_log_sum)


# ============================================================================
# AXIS COMBINATION FUNCTIONS
# ============================================================================

def _combine_components_properly(
    components: list,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
) -> pd.DataFrame:
    """Combine multiple z-score components using proper NaN handling.

    Instead of fillna(0) which biases toward neutral, we compute the mean
    only over available components for each (date, ticker) cell.
    """
    if not components:
        return pd.DataFrame(np.nan, index=dates, columns=tickers)

    # Stack all components and compute mean ignoring NaNs
    aligned = [comp.reindex(index=dates, columns=tickers) for _, comp in components]
    stacked = np.stack([df.values for df in aligned], axis=0)  # (n_components, n_dates, n_tickers)

    # Mean over components, ignoring NaNs
    with np.errstate(all='ignore'):
        result_values = np.nanmean(stacked, axis=0)

    # Where ALL components are NaN, result should be NaN
    all_nan_mask = np.all(np.isnan(stacked), axis=0)
    result_values[all_nan_mask] = np.nan

    return pd.DataFrame(result_values, index=dates, columns=tickers)


# Minimum data points required to include a component (proportional threshold)
MIN_COMPONENT_DATA_POINTS = 100


def combine_quality_axis(axis_panels: Dict[str, pd.DataFrame], dates: pd.DatetimeIndex, tickers: pd.Index) -> pd.DataFrame:
    """Combine quality sub-panels into a single quality axis (equal-weighted)."""
    components = []

    roe = axis_panels.get("quality_roe")
    if roe is not None and roe.notna().sum().sum() > MIN_COMPONENT_DATA_POINTS:
        components.append(("ROE", cross_sectional_zscore(roe)))

    roa = axis_panels.get("quality_roa")
    if roa is not None and roa.notna().sum().sum() > MIN_COMPONENT_DATA_POINTS:
        components.append(("ROA", cross_sectional_zscore(roa)))

    accruals = axis_panels.get("quality_accruals")
    if accruals is not None and accruals.notna().sum().sum() > MIN_COMPONENT_DATA_POINTS:
        # Low accruals = high quality, so invert
        components.append(("Accruals", -cross_sectional_zscore(accruals)))

    piotroski = axis_panels.get("quality_piotroski")
    if piotroski is not None and piotroski.notna().sum().sum() > MIN_COMPONENT_DATA_POINTS:
        components.append(("Piotroski", cross_sectional_zscore(piotroski)))

    if not components:
        print("    ⚠️  Quality axis has no valid components")
        return pd.DataFrame(np.nan, index=dates, columns=tickers)

    print(f"    Quality axis components: {[c[0] for c in components]}")

    return _combine_components_properly(components, dates, tickers)


def combine_liquidity_axis(axis_panels: Dict[str, pd.DataFrame], dates: pd.DatetimeIndex, tickers: pd.Index) -> pd.DataFrame:
    """Combine liquidity sub-panels into a single axis (equal-weighted).

    Liquidity measures ease of trading without price impact:
    - Amihud illiquidity (inverted: low = more liquid)
    - Real turnover (volume / shares outstanding)
    - Spread proxy (if available)
    """
    components = []

    amihud = axis_panels.get("liquidity_amihud")
    if amihud is not None and amihud.notna().sum().sum() > MIN_COMPONENT_DATA_POINTS:
        # Low Amihud = more liquid = good
        components.append(("Amihud", -cross_sectional_zscore(amihud)))

    turnover = axis_panels.get("liquidity_turnover")
    if turnover is not None and turnover.notna().sum().sum() > MIN_COMPONENT_DATA_POINTS:
        # Higher turnover = more liquid
        components.append(("Turnover", cross_sectional_zscore(turnover)))

    spread = axis_panels.get("liquidity_spread_proxy")
    if spread is not None and spread.notna().sum().sum() > MIN_COMPONENT_DATA_POINTS:
        # Low spread = more liquid = good
        components.append(("Spread", -cross_sectional_zscore(spread)))

    if not components:
        print("    ⚠️  Liquidity axis has no valid components")
        return pd.DataFrame(np.nan, index=dates, columns=tickers)

    print(f"    Liquidity axis components: {[c[0] for c in components]}")

    return _combine_components_properly(components, dates, tickers)


def combine_trading_intensity_axis(axis_panels: Dict[str, pd.DataFrame], dates: pd.DatetimeIndex, tickers: pd.Index) -> pd.DataFrame:
    """Combine trading intensity sub-panels into a single axis (equal-weighted).

    Trading intensity measures level of trading activity / market attention:
    - Relative volume (vs historical average)
    - Volume trend (recent vs older)
    - Turnover velocity (annualized turnover rate)

    This is distinct from liquidity which measures ease of trading.
    """
    components = []

    rel_vol = axis_panels.get("trading_intensity_relative_volume")
    if rel_vol is not None and rel_vol.notna().sum().sum() > MIN_COMPONENT_DATA_POINTS:
        components.append(("RelVol", cross_sectional_zscore(rel_vol)))

    vol_trend = axis_panels.get("trading_intensity_volume_trend")
    if vol_trend is not None and vol_trend.notna().sum().sum() > MIN_COMPONENT_DATA_POINTS:
        components.append(("VolTrend", cross_sectional_zscore(vol_trend)))

    turnover_vel = axis_panels.get("trading_intensity_turnover_velocity")
    if turnover_vel is not None and turnover_vel.notna().sum().sum() > MIN_COMPONENT_DATA_POINTS:
        components.append(("TurnoverVel", cross_sectional_zscore(turnover_vel)))

    if not components:
        print("    ⚠️  Trading Intensity axis has no valid components")
        return pd.DataFrame(np.nan, index=dates, columns=tickers)

    print(f"    Trading Intensity axis components: {[c[0] for c in components]}")

    return _combine_components_properly(components, dates, tickers)


def combine_sentiment_axis(axis_panels: Dict[str, pd.DataFrame], dates: pd.DatetimeIndex, tickers: pd.Index) -> pd.DataFrame:
    """Combine sentiment sub-panels into a single axis (equal-weighted)."""
    components = []

    high_pct = axis_panels.get("sentiment_52w_high_pct")
    if high_pct is not None and high_pct.notna().sum().sum() > MIN_COMPONENT_DATA_POINTS:
        components.append(("52wHigh", cross_sectional_zscore(high_pct)))

    accel = axis_panels.get("sentiment_price_acceleration")
    if accel is not None and accel.notna().sum().sum() > MIN_COMPONENT_DATA_POINTS:
        components.append(("PriceAccel", cross_sectional_zscore(accel)))

    reversal = axis_panels.get("sentiment_reversal")
    if reversal is not None and reversal.notna().sum().sum() > MIN_COMPONENT_DATA_POINTS:
        components.append(("Reversal", cross_sectional_zscore(reversal)))

    if not components:
        print("    ⚠️  Sentiment axis has no valid components")
        return pd.DataFrame(np.nan, index=dates, columns=tickers)

    print(f"    Sentiment axis components: {[c[0] for c in components]}")

    return _combine_components_properly(components, dates, tickers)


def combine_fundmom_axis(axis_panels: Dict[str, pd.DataFrame], dates: pd.DatetimeIndex, tickers: pd.Index) -> pd.DataFrame:
    """Combine fundamental momentum sub-panels (equal-weighted)."""
    components = []

    margin_chg = axis_panels.get("fundmom_margin_change")
    if margin_chg is not None and margin_chg.notna().sum().sum() > MIN_COMPONENT_DATA_POINTS:
        components.append(("MarginChg", cross_sectional_zscore(margin_chg)))

    sales_accel = axis_panels.get("fundmom_sales_accel")
    if sales_accel is not None and sales_accel.notna().sum().sum() > MIN_COMPONENT_DATA_POINTS:
        components.append(("SalesAccel", cross_sectional_zscore(sales_accel)))

    if not components:
        print("    ⚠️  FundMom axis has no valid components")
        return pd.DataFrame(np.nan, index=dates, columns=tickers)

    print(f"    FundMom axis components: {[c[0] for c in components]}")

    return _combine_components_properly(components, dates, tickers)


def combine_carry_axis(axis_panels: Dict[str, pd.DataFrame], dates: pd.DatetimeIndex, tickers: pd.Index) -> pd.DataFrame:
    """Combine carry sub-panels (dividend yield)."""
    components = []

    div_yield = axis_panels.get("carry_dividend_yield")
    if div_yield is not None and div_yield.notna().sum().sum() > MIN_COMPONENT_DATA_POINTS:
        components.append(("DivYield", cross_sectional_zscore(div_yield)))

    # Could add shareholder yield here if data becomes available
    sh_yield = axis_panels.get("carry_shareholder_yield")
    if sh_yield is not None and sh_yield.notna().sum().sum() > MIN_COMPONENT_DATA_POINTS:
        # Only add if it's different from div_yield
        if not div_yield.equals(sh_yield):
            components.append(("ShareholderYield", cross_sectional_zscore(sh_yield)))

    if not components:
        print("    ⚠️  Carry axis has no valid components")
        return pd.DataFrame(np.nan, index=dates, columns=tickers)

    print(f"    Carry axis components: {[c[0] for c in components]}")

    return _combine_components_properly(components, dates, tickers)


def combine_defensive_axis(axis_panels: Dict[str, pd.DataFrame], dates: pd.DatetimeIndex, tickers: pd.Index) -> pd.DataFrame:
    """Combine defensive sub-panels (equal-weighted)."""
    components = []

    stability = axis_panels.get("defensive_earnings_stability")
    if stability is not None and stability.notna().sum().sum() > MIN_COMPONENT_DATA_POINTS:
        components.append(("Stability", cross_sectional_zscore(stability)))

    beta = axis_panels.get("defensive_beta_to_market")
    if beta is not None and beta.notna().sum().sum() > MIN_COMPONENT_DATA_POINTS:
        # Low beta = defensive
        components.append(("LowBeta", -cross_sectional_zscore(beta)))

    if not components:
        print("    ⚠️  Defensive axis has no valid components")
        return pd.DataFrame(np.nan, index=dates, columns=tickers)

    print(f"    Defensive axis components: {[c[0] for c in components]}")

    return _combine_components_properly(components, dates, tickers)


# ============================================================================
# QUINTILE BUCKET SELECTION
# ============================================================================

def quintile_bucket_selection(
    axis_raw_scores: pd.DataFrame,
    daily_returns: pd.DataFrame,
    n_buckets: int = N_BUCKETS,
    lookback_windows: tuple = ENSEMBLE_LOOKBACK_WINDOWS,
    lookback_weights: tuple = ENSEMBLE_LOOKBACK_WEIGHTS,
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Quintile-based bucket selection - finds the BEST performing bucket.

    Returns:
        winning_bucket: Series of winning bucket index per date
        bucket_cum_returns: DataFrame of cumulative returns per bucket
        bucket_daily_returns: DataFrame of daily returns per bucket
        bucket_masks: Dict of bucket membership masks
    """
    axis = axis_raw_scores.reindex(daily_returns.index).astype(float)
    valid = axis.notna()
    dates = axis.index

    pct_ranks = axis.rank(axis=1, pct=True)

    bucket_edges = np.linspace(0, 1, n_buckets + 1)
    bucket_masks = {}
    bucket_daily = {}

    for i in range(n_buckets):
        lower = bucket_edges[i]
        upper = bucket_edges[i + 1]
        if i == n_buckets - 1:
            mask = (pct_ranks >= lower) & (pct_ranks <= upper) & valid
        else:
            mask = (pct_ranks >= lower) & (pct_ranks < upper) & valid

        bucket_masks[i] = mask
        bucket_w = mask.shift(1).astype(float)
        bucket_daily[i] = (daily_returns * bucket_w).sum(axis=1) / bucket_w.sum(axis=1).replace(0.0, np.nan)

    bucket_cum = {}
    for i in range(n_buckets):
        cum_scores = []
        for lb, wt in zip(lookback_windows, lookback_weights):
            min_obs = max(int(lb * MIN_ROLLING_OBS_RATIO), 10)
            cum = rolling_cumulative_return(bucket_daily[i], lb, min_obs)
            cum_scores.append(cum * wt)
        bucket_cum[i] = pd.concat(cum_scores, axis=1).sum(axis=1)

    bucket_cum_df = pd.DataFrame(bucket_cum, index=dates)
    bucket_daily_df = pd.DataFrame(bucket_daily, index=dates)

    winning_bucket = bucket_cum_df.idxmax(axis=1).fillna(n_buckets // 2)
    winning_bucket = winning_bucket.shift(1).fillna(n_buckets // 2)

    return winning_bucket, bucket_cum_df, bucket_daily_df, bucket_masks


def compute_quintile_axis_scores(
    axis_raw_scores: pd.DataFrame,
    winning_bucket: pd.Series,
    n_buckets: int = N_BUCKETS,
) -> pd.DataFrame:
    """Compute axis scores based on winning quintile bucket (vectorized)."""
    axis = axis_raw_scores.astype(float)
    valid = axis.notna()
    dates = axis.index
    tickers = axis.columns

    pct_ranks = axis.rank(axis=1, pct=True)
    # Use numpy floor directly (faster than apply)
    stock_buckets = np.floor((pct_ranks * n_buckets).clip(upper=n_buckets - 1e-9))

    winning_bucket_aligned = winning_bucket.reindex(dates).fillna(n_buckets // 2)
    winner_broadcast = pd.DataFrame(
        np.tile(winning_bucket_aligned.values[:, np.newaxis], (1, len(tickers))),
        index=dates,
        columns=tickers,
    )

    distance = (stock_buckets - winner_broadcast).abs()
    max_distance = n_buckets - 1
    scores = 100.0 - (distance / max_distance) * 100.0

    return scores.where(valid, np.nan)

# ============================================================================
# EXPONENTIALLY-WEIGHTED FACTOR SELECTION (Revealed Preference)
#
# Weight factors by their RECENT spread performance.
# Recent months matter exponentially more than old months.
#
# The only "parameter" is the exponential decay rate, which is derived from
# a natural market timescale: we use decay = 0.5^(1/6) so that 6 months ago
# has exactly half the weight of today. This is not tuned - it's a structural
# choice based on typical regime duration.
# ============================================================================

# Decay rate: 6-month half-life (not a tuning parameter, structural choice)
# decay^6 = 0.5, so decay = 0.5^(1/6) ≈ 0.891
_DECAY_PER_MONTH = 0.5 ** (1.0 / 6.0)


def _compute_rank_squared_weights(monthly_returns: pd.DataFrame, decay: float, n_factors: int) -> np.ndarray:
    """Convert monthly spread returns into rank-squared factor weights."""
    if monthly_returns.empty:
        return np.ones(n_factors) / n_factors

    weighted_sum = np.zeros(n_factors)
    weight_total = np.zeros(n_factors)

    n_months = len(monthly_returns)
    for row_idx in range(n_months):
        months_ago = n_months - row_idx - 1
        exp_weight = decay ** months_ago
        month_vals = monthly_returns.iloc[row_idx].values.astype(float)

        valid_mask = ~np.isnan(month_vals)
        weighted_sum[valid_mask] += exp_weight * month_vals[valid_mask]
        weight_total[valid_mask] += exp_weight

    with np.errstate(divide="ignore", invalid="ignore"):
        avg_spread = np.where(weight_total > 0, weighted_sum / weight_total, np.nan)

    avg_spread_for_rank = np.nan_to_num(avg_spread, nan=-1e9)
    ranks = np.argsort(np.argsort(avg_spread_for_rank)) + 1
    rank_weights = ranks.astype(float) ** 2
    return rank_weights / rank_weights.sum()


def compute_mwu_weights(
    axis_daily_returns: Dict[str, Tuple[pd.Series, pd.Series]],
    dates: pd.DatetimeIndex,
    warmup_months: int = 6,
    debug: bool = False,
    walkforward_train_years: int | None = None,
    walkforward_first_test_year: int | None = None,
    walkforward_last_test_year: int | None = None,
) -> pd.DataFrame:
    """
    Compute factor weights based on exponentially-weighted recent performance.

    Revealed preference model: factors that delivered spread returns RECENTLY
    get higher weights. Old performance fades with a 6-month half-life.

    Method:
      1. Compute monthly spread returns (Q5 - Q1) for each factor
      2. Exponentially weight: 6 months ago = 50%, 12 months ago = 25%
      3. Rank-based weights: best recent performer gets highest weight

    The weights are derived from ranks, not raw returns, to avoid
    sensitivity to outlier months.

    NaN handling:
      - Daily spreads: NaN means no data for that factor on that day (not 0)
      - Monthly returns: computed only from valid days; month with <5 days -> NaN
      - Ranking: NaN spreads are treated as worst (ranked lowest)

    Args:
      axis_daily_returns: Dict of (high_daily, low_daily) returns per factor
      dates: DatetimeIndex of trading days
      warmup_months: Number of months of historical data to use for initial weights.
                     If > 0, uses historical performance to initialize weights instead
                     of equal weighting. This eliminates the cold-start problem.
                     Default: 6 months (recommended to match decay half-life)
      debug: If True, print month-by-month weight diagnostics.
      walkforward_train_years: If set (>0), enable yearly walk-forward mode.
                              Each test year uses only prior N calendar years
                              to train factor weights.
      walkforward_first_test_year: Optional first OOS test year (inclusive).
      walkforward_last_test_year: Optional last OOS test year (inclusive).

    Returns:
      DataFrame (dates x factor_names) with weights summing to 1 per row.
    """
    factor_names = list(axis_daily_returns.keys())
    n_factors = len(factor_names)
    if n_factors == 0:
        return pd.DataFrame(index=dates)

    # Daily spread returns for each factor (Q5 - Q1)
    # Keep NaN where data is missing (don't fill with 0)
    # IMPORTANT: Use FULL date range from the data, not just backtest dates
    # This allows warm-up to access historical data before backtest start
    all_dates_list = []
    for name, (high_daily, low_daily) in axis_daily_returns.items():
        all_dates_list.extend(high_daily.index.tolist())
        all_dates_list.extend(low_daily.index.tolist())
    
    if all_dates_list:
        all_dates = pd.DatetimeIndex(sorted(set(all_dates_list)))
    else:
        all_dates = dates
    
    daily_spreads = pd.DataFrame(index=all_dates, dtype=float)
    daily_valid = pd.DataFrame(index=all_dates, dtype=bool)
    for name, (high_daily, low_daily) in axis_daily_returns.items():
        high = high_daily.reindex(all_dates)
        low = low_daily.reindex(all_dates)
        spread = high - low
        daily_spreads[name] = spread
        daily_valid[name] = spread.notna()

    # Compute monthly spread returns (only from valid days)
    # Use full date range to get ALL months including historical
    month_periods = all_dates.to_period("M")
    unique_months = month_periods.unique().sort_values()

    if debug:
        print(f"  [FIVE_FACTOR_DEBUG] MWU factors: {n_factors}, months: {len(unique_months)}")

    monthly_returns = pd.DataFrame(index=unique_months, columns=factor_names, dtype=float)
    monthly_valid_days = pd.DataFrame(index=unique_months, columns=factor_names, dtype=int)

    for month in unique_months:
        month_mask = month_periods == month
        month_spreads = daily_spreads.loc[month_mask]
        month_validity = daily_valid.loc[month_mask]

        for name in factor_names:
            valid_mask = month_validity[name]
            valid_spreads = month_spreads.loc[valid_mask, name]
            n_valid = len(valid_spreads)
            monthly_valid_days.loc[month, name] = n_valid

            if n_valid >= 5:  # Require at least 5 valid days
                monthly_returns.loc[month, name] = (1.0 + valid_spreads).prod() - 1.0
            else:
                monthly_returns.loc[month, name] = np.nan

    # Build weights month by month
    rows = []
    decay = _DECAY_PER_MONTH
    equal_weights = np.ones(n_factors) / n_factors

    # Find the first month that overlaps with backtest dates
    backtest_start_month = dates[0].to_period("M")
    first_backtest_month_idx = unique_months.get_loc(backtest_start_month) if backtest_start_month in unique_months else 0

    walkforward_enabled = walkforward_train_years is not None and walkforward_train_years > 0
    walkforward_weights_by_year: Dict[int, np.ndarray] = {}
    walkforward_train_ranges: Dict[int, tuple[int, int]] = {}

    if walkforward_enabled:
        if walkforward_first_test_year is None:
            walkforward_first_test_year = int(dates.min().year)
        if walkforward_last_test_year is None:
            walkforward_last_test_year = int(dates.max().year)

        if debug:
            print(
                f"  [FIVE_FACTOR_DEBUG] MWU walk-forward enabled: "
                f"train_years={walkforward_train_years}, "
                f"test_years={walkforward_first_test_year}-{walkforward_last_test_year}"
            )

        for test_year in range(walkforward_first_test_year, walkforward_last_test_year + 1):
            train_start_year = test_year - walkforward_train_years
            train_end_year = test_year - 1
            month_years = unique_months.year
            train_mask = (month_years >= train_start_year) & (month_years <= train_end_year)
            train_monthly = monthly_returns.loc[train_mask]

            if train_monthly.empty:
                w_year = equal_weights.copy()
                if debug:
                    print(
                        f"  [FIVE_FACTOR_DEBUG] MWU fold test={test_year}: "
                        f"train={train_start_year}-{train_end_year}, "
                        "months=0 -> equal weights"
                    )
            else:
                w_year = _compute_rank_squared_weights(train_monthly, decay, n_factors)
                if debug:
                    top_idx = np.argsort(-w_year)[:3]
                    top_weights = ", ".join([f"{factor_names[k]}={w_year[k]:.1%}" for k in top_idx])
                    print(
                        f"  [FIVE_FACTOR_DEBUG] MWU fold test={test_year}: "
                        f"train={train_start_year}-{train_end_year}, "
                        f"months={len(train_monthly)}, top_weights=[{top_weights}]"
                    )

            walkforward_weights_by_year[test_year] = w_year
            walkforward_train_ranges[test_year] = (train_start_year, train_end_year)

    for i, month in enumerate(unique_months):
        if walkforward_enabled and int(month.year) in walkforward_weights_by_year:
            w = walkforward_weights_by_year[int(month.year)].copy()
        elif i == 0 or (i < first_backtest_month_idx):
            # Pre-backtest months or very first month ever - equal weights
            w = equal_weights.copy()
        elif i == first_backtest_month_idx and warmup_months > 0:
            # WARM-UP: Use previous months to initialize weights
            if i >= warmup_months:
                print(f"  🔥 MWU Warm-up: Using {warmup_months} months of historical data for initial weights")
                warmup_monthly = monthly_returns.iloc[i - warmup_months : i]
                w = _compute_rank_squared_weights(warmup_monthly, decay, n_factors)
                print(f"  ✅ Initialized with warm-up weights (top 3: {', '.join([f'{factor_names[k]}={w[k]:.1%}' for k in np.argsort(-w)[:3]])})")
            else:
                w = equal_weights.copy()
                print(f"  ⚠️  Only {i} months of history (need {warmup_months}) - using equal weights")
        else:
            # Exponentially-weighted average of all available past monthly returns
            history_monthly = monthly_returns.iloc[:i]
            w = _compute_rank_squared_weights(history_monthly, decay, n_factors)

        if debug:
            top_idx = np.argsort(-w)[:3]
            top_weights = ", ".join([f"{factor_names[k]}={w[k]:.1%}" for k in top_idx])
            if i > 0:
                prev_month = unique_months[i - 1]
                prev_spreads = monthly_returns.iloc[i - 1].astype(float)
                prev_top = prev_spreads.sort_values(ascending=False).head(3)
                prev_top_str = ", ".join([f"{k}={v:.2%}" for k, v in prev_top.items()])
                walkforward_str = ""
                if walkforward_enabled and int(month.year) in walkforward_train_ranges:
                    tr_start, tr_end = walkforward_train_ranges[int(month.year)]
                    walkforward_str = f" train={tr_start}-{tr_end}"
                print(
                    f"  [FIVE_FACTOR_DEBUG] MWU month={month}: top_weights=[{top_weights}]"
                    f"{walkforward_str} prev_month={prev_month} spreads=[{prev_top_str}]"
                )
            else:
                print(f"  [FIVE_FACTOR_DEBUG] MWU month={month}: top_weights=[{top_weights}] (initial)")

        # Record for all days in this month
        month_mask = month_periods == month
        month_dates = all_dates[month_mask]  # Use all_dates, not dates
        for d in month_dates:
            rows.append((d, w.copy()))

    weight_df = pd.DataFrame(
        [r[1] for r in rows],
        index=pd.DatetimeIndex([r[0] for r in rows]),
        columns=factor_names,
    )
    
    # Reindex to backtest dates and forward-fill
    weight_df = weight_df.reindex(dates).ffill()

    # Normalize (should already sum to 1)
    s = weight_df.sum(axis=1).replace(0.0, np.nan)
    weight_df = weight_df.div(s, axis=0).fillna(1.0 / n_factors)

    return weight_df
