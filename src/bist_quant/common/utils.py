"""Common utilities for signal construction

This module provides centralized utility functions for:
- Signal panel validation and schema enforcement
- Cross-sectional transformations (z-score, rank)
- Fundamental data helpers (TTM, lag, quarter coercion)
- Debug logging
- Staleness tracking
"""

import json
import logging
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
# ============================================================================
# DEBUG LOGGING
# ============================================================================

DEBUG_ENV_VAR = "DEBUG"


def debug_enabled() -> bool:
    """Check if debug mode is enabled via environment variable."""
    return os.getenv(DEBUG_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}


def debug_log(message: str, prefix: str = "DEBUG") -> None:
    """Log a debug message if debug mode is enabled."""
    if debug_enabled():
        logger.info(f"  [{prefix}] {message}")


class SignalDataError(RuntimeError):
    """Raised when a signal cannot be built due to missing/invalid critical data."""


def raise_signal_data_error(signal_name: str, reason: str) -> None:
    """Raise a standardized signal data error."""
    raise SignalDataError(f"[{signal_name}] {reason}")


def assert_has_non_na_values(
    panel: pd.DataFrame,
    signal_name: str,
    context: str,
) -> None:
    """Ensure panel contains at least one non-NaN value."""
    if panel is None or panel.empty:
        raise_signal_data_error(signal_name, f"{context}: panel is empty")
    if int(panel.notna().sum().sum()) == 0:
        raise_signal_data_error(signal_name, f"{context}: panel has no non-NaN values")


def assert_has_cross_section(
    panel: pd.DataFrame,
    signal_name: str,
    context: str,
    min_valid_tickers: int = 5,
) -> None:
    """Ensure at least one date has enough non-NaN names for cross-sectional ranking."""
    assert_has_non_na_values(panel, signal_name, context)
    max_valid = int(panel.notna().sum(axis=1).max())
    if max_valid < min_valid_tickers:
        raise_signal_data_error(
            signal_name,
            f"{context}: max valid names per date is {max_valid} (< {min_valid_tickers})",
        )


def assert_panel_not_constant(
    panel: pd.DataFrame,
    signal_name: str,
    context: str,
    eps: float = 1e-9,
) -> None:
    """Ensure panel is not constant across tickers for all dates."""
    assert_has_non_na_values(panel, signal_name, context)
    # If every row has near-zero cross-sectional std, the signal is degenerate.
    row_std = panel.std(axis=1, skipna=True).fillna(0.0)
    if float(row_std.max()) <= eps:
        raise_signal_data_error(signal_name, f"{context}: cross-section is constant for all dates")


def validate_signal_panel_schema(
    panel: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
    signal_name: str,
    context: str = "final signal panel",
    dtype: type | np.dtype | None = None,
) -> pd.DataFrame:
    """Validate and align a signal output panel to the common schema contract.

    Contract:
      - DataFrame with DatetimeIndex dates x ticker columns
      - no duplicate/NaT dates, monotonic increasing index
      - no duplicate ticker columns
      - no object dtype columns
      - align to provided (dates, tickers) and cast to numeric dtype
    """
    if panel is None or not isinstance(panel, pd.DataFrame):
        raise_signal_data_error(signal_name, f"{context}: expected DataFrame, got {type(panel).__name__}")

    ref_dates = pd.DatetimeIndex(pd.to_datetime(pd.Index(dates), errors="coerce"))
    if len(ref_dates) == 0:
        raise_signal_data_error(signal_name, f"{context}: reference dates are empty")
    if ref_dates.hasnans:
        raise_signal_data_error(signal_name, f"{context}: reference dates contain NaT")
    if ref_dates.has_duplicates:
        raise_signal_data_error(signal_name, f"{context}: reference dates contain duplicates")
    if not ref_dates.is_monotonic_increasing:
        raise_signal_data_error(signal_name, f"{context}: reference dates must be monotonic increasing")

    ref_tickers = pd.Index(tickers)
    if len(ref_tickers) == 0:
        raise_signal_data_error(signal_name, f"{context}: reference tickers are empty")
    if ref_tickers.has_duplicates:
        dup = ref_tickers[ref_tickers.duplicated()].unique().tolist()[:5]
        raise_signal_data_error(signal_name, f"{context}: duplicate reference tickers (sample={dup})")

    out = panel.copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index, errors="coerce"))
    if out.index.hasnans:
        raise_signal_data_error(signal_name, f"{context}: panel index contains NaT")
    if out.index.has_duplicates:
        raise_signal_data_error(signal_name, f"{context}: panel index contains duplicate dates")
    if not out.index.is_monotonic_increasing:
        raise_signal_data_error(signal_name, f"{context}: panel index must be monotonic increasing")
    if out.columns.has_duplicates:
        dup_cols = out.columns[out.columns.duplicated()].unique().tolist()[:5]
        raise_signal_data_error(signal_name, f"{context}: duplicate ticker columns (sample={dup_cols})")

    object_cols = out.select_dtypes(include=["object"]).columns.tolist()
    if object_cols:
        raise_signal_data_error(signal_name, f"{context}: object dtype columns not allowed (sample={object_cols[:5]})")

    out = out.reindex(index=ref_dates, columns=ref_tickers)
    target_dtype = np.dtype(dtype if dtype is not None else np.float64)
    try:
        out = out.astype(target_dtype)
    except Exception as exc:
        raise_signal_data_error(
            signal_name,
            f"{context}: cannot cast to {target_dtype} ({exc})",
        )

    return out


def assert_recent_enough(
    series_dates: pd.DatetimeIndex,
    required_date: pd.Timestamp,
    signal_name: str,
    context: str,
    max_staleness_days: int = 400,
) -> None:
    """Ensure source dates are not excessively stale vs required date."""
    if len(series_dates) == 0:
        raise_signal_data_error(signal_name, f"{context}: no dates available")
    latest = pd.Timestamp(series_dates.max())
    staleness = (pd.Timestamp(required_date) - latest).days
    if staleness > max_staleness_days:
        raise_signal_data_error(
            signal_name,
            f"{context}: latest date {latest.date()} is stale by {staleness} days",
        )


def normalize_ticker(ticker: str) -> str:
    """Normalize ticker symbol"""
    return ticker.split('.')[0].upper()


def _turkish_match(s1: str, s2: str) -> bool:
    """Compare two strings case-insensitively, handling Turkish I/İ."""
    def _norm(s: str) -> str:
        return str(s).strip().replace('I', 'ı').replace('İ', 'i').lower()
    return _norm(s1) == _norm(s2)

def pick_row(df: pd.DataFrame, keys: tuple) -> pd.Series | None:
    """Pick first matching row from dataframe"""
    for key in keys:
        matches = df[df.apply(lambda r: _turkish_match(r.iloc[0], key), axis=1)]
        if not matches.empty:
            return matches.iloc[0]
    return None


def get_consolidated_sheet(
    consolidated: pd.DataFrame | None,
    ticker: str,
    sheet_name: str,
) -> pd.DataFrame:
    """Return a single sheet (row_name indexed) from consolidated fundamentals parquet."""
    if consolidated is None:
        return pd.DataFrame()
    try:
        sheet = consolidated.xs((ticker, sheet_name), level=("ticker", "sheet_name"))
    except Exception:
        return pd.DataFrame()
    sheet = sheet.copy()
    sheet.index = sheet.index.astype(str).str.strip()
    return sheet


def pick_row_from_sheet(sheet: pd.DataFrame, keys: tuple) -> pd.Series | None:
    """Pick first matching row from a consolidated sheet dataframe."""
    if sheet is None or sheet.empty:
        return None
    fallback = None
    for key in keys:
        mask = sheet.index.to_series().apply(lambda x: _turkish_match(x, key))
        if mask.any():
            orig_keys = sheet.index[mask].unique()
            for orig_key in orig_keys:
                row = sheet.loc[orig_key]
                if isinstance(row, pd.DataFrame):
                    candidates = [row.iloc[i] for i in range(len(row))]
                else:
                    candidates = [row]

            for candidate in candidates:
                if fallback is None:
                    fallback = candidate
                parsed = coerce_quarter_cols(candidate)
                if not parsed.empty:
                    return candidate
    return fallback


def coerce_quarter_cols(row: pd.Series) -> pd.Series:
    """Coerce quarter columns to datetime index"""
    dates = []
    values = []
    for col in row.index:
        if isinstance(col, str) and '/' in col:
            try:
                parts = col.split('/')
                if len(parts) == 2:
                    year = int(parts[0])
                    month = int(parts[1])
                    if year < 2000 or year > 2030:
                        continue
                    if month not in [3, 6, 9, 12]:
                        continue
                    dt = pd.Timestamp(year=year, month=month, day=1)
                    val = row[col]
                    if pd.notna(val):
                        try:
                            values.append(float(str(val).replace(',', '.').replace(' ', '')))
                            dates.append(dt)
                        except Exception:
                            pass
            except Exception:
                pass
    if not dates:
        return pd.Series(dtype=float)
    return pd.Series(values, index=pd.DatetimeIndex(dates))


def sum_ttm(series: pd.Series) -> pd.Series:
    """
    Calculate trailing twelve months sum.
    
    Handles missing quarters more robustly by:
    - Requiring at least 3 quarters (allowing 1 missing)
    - Only computing TTM where we have quarterly data
    
    If a company has gaps, the TTM will be less accurate but won't silently
    use stale data.
    """
    if series.empty:
        return pd.Series(dtype=float)
    
    series = series.sort_index()
    
    # Check for proper quarterly data (3 month gaps between observations)
    if len(series) >= 2:
        gaps = series.index.to_series().diff().dropna()
        median_gap_days = gaps.dt.days.median() if len(gaps) > 0 else 90
        
        # If median gap is > 120 days, data may be annual not quarterly
        if median_gap_days > 120:
            # Return the series as-is (already annualized)
            return series
    
    # Rolling 4-quarter sum with min_periods=3 (allows 1 missing quarter)
    ttm = series.rolling(window=4, min_periods=3).sum()
    
    # For cases with only 3 quarters, scale up to annual estimate
    valid_counts = series.rolling(window=4, min_periods=3).count()
    ttm = ttm * (4 / valid_counts)
    
    return ttm.dropna()


def apply_lag(
    series: pd.Series,
    dates: pd.DatetimeIndex,
    q4_lag_days: int = 70,
    other_lag_days: int = 40,
) -> pd.Series:
    """
    Apply reporting lag to fundamental data.

    In Turkey, financial statements are announced with a delay:
    - Q1, Q2, Q3 periods: ~40 days after quarter end
    - Q4 (annual): ~70 days after year end (more complex auditing)

    Args:
        series: Fundamental data indexed by calendar quarter end date
        dates: Target daily DatetimeIndex to align to
        q4_lag_days: Lag for Q4/December data (default 70)
        other_lag_days: Lag for Q1-Q3 data (default 40)

    Returns:
        Series aligned to dates with proper lag applied
    """
    min_valid_date = pd.Timestamp('2000-01-01')
    max_valid_date = pd.Timestamp('2030-12-31')

    effective_index = []
    effective_values = []

    for ts in series.index:
        try:
            ts_stamp = pd.Timestamp(ts)
            if ts_stamp < min_valid_date or ts_stamp > max_valid_date:
                continue
        except Exception:
            continue

        # Q4 (December) has longer lag due to annual audit requirements
        # Q1 (March), Q2 (June), Q3 (September) have shorter lag
        if ts.month == 12:
            lag_days = q4_lag_days
        else:
            lag_days = other_lag_days

        try:
            effective_date = (ts_stamp + pd.Timedelta(days=lag_days)).normalize()
            effective_index.append(effective_date)
            effective_values.append(series[ts])
        except Exception:
            continue

    if effective_index:
        effective = pd.Series(effective_values, index=pd.DatetimeIndex(effective_index)).sort_index()
        effective = effective[~effective.index.duplicated(keep="last")]
        return effective.reindex(dates, method="ffill")

    return pd.Series(dtype=float, index=dates)


# ═══════════════════════════════════════════════════════════════════════════════
# STALENESS WEIGHTING (Part D)
# ═══════════════════════════════════════════════════════════════════════════════

_STALENESS_WEIGHTS_CACHE: dict[str, float] | None = None


def load_staleness_weights(
    weight_file: Path | str | None = None,
) -> dict[str, float]:
    """
    Load per-ticker staleness weights.

    Weights range from 0.0 (no data) to 1.0 (fresh ≤60 days).
    Generated by fetch_integrate_fundamentals.py pipeline.
    """
    global _STALENESS_WEIGHTS_CACHE
    if _STALENESS_WEIGHTS_CACHE is not None:
        return _STALENESS_WEIGHTS_CACHE

    if weight_file is None:
        candidates = [
            Path(__file__).resolve().parent.parent.parent / "data" / "fundamental_staleness_weights.json",
        ]
        weight_file = next((f for f in candidates if f.exists()), None)

    if weight_file and Path(weight_file).exists():
        with open(weight_file) as f:
            _STALENESS_WEIGHTS_CACHE = json.load(f)
        return _STALENESS_WEIGHTS_CACHE

    return {}


def apply_staleness_weighting(
    signal_panel: pd.DataFrame,
    weights: dict[str, float] | None = None,
    min_weight: float = 0.1,
) -> pd.DataFrame:
    """
    Apply staleness-based down-weighting to a fundamental signal panel.

    Tickers with stale fundamentals (>60 days) have their signal scores
    scaled down, reducing their influence on ranking.

    Staleness thresholds:
      - ≤60 days:  full weight (1.0)
      - 60-120d:   linear decay → 0.2
      - ≥120 days: minimal weight (0.1)

    Args:
        signal_panel: DataFrame (dates × tickers) with signal scores
        weights: Dict mapping ticker → weight [0, 1]. If None, auto-loads.
        min_weight: Floor for weights (default 0.1)

    Returns:
        Weighted signal panel (same shape)
    """
    if weights is None:
        weights = load_staleness_weights()

    if not weights:
        return signal_panel  # No weights available, pass through

    weighted = signal_panel.copy()
    for ticker in weighted.columns:
        w = weights.get(ticker, 1.0)
        w = max(w, min_weight)
        if w < 1.0:
            weighted[ticker] = weighted[ticker] * w

    return weighted


def compute_staleness_for_ticker(
    series: pd.Series,
    reference_date: pd.Timestamp | None = None,
) -> int | None:
    """
    Compute staleness in days for a fundamental time series.

    Args:
        series: Fundamental data series (index = quarter end dates)
        reference_date: Date to measure staleness from (default: today)

    Returns:
        Staleness in days, or None if series is empty
    """
    if series.empty:
        return None
    if reference_date is None:
        reference_date = pd.Timestamp.now().normalize()
    latest = pd.Timestamp(series.index.max())
    return (reference_date - latest).days


# ============================================================================
# PANEL VALIDATION UTILITIES
# ============================================================================

def validate_reference_axes(
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
    context: str,
) -> Tuple[pd.DatetimeIndex, pd.Index]:
    """
    Validate reference axes used by all factor panels.

    Ensures dates and tickers meet the panel contract:
    - Non-empty
    - No NaT/duplicates in dates
    - Monotonic increasing dates
    - No duplicate tickers

    Args:
        dates: DatetimeIndex to validate
        tickers: Index of ticker symbols to validate
        context: Context string for error messages

    Returns:
        Tuple of (normalized_dates, normalized_tickers)

    Raises:
        ValueError: If validation fails
    """
    normalized_dates = pd.DatetimeIndex(pd.to_datetime(pd.Index(dates), errors="coerce"))
    if len(normalized_dates) == 0:
        raise ValueError(f"{context}: dates index is empty")
    if normalized_dates.hasnans:
        raise ValueError(f"{context}: dates index contains NaT")
    if normalized_dates.has_duplicates:
        raise ValueError(f"{context}: dates index contains duplicates")
    if not normalized_dates.is_monotonic_increasing:
        raise ValueError(f"{context}: dates index must be monotonic increasing")

    normalized_tickers = pd.Index(tickers)
    if len(normalized_tickers) == 0:
        raise ValueError(f"{context}: ticker index is empty")
    if normalized_tickers.has_duplicates:
        duplicate_tickers = normalized_tickers[normalized_tickers.duplicated()].unique().tolist()[:5]
        raise ValueError(f"{context}: ticker index contains duplicates (sample={duplicate_tickers})")

    return normalized_dates, normalized_tickers


def validate_numeric_panel(panel: pd.DataFrame, panel_name: str) -> pd.DataFrame:
    """
    Validate raw panel structure before alignment.

    Ensures panel meets the numeric panel contract:
    - Is a DataFrame
    - Has valid DatetimeIndex (no NaT, no duplicates, monotonic)
    - No duplicate columns
    - No object dtype columns
    - Can be cast to float

    Args:
        panel: DataFrame to validate
        panel_name: Name for error messages

    Returns:
        Validated panel cast to float

    Raises:
        TypeError: If panel is not a DataFrame or has object columns
        ValueError: If index/columns violate contract
    """
    if panel is None or not isinstance(panel, pd.DataFrame):
        raise TypeError(f"{panel_name}: expected pandas.DataFrame")

    normalized = panel.copy()
    normalized.index = pd.DatetimeIndex(pd.to_datetime(normalized.index, errors="coerce"))
    if normalized.index.hasnans:
        raise ValueError(f"{panel_name}: index contains NaT values")
    if normalized.index.has_duplicates:
        raise ValueError(f"{panel_name}: index contains duplicate dates")
    if not normalized.index.is_monotonic_increasing:
        raise ValueError(f"{panel_name}: index must be monotonic increasing")
    if normalized.columns.has_duplicates:
        duplicate_cols = normalized.columns[normalized.columns.duplicated()].unique().tolist()[:5]
        raise ValueError(f"{panel_name}: duplicate ticker columns found (sample={duplicate_cols})")

    object_cols = normalized.select_dtypes(include=["object"]).columns.tolist()
    if object_cols:
        raise TypeError(f"{panel_name}: object dtype columns are not allowed (sample={object_cols[:5]})")

    try:
        return normalized.astype(float)
    except Exception as exc:
        raise TypeError(f"{panel_name}: cannot cast panel to float ({exc})") from exc


def align_numeric_panel(
    panel: pd.DataFrame,
    panel_name: str,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
) -> pd.DataFrame:
    """
    Align panel to contract axes and reject object-typed payloads.

    Args:
        panel: DataFrame to align
        panel_name: Name for error messages and debug logging
        dates: Target DatetimeIndex
        tickers: Target ticker columns

    Returns:
        Aligned panel reindexed to (dates, tickers)
    """
    if panel is None or not isinstance(panel, pd.DataFrame):
        raise TypeError(f"{panel_name}: expected pandas.DataFrame")

    aligned = panel.copy()
    aligned.index = pd.DatetimeIndex(pd.to_datetime(aligned.index, errors="coerce"))
    if aligned.index.hasnans:
        raise ValueError(f"{panel_name}: index contains NaT")
    if aligned.index.has_duplicates:
        raise ValueError(f"{panel_name}: index contains duplicate dates")
    if not aligned.index.is_monotonic_increasing:
        raise ValueError(f"{panel_name}: index must be monotonic increasing")
    if aligned.columns.has_duplicates:
        duplicate_cols = aligned.columns[aligned.columns.duplicated()].unique().tolist()[:5]
        raise ValueError(f"{panel_name}: duplicate ticker columns found (sample={duplicate_cols})")

    object_cols = aligned.select_dtypes(include=["object"]).columns.tolist()
    if object_cols:
        raise TypeError(f"{panel_name}: object dtype columns are not allowed (sample={object_cols[:5]})")

    aligned = aligned.reindex(index=dates, columns=tickers)
    try:
        aligned = aligned.astype(float)
    except Exception as exc:
        raise TypeError(f"{panel_name}: cannot cast panel to float ({exc})") from exc

    debug_log(
        f"{panel_name}: shape={aligned.shape[0]}x{aligned.shape[1]}, "
        f"non_na={int(aligned.notna().sum().sum())}",
        prefix="PANEL"
    )
    return aligned


def align_panel_to_contract(
    panel: pd.DataFrame,
    panel_name: str,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
) -> pd.DataFrame:
    """
    Validate and align panel to contract axes (convenience wrapper).

    Args:
        panel: DataFrame to align
        panel_name: Name for error messages and debug logging
        dates: Target DatetimeIndex
        tickers: Target ticker columns

    Returns:
        Validated and aligned panel
    """
    validated = validate_numeric_panel(panel, panel_name)
    aligned = validated.reindex(index=dates, columns=tickers)
    debug_log(
        f"{panel_name}: shape={aligned.shape[0]}x{aligned.shape[1]}, "
        f"non_na={int(aligned.notna().sum().sum())}",
        prefix="PANEL"
    )
    return aligned


# ============================================================================
# CROSS-SECTIONAL TRANSFORMATIONS
# ============================================================================

MIN_ROLLING_OBS_RATIO = 0.5


def cross_sectional_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional z-score by date.

    For each row (date), compute z-score across all tickers:
    z = (x - mean) / std

    Args:
        panel: DataFrame with dates as index, tickers as columns

    Returns:
        Z-scored panel (same shape)
    """
    panel = validate_numeric_panel(panel, "cross_sectional_zscore_input")
    row_mean = panel.mean(axis=1)
    row_std = panel.std(axis=1).replace(0.0, np.nan)
    return panel.sub(row_mean, axis=0).div(row_std, axis=0)


def cross_sectional_rank(
    panel: pd.DataFrame,
    higher_is_better: bool = True
) -> pd.DataFrame:
    """
    Cross-sectional percentile rank by date, scaled to 0-100.

    Uses 'average' method for ties, producing ranks in (0, 100].
    The lowest ranked stock gets a small positive value, not 0.

    Args:
        panel: DataFrame with dates as index, tickers as columns
        higher_is_better: If True, higher values get higher ranks

    Returns:
        Percentile rank panel scaled to 0-100
    """
    panel = validate_numeric_panel(panel, "cross_sectional_rank_input")
    ranks = panel.rank(axis=1, pct=True, method="average")
    if not higher_is_better:
        ranks = 1.0 - ranks
    return (ranks * 100.0).where(panel.notna())


def rolling_cumulative_return(
    daily_returns: pd.Series,
    lookback: int,
    min_obs: int | None = None
) -> pd.Series:
    """
    Compute rolling compounded returns using log-sum for numerical stability.

    Missing days are excluded from the calculation (not treated as 0% return).
    Uses log1p/expm1 for numerical stability with small returns.

    Args:
        daily_returns: Series of daily returns
        lookback: Rolling window size in days
        min_obs: Minimum observations required (default: lookback * 0.5)

    Returns:
        Series of rolling compounded returns
    """
    if min_obs is None:
        min_obs = max(int(lookback * MIN_ROLLING_OBS_RATIO), 10)
    # Clip extreme negative returns to prevent log of negative numbers
    clipped = daily_returns.clip(lower=-0.99)
    log_growth = np.log1p(clipped)
    # Use min_periods to handle missing data properly (NaNs don't contribute)
    roll_log_sum = log_growth.rolling(lookback, min_periods=min_obs).sum()
    return np.expm1(roll_log_sum)
