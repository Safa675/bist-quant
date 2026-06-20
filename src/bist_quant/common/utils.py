"""Common utilities for signal construction

This module provides centralized utility functions for:
- Signal panel validation and schema enforcement
- Cross-sectional transformations (z-score, rank)
- Fundamental data helpers (TTM, lag, quarter coercion) -- re-exported from
  ``fundamental_utils`` for backwards compatibility
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

from bist_quant.common.fundamental_utils import (
    _turkish_match,
    apply_lag,
    coerce_quarter_cols,
    get_consolidated_sheet,
    pick_row,
    pick_row_from_sheet,
    sum_ttm,
)
from bist_quant.common.panel_validation import (
    _check_datetime_index,
    _check_no_duplicate_columns,
    _check_no_object_cols,
)

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
    def _err(msg: str) -> None:
        raise_signal_data_error(signal_name, msg)

    if panel is None or not isinstance(panel, pd.DataFrame):
        _err(f"{context}: expected DataFrame, got {type(panel).__name__}")

    ref_dates = pd.DatetimeIndex(pd.to_datetime(pd.Index(dates), errors="coerce"))
    if len(ref_dates) == 0:
        _err(f"{context}: reference dates are empty")
    _check_datetime_index(ref_dates, f"{context}: reference dates", _err)

    ref_tickers = pd.Index(tickers)
    if len(ref_tickers) == 0:
        _err(f"{context}: reference tickers are empty")
    if ref_tickers.has_duplicates:
        dup = ref_tickers[ref_tickers.duplicated()].unique().tolist()[:5]
        _err(f"{context}: duplicate reference tickers (sample={dup})")

    out = panel.copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index, errors="coerce"))
    _check_datetime_index(out.index, f"{context}: panel index", _err)
    _check_no_duplicate_columns(out, context, _err)
    _check_no_object_cols(out, context, _err)

    out = out.reindex(index=ref_dates, columns=ref_tickers)
    target_dtype = np.dtype(dtype if dtype is not None else np.float64)
    try:
        out = out.astype(target_dtype)
    except Exception as exc:
        _err(f"{context}: cannot cast to {target_dtype} ({exc})")

    return out


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


# ============================================================================
# PANEL VALIDATION UTILITIES
# ============================================================================


def _value_error(msg: str) -> None:
    raise ValueError(msg)


def _type_error(msg: str) -> None:
    raise TypeError(msg)


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
    _check_datetime_index(normalized_dates, f"{context}: dates index", _value_error)

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
    _check_datetime_index(normalized.index, panel_name, _value_error)
    _check_no_duplicate_columns(normalized, panel_name, _value_error)
    _check_no_object_cols(normalized, panel_name, _type_error)

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
    validated = validate_numeric_panel(panel, panel_name)
    aligned = validated.reindex(index=dates, columns=tickers)
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
