"""
Staleness-aware fundamental signal weighting.

Computes per-ticker, per-date staleness and applies decay weights
to fundamental-based signals so that stale data is progressively
down-weighted rather than used at full strength.

Thresholds:
    <= 60 days:  full weight (1.0)
    60–120 days: linear decay from 1.0 → 0.1
    >= 120 days: minimal weight (0.1) — fundamental component
                 effectively disabled

Usage in signal builders:
    from bist_quant.common.staleness import apply_staleness_decay

    signals = build_my_signal(...)
    signals = apply_staleness_decay(signals, fundamentals_parquet, data_loader)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

# Default thresholds (days)
FULL_WEIGHT_DAYS = 60
DECAY_END_DAYS = 120
MIN_WEIGHT = 0.1


def compute_fundamental_staleness_panel(
    parquet_path: Path | str | None = None,
    parquet_df: pd.DataFrame | None = None,
    dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """
    Build a (date × ticker) panel of staleness in days.

    For each ticker, identifies the latest non-NaN period column,
    converts it to a Timestamp (period-end), and computes
    staleness = date - period_end for every date in `dates`.

    Returns:
        DataFrame (dates × tickers) with staleness in days (float).
    """
    if parquet_df is None:
        if parquet_path is None:
            raise ValueError("Either parquet_path or parquet_df must be provided")
        parquet_df = pd.read_parquet(parquet_path)

    # Parse period columns → Timestamps
    period_cols = [c for c in parquet_df.columns if "/" in str(c)]
    period_map: dict[str, pd.Timestamp] = {}
    for c in period_cols:
        try:
            y, m = str(c).split("/")
            ts = pd.Timestamp(year=int(y), month=int(m), day=1) + pd.offsets.MonthEnd(0)
            period_map[c] = ts
        except Exception:
            pass

    if not period_map:
        return pd.DataFrame(dtype=float)

    # For each ticker, find the latest non-NaN period
    ticker_level = parquet_df.index.get_level_values("ticker")
    any_non_na = parquet_df[list(period_map.keys())].notna().groupby(ticker_level).any()

    ticker_latest: dict[str, pd.Timestamp] = {}
    for tk, row in any_non_na.iterrows():
        valid_cols = [c for c in period_map if bool(row.get(c, False))]
        if valid_cols:
            latest_col = max(valid_cols, key=lambda c: period_map[c])
            ticker_latest[tk] = period_map[latest_col]

    if not ticker_latest:
        return pd.DataFrame(dtype=float)

    if dates is None:
        # Generate a date range from earliest price date to today
        dates = pd.date_range("2013-01-01", pd.Timestamp.now().normalize(), freq="B")

    # Build staleness panel
    tickers = sorted(ticker_latest.keys())
    staleness = pd.DataFrame(np.nan, index=dates, columns=tickers)

    for tk, latest_ts in ticker_latest.items():
        if tk in staleness.columns:
            staleness[tk] = (dates - latest_ts).days.astype(float)

    return staleness


def staleness_decay_weight(staleness_days: float) -> float:
    """
    Compute decay weight for a single staleness value.

    Returns:
        Weight in [MIN_WEIGHT, 1.0]
    """
    if pd.isna(staleness_days) or staleness_days <= FULL_WEIGHT_DAYS:
        return 1.0
    if staleness_days >= DECAY_END_DAYS:
        return MIN_WEIGHT
    # Linear interpolation
    frac = (staleness_days - FULL_WEIGHT_DAYS) / (DECAY_END_DAYS - FULL_WEIGHT_DAYS)
    return 1.0 - frac * (1.0 - MIN_WEIGHT)


def compute_decay_panel(staleness_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Apply decay function element-wise to a staleness panel.

    Returns:
        DataFrame (dates × tickers) with weights in [MIN_WEIGHT, 1.0].
    """
    return staleness_panel.applymap(staleness_decay_weight)


def apply_staleness_decay(
    signal_panel: pd.DataFrame,
    parquet_path: Path | str | None = None,
    parquet_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Apply staleness decay weights to a signal panel.

    Fundamental signals are multiplied by the decay weight so that
    stale fundamentals contribute less to the composite score.

    Args:
        signal_panel: DataFrame (dates × tickers) with raw signal scores.
        parquet_path: Path to consolidated fundamentals parquet.
        parquet_df: Pre-loaded parquet DataFrame (avoids re-read).

    Returns:
        DataFrame with staleness-decayed signals (same shape).
    """
    staleness = compute_fundamental_staleness_panel(
        parquet_path=parquet_path,
        parquet_df=parquet_df,
        dates=signal_panel.index,
    )

    if staleness.empty:
        return signal_panel

    # Compute decay weights
    weights = compute_decay_panel(staleness)

    # Align columns
    common_tickers = signal_panel.columns.intersection(weights.columns)
    if common_tickers.empty:
        return signal_panel

    # Apply decay
    decayed = signal_panel.copy()
    for tk in common_tickers:
        if tk in weights.columns and tk in decayed.columns:
            w = weights[tk].reindex(decayed.index).fillna(1.0)
            decayed[tk] = decayed[tk] * w

    return decayed


def staleness_summary(
    parquet_path: Path | str | None = None,
    parquet_df: pd.DataFrame | None = None,
    as_of: pd.Timestamp | None = None,
) -> dict:
    """
    Compute summary staleness statistics.

    Returns:
        Dict with aggregate metrics.
    """
    if as_of is None:
        as_of = pd.Timestamp.now().normalize()

    staleness = compute_fundamental_staleness_panel(
        parquet_path=parquet_path,
        parquet_df=parquet_df,
        dates=pd.DatetimeIndex([as_of]),
    )

    if staleness.empty:
        return {"total_tickers": 0}

    s = staleness.iloc[0].dropna()

    return {
        "as_of": str(as_of.date()),
        "total_tickers": len(s),
        "staleness_median_days": float(s.median()),
        "staleness_mean_days": float(s.mean()),
        "staleness_min_days": float(s.min()),
        "staleness_max_days": float(s.max()),
        "pct_leq_60d": float((s <= 60).mean() * 100),
        "pct_leq_90d": float((s <= 90).mean() * 100),
        "pct_leq_120d": float((s <= 120).mean() * 100),
        "pct_gt_120d": float((s > 120).mean() * 100),
        "count_leq_60d": int((s <= 60).sum()),
        "count_leq_90d": int((s <= 90).sum()),
        "count_gt_120d": int((s > 120).sum()),
    }
