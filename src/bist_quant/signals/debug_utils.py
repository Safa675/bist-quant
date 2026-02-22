from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _debug_log(debug: bool, message: str) -> None:
    """Emit debug log message when detailed tracing is enabled."""
    if debug:
        logger.info(f"  [FIVE_FACTOR_DEBUG] {message}")


def _debug_panel_stats(debug: bool, name: str, panel: pd.DataFrame | None) -> None:
    """Print compact panel diagnostics."""
    if not debug:
        return
    if panel is None:
        _debug_log(debug, f"{name}: missing")
        return
    if panel.empty:
        _debug_log(debug, f"{name}: empty")
        return

    n_dates, n_tickers = panel.shape
    non_na = int(panel.notna().sum().sum())
    total = max(n_dates * n_tickers, 1)
    coverage = non_na / total
    latest_non_na = int(panel.iloc[-1].notna().sum()) if n_dates else 0

    valid_date_mask = panel.notna().any(axis=1).to_numpy()
    if valid_date_mask.any():
        valid_idx = np.flatnonzero(valid_date_mask)
        first_date = panel.index[valid_idx[0]].date()
        last_date = panel.index[valid_idx[-1]].date()
        span = f"{first_date}..{last_date}"
    else:
        span = "n/a"

    _debug_log(
        debug,
        f"{name}: shape={n_dates}x{n_tickers}, non_na={non_na}, coverage={coverage:.2%}, "
        f"latest_non_na={latest_non_na}, span={span}",
    )


def _debug_axis_component_stats(
    debug: bool,
    axis_name: str,
    component: pd.DataFrame,
    winning_bucket: pd.Series,
    high_daily: pd.Series,
    low_daily: pd.Series,
) -> None:
    """Print diagnostics for one computed axis component."""
    if not debug:
        return

    latest_scores = component.iloc[-1].dropna() if not component.empty else pd.Series(dtype=float)
    top_names = ", ".join(latest_scores.nlargest(3).index.tolist()) if not latest_scores.empty else "n/a"
    bottom_names = ", ".join(latest_scores.nsmallest(3).index.tolist()) if not latest_scores.empty else "n/a"

    non_na_buckets = winning_bucket.dropna()
    latest_bucket = non_na_buckets.iloc[-1] if not non_na_buckets.empty else "n/a"

    spread = (high_daily - low_daily).dropna()
    spread_mean = float(spread.mean()) if not spread.empty else np.nan
    spread_std = float(spread.std()) if not spread.empty else np.nan

    _debug_log(
        debug,
        f"axis={axis_name}: latest_bucket={latest_bucket}, spread_mean={spread_mean:.4%}, "
        f"spread_std={spread_std:.4%}, top={top_names}, bottom={bottom_names}",
    )
