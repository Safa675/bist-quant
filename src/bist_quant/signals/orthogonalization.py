from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from bist_quant.signals.factor_axes import cross_sectional_zscore

# Optional axis orthogonalization (cross-sectional, date-by-date)
DEFAULT_ORTHOGONALIZE_AXES = False
DEFAULT_ORTHOG_MIN_OVERLAP = 20
DEFAULT_ORTHOG_EPSILON = 1e-8


def _mean_abs_pairwise_corr(
    matrix: np.ndarray,
    min_overlap: int,
    epsilon: float,
) -> float:
    """Mean absolute pairwise correlation across axis columns for one date."""
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        return np.nan

    n_axes = matrix.shape[1]
    corr_sum = 0.0
    corr_count = 0

    for left in range(n_axes):
        x = matrix[:, left]
        for right in range(left + 1, n_axes):
            y = matrix[:, right]
            overlap = np.isfinite(x) & np.isfinite(y)
            if int(overlap.sum()) < min_overlap:
                continue

            xv = x[overlap]
            yv = y[overlap]
            xv = xv - xv.mean()
            yv = yv - yv.mean()

            denom = float(np.sqrt(np.dot(xv, xv) * np.dot(yv, yv)))
            if not np.isfinite(denom) or denom <= epsilon:
                continue

            corr = float(np.dot(xv, yv) / denom)
            corr_sum += abs(corr)
            corr_count += 1

    if corr_count == 0:
        return np.nan
    return corr_sum / corr_count


def _orthogonalize_axis_raw_scores(
    axis_raw_map: Dict[str, pd.DataFrame],
    axis_order: list[str],
    min_overlap: int = DEFAULT_ORTHOG_MIN_OVERLAP,
    epsilon: float = DEFAULT_ORTHOG_EPSILON,
) -> tuple[Dict[str, pd.DataFrame], Dict[str, object]]:
    """Orthogonalize raw axis scores cross-sectionally date-by-date."""
    if not axis_order:
        return axis_raw_map, {}

    first_panel = axis_raw_map[axis_order[0]]
    dates = first_panel.index
    tickers = first_panel.columns
    n_dates = len(dates)
    n_tickers = len(tickers)
    n_axes = len(axis_order)

    standardized_arrays: list[np.ndarray] = []
    for axis_name in axis_order:
        panel = axis_raw_map[axis_name].reindex(index=dates, columns=tickers).astype(float)
        standardized_arrays.append(cross_sectional_zscore(panel).to_numpy(dtype=float))

    orth_arrays = [np.full((n_dates, n_tickers), np.nan, dtype=float) for _ in axis_order]
    raw_daily_corr = np.full(n_dates, np.nan, dtype=float)
    orth_daily_corr = np.full(n_dates, np.nan, dtype=float)

    for date_idx in range(n_dates):
        raw_day = np.column_stack([arr[date_idx, :] for arr in standardized_arrays])
        raw_daily_corr[date_idx] = _mean_abs_pairwise_corr(raw_day, min_overlap, epsilon)

        orth_day = np.full((n_tickers, n_axes), np.nan, dtype=float)

        for axis_idx in range(n_axes):
            residual = raw_day[:, axis_idx].copy()
            if int(np.isfinite(residual).sum()) < 2:
                continue

            for prev_idx in range(axis_idx):
                prev_axis = orth_day[:, prev_idx]
                overlap = np.isfinite(residual) & np.isfinite(prev_axis)
                if int(overlap.sum()) < min_overlap:
                    continue

                x = prev_axis[overlap]
                y = residual[overlap]
                denom = float(np.dot(x, x))
                if not np.isfinite(denom) or denom <= epsilon:
                    continue

                beta = float(np.dot(x, y) / denom)
                residual[overlap] = y - beta * x

            valid = np.isfinite(residual)
            n_valid = int(valid.sum())
            if n_valid < 2:
                continue

            centered = residual[valid] - residual[valid].mean()
            std = float(centered.std(ddof=1))
            if not np.isfinite(std) or std <= epsilon:
                continue

            normalized = centered / std

            # Keep orientation stable versus original axis direction.
            alignment = float(np.dot(raw_day[valid, axis_idx], normalized))
            if np.isfinite(alignment) and alignment < 0:
                normalized = -normalized

            orth_col = np.full(n_tickers, np.nan, dtype=float)
            orth_col[valid] = normalized
            orth_day[:, axis_idx] = orth_col

        orth_daily_corr[date_idx] = _mean_abs_pairwise_corr(orth_day, min_overlap, epsilon)

        for axis_idx in range(n_axes):
            orth_arrays[axis_idx][date_idx, :] = orth_day[:, axis_idx]

    orthogonalized = {
        axis_name: pd.DataFrame(orth_arrays[idx], index=dates, columns=tickers)
        for idx, axis_name in enumerate(axis_order)
    }

    diagnostics = {
        "axis_order": axis_order,
        "raw_daily_mean_abs_corr": pd.Series(raw_daily_corr, index=dates, name="raw_mean_abs_corr"),
        "orth_daily_mean_abs_corr": pd.Series(orth_daily_corr, index=dates, name="orth_mean_abs_corr"),
        "raw_mean_abs_corr": float(np.nanmean(raw_daily_corr)) if np.isfinite(raw_daily_corr).any() else np.nan,
        "orth_mean_abs_corr": float(np.nanmean(orth_daily_corr)) if np.isfinite(orth_daily_corr).any() else np.nan,
    }
    return orthogonalized, diagnostics
