"""
Multi-Factor Rotation Signal Construction

Builds a cross-sectional stock score by combining 13 factor axes:

ORIGINAL AXES:
1. Size: Small vs Big
2. Value Style: Value Level vs Value Growth
3. Profitability Style: Profitable Now vs Future Profitability
4. Investment Style: Conservative vs Reinvestment
5. Momentum: Winner vs Loser
6. Risk: Low Volatility vs High Beta

NEW AXES:
7. Quality: High Quality vs Low Quality (ROE, ROA, Piotroski F-score, accruals)
8. Liquidity: Liquid vs Illiquid (Amihud, real turnover = volume/shares_outstanding)
9. Trading Intensity: High Activity vs Low Activity (relative volume, volume trend, turnover velocity)
10. Sentiment: Strong Price Action vs Weak (52w high proximity, price acceleration)
11. Fundamental Momentum: Improving vs Deteriorating (margin change, sales accel)
12. Carry: High Yield vs Low Yield (dividend yield)
13. Defensive: Stable vs Cyclical (earnings stability, low beta)

Note: Liquidity and Trading Intensity are separate concepts:
- Liquidity = ease of trading without price impact (Amihud illiquidity, spreads)
- Trading Intensity = level of trading activity / market attention (volume trends)

Key features:
- Quintile-based bucket selection (captures center returns, not just edges)
- Multi-lookback ensemble with equal weights (21, 63, 126, 252 days)
- Exponentially-weighted factor selection favoring recent performance
  - 6-month half-life decay on past returns
  - Rank-based weights (rank^2) emphasizing best recent performers

Final score is the weighted average of all axis scores (0-100).
"""

from pathlib import Path
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.utils import (
    apply_lag,
    assert_has_cross_section,
    assert_panel_not_constant,
    coerce_quarter_cols,
    get_consolidated_sheet,
    pick_row,
    pick_row_from_sheet,
    raise_signal_data_error,
    sum_ttm,
)
from signals.investment_signals import build_investment_signals
from signals.small_cap_signals import build_small_cap_signals
from signals.value_signals import build_value_signals

# Import from modular files
from signals.factor_builders import (
    build_quality_panels,
    build_liquidity_panels,
    build_trading_intensity_panels,
    build_sentiment_panels,
    build_fundamental_momentum_panels,
    build_carry_panels,
    build_defensive_panels,
    build_volatility_beta_panels,
    build_realized_volatility_panel,
    build_market_beta_panel,
    INCOME_SHEET,
    REVENUE_KEYS,
    OPERATING_INCOME_KEYS,
    GROSS_PROFIT_KEYS,
)
from signals.factor_axes import (
    cross_sectional_zscore,
    cross_sectional_rank,
    rolling_cumulative_return,
    combine_quality_axis,
    combine_liquidity_axis,
    combine_trading_intensity_axis,
    combine_sentiment_axis,
    combine_fundmom_axis,
    combine_carry_axis,
    combine_defensive_axis,
    quintile_bucket_selection,
    compute_quintile_axis_scores,
    compute_mwu_weights,
    N_BUCKETS,
    BUCKET_LABELS,
    ENSEMBLE_LOOKBACK_WINDOWS,
    ENSEMBLE_LOOKBACK_WEIGHTS,
    MIN_ROLLING_OBS_RATIO,
)


# ============================================================================
# PARAMETERS
# ============================================================================

# Legacy single lookback (used only for binary-split fallback)
ROTATION_LOOKBACK_DAYS = 126
MIN_ROLLING_OBS = 63

# Momentum
MOMENTUM_LOOKBACK_DAYS = 126
MOMENTUM_SKIP_DAYS = 21
VALUE_GROWTH_LOOKBACK_DAYS = 252

# Quintile settings
USE_QUINTILE_BUCKETS = True

# Optional axis orthogonalization (cross-sectional, date-by-date)
DEFAULT_ORTHOGONALIZE_AXES = False
DEFAULT_ORTHOG_MIN_OVERLAP = 20
DEFAULT_ORTHOG_EPSILON = 1e-8

# Axis cache
AXIS_CACHE_FILENAME = "multi_factor_axis_construction.parquet"
AXIS_PANEL_NAMES = (
    # Original panels
    "size_small_signal",
    "value_level_signal",
    "value_growth_signal",
    "profit_margin_level",
    "profit_margin_growth",
    "investment_reinvestment_signal",
    "realized_volatility",
    "market_beta",
    # New panels
    "quality_roe",
    "quality_roa",
    "quality_accruals",
    "quality_piotroski",
    "liquidity_amihud",
    "liquidity_turnover",
    "liquidity_spread_proxy",
    # Trading intensity (separate from liquidity)
    "trading_intensity_relative_volume",
    "trading_intensity_volume_trend",
    "trading_intensity_turnover_velocity",
    "sentiment_52w_high_pct",
    "sentiment_price_acceleration",
    "sentiment_reversal",
    "fundmom_margin_change",
    "fundmom_sales_accel",
    "carry_dividend_yield",
    "carry_shareholder_yield",
    "defensive_earnings_stability",
    "defensive_beta_to_market",
)

# Optional panels that may legitimately be empty in some environments.
# Keep them in cache if available, but do not treat emptiness as stale/corrupt.
OPTIONAL_EMPTY_CACHE_PANELS = {
    "liquidity_spread_proxy",   # Requires intraday/spread-like inputs we may not have.
    "carry_shareholder_yield",  # Often unavailable in Turkey; carry uses dividend yield.
}

# Required panel groups for rebuild triggers.
REQUIRED_LIQUIDITY_PANELS = (
    "liquidity_amihud",
    "liquidity_turnover",
)
REQUIRED_CARRY_PANELS = (
    "carry_dividend_yield",
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _squash_metric(
    panel: pd.DataFrame,
    scale: float,
    lower: float | None = None,
    upper: float | None = None,
) -> pd.DataFrame:
    """Clip and squash a metric panel to a bounded range via tanh."""
    clipped = panel
    if lower is not None or upper is not None:
        clipped = clipped.clip(lower=lower, upper=upper)
    return np.tanh(clipped / scale)


def _debug_log(debug: bool, message: str) -> None:
    """Emit debug log message when detailed tracing is enabled."""
    if debug:
        print(f"  [FIVE_FACTOR_DEBUG] {message}")


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
    """Orthogonalize raw axis scores cross-sectionally date-by-date.

    Uses sequential residualization (Gram-Schmidt style) in the supplied axis
    order. Earlier axes are preserved; later axes keep only variation not
    explained by earlier axes.
    """
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


def _build_two_sided_axis_raw(
    high_side_panel: pd.DataFrame,
    low_side_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Build one raw axis from two distinct side scores.

    When only one side has data, we use the available side's z-score directly
    (positive for high side, negative for low side) rather than treating
    the missing side as 0.
    """
    high_z = cross_sectional_zscore(high_side_panel)
    low_z = cross_sectional_zscore(low_side_panel)
    high_z, low_z = high_z.align(low_z, join="outer")

    # Handle cases where only one side has data
    high_only = high_z.notna() & low_z.isna()
    low_only = low_z.notna() & high_z.isna()
    both_valid = high_z.notna() & low_z.notna()
    both_missing = high_z.isna() & low_z.isna()

    # When both valid: difference; when one valid: use that side's score
    axis_raw = pd.DataFrame(np.nan, index=high_z.index, columns=high_z.columns)
    axis_raw = axis_raw.where(~both_valid, high_z - low_z)
    axis_raw = axis_raw.where(~high_only, high_z)  # Use high z-score when only high available
    axis_raw = axis_raw.where(~low_only, -low_z)   # Use negative low z-score when only low available

    return axis_raw.where(~both_missing)


def _build_metric_panel(
    metrics_df: pd.DataFrame,
    metric_name: str,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
) -> pd.DataFrame:
    """Convert (ticker, date) metric series into a daily Date x Ticker panel."""
    panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)

    if metrics_df.empty or metric_name not in metrics_df.columns:
        return panel

    available_tickers = set(metrics_df.index.get_level_values(0))
    for ticker in tickers:
        if ticker not in available_tickers:
            continue
        try:
            series = metrics_df.loc[ticker, metric_name]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            series = series.sort_index()
            series = series[~series.index.duplicated(keep="last")]
            series.index = pd.to_datetime(series.index)
            panel[ticker] = apply_lag(series, dates)
        except KeyError:
            # Ticker or metric not found in index
            continue
        except (ValueError, TypeError) as e:
            # Data conversion issues
            print(f"    Warning: Could not process {metric_name} for {ticker}: {e}")
            continue

    return panel


def _calculate_margin_level_growth_for_ticker(
    xlsx_path: Path | None,
    ticker: str,
    fundamentals_parquet: pd.DataFrame | None = None,
) -> tuple[pd.Series | None, pd.Series | None]:
    """Return profitability margin level and YoY margin growth series for one ticker."""
    if fundamentals_parquet is not None:
        inc = get_consolidated_sheet(fundamentals_parquet, ticker, INCOME_SHEET)
        if inc.empty:
            return None, None
        rev_row = pick_row_from_sheet(inc, REVENUE_KEYS)
        op_row = pick_row_from_sheet(inc, OPERATING_INCOME_KEYS)
        gp_row = pick_row_from_sheet(inc, GROSS_PROFIT_KEYS)
    else:
        if xlsx_path is None:
            return None, None
        try:
            inc = pd.read_excel(xlsx_path, sheet_name=INCOME_SHEET)
        except Exception:
            return None, None
        rev_row = pick_row(inc, REVENUE_KEYS)
        op_row = pick_row(inc, OPERATING_INCOME_KEYS)
        gp_row = pick_row(inc, GROSS_PROFIT_KEYS)

    if rev_row is None or op_row is None or gp_row is None:
        return None, None

    rev = coerce_quarter_cols(rev_row)
    op = coerce_quarter_cols(op_row)
    gp = coerce_quarter_cols(gp_row)
    if rev.empty or op.empty or gp.empty:
        return None, None

    rev_ttm = sum_ttm(rev)
    op_ttm = sum_ttm(op)
    gp_ttm = sum_ttm(gp)
    if rev_ttm.empty or op_ttm.empty or gp_ttm.empty:
        return None, None

    combined = pd.concat([rev_ttm, op_ttm, gp_ttm], axis=1, join="inner").dropna()
    if combined.empty:
        return None, None
    combined.columns = ["RevenueTTM", "OperatingIncomeTTM", "GrossProfitTTM"]

    revenue = combined["RevenueTTM"].replace(0.0, np.nan)
    op_margin = combined["OperatingIncomeTTM"] / revenue
    gp_margin = combined["GrossProfitTTM"] / revenue
    # Weighted average of operating and gross margins (weights configurable)
    OP_MARGIN_WEIGHT = 0.60
    GP_MARGIN_WEIGHT = 0.40
    margin_level = (OP_MARGIN_WEIGHT * op_margin + GP_MARGIN_WEIGHT * gp_margin).replace([np.inf, -np.inf], np.nan).dropna()
    if margin_level.empty:
        return None, None

    # YoY margin growth (diff of 4 quarters)
    # Note: This assumes quarterly data frequency
    margin_growth = margin_level.diff(4).replace([np.inf, -np.inf], np.nan).dropna()
    return margin_level.sort_index(), margin_growth.sort_index() if not margin_growth.empty else None


def _build_profitability_margin_panels(
    fundamentals: Dict,
    dates: pd.DatetimeIndex,
    data_loader,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build lagged margin-level and margin-growth panels."""
    fundamentals_parquet = data_loader.load_fundamentals_parquet() if data_loader is not None else None
    level_panel: Dict[str, pd.Series] = {}
    growth_panel: Dict[str, pd.Series] = {}

    count = 0
    for ticker, fund_data in fundamentals.items():
        xlsx_path = fund_data.get("path") if isinstance(fund_data, dict) else None
        margin_level, margin_growth = _calculate_margin_level_growth_for_ticker(
            xlsx_path=xlsx_path,
            ticker=ticker,
            fundamentals_parquet=fundamentals_parquet,
        )

        if margin_level is not None and not margin_level.empty:
            lagged_level = apply_lag(margin_level, dates)
            if not lagged_level.empty:
                level_panel[ticker] = lagged_level

        if margin_growth is not None and not margin_growth.empty:
            lagged_growth = apply_lag(margin_growth, dates)
            if not lagged_growth.empty:
                growth_panel[ticker] = lagged_growth

        count += 1
        if count % 100 == 0:
            print(f"  Profitability margin progress: {count}/{len(fundamentals)}")

    level_df = pd.DataFrame(level_panel, index=dates)
    growth_df = pd.DataFrame(growth_panel, index=dates)
    return level_df, growth_df


def _build_value_level_growth_panels(
    fundamentals: Dict,
    close: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
    data_loader,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build value level and lagged YoY value-growth panels."""
    level_df = build_value_signals(
        fundamentals,
        close,
        dates,
        data_loader,
    ).reindex(index=dates, columns=tickers)

    growth_df = level_df.diff(VALUE_GROWTH_LOOKBACK_DAYS)
    growth_df = growth_df.replace([np.inf, -np.inf], np.nan)
    return level_df, growth_df


# ============================================================================
# AXIS CACHE MANAGEMENT
# ============================================================================

def _resolve_axis_cache_path(data_loader, cache_path: Path | str | None = None) -> Path:
    """Resolve on-disk path for axis construction cache parquet."""
    if cache_path is not None:
        return Path(cache_path)
    if data_loader is not None and hasattr(data_loader, "data_dir"):
        return Path(data_loader.data_dir) / AXIS_CACHE_FILENAME
    return Path(__file__).resolve().parents[2] / "data" / AXIS_CACHE_FILENAME


def _load_axis_construction_cache(
    cache_path: Path,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
) -> Dict[str, pd.DataFrame]:
    """Load cached axis construction panels from parquet."""
    if not cache_path.exists():
        return {}

    try:
        cached = pd.read_parquet(cache_path)
    except Exception as exc:
        print(f"  ⚠️  Failed to read axis cache {cache_path}: {exc}")
        return {}

    if cached.empty:
        return {}

    if not isinstance(cached.columns, pd.MultiIndex):
        try:
            cached.columns = pd.MultiIndex.from_tuples(cached.columns)
        except Exception:
            print(f"  ⚠️  Axis cache columns are not multi-indexed: {cache_path}")
            return {}

    lvl0 = cached.columns.get_level_values(0)
    loaded: Dict[str, pd.DataFrame] = {}
    for panel_name in AXIS_PANEL_NAMES:
        if panel_name not in lvl0:
            continue
        panel = cached[panel_name]
        panel.index = pd.to_datetime(panel.index)
        panel.columns = pd.Index(panel.columns).astype(str)
        panel = panel.reindex(index=dates, columns=tickers)
        loaded[panel_name] = panel

    return loaded


def _save_axis_construction_cache(
    cache_path: Path,
    panels: Dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
) -> None:
    """Persist axis construction panels to parquet."""
    to_save: Dict[str, pd.DataFrame] = {}
    for panel_name in AXIS_PANEL_NAMES:
        panel = panels.get(panel_name)
        if panel is None or panel.empty:
            continue
        to_save[panel_name] = panel.reindex(index=dates, columns=tickers).astype(float)

    if not to_save:
        return

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    wide = pd.concat(to_save, axis=1)
    wide.to_parquet(cache_path)
    print(f"  💾 Saved axis construction cache: {cache_path}")


def _cache_panel_stale_reason(panel_name: str, panel: pd.DataFrame | None) -> str | None:
    """Return reason string when cached panel looks stale/corrupt, else None."""
    if panel is None:
        return "missing"
    if panel.empty:
        return "empty_shape"

    # Some panels are optional and may be all-NaN by design.
    if panel_name in OPTIONAL_EMPTY_CACHE_PANELS:
        return None

    non_na_total = int(panel.notna().sum().sum())
    if non_na_total == 0:
        return "all_nan"

    # If historical values exist but latest row is fully empty, cache is stale
    # relative to current date/ticker universe.
    latest_non_na = int(panel.iloc[-1].notna().sum()) if len(panel.index) > 0 else 0
    if latest_non_na == 0:
        return "no_latest_coverage"

    return None


def _build_axis_construction_panels(
    close: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
    data_loader,
    fundamentals: Dict,
    volume_df: pd.DataFrame,
    use_cache: bool = True,
    force_rebuild_cache: bool = False,
    save_cache: bool = True,
    cache_path: Path | str | None = None,
) -> Dict[str, pd.DataFrame]:
    """Build heavy axis-construction panels with optional parquet cache reuse."""
    resolved_cache_path = _resolve_axis_cache_path(data_loader, cache_path)
    panels: Dict[str, pd.DataFrame] = {}

    if use_cache and not force_rebuild_cache:
        panels = _load_axis_construction_cache(resolved_cache_path, dates, tickers)
        if panels:
            print(f"  📦 Loaded axis construction cache ({len(panels)}/{len(AXIS_PANEL_NAMES)} panels)")
            stale_panel_reasons: Dict[str, str] = {}
            for panel_name in AXIS_PANEL_NAMES:
                panel = panels.get(panel_name)
                reason = _cache_panel_stale_reason(panel_name, panel)
                if reason is not None:
                    stale_panel_reasons[panel_name] = reason

            if stale_panel_reasons:
                for panel_name in sorted(stale_panel_reasons.keys()):
                    panels.pop(panel_name, None)
                pretty = ", ".join(
                    f"{name}({stale_panel_reasons[name]})"
                    for name in sorted(stale_panel_reasons.keys())
                )
                print(f"  ♻️  Rebuilding stale cached panels: {pretty}")

    computed_any = False

    # Original axis panels
    if "size_small_signal" not in panels:
        print("  Building size axis inputs...")
        panels["size_small_signal"] = build_small_cap_signals(
            fundamentals, close, volume_df.reindex(dates), dates, data_loader,
        ).reindex(index=dates, columns=tickers)
        computed_any = True

    need_value = ("value_level_signal" not in panels) or ("value_growth_signal" not in panels)
    if need_value:
        print("  Building value axis inputs...")
        value_level_df, value_growth_df = _build_value_level_growth_panels(
            fundamentals, close, dates, tickers, data_loader,
        )
        panels["value_level_signal"] = value_level_df.reindex(index=dates, columns=tickers)
        panels["value_growth_signal"] = value_growth_df.reindex(index=dates, columns=tickers)
        computed_any = True

    need_profit = ("profit_margin_level" not in panels) or ("profit_margin_growth" not in panels)
    if need_profit:
        print("  Building profitability axis inputs...")
        margin_level_df, margin_growth_df = _build_profitability_margin_panels(fundamentals, dates, data_loader)
        panels["profit_margin_level"] = margin_level_df.reindex(index=dates, columns=tickers)
        panels["profit_margin_growth"] = margin_growth_df.reindex(index=dates, columns=tickers)
        computed_any = True

    if "investment_reinvestment_signal" not in panels:
        print("  Building investment axis inputs...")
        panels["investment_reinvestment_signal"] = build_investment_signals(
            fundamentals, close, dates, data_loader,
        ).reindex(index=dates, columns=tickers)
        computed_any = True

    need_risk = ("realized_volatility" not in panels) or ("market_beta" not in panels)
    if need_risk:
        print("  Building risk axis inputs...")
        vol_panel, beta_panel = build_volatility_beta_panels(close, dates, data_loader)
        panels["realized_volatility"] = vol_panel.reindex(index=dates, columns=tickers)
        panels["market_beta"] = beta_panel.reindex(index=dates, columns=tickers)
        computed_any = True

    # NEW FACTOR PANELS
    need_quality = any(p not in panels for p in ["quality_roe", "quality_roa", "quality_accruals", "quality_piotroski"])
    if need_quality:
        quality_panels = build_quality_panels(fundamentals, close, dates, tickers, data_loader)
        for name, panel in quality_panels.items():
            panels[name] = panel.reindex(index=dates, columns=tickers)
        computed_any = True

    need_liquidity = any(p not in panels for p in REQUIRED_LIQUIDITY_PANELS)
    if need_liquidity:
        liquidity_panels = build_liquidity_panels(close, volume_df, dates, tickers, data_loader)
        for name, panel in liquidity_panels.items():
            panels[name] = panel.reindex(index=dates, columns=tickers)
        computed_any = True

    need_trading_intensity = any(p not in panels for p in [
        "trading_intensity_relative_volume", "trading_intensity_volume_trend", "trading_intensity_turnover_velocity"
    ])
    if need_trading_intensity:
        trading_intensity_panels = build_trading_intensity_panels(close, volume_df, dates, tickers, data_loader)
        for name, panel in trading_intensity_panels.items():
            panels[name] = panel.reindex(index=dates, columns=tickers)
        computed_any = True

    need_sentiment = any(p not in panels for p in ["sentiment_52w_high_pct", "sentiment_price_acceleration", "sentiment_reversal"])
    if need_sentiment:
        sentiment_panels = build_sentiment_panels(close, dates, tickers)
        for name, panel in sentiment_panels.items():
            panels[name] = panel.reindex(index=dates, columns=tickers)
        computed_any = True

    need_fundmom = any(p not in panels for p in ["fundmom_margin_change", "fundmom_sales_accel"])
    if need_fundmom:
        fundmom_panels = build_fundamental_momentum_panels(fundamentals, dates, tickers, data_loader)
        for name, panel in fundmom_panels.items():
            panels[name] = panel.reindex(index=dates, columns=tickers)
        computed_any = True

    need_carry = any(p not in panels for p in REQUIRED_CARRY_PANELS)
    if need_carry:
        carry_panels = build_carry_panels(fundamentals, close, dates, tickers, data_loader)
        for name, panel in carry_panels.items():
            panels[name] = panel.reindex(index=dates, columns=tickers)
        computed_any = True

    need_defensive = any(p not in panels for p in ["defensive_earnings_stability", "defensive_beta_to_market"])
    if need_defensive:
        defensive_panels = build_defensive_panels(fundamentals, close, dates, tickers, data_loader)
        for name, panel in defensive_panels.items():
            panels[name] = panel.reindex(index=dates, columns=tickers)
        computed_any = True

    if save_cache and (computed_any or force_rebuild_cache):
        _save_axis_construction_cache(resolved_cache_path, panels, dates, tickers)

    return panels


# ============================================================================
# AXIS SCORE COMPUTATION
# ============================================================================

def _compute_axis_component(
    axis_raw_scores: pd.DataFrame,
    daily_returns: pd.DataFrame,
    lookback: int = ROTATION_LOOKBACK_DAYS,
    use_quintiles: bool = USE_QUINTILE_BUCKETS,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    """Compute one axis score component by selecting the winning bucket."""
    axis = axis_raw_scores.reindex(daily_returns.index).astype(float)

    if use_quintiles:
        winning_bucket, bucket_cum_df, bucket_daily_df, bucket_masks = quintile_bucket_selection(
            axis_raw_scores=axis,
            daily_returns=daily_returns,
            n_buckets=N_BUCKETS,
        )

        component = compute_quintile_axis_scores(axis, winning_bucket, N_BUCKETS)
        component = component.clip(0.0, 100.0)

        # Track best/worst bucket daily returns (for MWU weighting)
        # IMPORTANT: Shift by 1 to avoid lookahead - we select bucket based on
        # YESTERDAY's cumulative returns, then get TODAY's return of that bucket
        best_bucket_idx = bucket_cum_df.idxmax(axis=1).shift(1).fillna(N_BUCKETS - 1)
        worst_bucket_idx = bucket_cum_df.idxmin(axis=1).shift(1).fillna(0)

        # Vectorized lookup: for each date, get the return of the selected bucket
        # Convert bucket indices to integers for indexing
        best_idx_int = best_bucket_idx.astype(int)
        worst_idx_int = worst_bucket_idx.astype(int)

        # Use numpy advanced indexing for speed
        dates_idx = np.arange(len(bucket_daily_df))
        high_daily = pd.Series(
            bucket_daily_df.values[dates_idx, best_idx_int.values],
            index=bucket_daily_df.index,
            dtype=float
        )
        low_daily = pd.Series(
            bucket_daily_df.values[dates_idx, worst_idx_int.values],
            index=bucket_daily_df.index,
            dtype=float
        )

        return component, winning_bucket, bucket_cum_df, high_daily, low_daily

    else:
        # Binary split (legacy)
        valid = axis.notna()
        median = axis.median(axis=1, skipna=True)
        high_mask = axis.ge(median, axis=0) & valid
        low_mask = axis.lt(median, axis=0) & valid

        high_w = high_mask.shift(1).astype(float)
        low_w = low_mask.shift(1).astype(float)

        high_daily = (daily_returns * high_w).sum(axis=1) / high_w.sum(axis=1).replace(0.0, np.nan)
        low_daily = (daily_returns * low_w).sum(axis=1) / low_w.sum(axis=1).replace(0.0, np.nan)

        high_cum = rolling_cumulative_return(high_daily, lookback)
        low_cum = rolling_cumulative_return(low_daily, lookback)
        winner_is_high = (high_cum >= low_cum).fillna(True)

        rank_high = cross_sectional_rank(axis, higher_is_better=True)
        rank_low = cross_sectional_rank(axis, higher_is_better=False)
        component = rank_high.where(winner_is_high, rank_low)
        component = component.clip(0.0, 100.0)

        bucket_cum_df = pd.DataFrame({"high": high_cum, "low": low_cum})
        return component, winner_is_high, bucket_cum_df, high_daily, low_daily


# ============================================================================
# YEARLY REPORT
# ============================================================================

def _build_yearly_axis_winner_report(
    axis_summary: Dict[str, Tuple],
    axis_daily_returns: Dict[str, Tuple[pd.Series, pd.Series]],
    axis_bucket_returns: Dict[str, pd.DataFrame],
    use_quintiles: bool = USE_QUINTILE_BUCKETS,
) -> pd.DataFrame:
    """Build yearly winner report for each axis."""
    rows = []

    for axis_name, (_, _, high_label, low_label) in axis_summary.items():
        high_daily, low_daily = axis_daily_returns[axis_name]
        bucket_cum_df = axis_bucket_returns[axis_name]

        combined = pd.DataFrame({"high_daily": high_daily, "low_daily": low_daily}).dropna(how="all")
        if combined.empty:
            continue

        combined["year"] = combined.index.year

        for year, grp in combined.groupby("year"):
            high_total = (1.0 + grp["high_daily"].fillna(0.0)).prod() - 1.0
            low_total = (1.0 + grp["low_daily"].fillna(0.0)).prod() - 1.0

            row_data = {
                "Year": int(year),
                "Axis": axis_name,
                "High_Side": high_label,
                "Low_Side": low_label,
                "High_Side_Return": high_total,
                "Low_Side_Return": low_total,
            }

            if use_quintiles and isinstance(bucket_cum_df.columns[0], int):
                year_mask = bucket_cum_df.index.year == year
                if year_mask.any():
                    year_end_rets = bucket_cum_df[year_mask].iloc[-1]
                    best_bucket = year_end_rets.idxmax()
                    worst_bucket = year_end_rets.idxmin()
                    row_data["Winner"] = BUCKET_LABELS[best_bucket] if best_bucket < len(BUCKET_LABELS) else f"Q{best_bucket+1}"
                    row_data["Winner_Return"] = float(year_end_rets[best_bucket])
                    row_data["Loser"] = BUCKET_LABELS[worst_bucket] if worst_bucket < len(BUCKET_LABELS) else f"Q{worst_bucket+1}"
                    row_data["Loser_Return"] = float(year_end_rets[worst_bucket])
                    for i in range(N_BUCKETS):
                        if i in year_end_rets.index:
                            row_data[f"Q{i+1}_Return"] = float(year_end_rets[i])
                else:
                    row_data["Winner"] = "N/A"
                    row_data["Winner_Return"] = 0.0
                    row_data["Loser"] = "N/A"
                    row_data["Loser_Return"] = 0.0
            else:
                if high_total >= low_total:
                    row_data["Winner"] = high_label
                    row_data["Winner_Return"] = high_total
                    row_data["Loser"] = low_label
                    row_data["Loser_Return"] = low_total
                else:
                    row_data["Winner"] = low_label
                    row_data["Winner_Return"] = low_total
                    row_data["Loser"] = high_label
                    row_data["Loser_Return"] = high_total

            row_data["Spread_Winner_Minus_Loser"] = row_data["Winner_Return"] - row_data["Loser_Return"]
            rows.append(row_data)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["Year", "Axis"]).reset_index(drop=True)


# ============================================================================
# MAIN SIGNAL BUILDER
# ============================================================================

def build_five_factor_rotation_signals(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
    fundamentals: Dict | None = None,
    volume_df: pd.DataFrame | None = None,
    use_construction_cache: bool = True,
    force_rebuild_construction_cache: bool = False,
    construction_cache_path: Path | str | None = None,
    mwu_walkforward_config: dict | None = None,
    axis_orthogonalization_config: dict | None = None,
    return_details: bool = False,
    debug: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
    Build multi-factor rotation signal panel.

    Returns:
        DataFrame (dates x tickers) with multi-factor rotation scores (0-100)
    """
    print("\n🔧 Building multi-factor rotation signals (13 axes, exponential weighting)...")
    print(f"  Multi-lookback ensemble: {ENSEMBLE_LOOKBACK_WINDOWS} days")
    print("  Original: Size / Value / Profitability / Investment / Momentum / Risk")
    print("  New: Quality / Liquidity / TradingIntensity / Sentiment / FundMom / Carry / Defensive")
    print("  Axis weighting: Exponentially-weighted factor selection (6mo half-life)")
    _debug_log(debug, "Detailed line-by-line debug tracing is enabled")

    if data_loader is None:
        raise_signal_data_error(
            "five_factor_rotation",
            "no data_loader provided for required data dependencies",
        )

    if fundamentals is None:
        try:
            fundamentals = data_loader.load_fundamentals()
        except Exception as exc:
            raise_signal_data_error(
                "five_factor_rotation",
                f"failed to load fundamentals: {exc}",
            )

    if not fundamentals:
        raise_signal_data_error(
            "five_factor_rotation",
            "no fundamentals available",
        )

    if volume_df is None:
        volume_df = getattr(data_loader, "_volume_df", None)
    if volume_df is None:
        try:
            prices_file = data_loader.data_dir / "bist_prices_full.csv"
            prices = data_loader.load_prices(prices_file)
            volume_df = data_loader.build_volume_panel(prices)
        except Exception:
            volume_df = pd.DataFrame(np.nan, index=dates, columns=close_df.columns)

    # IMPORTANT: Use FULL close_df (not reindexed to dates) for axis computation
    # This allows MWU warm-up to access historical data before backtest start
    close = close_df.astype(float)
    daily_returns = close.pct_change()
    tickers = close.columns

    # Load fundamental metrics
    try:
        metrics_df = data_loader.load_fundamental_metrics()
    except Exception:
        metrics_df = pd.DataFrame()

    # Build axis panels (using full date range)
    axis_panels = _build_axis_construction_panels(
        close=close,
        dates=close.index,  # Use full date range, not just backtest dates
        tickers=tickers,
        data_loader=data_loader,
        fundamentals=fundamentals,
        volume_df=volume_df,
        use_cache=use_construction_cache,
        force_rebuild_cache=force_rebuild_construction_cache,
        save_cache=True,
        cache_path=construction_cache_path,
    )
    for panel_name in AXIS_PANEL_NAMES:
        _debug_panel_stats(debug, f"panel:{panel_name}", axis_panels.get(panel_name))

    # ========================================================================
    # BUILD ALL 13 AXES
    # ========================================================================

    # Axis 1: Size
    size_raw = cross_sectional_zscore(axis_panels["size_small_signal"])
    _debug_panel_stats(debug, "axis_raw:size", size_raw)

    # Axis 2: Value
    value_raw = _build_two_sided_axis_raw(
        axis_panels["value_level_signal"],
        axis_panels["value_growth_signal"],
    )
    _debug_panel_stats(debug, "axis_raw:value", value_raw)

    # Axis 3: Profitability
    profitability_raw = _build_two_sided_axis_raw(
        axis_panels["profit_margin_level"],
        axis_panels["profit_margin_growth"],
    )
    _debug_panel_stats(debug, "axis_raw:profitability", profitability_raw)

    # Axis 4: Investment
    print("  Building conservative side from fundamental metrics...")
    debt_panel = _build_metric_panel(metrics_df, "debt_to_equity", close.index, tickers)
    cash_panel = _build_metric_panel(metrics_df, "cash_ratio", close.index, tickers)
    current_panel = _build_metric_panel(metrics_df, "current_ratio", close.index, tickers)
    payout_panel = _build_metric_panel(metrics_df, "dividend_payout_ratio", close.index, tickers)

    debt_score = _squash_metric(debt_panel, scale=1.5, lower=-2.0, upper=6.0).fillna(0.0)
    cash_score = _squash_metric(cash_panel, scale=1.0, lower=0.0, upper=8.0).fillna(0.0)
    current_score = _squash_metric(current_panel, scale=2.0, lower=0.0, upper=12.0).fillna(0.0)
    payout_score = _squash_metric(payout_panel, scale=0.5, lower=-1.0, upper=2.0).fillna(0.0)

    # Conservative profile: low debt, high cash, good current ratio, reasonable payout
    # Weights are intentionally not normalized to 1.0 since debt has negative sign
    # Effective interpretation: debt penalty of 0.55, benefits from cash (0.30),
    # current ratio (0.30), and payout (0.20)
    conservative_profile = (
        -0.55 * debt_score + 0.30 * cash_score + 0.30 * current_score + 0.20 * payout_score
    )
    conservative_valid = debt_panel.notna() | cash_panel.notna() | current_panel.notna() | payout_panel.notna()
    conservative_profile = conservative_profile.where(conservative_valid)

    investment_raw = _build_two_sided_axis_raw(
        conservative_profile,
        axis_panels["investment_reinvestment_signal"],
    )
    _debug_panel_stats(debug, "axis_raw:investment", investment_raw)

    # Axis 5: Momentum
    momentum_raw = close.shift(MOMENTUM_SKIP_DAYS) / close.shift(MOMENTUM_LOOKBACK_DAYS + MOMENTUM_SKIP_DAYS) - 1.0
    _debug_panel_stats(debug, "axis_raw:momentum", momentum_raw)

    # Axis 6: Beta (Market Sensitivity)
    # High side: High beta stocks (cyclical/aggressive - outperform in bull markets)
    # Low side: Low beta stocks (defensive/stable - outperform in bear markets)
    # MWU will learn when to weight this factor based on recent performance
    print("  Building beta axis from market sensitivity...")
    beta_panel = axis_panels.get("market_beta")

    if beta_panel is not None:
        # Simple: high beta = high score
        # MWU will upweight in bull markets, downweight in bear markets
        risk_raw = cross_sectional_zscore(beta_panel)
    else:
        # Fallback: use inverse volatility if beta not available
        vol_panel = axis_panels.get("realized_volatility")
        risk_raw = -cross_sectional_zscore(vol_panel) if vol_panel is not None else pd.DataFrame(0.0, index=close.index, columns=tickers)
    _debug_panel_stats(debug, "axis_raw:risk", risk_raw)

    # Axis 7-13: New factors
    quality_raw = combine_quality_axis(axis_panels, close.index, tickers)
    liquidity_raw = combine_liquidity_axis(axis_panels, close.index, tickers)
    trading_intensity_raw = combine_trading_intensity_axis(axis_panels, close.index, tickers)
    sentiment_raw = combine_sentiment_axis(axis_panels, close.index, tickers)
    fundmom_raw = combine_fundmom_axis(axis_panels, close.index, tickers)
    carry_raw = combine_carry_axis(axis_panels, close.index, tickers)
    defensive_raw = combine_defensive_axis(axis_panels, close.index, tickers)
    _debug_panel_stats(debug, "axis_raw:quality", quality_raw)
    _debug_panel_stats(debug, "axis_raw:liquidity", liquidity_raw)
    _debug_panel_stats(debug, "axis_raw:trading_intensity", trading_intensity_raw)
    _debug_panel_stats(debug, "axis_raw:sentiment", sentiment_raw)
    _debug_panel_stats(debug, "axis_raw:fundmom", fundmom_raw)
    _debug_panel_stats(debug, "axis_raw:carry", carry_raw)
    _debug_panel_stats(debug, "axis_raw:defensive", defensive_raw)

    # All 13 axes
    axis_specs = {
        "size": (size_raw, "Small", "Big"),
        "value": (value_raw, "Value Level", "Value Growth"),
        "profitability": (profitability_raw, "Margin Level", "Margin Growth"),
        "investment": (investment_raw, "Conservative", "Reinvestment"),
        "momentum": (momentum_raw, "Winner", "Loser"),
        "risk": (risk_raw, "High Beta", "Low Beta"),
        "quality": (quality_raw, "High Quality", "Low Quality"),
        "liquidity": (liquidity_raw, "Liquid", "Illiquid"),
        "trading_intensity": (trading_intensity_raw, "High Activity", "Low Activity"),
        "sentiment": (sentiment_raw, "Strong Action", "Weak Action"),
        "fundmom": (fundmom_raw, "Improving", "Deteriorating"),
        "carry": (carry_raw, "High Yield", "Low Yield"),
        "defensive": (defensive_raw, "Stable", "Cyclical"),
    }

    orth_cfg = axis_orthogonalization_config if isinstance(axis_orthogonalization_config, dict) else {}
    orth_enabled = bool(orth_cfg.get("enabled", DEFAULT_ORTHOGONALIZE_AXES))
    orth_details: Dict[str, object] = {}

    if orth_enabled:
        min_overlap = max(int(orth_cfg.get("min_overlap", DEFAULT_ORTHOG_MIN_OVERLAP)), 2)
        epsilon = float(orth_cfg.get("epsilon", DEFAULT_ORTHOG_EPSILON))
        axis_order = list(axis_specs.keys())
        requested_order = orth_cfg.get("order")
        if isinstance(requested_order, (list, tuple)) and requested_order:
            preferred = list(dict.fromkeys([str(name) for name in requested_order]))
            valid = [name for name in preferred if name in axis_specs]
            missing = [name for name in preferred if name not in axis_specs]
            remaining = [name for name in axis_order if name not in valid]
            if valid:
                axis_order = valid + remaining
            if missing:
                print(f"  ⚠️  Ignoring unknown orthogonalization axes: {', '.join(missing)}")

        print(
            "  Orthogonalizing axis raws "
            f"(cross-sectional residualization, min_overlap={min_overlap})..."
        )
        raw_axis_map = {axis_name: axis_specs[axis_name][0] for axis_name in axis_order}
        orth_axis_map, orth_details = _orthogonalize_axis_raw_scores(
            axis_raw_map=raw_axis_map,
            axis_order=axis_order,
            min_overlap=min_overlap,
            epsilon=epsilon,
        )

        axis_specs = {
            axis_name: (orth_axis_map[axis_name], high_label, low_label)
            for axis_name, (_, high_label, low_label) in axis_specs.items()
        }

        before_corr = orth_details.get("raw_mean_abs_corr", np.nan)
        after_corr = orth_details.get("orth_mean_abs_corr", np.nan)
        if np.isfinite(before_corr) and np.isfinite(after_corr):
            print(f"  Axis overlap reduced: mean |corr| {before_corr:.3f} -> {after_corr:.3f}")

        for axis_name, (axis_raw, _, _) in axis_specs.items():
            _debug_panel_stats(debug, f"axis_raw_orth:{axis_name}", axis_raw)

    # Compute axis components for all configured axes (MWU weights all of them)
    axis_components: Dict[str, pd.DataFrame] = {}
    axis_summary: Dict[str, Tuple] = {}
    axis_daily_returns: Dict[str, Tuple[pd.Series, pd.Series]] = {}
    axis_bucket_returns: Dict[str, pd.DataFrame] = {}
    insufficient_axes: list[str] = []

    use_quintiles = USE_QUINTILE_BUCKETS
    n_total = len(axis_specs)
    print(f"  Computing {n_total} axis components with {N_BUCKETS}-bucket selection...")

    for axis_name, (axis_raw, high_label, low_label) in axis_specs.items():
        component, winning_bucket, bucket_cum_df, high_daily, low_daily = _compute_axis_component(
            axis_raw_scores=axis_raw,
            daily_returns=daily_returns,
            lookback=ROTATION_LOOKBACK_DAYS,
            use_quintiles=use_quintiles,
        )
        axis_components[axis_name] = component
        axis_summary[axis_name] = (winning_bucket, bucket_cum_df, high_label, low_label)
        axis_daily_returns[axis_name] = (high_daily, low_daily)
        axis_bucket_returns[axis_name] = bucket_cum_df
        max_valid_names = int(component.notna().sum(axis=1).max()) if not component.empty else 0
        if max_valid_names < 5:
            insufficient_axes.append(f"{axis_name}(max_valid_names={max_valid_names})")
        _debug_axis_component_stats(debug, axis_name, component, winning_bucket, high_daily, low_daily)

    if insufficient_axes:
        raise_signal_data_error(
            "five_factor_rotation",
            "insufficient cross-sectional coverage in axis components: "
            + ", ".join(insufficient_axes),
        )

    # Compute MWU weights (online learning — no screening needed)
    # Supports optional calendar-year walk-forward (e.g., 3y train / 1y test)
    print("  Computing MWU axis weights...")
    walk_cfg = mwu_walkforward_config if isinstance(mwu_walkforward_config, dict) else {}
    walk_enabled = bool(walk_cfg.get("enabled", False))
    walk_train_years = int(walk_cfg.get("train_years", 0)) if walk_enabled else None
    walk_first_test_year = walk_cfg.get("first_test_year") if walk_enabled else None
    walk_last_test_year = walk_cfg.get("last_test_year") if walk_enabled else None
    if walk_first_test_year is not None:
        walk_first_test_year = int(walk_first_test_year)
    if walk_last_test_year is not None:
        walk_last_test_year = int(walk_last_test_year)

    if walk_enabled:
        _debug_log(
            debug,
            f"MWU walk-forward: train_years={walk_train_years}, "
            f"first_test_year={walk_first_test_year}, last_test_year={walk_last_test_year}",
        )

    axis_weights = compute_mwu_weights(
        axis_daily_returns,
        dates,
        warmup_months=6,
        debug=debug,
        walkforward_train_years=walk_train_years,
        walkforward_first_test_year=walk_first_test_year,
        walkforward_last_test_year=walk_last_test_year,
    )

    # Weighted average using MWU weights
    aligned_components = {
        axis_name: comp.reindex(index=dates, columns=tickers)
        for axis_name, comp in axis_components.items()
    }

    weighted_sum = pd.DataFrame(0.0, index=dates, columns=tickers)
    weight_sum = pd.DataFrame(0.0, index=dates, columns=tickers)

    for axis_name, comp in aligned_components.items():
        # Reindex weights to match component dates (defensive programming)
        w = axis_weights[axis_name].reindex(dates).values[:, np.newaxis]
        valid_mask = comp.notna()
        weighted_sum = weighted_sum + comp.fillna(0.0) * w
        weight_sum = weight_sum + valid_mask.astype(float) * w

    weight_sum = weight_sum.replace(0.0, np.nan)
    final_scores = weighted_sum / weight_sum

    final_scores = final_scores.clip(0.0, 100.0)
    assert_has_cross_section(
        final_scores,
        "five_factor_rotation",
        "final score panel",
        min_valid_tickers=5,
    )
    assert_panel_not_constant(final_scores, "five_factor_rotation", "final score panel")
    latest_valid = int(final_scores.iloc[-1].notna().sum()) if len(final_scores.index) else 0
    if latest_valid < 5:
        raise_signal_data_error(
            "five_factor_rotation",
            f"latest date has insufficient coverage: {latest_valid} valid names (< 5)",
        )
    _debug_panel_stats(debug, "final_scores", final_scores)

    # Summary
    latest_date = final_scores.index[-1]
    latest_scores = final_scores.loc[latest_date].dropna()

    print(f"  Latest date: {latest_date.date()}")
    if len(latest_scores) > 0:
        print(f"  Latest scores - Mean: {latest_scores.mean():.1f}, Std: {latest_scores.std():.1f}")
        top_5 = latest_scores.nlargest(5)
        print(f"  Top 5 stocks: {', '.join(top_5.index.tolist())}")

    latest_w = axis_weights.loc[latest_date] if latest_date in axis_weights.index else axis_weights.iloc[-1]
    if debug and not axis_weights.empty:
        sample_indices = [0, len(axis_weights) // 2, len(axis_weights) - 1]
        seen_dates = set()
        for sample_idx in sample_indices:
            sample_date = axis_weights.index[sample_idx]
            if sample_date in seen_dates:
                continue
            seen_dates.add(sample_date)
            sample_w = axis_weights.loc[sample_date].sort_values(ascending=False).head(5)
            sample_str = ", ".join([f"{k}={v:.1%}" for k, v in sample_w.items()])
            _debug_log(debug, f"axis_weights@{sample_date.date()}: {sample_str}")
    print("  MWU axis weights:")
    for name in sorted(latest_w.index, key=lambda n: -latest_w[n]):
        print(f"    {name:<16}: {latest_w[name]:.1%}")

    yearly_report = _build_yearly_axis_winner_report(axis_summary, axis_daily_returns, axis_bucket_returns, use_quintiles)

    print(f"  Multi-factor rotation signals: {final_scores.shape[0]} days x {final_scores.shape[1]} tickers ({n_total} axes)")

    if return_details:
        details = {
            "yearly_axis_winners": yearly_report,
            "axis_weights": axis_weights,
            "axis_components": aligned_components,
            "active_axes": list(axis_specs.keys()),
        }
        if orth_enabled:
            details["axis_orthogonalization"] = {
                "method": "cross_sectional_residualization",
                "axis_order": orth_details.get("axis_order", list(axis_specs.keys())),
                "raw_mean_abs_corr": orth_details.get("raw_mean_abs_corr", np.nan),
                "orth_mean_abs_corr": orth_details.get("orth_mean_abs_corr", np.nan),
                "raw_daily_mean_abs_corr": orth_details.get("raw_daily_mean_abs_corr"),
                "orth_daily_mean_abs_corr": orth_details.get("orth_daily_mean_abs_corr"),
            }
        return final_scores, details

    return final_scores


# ============================================================================
# CACHE BUILDER
# ============================================================================

def build_five_factor_rotation_axis_cache(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader,
    fundamentals: Dict | None = None,
    volume_df: pd.DataFrame | None = None,
    cache_path: Path | str | None = None,
    force_rebuild: bool = True,
) -> Path:
    """Precompute heavy axis-construction inputs and persist to parquet."""
    if fundamentals is None:
        fundamentals = data_loader.load_fundamentals() if data_loader is not None else {}

    if volume_df is None:
        volume_df = getattr(data_loader, "_volume_df", None)
    if volume_df is None and data_loader is not None:
        prices_file = data_loader.data_dir / "bist_prices_full.csv"
        prices = data_loader.load_prices(prices_file)
        volume_df = data_loader.build_volume_panel(prices)
    if volume_df is None:
        volume_df = pd.DataFrame(np.nan, index=dates, columns=close_df.columns)

    close = close_df.reindex(dates).astype(float)
    tickers = close.columns

    _build_axis_construction_panels(
        close=close,
        dates=dates,
        tickers=tickers,
        data_loader=data_loader,
        fundamentals=fundamentals,
        volume_df=volume_df,
        use_cache=True,
        force_rebuild_cache=force_rebuild,
        save_cache=True,
        cache_path=cache_path,
    )
    return _resolve_axis_cache_path(data_loader, cache_path)
