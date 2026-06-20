"""Fundamental data helpers for consolidated financial-statement data.

Extracted from ``common/utils.py`` to keep the fundamentals-specific logic
(security-row lookup, Turkish-key matching, quarter coercion, TTM sums and
reporting-lag alignment) in one cohesive module.

These helpers are re-exported from ``bist_quant.common.utils`` for backwards
compatibility, so existing ``from bist_quant.common.utils import pick_row``
call sites continue to work unchanged.
"""

import pandas as pd


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


__all__ = [
    "_turkish_match",
    "pick_row",
    "get_consolidated_sheet",
    "pick_row_from_sheet",
    "coerce_quarter_cols",
    "sum_ttm",
    "apply_lag",
]
