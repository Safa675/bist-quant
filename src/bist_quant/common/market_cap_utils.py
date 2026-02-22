from __future__ import annotations

import pandas as pd

LARGE_CAP_PERCENTILE = 90
SMALL_CAP_PERCENTILE = 10
SIZE_LIQUIDITY_QUANTILE = 0.25
MIN_BUCKET_NAMES = 10


def get_size_buckets_for_date(
    market_cap_row: pd.Series,
    liquidity_row: pd.Series,
    large_percentile: int = LARGE_CAP_PERCENTILE,
    small_percentile: int = SMALL_CAP_PERCENTILE,
    liquidity_quantile: float = SIZE_LIQUIDITY_QUANTILE,
    min_bucket_names: int = MIN_BUCKET_NAMES,
) -> tuple[set[str], set[str], set[str]]:
    """Build liquid universe then split by market-cap deciles."""
    if market_cap_row is None or market_cap_row.empty:
        return set(), set(), set()

    if liquidity_row is None or liquidity_row.empty:
        combined = pd.DataFrame({"mcap": market_cap_row}).dropna()
    else:
        combined = pd.concat(
            [market_cap_row.rename("mcap"), liquidity_row.rename("liq")],
            axis=1,
            join="inner",
        ).dropna()

    if combined.empty:
        return set(), set(), set()

    if "liq" in combined.columns:
        liq_threshold = combined["liq"].quantile(liquidity_quantile)
        liquid_df = combined[combined["liq"] >= liq_threshold]
    else:
        liquid_df = combined

    if len(liquid_df) < 2 * min_bucket_names:
        return set(liquid_df.index), set(), set()

    large_thr = liquid_df["mcap"].quantile(large_percentile / 100.0)
    small_thr = liquid_df["mcap"].quantile(small_percentile / 100.0)

    large_caps = set(liquid_df[liquid_df["mcap"] >= large_thr].index)
    small_caps = set(liquid_df[liquid_df["mcap"] <= small_thr].index) - large_caps

    if len(large_caps) < min_bucket_names or len(small_caps) < min_bucket_names:
        ordered = liquid_df["mcap"].sort_values()
        n = max(min_bucket_names, int(len(ordered) * 0.10))
        n = min(n, len(ordered) // 2)
        small_caps = set(ordered.head(n).index)
        large_caps = set(ordered.tail(n).index) - small_caps

    return set(liquid_df.index), small_caps, large_caps
