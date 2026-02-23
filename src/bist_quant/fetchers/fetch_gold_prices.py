#!/usr/bin/env python3
"""
Fetch 10 years of daily XAU/TRY gold prices and store in data/borsapy_cache/gold/.

Usage:
    python scripts/fetch_gold_prices.py
    python scripts/fetch_gold_prices.py --years 5
    python scripts/fetch_gold_prices.py --output-csv   # also writes data/gold_prices_daily.csv
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project src is on path when run as a script
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import pandas as pd

CACHE_DIR = _ROOT / "data" / "borsapy_cache"
DEFAULT_YEARS = 10


def _fetch_borsapy(start: str, end: str) -> pd.DataFrame:
    """Fetch XAU/TRY and USD/TRY daily prices via borsapy."""
    try:
        import borsapy as bp  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SystemExit("borsapy is not installed. Run: pip install borsapy") from exc

    print(f"  Fetching XAU/TRY (ons-altin) {start} → {end} ...")
    xau_hist = bp.FX("ons-altin").history(start=start, end=end, interval="1d")

    print(f"  Fetching USD/TRY {start} → {end} ...")
    usd_hist = bp.FX("USD").history(start=start, end=end, interval="1d")

    # Normalise indices to date-only
    def _to_date_index(df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=float)
        idx = pd.to_datetime(df.index).normalize()
        return pd.Series(df["Close"].values, index=idx, name="close")

    xau_s = _to_date_index(xau_hist)
    usd_s = _to_date_index(usd_hist)

    df = pd.DataFrame({"XAU_TRY": xau_s, "USD_TRY": usd_s})
    df.index.name = "Date"
    df["XAU_USD"] = (df["XAU_TRY"] / df["USD_TRY"]).round(4)
    df = df.dropna(subset=["XAU_TRY"]).sort_index()
    return df


def fetch_gold(years: int = DEFAULT_YEARS, output_csv: bool = False) -> pd.DataFrame:
    """Fetch daily gold prices and persist to disk cache."""
    from bist_quant.common.disk_cache import DiskCache  # type: ignore[import]

    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=365 * years)).strftime("%Y-%m-%d")

    print(f"Fetching {years}-year daily gold prices: {start} → {end}")
    df = _fetch_borsapy(start, end)

    if df.empty:
        print("WARNING: No data returned from borsapy.", file=sys.stderr)
        return df

    cache = DiskCache(CACHE_DIR)
    cache.set_dataframe("gold", "xau_try_daily", df, ttl_seconds=86400)
    parquet_path = CACHE_DIR / "gold" / "xau_try_daily.parquet"
    print(f"  Cached {len(df):,} rows → {parquet_path.relative_to(_ROOT)}")

    if output_csv:
        csv_path = _ROOT / "data" / "gold_prices_daily.csv"
        df.to_csv(csv_path)
        print(f"  CSV written → {csv_path.relative_to(_ROOT)}")

    print("\nLast 5 rows:")
    print(df.tail().to_string())
    print(f"\nDate range : {df.index[0].date()} – {df.index[-1].date()}")
    print(f"Rows       : {len(df):,}")
    print(f"XAU/TRY    : min={df['XAU_TRY'].min():.2f}  max={df['XAU_TRY'].max():.2f}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch daily gold (XAU/TRY) prices via borsapy")
    parser.add_argument(
        "--years",
        type=int,
        default=DEFAULT_YEARS,
        help=f"Number of years of history to fetch (default: {DEFAULT_YEARS})",
    )
    parser.add_argument(
        "--output-csv",
        action="store_true",
        help="Also write data/gold_prices_daily.csv",
    )
    args = parser.parse_args()
    fetch_gold(years=args.years, output_csv=args.output_csv)


if __name__ == "__main__":
    main()
