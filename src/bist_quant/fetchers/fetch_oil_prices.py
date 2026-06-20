#!/usr/bin/env python3
"""
Fetch 10 years of daily WTI and Brent crude oil prices and store in
data/borsapy_cache/commodities/.

Usage:
    python -m bist_quant.fetchers.fetch_oil_prices
    python -m bist_quant.fetchers.fetch_oil_prices --years 5
    python -m bist_quant.fetchers.fetch_oil_prices --output-csv
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Ensure project src is on path when run as a script
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

CACHE_DIR = _ROOT / "data" / "borsapy_cache"
DEFAULT_YEARS = 10

# yfinance tickers
_TICKERS = {
    "CL=F": "WTI",     # WTI crude oil futures
    "BZ=F": "Brent",   # Brent crude oil futures
}


def _fetch_yfinance(start: str, end: str) -> pd.DataFrame:
    """Fetch WTI and Brent daily close prices via yfinance."""
    try:
        import yfinance as yf  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SystemExit(
            "yfinance is not installed. Run: pip install yfinance"
        ) from exc

    frames: dict[str, pd.Series] = {}
    for ticker, label in _TICKERS.items():
        print(f"  Fetching {label} ({ticker}) {start} → {end} ...")
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

        if raw is None or raw.empty:
            print(f"    ⚠️  No data returned for {ticker}", file=sys.stderr)
            continue

        # yfinance returns MultiIndex columns with ticker; flatten
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        idx = pd.to_datetime(raw.index).normalize()
        s = pd.Series(
            raw["Close"].values,
            index=idx,
            name=label,
        )
        frames[label] = s

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df.index.name = "Date"
    df = df.dropna(how="all").sort_index()
    return df


def fetch_oil(years: int = DEFAULT_YEARS, output_csv: bool = False) -> pd.DataFrame:
    """Fetch daily oil prices and persist to disk cache."""
    from bist_quant.common.disk_cache import DiskCache

    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=365 * years)).strftime("%Y-%m-%d")

    print(f"Fetching {years}-year daily oil prices: {start} → {end}")
    df = _fetch_yfinance(start, end)

    if df.empty:
        print("WARNING: No data returned.", file=sys.stderr)
        return df

    cache = DiskCache(CACHE_DIR)
    cache.set_dataframe("commodities", "oil_daily", df, ttl_seconds=86400)
    parquet_path = CACHE_DIR / "commodities" / "oil_daily.parquet"
    print(f"  Cached {len(df):,} rows → {parquet_path.relative_to(_ROOT)}")

    if output_csv:
        csv_path = _ROOT / "data" / "oil_prices_daily.csv"
        df.to_csv(csv_path)
        print(f"  CSV written → {csv_path.relative_to(_ROOT)}")

    print("\nLast 5 rows:")
    print(df.tail().to_string())
    print(f"\nDate range : {df.index[0].date()} – {df.index[-1].date()}")
    print(f"Rows       : {len(df):,}")
    for col in df.columns:
        print(f"{col:8s}  : min={df[col].min():.2f}  max={df[col].max():.2f}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch daily WTI/Brent crude oil prices via yfinance"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=DEFAULT_YEARS,
        help=f"Number of years of history to fetch (default: {DEFAULT_YEARS})",
    )
    parser.add_argument(
        "--output-csv",
        action="store_true",
        help="Also write data/oil_prices_daily.csv",
    )
    args = parser.parse_args()
    fetch_oil(years=args.years, output_csv=args.output_csv)


if __name__ == "__main__":
    main()
