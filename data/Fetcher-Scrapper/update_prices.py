#!/usr/bin/env python3
"""
Incremental price updater for BIST strategy data.

Updates both CSV and Parquet files to the latest available date:
  1. bist_prices_full.csv/.parquet   – all BIST stock OHLCV
  2. xu100_prices.csv/.parquet       – XU100 index OHLCV
  3. xau_try_2013_2026.csv/.parquet  – XAU/TRY (gold in lira)

Both CSV and Parquet versions are kept in sync for flexibility.
Parquet files are used for faster data loading in the portfolio engine.

Usage:
    python data/Fetcher-Scrapper/update_prices.py
    python data/Fetcher-Scrapper/update_prices.py --source auto
    python data/Fetcher-Scrapper/update_prices.py --source borsapy
    python data/Fetcher-Scrapper/update_prices.py --source yfinance --dry-run

Schedule with cron (every weekday at 18:45 Istanbul time):
    45 18 * * 1-5  cd /home/safa/Documents/Markets/BIST && python data/Fetcher-Scrapper/update_prices.py >> data/update.log 2>&1
"""

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd
import yfinance as yf

try:
    import borsapy as bp
    BORSAPY_AVAILABLE = True
except ImportError:
    bp = None
    BORSAPY_AVAILABLE = False

try:
    from borsapy_client import BorsapyClient
except Exception:
    BorsapyClient = None

DATA_DIR = Path(__file__).resolve().parent.parent
BIST_PRICES = DATA_DIR / "bist_prices_full.csv"
BIST_PRICES_PARQUET = DATA_DIR / "bist_prices_full.parquet"
XU100_PRICES = DATA_DIR / "xu100_prices.csv"
XU100_PRICES_PARQUET = DATA_DIR / "xu100_prices.parquet"
XAU_TRY_PRICES = DATA_DIR / "xau_try_2013_2026.csv"
XAU_TRY_PARQUET = DATA_DIR / "xau_try_2013_2026.parquet"

MNYET_URL = "https://finans.mynet.com/borsa/hisseler/"

BIST_COLS = ["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
XU100_COLS = BIST_COLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def last_date_in_csv(path: Path, date_col: str = "Date") -> dt.date:
    """Read only the Date column and return the latest date."""
    df = pd.read_csv(path, usecols=[date_col], parse_dates=[date_col])
    return df[date_col].max().date()


def _wants_borsapy(source: str) -> bool:
    """Return whether borsapy should be used for this run."""
    if source == "yfinance":
        return False
    return BORSAPY_AVAILABLE and BorsapyClient is not None


def _coerce_price_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Normalize price frame to expected schema and ordering."""
    normalized = df.rename(columns=str.title).copy()
    for col in columns:
        if col not in normalized.columns:
            normalized[col] = None
    normalized = normalized[columns]
    normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce")
    ohlcv = ["Open", "High", "Low", "Close", "Volume"]
    normalized = normalized.dropna(subset=ohlcv, how="all")
    return normalized


def _append_and_persist(
    existing_path: Path,
    parquet_path: Path,
    new_df: pd.DataFrame,
    dedupe_cols: list[str],
    sort_cols: list[str],
) -> pd.DataFrame:
    """Append new rows, dedupe, and write both CSV and parquet."""
    existing = pd.read_csv(existing_path, parse_dates=["Date"])
    combined = pd.concat([existing, new_df], ignore_index=True)
    if "Date" in combined.columns:
        combined = combined[combined["Date"].notna()]
    combined = combined.drop_duplicates(subset=dedupe_cols, keep="last")
    combined = combined.sort_values(sort_cols).reset_index(drop=True)
    combined.to_csv(existing_path, index=False)
    combined.to_parquet(parquet_path, index=False)
    return combined


def _fallback_bist_tickers_from_existing() -> list[str]:
    """Fallback ticker universe from local historical file."""
    if not BIST_PRICES.exists():
        return []

    try:
        existing = pd.read_csv(BIST_PRICES, usecols=["Ticker"])
    except Exception:
        return []

    tickers = (
        existing["Ticker"]
        .dropna()
        .astype(str)
        .str.replace(".IS", "", regex=False)
        .str.upper()
        .unique()
        .tolist()
    )
    return sorted(t for t in tickers if t.isalpha())


def fetch_bist_tickers(prefer_borsapy: bool = False) -> list[str]:
    if prefer_borsapy and BORSAPY_AVAILABLE:
        try:
            return sorted(bp.Index("XUTUM").component_symbols or [])
        except Exception as exc:
            print(f"  Warning: could not fetch ticker list from borsapy/XUTUM: {exc}")

    try:
        tables = pd.read_html(MNYET_URL)
        if not tables:
            raise RuntimeError(f"No tables found at {MNYET_URL}")
        tickers = (
            tables[0]["Hisseler"]
            .astype(str)
            .str.split()
            .str[0]
            .dropna()
            .unique()
            .tolist()
        )
        return sorted(t for t in tickers if t.isalpha())
    except Exception as exc:
        print(f"  Warning: could not fetch ticker list from {MNYET_URL}: {exc}")
        fallback = _fallback_bist_tickers_from_existing()
        if fallback:
            print(f"  Using {len(fallback)} tickers from local price history fallback")
            return fallback
        raise RuntimeError("Ticker universe fetch failed and no local fallback is available") from exc


# ---------------------------------------------------------------------------
# 1. BIST all-stock prices
# ---------------------------------------------------------------------------

def update_bist_prices(dry_run: bool = False, source: str = "auto") -> None:
    print("\n" + "=" * 60)
    print("BIST STOCK PRICES")
    print("=" * 60)

    last = last_date_in_csv(BIST_PRICES)
    # Start from the day after the last row (yfinance start is inclusive)
    start = (last + dt.timedelta(days=1)).isoformat()
    end = dt.date.today().isoformat()

    print(f"  Last date in CSV : {last}")
    print(f"  Fetch window     : {start}  ->  {end}")

    if start >= end:
        print("  Already up to date.")
        return

    use_borsapy = _wants_borsapy(source)
    if dry_run:
        provider = "borsapy" if use_borsapy else "yfinance"
        print(f"  [dry-run] Would fetch BIST prices via {provider}.")
        return

    tickers = fetch_bist_tickers(prefer_borsapy=use_borsapy)
    print(f"  Ticker list      : {len(tickers)} BIST tickers")

    new_df = pd.DataFrame()

    if use_borsapy:
        try:
            client = BorsapyClient(cache_dir=DATA_DIR / "borsapy_cache")
            new_df = client.batch_download_to_long(
                symbols=tickers,
                start=start,
                end=end,
                group_by="ticker",
                add_is_suffix=True,
            )
            new_df = _coerce_price_columns(new_df, BIST_COLS)
            if not new_df.empty:
                print("  Data source      : borsapy")
        except Exception as exc:
            if source == "borsapy":
                raise
            print(f"  Warning: borsapy fetch failed, falling back to yfinance: {exc}")

    if new_df.empty and source != "borsapy":
        yf_tickers = [f"{t}.IS" for t in tickers]
        data = yf.download(
            yf_tickers,
            start=start,
            end=end,
            progress=True,
            group_by="ticker",
            auto_adjust=False,
            threads=True,
        )

        if data is None or data.empty:
            print("  No new data returned by yfinance.")
            return

        records = []
        if isinstance(data.columns, pd.MultiIndex):
            for ticker in yf_tickers:
                if ticker not in data.columns.get_level_values(0):
                    continue
                df_t = data[ticker].copy().reset_index()
                df_t["Ticker"] = ticker
                records.append(df_t)
        else:
            df_single = data.copy().reset_index()
            df_single["Ticker"] = yf_tickers[0]
            records.append(df_single)

        if not records:
            print("  No records after parsing.")
            return

        new_df = _coerce_price_columns(pd.concat(records, ignore_index=True), BIST_COLS)
        print("  Data source      : yfinance")

    if new_df.empty:
        print("  No valid new rows after cleanup.")
        return

    combined = _append_and_persist(
        existing_path=BIST_PRICES,
        parquet_path=BIST_PRICES_PARQUET,
        new_df=new_df,
        dedupe_cols=["Date", "Ticker"],
        sort_cols=["Ticker", "Date"],
    )
    print(f"  ✅ Parquet updated: {BIST_PRICES_PARQUET.name}")

    new_last = combined["Date"].max().date()
    print(f"  Appended {len(new_df)} rows  ->  new last date: {new_last}")
    print(f"  Total rows: {len(combined)}")


# ---------------------------------------------------------------------------
# 2. XU100 index prices
# ---------------------------------------------------------------------------

def update_xu100_prices(dry_run: bool = False, source: str = "auto") -> None:
    print("\n" + "=" * 60)
    print("XU100 INDEX PRICES")
    print("=" * 60)

    last = last_date_in_csv(XU100_PRICES)
    start = (last + dt.timedelta(days=1)).isoformat()
    end = dt.date.today().isoformat()

    print(f"  Last date in CSV : {last}")
    print(f"  Fetch window     : {start}  ->  {end}")

    if start >= end:
        print("  Already up to date.")
        return

    use_borsapy = _wants_borsapy(source)
    if dry_run:
        provider = "borsapy" if use_borsapy else "yfinance"
        print(f"  [dry-run] Would fetch XU100 prices via {provider}.")
        return

    new_df = pd.DataFrame()

    if use_borsapy:
        try:
            hist = bp.Index("XU100").history(start=start, end=end, interval="1d")
            if hist is not None and not hist.empty:
                new_df = hist.copy().reset_index()
                new_df["Ticker"] = "XU100.IS"
                new_df = _coerce_price_columns(new_df, XU100_COLS)
                print("  Data source      : borsapy")
        except Exception as exc:
            if source == "borsapy":
                raise
            print(f"  Warning: borsapy fetch failed, falling back to yfinance: {exc}")

    if new_df.empty and source != "borsapy":
        for ticker in ("XU100.IS", "XU100"):
            data = yf.download(
                ticker,
                start=start,
                end=end,
                progress=True,
                auto_adjust=False,
                threads=True,
            )
            if data is not None and not data.empty:
                break
        else:
            print("  No XU100 data returned.")
            return

        # Flatten MultiIndex columns that yfinance may return for single tickers
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        new_df = data.copy().reset_index()
        new_df["Ticker"] = "XU100.IS"
        new_df = _coerce_price_columns(new_df, XU100_COLS)
        print("  Data source      : yfinance")

    if new_df.empty:
        print("  No valid new rows.")
        return

    combined = _append_and_persist(
        existing_path=XU100_PRICES,
        parquet_path=XU100_PRICES_PARQUET,
        new_df=new_df,
        dedupe_cols=["Date"],
        sort_cols=["Date"],
    )
    print(f"  ✅ Parquet updated: {XU100_PRICES_PARQUET.name}")

    new_last = combined["Date"].max().date()
    print(f"  Appended {len(new_df)} rows  ->  new last date: {new_last}")


# ---------------------------------------------------------------------------
# 3. XAU/TRY (gold price in Turkish lira)
# ---------------------------------------------------------------------------

def update_xau_try(dry_run: bool = False, source: str = "auto") -> None:
    print("\n" + "=" * 60)
    print("XAU/TRY PRICES")
    print("=" * 60)

    last = last_date_in_csv(XAU_TRY_PRICES)
    start = (last + dt.timedelta(days=1)).isoformat()
    end = dt.date.today().isoformat()

    print(f"  Last date in CSV : {last}")
    print(f"  Fetch window     : {start}  ->  {end}")

    if start >= end:
        print("  Already up to date.")
        return

    use_borsapy = _wants_borsapy(source)
    if dry_run:
        provider = "borsapy" if use_borsapy else "yfinance"
        print(f"  [dry-run] Would fetch XAU/TRY prices via {provider}.")
        return

    new_df = pd.DataFrame()

    if use_borsapy:
        try:
            xau_hist = bp.FX("ons-altin").history(start=start, end=end, interval="1d")
            usd_hist = bp.FX("USD").history(start=start, end=end, interval="1d")
            xau = xau_hist["Close"] if xau_hist is not None and not xau_hist.empty else pd.Series(dtype=float)
            usd_try = usd_hist["Close"] if usd_hist is not None and not usd_hist.empty else pd.Series(dtype=float)

            new_df = pd.concat([xau, usd_try], axis=1)
            new_df.columns = ["XAU_USD", "USD_TRY"]
            new_df["XAU_TRY"] = new_df["XAU_USD"] * new_df["USD_TRY"]
            new_df = new_df.dropna()
            new_df.index.name = "Date"
            if not new_df.empty:
                print("  Data source      : borsapy")
        except Exception as exc:
            if source == "borsapy":
                raise
            print(f"  Warning: borsapy fetch failed, falling back to yfinance: {exc}")

    if new_df.empty and source != "borsapy":
        # Download gold (USD) and USD/TRY from yfinance
        def _get_close(ticker: str, start_date: str, end_date: str) -> pd.Series:
            raw = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            return raw["Close"]

        try:
            xau = _get_close("XAUUSD=X", start, end)
        except Exception:
            xau = _get_close("GC=F", start, end)

        usd_try = _get_close("USDTRY=X", start, end)
        new_df = pd.concat([xau, usd_try], axis=1)
        new_df.columns = ["XAU_USD", "USD_TRY"]
        new_df["XAU_TRY"] = new_df["XAU_USD"] * new_df["USD_TRY"]
        new_df = new_df.dropna()
        new_df.index.name = "Date"
        print("  Data source      : yfinance")

    if new_df.empty:
        print("  No valid new rows.")
        return

    existing = pd.read_csv(XAU_TRY_PRICES, parse_dates=["Date"], index_col="Date")
    combined = pd.concat([existing, new_df])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    combined.to_csv(XAU_TRY_PRICES)
    
    # Also save as parquet for faster loading
    combined.to_parquet(XAU_TRY_PARQUET)
    print(f"  ✅ Parquet updated: {XAU_TRY_PARQUET.name}")

    new_last = combined.index.max().date()
    print(f"  Appended {len(new_df)} rows  ->  new last date: {new_last}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Incrementally update BIST price data files."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fetched without downloading.",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "borsapy", "yfinance"],
        default="auto",
        help="Data source: auto (prefer borsapy), borsapy only, or yfinance only.",
    )
    args = parser.parse_args()

    print(f"{'=' * 60}")
    print(f"BIST DATA UPDATER  —  {dt.datetime.now():%Y-%m-%d %H:%M}")
    print(f"{'=' * 60}")
    print(f"Source mode: {args.source}")

    if args.source == "borsapy" and not _wants_borsapy("borsapy"):
        print("ERROR: --source borsapy selected but borsapy integration is not available.")
        print("Install borsapy and ensure `data/Fetcher-Scrapper/borsapy_client.py` is importable.")
        return 1

    failures: list[tuple[str, Exception]] = []

    steps = [
        ("BIST STOCK PRICES", update_bist_prices),
        ("XU100 INDEX PRICES", update_xu100_prices),
        ("XAU/TRY PRICES", update_xau_try),
    ]
    for label, step in steps:
        try:
            step(dry_run=args.dry_run, source=args.source)
        except Exception as exc:
            failures.append((label, exc))
            print(f"\n  ERROR in {label}: {exc}")

    print("\n" + "=" * 60)
    if failures:
        print(f"UPDATES COMPLETED WITH {len(failures)} FAILURE(S)")
        for label, exc in failures:
            print(f"  - {label}: {exc}")
        print("=" * 60)
        return 1

    print("ALL UPDATES COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
