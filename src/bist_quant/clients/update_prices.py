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
    python src/bist_quant/data_pipeline/fetcher_scripts/update_prices.py
    python src/bist_quant/data_pipeline/fetcher_scripts/update_prices.py --source auto
    python src/bist_quant/data_pipeline/fetcher_scripts/update_prices.py --source borsapy
    python src/bist_quant/data_pipeline/fetcher_scripts/update_prices.py --source yfinance --dry-run

Schedule with cron (every weekday at 18:45 Istanbul time):
    45 18 * * 1-5  cd "/home/safa/Documents/Market Research" && python src/bist_quant/data_pipeline/fetcher_scripts/update_prices.py >> data/update.log 2>&1
"""

import logging
import argparse
import datetime as dt
import math
from pathlib import Path

import pandas as pd
import yfinance as yf
logger = logging.getLogger(__name__)

try:
    import borsapy as bp
    BORSAPY_AVAILABLE = True
except ImportError:
    bp = None
    BORSAPY_AVAILABLE = False

try:
    from bist_quant.clients.borsapy_adapter import BorsapyAdapter
except ImportError:
    BorsapyAdapter = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
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
    try:
        df = pd.read_csv(path, usecols=[date_col], parse_dates=[date_col])
        return df[date_col].max().date()
    except FileNotFoundError:
        return dt.date(2013, 1, 1)


def _wants_borsapy(source: str) -> bool:
    """Return whether borsapy should be used for this run."""
    if source == "yfinance":
        return False
    return BORSAPY_AVAILABLE


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
    try:
        existing = pd.read_csv(existing_path, parse_dates=["Date"])
        combined = pd.concat([existing, new_df], ignore_index=True)
    except FileNotFoundError:
        combined = new_df

    if "Date" in combined.columns:
        combined = combined[combined["Date"].notna()]
    combined = combined.drop_duplicates(subset=dedupe_cols, keep="last")
    combined = combined.sort_values(sort_cols).reset_index(drop=True)
    
    existing_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(existing_path, index=False)
    combined.to_parquet(parquet_path, index=False)
    return combined


def _normalize_borsapy_xau(
    xau_raw: pd.Series,
    usd_try: pd.Series,
    anchor_try: float | None = None,
) -> tuple[pd.Series, pd.Series, int]:
    """
    Normalize borsapy ons-altin rows to consistent XAU_USD and XAU_TRY series.

    borsapy payloads may mix quote units; this picks the smoother interpretation
    row-by-row to avoid synthetic spikes.
    """
    work = pd.concat(
        [
            pd.to_numeric(xau_raw, errors="coerce").rename("XAU_RAW"),
            pd.to_numeric(usd_try, errors="coerce").rename("USD_TRY"),
        ],
        axis=1,
    ).dropna()

    if work.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), 0

    prev = None
    if anchor_try is not None and pd.notna(anchor_try):
        try:
            prev_val = float(anchor_try)
            if prev_val > 0:
                prev = prev_val
        except (TypeError, ValueError):
            prev = None

    xau_try_vals: list[float] = []
    xau_usd_vals: list[float] = []
    converted_try_rows = 0

    for row in work.itertuples(index=False):
        raw = float(row.XAU_RAW)
        fx = float(row.USD_TRY)
        try_quote = raw
        usd_quote = raw * fx

        # Heuristic 1: very large raw ons-altin values are TRY-quoted.
        prefer_try_quote = raw >= 20_000.0

        # Heuristic 2: otherwise choose the interpretation closer to prior level.
        if not prefer_try_quote and prev is not None and prev > 0 and try_quote > 0 and usd_quote > 0:
            direct_jump = abs(math.log(try_quote / prev))
            multiplied_jump = abs(math.log(usd_quote / prev))
            prefer_try_quote = direct_jump < multiplied_jump

        chosen_try = try_quote if prefer_try_quote else usd_quote
        alt_try = usd_quote if prefer_try_quote else try_quote

        # Safety net against >400% synthetic jumps when an alternative exists.
        if prev is not None and prev > 0 and chosen_try > 0 and alt_try > 0:
            chosen_jump = abs(chosen_try / prev - 1.0)
            alt_jump = abs(alt_try / prev - 1.0)
            if chosen_jump > 4.0 and alt_jump < 2.0:
                chosen_try = alt_try
                prefer_try_quote = not prefer_try_quote

        if prefer_try_quote:
            converted_try_rows += 1

        xau_try_vals.append(chosen_try)
        xau_usd_vals.append(chosen_try / fx if fx > 0 else float("nan"))
        if chosen_try > 0:
            prev = chosen_try

    index = work.index
    return (
        pd.Series(xau_usd_vals, index=index, name="XAU_USD"),
        pd.Series(xau_try_vals, index=index, name="XAU_TRY"),
        converted_try_rows,
    )


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
            logger.info(f"  Warning: could not fetch ticker list from borsapy/XUTUM: {exc}")

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
        logger.info(f"  Warning: could not fetch ticker list from {MNYET_URL}: {exc}")
        fallback = _fallback_bist_tickers_from_existing()
        if fallback:
            logger.info(f"  Using {len(fallback)} tickers from local price history fallback")
            return fallback
        raise RuntimeError("Ticker universe fetch failed and no local fallback is available") from exc


# ---------------------------------------------------------------------------
# 1. BIST all-stock prices
# ---------------------------------------------------------------------------

def update_bist_prices(dry_run: bool = False, source: str = "auto") -> None:
    logger.info("\n" + "=" * 60)
    logger.info("BIST STOCK PRICES")
    logger.info("=" * 60)

    last = last_date_in_csv(BIST_PRICES)
    # Start from the day after the last row (yfinance start is inclusive)
    start = (last + dt.timedelta(days=1)).isoformat()
    end = dt.date.today().isoformat()

    logger.info(f"  Last date in CSV : {last}")
    logger.info(f"  Fetch window     : {start}  ->  {end}")

    if start >= end:
        logger.info("  Already up to date.")
        return

    use_borsapy = _wants_borsapy(source)
    if dry_run:
        provider = "borsapy" if use_borsapy else "yfinance"
        logger.info(f"  [dry-run] Would fetch BIST prices via {provider}.")
        return

    tickers = fetch_bist_tickers(prefer_borsapy=use_borsapy)
    logger.info(f"  Ticker list      : {len(tickers)} BIST tickers")

    new_df = pd.DataFrame()

    if use_borsapy:
        try:
            mock_loader = type('MockLoader', (), {'data_dir': DATA_DIR})
            adapter = BorsapyAdapter(loader=mock_loader)
            if adapter.client is not None:
                new_df = adapter.load_prices(
                    symbols=tickers,
                    period="14y", 
                )
                if not new_df.empty:
                    new_df = _coerce_price_columns(new_df, BIST_COLS)
                    logger.info("  Data source      : borsapy (adapter)")
            
        except Exception as exc:
            if source == "borsapy":
                raise
            logger.info(f"  Warning: borsapy fetch failed, falling back to yfinance: {exc}")

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
            logger.info("  No new data returned by yfinance.")
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
            logger.info("  No records after parsing.")
            return

        new_df = _coerce_price_columns(pd.concat(records, ignore_index=True), BIST_COLS)
        logger.info("  Data source      : yfinance")

    if new_df.empty:
        logger.info("  No valid new rows after cleanup.")
        return

    combined = _append_and_persist(
        existing_path=BIST_PRICES,
        parquet_path=BIST_PRICES_PARQUET,
        new_df=new_df,
        dedupe_cols=["Date", "Ticker"],
        sort_cols=["Ticker", "Date"],
    )
    logger.info(f"  ✅ Parquet updated: {BIST_PRICES_PARQUET.name}")

    new_last = combined["Date"].max().date()
    logger.info(f"  Appended {len(new_df)} rows  ->  new last date: {new_last}")
    logger.info(f"  Total rows: {len(combined)}")


# ---------------------------------------------------------------------------
# 2. XU100 index prices
# ---------------------------------------------------------------------------

def update_xu100_prices(dry_run: bool = False, source: str = "auto") -> None:
    logger.info("\n" + "=" * 60)
    logger.info("XU100 INDEX PRICES")
    logger.info("=" * 60)

    last = last_date_in_csv(XU100_PRICES)
    start = (last + dt.timedelta(days=1)).isoformat()
    end = dt.date.today().isoformat()

    logger.info(f"  Last date in CSV : {last}")
    logger.info(f"  Fetch window     : {start}  ->  {end}")

    if start >= end:
        logger.info("  Already up to date.")
        return

    use_borsapy = _wants_borsapy(source)
    if dry_run:
        provider = "borsapy" if use_borsapy else "yfinance"
        logger.info(f"  [dry-run] Would fetch XU100 prices via {provider}.")
        return

    new_df = pd.DataFrame()

    if use_borsapy:
        try:
            hist = bp.Index("XU100").history(start=start, end=end, interval="1d")
            if hist is not None and not hist.empty:
                new_df = hist.copy().reset_index()
                new_df["Ticker"] = "XU100.IS"
                new_df = _coerce_price_columns(new_df, XU100_COLS)
                logger.info("  Data source      : borsapy")
        except Exception as exc:
            if source == "borsapy":
                raise
            logger.info(f"  Warning: borsapy fetch failed, falling back to yfinance: {exc}")

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
            logger.info("  No XU100 data returned.")
            return

        # Flatten MultiIndex columns that yfinance may return for single tickers
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        new_df = data.copy().reset_index()
        new_df["Ticker"] = "XU100.IS"
        new_df = _coerce_price_columns(new_df, XU100_COLS)
        logger.info("  Data source      : yfinance")

    if new_df.empty:
        logger.info("  No valid new rows.")
        return

    combined = _append_and_persist(
        existing_path=XU100_PRICES,
        parquet_path=XU100_PRICES_PARQUET,
        new_df=new_df,
        dedupe_cols=["Date"],
        sort_cols=["Date"],
    )
    logger.info(f"  ✅ Parquet updated: {XU100_PRICES_PARQUET.name}")

    new_last = combined["Date"].max().date()
    logger.info(f"  Appended {len(new_df)} rows  ->  new last date: {new_last}")


# ---------------------------------------------------------------------------
# 3. XAU/TRY (gold price in Turkish lira)
# ---------------------------------------------------------------------------

def update_xau_try(dry_run: bool = False, source: str = "auto") -> None:
    logger.info("\n" + "=" * 60)
    logger.info("XAU/TRY PRICES")
    logger.info("=" * 60)

    last = last_date_in_csv(XAU_TRY_PRICES)
    start = (last + dt.timedelta(days=1)).isoformat()
    end = dt.date.today().isoformat()

    logger.info(f"  Last date in CSV : {last}")
    logger.info(f"  Fetch window     : {start}  ->  {end}")

    if start >= end:
        logger.info("  Already up to date.")
        return

    anchor_try: float | None = None
    try:
        anchor_df = pd.read_csv(XAU_TRY_PRICES, usecols=["Date", "XAU_TRY"], parse_dates=["Date"])
        anchor_df = anchor_df.dropna(subset=["XAU_TRY"]).sort_values("Date")
        if not anchor_df.empty:
            anchor_try = float(anchor_df["XAU_TRY"].iloc[-1])
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.info(f"  Warning: could not read XAU anchor value: {exc}")

    use_borsapy = _wants_borsapy(source)
    if dry_run:
        provider = "borsapy" if use_borsapy else "yfinance"
        logger.info(f"  [dry-run] Would fetch XAU/TRY prices via {provider}.")
        return

    new_df = pd.DataFrame()

    if use_borsapy:
        try:
            xau_hist = bp.FX("ons-altin").history(start=start, end=end, interval="1d")
            usd_hist = bp.FX("USD").history(start=start, end=end, interval="1d")
            xau = xau_hist["Close"] if xau_hist is not None and not xau_hist.empty else pd.Series(dtype=float)
            usd_try = usd_hist["Close"] if usd_hist is not None and not usd_hist.empty else pd.Series(dtype=float)

            raw = pd.concat([xau, usd_try], axis=1)
            raw.columns = ["XAU_RAW", "USD_TRY"]
            xau_usd, xau_try, converted_try_rows = _normalize_borsapy_xau(
                xau_raw=raw["XAU_RAW"],
                usd_try=raw["USD_TRY"],
                anchor_try=anchor_try,
            )
            new_df = pd.concat([xau_usd, raw["USD_TRY"], xau_try], axis=1)
            new_df = new_df.dropna()
            new_df.index.name = "Date"
            if not new_df.empty:
                logger.info("  Data source      : borsapy")
                if converted_try_rows > 0:
                    logger.info(
                        "  Unit normalization: treated %d rows as TRY-quoted ons-altin",
                        converted_try_rows,
                    )
        except Exception as exc:
            if source == "borsapy":
                raise
            logger.info(f"  Warning: borsapy fetch failed, falling back to yfinance: {exc}")

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
        logger.info("  Data source      : yfinance")

    if new_df.empty:
        logger.info("  No valid new rows.")
        return

    try:
        existing = pd.read_csv(XAU_TRY_PRICES, parse_dates=["Date"], index_col="Date")
        combined = pd.concat([existing, new_df])
    except FileNotFoundError:
        combined = new_df
        
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    XAU_TRY_PRICES.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(XAU_TRY_PRICES)
    
    # Also save as parquet for faster loading
    combined.to_parquet(XAU_TRY_PARQUET)
    logger.info(f"  ✅ Parquet updated: {XAU_TRY_PARQUET.name}")

    new_last = combined.index.max().date()
    logger.info(f"  Appended {len(new_df)} rows  ->  new last date: {new_last}")


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

    logger.info(f"{'=' * 60}")
    logger.info(f"BIST DATA UPDATER  —  {dt.datetime.now():%Y-%m-%d %H:%M}")
    logger.info(f"{'=' * 60}")
    logger.info(f"Source mode: {args.source}")

    if args.source == "borsapy" and not _wants_borsapy("borsapy"):
        logger.error("ERROR: --source borsapy selected but borsapy integration is not available.")
        logger.info("Install borsapy and ensure `bist_quant.clients.borsapy_client` is importable.")
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
            logger.error(f"\n  ERROR in {label}: {exc}")

    logger.info("\n" + "=" * 60)
    if failures:
        logger.info(f"UPDATES COMPLETED WITH {len(failures)} FAILURE(S)")
        for label, exc in failures:
            logger.info(f"  - {label}: {exc}")
        logger.info("=" * 60)
        return 1

    logger.info("ALL UPDATES COMPLETE")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
