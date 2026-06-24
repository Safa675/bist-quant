"""Helper module for fetching and formatting price data via borsapy."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import borsapy as bp

logger = logging.getLogger(__name__)


def get_history(
    client: Any,
    symbol: str,
    period: str = "5y",
    interval: str = "1d",
    start: datetime | str | None = None,
    end: datetime | str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch OHLCV history for a single ticker."""
    symbol = client._normalize_symbol(symbol)
    cache_key = f"{symbol}_{period}_{interval}"

    # Try disk cache first
    if use_cache and client._disk_cache is not None:
        cached = client._disk_cache.get_dataframe("prices", cache_key)
        if cached is not None and not cached.empty:
            logger.debug("  Cache hit for %s history", symbol)
            return cached

    ticker = client.get_ticker(symbol)
    try:
        df = ticker.history(period=period, interval=interval, start=start, end=end)
        if df is not None and not df.empty:
            df["Ticker"] = symbol
            # Persist to disk cache
            if use_cache and client._disk_cache is not None:
                client._disk_cache.set_dataframe("prices", cache_key, df)
        return df
    except Exception as e:
        logger.info(f"  Warning: Failed to fetch history for {symbol}: {e}")
        return pd.DataFrame()


def batch_download(
    client: Any,
    symbols: list[str],
    period: str = "5y",
    interval: str = "1d",
    start: datetime | str | None = None,
    end: datetime | str | None = None,
    group_by: str = "column",
    progress: bool = False,
) -> pd.DataFrame:
    """Batch download OHLCV data for multiple tickers."""
    symbols = [client._normalize_symbol(s) for s in symbols]
    try:
        return bp.download(
            symbols,
            period=period,
            interval=interval,
            start=start,
            end=end,
            group_by=group_by,
            progress=progress,
        )
    except Exception as e:
        logger.info(f"  Warning: Batch download failed: {e}")
        return pd.DataFrame()


def to_long_ohlcv(
    client: Any,
    downloaded: pd.DataFrame,
    symbol_hint: str | None = None,
    add_is_suffix: bool = False,
) -> pd.DataFrame:
    """Convert borsapy download output to long format (Date, Ticker, OHLCV...)."""
    if downloaded is None or downloaded.empty:
        return pd.DataFrame(columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"])

    data = downloaded.copy()
    records: list[pd.DataFrame] = []

    def _format_ticker(ticker: str) -> str:
        base = client._normalize_symbol(ticker)
        return f"{base}.IS" if add_is_suffix else base

    def _frame_to_records(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
        part = frame.copy()
        part.columns = [str(c).title() for c in part.columns]
        part = part.reset_index()

        if "Date" not in part.columns and "index" in part.columns:
            part = part.rename(columns={"index": "Date"})

        part["Ticker"] = _format_ticker(ticker)

        wanted = ["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
        for col in wanted:
            if col not in part.columns:
                part[col] = pd.NA

        return part[wanted]

    if isinstance(data.columns, pd.MultiIndex):
        level0 = [str(v) for v in data.columns.get_level_values(0)]
        level1 = [str(v) for v in data.columns.get_level_values(1)]
        ohlcv_set = set(client._ohlcv_columns)

        # group_by="ticker" -> (Ticker, Field)
        if set(level0) - ohlcv_set:
            tickers = list(dict.fromkeys(level0))
            for ticker in tickers:
                try:
                    records.append(_frame_to_records(data[ticker], ticker))
                except Exception:
                    continue
        # group_by="column" -> (Field, Ticker)
        else:
            tickers = list(dict.fromkeys(level1))
            for ticker in tickers:
                try:
                    records.append(_frame_to_records(data.xs(ticker, axis=1, level=1), ticker))
                except Exception:
                    continue
    else:
        fallback_ticker = client._normalize_symbol(symbol_hint or "UNKNOWN")
        records.append(_frame_to_records(data, fallback_ticker))

    if not records:
        return pd.DataFrame(columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"])

    result = pd.concat(records, ignore_index=True)
    result["Date"] = pd.to_datetime(result["Date"], errors="coerce")

    # Keep data aligned with the rest of the project (timezone-naive timestamps).
    if pd.api.types.is_datetime64tz_dtype(result["Date"]):
        result["Date"] = result["Date"].dt.tz_convert(None)

    result = result.dropna(subset=["Date"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return result


def batch_download_to_long(
    client: Any,
    symbols: list[str],
    period: str = "5y",
    interval: str = "1d",
    start: datetime | str | None = None,
    end: datetime | str | None = None,
    group_by: str = "ticker",
    add_is_suffix: bool = False,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Batch download and return data in long OHLCV format."""
    normalized = [client._normalize_symbol(s) for s in symbols]

    # --- Disk-cache partition -----------------------------------------
    cached_frames: list[pd.DataFrame] = []
    fetch_symbols = list(normalized)

    if use_cache and client._disk_cache is not None:
        for sym in normalized:
            cache_key = f"{sym}_{period}_{interval}"
            frame = client._disk_cache.get_dataframe("prices", cache_key)
            if frame is not None and not frame.empty:
                cached_frames.append(frame)
                fetch_symbols.remove(sym)
        if cached_frames:
            logger.info(
                "  📦 Disk cache hit for %d/%d symbols",
                len(cached_frames),
                len(normalized),
            )

    # --- API fetch for missing symbols --------------------------------
    if fetch_symbols:
        logger.info(
            "  ⬇️  Downloading %d symbols from TradingView "
            "(this may take a while — ~%.0f min estimated)...",
            len(fetch_symbols),
            len(fetch_symbols) * 0.5 / 60,
        )
        downloaded = batch_download(
            client,
            fetch_symbols,
            period=period,
            interval=interval,
            start=start,
            end=end,
            group_by=group_by,
            progress=True,
        )
        result = to_long_ohlcv(client, downloaded, add_is_suffix=add_is_suffix)
    else:
        result = pd.DataFrame()

    # Fallback fetch for symbols omitted by a partial batch response.
    if result.empty:
        present: set[str] = set()
    else:
        present = {
            client._normalize_symbol(t)
            for t in result["Ticker"].dropna().astype(str)
        }

    missing = [s for s in fetch_symbols if s not in present]
    for symbol in missing:
        single = get_history(
            client,
            symbol,
            period=period,
            interval=interval,
            start=start,
            end=end,
            use_cache=False,
        )
        if single is None or single.empty:
            continue
        long_single = to_long_ohlcv(
            client,
            single.drop(columns=["Ticker"], errors="ignore"),
            symbol_hint=symbol,
            add_is_suffix=add_is_suffix,
        )
        if not long_single.empty:
            result = pd.concat([result, long_single], ignore_index=True)

    # --- Persist newly fetched data to disk cache ---------------------
    if use_cache and client._disk_cache is not None and not result.empty:
        for ticker_val in result["Ticker"].dropna().unique():
            ticker_str = client._normalize_symbol(str(ticker_val))
            cache_key = f"{ticker_str}_{period}_{interval}"
            ticker_slice = result[result["Ticker"] == ticker_val]
            client._disk_cache.set_dataframe("prices", cache_key, ticker_slice)

    # --- Combine cached + freshly fetched data ------------------------
    if cached_frames:
        # Reset index on cached frames before concatenation
        normalized_frames = []
        for frame in cached_frames:
            if isinstance(frame.index, pd.DatetimeIndex) or frame.index.name == "Date":
                frame = frame.reset_index()
            normalized_frames.append(frame)

        # Also reset result if it has data
        if not result.empty and (isinstance(result.index, pd.DatetimeIndex) or result.index.name == "Date"):
            result = result.reset_index()

        result = pd.concat([*normalized_frames, result], ignore_index=True)

    if result.empty:
        return result

    # Ensure Date column exists for deduplication
    if "Date" not in result.columns:
        logger.warning("Date column missing after concatenation, returning empty")
        return pd.DataFrame()

    return result.drop_duplicates(subset=["Date", "Ticker"], keep="last").sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)


def save_to_cache(
    client: Any,
    data: pd.DataFrame,
    filename: str,
    format: str = "parquet",
) -> None:
    """Save DataFrame to cache directory."""
    if format == "parquet":
        path = client.cache_dir / f"{filename}.parquet"
        data.to_parquet(path, compression="zstd")
    else:
        path = client.cache_dir / f"{filename}.csv"
        data.to_csv(path)

    logger.info(f"  Saved to cache: {path}")


def load_from_cache(
    client: Any,
    filename: str,
    format: str = "parquet",
    max_age_hours: int = 24,
) -> Optional[pd.DataFrame]:
    """Load DataFrame from cache if fresh enough."""
    ext = "parquet" if format == "parquet" else "csv"
    path = client.cache_dir / f"{filename}.{ext}"

    if not path.exists():
        return None

    # Check age
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    if age.total_seconds() > max_age_hours * 3600:
        return None

    if format == "parquet":
        return pd.read_parquet(path)
    else:
        return pd.read_csv(path)
