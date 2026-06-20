"""Price data loading sub-loader.

Handles loading, caching, and panel construction for stock prices,
XAU/TRY, XU100, USD/TRY, and oil prices from borsapy, local files,
and other data sources.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from bist_quant.common.data_paths import DataPaths
    from bist_quant.common.panel_cache import PanelCache

logger = logging.getLogger(__name__)


def _normalize_dt_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Strip timezone and floor to midnight for reliable date alignment."""
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.floor("D")


def _normalize_dt_series(s: pd.Series) -> pd.Series:
    """Strip timezone and floor to midnight for a datetime Series."""
    if hasattr(s.dt, "tz") and s.dt.tz is not None:
        s = s.dt.tz_localize(None)
    return s.dt.floor("D")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _canonical_csv_path(path: Path) -> Path:
    text = str(path)
    if text.endswith(".csv.gz"):
        return path.with_suffix("")
    if path.suffix == ".parquet":
        return path.with_suffix(".csv")
    if path.suffix == ".csv":
        return path
    return path.with_suffix(".csv")


def _price_source_candidates(path: Path) -> list[Path]:
    csv_path = _canonical_csv_path(path)
    return [
        csv_path.with_suffix(".parquet"),
        Path(f"{csv_path}.gz"),
        csv_path,
    ]


class PriceLoader:
    """Load and cache price data.

    This loader is owned by :class:`DataLoader` and should not be
    instantiated directly.
    """

    def __init__(
        self,
        paths: DataPaths,
        data_dir: Path,
        data_source_priority: str,
        borsapy_adapter: Any,
        panel_cache: PanelCache | None = None,
    ) -> None:
        self.paths = paths
        self.data_dir = data_dir
        self._data_source_priority = data_source_priority
        self._borsapy_adapter = borsapy_adapter
        self._panel_cache = panel_cache

        # Caches
        self._prices: pd.DataFrame | None = None
        self._close_df: pd.DataFrame | None = None
        self._open_df: pd.DataFrame | None = None
        self._volume_df: pd.DataFrame | None = None
        self._volume_lookback: int | None = None
        self._xautry_prices: pd.Series | None = None
        self._xu100_prices: pd.Series | None = None
        self._oil_prices: pd.DataFrame | None = None
        self._native_borsapy_module = None

    # ------------------------------------------------------------------
    # Core price loading
    # ------------------------------------------------------------------

    def load_prices(
        self,
        prices_file: Path | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        symbols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Load stock prices.

        In ``auto`` / ``borsapy`` mode the **only** local data source is
        ``borsapy_cache/``.  ``local`` mode preserves the legacy behaviour
        (read from ``data/bist_prices_full.*``).
        """
        if self._prices is None:
            logger.info("\n📊 Loading price data...")

            # --- Borsapy / cache path (auto or borsapy mode) ---
            if prices_file is None and self._should_use_borsapy_for("prices"):
                borsapy_prices = self._load_prices_via_borsapy(
                    symbols=symbols,
                    index="XUTUM",
                )
                if not borsapy_prices.empty:
                    self._prices = borsapy_prices
                    logger.info(
                        "  ✅ Loaded %d price records via borsapy (cache)",
                        len(self._prices),
                    )
                else:
                    raise RuntimeError(
                        "Borsapy price fetch returned no data. "
                        "Run 'python -m bist_quant.cli.cache_cli warm' to "
                        "populate the cache, or set BIST_DATA_SOURCE=local "
                        "to use old local files."
                    )

            # --- Legacy local file path (only when mode=local) ---
            if self._prices is None:
                requested_path = (
                    Path(prices_file) if prices_file is not None else self.paths.prices_file
                )
                source = next(
                    (c for c in _price_source_candidates(requested_path) if c.exists()), None
                )
                if source is None:
                    candidates = ", ".join(
                        str(c) for c in _price_source_candidates(requested_path)
                    )
                    raise FileNotFoundError(f"Price file not found. Tried: {candidates}")

                if source.suffix == ".parquet":
                    logger.info(f"  📦 Using legacy Parquet: {source.name}")
                    self._prices = pd.read_parquet(
                        source,
                        columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"],
                    )
                elif source.suffix == ".gz":
                    logger.info(f"  🗜️  Using legacy CSV.GZ: {source.name}")
                    self._prices = pd.read_csv(
                        source,
                        compression="gzip",
                        usecols=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"],
                    )
                else:
                    logger.info(f"  📄 Using legacy CSV: {source.name}")
                    self._prices = pd.read_csv(
                        source,
                        usecols=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"],
                    )
                if "Date" in self._prices.columns:
                    self._prices["Date"] = pd.to_datetime(self._prices["Date"], errors="coerce")
                logger.info(f"  ✅ Loaded {len(self._prices)} price records (legacy)")

        prices = self._prices.copy()
        if start_date is not None:
            if "Date" in prices.columns:
                prices = prices[pd.to_datetime(prices["Date"], errors="coerce") >= start_date]
            elif isinstance(prices.index, pd.DatetimeIndex):
                prices = prices[prices.index >= start_date]
        if end_date is not None:
            if "Date" in prices.columns:
                prices = prices[pd.to_datetime(prices["Date"], errors="coerce") <= end_date]
            elif isinstance(prices.index, pd.DatetimeIndex):
                prices = prices[prices.index <= end_date]

        if symbols:
            normalized = {str(symbol).upper().split(".")[0] for symbol in symbols}
            if "Ticker" in prices.columns:
                tickers = prices["Ticker"].astype(str).str.upper().str.split(".").str[0]
                prices = prices[tickers.isin(normalized)]
            elif not prices.empty:
                keep = [
                    col for col in prices.columns if str(col).upper().split(".")[0] in normalized
                ]
                prices = prices[keep]

        return prices

    def build_close_panel(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Build close price panel (Date x Ticker)."""
        if self._close_df is None:
            logger.info("  Building close price panel...")
            prices = prices.copy()
            prices["Ticker"] = prices["Ticker"].apply(lambda x: str(x).split(".")[0].upper())
            close_df = prices.pivot_table(
                index="Date",
                columns="Ticker",
                values="Close",
                aggfunc="last",
            ).sort_index()
            self._close_df = close_df
            logger.info(f"  ✅ Close panel: {close_df.shape[0]} days × {close_df.shape[1]} tickers")
        return self._close_df

    def build_open_panel(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Build open price panel (Date x Ticker)."""
        if self._open_df is None:
            logger.info("  Building open price panel...")
            prices = prices.copy()
            prices["Ticker"] = prices["Ticker"].apply(lambda x: str(x).split(".")[0].upper())
            open_df = prices.pivot_table(
                index="Date",
                columns="Ticker",
                values="Open",
                aggfunc="last",
            ).sort_index()
            self._open_df = open_df
            logger.info(f"  ✅ Open panel: {open_df.shape[0]} days × {open_df.shape[1]} tickers")
        return self._open_df

    def build_volume_panel(self, prices: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
        """Build rolling median volume panel."""
        panel_cache = self._panel_cache
        cache_key = None
        if panel_cache is not None:
            cache_key = panel_cache.make_key(
                "volume",
                lookback=int(lookback),
                rows=int(len(prices)),
                date_start=(
                    str(pd.to_datetime(prices["Date"], errors="coerce").min())
                    if "Date" in prices.columns
                    else None
                ),
                date_end=(
                    str(pd.to_datetime(prices["Date"], errors="coerce").max())
                    if "Date" in prices.columns
                    else None
                ),
                ticker_count=(
                    int(prices["Ticker"].nunique()) if "Ticker" in prices.columns else None
                ),
            )
            cached = panel_cache.get(cache_key)
            if isinstance(cached, pd.DataFrame):
                self._volume_df = cached
                self._volume_lookback = int(lookback)
                return cached

        if self._volume_df is None or self._volume_lookback != int(lookback):
            logger.info(f"  Building volume panel (lookback={lookback})...")
            prices = prices.copy()
            prices["Ticker"] = prices["Ticker"].apply(lambda x: str(x).split(".")[0].upper())
            vol_pivot = prices.pivot_table(
                index="Date",
                columns="Ticker",
                values="Volume",
                aggfunc="last",
            ).sort_index()

            # Drop holiday rows
            valid_pct = vol_pivot.notna().mean(axis=1)
            holiday_mask = valid_pct < 0.5
            if holiday_mask.any():
                vol_clean = vol_pivot.loc[~holiday_mask]
            else:
                vol_clean = vol_pivot

            median_adv = vol_clean.rolling(lookback, min_periods=lookback).median()
            median_adv = median_adv.reindex(vol_pivot.index).ffill()
            self._volume_df = median_adv
            self._volume_lookback = int(lookback)
            if panel_cache is not None and cache_key is not None:
                panel_cache.set(cache_key, median_adv)
            logger.info(
                f"  ✅ Volume panel: {median_adv.shape[0]} days × {median_adv.shape[1]} tickers"
            )
        return self._volume_df

    # ------------------------------------------------------------------
    # Benchmark / FX / commodity loading
    # ------------------------------------------------------------------

    def load_xautry_prices(
        self,
        csv_path: Path | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.Series:
        """Load XAU/TRY prices."""
        if self._xautry_prices is None:
            logger.info("\n💰 Loading XAU/TRY prices...")
            target_path = Path(csv_path) if csv_path is not None else self.paths.gold_try_file
            if target_path.suffix == ".parquet":
                df = pd.read_parquet(target_path)
                if "Date" not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                    df = df.reset_index()
            else:
                df = pd.read_csv(target_path, parse_dates=["Date"])
            if "Date" in df.columns:
                df["Date"] = _normalize_dt_series(pd.to_datetime(df["Date"], errors="coerce"))
            if "XAU_TRY" not in df.columns:
                # Try common column name variants
                for col in ("xau_try", "Close", "close", "price"):
                    if col in df.columns:
                        df = df.rename(columns={col: "XAU_TRY"})
                        break
            if "XAU_TRY" not in df.columns:
                raise ValueError(
                    f"XAU_TRY column not found in {target_path.name}. Columns: {list(df.columns)}"
                )
            if "Date" in df.columns:
                series = df.set_index("Date")["XAU_TRY"].astype(float)
                series.index = _normalize_dt_index(series.index)
            else:
                series = df["XAU_TRY"].astype(float)
                if isinstance(series.index, pd.DatetimeIndex):
                    series.index = _normalize_dt_index(series.index)

            series.name = "XAU_TRY"
            self._xautry_prices = series
            logger.info(f"  ✅ Loaded {len(series)} XAU/TRY observations")

        series = self._xautry_prices
        if start_date is not None:
            series = series.loc[series.index >= pd.Timestamp(start_date).floor("D")]
        if end_date is not None:
            series = series.loc[series.index <= pd.Timestamp(end_date).floor("D")]
        return series

    def load_xu100_prices(self, csv_path: Path | None = None) -> pd.Series:
        """Load XU100 benchmark prices (borsapy-first with local fallback)."""
        if self._xu100_prices is None:
            logger.info("\n📊 Loading XU100 benchmark...")

            # Try borsapy first
            if csv_path is None and self._should_use_borsapy_for("xu100"):
                try:
                    hist = self._borsapy_adapter.client
                    if hist is not None:
                        xu100_df = hist.get_history("XU100", period="5y", interval="1d")
                        if xu100_df is not None and not xu100_df.empty:
                            if "Close" in xu100_df.columns:
                                xu100_df.index = _normalize_dt_index(
                                    pd.to_datetime(xu100_df.index, errors="coerce")
                                )
                                self._xu100_prices = xu100_df["Close"].sort_index()
                                logger.info(
                                    "  ✅ Loaded %d XU100 observations via borsapy",
                                    len(self._xu100_prices),
                                )
                                return self._xu100_prices
                except Exception as exc:
                    logger.warning("  ⚠️  Borsapy XU100 fetch failed: %s", exc)

            # Local file fallback
            target_path = Path(csv_path) if csv_path is not None else self.paths.xu100_prices
            if target_path.suffix == ".parquet":
                df = pd.read_parquet(target_path)
            else:
                df = pd.read_csv(target_path)
            if "Date" in df.columns:
                df["Date"] = _normalize_dt_series(pd.to_datetime(df["Date"], errors="coerce"))
                df = df.set_index("Date")
            elif isinstance(df.index, pd.DatetimeIndex):
                df.index = _normalize_dt_index(df.index)
            df = df.sort_index()
            # Prefer close for return calculations and benchmark alignment.
            if "Close" in df.columns:
                self._xu100_prices = df["Close"]
            elif "close" in df.columns:
                self._xu100_prices = df["close"]
            else:
                self._xu100_prices = df.iloc[:, 0]
            logger.info(f"  ✅ Loaded {len(self._xu100_prices)} XU100 observations")
        return self._xu100_prices

    def load_xu100(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.Series:
        """Load XU100 series using canonical DataPaths resolution."""
        series = self.load_xu100_prices()
        out = series.copy()
        if start_date is not None:
            out = out[out.index >= start_date]
        if end_date is not None:
            out = out[out.index <= end_date]
        return out

    def load_usdtry(self) -> pd.DataFrame:
        """Load USD/TRY exchange rate data."""
        logger.info("\n💱 Loading USD/TRY data...")
        usdtry_file = self.paths.usdtry_file

        if not usdtry_file.exists():
            logger.warning(f"  ⚠️  USD/TRY file not found: {usdtry_file}")
            return pd.DataFrame()

        if usdtry_file.suffix == ".parquet":
            df = pd.read_parquet(usdtry_file)
            if "Date" not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
            if "Date" in df.columns:
                df["Date"] = _normalize_dt_series(pd.to_datetime(df["Date"], errors="coerce"))
                df = df.set_index("Date")
            df = df.sort_index()
        else:
            df = pd.read_csv(usdtry_file, parse_dates=["Date"])
            if "Date" in df.columns:
                df["Date"] = _normalize_dt_series(pd.to_datetime(df["Date"], errors="coerce"))
                df = df.set_index("Date").sort_index()

        # Rename column to 'Close' for consistency
        if "USDTRY" in df.columns:
            df = df.rename(columns={"USDTRY": "Close"})

        logger.info(f"  ✅ Loaded {len(df)} USD/TRY observations")
        return df

    def load_oil_prices(self) -> pd.DataFrame | None:
        """Load daily WTI and Brent crude oil prices.

        Reads from the borsapy_cache commodities parquet produced by
        ``fetchers/fetch_oil_prices.py``.  Returns ``None`` when no
        cached data is available.
        """
        if self._oil_prices is not None:
            return self._oil_prices

        cache_pq = self.paths.borsapy_cache_dir / "commodities" / "oil_daily.parquet"
        fallback_csv = self.data_dir / "oil_prices_daily.csv"

        target = cache_pq if cache_pq.exists() else (fallback_csv if fallback_csv.exists() else None)
        if target is None:
            logger.debug("Oil price data not found (checked %s)", cache_pq)
            self._oil_prices = None
            return None

        try:
            if target.suffix == ".parquet":
                df = pd.read_parquet(target)
            else:
                df = pd.read_csv(target, parse_dates=["Date"])
            if "Date" in df.columns:
                df["Date"] = _normalize_dt_series(pd.to_datetime(df["Date"], errors="coerce"))
                df = df.set_index("Date")
            elif isinstance(df.index, pd.DatetimeIndex):
                df.index = _normalize_dt_index(df.index)
            df = df.sort_index()
            self._oil_prices = df
            logger.info(f"  ✅ Loaded {len(df)} oil price observations")
            return self._oil_prices
        except Exception as exc:
            logger.warning("Failed to load oil prices: %s", exc)
            self._oil_prices = None
            return None

    # ------------------------------------------------------------------
    # Borsapy-native helpers
    # ------------------------------------------------------------------

    def load_prices_borsapy(
        self,
        symbols: list[str] | None = None,
        period: str = "5y",
        index: str = "XU100",
    ) -> pd.DataFrame:
        resolved_symbols = [str(item).upper().split(".")[0] for item in (symbols or []) if item]
        if not resolved_symbols and symbols is None:
            resolved_symbols = self._borsapy_adapter.get_index_components(index=index)
        if not resolved_symbols and symbols is None:
            resolved_symbols = self._resolve_index_components_native(index=index)

        adapter_result = self._borsapy_adapter.load_prices(
            symbols=resolved_symbols or symbols,
            period=period,
            index=index,
        )
        if not adapter_result.empty:
            return adapter_result

        if not resolved_symbols:
            logger.warning("  ⚠️  No symbols resolved for borsapy load; returning empty frame")
            return pd.DataFrame()

        logger.warning("  ⚠️  Adapter returned no data, trying native borsapy fallback...")
        fallback = self._borsapy_download_to_long(
            symbols=resolved_symbols,
            period=period,
            interval="1d",
        )
        if fallback.empty:
            logger.warning("  ⚠️  Native borsapy fallback also returned no data")
        else:
            loaded = fallback["Ticker"].dropna().nunique() if "Ticker" in fallback.columns else 0
            logger.info(
                f"  ✅ Native fallback loaded {len(fallback)} price records for {loaded}/{len(resolved_symbols)} tickers"
            )
        return fallback

    def _resolve_index_components_native(self, index: str = "XU100") -> list[str]:
        """Best-effort index component resolution via native borsapy module."""
        if self._native_borsapy_module is None:
            try:
                import borsapy as bp  # type: ignore[import-not-found]

                self._native_borsapy_module = bp
            except Exception:
                return []

        bp = self._native_borsapy_module
        if bp is None or not hasattr(bp, "index"):
            return []

        try:
            idx = bp.index(index)
        except Exception as exc:
            logger.warning(f"  ⚠️  Native borsapy index lookup failed for {index}: {exc}")
            return []

        symbols = getattr(idx, "component_symbols", None)
        if isinstance(symbols, list) and symbols:
            return [str(item).upper().split(".")[0] for item in symbols if item]

        components = getattr(idx, "components", None)
        if isinstance(components, list):
            out: list[str] = []
            seen: set[str] = set()
            for item in components:
                if not isinstance(item, dict):
                    continue
                symbol = str(item.get("symbol", "")).upper().split(".")[0]
                if not symbol or symbol in seen:
                    continue
                seen.add(symbol)
                out.append(symbol)
            return out

        return []

    def _borsapy_download_to_long(
        self,
        symbols: list[str],
        period: str = "5y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Download prices via native borsapy and normalize to long format."""
        if not symbols:
            return pd.DataFrame()

        if self._native_borsapy_module is None:
            try:
                import borsapy as bp  # type: ignore[import-not-found]

                self._native_borsapy_module = bp
                logger.info("  ✅ Native borsapy module initialized")
            except Exception as exc:
                logger.warning(f"  ⚠️  Native borsapy unavailable for fallback download: {exc}")
                return pd.DataFrame()

        bp = self._native_borsapy_module
        if bp is None:
            return pd.DataFrame()

        try:
            raw = bp.download(
                symbols,
                period=period,
                interval=interval,
                group_by="ticker",
                progress=False,
            )
        except Exception as exc:
            logger.warning(f"  ⚠️  Native borsapy fallback download failed: {exc}")
            return pd.DataFrame()

        if raw is None or raw.empty:
            return pd.DataFrame()

        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        frames: list[pd.DataFrame] = []

        if isinstance(raw.columns, pd.MultiIndex):
            lvl0 = raw.columns.get_level_values(0)
            if {"Open", "High", "Low", "Close"}.issubset(set(lvl0)):
                raw = raw.swaplevel(0, 1, axis=1).sort_index(axis=1)

            for ticker in dict.fromkeys(raw.columns.get_level_values(0)):
                sub = raw[ticker]
                if not isinstance(sub, pd.DataFrame) or sub.empty:
                    continue
                sub = sub.rename_axis("Date").reset_index()
                sub["Ticker"] = str(ticker).upper().split(".")[0]
                for col in required_cols:
                    if col not in sub.columns:
                        sub[col] = pd.NA
                frames.append(sub[["Date", "Ticker", *required_cols]])
        else:
            sub = raw.rename_axis("Date").reset_index()
            ticker = symbols[0]
            sub["Ticker"] = str(ticker).upper().split(".")[0]
            for col in required_cols:
                if col not in sub.columns:
                    sub[col] = pd.NA
            frames.append(sub[["Date", "Ticker", *required_cols]])

        if not frames:
            return pd.DataFrame()

        out = pd.concat(frames, ignore_index=True)
        if "Date" in out.columns:
            out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        return out

    def _load_prices_via_borsapy(
        self,
        symbols: list[str] | None = None,
        period: str = "5y",
        index: str = "XUTUM",
    ) -> pd.DataFrame:
        """Attempt to load prices through the borsapy adapter + disk cache."""
        try:
            result = self.load_prices_borsapy(
                symbols=symbols,
                period=period,
                index=index,
            )
            if not result.empty:
                return result
        except Exception as exc:
            logger.warning("  ⚠️  Borsapy price fetch failed: %s", exc)
        return pd.DataFrame()

    def _should_use_borsapy_for(self, category: str = "prices") -> bool:
        """Determine whether borsapy should be used for *category*."""
        prio = self._data_source_priority
        if prio == "local":
            return False
        return True

    # ------------------------------------------------------------------
    # Public cache accessors (used by data_manager.py)
    # ------------------------------------------------------------------

    @property
    def prices(self) -> pd.DataFrame | None:
        return self._prices

    @prices.setter
    def prices(self, value: pd.DataFrame | None) -> None:
        """Allow external caching (e.g. from consolidated panel)."""
        self._prices = value
