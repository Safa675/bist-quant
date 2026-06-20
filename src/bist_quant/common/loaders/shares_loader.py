"""Shares outstanding and market-cap loading sub-loader.

Handles loading and caching of shares outstanding panels from
consolidated files, fundamentals parquet, and isyatirim sources.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from bist_quant.common.data_paths import DataPaths
    from bist_quant.common.panel_cache import PanelCache

logger = logging.getLogger(__name__)


class SharesLoader:
    """Load and cache shares outstanding data.

    This loader is owned by :class:`DataLoader` and should not be
    instantiated directly.
    """

    def __init__(
        self,
        paths: DataPaths,
        data_dir: Path,
        isyatirim_dir: Path,
        panel_cache: PanelCache | None = None,
        price_loader: Any | None = None,
        fundamentals_loader: Any | None = None,
    ) -> None:
        self.paths = paths
        self.data_dir = data_dir
        self.isyatirim_dir = isyatirim_dir
        self._panel_cache = panel_cache
        self._price_loader = price_loader
        self._fundamentals_loader = fundamentals_loader

        # Caches
        self._shares_consolidated: pd.DataFrame | None = None
        self._isyatirim_parquet: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_shares_outstanding_panel(self) -> pd.DataFrame:
        """Load Date x Ticker shares outstanding panel."""
        panel_cache = self._panel_cache
        cache_key = None
        if panel_cache is not None:
            cache_key = panel_cache.make_key(
                "shares_outstanding",
                data_dir=str(self.data_dir),
            )
            cached = panel_cache.get(cache_key)
            if isinstance(cached, pd.DataFrame):
                self._shares_consolidated = cached

        if self._shares_consolidated is None:
            shares_csv = self.paths.shares_outstanding
            shares_parquet = shares_csv.with_suffix(".parquet")
            shares_csv_gz = shares_csv.with_suffix(f"{shares_csv.suffix}.gz")
            panel: pd.DataFrame | None = None

            if shares_parquet.exists():
                logger.info("  📦 Loading consolidated shares file (Parquet)...")
                panel = pd.read_parquet(shares_parquet)
            elif shares_csv_gz.exists() or shares_csv.exists():
                source = shares_csv_gz if shares_csv_gz.exists() else shares_csv
                logger.info(f"  📊 Loading consolidated shares file ({source.name})...")
                panel = pd.read_csv(
                    source,
                    index_col=0,
                    parse_dates=True,
                    compression="gzip" if source.suffix == ".gz" else "infer",
                )

            if panel is not None:
                if "Date" in panel.columns and not isinstance(panel.index, pd.DatetimeIndex):
                    panel = panel.set_index("Date")
                panel.index = pd.to_datetime(panel.index, errors="coerce")
                panel = panel.sort_index()
                panel.columns = [str(c).upper() for c in panel.columns]
                self._shares_consolidated = panel
                logger.info(f"  ✅ Loaded shares for {panel.shape[1]} tickers")
            else:
                # Fallback: build panel from consolidated isyatirim parquet
                isy = self._load_isyatirim_parquet()
                if isy is not None and not isy.empty:
                    try:
                        daily = isy[isy["sheet_type"] == "daily"]
                        required_cols = {"ticker", "HGDG_TARIH", "SERMAYE"}
                        if not daily.empty and required_cols.issubset(daily.columns):
                            panel = daily.pivot_table(
                                index="HGDG_TARIH",
                                columns="ticker",
                                values="SERMAYE",
                                aggfunc="last",
                            )
                            panel.index = pd.to_datetime(panel.index, errors="coerce")
                            panel = panel.sort_index()
                            panel.columns = [str(c).upper() for c in panel.columns]
                            self._shares_consolidated = panel
                            logger.info(
                                f"  ✅ Built shares panel from isyatirim parquet for {panel.shape[1]} tickers"
                            )
                        else:
                            self._shares_consolidated = None
                    except Exception as exc:
                        logger.warning(
                            f"  ⚠️  Failed to build shares panel from isyatirim parquet: {exc}"
                        )
                        self._shares_consolidated = None
                else:
                    logger.warning("  ⚠️  Consolidated shares file not found")
                    self._shares_consolidated = None

        if self._shares_consolidated is None:
            return pd.DataFrame()
        if panel_cache is not None and cache_key is not None:
            panel_cache.set(cache_key, self._shares_consolidated)
        return self._shares_consolidated

    def load_shares_outstanding(self, ticker: str) -> pd.Series:
        """Load shares outstanding from consolidated file or fundamentals (fast!)."""
        shares_panel = self.load_shares_outstanding_panel()
        if not shares_panel.empty and ticker in shares_panel.columns:
            return shares_panel[ticker].dropna()

        # Extract from fundamentals parquet (Ödenmiş Sermaye)
        try:
            from bist_quant.common.utils import (
                coerce_quarter_cols,
                get_consolidated_sheet,
                pick_row_from_sheet,
            )

            fund_parquet = (
                self._fundamentals_loader.load_fundamentals_parquet()
                if self._fundamentals_loader is not None
                else None
            )
            if fund_parquet is not None:
                bs = get_consolidated_sheet(fund_parquet, ticker, "Bilanço")
                if not bs.empty:
                    sermaye_row = pick_row_from_sheet(bs, ("Ödenmiş Sermaye", "ÖDENMİŞ SERMAYE"))
                    if sermaye_row is not None and not sermaye_row.empty:
                        sermaye_parsed = coerce_quarter_cols(sermaye_row)
                        if not sermaye_parsed.empty:
                            return sermaye_parsed.dropna()
        except Exception as e:
            logger.warning(f"Error extracting shares from fundamentals for {ticker}: {e}")

        # Fallback: consolidated isyatirim parquet (if available)
        isy = self._load_isyatirim_parquet()
        if isy is not None:
            try:
                daily = isy[(isy["ticker"] == ticker) & (isy["sheet_type"] == "daily")]
                if not daily.empty and "HGDG_TARIH" in daily.columns and "SERMAYE" in daily.columns:
                    series = daily.set_index("HGDG_TARIH")["SERMAYE"].dropna()
                    return series
            except Exception:
                pass

        # Fallback to individual Excel file (slow)
        excel_path = self.isyatirim_dir / f"{ticker}_2016_2026_daily_and_quarterly.xlsx"

        if not excel_path.exists():
            return pd.Series(dtype=float)

        try:
            df = pd.read_excel(excel_path, sheet_name="daily")
            if "HGDG_TARIH" not in df.columns or "SERMAYE" not in df.columns:
                return pd.Series(dtype=float)

            df["HGDG_TARIH"] = pd.to_datetime(df["HGDG_TARIH"])
            df = df.set_index("HGDG_TARIH").sort_index()
            return df["SERMAYE"].dropna()
        except Exception:
            return pd.Series(dtype=float)

    def load_market_caps(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        symbols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Load or compute Date x Ticker market capitalization panel."""
        if self._price_loader is None:
            return pd.DataFrame()

        prices = self._price_loader.load_prices(start_date=start_date, end_date=end_date, symbols=symbols)
        if prices.empty:
            return pd.DataFrame()

        if {"Date", "Ticker", "Close"}.issubset(set(prices.columns)):
            close_panel = self._price_loader.build_close_panel(prices)
        else:
            close_panel = prices.copy()

        shares = self.load_shares_outstanding_panel()
        if shares.empty:
            return close_panel

        shares = shares.reindex(close_panel.index).ffill()
        shares = shares.reindex(columns=close_panel.columns)
        return close_panel * shares

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_isyatirim_parquet(self) -> pd.DataFrame | None:
        """Load consolidated isyatirim prices parquet (used for shares fallback)."""
        if self._isyatirim_parquet is None:
            legacy_file = self.data_dir / "isyatirim_prices_consolidated.parquet"
            parquet_file = (
                self.paths.isyatirim_prices if self.paths.isyatirim_prices.exists() else legacy_file
            )
            if parquet_file.exists():
                logger.info("  📦 Loading consolidated isyatirim prices (Parquet)...")
                try:
                    self._isyatirim_parquet = pd.read_parquet(
                        parquet_file,
                        columns=["ticker", "sheet_type", "HGDG_TARIH", "SERMAYE"],
                    )
                except Exception:
                    frame = pd.read_parquet(
                        parquet_file,
                        columns=["HGDG_HS_KODU", "HGDG_TARIH", "SERMAYE"],
                    )
                    frame = frame.rename(columns={"HGDG_HS_KODU": "ticker"})
                    frame["sheet_type"] = "daily"
                    self._isyatirim_parquet = frame
        return self._isyatirim_parquet
