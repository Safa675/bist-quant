"""
Unified borsapy client for BIST data fetching.

Wraps borsapy with caching, error handling, and integration
with the existing data pipeline.
"""

from pathlib import Path
from datetime import datetime
from typing import Any, Optional
import warnings

import pandas as pd

try:
    import borsapy as bp
    BORSAPY_AVAILABLE = True
except ImportError:
    BORSAPY_AVAILABLE = False
    bp = None

warnings.filterwarnings("ignore")


class BorsapyClient:
    """
    Centralized borsapy interface with caching.

    Provides unified access to:
    - Stock prices and history
    - Fundamental data (financials, dividends, shareholders)
    - Index constituents
    - Technical indicators
    - Stock screening
    - Real-time quotes (15-min delayed by default)
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the borsapy client.

        Args:
            cache_dir: Directory for caching data. Defaults to data/borsapy_cache
        """
        if not BORSAPY_AVAILABLE:
            raise ImportError(
                "borsapy is not installed. Install with: pip install borsapy"
            )

        self.cache_dir = cache_dir or Path(__file__).parent.parent / "borsapy_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache for frequently accessed data
        self._ticker_cache: dict[str, bp.Ticker] = {}
        self._index_cache: dict[str, bp.Index] = {}
        self._index_components_cache: dict[str, list[str]] = {}
        self._ohlcv_columns = ("Open", "High", "Low", "Close", "Volume", "Adj Close")

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Normalize symbol strings to BIST short form (THYAO instead of THYAO.IS)."""
        return symbol.upper().split(".")[0]

    @staticmethod
    def _period_from_days(days_ahead: int) -> str:
        """Map day window to borsapy calendar period argument."""
        if days_ahead <= 1:
            return "1d"
        if days_ahead <= 7:
            return "1w"
        if days_ahead <= 14:
            return "2w"
        return "1mo"

    # -------------------------------------------------------------------------
    # Ticker Access
    # -------------------------------------------------------------------------

    def get_ticker(self, symbol: str) -> "bp.Ticker":
        """
        Get ticker object for a single stock.

        Args:
            symbol: Stock symbol (e.g., "THYAO", "AKBNK")

        Returns:
            borsapy Ticker object
        """
        symbol = self._normalize_symbol(symbol)

        if symbol not in self._ticker_cache:
            self._ticker_cache[symbol] = bp.Ticker(symbol)

        return self._ticker_cache[symbol]

    # -------------------------------------------------------------------------
    # Price Data
    # -------------------------------------------------------------------------

    def get_history(
        self,
        symbol: str,
        period: str = "5y",
        interval: str = "1d",
        start: datetime | str | None = None,
        end: datetime | str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV history for a single ticker.

        Args:
            symbol: Stock symbol
            period: Data period (e.g., "1y", "5y", "max")
            interval: Data interval (e.g., "1d", "1h", "1m")
            start: Optional start date
            end: Optional end date

        Returns:
            DataFrame with Date index and OHLCV columns
        """
        ticker = self.get_ticker(symbol)
        try:
            df = ticker.history(period=period, interval=interval, start=start, end=end)
            if df is not None and not df.empty:
                df["Ticker"] = symbol
            return df
        except Exception as e:
            print(f"  Warning: Failed to fetch history for {symbol}: {e}")
            return pd.DataFrame()

    def batch_download(
        self,
        symbols: list[str],
        period: str = "5y",
        interval: str = "1d",
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        group_by: str = "column",
        progress: bool = False,
    ) -> pd.DataFrame:
        """
        Batch download OHLCV data for multiple tickers.

        Args:
            symbols: List of stock symbols
            period: Data period
            interval: Data interval
            start: Optional start date (overrides period when provided)
            end: Optional end date
            group_by: How to organize output ("column" or "ticker")
            progress: Whether to show borsapy progress output

        Returns:
            DataFrame with multi-level columns or grouped by ticker
        """
        symbols = [self._normalize_symbol(s) for s in symbols]
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
            print(f"  Warning: Batch download failed: {e}")
            return pd.DataFrame()

    def to_long_ohlcv(
        self,
        downloaded: pd.DataFrame,
        symbol_hint: str | None = None,
        add_is_suffix: bool = False,
    ) -> pd.DataFrame:
        """
        Convert borsapy download output to long format (Date, Ticker, OHLCV...).

        Handles all supported borsapy output layouts:
        - MultiIndex columns with group_by="ticker"
        - MultiIndex columns with group_by="column"
        - Single ticker (single-level columns)
        """
        if downloaded is None or downloaded.empty:
            return pd.DataFrame(columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"])

        data = downloaded.copy()
        records: list[pd.DataFrame] = []

        def _format_ticker(ticker: str) -> str:
            base = self._normalize_symbol(ticker)
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
            ohlcv_set = set(self._ohlcv_columns)

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
            fallback_ticker = self._normalize_symbol(symbol_hint or "UNKNOWN")
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
        self,
        symbols: list[str],
        period: str = "5y",
        interval: str = "1d",
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        group_by: str = "ticker",
        add_is_suffix: bool = False,
    ) -> pd.DataFrame:
        """Batch download and return data in long OHLCV format."""
        normalized = [self._normalize_symbol(s) for s in symbols]
        downloaded = self.batch_download(
            normalized,
            period=period,
            interval=interval,
            start=start,
            end=end,
            group_by=group_by,
        )
        result = self.to_long_ohlcv(downloaded, add_is_suffix=add_is_suffix)

        # Fallback fetch for symbols omitted by a partial batch response.
        if result.empty:
            present = set()
        else:
            present = {
                self._normalize_symbol(t)
                for t in result["Ticker"].dropna().astype(str)
            }

        missing = [s for s in normalized if s not in present]
        for symbol in missing:
            single = self.get_history(symbol, period=period, interval=interval, start=start, end=end)
            if single is None or single.empty:
                continue
            long_single = self.to_long_ohlcv(
                single.drop(columns=["Ticker"], errors="ignore"),
                symbol_hint=symbol,
                add_is_suffix=add_is_suffix,
            )
            if not long_single.empty:
                result = pd.concat([result, long_single], ignore_index=True)

        if result.empty:
            return result

        return result.drop_duplicates(subset=["Date", "Ticker"], keep="last").sort_values(
            ["Ticker", "Date"]
        ).reset_index(drop=True)

    def get_fast_info(self, symbol: str) -> dict:
        """
        Get current quote/fast info for a ticker.

        Note: Data has ~15 minute delay unless using TradingView Pro.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with current price, volume, market cap, etc.
        """
        ticker = self.get_ticker(symbol)
        try:
            return dict(ticker.fast_info) if ticker.fast_info else {}
        except Exception as e:
            print(f"  Warning: Failed to get fast_info for {symbol}: {e}")
            return {}

    # -------------------------------------------------------------------------
    # Fundamental Data
    # -------------------------------------------------------------------------

    def get_financials(self, symbol: str) -> dict[str, pd.DataFrame]:
        """
        Get all financial statements for a ticker.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with balance_sheet, income_stmt, cashflow DataFrames
        """
        ticker = self.get_ticker(symbol)
        result = {}

        try:
            result["balance_sheet"] = ticker.balance_sheet
        except Exception:
            result["balance_sheet"] = pd.DataFrame()

        try:
            result["income_stmt"] = ticker.income_stmt
        except Exception:
            result["income_stmt"] = pd.DataFrame()

        try:
            result["cashflow"] = ticker.cashflow
        except Exception:
            result["cashflow"] = pd.DataFrame()

        return result

    def get_dividends(self, symbol: str) -> pd.DataFrame:
        """Get dividend history for a ticker."""
        ticker = self.get_ticker(symbol)
        try:
            return ticker.dividends if ticker.dividends is not None else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def get_splits(self, symbol: str) -> pd.DataFrame:
        """Get stock split history for a ticker."""
        ticker = self.get_ticker(symbol)
        try:
            return ticker.splits if ticker.splits is not None else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def get_major_holders(self, symbol: str) -> pd.DataFrame:
        """Get major shareholder composition."""
        ticker = self.get_ticker(symbol)
        try:
            return ticker.major_holders if ticker.major_holders is not None else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def get_analyst_targets(self, symbol: str) -> dict:
        """Get analyst price targets and recommendations."""
        ticker = self.get_ticker(symbol)
        try:
            return {
                "price_targets": ticker.analyst_price_targets,
                "recommendations": ticker.recommendations,
            }
        except Exception:
            return {"price_targets": None, "recommendations": None}

    def get_news(self, symbol: str, limit: int = 10) -> list[dict]:
        """Get recent KAP announcements/news for a ticker."""
        ticker = self.get_ticker(symbol)
        try:
            news = ticker.news
            return news[:limit] if news else []
        except Exception:
            return []

    # -------------------------------------------------------------------------
    # Index Data
    # -------------------------------------------------------------------------

    def get_index(self, index: str = "XU100") -> "bp.Index":
        """
        Get index object.

        Args:
            index: Index name (e.g., "XU100", "XU030", "XBANK")

        Returns:
            borsapy Index object
        """
        index = index.upper()
        if index not in self._index_cache:
            self._index_cache[index] = bp.Index(index)
        return self._index_cache[index]

    def get_index_components(self, index: str = "XU100") -> list[str]:
        """
        Get index constituent symbols.

        Args:
            index: Index name

        Returns:
            List of ticker symbols in the index
        """
        index = index.upper()
        if index not in self._index_components_cache:
            try:
                idx = self.get_index(index)
                self._index_components_cache[index] = list(idx.component_symbols or [])
            except Exception as e:
                print(f"  Warning: Failed to get components for {index}: {e}")
                self._index_components_cache[index] = []

        return self._index_components_cache[index]

    def get_all_indices(self) -> list[dict[str, Any]]:
        """Get all available BIST indices."""
        try:
            return bp.all_indices()
        except Exception:
            return []

    # -------------------------------------------------------------------------
    # Technical Indicators
    # -------------------------------------------------------------------------

    def get_history_with_indicators(
        self,
        symbol: str,
        indicators: list[str] = None,
        period: str = "2y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Get OHLCV history with built-in technical indicators.

        Args:
            symbol: Stock symbol
            indicators: List of indicators (e.g., ["rsi", "macd", "bb"])
                       If None, returns all available indicators
            period: Data period
            interval: Data interval

        Returns:
            DataFrame with OHLCV + indicator columns
        """
        ticker = self.get_ticker(symbol)
        try:
            if indicators:
                return ticker.history_with_indicators(
                    period=period, interval=interval, indicators=indicators
                )
            else:
                return ticker.history_with_indicators(period=period, interval=interval)
        except Exception as e:
            print(f"  Warning: Failed to get indicators for {symbol}: {e}")
            return self.get_history(symbol, period=period, interval=interval)

    def calculate_rsi(
        self,
        prices: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Calculate RSI indicator."""
        try:
            return bp.calculate_rsi(prices, period=period)
        except Exception:
            return pd.Series(dtype=float)

    def calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """Calculate MACD indicator."""
        try:
            return bp.calculate_macd(prices, fast=fast, slow=slow, signal=signal)
        except Exception:
            return pd.DataFrame()

    def calculate_supertrend(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 10,
        multiplier: float = 3.0,
    ) -> pd.DataFrame:
        """Calculate Supertrend indicator."""
        try:
            return bp.calculate_supertrend(high, low, close, period, multiplier)
        except Exception:
            return pd.DataFrame()

    # -------------------------------------------------------------------------
    # Stock Screening
    # -------------------------------------------------------------------------

    def screen_stocks(self, **filters) -> pd.DataFrame:
        """
        Run stock screener with fundamental/technical filters.

        Example filters:
            pe_max=10, pb_max=1.5, div_yield_min=3,
            roe_min=15, market_cap_min=1_000_000_000,
            index="XU100"

        Returns:
            DataFrame with matching stocks
        """
        try:
            return bp.screen_stocks(**filters)
        except Exception as e:
            print(f"  Warning: Screening failed: {e}")
            return pd.DataFrame()

    def technical_scan(
        self,
        condition: str,
        symbols: list[str] | str | None = None,
        timeframe: str = "1d",
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Technical condition scanner.

        Args:
            condition: Scan condition (e.g., "crosses_above", "crosses_below")
            symbols: Symbols or index universe. If None, scans XU100.
            timeframe: Timeframe for scan
            limit: Maximum result count

        Returns:
            DataFrame with scan results
        """
        if symbols is None:
            universe: str | list[str] = "XU100"
        elif isinstance(symbols, str):
            universe = symbols.upper()
        else:
            universe = [self._normalize_symbol(s) for s in symbols]

        try:
            return bp.scan(universe=universe, condition=condition, interval=timeframe, limit=limit)
        except Exception as e:
            print(f"  Warning: Technical scan failed: {e}")
            return pd.DataFrame()

    # -------------------------------------------------------------------------
    # Macro/Economic Data
    # -------------------------------------------------------------------------

    def get_inflation_data(self) -> pd.DataFrame:
        """Get TCMB inflation data."""
        try:
            return bp.Inflation().tufe()
        except Exception:
            return pd.DataFrame()

    def get_bond_yields(self) -> dict:
        """Get TCMB policy and corridor rates."""
        try:
            tcmb = bp.TCMB()
            rates = {
                "policy_rate": tcmb.policy_rate,
                "overnight": tcmb.overnight,
                "late_liquidity": tcmb.late_liquidity,
            }
            if isinstance(getattr(tcmb, "rates", None), pd.DataFrame):
                rates["rates_table"] = tcmb.rates
            return rates
        except Exception:
            return {}

    def get_economic_calendar(
        self,
        days_ahead: int = 7,
        countries: list[str] = None,
        importance: str | None = None,
    ) -> pd.DataFrame:
        """
        Get upcoming economic events.

        Args:
            days_ahead: Number of days to look ahead
            countries: Country codes (e.g., ["TR", "US"])
            importance: Optional importance filter (low/mid/high)

        Returns:
            DataFrame with upcoming events
        """
        if countries is None:
            countries = ["TR", "US"]

        try:
            return bp.economic_calendar(
                period=self._period_from_days(days_ahead),
                country=countries,
                importance=importance,
            )
        except Exception:
            return pd.DataFrame()

    # -------------------------------------------------------------------------
    # FX and Crypto
    # -------------------------------------------------------------------------

    def get_fx_rate(self, pair: str = "USD") -> dict:
        """Get current FX rate."""
        try:
            asset = pair.split("/")[0] if "/" in pair else pair
            return bp.FX(asset.upper()).info
        except Exception:
            return {}

    def get_crypto_price(self, symbol: str = "BTC") -> dict:
        """Get cryptocurrency price from BtcTurk."""
        try:
            return bp.Crypto(symbol).info
        except Exception:
            return {}

    # -------------------------------------------------------------------------
    # Cache Management
    # -------------------------------------------------------------------------

    def clear_cache(self):
        """Clear in-memory caches."""
        self._ticker_cache.clear()
        self._index_cache.clear()
        self._index_components_cache.clear()

    def save_to_cache(
        self,
        data: pd.DataFrame,
        filename: str,
        format: str = "parquet",
    ):
        """
        Save DataFrame to cache directory.

        Args:
            data: DataFrame to save
            filename: Output filename (without extension)
            format: Output format ("parquet" or "csv")
        """
        if format == "parquet":
            path = self.cache_dir / f"{filename}.parquet"
            data.to_parquet(path, compression="zstd")
        else:
            path = self.cache_dir / f"{filename}.csv"
            data.to_csv(path)

        print(f"  Saved to cache: {path}")

    def load_from_cache(
        self,
        filename: str,
        format: str = "parquet",
        max_age_hours: int = 24,
    ) -> Optional[pd.DataFrame]:
        """
        Load DataFrame from cache if fresh enough.

        Args:
            filename: Cache filename (without extension)
            format: File format
            max_age_hours: Maximum cache age in hours

        Returns:
            Cached DataFrame or None if stale/missing
        """
        ext = "parquet" if format == "parquet" else "csv"
        path = self.cache_dir / f"{filename}.{ext}"

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


# Convenience function for quick access
def get_client(cache_dir: Optional[Path] = None) -> BorsapyClient:
    """Get a BorsapyClient instance."""
    return BorsapyClient(cache_dir=cache_dir)
