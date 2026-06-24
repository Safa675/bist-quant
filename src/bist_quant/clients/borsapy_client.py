"""
Unified borsapy client for BIST data fetching.

Wraps borsapy with caching, error handling, and integration
with the existing data pipeline.

Resilience primitives (circuit breaker, retry) are in
:mod:`bist_quant.common.resilience`.
"""

import logging
import warnings
from datetime import datetime
from pathlib import Path
from ssl import SSLError
from typing import Any, Optional

try:
    from bist_quant.common.disk_cache import DiskCache
    from bist_quant.common.cache_config import CacheTTL
    _DISK_CACHE_AVAILABLE = True
except ImportError:
    _DISK_CACHE_AVAILABLE = False
    DiskCache = None  # type: ignore[assignment,misc]
    CacheTTL = None   # type: ignore[assignment,misc]

import pandas as pd

from bist_quant.common.resilience import (
    CircuitBreaker,
    CircuitBreakerError,
    configure_borsapy_logging,
    retry_with_backoff,
)
from bist_quant.settings import get_borsapy_config_path
from bist_quant.clients import borsapy_prices, borsapy_financials, borsapy_indices
from bist_quant.clients.utils import as_frame

logger = logging.getLogger(__name__)

try:
    import borsapy as bp
    BORSAPY_AVAILABLE = True
except ImportError:
    BORSAPY_AVAILABLE = False
    bp = None

warnings.filterwarnings("ignore")

DEFAULT_CONFIG_PATH = get_borsapy_config_path()


def _default_borsapy_cache_dir() -> Path:
    """Resolve the canonical borsapy cache directory (``<data_dir>/borsapy_cache``)."""
    from bist_quant.common.data_paths import get_data_paths

    return get_data_paths().borsapy_cache_dir





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

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        config_path: Path | str | None = None,
        disk_cache: "DiskCache | None" = None,
        **kwargs,
    ):
        """
        Initialize the borsapy client.

        Args:
            cache_dir: Directory for caching data. Defaults to data/borsapy_cache
            config_path: Optional path to borsapy YAML config.
            disk_cache: Optional pre-configured DiskCache instance.
                        If *None* and the disk_cache module is available a
                        default instance is created using ``cache_dir``.
        """
        if not BORSAPY_AVAILABLE:
            raise ImportError(
                "borsapy is not installed. Install with: pip install borsapy"
            )

        config = self._load_config(config_path)
        borsapy_config = config.get("borsapy", {}) if isinstance(config, dict) else {}

        # Persistent disk cache

        cb_config = borsapy_config.get("circuit_breaker", {}) if isinstance(borsapy_config, dict) else {}
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=int(cb_config.get("failure_threshold", 3)),
            recovery_timeout=int(cb_config.get("recovery_timeout", 30)),
        )

        retry_policy = borsapy_config.get("retry_policy", {}) if isinstance(borsapy_config, dict) else {}
        self._retry_policy: dict[str, Any] = {
            "max_retries": int(retry_policy.get("max_retries", 3)),
            "base_delay": float(retry_policy.get("base_delay", 1)),
            "max_delay": float(retry_policy.get("max_delay", 10)),
            "backoff_factor": float(retry_policy.get("backoff_factor", 2)),
            "jitter": bool(retry_policy.get("jitter", True)),
        }

        self.cache_dir = cache_dir or _default_borsapy_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Persistent disk cache
        if disk_cache is not None:
            self._disk_cache = disk_cache
        elif _DISK_CACHE_AVAILABLE and DiskCache is not None:
            self._disk_cache = DiskCache(
                cache_dir=self.cache_dir,
                ttl=CacheTTL.from_env() if CacheTTL is not None else None,
            )
        else:
            self._disk_cache = None

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

    @staticmethod
    def _load_config(config_path: Path | str | None) -> dict[str, Any]:
        path = Path(config_path) if config_path is not None else get_borsapy_config_path()
        if not path.exists():
            return {}

        try:
            import yaml
        except ImportError:
            logger.warning("PyYAML is unavailable; skipping borsapy config load.")
            return {}

        try:
            with open(path, "r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}
            return config if isinstance(config, dict) else {}
        except Exception as exc:
            logger.warning(f"Could not load borsapy config from {path}: {exc}")
            return {}

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
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch OHLCV history for a single ticker."""
        return borsapy_prices.get_history(
            self, symbol, period, interval, start, end, use_cache
        )

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
        """Batch download OHLCV data for multiple tickers."""
        return borsapy_prices.batch_download(
            self, symbols, period, interval, start, end, group_by, progress
        )

    def to_long_ohlcv(
        self,
        downloaded: pd.DataFrame,
        symbol_hint: str | None = None,
        add_is_suffix: bool = False,
    ) -> pd.DataFrame:
        """Convert borsapy download output to long format (Date, Ticker, OHLCV...)."""
        return borsapy_prices.to_long_ohlcv(
            self, downloaded, symbol_hint, add_is_suffix
        )

    def batch_download_to_long(
        self,
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
        return borsapy_prices.batch_download_to_long(
            self,
            symbols,
            period,
            interval,
            start,
            end,
            group_by,
            add_is_suffix,
            use_cache,
        )

    def get_fast_info(self, symbol: str) -> dict:
        """
        Get current quote/fast info for a ticker.

        Results are disk-cached with a short TTL (15 min default).

        Note: Data has ~15 minute delay unless using TradingView Pro.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with current price, volume, market cap, etc.
        """
        symbol = self._normalize_symbol(symbol)

        # Try disk cache
        if self._disk_cache is not None:
            cached = self._disk_cache.get_json("fast_info", symbol)
            if isinstance(cached, dict) and cached:
                return cached

        ticker = self.get_ticker(symbol)
        try:
            info = dict(ticker.fast_info) if ticker.fast_info else {}
            if self._disk_cache is not None and info:
                self._disk_cache.set_json("fast_info", symbol, info)
            return info
        except Exception as e:
            logger.info(f"  Warning: Failed to get fast_info for {symbol}: {e}")
            return {}

    # -------------------------------------------------------------------------
    # Fundamental Data
    # -------------------------------------------------------------------------

    def get_financials(self, symbol: str) -> dict[str, pd.DataFrame]:
        """Backward-compatible alias for financial statements."""
        return borsapy_financials.get_financials(self, symbol)

    def get_financial_statements(
        self, symbol: str, last_n: int = 20
    ) -> dict[str, pd.DataFrame]:
        """Get financial statements with disk caching."""
        return borsapy_financials.get_financial_statements(self, symbol, last_n)

    def get_financial_ratios(self, symbol: str) -> pd.DataFrame:
        """Get financial ratios."""
        return borsapy_financials.get_financial_ratios(self, symbol)

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
        """Get index object."""
        return borsapy_indices.get_index(self, index)

    def get_index_components(self, index: str = "XU100") -> list[str]:
        """Get index constituent symbols."""
        return borsapy_indices.get_index_components(self, index)

    def get_all_indices(self) -> list[dict[str, Any]]:
        """Get all available BIST indices."""
        return borsapy_indices.get_all_indices(self)

    def get_history_with_indicators(
        self,
        symbol: str,
        indicators: list[str] = None,
        period: str = "2y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Get OHLCV history with built-in technical indicators."""
        return borsapy_indices.get_history_with_indicators(
            self, symbol, indicators, period, interval
        )

    def calculate_rsi(
        self,
        prices: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Calculate RSI indicator."""
        return borsapy_indices.calculate_rsi(prices, period)

    def calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """Calculate MACD indicator."""
        return borsapy_indices.calculate_macd(prices, fast, slow, signal)

    def calculate_supertrend(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 10,
        multiplier: float = 3.0,
    ) -> pd.DataFrame:
        """Calculate Supertrend indicator."""
        return borsapy_indices.calculate_supertrend(
            high, low, close, period, multiplier
        )

    # -------------------------------------------------------------------------
    # Stock Screening
    # -------------------------------------------------------------------------

    def screen_stocks(
        self,
        template: str | None = None,
        filters: dict[str, Any] | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Run stock screener with fundamental/technical filters.

        Example filters:
            pe_max=10, pb_max=1.5, div_yield_min=3,
            roe_min=15, market_cap_min=1_000_000_000,
            index="XU100"

        Returns:
            DataFrame with matching stocks
        """
        merged_filters = dict(filters or {})
        merged_filters.update(kwargs)

        try:
            if template:
                result = bp.screen_stocks(template=template, **merged_filters)
            else:
                result = bp.screen_stocks(**merged_filters)
            result_df = as_frame(result)
            if result_df.empty:
                logger.warning("Empty screening result from borsapy.")
            return result_df
        except SSLError as exc:
            logger.warning(f"SSL error in borsapy.screen_stocks: {exc}")
            return pd.DataFrame()
        except Exception as exc:
            logger.error(f"Unexpected error in borsapy.screen_stocks: {exc}")
            return pd.DataFrame()

    def technical_scan(
        self,
        condition: str | None = None,
        symbols: list[str] | str | None = None,
        timeframe: str = "1d",
        limit: int = 100,
        *,
        template: str | None = None,
        conditions: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Technical condition scanner.

        Args:
            condition: Scan expression (e.g. ``"rsi < 30"``) or predefined template name.
            symbols: Symbols or index universe. If None, scans XU100.
            timeframe: Timeframe for scan.
            limit: Maximum result count (best-effort; depends on borsapy backend).
            template: Named predefined scan (see :data:`~bist_quant.clients.technical_scan.PREDEFINED_SCANS`).
            conditions: Multiple expressions combined with logical AND.

        Returns:
            DataFrame with scan results.
        """
        from bist_quant.clients.technical_scan import TechnicalScanner

        if symbols is None:
            universe: str | list[str] = "XU100"
        elif isinstance(symbols, str):
            universe = symbols.upper()
        else:
            universe = [self._normalize_symbol(s) for s in symbols]

        scanner = TechnicalScanner(borsapy_module=bp)
        try:
            if conditions:
                frame = scanner.scan_multi(
                    universe=universe,
                    conditions=conditions,
                    interval=timeframe,
                )
            else:
                frame = scanner.scan(
                    universe=universe,
                    condition=condition or "",
                    interval=timeframe,
                    template=template,
                )
        except ValueError:
            raise
        except Exception as exc:
            logger.info(f"  Warning: Technical scan failed: {exc}")
            return pd.DataFrame()

        if limit > 0 and not frame.empty and len(frame) > limit:
            return frame.head(limit).copy()
        return frame


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

    def close(self):
        """Close resources (no-op)."""
        pass

    def save_to_cache(
        self,
        data: pd.DataFrame,
        filename: str,
        format: str = "parquet",
    ) -> None:
        """Save DataFrame to cache directory."""
        return borsapy_prices.save_to_cache(self, data, filename, format)

    def load_from_cache(
        self,
        filename: str,
        format: str = "parquet",
        max_age_hours: int = 24,
    ) -> Optional[pd.DataFrame]:
        """Load DataFrame from cache if fresh enough."""
        return borsapy_prices.load_from_cache(self, filename, format, max_age_hours)


# Convenience function for quick access
def get_client(
    cache_dir: Optional[Path] = None,
) -> BorsapyClient:
    """Get a BorsapyClient instance."""
    return BorsapyClient(cache_dir=cache_dir)
