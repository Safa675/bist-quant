"""
Unified borsapy client for BIST data fetching.

Wraps borsapy with caching, error handling, and integration
with the existing data pipeline.
"""

import ast
import json
import logging
import random
import re
import time
import uuid
import warnings
from datetime import datetime
from enum import Enum
from functools import wraps
from pathlib import Path
from ssl import SSLError
from threading import Lock
from typing import Any, Callable, Optional

try:
    from bist_quant.common.disk_cache import DiskCache
    from bist_quant.common.cache_config import CacheTTL
    _DISK_CACHE_AVAILABLE = True
except ImportError:
    _DISK_CACHE_AVAILABLE = False
    DiskCache = None  # type: ignore[assignment,misc]
    CacheTTL = None   # type: ignore[assignment,misc]

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import borsapy as bp
    BORSAPY_AVAILABLE = True
except ImportError:
    BORSAPY_AVAILABLE = False
    bp = None

warnings.filterwarnings("ignore")

DEFAULT_MCP_ENDPOINT = "https://borsamcp.fastmcp.app/mcp"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "borsapy_config.yaml"

# Bank and financial institution tickers that require UFRS accounting group
# on the İş Yatırım MaliTablo endpoint (instead of default XI_29).
UFRS_TICKERS = {
    # Banks
    "AKBNK", "ALBRK", "DENIZ", "GARAN", "HALKB", "ICBCT",
    "ISCTR", "KLNMA", "QNBFB", "QNBFK", "QNBTR", "SKBNK",
    "TSKB", "VAKBN", "YKBNK",
    # Financial services / insurance
    "AGESA", "AKGRT", "ANHYT", "ANSGR", "AVHOL", "AVIVA",
    "GUSGR", "HDFGS", "ISFIN", "ISGSY", "ISYAT", "RAYSG",
    "SEKFK", "TURSG", "VAKFN", "VKFYO",
}


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerError(RuntimeError):
    """Raised when the circuit breaker blocks a call."""


class CircuitBreaker:
    """Simple thread-safe circuit breaker."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = int(failure_threshold)
        self.recovery_timeout = int(recovery_timeout)

        self.failure_count = 0
        self.last_failure_time: float | None = None
        self.state = CircuitState.CLOSED
        self.lock = Lock()

    def call(self, func: Callable[..., Any], *args, **kwargs):
        with self.lock:
            if self.state == CircuitState.OPEN:
                elapsed = time.time() - (self.last_failure_time or 0)
                if elapsed > self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                else:
                    raise CircuitBreakerError("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except Exception:
            self.on_failure()
            raise

    def on_success(self):
        with self.lock:
            self.failure_count = 0
            self.state = CircuitState.CLOSED
            self.last_failure_time = None

    def on_failure(self):
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
):
    """Retry decorator with exponential backoff and optional jitter."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = int(max_retries)
            delay = float(base_delay)
            cap = float(max_delay)
            factor = float(backoff_factor)
            use_jitter = bool(jitter)

            if args:
                policy = getattr(args[0], "_retry_policy", None)
                if isinstance(policy, dict):
                    retries = int(policy.get("max_retries", retries))
                    delay = float(policy.get("base_delay", delay))
                    cap = float(policy.get("max_delay", cap))
                    factor = float(policy.get("backoff_factor", factor))
                    use_jitter = bool(policy.get("jitter", use_jitter))

            current_delay = max(0.0, delay)
            last_exception = None

            for attempt in range(retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exception = exc
                    if attempt == retries:
                        break

                    sleep_time = current_delay
                    if use_jitter and current_delay > 0:
                        sleep_time = random.uniform(0, current_delay)

                    logger.warning(
                        f"Attempt {attempt + 1} failed: {exc}. Retrying in {sleep_time:.2f}s..."
                    )
                    time.sleep(max(0.0, sleep_time))
                    current_delay = min(max(0.0, current_delay * factor), cap)

            raise last_exception

        return wrapper

    return decorator


def configure_borsapy_logging(
    log_file: str | Path = "borsapy_integration.log",
    level: int = logging.INFO,
) -> None:
    """
    Configure default logging for borsapy integrations.

    This only configures the root logger if no handlers are present.
    """
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )


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
        use_mcp_fallback: bool = True,
        mcp_endpoint: str | None = None,
        timeout: float | None = None,
        config_path: Path | str | None = None,
        disk_cache: "DiskCache | None" = None,
    ):
        """
        Initialize the borsapy client.

        Args:
            cache_dir: Directory for caching data. Defaults to data/borsapy_cache
            use_mcp_fallback: Enable Borsa-MCP fallback for SSL/data issues.
            mcp_endpoint: Optional MCP endpoint override.
            timeout: Optional MCP HTTP timeout override.
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

        self.use_mcp_fallback = bool(
            borsapy_config.get("use_mcp_fallback", use_mcp_fallback)
        )
        self._mcp_endpoint = str(borsapy_config.get("mcp_endpoint", mcp_endpoint or DEFAULT_MCP_ENDPOINT))
        timeout_seconds = float(borsapy_config.get("timeout", timeout or 15.0))
        self._session = httpx.Client(
            timeout=timeout_seconds,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
        )

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

        self.cache_dir = cache_dir or Path(__file__).parent.parent / "borsapy_cache"
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
        path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
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
        """
        Fetch OHLCV history for a single ticker.

        Args:
            symbol: Stock symbol
            period: Data period (e.g., "1y", "5y", "max")
            interval: Data interval (e.g., "1d", "1h", "1m")
            start: Optional start date
            end: Optional end date
            use_cache: Check / populate disk cache (default True)

        Returns:
            DataFrame with Date index and OHLCV columns
        """
        symbol = self._normalize_symbol(symbol)
        cache_key = f"{symbol}_{period}_{interval}"

        # Try disk cache first
        if use_cache and self._disk_cache is not None:
            cached = self._disk_cache.get_dataframe("prices", cache_key)
            if cached is not None and not cached.empty:
                logger.debug("  Cache hit for %s history", symbol)
                return cached

        ticker = self.get_ticker(symbol)
        try:
            df = ticker.history(period=period, interval=interval, start=start, end=end)
            if df is not None and not df.empty:
                df["Ticker"] = symbol
                # Persist to disk cache
                if use_cache and self._disk_cache is not None:
                    self._disk_cache.set_dataframe("prices", cache_key, df)
            return df
        except Exception as e:
            logger.info(f"  Warning: Failed to fetch history for {symbol}: {e}")
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
            logger.info(f"  Warning: Batch download failed: {e}")
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
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Batch download and return data in long OHLCV format.

        When ``use_cache`` is *True* (default), per-symbol disk cache is
        checked first.  Only symbols whose cache has expired or is missing
        are fetched from the borsapy API.
        """
        normalized = [self._normalize_symbol(s) for s in symbols]

        # --- Disk-cache partition -----------------------------------------
        cached_frames: list[pd.DataFrame] = []
        fetch_symbols = list(normalized)

        if use_cache and self._disk_cache is not None:
            for sym in normalized:
                cache_key = f"{sym}_{period}_{interval}"
                frame = self._disk_cache.get_dataframe("prices", cache_key)
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
            downloaded = self.batch_download(
                fetch_symbols,
                period=period,
                interval=interval,
                start=start,
                end=end,
                group_by=group_by,
            )
            result = self.to_long_ohlcv(downloaded, add_is_suffix=add_is_suffix)
        else:
            result = pd.DataFrame()

        # Fallback fetch for symbols omitted by a partial batch response.
        if result.empty:
            present: set[str] = set()
        else:
            present = {
                self._normalize_symbol(t)
                for t in result["Ticker"].dropna().astype(str)
            }

        missing = [s for s in fetch_symbols if s not in present]
        for symbol in missing:
            single = self.get_history(
                symbol, period=period, interval=interval,
                start=start, end=end, use_cache=False,
            )
            if single is None or single.empty:
                continue
            long_single = self.to_long_ohlcv(
                single.drop(columns=["Ticker"], errors="ignore"),
                symbol_hint=symbol,
                add_is_suffix=add_is_suffix,
            )
            if not long_single.empty:
                result = pd.concat([result, long_single], ignore_index=True)

        # --- Persist newly fetched data to disk cache ---------------------
        if use_cache and self._disk_cache is not None and not result.empty:
            for ticker_val in result["Ticker"].dropna().unique():
                ticker_str = self._normalize_symbol(str(ticker_val))
                cache_key = f"{ticker_str}_{period}_{interval}"
                ticker_slice = result[result["Ticker"] == ticker_val]
                self._disk_cache.set_dataframe("prices", cache_key, ticker_slice)

        # --- Combine cached + freshly fetched data ------------------------
        if cached_frames:
            # FIX: Reset index on cached frames before concatenation (Date is index in cache)
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
        """
        Backward-compatible alias for financial statements.

        Returns:
            Dict with balance_sheet, income_stmt, cashflow and cash_flow DataFrames.
        """
        statements = self.get_financial_statements(symbol)
        cash_flow = statements.get("cash_flow", pd.DataFrame())
        return {
            "balance_sheet": statements.get("balance_sheet", pd.DataFrame()),
            "income_stmt": statements.get("income_stmt", pd.DataFrame()),
            "cashflow": cash_flow,
            "cash_flow": cash_flow,
        }
    def _format_quarterly_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Format borsapy quarterly column names (e.g. 2025Q3, 2024Q4) to YYYY/MM.
        TradingView/borsapy tends to shift the fiscal year forward by 1.
        e.g., 2025Q3 -> 2024/09, 2024Q4 -> 2023/12.
        """
        if df is None or df.empty:
            return df
            
        new_cols = []
        for col in df.columns:
            col_str = str(col).strip()
            if len(col_str) == 6 and col_str[4] == 'Q':
                try:
                    year = int(col_str[:4])
                    q = int(col_str[5])
                    real_year = year - 1
                    month = q * 3
                    new_cols.append(f"{real_year}/{month:02d}")
                    continue
                except ValueError:
                    pass
            new_cols.append(col)
            
        df.columns = new_cols
        return df

    @staticmethod
    def _detect_financial_group(symbol: str) -> str | None:
        """Return UFRS for bank/financial tickers, None (=XI_29 default) otherwise."""
        return "UFRS" if symbol.upper() in UFRS_TICKERS else None

    def get_financial_statements(self, symbol: str) -> dict[str, pd.DataFrame]:
        """
        Get financial statements with disk caching and MCP fallback.

        Automatically detects bank/financial tickers that require the UFRS
        accounting group on the İş Yatırım API.

        Args:
            symbol: Stock symbol (e.g., THYAO, GARAN)

        Returns:
            Dictionary containing balance_sheet, income_stmt, cash_flow.
        """
        symbol = self._normalize_symbol(symbol)

        # Try disk cache first
        if self._disk_cache is not None:
            cached_bs = self._disk_cache.get_dataframe("financials", f"{symbol}/balance_sheet")
            cached_is = self._disk_cache.get_dataframe("financials", f"{symbol}/income_stmt")
            cached_cf = self._disk_cache.get_dataframe("financials", f"{symbol}/cash_flow")
            if cached_bs is not None or cached_is is not None or cached_cf is not None:
                logger.debug("  Cache hit for %s financials", symbol)
                return {
                    "balance_sheet": cached_bs if cached_bs is not None else pd.DataFrame(),
                    "income_stmt": cached_is if cached_is is not None else pd.DataFrame(),
                    "cash_flow": cached_cf if cached_cf is not None else pd.DataFrame(),
                }

        try:
            ticker = self.get_ticker(symbol)
            fg = self._detect_financial_group(symbol)

            # Use explicit method calls so we can pass financial_group for
            # bank/financial tickers that need UFRS instead of the default XI_29.
            statements: dict[str, pd.DataFrame] = {}
            for sheet_name, method_name in [
                ("balance_sheet", "get_balance_sheet"),
                ("income_stmt", "get_income_stmt"),
                ("cash_flow", "get_cashflow"),
            ]:
                try:
                    raw = getattr(ticker, method_name)(
                        quarterly=True, financial_group=fg
                    )
                    statements[sheet_name] = self._format_quarterly_columns(
                        self._coerce_dataframe(raw)
                    )
                except Exception as inner_exc:
                    logger.debug(
                        "  %s %s fetch failed: %s", symbol, sheet_name, inner_exc
                    )
                    statements[sheet_name] = pd.DataFrame()

            if all(frame.empty for frame in statements.values()):
                logger.warning(
                    f"Empty financial statements from borsapy for {symbol}, trying MCP fallback."
                )
                if self.use_mcp_fallback:
                    statements = self._get_financials_via_mcp(symbol)
            # Cache non-empty results
            if self._disk_cache is not None:
                for sheet_name, df in statements.items():
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        self._disk_cache.set_dataframe(
                            "financials", f"{symbol}/{sheet_name}", df,
                        )
            return statements
        except SSLError as exc:
            logger.warning(f"SSL error in borsapy financial statements for {symbol}: {exc}")
            if self.use_mcp_fallback:
                return self._get_financials_via_mcp(symbol)
            raise
        except Exception as exc:
            logger.error(f"Unexpected error in borsapy financial statements for {symbol}: {exc}")
            if self.use_mcp_fallback:
                return self._get_financials_via_mcp(symbol)
            raise

    def get_financial_ratios(self, symbol: str) -> pd.DataFrame:
        """
        Get financial ratios with MCP fallback.

        Args:
            symbol: Stock symbol (e.g., THYAO)

        Returns:
            DataFrame with financial ratios.
        """
        symbol = self._normalize_symbol(symbol)
        try:
            ticker = self.get_ticker(symbol)
            ratios = self._coerce_dataframe(getattr(ticker, "financial_ratios", None))
            if ratios.empty:
                logger.warning(f"Empty financial ratios from borsapy for {symbol}, trying MCP fallback.")
                if self.use_mcp_fallback:
                    return self._get_ratios_via_mcp(symbol)
            return ratios
        except SSLError as exc:
            logger.warning(f"SSL error in borsapy financial ratios for {symbol}: {exc}")
            if self.use_mcp_fallback:
                return self._get_ratios_via_mcp(symbol)
            raise
        except Exception as exc:
            logger.error(f"Unexpected error in borsapy financial ratios for {symbol}: {exc}")
            if self.use_mcp_fallback:
                return self._get_ratios_via_mcp(symbol)
            raise

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

        Results are cached to disk (24h default TTL) and in-memory.

        Args:
            index: Index name

        Returns:
            List of ticker symbols in the index
        """
        index = index.upper()
        if index not in self._index_components_cache:
            # Try disk cache
            if self._disk_cache is not None:
                cached = self._disk_cache.get_json("index_components", index)
                if isinstance(cached, list) and cached:
                    self._index_components_cache[index] = cached
                    return cached

            try:
                idx = self.get_index(index)
                components = list(idx.component_symbols or [])
                self._index_components_cache[index] = components
                # Persist to disk cache
                if self._disk_cache is not None and components:
                    self._disk_cache.set_json("index_components", index, components)
            except Exception as e:
                logger.info(f"  Warning: Failed to get components for {index}: {e}")
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
            logger.info(f"  Warning: Failed to get indicators for {symbol}: {e}")
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
            result_df = self._coerce_dataframe(result)
            if not result_df.empty or not self.use_mcp_fallback:
                return result_df

            logger.warning("Empty screening result from borsapy; trying MCP fallback.")
            return self._screen_via_mcp(template=template, filters=merged_filters)
        except SSLError as exc:
            logger.warning(f"SSL error in borsapy.screen_stocks: {exc}")
            if self.use_mcp_fallback:
                return self._screen_via_mcp(template=template, filters=merged_filters)
            raise
        except Exception as exc:
            logger.error(f"Unexpected error in borsapy.screen_stocks: {exc}")
            if self.use_mcp_fallback:
                return self._screen_via_mcp(template=template, filters=merged_filters)
            raise

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
            logger.info(f"  Warning: Technical scan failed: {e}")
            return pd.DataFrame()

    # -------------------------------------------------------------------------
    # MCP Fallback Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _coerce_dataframe(value: Any) -> pd.DataFrame:
        """Normalize arbitrary payloads into a DataFrame."""
        if isinstance(value, pd.DataFrame):
            return value.copy()
        if isinstance(value, pd.Series):
            return value.to_frame().T
        if isinstance(value, list):
            if not value:
                return pd.DataFrame()
            if all(isinstance(item, dict) for item in value):
                return pd.DataFrame(value)
            return pd.DataFrame({"value": value})
        if isinstance(value, dict):
            if not value:
                return pd.DataFrame()
            try:
                if all(isinstance(v, (list, tuple, pd.Series)) for v in value.values()):
                    return pd.DataFrame(value)
            except Exception:
                pass
            return pd.DataFrame([value])
        return pd.DataFrame()

    @staticmethod
    def _first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
        for key in keys:
            if key in mapping:
                return mapping[key]
        return None

    @staticmethod
    def _extract_mcp_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(part for part in parts if part).strip()
        if isinstance(content, dict):
            text = content.get("text")
            if isinstance(text, str):
                return text.strip()
        return ""

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            return fenced.group(1).strip()
        return text.strip()

    @classmethod
    def _parse_content_text(cls, text: str) -> Any:
        cleaned = cls._strip_code_fence(text)
        if not cleaned:
            return None

        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(cleaned)
            except Exception:
                continue
        return None

    @classmethod
    def _extract_payload_from_result(cls, result: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(result, dict):
            return {}

        payload: dict[str, Any] = {}
        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            payload.update(structured)

        for key, value in result.items():
            if key == "structuredContent":
                continue
            payload.setdefault(key, value)

        text = cls._extract_mcp_text(result.get("content"))
        parsed = cls._parse_content_text(text) if text else None
        if isinstance(parsed, dict):
            for key, value in parsed.items():
                payload.setdefault(key, value)
        elif isinstance(parsed, list):
            payload.setdefault("data", parsed)

        return payload

    def _screen_via_mcp(
        self,
        template: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Use Borsa-MCP for screening when borsapy fails."""
        params: dict[str, Any] = {}
        if template:
            # Borsa-MCP uses "preset", but we keep "template" compatibility.
            params["preset"] = template
            params["template"] = template
        if filters:
            params.update(filters)

        try:
            response = self._call_mcp_tool_with_circuit_breaker("screen_securities", params)
            payload = self._extract_payload_from_result(response)
            rows = self._first_present(
                payload,
                ("data", "results", "securities", "stocks", "items", "matches"),
            )
            return self._coerce_dataframe(rows)
        except Exception as exc:
            logger.error(f"MCP screening fallback failed: {exc}")
            return pd.DataFrame()

    def _get_financials_via_mcp(self, symbol: str) -> dict[str, pd.DataFrame]:
        """Use Borsa-MCP for financial statements when borsapy fails."""
        empty = {
            "balance_sheet": pd.DataFrame(),
            "income_stmt": pd.DataFrame(),
            "cash_flow": pd.DataFrame(),
        }

        try:
            response = self._call_mcp_tool_with_circuit_breaker(
                "get_financial_statements",
                {
                    "symbols": [self._normalize_symbol(symbol)],
                    "market": "bist"
                },
            )
            payload = self._extract_payload_from_result(response)

            data_payload = payload.get("data")
            if isinstance(data_payload, dict):
                for key, value in data_payload.items():
                    payload.setdefault(key, value)

            balance_sheet = self._coerce_dataframe(
                self._first_present(payload, ("balance_sheet", "balanceSheet", "balance"))
            )
            income_stmt = self._coerce_dataframe(
                self._first_present(payload, ("income_stmt", "income_statement", "incomeStatement"))
            )
            cash_flow = self._coerce_dataframe(
                self._first_present(payload, ("cash_flow", "cashflow", "cashFlow"))
            )

            return {
                "balance_sheet": balance_sheet,
                "income_stmt": income_stmt,
                "cash_flow": cash_flow,
            }
        except Exception as exc:
            logger.error(f"MCP financial statements fallback failed for {symbol}: {exc}")
            return empty

    def _get_ratios_via_mcp(self, symbol: str) -> pd.DataFrame:
        """Use Borsa-MCP for financial ratios when borsapy fails."""
        try:
            response = self._call_mcp_tool_with_circuit_breaker(
                "get_financial_ratios",
                {
                    "symbols": [self._normalize_symbol(symbol)],
                    "market": "bist"
                },
            )
            payload = self._extract_payload_from_result(response)

            ratios = self._first_present(payload, ("ratios", "financial_ratios", "data"))
            if isinstance(ratios, dict) and "ratios" in ratios:
                ratios = ratios.get("ratios")
            return self._coerce_dataframe(ratios)
        except Exception as exc:
            logger.error(f"MCP financial ratios fallback failed for {symbol}: {exc}")
            return pd.DataFrame()

    def _call_mcp_tool_with_circuit_breaker(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Call MCP tool with circuit breaker protection."""
        return self._circuit_breaker.call(self._call_mcp_tool, tool_name, params)

    @staticmethod
    def _parse_mcp_jsonrpc_response(response: httpx.Response) -> dict[str, Any]:
        content_type = response.headers.get("content-type", "").lower()
        if "text/event-stream" in content_type:
            payloads: list[dict[str, Any]] = []
            for line in response.text.splitlines():
                if not line.startswith("data:"):
                    continue
                content = line[5:].strip()
                if not content or content == "[DONE]":
                    continue
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict):
                    payloads.append(data)

            if not payloads:
                raise ValueError("MCP SSE response contained no JSON payloads.")

            for payload in reversed(payloads):
                if "result" in payload or "error" in payload:
                    return payload
            return payloads[-1]

        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("MCP response payload is not a JSON object.")
        return payload

    @retry_with_backoff(max_retries=3, base_delay=1, max_delay=10, backoff_factor=2, jitter=True)
    def _call_mcp_tool(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Make a call to the Borsa-MCP endpoint with retry logic."""
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": params,
            },
        }

        try:
            response = self._session.post(self._mcp_endpoint, json=payload)
            response.raise_for_status()

            rpc_payload = self._parse_mcp_jsonrpc_response(response)
            if "error" in rpc_payload:
                error = rpc_payload.get("error")
                if isinstance(error, dict):
                    message = str(error.get("message", error))
                else:
                    message = str(error)
                raise RuntimeError(message)

            result = rpc_payload.get("result", {})
            if isinstance(result, dict) and result.get("isError") is True:
                message = self._extract_mcp_text(result.get("content")) or "MCP tool returned isError=true."
                raise RuntimeError(message)

            if isinstance(result, dict):
                return result
            if result is None:
                return {}
            return {"data": result}
        except httpx.RequestError as exc:
            logger.error(f"Request error calling MCP tool {tool_name}: {exc}")
            raise
        except Exception as exc:
            logger.error(f"Error calling MCP tool {tool_name}: {exc}")
            raise

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
        """Close HTTP resources used by MCP fallback."""
        try:
            self._session.close()
        except Exception:
            pass

    def __del__(self):
        self.close()

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

        logger.info(f"  Saved to cache: {path}")

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
def get_client(
    cache_dir: Optional[Path] = None,
    use_mcp_fallback: bool = True,
) -> BorsapyClient:
    """Get a BorsapyClient instance."""
    return BorsapyClient(cache_dir=cache_dir, use_mcp_fallback=use_mcp_fallback)
