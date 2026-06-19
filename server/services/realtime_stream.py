"""
Real-Time Data Streaming Service via Borsapy

Provides real-time (15-min delayed by default) quote access for:
- Current prices and quotes
- Batch quote fetching for multiple tickers
- Quote caching with TTL
- Portfolio position updates

Note: Real-time data requires TradingView Pro + BIST package for
true real-time access. Default is ~15 minute delay.

Usage:
    from realtime_stream import RealtimeQuoteService

    service = RealtimeQuoteService()

    # Get single quote
    quote = service.get_quote("THYAO")

    # Get batch quotes
    quotes = service.get_quotes_batch(["THYAO", "AKBNK", "GARAN"])

    # Get portfolio snapshot
    snapshot = service.get_portfolio_snapshot(holdings)
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Callable
import json
import threading
import time
logger = logging.getLogger(__name__)

try:
    import borsapy as bp
    BORSAPY_AVAILABLE = True
except ImportError:
    BORSAPY_AVAILABLE = False
    bp = None


class QuoteCache:
    """Thread-safe quote cache with TTL."""

    def __init__(self, ttl_seconds: int = 60):
        """
        Initialize cache.

        Args:
            ttl_seconds: Time-to-live for cached quotes (default 60s)
        """
        self.ttl = timedelta(seconds=ttl_seconds)
        self._cache: dict[str, tuple[dict, datetime]] = {}
        self._lock = threading.Lock()

    def get(self, symbol: str) -> Optional[dict]:
        """Get cached quote if fresh."""
        with self._lock:
            if symbol in self._cache:
                quote, timestamp = self._cache[symbol]
                if datetime.now() - timestamp < self.ttl:
                    return quote
                # Expired, remove from cache
                del self._cache[symbol]
        return None

    def set(self, symbol: str, quote: dict):
        """Cache a quote."""
        with self._lock:
            self._cache[symbol] = (quote, datetime.now())

    def clear(self):
        """Clear all cached quotes."""
        with self._lock:
            self._cache.clear()

    def get_all_fresh(self) -> dict[str, dict]:
        """Get all fresh quotes from cache."""
        with self._lock:
            now = datetime.now()
            fresh = {}
            expired = []
            for symbol, (quote, timestamp) in self._cache.items():
                if now - timestamp < self.ttl:
                    fresh[symbol] = quote
                else:
                    expired.append(symbol)
            # Clean up expired
            for symbol in expired:
                del self._cache[symbol]
            return fresh


class RealtimeQuoteService:
    """
    Real-time quote service for dashboard integration.

    Provides:
    - Single and batch quote fetching
    - Quote caching to reduce API calls
    - Portfolio value calculations
    - Price change tracking
    """

    def __init__(self, cache_ttl: int = 60):
        """
        Initialize the service.

        Args:
            cache_ttl: Cache TTL in seconds (default 60)
        """
        if not BORSAPY_AVAILABLE:
            raise ImportError(
                "borsapy is not installed. Install with: pip install borsapy"
            )

        self._cache = QuoteCache(ttl_seconds=cache_ttl)
        self._ticker_cache: dict[str, bp.Ticker] = {}

    def _get_ticker(self, symbol: str) -> "bp.Ticker":
        """Get or create ticker object."""
        symbol = symbol.upper().split(".")[0]
        if symbol not in self._ticker_cache:
            self._ticker_cache[symbol] = bp.Ticker(symbol)
        return self._ticker_cache[symbol]

    @staticmethod
    def _index_snapshot(info: dict) -> dict:
        """Normalize borsapy index info payload."""
        if not isinstance(info, dict) or not info:
            return {"error": "No data"}

        value = info.get("last")
        prev = info.get("close")
        change_pct = info.get("change_percent")
        if change_pct is None and value is not None and prev not in (None, 0):
            change_pct = ((value - prev) / prev) * 100

        return {"value": value, "change_pct": change_pct}

    def get_quote(self, symbol: str, use_cache: bool = True) -> dict:
        """
        Get current quote for a ticker.

        Args:
            symbol: Stock symbol (e.g., "THYAO")
            use_cache: Whether to use cached data if available

        Returns:
            Dict with quote data:
            - symbol: Ticker symbol
            - last_price: Current price
            - change: Price change from previous close
            - change_pct: Percent change
            - volume: Trading volume
            - high: Day high
            - low: Day low
            - open: Opening price
            - prev_close: Previous close
            - market_cap: Market capitalization
            - pe_ratio: P/E ratio
            - pb_ratio: P/B ratio
            - timestamp: Quote timestamp
        """
        symbol = symbol.upper().split(".")[0]

        # Check cache first
        if use_cache:
            cached = self._cache.get(symbol)
            if cached:
                return cached

        # Fetch fresh quote
        try:
            ticker = self._get_ticker(symbol)
            info = ticker.fast_info

            if not info:
                return {"symbol": symbol, "error": "No data available"}

            # FastInfo is an object with attributes, not a dict
            last_price = getattr(info, "last_price", None)
            prev_close = getattr(info, "previous_close", None)

            # Calculate change and change_pct
            change = None
            change_pct = None
            if last_price is not None and prev_close is not None and prev_close != 0:
                change = last_price - prev_close
                change_pct = (change / prev_close) * 100

            quote = {
                "symbol": symbol,
                "last_price": last_price,
                "change": change,
                "change_pct": change_pct,
                "volume": getattr(info, "volume", None),
                "amount": getattr(info, "amount", None),
                "high": getattr(info, "day_high", None),
                "low": getattr(info, "day_low", None),
                "open": getattr(info, "open", None),
                "prev_close": prev_close,
                "market_cap": getattr(info, "market_cap", None),
                "pe_ratio": getattr(info, "pe_ratio", None),
                "pb_ratio": getattr(info, "pb_ratio", None),
                "year_high": getattr(info, "year_high", None),
                "year_low": getattr(info, "year_low", None),
                "fifty_day_avg": getattr(info, "fifty_day_average", None),
                "two_hundred_day_avg": getattr(info, "two_hundred_day_average", None),
                "free_float": getattr(info, "free_float", None),
                "foreign_ratio": getattr(info, "foreign_ratio", None),
                "timestamp": datetime.now().isoformat(),
            }

            # Cache the quote
            self._cache.set(symbol, quote)
            return quote

        except Exception as e:
            return {"symbol": symbol, "error": str(e)}

    def get_quotes_batch(
        self,
        symbols: list[str],
        use_cache: bool = True,
    ) -> dict[str, dict]:
        """
        Get quotes for multiple tickers.

        Args:
            symbols: List of stock symbols
            use_cache: Whether to use cached data

        Returns:
            Dict mapping symbol -> quote dict
        """
        results = {}
        symbols_to_fetch = []

        # Check cache first
        if use_cache:
            for symbol in symbols:
                symbol = symbol.upper().split(".")[0]
                cached = self._cache.get(symbol)
                if cached:
                    results[symbol] = cached
                else:
                    symbols_to_fetch.append(symbol)
        else:
            symbols_to_fetch = [s.upper().split(".")[0] for s in symbols]

        # Fetch remaining symbols
        for symbol in symbols_to_fetch:
            quote = self.get_quote(symbol, use_cache=False)
            results[symbol] = quote

        return results

    def get_index_quotes(
        self,
        index: str = "XU100",
        max_symbols: int | None = None,
    ) -> dict[str, dict]:
        """
        Get quotes for all stocks in an index.

        Args:
            index: Index name (e.g., "XU100", "XU030")
            max_symbols: Optional cap on number of index components

        Returns:
            Dict mapping symbol -> quote dict
        """
        try:
            idx = bp.Index(index)
            symbols = list(idx.component_symbols or [])
            if max_symbols is not None:
                symbols = symbols[:max_symbols]
            return self.get_quotes_batch(symbols)
        except Exception as e:
            return {"error": str(e)}

    def get_portfolio_snapshot(
        self,
        holdings: dict[str, float],
        cost_basis: dict[str, float] = None,
    ) -> dict:
        """
        Get real-time portfolio snapshot.

        Args:
            holdings: Dict mapping symbol -> quantity
            cost_basis: Optional dict mapping symbol -> average cost per share

        Returns:
            Dict with portfolio summary:
            - total_value: Current portfolio value
            - total_cost: Total cost basis (if provided)
            - total_pnl: Unrealized P&L
            - total_pnl_pct: P&L percentage
            - positions: List of position details
            - timestamp: Snapshot timestamp
        """
        symbols = list(holdings.keys())
        quotes = self.get_quotes_batch(symbols)

        positions = []
        total_value = 0.0
        total_cost = 0.0

        for symbol, quantity in holdings.items():
            quote = quotes.get(symbol.upper(), {})
            price = quote.get("last_price")

            if price is None:
                positions.append({
                    "symbol": symbol,
                    "quantity": quantity,
                    "error": quote.get("error", "No price data"),
                })
                continue

            market_value = price * quantity
            total_value += market_value

            position = {
                "symbol": symbol,
                "quantity": quantity,
                "price": price,
                "market_value": market_value,
                "change_pct": quote.get("change_pct"),
            }

            # Add P&L if cost basis provided
            if cost_basis and symbol in cost_basis:
                cost = cost_basis[symbol] * quantity
                total_cost += cost
                pnl = market_value - cost
                pnl_pct = (pnl / cost * 100) if cost > 0 else 0

                position["cost_basis"] = cost_basis[symbol]
                position["cost"] = cost
                position["pnl"] = pnl
                position["pnl_pct"] = pnl_pct

            positions.append(position)

        # Sort by market value descending
        positions.sort(key=lambda p: p.get("market_value", 0), reverse=True)

        result = {
            "total_value": total_value,
            "positions": positions,
            "timestamp": datetime.now().isoformat(),
        }

        if cost_basis:
            result["total_cost"] = total_cost
            result["total_pnl"] = total_value - total_cost
            result["total_pnl_pct"] = (
                (total_value - total_cost) / total_cost * 100
                if total_cost > 0 else 0
            )

        return result

    def get_market_summary(self) -> dict:
        """
        Get market summary with key indices and stats.

        Returns:
            Dict with market overview:
            - xu100: XU100 index quote
            - xu030: XU030 index quote
            - usdtry: USD/TRY rate
            - timestamp: Summary timestamp
        """
        summary = {"timestamp": datetime.now().isoformat()}

        # Get index data
        try:
            summary["xu100"] = self._index_snapshot(bp.Index("XU100").info)
        except Exception as e:
            summary["xu100"] = {"error": str(e)}

        try:
            summary["xu030"] = self._index_snapshot(bp.Index("XU030").info)
        except Exception as e:
            summary["xu030"] = {"error": str(e)}

        # Get FX rates
        try:
            usdtry = bp.FX("USD")
            info = usdtry.info
            if info:
                rate = info.get("last") if isinstance(info, dict) else None
                open_ = info.get("open") if isinstance(info, dict) else None
                change_pct = None
                if rate is not None and open_ not in (None, 0):
                    change_pct = ((rate - open_) / open_) * 100
                summary["usdtry"] = {"rate": rate, "change_pct": change_pct}
            else:
                summary["usdtry"] = {"error": "No data"}
        except Exception as e:
            summary["usdtry"] = {"error": str(e)}

        return summary

    def clear_cache(self):
        """Clear the quote cache."""
        self._cache.clear()
        self._ticker_cache.clear()

    def to_json(self, data: dict) -> str:
        """Convert data to JSON string."""
        return json.dumps(data, ensure_ascii=False, indent=2, default=str)


class RealtimeWatcher:
    """
    Background watcher for real-time price updates.

    Periodically fetches quotes and triggers callbacks on changes.
    """

    def __init__(
        self,
        symbols: list[str],
        interval_seconds: int = 60,
        on_update: Callable[[dict[str, dict]], None] = None,
    ):
        """
        Initialize the watcher.

        Args:
            symbols: Symbols to watch
            interval_seconds: Update interval
            on_update: Callback function for updates
        """
        self.symbols = [s.upper() for s in symbols]
        self.interval = interval_seconds
        self.on_update = on_update

        self._service = RealtimeQuoteService(cache_ttl=interval_seconds // 2)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_quotes: dict[str, dict] = {}

    def start(self):
        """Start the background watcher."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        logger.info(f"RealtimeWatcher started: {len(self.symbols)} symbols, {self.interval}s interval")

    def stop(self):
        """Stop the background watcher."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("RealtimeWatcher stopped")

    def _watch_loop(self):
        """Background watch loop."""
        while self._running:
            try:
                quotes = self._service.get_quotes_batch(self.symbols, use_cache=False)

                # Check for changes
                changed = {}
                for symbol, quote in quotes.items():
                    if "error" in quote:
                        continue

                    last = self._last_quotes.get(symbol, {})
                    if quote.get("last_price") != last.get("last_price"):
                        changed[symbol] = quote

                self._last_quotes = quotes

                # Trigger callback if there are changes
                if changed and self.on_update:
                    self.on_update(changed)

            except Exception as e:
                logger.info(f"RealtimeWatcher error: {e}")

            # Wait for next interval
            time.sleep(self.interval)

    def add_symbol(self, symbol: str):
        """Add a symbol to watch."""
        symbol = symbol.upper()
        if symbol not in self.symbols:
            self.symbols.append(symbol)

    def remove_symbol(self, symbol: str):
        """Remove a symbol from watching."""
        symbol = symbol.upper()
        if symbol in self.symbols:
            self.symbols.remove(symbol)

    @property
    def last_quotes(self) -> dict[str, dict]:
        """Get the last fetched quotes."""
        return self._last_quotes.copy()


# Convenience function
def get_realtime_service(cache_ttl: int = 60) -> RealtimeQuoteService:
    """Get a RealtimeQuoteService instance."""
    return RealtimeQuoteService(cache_ttl=cache_ttl)
