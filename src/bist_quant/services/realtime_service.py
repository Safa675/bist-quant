"""Realtime market data service built on top of borsapy-compatible clients."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any
from zoneinfo import ZoneInfo


class MarketStatus(Enum):
    """Market status for BIST trading session."""

    OPEN = "open"
    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    POST_MARKET = "post_market"
    UNKNOWN = "unknown"


@dataclass
class Quote:
    """Real-time quote for a symbol."""

    symbol: str
    price: Decimal
    change: Decimal
    change_percent: Decimal
    volume: int
    bid: Decimal | None
    ask: Decimal | None
    high: Decimal
    low: Decimal
    open: Decimal
    previous_close: Decimal
    timestamp: datetime
    market_status: MarketStatus


@dataclass
class IndexData:
    """Index data (XU100, XU030, etc.)."""

    symbol: str
    value: Decimal
    change: Decimal
    change_percent: Decimal
    high: Decimal
    low: Decimal
    timestamp: datetime


@dataclass
class FXRate:
    """Foreign exchange rate."""

    pair: str
    rate: Decimal
    change: Decimal
    change_percent: Decimal
    timestamp: datetime


@dataclass
class PortfolioPosition:
    """Single position in a realtime portfolio valuation."""

    symbol: str
    quantity: Decimal
    average_cost: Decimal
    current_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    unrealized_pnl_percent: Decimal


@dataclass
class PortfolioValuation:
    """Portfolio valuation with realtime prices."""

    positions: list[PortfolioPosition]
    total_value: Decimal
    total_cost: Decimal
    total_pnl: Decimal
    total_pnl_percent: Decimal
    cash: Decimal
    timestamp: datetime


@dataclass
class MarketSummary:
    """Market summary with indices and FX data."""

    indices: list[IndexData]
    fx_rates: list[FXRate]
    market_status: MarketStatus
    timestamp: datetime


class QuoteCache:
    """TTL-based in-memory cache for realtime quotes."""

    def __init__(self, ttl_seconds: int = 5):
        self.ttl = timedelta(seconds=max(0, int(ttl_seconds)))
        self._cache: dict[str, tuple[Quote, datetime]] = {}

    def get(self, symbol: str) -> Quote | None:
        """Get cached quote if it exists and is still valid."""
        key = str(symbol or "").strip().upper()
        if not key:
            return None

        row = self._cache.get(key)
        if row is None:
            return None

        quote, created_at = row
        if datetime.now(timezone.utc) - created_at > self.ttl:
            self._cache.pop(key, None)
            return None
        return quote

    def set(self, symbol: str, quote: Quote) -> None:
        """Store quote in cache."""
        key = str(symbol or "").strip().upper()
        if not key:
            return
        self._cache[key] = (quote, datetime.now(timezone.utc))

    def invalidate(self, symbol: str | None = None) -> None:
        """Invalidate one symbol or clear the entire cache."""
        if symbol is None:
            self._cache.clear()
            return
        key = str(symbol or "").strip().upper()
        if key:
            self._cache.pop(key, None)


class RealtimeServiceError(Exception):
    """Base exception for realtime service failures."""


class SymbolNotFoundError(RealtimeServiceError):
    """Raised when a symbol cannot be resolved by the data source."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        super().__init__(f"Symbol not found: {symbol}")


class MarketDataUnavailableError(RealtimeServiceError):
    """Raised when market data provider is temporarily unavailable."""


class RealtimeService:
    """Service for realtime quotes, indices, FX, and portfolio valuation."""

    def __init__(
        self,
        cache_ttl_seconds: int = 5,
        borsapy_client: Any | None = None,
        fx_client: Any | None = None,
    ):
        self.cache = QuoteCache(ttl_seconds=cache_ttl_seconds)
        self._borsapy_client = borsapy_client
        self._fx_client = fx_client
        self._market_tz = ZoneInfo("Europe/Istanbul")

    @property
    def borsapy(self) -> Any:
        """Lazy-load borsapy-compatible client."""
        if self._borsapy_client is None:
            try:
                from borsapy import BorsaClient  # type: ignore

                self._borsapy_client = BorsaClient()
            except Exception:
                import borsapy as bp  # type: ignore

                self._borsapy_client = bp
        return self._borsapy_client

    @property
    def fx(self) -> Any:
        """Lazy-load FX client with fallbacks to borsapy."""
        if self._fx_client is None:
            borsapy_client = self.borsapy
            if hasattr(borsapy_client, "get_fx_rate") or hasattr(borsapy_client, "FX"):
                self._fx_client = borsapy_client
                return self._fx_client

            try:
                from bist_quant.data import FXClient  # type: ignore

                self._fx_client = FXClient()
            except Exception:
                try:
                    from bist_quant.clients.fx_commodities_client import FXCommoditiesClient

                    self._fx_client = FXCommoditiesClient()
                except Exception:
                    self._fx_client = self.borsapy
        return self._fx_client

    async def get_quote(self, symbol: str, use_cache: bool = True) -> Quote:
        """
        Get realtime quote for a symbol.

        Raises:
            SymbolNotFoundError: Symbol is invalid or unknown.
            MarketDataUnavailableError: Upstream market data source failed.
        """
        normalized = self._normalize_symbol(symbol)
        if not normalized:
            raise SymbolNotFoundError(str(symbol))

        if use_cache:
            cached = self.cache.get(normalized)
            if cached is not None:
                return cached

        payload = await self._fetch_quote_payload(normalized)
        quote = self._build_quote(normalized, payload)
        self.cache.set(normalized, quote)
        return quote

    async def get_quotes(
        self,
        symbols: list[str],
        use_cache: bool = True,
    ) -> dict[str, Quote]:
        """Get realtime quotes for multiple symbols."""
        normalized: list[str] = []
        seen: set[str] = set()
        for symbol in symbols:
            key = self._normalize_symbol(symbol)
            if not key or key in seen:
                continue
            seen.add(key)
            normalized.append(key)

        tasks = [self.get_quote(symbol, use_cache=use_cache) for symbol in normalized]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        out: dict[str, Quote] = {}
        for symbol, result in zip(normalized, results, strict=False):
            if isinstance(result, Exception):
                raise result
            out[symbol] = result
        return out

    async def get_index(self, symbol: str = "XU100") -> IndexData:
        """Get realtime index data."""
        normalized = self._normalize_symbol(symbol) or "XU100"
        payload = await self._fetch_index_payload(normalized)

        value = self._pick_decimal(payload, ["value", "last"])
        if value is None:
            self._raise_from_payload_error(normalized, payload)
            raise MarketDataUnavailableError(f"No index value available for {normalized}")

        prev_close = self._pick_decimal(payload, ["previous_close", "prev_close", "close"])
        change = self._pick_decimal(payload, ["change"])
        if change is None and prev_close is not None:
            change = value - prev_close
        if change is None:
            change = Decimal("0")

        change_percent = self._normalize_change_percent(
            payload.get("change_percent", payload.get("change_pct")),
            value,
            prev_close,
        )
        if change_percent is None:
            change_percent = Decimal("0")

        high = self._pick_decimal(payload, ["high", "day_high"]) or value
        low = self._pick_decimal(payload, ["low", "day_low"]) or value
        timestamp = self._normalize_timestamp(payload.get("timestamp", payload.get("update_time")))

        return IndexData(
            symbol=normalized,
            value=value,
            change=change,
            change_percent=change_percent,
            high=high,
            low=low,
            timestamp=timestamp,
        )

    async def get_indices(self, symbols: list[str] | None = None) -> list[IndexData]:
        """Get multiple index snapshots."""
        targets = symbols or ["XU100", "XU030", "XU050", "XBANK"]
        tasks = [self.get_index(symbol) for symbol in targets]
        return list(await asyncio.gather(*tasks))

    async def get_fx_rate(self, pair: str = "USD/TRY") -> FXRate:
        """Get realtime FX rate for a currency pair."""
        normalized_pair = self._normalize_pair(pair)
        payload = await self._fetch_fx_payload(normalized_pair)

        rate = self._pick_decimal(payload, ["rate", "last", "price", "last_price"])
        if rate is None:
            self._raise_from_payload_error(normalized_pair, payload)
            raise MarketDataUnavailableError(f"No FX rate available for {normalized_pair}")

        open_rate = self._pick_decimal(payload, ["open", "previous_close", "prev_close", "close"])
        change = self._pick_decimal(payload, ["change"])
        if change is None and open_rate is not None:
            change = rate - open_rate
        if change is None:
            change = Decimal("0")

        change_percent = self._normalize_change_percent(
            payload.get("change_percent", payload.get("change_pct")),
            rate,
            open_rate,
        )
        if change_percent is None:
            change_percent = Decimal("0")

        timestamp = self._normalize_timestamp(payload.get("timestamp", payload.get("update_time")))
        return FXRate(
            pair=normalized_pair,
            rate=rate,
            change=change,
            change_percent=change_percent,
            timestamp=timestamp,
        )

    async def get_fx_rates(self, pairs: list[str] | None = None) -> list[FXRate]:
        """Get multiple FX rates."""
        targets = pairs or ["USD/TRY", "EUR/TRY", "GBP/TRY"]
        tasks = [self.get_fx_rate(pair) for pair in targets]
        return list(await asyncio.gather(*tasks))

    async def get_portfolio_valuation(
        self,
        positions: list[dict[str, Any]],
        cash: Decimal = Decimal("0"),
    ) -> PortfolioValuation:
        """Compute portfolio valuation using realtime quote prices."""
        normalized_positions: list[tuple[str, Decimal, Decimal]] = []
        symbols: list[str] = []
        for row in positions:
            symbol = self._normalize_symbol(row.get("symbol", ""))
            if not symbol:
                continue
            quantity = self._normalize_price(row.get("quantity", 0))
            average_cost = self._normalize_price(row.get("average_cost", 0))
            normalized_positions.append((symbol, quantity, average_cost))
            symbols.append(symbol)

        if not normalized_positions:
            safe_cash = self._normalize_price(cash)
            return PortfolioValuation(
                positions=[],
                total_value=safe_cash,
                total_cost=Decimal("0"),
                total_pnl=Decimal("0"),
                total_pnl_percent=Decimal("0"),
                cash=safe_cash,
                timestamp=datetime.now(timezone.utc),
            )

        quote_map = await self.get_quotes(symbols, use_cache=True)

        portfolio_positions: list[PortfolioPosition] = []
        total_market_value = Decimal("0")
        total_cost = Decimal("0")
        for symbol, quantity, average_cost in normalized_positions:
            quote = quote_map[symbol]
            market_value = quote.price * quantity
            cost_basis = average_cost * quantity
            unrealized_pnl = market_value - cost_basis
            unrealized_pnl_percent = (
                (unrealized_pnl / cost_basis) * Decimal("100")
                if cost_basis != Decimal("0")
                else Decimal("0")
            )
            total_market_value += market_value
            total_cost += cost_basis
            portfolio_positions.append(
                PortfolioPosition(
                    symbol=symbol,
                    quantity=quantity,
                    average_cost=average_cost,
                    current_price=quote.price,
                    market_value=market_value,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_percent=unrealized_pnl_percent,
                )
            )

        safe_cash = self._normalize_price(cash)
        total_pnl = total_market_value - total_cost
        total_pnl_percent = (
            (total_pnl / total_cost) * Decimal("100") if total_cost != Decimal("0") else Decimal("0")
        )
        return PortfolioValuation(
            positions=portfolio_positions,
            total_value=total_market_value + safe_cash,
            total_cost=total_cost,
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_percent,
            cash=safe_cash,
            timestamp=datetime.now(timezone.utc),
        )

    async def get_market_summary(self) -> MarketSummary:
        """Get market summary with major indices and FX rates."""
        indices_task = asyncio.create_task(self.get_indices())
        fx_task = asyncio.create_task(self.get_fx_rates())
        indices, fx_rates = await asyncio.gather(indices_task, fx_task)
        return MarketSummary(
            indices=indices,
            fx_rates=fx_rates,
            market_status=self._determine_market_status(),
            timestamp=datetime.now(timezone.utc),
        )

    def _normalize_price(self, raw_price: Any) -> Decimal:
        """Normalize number-like values (including Turkish format) to Decimal."""
        if isinstance(raw_price, Decimal):
            return raw_price

        if raw_price is None:
            raise ValueError("Price cannot be None.")

        if isinstance(raw_price, bool):
            raise ValueError("Boolean cannot be parsed as price.")

        if isinstance(raw_price, int):
            return Decimal(raw_price)

        if isinstance(raw_price, float):
            if raw_price != raw_price or raw_price in (float("inf"), float("-inf")):
                raise ValueError("Price must be finite.")
            return Decimal(str(raw_price))

        text = str(raw_price).strip()
        if not text:
            raise ValueError("Price cannot be empty.")

        compact = text.replace(" ", "")
        if "," in compact and "." in compact:
            if compact.rfind(",") > compact.rfind("."):
                compact = compact.replace(".", "").replace(",", ".")
            else:
                compact = compact.replace(",", "")
        elif "," in compact:
            compact = compact.replace(",", ".")

        try:
            return Decimal(compact)
        except InvalidOperation as exc:
            raise ValueError(f"Invalid price value: {raw_price}") from exc

    def _determine_market_status(self) -> MarketStatus:
        """Determine market status using Europe/Istanbul trading hours."""
        now = datetime.now(self._market_tz)
        if now.weekday() >= 5:
            return MarketStatus.CLOSED

        current = now.time()
        if time(9, 0) <= current < time(10, 0):
            return MarketStatus.PRE_MARKET
        if time(10, 0) <= current < time(18, 0):
            return MarketStatus.OPEN
        if time(18, 0) <= current < time(19, 0):
            return MarketStatus.POST_MARKET
        return MarketStatus.CLOSED

    async def _fetch_quote_payload(self, symbol: str) -> dict[str, Any]:
        client = self.borsapy
        if hasattr(client, "get_quote"):
            try:
                payload = await self._call_client_method(client.get_quote, symbol)
            except Exception as exc:
                self._raise_from_exception(symbol, exc)
            if isinstance(payload, dict):
                return payload

        if hasattr(client, "Ticker"):
            try:
                ticker = await self._call_client_method(client.Ticker, symbol)
                fast_info = getattr(ticker, "fast_info", {})
                if hasattr(fast_info, "todict"):
                    fast_info = fast_info.todict()
                return dict(fast_info or {})
            except Exception as exc:
                self._raise_from_exception(symbol, exc)

        self._raise_data_unavailable(symbol)

    async def _fetch_index_payload(self, symbol: str) -> dict[str, Any]:
        client = self.borsapy
        if hasattr(client, "get_index"):
            try:
                payload = await self._call_client_method(client.get_index, symbol)
            except Exception as exc:
                self._raise_from_exception(symbol, exc)
            if isinstance(payload, dict):
                return payload
            if hasattr(payload, "info") and isinstance(payload.info, dict):
                return payload.info

        if hasattr(client, "index"):
            try:
                idx = await self._call_client_method(client.index, symbol)
                info = getattr(idx, "info", idx)
                if isinstance(info, dict):
                    return info
            except Exception as exc:
                self._raise_from_exception(symbol, exc)

        self._raise_data_unavailable(symbol)

    async def _fetch_fx_payload(self, pair: str) -> dict[str, Any]:
        fx_client = self.fx
        if hasattr(fx_client, "get_fx_rate"):
            try:
                payload = await self._call_client_method(fx_client.get_fx_rate, pair)
            except Exception as exc:
                self._raise_from_exception(pair, exc)
            if isinstance(payload, dict):
                return payload

        if hasattr(fx_client, "get_fx_rates"):
            try:
                payload = await self._call_client_method(fx_client.get_fx_rates, [pair])
            except Exception as exc:
                self._raise_from_exception(pair, exc)
            row = self._extract_fx_row(payload, pair)
            if row:
                return row

        asset = pair.split("/", 1)[0]
        if hasattr(fx_client, "FX"):
            try:
                fx_obj = await self._call_client_method(fx_client.FX, asset)
                info = getattr(fx_obj, "info", fx_obj)
                if isinstance(info, dict):
                    return info
            except Exception as exc:
                self._raise_from_exception(pair, exc)

        self._raise_data_unavailable(pair)

    def _extract_fx_row(self, payload: Any, pair: str) -> dict[str, Any]:
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list) and payload:
            first = payload[0]
            return first if isinstance(first, dict) else {}

        if hasattr(payload, "empty") and hasattr(payload, "to_dict"):
            if bool(getattr(payload, "empty", True)):
                return {}
            records = payload.to_dict("records")
            if not records:
                return {}
            for row in records:
                if not isinstance(row, dict):
                    continue
                row_pair = self._normalize_pair(row.get("pair", ""))
                if row_pair == pair:
                    return row
            first = records[0]
            return first if isinstance(first, dict) else {}
        return {}

    async def _call_client_method(self, method: Any, *args: Any) -> Any:
        if asyncio.iscoroutinefunction(method):
            return await method(*args)
        result = await asyncio.to_thread(method, *args)
        if asyncio.iscoroutine(result):
            return await result
        return result

    def _build_quote(self, symbol: str, payload: dict[str, Any]) -> Quote:
        price = self._pick_decimal(payload, ["price", "last_price", "last", "value"])
        if price is None:
            self._raise_from_payload_error(symbol, payload)
            raise MarketDataUnavailableError(f"No quote price available for {symbol}")

        prev_close = self._pick_decimal(payload, ["previous_close", "prev_close", "close"]) or price
        change = self._pick_decimal(payload, ["change"])
        if change is None:
            change = price - prev_close

        change_percent = self._normalize_change_percent(
            payload.get("change_percent", payload.get("change_pct")),
            price,
            prev_close,
        )
        if change_percent is None:
            if prev_close != Decimal("0"):
                change_percent = (change / prev_close) * Decimal("100")
            else:
                change_percent = Decimal("0")

        volume = self._pick_decimal(payload, ["volume"])
        volume_int = int(volume) if volume is not None else 0
        bid = self._pick_decimal(payload, ["bid"])
        ask = self._pick_decimal(payload, ["ask"])
        high = self._pick_decimal(payload, ["high", "day_high"]) or price
        low = self._pick_decimal(payload, ["low", "day_low"]) or price
        open_price = self._pick_decimal(payload, ["open"]) or prev_close
        timestamp = self._normalize_timestamp(payload.get("timestamp", payload.get("update_time")))

        return Quote(
            symbol=symbol,
            price=price,
            change=change,
            change_percent=change_percent,
            volume=volume_int,
            bid=bid,
            ask=ask,
            high=high,
            low=low,
            open=open_price,
            previous_close=prev_close,
            timestamp=timestamp,
            market_status=self._determine_market_status(),
        )

    def _normalize_symbol(self, symbol: str) -> str:
        return str(symbol or "").strip().upper().split(".")[0]

    def _normalize_pair(self, pair: str) -> str:
        text = str(pair or "").strip().upper().replace("-", "/")
        if not text:
            return "USD/TRY"
        if "/" in text:
            base, quote = text.split("/", 1)
            if base and quote:
                return f"{base}/{quote}"
            return "USD/TRY"
        if len(text) == 6:
            return f"{text[:3]}/{text[3:]}"
        if len(text) == 3:
            return f"{text}/TRY"
        return text

    def _pick_decimal(self, payload: dict[str, Any], keys: list[str]) -> Decimal | None:
        for key in keys:
            if key not in payload:
                continue
            raw = payload.get(key)
            if raw is None:
                continue
            try:
                return self._normalize_price(raw)
            except ValueError:
                continue
        return None

    def _normalize_timestamp(self, raw_timestamp: Any) -> datetime:
        if isinstance(raw_timestamp, datetime):
            if raw_timestamp.tzinfo is None:
                return raw_timestamp.replace(tzinfo=timezone.utc)
            return raw_timestamp.astimezone(timezone.utc)

        if isinstance(raw_timestamp, str):
            text = raw_timestamp.strip()
            if text:
                if text.endswith("Z"):
                    text = f"{text[:-1]}+00:00"
                try:
                    parsed = datetime.fromisoformat(text)
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=timezone.utc)
                    return parsed.astimezone(timezone.utc)
                except ValueError:
                    pass

        return datetime.now(timezone.utc)

    def _normalize_change_percent(
        self,
        raw_change_percent: Any,
        value: Decimal | None,
        previous: Decimal | None,
    ) -> Decimal | None:
        computed: Decimal | None = None
        if value is not None and previous not in (None, Decimal("0")):
            computed = ((value - previous) / previous) * Decimal("100")

        provided: Decimal | None = None
        if raw_change_percent is not None:
            try:
                provided = self._normalize_price(raw_change_percent)
            except ValueError:
                provided = None

        if provided is None:
            return computed
        if computed is None:
            return provided

        if abs(provided - computed) <= Decimal("0.15"):
            return provided
        scaled = provided * Decimal("100")
        if abs(scaled - computed) <= Decimal("0.15"):
            return scaled
        return computed

    def _raise_from_payload_error(self, symbol: str, payload: dict[str, Any]) -> None:
        message = str(payload.get("error", "")).strip()
        if not message:
            return
        if self._looks_like_symbol_not_found(message):
            raise SymbolNotFoundError(symbol)
        raise MarketDataUnavailableError(message)

    def _raise_from_exception(self, symbol: str, exc: Exception) -> None:
        message = str(exc).strip() or f"Failed to fetch market data for {symbol}"
        if self._looks_like_symbol_not_found(message):
            raise SymbolNotFoundError(symbol) from exc
        raise MarketDataUnavailableError(message) from exc

    def _raise_data_unavailable(self, symbol: str) -> None:
        raise MarketDataUnavailableError(f"Market data unavailable for {symbol}")

    def _looks_like_symbol_not_found(self, message: str) -> bool:
        lowered = message.lower()
        if "not found" in lowered:
            return True
        if "unknown symbol" in lowered:
            return True
        if "invalid symbol" in lowered:
            return True
        if "symbol not found" in lowered:
            return True
        return False


__all__ = [
    "FXRate",
    "IndexData",
    "MarketDataUnavailableError",
    "MarketStatus",
    "MarketSummary",
    "PortfolioPosition",
    "PortfolioValuation",
    "Quote",
    "QuoteCache",
    "RealtimeService",
    "RealtimeServiceError",
    "SymbolNotFoundError",
]
