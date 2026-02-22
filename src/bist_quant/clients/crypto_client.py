"""Crypto market client via Borsa MCP."""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CryptoAsset:
    symbol: str
    name: str
    price: float
    change_24h: float
    volume_24h: float
    market_cap: float
    source: str


class CryptoClient:
    """Client for crypto market data from BtcTurk/Coinbase via Borsa MCP."""

    def __init__(
        self,
        mcp_endpoint: str = "https://borsamcp.fastmcp.app/mcp",
        timeout: float = 15.0,
        cache_ttl: int = 300,
    ):
        self._mcp_endpoint = mcp_endpoint
        self._session = httpx.AsyncClient(timeout=timeout)
        self._cache: Dict[str, tuple[pd.DataFrame, float]] = {}
        self._cache_ttl = cache_ttl

    async def get_crypto_markets(self, exchange: Optional[str] = None) -> pd.DataFrame:
        """Get crypto market data."""
        ex_key = (exchange or "all").lower()
        cache_key = f"crypto_markets_{ex_key}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        params: Dict[str, Any] = {}
        if exchange:
            params["exchange"] = exchange.lower()

        result = await self._call_mcp_async("get_crypto_market", params)
        records = self._extract_market_records(result, exchange)
        df = pd.DataFrame(records)
        self._cache_set(cache_key, df)
        return df

    async def get_crypto_history(
        self,
        symbol: str,
        exchange: str,
        period: str = "1d",
        interval: str = "1h",
    ) -> pd.DataFrame:
        """Get historical crypto OHLCV data."""
        result = await self._call_mcp_async(
            "get_crypto_market",
            {
                "symbol": symbol.upper(),
                "exchange": exchange.lower(),
                "period": period,
                "interval": interval,
                "history": True,
            },
        )

        records: List[Dict[str, Any]] = []
        data_block = result.get("data")
        if isinstance(data_block, dict):
            candidate = data_block.get("history", data_block.get("ohlc", []))
            if isinstance(candidate, list):
                records.extend(self._normalize_history_record(item) for item in candidate if isinstance(item, dict))
        elif isinstance(data_block, list):
            records.extend(self._normalize_history_record(item) for item in data_block if isinstance(item, dict))

        if not records:
            for text in self._extract_text_blocks(result):
                records.extend(self._parse_history_response(text))

        df = pd.DataFrame([item for item in records if item])
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            numeric_cols = ["open", "high", "low", "close", "volume"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        return df

    async def get_top_gainers_losers(self, exchange: str, count: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Get top gainers and losers by 24h change."""
        all_crypto = await self.get_crypto_markets(exchange=exchange)
        if all_crypto.empty or "change_24h" not in all_crypto.columns:
            return {"gainers": [], "losers": []}

        sorted_crypto = all_crypto.sort_values("change_24h", ascending=False)
        return {
            "gainers": sorted_crypto.head(count).to_dict("records"),
            "losers": sorted_crypto.tail(count).to_dict("records"),
        }

    async def get_crypto_pairs(self, exchange: str) -> List[str]:
        """Get available crypto pairs for an exchange."""
        result = await self._call_mcp_async(
            "get_crypto_market",
            {"exchange": exchange.lower(), "action": "pairs"},
        )

        pairs: List[str] = []
        data_block = result.get("data")
        if isinstance(data_block, dict) and isinstance(data_block.get("pairs"), list):
            for pair in data_block["pairs"]:
                if isinstance(pair, str):
                    pairs.append(pair)
        elif isinstance(data_block, list):
            for pair in data_block:
                if isinstance(pair, str):
                    pairs.append(pair)

        if not pairs:
            for text in self._extract_text_blocks(result):
                pairs.extend(self._parse_pairs_response(text))

        # Preserve order while deduping.
        seen: set[str] = set()
        deduped: List[str] = []
        for pair in pairs:
            norm = pair.strip().upper().replace("-", "/")
            if norm and norm not in seen:
                seen.add(norm)
                deduped.append(norm)
        return deduped

    async def _call_mcp_async(self, tool: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call Borsa MCP endpoint asynchronously."""
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {"name": tool, "arguments": params},
        }

        try:
            response = await self._session.post(
                self._mcp_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            body = response.json()
            if "error" in body:
                message = body["error"].get("message", "Unknown MCP error")
                raise RuntimeError(message)
            return body.get("result", {})
        except httpx.RequestError as exc:
            logger.error("Request error calling MCP: %s", exc)
            raise

    async def close(self) -> None:
        """Close async HTTP client."""
        await self._session.aclose()

    def _extract_market_records(self, result: Dict[str, Any], exchange: Optional[str]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []

        data_block = result.get("data")
        if isinstance(data_block, dict):
            candidate = data_block.get("markets", data_block.get("tickers", data_block))
            if isinstance(candidate, list):
                records.extend(self._normalize_market_record(item, exchange) for item in candidate if isinstance(item, dict))
            elif isinstance(candidate, dict):
                records.append(self._normalize_market_record(candidate, exchange))
        elif isinstance(data_block, list):
            records.extend(self._normalize_market_record(item, exchange) for item in data_block if isinstance(item, dict))

        if records:
            return [row for row in records if row]

        for text in self._extract_text_blocks(result):
            records.extend(self._parse_crypto_response(text, exchange))

        return [row for row in records if row]

    def _parse_crypto_response(self, response_text: str, exchange: Optional[str]) -> List[Dict[str, Any]]:
        """Parse crypto market records from text payload."""
        text = response_text.strip()
        if not text:
            return []

        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            decoded = None

        if decoded is not None:
            out: List[Dict[str, Any]] = []
            if isinstance(decoded, dict):
                candidate = decoded.get("markets", decoded.get("tickers", decoded.get("data", decoded)))
                if isinstance(candidate, list):
                    out.extend(self._normalize_market_record(item, exchange) for item in candidate if isinstance(item, dict))
                elif isinstance(candidate, dict):
                    out.append(self._normalize_market_record(candidate, exchange))
            elif isinstance(decoded, list):
                out.extend(self._normalize_market_record(item, exchange) for item in decoded if isinstance(item, dict))
            return [row for row in out if row]

        records: List[Dict[str, Any]] = []
        for raw in text.splitlines():
            row = self._extract_crypto_info(raw, exchange)
            if row:
                records.append(row)
        return records

    def _extract_crypto_info(self, line: str, exchange: Optional[str]) -> Optional[Dict[str, Any]]:
        """Extract symbol/price/change from a plain text line."""
        text = line.strip()
        if not text:
            return None

        symbol_match = re.search(r"\b([A-Z]{2,10})(?:[/\-]([A-Z]{2,10}))?\b", text)
        price_match = re.search(r"([0-9]+(?:[\.,][0-9]+)?)", text)
        change_match = re.search(r"([+-]?\d+(?:[\.,]\d+)?)%", text)

        if not symbol_match or not price_match:
            return None

        base = symbol_match.group(1)
        quote = symbol_match.group(2)
        symbol = f"{base}/{quote}" if quote else base

        return {
            "symbol": symbol,
            "name": symbol,
            "price": self._to_float(price_match.group(1)),
            "change_24h": self._to_float(change_match.group(1) if change_match else 0.0),
            "volume_24h": 0.0,
            "market_cap": 0.0,
            "source": (exchange or "both").lower(),
        }

    def _parse_history_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse CSV-like OHLCV rows from text payload."""
        rows: List[Dict[str, Any]] = []
        for line in response_text.splitlines():
            cleaned = line.strip()
            if not cleaned or "," not in cleaned:
                continue
            parts = [p.strip() for p in cleaned.split(",")]
            if len(parts) < 5:
                continue
            rows.append(
                {
                    "date": parts[0],
                    "open": self._to_float(parts[1]),
                    "high": self._to_float(parts[2]),
                    "low": self._to_float(parts[3]),
                    "close": self._to_float(parts[4]),
                    "volume": self._to_float(parts[5]) if len(parts) > 5 else 0.0,
                }
            )
        return rows

    def _parse_pairs_response(self, response_text: str) -> List[str]:
        """Parse available trading pairs from text payload."""
        matches = re.findall(r"\b([A-Z]{2,10})[/\-]([A-Z]{2,10})\b", response_text.upper())
        return [f"{base}/{quote}" for base, quote in matches]

    def _normalize_market_record(self, raw: Dict[str, Any], exchange: Optional[str]) -> Dict[str, Any]:
        def pick(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in raw and raw[key] is not None:
                    return raw[key]
            return default

        symbol = str(pick("symbol", "pair", "market", default="")).upper()
        if not symbol:
            return {}

        return {
            "symbol": symbol,
            "name": str(pick("name", default=symbol)),
            "price": self._to_float(pick("price", "last", "last_price", default=0.0)),
            "change_24h": self._to_float(pick("change_24h", "change_percent", "change_pct", default=0.0)),
            "volume_24h": self._to_float(pick("volume_24h", "volume", default=0.0)),
            "market_cap": self._to_float(pick("market_cap", default=0.0)),
            "source": str(pick("source", "exchange", default=(exchange or "unknown"))).lower(),
        }

    def _normalize_history_record(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        def pick(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in raw and raw[key] is not None:
                    return raw[key]
            return default

        return {
            "date": pick("date", "timestamp", "time", default=""),
            "open": self._to_float(pick("open", default=0.0)),
            "high": self._to_float(pick("high", default=0.0)),
            "low": self._to_float(pick("low", default=0.0)),
            "close": self._to_float(pick("close", "price", "last", default=0.0)),
            "volume": self._to_float(pick("volume", default=0.0)),
        }

    def _extract_text_blocks(self, result: Dict[str, Any]) -> List[str]:
        texts: List[str] = []
        content = result.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    texts.append(item["text"])
        return texts

    def _cache_get(self, key: str) -> Optional[pd.DataFrame]:
        item = self._cache.get(key)
        if not item:
            return None
        cached_df, timestamp = item
        if datetime.now().timestamp() - timestamp > self._cache_ttl:
            self._cache.pop(key, None)
            return None
        logger.info("Returning cached crypto data for %s", key)
        return cached_df.copy()

    def _cache_set(self, key: str, value: pd.DataFrame) -> None:
        self._cache[key] = (value.copy(), datetime.now().timestamp())

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip().replace("%", "")
        if not text:
            return default
        if "," in text and "." not in text:
            text = text.replace(",", ".")
        else:
            text = text.replace(",", "")
        try:
            return float(text)
        except ValueError:
            return default
