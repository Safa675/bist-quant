"""US stock market client via Borsa MCP."""

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
class USStockInfo:
    symbol: str
    name: str
    sector: str
    industry: str
    market_cap: float
    pe_ratio: float
    pb_ratio: float
    dividend_yield: float
    eps: float
    revenue: float
    profit_margin: float


class USStockClient:
    """Client for US stock market data via Borsa MCP."""

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

    async def search_us_stocks(self, query: str) -> pd.DataFrame:
        """Search US stocks by symbol/company name."""
        query = query.strip()
        if not query:
            return pd.DataFrame()

        cache_key = f"us_stock_search_{query.lower()}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        result = await self._call_mcp_async(
            "search_symbol",
            {"query": query, "market": "us"},
        )

        records = self._extract_search_records(result)
        df = pd.DataFrame(records)
        self._cache_set(cache_key, df)
        return df

    async def get_us_stock_info(self, symbol: str) -> USStockInfo:
        """Get detailed profile for a US stock."""
        symbol = symbol.upper().strip()
        if not symbol:
            raise ValueError("symbol is required")

        result = await self._call_mcp_async(
            "get_profile",
            {"symbol": symbol, "market": "us"},
        )

        data_block = result.get("data")
        if isinstance(data_block, dict):
            return self._normalize_profile_record(data_block, symbol)

        for text in self._extract_text_blocks(result):
            return self._parse_profile_response(text, symbol)

        raise ValueError(f"No profile data found for {symbol}")

    async def get_us_stock_quotes(self, symbols: List[str]) -> pd.DataFrame:
        """Get quick quote data for multiple symbols."""
        if not symbols:
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        for symbol in symbols:
            try:
                rows.append(await self._get_single_quote(symbol.upper()))
            except Exception as exc:
                logger.warning("Could not get quote for %s: %s", symbol, exc)

        return pd.DataFrame(rows)

    async def get_us_stock_history(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Get historical OHLCV for a US stock."""
        result = await self._call_mcp_async(
            "get_historical_data",
            {
                "symbol": symbol.upper(),
                "market": "us",
                "period": period,
                "interval": interval,
            },
        )

        records: List[Dict[str, Any]] = []
        data_block = result.get("data")
        if isinstance(data_block, dict):
            candidate = data_block.get("history", data_block.get("prices", []))
            if isinstance(candidate, list):
                records.extend(self._normalize_history_record(item) for item in candidate if isinstance(item, dict))
        elif isinstance(data_block, list):
            records.extend(self._normalize_history_record(item) for item in data_block if isinstance(item, dict))

        if not records:
            for text in self._extract_text_blocks(result):
                records.extend(self._parse_history_response(text))

        df = pd.DataFrame([row for row in records if row])
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        return df

    async def get_us_sector_performance(self) -> pd.DataFrame:
        """Get US sector comparison/performance."""
        result = await self._call_mcp_async(
            "get_sector_comparison",
            {"market": "us"},
        )

        rows: List[Dict[str, Any]] = []
        data_block = result.get("data")
        if isinstance(data_block, list):
            for item in data_block:
                if isinstance(item, dict):
                    rows.append(
                        {
                            "sector": str(item.get("sector", item.get("name", ""))),
                            "performance": self._to_float(item.get("performance", item.get("change_percent", 0))),
                        }
                    )
        elif isinstance(data_block, dict):
            candidate = data_block.get("sectors")
            if isinstance(candidate, list):
                for item in candidate:
                    if isinstance(item, dict):
                        rows.append(
                            {
                                "sector": str(item.get("sector", item.get("name", ""))),
                                "performance": self._to_float(item.get("performance", item.get("change_percent", 0))),
                            }
                        )

        if not rows:
            for text in self._extract_text_blocks(result):
                rows.extend(self._parse_sector_response(text))

        return pd.DataFrame(rows)

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

    def _parse_search_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse search results from text payload."""
        text = response_text.strip()
        if not text:
            return []

        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            decoded = None

        if decoded is not None:
            records: List[Dict[str, Any]] = []
            if isinstance(decoded, dict):
                candidate = decoded.get("symbols", decoded.get("results", decoded.get("data", decoded)))
                if isinstance(candidate, list):
                    for item in candidate:
                        if isinstance(item, dict):
                            records.append(self._normalize_search_record(item))
                elif isinstance(candidate, dict):
                    records.append(self._normalize_search_record(candidate))
            elif isinstance(decoded, list):
                for item in decoded:
                    if isinstance(item, dict):
                        records.append(self._normalize_search_record(item))
            return [row for row in records if row]

        rows: List[Dict[str, Any]] = []
        for line in text.splitlines():
            row = self._extract_stock_info(line)
            if row:
                rows.append(row)
        return rows

    def _extract_stock_info(self, line: str) -> Optional[Dict[str, Any]]:
        """Extract symbol/name from plain text search line."""
        text = line.strip()
        if not text:
            return None

        symbol_match = re.search(r"\b([A-Z]{1,5})\b", text)
        if not symbol_match:
            return None

        symbol = symbol_match.group(1)
        name_match = re.search(r"\"([^\"]+)\"", text)
        name = name_match.group(1).strip() if name_match else text.replace(symbol, "").strip(" -:") or symbol

        return {
            "symbol": symbol,
            "name": name,
            "exchange": self._infer_exchange(symbol),
        }

    def _infer_exchange(self, symbol: str) -> str:
        """Infer exchange from symbol length (rough heuristic)."""
        if len(symbol) <= 3:
            return "NYSE"
        if len(symbol) == 4:
            return "NASDAQ"
        return "OTHER"

    def _parse_profile_response(self, response_text: str, symbol: str) -> USStockInfo:
        """Parse profile text into USStockInfo."""
        info = USStockInfo(
            symbol=symbol,
            name=symbol,
            sector="Unknown",
            industry="Unknown",
            market_cap=0.0,
            pe_ratio=0.0,
            pb_ratio=0.0,
            dividend_yield=0.0,
            eps=0.0,
            revenue=0.0,
            profit_margin=0.0,
        )

        for raw in response_text.splitlines():
            line = raw.strip()
            if not line:
                continue
            lower = line.lower()
            if ":" in line:
                key, value = line.split(":", 1)
                k = key.strip().lower()
                v = value.strip()
                if k == "name":
                    info.name = v
                elif k == "sector":
                    info.sector = v
                elif k == "industry":
                    info.industry = v
                elif "market cap" in k:
                    info.market_cap = self._to_float(v)
                elif k in {"p/e", "pe", "pe ratio"}:
                    info.pe_ratio = self._to_float(v)
                elif k in {"p/b", "pb", "pb ratio"}:
                    info.pb_ratio = self._to_float(v)
                elif "dividend" in k:
                    info.dividend_yield = self._to_float(v)
                elif k == "eps":
                    info.eps = self._to_float(v)
                elif "revenue" in k:
                    info.revenue = self._to_float(v)
                elif "margin" in k:
                    info.profit_margin = self._to_float(v)
            elif "market cap" in lower:
                info.market_cap = self._to_float(line)
        return info

    async def _get_single_quote(self, symbol: str) -> Dict[str, Any]:
        """Get quote for one symbol."""
        result = await self._call_mcp_async(
            "get_quick_info",
            {"symbol": symbol.upper(), "market": "us"},
        )

        data_block = result.get("data")
        if isinstance(data_block, dict):
            return self._normalize_quote_record(data_block, symbol)

        for text in self._extract_text_blocks(result):
            return self._parse_quote_response(text, symbol)

        raise ValueError(f"No quote data found for {symbol}")

    def _parse_quote_response(self, response_text: str, symbol: str) -> Dict[str, Any]:
        """Parse quote text payload."""
        quote = {
            "symbol": symbol,
            "price": 0.0,
            "change": 0.0,
            "change_percent": 0.0,
            "volume": 0.0,
            "high": 0.0,
            "low": 0.0,
            "open": 0.0,
        }

        for raw in response_text.splitlines():
            line = raw.strip()
            if not line:
                continue
            lower = line.lower()
            if "price" in lower:
                quote["price"] = self._to_float(line)
            elif "change" in lower:
                pct_match = re.search(r"([+-]?\d+(?:[\.,]\d+)?)%", line)
                if pct_match:
                    quote["change_percent"] = self._to_float(pct_match.group(1))
                abs_match = re.search(r"([+-]?\d+(?:[\.,]\d+)?)", line)
                if abs_match:
                    quote["change"] = self._to_float(abs_match.group(1))
            elif "volume" in lower:
                quote["volume"] = self._to_float(line)
            elif lower.startswith("high"):
                quote["high"] = self._to_float(line)
            elif lower.startswith("low"):
                quote["low"] = self._to_float(line)
            elif lower.startswith("open"):
                quote["open"] = self._to_float(line)

        return quote

    def _parse_history_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse CSV-like OHLCV rows from text payload."""
        rows: List[Dict[str, Any]] = []
        for raw in response_text.splitlines():
            line = raw.strip()
            if not line or "," not in line:
                continue
            parts = [item.strip() for item in line.split(",")]
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

    def _parse_sector_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse sector comparison text payload."""
        rows: List[Dict[str, Any]] = []
        for raw in response_text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if ":" not in line:
                continue
            sector, rest = line.split(":", 1)
            pct_match = re.search(r"([+-]?\d+(?:[\.,]\d+)?)%", rest)
            rows.append(
                {
                    "sector": sector.strip(),
                    "performance": self._to_float(pct_match.group(1) if pct_match else rest),
                }
            )
        return rows

    def _extract_search_records(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        data_block = result.get("data")
        if isinstance(data_block, list):
            for item in data_block:
                if isinstance(item, dict):
                    rows.append(self._normalize_search_record(item))
        elif isinstance(data_block, dict):
            candidate = data_block.get("results", data_block.get("symbols", data_block))
            if isinstance(candidate, list):
                for item in candidate:
                    if isinstance(item, dict):
                        rows.append(self._normalize_search_record(item))
            elif isinstance(candidate, dict):
                rows.append(self._normalize_search_record(candidate))

        if rows:
            return [row for row in rows if row]

        for text in self._extract_text_blocks(result):
            rows.extend(self._parse_search_response(text))
        return [row for row in rows if row]

    def _normalize_search_record(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        symbol = str(raw.get("symbol") or raw.get("ticker") or "").upper()
        if not symbol:
            return {}
        return {
            "symbol": symbol,
            "name": str(raw.get("name") or raw.get("company") or symbol),
            "exchange": str(raw.get("exchange") or self._infer_exchange(symbol)),
        }

    def _normalize_profile_record(self, raw: Dict[str, Any], symbol: str) -> USStockInfo:
        return USStockInfo(
            symbol=symbol,
            name=str(raw.get("name", symbol)),
            sector=str(raw.get("sector", "Unknown")),
            industry=str(raw.get("industry", "Unknown")),
            market_cap=self._to_float(raw.get("market_cap", 0.0)),
            pe_ratio=self._to_float(raw.get("pe_ratio", raw.get("pe", 0.0))),
            pb_ratio=self._to_float(raw.get("pb_ratio", raw.get("pb", 0.0))),
            dividend_yield=self._to_float(raw.get("dividend_yield", 0.0)),
            eps=self._to_float(raw.get("eps", 0.0)),
            revenue=self._to_float(raw.get("revenue", 0.0)),
            profit_margin=self._to_float(raw.get("profit_margin", 0.0)),
        )

    def _normalize_quote_record(self, raw: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "price": self._to_float(raw.get("price", raw.get("last", raw.get("last_price", 0.0)))),
            "change": self._to_float(raw.get("change", 0.0)),
            "change_percent": self._to_float(raw.get("change_percent", raw.get("change_pct", 0.0))),
            "volume": self._to_float(raw.get("volume", 0.0)),
            "high": self._to_float(raw.get("high", 0.0)),
            "low": self._to_float(raw.get("low", 0.0)),
            "open": self._to_float(raw.get("open", 0.0)),
        }

    def _normalize_history_record(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "date": raw.get("date") or raw.get("timestamp") or raw.get("time") or "",
            "open": self._to_float(raw.get("open", 0.0)),
            "high": self._to_float(raw.get("high", 0.0)),
            "low": self._to_float(raw.get("low", 0.0)),
            "close": self._to_float(raw.get("close", raw.get("price", 0.0))),
            "volume": self._to_float(raw.get("volume", 0.0)),
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
        logger.info("Returning cached US stock data for %s", key)
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
