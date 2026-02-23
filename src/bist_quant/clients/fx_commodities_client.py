"""FX and commodities client via Borsa MCP."""

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

from bist_quant.clients.fx_enhanced_provider import FXEnhancedProvider

logger = logging.getLogger(__name__)


@dataclass
class FXPair:
    pair: str
    bid: float
    ask: float
    last_price: float
    change: float
    change_percent: float
    high: float
    low: float
    volume: float
    timestamp: str


class FXCommoditiesClient:
    """Client for FX and commodities data via Borsa MCP."""

    def __init__(
        self,
        mcp_endpoint: str = "https://borsamcp.fastmcp.app/mcp",
        timeout: float = 15.0,
        cache_ttl: int = 60,
        enhanced_provider: FXEnhancedProvider | None = None,
        include_bank_rates: bool = True,
        cache_dir: Any = None,
    ):
        self._mcp_endpoint = mcp_endpoint
        self._session = httpx.AsyncClient(timeout=timeout)
        self._cache: Dict[str, tuple[pd.DataFrame, float]] = {}
        self._cache_ttl = cache_ttl
        self._enhanced_provider = enhanced_provider or FXEnhancedProvider()
        self._include_bank_rates = include_bank_rates
        self._disk_cache: Any | None = None
        if cache_dir is not None:
            try:
                from pathlib import Path as _Path
                from bist_quant.common.disk_cache import DiskCache as _DiskCache
                self._disk_cache = _DiskCache(_Path(cache_dir))
            except Exception:
                pass

    async def get_fx_rates(self, pairs: Optional[List[str]] = None) -> pd.DataFrame:
        """Get FX rates for given pairs or all available."""
        normalized_pairs = sorted(self._normalize_pair(pair) for pair in (pairs or []) if pair)
        cache_key = f"fx_rates_{'_'.join(normalized_pairs) if normalized_pairs else 'all'}"

        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        params: Dict[str, Any] = {}
        if normalized_pairs:
            params["pairs"] = normalized_pairs

        result = await self._call_mcp_async("get_fx_data", params)
        rows = self._extract_fx_records(result)
        df = pd.DataFrame(rows)
        if not df.empty and "source" not in df.columns:
            df["source"] = "mcp"

        if self._include_bank_rates:
            try:
                bank_rows = self._build_bank_rate_rows(normalized_pairs)
                if not bank_rows.empty:
                    if df.empty:
                        df = bank_rows
                    else:
                        df = pd.concat([df, bank_rows], ignore_index=True, sort=False)
                    if normalized_pairs and "pair" in df.columns:
                        df = df[df["pair"].isin(normalized_pairs)].reset_index(drop=True)
            except Exception as exc:
                logger.info("Bank-rate FX supplement skipped: %s", exc)

        self._cache_set(cache_key, df)
        return df

    async def get_commodity_prices(self, commodities: Optional[List[str]] = None) -> pd.DataFrame:
        """Get commodity prices."""
        normalized_assets = sorted(item.lower() for item in (commodities or []) if item)
        cache_key = f"commodities_{'_'.join(normalized_assets) if normalized_assets else 'all'}"

        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        params: Dict[str, Any] = {"asset_type": "commodity"}
        if normalized_assets:
            params["assets"] = normalized_assets

        result = await self._call_mcp_async("get_fx_data", params)
        rows = self._extract_commodity_records(result)
        df = pd.DataFrame(rows)

        self._cache_set(cache_key, df)
        return df

    async def get_precious_metals(self) -> pd.DataFrame:
        """Get precious metals snapshot."""
        return await self.get_commodity_prices(["gold", "silver", "platinum", "palladium"])

    async def get_energy_prices(self) -> pd.DataFrame:
        """Get energy commodity snapshot."""
        return await self.get_commodity_prices(["oil", "gas", "coal", "uranium"])

    async def get_agricultural_prices(self) -> pd.DataFrame:
        """Get agricultural commodity snapshot."""
        return await self.get_commodity_prices(
            ["wheat", "corn", "soybean", "cotton", "coffee", "sugar", "cocoa", "orange juice"]
        )

    async def get_fx_history(
        self,
        pair: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Get historical FX OHLCV data."""
        result = await self._call_mcp_async(
            "get_fx_data",
            {
                "pair": self._normalize_pair(pair),
                "period": period,
                "interval": interval,
                "history": True,
            },
        )

        rows: List[Dict[str, Any]] = []
        data_block = result.get("data")
        if isinstance(data_block, dict):
            candidate = data_block.get("history", data_block.get("ohlc", []))
            if isinstance(candidate, list):
                rows.extend(self._normalize_history_record(item) for item in candidate if isinstance(item, dict))
        elif isinstance(data_block, list):
            rows.extend(self._normalize_history_record(item) for item in data_block if isinstance(item, dict))

        if not rows:
            for text in self._extract_text_blocks(result):
                rows.extend(self._parse_history_response(text))

        df = pd.DataFrame([row for row in rows if row])
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        return df

    async def get_asset_history(self, asset: str) -> pd.DataFrame:
        """Get historical data for an FX pair or commodity."""
        normalized = asset.strip().upper()
        if "/" in normalized or len(normalized) == 6:
            return await self.get_fx_history(normalized)
        return pd.DataFrame()

    async def get_correlation_matrix(self, assets: List[str]) -> pd.DataFrame:
        """Build simple correlation matrix from available close series."""
        if not assets:
            return pd.DataFrame()

        returns_map: Dict[str, pd.Series] = {}
        for asset in assets:
            try:
                hist = await self.get_asset_history(asset)
            except Exception as exc:
                logger.warning("Could not fetch history for %s: %s", asset, exc)
                continue
            if hist.empty or "close" not in hist.columns:
                continue
            series = pd.to_numeric(hist["close"], errors="coerce").pct_change().dropna()
            if not series.empty:
                returns_map[asset] = series.reset_index(drop=True)

        if len(returns_map) < 2:
            return pd.DataFrame()

        aligned = pd.DataFrame(returns_map)
        return aligned.corr()

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

    def _extract_fx_records(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        data_block = result.get("data")
        if isinstance(data_block, dict):
            candidate = data_block.get("rates", data_block.get("pairs", data_block))
            if isinstance(candidate, list):
                rows.extend(self._normalize_fx_record(item) for item in candidate if isinstance(item, dict))
            elif isinstance(candidate, dict):
                rows.append(self._normalize_fx_record(candidate))
        elif isinstance(data_block, list):
            rows.extend(self._normalize_fx_record(item) for item in data_block if isinstance(item, dict))

        if rows:
            return [row for row in rows if row]

        for text in self._extract_text_blocks(result):
            rows.extend(self._parse_fx_response(text))
        return [row for row in rows if row]

    def _extract_commodity_records(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        data_block = result.get("data")
        if isinstance(data_block, dict):
            candidate = data_block.get("commodities", data_block.get("assets", data_block))
            if isinstance(candidate, list):
                rows.extend(self._normalize_commodity_record(item) for item in candidate if isinstance(item, dict))
            elif isinstance(candidate, dict):
                rows.append(self._normalize_commodity_record(candidate))
        elif isinstance(data_block, list):
            rows.extend(self._normalize_commodity_record(item) for item in data_block if isinstance(item, dict))

        if rows:
            return [row for row in rows if row]

        for text in self._extract_text_blocks(result):
            rows.extend(self._parse_commodity_response(text))
        return [row for row in rows if row]

    def _parse_fx_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse FX rows from text payload."""
        text = response_text.strip()
        if not text:
            return []

        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            decoded = None

        if decoded is not None:
            rows: List[Dict[str, Any]] = []
            if isinstance(decoded, list):
                rows.extend(self._normalize_fx_record(item) for item in decoded if isinstance(item, dict))
            elif isinstance(decoded, dict):
                candidate = decoded.get("rates", decoded.get("pairs", decoded))
                if isinstance(candidate, list):
                    rows.extend(self._normalize_fx_record(item) for item in candidate if isinstance(item, dict))
                elif isinstance(candidate, dict):
                    rows.append(self._normalize_fx_record(candidate))
            return [row for row in rows if row]

        rows = []
        for line in text.splitlines():
            rec = self._extract_fx_info(line)
            if rec:
                rows.append(rec)
        return rows

    def _extract_fx_info(self, line: str) -> Optional[Dict[str, Any]]:
        """Extract pair, price and daily change from plain text row."""
        raw = line.strip()
        if not raw:
            return None

        pair_match = re.search(r"\b([A-Z]{3})[/\-]?([A-Z]{3})\b", raw.upper())
        if not pair_match:
            return None

        pair = f"{pair_match.group(1)}/{pair_match.group(2)}"
        price_match = re.search(r"([0-9]+(?:[\.,][0-9]+)?)", raw)
        change_match = re.search(r"([+-]?\d+(?:[\.,]\d+)?)%", raw)

        last_price = self._to_float(price_match.group(1) if price_match else 0.0)
        change_percent = self._to_float(change_match.group(1) if change_match else 0.0)
        change = last_price * (change_percent / 100.0)

        return {
            "pair": pair,
            "bid": last_price,
            "ask": last_price,
            "last_price": last_price,
            "change": change,
            "change_percent": change_percent,
            "high": last_price,
            "low": last_price,
            "volume": 0.0,
            "timestamp": datetime.now().isoformat(),
        }

    def _parse_commodity_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse commodity rows from text payload."""
        text = response_text.strip()
        if not text:
            return []

        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            decoded = None

        if decoded is not None:
            rows: List[Dict[str, Any]] = []
            if isinstance(decoded, list):
                rows.extend(self._normalize_commodity_record(item) for item in decoded if isinstance(item, dict))
            elif isinstance(decoded, dict):
                candidate = decoded.get("commodities", decoded.get("assets", decoded))
                if isinstance(candidate, list):
                    rows.extend(self._normalize_commodity_record(item) for item in candidate if isinstance(item, dict))
                elif isinstance(candidate, dict):
                    rows.append(self._normalize_commodity_record(candidate))
            return [row for row in rows if row]

        rows: List[Dict[str, Any]] = []
        for line in text.splitlines():
            rec = self._extract_commodity_info(line)
            if rec:
                rows.append(rec)
        return rows

    def _extract_commodity_info(self, line: str) -> Optional[Dict[str, Any]]:
        """Extract commodity snapshot from plain text row."""
        raw = line.strip()
        if not raw:
            return None

        name_match = re.match(r"^([A-Za-z\s\-_/]+)", raw)
        if not name_match:
            return None
        name = name_match.group(1).strip().lower()
        commodity_type = self._infer_commodity_type(name)
        if commodity_type == "unknown":
            return None

        price_match = re.search(r"([0-9]+(?:[\.,][0-9]+)?)", raw)
        change_match = re.search(r"([+-]?\d+(?:[\.,]\d+)?)%", raw)
        price = self._to_float(price_match.group(1) if price_match else 0.0)
        change_percent = self._to_float(change_match.group(1) if change_match else 0.0)

        return {
            "commodity": name,
            "type": commodity_type,
            "unit": self._get_unit(name),
            "price": price,
            "change": price * (change_percent / 100.0),
            "change_percent": change_percent,
            "high": price,
            "low": price,
            "timestamp": datetime.now().isoformat(),
        }

    def _parse_history_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse CSV-like OHLC rows from text payload."""
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

    def _normalize_fx_record(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        pair = raw.get("pair") or raw.get("symbol") or raw.get("ticker") or ""
        pair = self._normalize_pair(str(pair))
        if not pair:
            return {}

        last_price = self._to_float(raw.get("last_price", raw.get("price", raw.get("last", 0.0))))
        bid = self._to_float(raw.get("bid", last_price))
        ask = self._to_float(raw.get("ask", last_price))
        change_percent = self._to_float(raw.get("change_percent", raw.get("change_pct", 0.0)))

        return {
            "pair": pair,
            "bid": bid,
            "ask": ask,
            "last_price": last_price,
            "change": self._to_float(raw.get("change", last_price * (change_percent / 100.0))),
            "change_percent": change_percent,
            "high": self._to_float(raw.get("high", last_price)),
            "low": self._to_float(raw.get("low", last_price)),
            "volume": self._to_float(raw.get("volume", 0.0)),
            "timestamp": str(raw.get("timestamp", datetime.now().isoformat())),
        }

    def _normalize_commodity_record(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        name = str(raw.get("commodity") or raw.get("name") or raw.get("asset") or "").strip().lower()
        if not name:
            return {}
        price = self._to_float(raw.get("price", raw.get("last", 0.0)))
        change_percent = self._to_float(raw.get("change_percent", raw.get("change_pct", 0.0)))

        return {
            "commodity": name,
            "type": str(raw.get("type") or self._infer_commodity_type(name)),
            "unit": str(raw.get("unit") or self._get_unit(name)),
            "price": price,
            "change": self._to_float(raw.get("change", price * (change_percent / 100.0))),
            "change_percent": change_percent,
            "high": self._to_float(raw.get("high", price)),
            "low": self._to_float(raw.get("low", price)),
            "timestamp": str(raw.get("timestamp", datetime.now().isoformat())),
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

    def _infer_commodity_type(self, commodity_name: str) -> str:
        lower = commodity_name.lower()
        if any(metal in lower for metal in ["gold", "silver", "platinum", "palladium"]):
            return "metal"
        if any(oil in lower for oil in ["oil", "crude", "wti", "brent"]):
            return "oil"
        if any(gas in lower for gas in ["gas", "lng", "natural gas"]):
            return "gas"
        if any(agri in lower for agri in ["wheat", "corn", "soy", "cotton", "coffee", "sugar", "cocoa"]):
            return "agricultural"
        return "unknown"

    def _get_unit(self, commodity_name: str) -> str:
        lower = commodity_name.lower()
        if any(metal in lower for metal in ["gold", "silver", "platinum", "palladium"]):
            return "oz"
        if any(oil in lower for oil in ["oil", "crude", "wti", "brent"]):
            return "barrel"
        if any(gas in lower for gas in ["gas", "natural gas"]):
            return "mmBtu"
        return "unit"

    def _extract_text_blocks(self, result: Dict[str, Any]) -> List[str]:
        texts: List[str] = []
        content = result.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    texts.append(item["text"])
        return texts

    def _bank_currencies_for_pairs(self, pairs: List[str]) -> List[str]:
        if not pairs:
            return ["USD", "EUR", "GBP"]

        out: List[str] = []
        seen: set[str] = set()
        for pair in pairs:
            normalized = self._normalize_pair(pair)
            if "/" not in normalized:
                continue
            base, quote = normalized.split("/", 1)
            if quote != "TRY":
                continue
            if base in {"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD"} and base not in seen:
                seen.add(base)
                out.append(base)
        return out

    @staticmethod
    def _bank_rates_to_fx_rows(currency: str, rates: pd.DataFrame) -> pd.DataFrame:
        if rates.empty:
            return pd.DataFrame()

        work = rates.copy()
        buy = pd.to_numeric(work.get("buy"), errors="coerce")
        sell = pd.to_numeric(work.get("sell"), errors="coerce")
        spread = pd.to_numeric(work.get("spread"), errors="coerce")
        valid = buy.notna() & sell.notna()
        if not valid.any():
            return pd.DataFrame()

        work = work.loc[valid].copy()
        buy = buy.loc[valid]
        sell = sell.loc[valid]
        spread = spread.loc[valid]
        mid = (buy + sell) / 2.0

        if "timestamp" in work.columns:
            timestamp_series = pd.to_datetime(work["timestamp"], errors="coerce")
        else:
            timestamp_series = pd.Series(pd.NaT, index=work.index, dtype="datetime64[ns]")
        timestamp_fallback = datetime.now().isoformat()
        timestamps = timestamp_series.fillna(pd.Timestamp(datetime.now())).astype(str)

        return pd.DataFrame(
            {
                "pair": f"{currency}/TRY",
                "bid": buy.astype(float),
                "ask": sell.astype(float),
                "last_price": mid.astype(float),
                "change": 0.0,
                "change_percent": 0.0,
                "high": sell.astype(float),
                "low": buy.astype(float),
                "volume": 0.0,
                "timestamp": timestamps if len(timestamps) else timestamp_fallback,
                "source": "bank_rates",
                "bank": work.get("bank"),
                "bank_name": work.get("bank_name"),
                "spread": spread.astype(float),
            }
        )

    def _build_bank_rate_rows(self, pairs: List[str]) -> pd.DataFrame:
        currencies = self._bank_currencies_for_pairs(pairs)
        if not currencies:
            return pd.DataFrame()

        frames: List[pd.DataFrame] = []
        for currency in currencies:
            rates = self._enhanced_provider.get_bank_rates(currency=currency)
            frame = self._bank_rates_to_fx_rows(currency=currency, rates=rates)
            if not frame.empty:
                frames.append(frame)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True, sort=False)

    def _cache_get(self, key: str) -> Optional[pd.DataFrame]:
        if self._disk_cache is not None:
            _disk = self._disk_cache.get_dataframe("commodities", key)
            if _disk is not None:
                return _disk
        item = self._cache.get(key)
        if not item:
            return None
        cached_df, timestamp = item
        if datetime.now().timestamp() - timestamp > self._cache_ttl:
            self._cache.pop(key, None)
            return None
        logger.info("Returning cached FX/commodity data for %s", key)
        return cached_df.copy()

    def _cache_set(self, key: str, value: pd.DataFrame) -> None:
        self._cache[key] = (value.copy(), datetime.now().timestamp())
        if self._disk_cache is not None and not value.empty:
            self._disk_cache.set_dataframe("commodities", key, value)

    @staticmethod
    def _normalize_pair(pair: str) -> str:
        cleaned = pair.strip().upper().replace("-", "")
        if "/" in cleaned:
            base, quote = cleaned.split("/", 1)
            if len(base) == 3 and len(quote) == 3:
                return f"{base}/{quote}"
        if len(cleaned) == 6:
            return f"{cleaned[:3]}/{cleaned[3:]}"
        return pair.strip().upper()

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
