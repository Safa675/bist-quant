"""TEFAS fund analytics via Borsa MCP."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd

logger = logging.getLogger(__name__)


class FundCategory(Enum):
    EQUITY = "equity"
    BOND = "bond"
    MONEY_MARKET = "money_market"
    MIXED = "mixed"
    PENSION = "pension"
    ISLAMIC = "islamic"


@dataclass
class FundMetrics:
    fund_code: str
    name: str
    category: str
    nav: float
    nav_date: str
    return_1d: float
    return_1w: float
    return_1m: float
    return_3m: float
    return_6m: float
    return_1y: float
    return_ytd: float
    expense_ratio: float
    aum: float
    aum_date: str
    manager: str
    inception_date: str
    min_investment: float
    is_liquid: bool
    is_islamic: bool


class TEFASAnalyzer:
    """Analyze Turkish investment funds via Borsa MCP."""

    def __init__(
        self,
        mcp_endpoint: str = "https://borsamcp.fastmcp.app/mcp",
        timeout: float = 15.0,
        cache_ttl: int = 300,
        cache_dir: Any = None,
    ):
        self._mcp_endpoint = mcp_endpoint
        self._session = httpx.AsyncClient(timeout=timeout)
        self._cache: Dict[str, tuple[pd.DataFrame, float]] = {}
        self._cache_ttl = cache_ttl
        self._disk_cache: Any | None = None
        if cache_dir is not None:
            try:
                from pathlib import Path as _Path
                from bist_quant.common.disk_cache import DiskCache as _DiskCache
                self._disk_cache = _DiskCache(_Path(cache_dir))
            except Exception:
                pass

    async def get_fund_data(
        self,
        fund_code: Optional[str] = None,
        category: Optional[FundCategory] = None,
        sort_by: str = "return_1y",
        ascending: bool = False,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch TEFAS fund data with light in-memory caching."""
        cache_key = f"fund_data_{fund_code}_{category}_{sort_by}_{ascending}_{limit}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        params: Dict[str, Any] = {"sort_by": sort_by, "ascending": ascending}
        if fund_code:
            params["fund_code"] = fund_code.upper()
        if category:
            params["category"] = category.value
        if limit is not None:
            params["limit"] = int(limit)

        result = await self._call_mcp_async("get_fund_data", params)
        records = self._extract_fund_records(result)
        df = pd.DataFrame(records)

        if not df.empty and sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=ascending)
        if limit is not None and not df.empty:
            df = df.head(limit)

        self._cache_set(cache_key, df)
        return df

    async def compare_funds(self, fund_codes: List[str], period: str = "1y") -> pd.DataFrame:
        """Compare multiple funds side-by-side."""
        if not fund_codes:
            return pd.DataFrame()

        calls = [self.get_fund_data(fund_code=code) for code in fund_codes]
        results = await asyncio.gather(*calls, return_exceptions=True)

        rows: List[pd.Series] = []
        for code, item in zip(fund_codes, results):
            if isinstance(item, Exception):
                logger.warning("Could not fetch data for fund %s: %s", code, item)
                continue
            if isinstance(item, pd.DataFrame) and not item.empty:
                rows.append(item.iloc[0])

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        period_col = f"return_{period}"
        if period_col in df.columns:
            df = df.sort_values(period_col, ascending=False)
        return df.reset_index(drop=True)

    async def get_top_performers(
        self,
        category: FundCategory,
        period: str = "1y",
        limit: int = 10,
    ) -> pd.DataFrame:
        """Get top performing funds in a category."""
        sort_key = f"return_{period}"
        df = await self.get_fund_data(
            category=category,
            sort_by=sort_key,
            ascending=False,
            limit=limit,
        )
        return df.head(limit)

    async def get_fund_allocation(self, fund_code: str) -> pd.DataFrame:
        """Get fund allocation breakdown."""
        result = await self._call_mcp_async(
            "get_fund_data",
            {"fund_code": fund_code.upper(), "allocation": True},
        )

        allocations: List[Dict[str, Any]] = []
        data_block = result.get("data")
        if isinstance(data_block, dict):
            if isinstance(data_block.get("allocation"), list):
                for item in data_block["allocation"]:
                    if isinstance(item, dict):
                        allocations.append(
                            {
                                "asset_type": str(item.get("asset_type", item.get("name", "unknown"))),
                                "percentage": self._to_float(item.get("percentage", item.get("weight", 0))),
                            }
                        )
        elif isinstance(data_block, list):
            for item in data_block:
                if isinstance(item, dict) and "percentage" in item:
                    allocations.append(
                        {
                            "asset_type": str(item.get("asset_type", item.get("name", "unknown"))),
                            "percentage": self._to_float(item.get("percentage", item.get("weight", 0))),
                        }
                    )

        if not allocations:
            for text in self._extract_text_blocks(result):
                allocations.extend(self._parse_allocation_response(text))

        return pd.DataFrame(allocations)

    async def get_fund_risk_metrics(self, fund_code: str) -> Dict[str, float]:
        """Calculate simple risk metrics for a fund using NAV history."""
        hist_data = await self.get_fund_history(fund_code, period="1y")
        if hist_data.empty or "nav" not in hist_data.columns:
            return {}

        returns = pd.to_numeric(hist_data["nav"], errors="coerce").pct_change().dropna()
        if returns.empty:
            return {}

        std = returns.std()
        metrics = {
            "volatility": float(std * (252**0.5)),
            "sharpe_ratio": float((returns.mean() / std) if std != 0 else 0.0),
            "max_drawdown": float(self._calculate_max_drawdown(pd.to_numeric(hist_data["nav"], errors="coerce"))),
            "beta": float(await self._calculate_beta(fund_code, returns)),
            "alpha": float(await self._calculate_alpha(fund_code, returns)),
        }
        return metrics

    async def get_fund_history(self, fund_code: str, period: str = "1y") -> pd.DataFrame:
        """Get fund NAV history if available through MCP."""
        result = await self._call_mcp_async(
            "get_fund_data",
            {"fund_code": fund_code.upper(), "period": period, "history": True},
        )

        rows: List[Dict[str, Any]] = []
        data_block = result.get("data")
        if isinstance(data_block, dict):
            history = data_block.get("history")
            if isinstance(history, list):
                for item in history:
                    if not isinstance(item, dict):
                        continue
                    rows.append(
                        {
                            "date": item.get("date"),
                            "nav": self._to_float(item.get("nav", item.get("close"))),
                        }
                    )
        elif isinstance(data_block, list):
            for item in data_block:
                if isinstance(item, dict) and "date" in item:
                    rows.append(
                        {
                            "date": item.get("date"),
                            "nav": self._to_float(item.get("nav", item.get("close"))),
                        }
                    )

        if not rows:
            for text in self._extract_text_blocks(result):
                rows.extend(self._parse_history_response(text))

        df = pd.DataFrame(rows)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
            df = df.dropna(subset=["date", "nav"]).sort_values("date").reset_index(drop=True)
        return df

    def _parse_fund_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse fund data from MCP response text."""
        text = response_text.strip()
        if not text:
            return []

        # Prefer JSON text payload when provided.
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            decoded = None

        if decoded is not None:
            records: List[Dict[str, Any]] = []
            if isinstance(decoded, dict):
                candidate = decoded.get("funds", decoded.get("data", decoded))
                if isinstance(candidate, list):
                    records.extend(self._normalize_record(item) for item in candidate if isinstance(item, dict))
                elif isinstance(candidate, dict):
                    records.append(self._normalize_record(candidate))
            elif isinstance(decoded, list):
                records.extend(self._normalize_record(item) for item in decoded if isinstance(item, dict))
            return [record for record in records if record]

        rows: List[Dict[str, Any]] = []
        current: Dict[str, Any] = {}

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                if current:
                    rows.append(self._normalize_record(current))
                    current = {}
                continue

            code_match = re.match(r"^([A-Z0-9]{3,5})\b(?:\s*[-:]\s*(.*))?$", line)
            if code_match and code_match.group(1):
                if current:
                    rows.append(self._normalize_record(current))
                current = {
                    "fund_code": code_match.group(1),
                    "name": (code_match.group(2) or code_match.group(1)).strip(),
                }
                continue

            if ":" in line:
                key, value = line.split(":", 1)
                current[key.strip()] = value.strip()

        if current:
            rows.append(self._normalize_record(current))

        return [row for row in rows if row]

    def _parse_allocation_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse fund allocation from text response."""
        allocations: List[Dict[str, Any]] = []
        pattern = re.compile(r"([A-Za-z\s\-_/]+?)\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)%")

        for line in response_text.splitlines():
            match = pattern.search(line.strip())
            if not match:
                continue
            allocations.append(
                {
                    "asset_type": match.group(1).strip().lower(),
                    "percentage": float(match.group(2)),
                }
            )
        return allocations

    def _parse_history_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse NAV history from text response."""
        rows: List[Dict[str, Any]] = []
        for line in response_text.splitlines():
            cleaned = line.strip()
            if not cleaned or "," not in cleaned:
                continue
            parts = [item.strip() for item in cleaned.split(",")]
            if len(parts) < 2:
                continue
            rows.append({"date": parts[0], "nav": self._to_float(parts[1])})
        return rows

    def _calculate_max_drawdown(self, nav_series: pd.Series) -> float:
        """Calculate maximum drawdown from a NAV series."""
        if nav_series.empty:
            return 0.0
        rolling_max = nav_series.expanding().max()
        drawdown = (nav_series - rolling_max) / rolling_max
        return float(drawdown.min())

    async def _calculate_beta(self, fund_code: str, fund_returns: pd.Series) -> float:
        """Placeholder beta calculation."""
        _ = fund_code, fund_returns
        return 1.0

    async def _calculate_alpha(self, fund_code: str, fund_returns: pd.Series) -> float:
        """Placeholder alpha calculation."""
        _ = fund_code, fund_returns
        return 0.0

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
        """Close the HTTP session."""
        await self._session.aclose()

    def _cache_get(self, key: str) -> Optional[pd.DataFrame]:
        if self._disk_cache is not None:
            _disk = self._disk_cache.get_dataframe("funds", key)
            if _disk is not None:
                return _disk
        item = self._cache.get(key)
        if not item:
            return None
        cached_df, timestamp = item
        if datetime.now().timestamp() - timestamp > self._cache_ttl:
            self._cache.pop(key, None)
            return None
        logger.info("Returning cached fund data for %s", key)
        return cached_df.copy()

    def _cache_set(self, key: str, value: pd.DataFrame) -> None:
        self._cache[key] = (value.copy(), datetime.now().timestamp())
        if self._disk_cache is not None and not value.empty:
            self._disk_cache.set_dataframe("funds", key, value)

    def _extract_text_blocks(self, result: Dict[str, Any]) -> List[str]:
        texts: List[str] = []
        content = result.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    texts.append(item["text"])
        return texts

    def _extract_fund_records(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []

        data_block = result.get("data")
        if isinstance(data_block, dict):
            candidate = data_block.get("funds", data_block.get("items", data_block))
            if isinstance(candidate, list):
                records.extend(self._normalize_record(item) for item in candidate if isinstance(item, dict))
            elif isinstance(candidate, dict):
                records.append(self._normalize_record(candidate))
        elif isinstance(data_block, list):
            records.extend(self._normalize_record(item) for item in data_block if isinstance(item, dict))

        if records:
            return [row for row in records if row]

        for text in self._extract_text_blocks(result):
            records.extend(self._parse_fund_response(text))

        return [row for row in records if row]

    def _normalize_record(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize field names from heterogeneous providers."""
        if not isinstance(raw, dict):
            return {}

        def pick(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in raw and raw[key] is not None:
                    return raw[key]
            return default

        record = {
            "fund_code": str(pick("fund_code", "code", "symbol", "ticker", default="")).upper(),
            "name": str(pick("name", "fund_name", "title", default="")).strip(),
            "category": str(pick("category", "type", default="")).strip().lower(),
            "nav": self._to_float(pick("nav", "price", "close", default=0.0)),
            "nav_date": str(pick("nav_date", "date", default="")),
            "return_1d": self._to_float(pick("return_1d", "daily_return", default=0.0)),
            "return_1w": self._to_float(pick("return_1w", "weekly_return", default=0.0)),
            "return_1m": self._to_float(pick("return_1m", "monthly_return", default=0.0)),
            "return_3m": self._to_float(pick("return_3m", default=0.0)),
            "return_6m": self._to_float(pick("return_6m", default=0.0)),
            "return_1y": self._to_float(pick("return_1y", "annual_return", default=0.0)),
            "return_ytd": self._to_float(pick("return_ytd", "ytd_return", default=0.0)),
            "expense_ratio": self._to_float(pick("expense_ratio", "expense", default=0.0)),
            "aum": self._to_float(pick("aum", "assets_under_management", default=0.0)),
            "aum_date": str(pick("aum_date", default="")),
            "manager": str(pick("manager", "portfolio_manager", default="")).strip(),
            "inception_date": str(pick("inception_date", default="")),
            "min_investment": self._to_float(pick("min_investment", default=0.0)),
            "is_liquid": self._to_bool(pick("is_liquid", default=False)),
            "is_islamic": self._to_bool(pick("is_islamic", default=False)),
        }

        if not record["fund_code"] and record["name"]:
            m = re.match(r"^([A-Z0-9]{3,5})\b", record["name"].upper())
            if m:
                record["fund_code"] = m.group(1)

        if not record["fund_code"]:
            return {}

        if not record["name"]:
            record["name"] = record["fund_code"]

        return record

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

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "yes", "y", "evet"}


# Backward-compatible alias for callers expecting FundAnalyzer.
FundAnalyzer = TEFASAnalyzer
