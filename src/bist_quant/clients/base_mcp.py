"""Base class for Borsa MCP JSON-RPC clients.

Handles common tasks such as HTTP requests, standard caching (in-memory + disk),
error boundaries, and utilities.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List
import httpx
import pandas as pd

from bist_quant.clients.utils import to_float

logger = logging.getLogger(__name__)


class BaseMCPClient:
    """Base class for Borsa MCP JSON-RPC endpoints."""

    def __init__(
        self,
        mcp_endpoint: str = "https://borsamcp.fastmcp.app/mcp",
        cache_ttl: int = 3600,
        cache_dir: Optional[Path | str] = None,
        disk_cache_category: str = "mcp",
    ) -> None:
        self._mcp_endpoint = mcp_endpoint
        self._cache_ttl = cache_ttl
        self._cache: Dict[str, tuple[pd.DataFrame, float]] = {}
        self._disk_cache_category = disk_cache_category

        self._disk_cache: Any | None = None
        if cache_dir is not None:
            try:
                from bist_quant.common.disk_cache import DiskCache
                self._disk_cache = DiskCache(Path(cache_dir))
            except Exception:
                pass

        self._session = httpx.AsyncClient(timeout=30.0)

    async def _call_mcp_async(self, tool: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call Borsa MCP endpoint asynchronously using JSON-RPC."""
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

    def _cache_get(self, key: str) -> Optional[pd.DataFrame]:
        if self._disk_cache is not None:
            _disk = self._disk_cache.get_dataframe(self._disk_cache_category, key)
            if _disk is not None:
                return _disk
        item = self._cache.get(key)
        if not item:
            return None
        cached_df, timestamp = item
        if datetime.now().timestamp() - timestamp > self._cache_ttl:
            self._cache.pop(key, None)
            return None
        logger.info("Returning cached data for %s", key)
        return cached_df.copy()

    def _cache_set(self, key: str, value: pd.DataFrame) -> None:
        self._cache[key] = (value.copy(), datetime.now().timestamp())
        if self._disk_cache is not None and not value.empty:
            self._disk_cache.set_dataframe(self._disk_cache_category, key, value)

    @staticmethod
    def _extract_text_blocks(result: Dict[str, Any]) -> List[str]:
        texts: List[str] = []
        content = result.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    texts.append(item["text"])
        return texts

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        parsed = to_float(value)
        return parsed if parsed is not None else default
