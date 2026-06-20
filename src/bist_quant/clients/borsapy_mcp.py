"""Borsa-MCP fallback layer for borsapy operations.

Provides MCP (Model Context Protocol) client helpers for fetching BIST
data via an external HTTP endpoint when the native borsapy library
encounters SSL or data issues.  Used internally by
:class:`bist_quant.clients.borsapy_client.BorsapyClient`.

The module is split out from ``borsapy_client.py`` so that the MCP
transport/parsing logic can evolve independently of the main client
class.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import uuid
from typing import Any

import httpx
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MCP JSON-RPC / SSE response parsing
# ---------------------------------------------------------------------------

def extract_mcp_text(content: Any) -> str:
    """Normalise various MCP *content* shapes into a plain string."""
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


def strip_code_fence(text: str) -> str:
    """Remove ```` ```json ```` fences around a payload string."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def parse_content_text(text: str) -> Any:
    """Try to deserialise MCP text as JSON / Python literal."""
    cleaned = strip_code_fence(text)
    if not cleaned:
        return None

    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(cleaned)
        except Exception:
            continue
    return None


def extract_payload_from_result(result: dict[str, Any]) -> dict[str, Any]:
    """Merge *structuredContent* and text-parsed payloads from an MCP result."""
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

    text = extract_mcp_text(result.get("content"))
    parsed = parse_content_text(text) if text else None
    if isinstance(parsed, dict):
        for key, value in parsed.items():
            payload.setdefault(key, value)
    elif isinstance(parsed, list):
        payload.setdefault("data", parsed)

    return payload


def first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Return the first value found for *keys* in *mapping*."""
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


# ---------------------------------------------------------------------------
# MCP JSON-RPC transport
# ---------------------------------------------------------------------------

def parse_mcp_jsonrpc_response(response: httpx.Response) -> dict[str, Any]:
    """Parse an MCP JSON-RPC response, handling both JSON and SSE."""
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


# ---------------------------------------------------------------------------
# High-level MCP fallback helper (mixed into BorsapyClient)
# ---------------------------------------------------------------------------

class MCPFallbackClient:
    """Stateless helper that executes Borsa-MCP tool calls.

    Designed to be *composed into* ``BorsapyClient`` rather than inherited,
    keeping the MCP transport layer independent of caching and config.
    """

    def __init__(self, mcp_endpoint: str, session: httpx.Client) -> None:
        self._mcp_endpoint = mcp_endpoint
        self._session = session

    # ---- tool invocation ---------------------------------------------------

    def call_tool(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Build and send a JSON-RPC ``tools/call`` request."""
        payload: dict[str, Any] = {
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

            rpc_payload = parse_mcp_jsonrpc_response(response)
            if "error" in rpc_payload:
                error = rpc_payload.get("error")
                if isinstance(error, dict):
                    message = str(error.get("message", error))
                else:
                    message = str(error)
                raise RuntimeError(message)

            result = rpc_payload.get("result", {})
            if isinstance(result, dict) and result.get("isError") is True:
                message = extract_mcp_text(result.get("content")) or "MCP tool returned isError=true."
                raise RuntimeError(message)

            if isinstance(result, dict):
                return result
            if result is None:
                return {}
            return {"data": result}
        except httpx.RequestError as exc:
            logger.error("Request error calling MCP tool %s: %s", tool_name, exc)
            raise
        except Exception as exc:
            logger.error("Error calling MCP tool %s: %s", tool_name, exc)
            raise

    # ---- domain-specific fallbacks -----------------------------------------

    def screen_securities(
        self,
        template: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Screen securities via MCP ``screen_securities`` tool."""
        params: dict[str, Any] = {}
        if template:
            params["preset"] = template
            params["template"] = template
        if filters:
            params.update(filters)

        try:
            response = self.call_tool("screen_securities", params)
            payload = extract_payload_from_result(response)
            rows = first_present(
                payload,
                ("data", "results", "securities", "stocks", "items", "matches"),
            )
            return coerce_dataframe(rows)
        except Exception as exc:
            logger.error("MCP screening fallback failed: %s", exc)
            return pd.DataFrame()

    def get_financial_statements(self, symbol: str) -> dict[str, pd.DataFrame]:
        """Fetch financial statements via MCP ``get_financial_statements`` tool."""
        empty = {
            "balance_sheet": pd.DataFrame(),
            "income_stmt": pd.DataFrame(),
            "cash_flow": pd.DataFrame(),
        }

        try:
            response = self.call_tool(
                "get_financial_statements",
                {
                    "symbols": [symbol],
                    "market": "bist",
                },
            )
            payload = extract_payload_from_result(response)

            data_payload = payload.get("data")
            if isinstance(data_payload, dict):
                for key, value in data_payload.items():
                    payload.setdefault(key, value)

            balance_sheet = coerce_dataframe(
                first_present(payload, ("balance_sheet", "balanceSheet", "balance"))
            )
            income_stmt = coerce_dataframe(
                first_present(payload, ("income_stmt", "income_statement", "incomeStatement"))
            )
            cash_flow = coerce_dataframe(
                first_present(payload, ("cash_flow", "cashflow", "cashFlow"))
            )

            return {
                "balance_sheet": balance_sheet,
                "income_stmt": income_stmt,
                "cash_flow": cash_flow,
            }
        except Exception as exc:
            logger.error("MCP financial statements fallback failed for %s: %s", symbol, exc)
            return empty

    def get_financial_ratios(self, symbol: str) -> pd.DataFrame:
        """Fetch financial ratios via MCP ``get_financial_ratios`` tool."""
        try:
            response = self.call_tool(
                "get_financial_ratios",
                {
                    "symbols": [symbol],
                    "market": "bist",
                },
            )
            payload = extract_payload_from_result(response)

            ratios = first_present(payload, ("ratios", "financial_ratios", "data"))
            if isinstance(ratios, dict) and "ratios" in ratios:
                ratios = ratios.get("ratios")
            return coerce_dataframe(ratios)
        except Exception as exc:
            logger.error("MCP financial ratios fallback failed for %s: %s", symbol, exc)
            return pd.DataFrame()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def coerce_dataframe(value: Any) -> pd.DataFrame:
    """Coerce *value* to a :class:`pd.DataFrame` when possible."""
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, list) and value:
        return pd.DataFrame(value)
    if isinstance(value, dict) and value:
        try:
            return pd.DataFrame(value)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()
