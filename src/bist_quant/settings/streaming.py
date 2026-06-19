"""Library-side streaming auth configuration (TradingView)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .environment import parse_env_int, parse_env_str


def load_streaming_auth_config(
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    """Load TradingView streaming auth from environment variables."""
    token = parse_env_str("BIST_API_TRADINGVIEW_AUTH_TOKEN", "", environ=environ)
    if not token:
        token = parse_env_str("BIST_TRADINGVIEW_AUTH_TOKEN", "", environ=environ)
    if not token:
        token = parse_env_str("TRADINGVIEW_AUTH_TOKEN", "", environ=environ)
    if not token:
        return None

    timeout = parse_env_int(
        "BIST_API_TRADINGVIEW_CONNECT_TIMEOUT_SECONDS",
        10,
        1,
        120,
        environ=environ,
    )
    return {"auth_token": token, "connect_timeout": float(timeout)}
