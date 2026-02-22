"""
Cache TTL Configuration for Borsapy Integration.

Centralizes TTL defaults with environment variable overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass
class CacheTTL:
    """Per-category TTL settings (in seconds).

    Defaults can be overridden via ``BIST_CACHE_TTL_*`` environment variables.

    Attributes:
        prices: OHLCV price data.  Default 4 hours.
        index_components: Index constituent lists.  Default 24 hours.
        financials: Balance sheet / income / cash-flow.  Default 7 days.
        financial_ratios: Financial ratios.  Default 7 days.
        fast_info: Current quote / market snapshot.  Default 15 minutes.
        dividends: Dividend history.  Default 7 days.
        news: News / KAP announcements.  Default 1 hour.
    """

    prices: int = 14400           # 4 hours
    index_components: int = 86400  # 24 hours
    financials: int = 604800       # 7 days
    financial_ratios: int = 604800 # 7 days
    fast_info: int = 900           # 15 minutes
    dividends: int = 604800        # 7 days
    news: int = 3600               # 1 hour

    @classmethod
    def from_env(cls) -> CacheTTL:
        """Build a ``CacheTTL`` instance, applying environment overrides."""
        return cls(
            prices=_env_int("BIST_CACHE_TTL_PRICES", cls.prices),
            index_components=_env_int("BIST_CACHE_TTL_INDEX_COMPONENTS", cls.index_components),
            financials=_env_int("BIST_CACHE_TTL_FINANCIALS", cls.financials),
            financial_ratios=_env_int("BIST_CACHE_TTL_FINANCIAL_RATIOS", cls.financial_ratios),
            fast_info=_env_int("BIST_CACHE_TTL_FAST_INFO", cls.fast_info),
            dividends=_env_int("BIST_CACHE_TTL_DIVIDENDS", cls.dividends),
            news=_env_int("BIST_CACHE_TTL_NEWS", cls.news),
        )

    def ttl_for(self, category: str) -> int:
        """Return TTL seconds for a named category, falling back to ``prices``."""
        return getattr(self, category, self.prices)
