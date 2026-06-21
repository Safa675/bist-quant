"""Screener field definitions, templates, and universe options."""

from __future__ import annotations

from typing import Any

FILTER_FIELD_DEFS: list[dict[str, Any]] = [
    {"key": "market_cap_usd", "label": "Market Cap (USD mn)", "group": "valuation"},
    {"key": "market_cap", "label": "Market Cap (TL mn)", "group": "valuation"},
    {"key": "pe", "label": "P/E", "group": "valuation"},
    {"key": "forward_pe", "label": "Forward P/E", "group": "valuation"},
    {"key": "pb", "label": "P/B", "group": "valuation"},
    {"key": "ev_ebitda", "label": "EV/EBITDA", "group": "valuation"},
    {"key": "ev_sales", "label": "EV/Sales", "group": "valuation"},
    {"key": "dividend_yield", "label": "Dividend Yield (%)", "group": "income"},
    {"key": "upside_potential", "label": "Upside Potential (%)", "group": "analyst"},
    {"key": "analyst_target_price", "label": "Analyst Target Price", "group": "analyst"},
    {"key": "roe", "label": "ROE (%)", "group": "quality"},
    {"key": "roa", "label": "ROA (%)", "group": "quality"},
    {"key": "net_margin", "label": "Net Margin (%)", "group": "quality"},
    {"key": "ebitda_margin", "label": "EBITDA Margin (%)", "group": "quality"},
    {"key": "foreign_ratio", "label": "Foreign Ownership (%)", "group": "flow"},
    {"key": "foreign_change_1w", "label": "Foreign Ownership 1W Change (pp)", "group": "flow"},
    {"key": "foreign_change_1m", "label": "Foreign Ownership 1M Change (pp)", "group": "flow"},
    {"key": "float_ratio", "label": "Free Float (%)", "group": "flow"},
    {"key": "volume_3m", "label": "Avg Volume 3M (mn)", "group": "liquidity"},
    {"key": "volume_12m", "label": "Avg Volume 12M (mn)", "group": "liquidity"},
    {"key": "return_1w", "label": "Return 1W (%)", "group": "momentum"},
    {"key": "return_1m", "label": "Return 1M (%)", "group": "momentum"},
    {"key": "return_1y", "label": "Return 1Y (%)", "group": "momentum"},
    {"key": "return_ytd", "label": "Return YTD (%)", "group": "momentum"},
    {"key": "rsi_14", "label": "RSI 14", "group": "technical"},
    {"key": "macd_hist", "label": "MACD Histogram", "group": "technical"},
    {"key": "atr_14_pct", "label": "ATR 14 (% of Price)", "group": "technical"},
    {"key": "revenue_growth_yoy", "label": "Revenue Growth YoY (%)", "group": "growth"},
    {"key": "net_income_growth_yoy", "label": "Net Income Growth YoY (%)", "group": "growth"},
]

FIELD_LABELS = {row["key"]: row["label"] for row in FILTER_FIELD_DEFS}

DISPLAY_COLUMNS_DEFAULT: list[str] = [
    "symbol",
    "name",
    "market_cap_usd",
    "pe",
    "forward_pe",
    "pb",
    "dividend_yield",
    "upside_potential",
    "roe",
    "net_margin",
    "rsi_14",
    "return_1m",
    "recommendation",
]

INDEX_OPTIONS = ["XU030", "XU050", "XU100", "XUTUM", "CUSTOM"]
RECOMMENDATION_OPTIONS = ["AL", "TUT", "SAT"]

DEFAULT_TEMPLATES = [
    "small_cap",
    "mid_cap",
    "large_cap",
    "high_dividend",
    "high_upside",
    "low_upside",
    "high_volume",
    "low_volume",
    "buy_recommendation",
    "sell_recommendation",
    "high_net_margin",
    "high_return",
    "low_pe",
    "high_roe",
    "high_foreign_ownership",
]

DATA_SOURCE_OPTIONS = ["local", "isyatirim", "hybrid"]

TEMPLATE_PRESETS: dict[str, dict[str, Any]] = {
    "small_cap": {"market_cap_usd": {"max": 1_000.0}},
    "mid_cap": {"market_cap_usd": {"min": 1_000.0, "max": 10_000.0}},
    "large_cap": {"market_cap_usd": {"min": 10_000.0}},
    "high_dividend": {"dividend_yield": {"min": 4.0}},
    "high_upside": {"upside_potential": {"min": 15.0}},
    "low_upside": {"upside_potential": {"max": 5.0}},
    "high_volume": {"volume_3m": {"min": 20.0}},
    "low_volume": {"volume_3m": {"max": 2.0}},
    "buy_recommendation": {"recommendation": "AL"},
    "sell_recommendation": {"recommendation": "SAT"},
    "high_net_margin": {"net_margin": {"min": 10.0}},
    "high_return": {"return_1y": {"min": 20.0}},
    "low_pe": {"pe": {"max": 12.0}},
    "high_roe": {"roe": {"min": 15.0}},
    "high_foreign_ownership": {"foreign_ratio": {"min": 50.0}},
}
