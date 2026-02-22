from __future__ import annotations

from typing import Any, TypedDict


class FactorCatalogResult(TypedDict):
    factors: list[dict[str, Any]]
    default_portfolio_options: dict[str, Any]


class SignalConstructionSnapshotResult(TypedDict):
    meta: dict[str, Any]
    indicator_summaries: list[dict[str, Any]]
    signals: list[dict[str, Any]]


class SignalBacktestResult(TypedDict):
    meta: dict[str, Any]
    metrics: dict[str, Any]
    signals: list[dict[str, Any]]
    indicator_summaries: list[dict[str, Any]]
    current_holdings: list[str]
    equity_curve: list[dict[str, Any]]
    benchmark_curve: list[dict[str, Any]]
    price_overlay: dict[str, Any]
    validation: dict[str, Any]
    analytics_v2: dict[str, Any]


class StockScreenerResult(TypedDict):
    meta: dict[str, Any]
    columns: list[dict[str, Any]]
    rows: list[dict[str, Any]]
    applied_filters: list[dict[str, Any]]
    applied_percentile_filters: list[dict[str, Any]]
    chart: dict[str, Any]
