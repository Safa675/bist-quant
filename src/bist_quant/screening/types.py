from __future__ import annotations

from typing import Any, TypedDict


class StockScreenerResult(TypedDict, total=False):
    meta: dict[str, Any]
    columns: list[dict[str, Any]]
    rows: list[dict[str, Any]]
    applied_filters: list[dict[str, Any]]
    applied_percentile_filters: list[dict[str, Any]]
    chart: dict[str, Any]
