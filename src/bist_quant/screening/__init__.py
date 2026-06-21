"""Multi-dimensional BIST stock screener."""

from bist_quant.screening.errors import (
    ScreeningDataError,
    ScreeningError,
    ScreeningExecutionError,
    ScreeningValidationError,
)
from bist_quant.screening.presets import (
    DATA_SOURCE_OPTIONS,
    DEFAULT_TEMPLATES,
    FILTER_FIELD_DEFS,
    INDEX_OPTIONS,
    RECOMMENDATION_OPTIONS,
    TEMPLATE_PRESETS,
)
from bist_quant.screening.screener import (
    get_screener_metadata,
    get_stock_filter_metadata,
    run_screener,
    run_stock_filter,
)
from bist_quant.screening.types import StockScreenerResult

__all__ = [
    "DATA_SOURCE_OPTIONS",
    "DEFAULT_TEMPLATES",
    "FILTER_FIELD_DEFS",
    "INDEX_OPTIONS",
    "RECOMMENDATION_OPTIONS",
    "TEMPLATE_PRESETS",
    "ScreeningDataError",
    "ScreeningError",
    "ScreeningExecutionError",
    "ScreeningValidationError",
    "StockScreenerResult",
    "get_screener_metadata",
    "get_stock_filter_metadata",
    "run_screener",
    "run_stock_filter",
]
