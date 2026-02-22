"""
BIST Quant Data Pipeline.

ETL pipeline for fetching, normalizing, and storing market data.
"""

from __future__ import annotations

from .errors import (
    CircuitBreakerOpenError,
    FetchError,
    FreshnessGateError,
    FundamentalsPipelineError,
    MergeError,
    ProvenanceError,
    SchemaValidationError,
)
from .fetcher import FundamentalsFetcher
from .merge import merge_consolidated_panels
from .normalize import build_consolidated_panel, build_flat_normalized
from .pipeline import (
    FundamentalsPipeline,
    PipelineRunResult,
    build_default_config,
    build_default_paths,
    compute_default_periods,
)
from .schemas import validate_consolidated_panel, validate_flat_normalized
from .types import FreshnessThresholds, PipelineConfig, PipelinePaths, RawDataBundle

# Convenience aliases aligned with broader package naming.
Pipeline = FundamentalsPipeline
Fetcher = FundamentalsFetcher
normalize_prices = build_consolidated_panel
normalize_fundamentals = build_flat_normalized
merge_data = merge_consolidated_panels
DataPipelineError = FundamentalsPipelineError
ValidationError = SchemaValidationError

__all__ = [
    "Pipeline",
    "Fetcher",
    "normalize_prices",
    "normalize_fundamentals",
    "merge_data",
    "FundamentalsPipeline",
    "FundamentalsFetcher",
    "PipelineRunResult",
    "PipelineConfig",
    "PipelinePaths",
    "FreshnessThresholds",
    "RawDataBundle",
    "build_default_config",
    "build_default_paths",
    "compute_default_periods",
    "validate_consolidated_panel",
    "validate_flat_normalized",
    "DataPipelineError",
    "ValidationError",
    "FetchError",
    "MergeError",
    "FreshnessGateError",
    "ProvenanceError",
    "CircuitBreakerOpenError",
    "SchemaValidationError",
]
