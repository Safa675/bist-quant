from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class FreshnessThresholds:
    """Freshness rules used to block stale fundamentals."""

    max_median_staleness_days: int = 120
    max_pct_over_120_days: float = 0.90
    min_q4_coverage_pct: float = 0.10
    max_max_staleness_days: int = 500
    grace_days: int = 0


@dataclass(frozen=True)
class PipelinePaths:
    """Filesystem paths used by the unified fundamentals pipeline."""

    base_dir: Path
    data_dir: Path
    fundamentals_dir: Path
    raw_dir: Path
    normalized_json_dir: Path
    log_dir: Path
    consolidated_parquet: Path
    normalized_parquet: Path
    normalized_csv: Path
    staleness_weights_json: Path
    freshness_report_csv: Path
    quality_metrics_json: Path
    provenance_dir: Path
    audit_log_jsonl: Path
    alerts_log_jsonl: Path
    cache_state_json: Path


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime config for the fundamentals pipeline."""

    periods: tuple[tuple[int, int], ...]
    request_delay_seconds: float = 0.35
    retry_base_seconds: float = 2.0
    max_retries: int = 3
    enforce_freshness_gate: bool = True
    allow_stale_override: bool = False
    circuit_breaker_failures: int = 12
    circuit_breaker_timeout_seconds: int = 120
    prefer_existing_values: bool = True
    schema_version: str = "v1"
    pipeline_version: str = "phase4"


@dataclass
class RawDataBundle:
    """Raw fetch payloads and fetch failures."""

    raw_by_ticker: dict[str, dict[str, Any]]
    errors: list[dict[str, Any]]
    source_name: str
    fetched_at: datetime


@dataclass
class ValidatedDataBundle:
    """Schema-validated raw payloads."""

    raw_bundle: RawDataBundle


@dataclass
class NormalizedDataBundle:
    """Normalized datasets used for merge + reporting."""

    consolidated_like: pd.DataFrame
    flat_normalized: pd.DataFrame
    raw_bundle: RawDataBundle


@dataclass
class MergedDataBundle:
    """Final merged data plus quality/freshness diagnostics."""

    merged_consolidated: pd.DataFrame
    staleness_report: pd.DataFrame
    quality_metrics: dict[str, Any]
    merge_stats: dict[str, Any]
    input_fingerprint: str
    warnings: list[str] = field(default_factory=list)
