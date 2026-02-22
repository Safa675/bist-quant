from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from bist_quant.data_pipeline.fetcher import FundamentalsFetcher
from bist_quant.data_pipeline.freshness import (
    compute_staleness_report,
    enforce_freshness_gate,
    evaluate_freshness,
    summarize_quality_metrics,
)
from bist_quant.data_pipeline.logging_utils import configure_structured_logger, log_event
from bist_quant.data_pipeline.merge import merge_consolidated_panels
from bist_quant.data_pipeline.normalize import (
    build_consolidated_panel,
    build_flat_normalized,
    build_staleness_weights,
    save_normalized_per_ticker_json,
)
from bist_quant.data_pipeline.provenance import (
    file_checksum_sha256,
    stable_json_sha256,
    write_dataset_provenance,
)
from bist_quant.data_pipeline.schemas import (
    validate_consolidated_panel,
    validate_flat_normalized,
    validate_raw_payload_structure,
)
from bist_quant.data_pipeline.types import (
    FreshnessThresholds,
    MergedDataBundle,
    NormalizedDataBundle,
    PipelineConfig,
    PipelinePaths,
    RawDataBundle,
    ValidatedDataBundle,
)

logger = logging.getLogger(__name__)


def compute_default_periods(count: int = 5, as_of: datetime | None = None) -> tuple[tuple[int, int], ...]:
    """Generate quarterly fetch periods similar to legacy fundamentals scripts."""
    now = as_of or datetime.now(timezone.utc)
    year, month = now.year, now.month
    
    # Financial reporting is typically delayed.
    # If we are in Q1, the last expected report is Q3 of the prior year
    if month <= 3:
        start_year, start_period = year - 1, 9
    elif month <= 6:
        start_year, start_period = year - 1, 12
    elif month <= 9:
        start_year, start_period = year, 3
    else:
        start_year, start_period = year, 6

    periods: list[tuple[int, int]] = []
    current_year = start_year
    current_period = start_period
    for _ in range(count):
        periods.append((current_year, current_period))
        current_period -= 3
        if current_period <= 0:
            current_period = 12
            current_year -= 1
    return tuple(periods)


def build_default_paths(base_dir: Path | None = None) -> PipelinePaths:
    """Create default path contract for the fundamentals reliability pipeline."""
    root = (base_dir or Path(__file__).resolve().parents[3]).resolve()
    data_dir = root / "data"
    fundamentals_dir = data_dir / "fundamentals"
    log_dir = root / "logs"
    provenance_dir = fundamentals_dir / "provenance"
    return PipelinePaths(
        base_dir=root,
        data_dir=data_dir,
        fundamentals_dir=fundamentals_dir,
        raw_dir=fundamentals_dir / "raw",
        normalized_json_dir=fundamentals_dir / "normalized_json",
        log_dir=log_dir,
        consolidated_parquet=data_dir / "fundamental_data_consolidated.parquet",
        normalized_parquet=fundamentals_dir / "normalized.parquet",
        normalized_csv=fundamentals_dir / "normalized.csv",
        staleness_weights_json=data_dir / "fundamental_staleness_weights.json",
        freshness_report_csv=log_dir / "staleness_diagnostics.csv",
        quality_metrics_json=log_dir / "fundamentals_quality_metrics.json",
        provenance_dir=provenance_dir,
        audit_log_jsonl=log_dir / "fundamentals_audit.jsonl",
        alerts_log_jsonl=log_dir / "fundamentals_alerts.jsonl",
        cache_state_json=fundamentals_dir / "cache_state.json",
    )


def build_default_config(
    *,
    periods: tuple[tuple[int, int], ...] | None = None,
    enforce_freshness_gate: bool = True,
    allow_stale_override: bool = False,
) -> PipelineConfig:
    """Create runtime config with stable defaults for production usage."""
    resolved_periods = periods or compute_default_periods(count=5)
    return PipelineConfig(
        periods=resolved_periods,
        enforce_freshness_gate=enforce_freshness_gate,
        allow_stale_override=allow_stale_override,
    )


@dataclass
class PipelineRunResult:
    raw_bundle: RawDataBundle | None = None
    normalized_bundle: NormalizedDataBundle | None = None
    merged_bundle: MergedDataBundle | None = None
    outputs: dict[str, Path] = field(default_factory=dict)
    freshness_passed: bool = True


class FundamentalsPipeline:
    """Unified, typed fundamentals pipeline with schema and freshness controls."""

    def __init__(
        self,
        *,
        paths: PipelinePaths | None = None,
        config: PipelineConfig | None = None,
        thresholds: FreshnessThresholds | None = None,
        logger=None,
        fetcher: FundamentalsFetcher | None = None,
    ) -> None:
        self.paths = paths or build_default_paths()
        self.config = config or build_default_config()
        self.thresholds = thresholds or FreshnessThresholds()
        self.logger = logger or configure_structured_logger(
            name="bist_fundamentals_pipeline",
            log_file=self.paths.log_dir / "fundamentals_fetch_2025Q4.log",
        )
        for directory in (
            self.paths.fundamentals_dir,
            self.paths.raw_dir,
            self.paths.normalized_json_dir,
            self.paths.log_dir,
            self.paths.provenance_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.fetcher = fetcher or FundamentalsFetcher(
            config=self.config,
            paths=self.paths,
            logger=self.logger,
        )

    def fetch_data(
        self,
        *,
        tickers: list[str] | None = None,
        force: bool = False,
        max_tickers: int | None = None,
    ) -> RawDataBundle:
        ticker_list = tickers or self.fetcher.get_ticker_universe()
        if max_tickers is not None:
            ticker_list = ticker_list[: max(0, max_tickers)]
        return self.fetcher.fetch_tickers(tickers=ticker_list, force=force)

    def load_cached_raw_data(
        self,
        *,
        tickers: list[str] | None = None,
        max_tickers: int | None = None,
    ) -> RawDataBundle:
        ticker_list = tickers
        if ticker_list is not None and max_tickers is not None:
            ticker_list = ticker_list[: max(0, max_tickers)]
        return self.fetcher.load_cached_raw(tickers=ticker_list)

    def validate_schema(self, data: RawDataBundle) -> ValidatedDataBundle:
        validate_raw_payload_structure(data.raw_by_ticker)
        log_event(
            self.logger,
            "raw_schema_validated",
            ticker_count=len(data.raw_by_ticker),
            error_count=len(data.errors),
        )
        return ValidatedDataBundle(raw_bundle=data)

    def normalize_data(self, data: ValidatedDataBundle) -> NormalizedDataBundle:
        flat = validate_flat_normalized(build_flat_normalized(data.raw_bundle.raw_by_ticker))
        consolidated = validate_consolidated_panel(
            build_consolidated_panel(data.raw_bundle.raw_by_ticker)
        )
        log_event(
            self.logger,
            "normalization_complete",
            flat_rows=int(flat.shape[0]),
            consolidated_rows=int(consolidated.shape[0]),
        )
        return NormalizedDataBundle(
            consolidated_like=consolidated,
            flat_normalized=flat,
            raw_bundle=data.raw_bundle,
        )

    def merge_data(
        self,
        data: NormalizedDataBundle,
        *,
        force_merge: bool = False,
    ) -> MergedDataBundle:
        existing = self._load_existing_consolidated()
        input_fingerprint = self._compute_input_fingerprint(data.raw_bundle)

        cache_state = self._load_cache_state()
        cache_hit = (
            bool(cache_state)
            and cache_state.get("input_fingerprint") == input_fingerprint
            and self.paths.consolidated_parquet.exists()
            and not force_merge
        )
        if cache_hit:
            merged = existing
            merge_stats = {
                "cache_hit": True,
                "cache_reason": "input_fingerprint_unchanged",
                "existing_rows": int(existing.shape[0]),
                "new_rows": int(data.consolidated_like.shape[0]),
                "merged_rows": int(existing.shape[0]),
                "new_columns_added": 0,
                "cells_filled_from_new": 0,
                "cells_overwritten_from_new": 0,
            }
        else:
            merged, stats = merge_consolidated_panels(
                existing=existing,
                new_data=data.consolidated_like,
                prefer_existing_values=self.config.prefer_existing_values,
            )
            merge_stats = {"cache_hit": False, **stats}

        staleness_report = compute_staleness_report(merged)
        quality_metrics = summarize_quality_metrics(staleness_report)
        warnings = evaluate_freshness(quality_metrics, self.thresholds)
        log_event(
            self.logger,
            "merge_complete",
            **merge_stats,
            quality_metrics=quality_metrics,
            freshness_violations=warnings,
        )
        return MergedDataBundle(
            merged_consolidated=merged,
            staleness_report=staleness_report,
            quality_metrics=quality_metrics,
            merge_stats=merge_stats,
            input_fingerprint=input_fingerprint,
            warnings=warnings,
        )

    def validate_freshness(self, data: MergedDataBundle) -> bool:
        if not self.config.enforce_freshness_gate:
            if data.warnings:
                log_event(
                    self.logger,
                    "freshness_gate_warning_only",
                    violations=data.warnings,
                )
            return len(data.warnings) == 0

        enforce_freshness_gate(
            quality_metrics=data.quality_metrics,
            thresholds=self.thresholds,
            allow_override=self.config.allow_stale_override,
            alerts_log_path=self.paths.alerts_log_jsonl,
        )
        return len(data.warnings) == 0

    def save_data(
        self,
        *,
        normalized: NormalizedDataBundle,
        merged: MergedDataBundle,
    ) -> dict[str, Path]:
        outputs: dict[str, Path] = {}

        normalized.flat_normalized.to_parquet(self.paths.normalized_parquet, index=False)
        outputs["normalized_parquet"] = self.paths.normalized_parquet
        normalized.flat_normalized.to_csv(self.paths.normalized_csv, index=False)
        outputs["normalized_csv"] = self.paths.normalized_csv
        save_normalized_per_ticker_json(normalized.flat_normalized, self.paths.normalized_json_dir)
        outputs["normalized_json_dir"] = self.paths.normalized_json_dir

        if (
            self.paths.consolidated_parquet.exists()
            and not merged.merge_stats.get("cache_hit", False)
        ):
            backup = self.paths.consolidated_parquet.with_suffix(".parquet.bak")
            shutil.copy2(self.paths.consolidated_parquet, backup)
            outputs["consolidated_backup"] = backup
        if (
            not merged.merge_stats.get("cache_hit", False)
            or not self.paths.consolidated_parquet.exists()
        ):
            merged.merged_consolidated.to_parquet(self.paths.consolidated_parquet)
        outputs["consolidated_parquet"] = self.paths.consolidated_parquet

        merged.staleness_report.to_csv(self.paths.freshness_report_csv, index=False)
        outputs["freshness_report_csv"] = self.paths.freshness_report_csv

        self.paths.quality_metrics_json.write_text(
            json.dumps(merged.quality_metrics, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        outputs["quality_metrics_json"] = self.paths.quality_metrics_json

        weights = build_staleness_weights(merged.staleness_report)
        self.paths.staleness_weights_json.write_text(
            json.dumps(weights, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        outputs["staleness_weights_json"] = self.paths.staleness_weights_json

        write_dataset_provenance(
            dataset_name="fundamentals_consolidated",
            dataset_path=self.paths.consolidated_parquet,
            dataframe=merged.merged_consolidated,
            pipeline_version=self.config.pipeline_version,
            schema_version=self.config.schema_version,
            source_name=normalized.raw_bundle.source_name,
            source_timestamp=normalized.raw_bundle.fetched_at,
            input_fingerprint=merged.input_fingerprint,
            quality_metrics=merged.quality_metrics,
            audit_log_path=self.paths.audit_log_jsonl,
        )
        write_dataset_provenance(
            dataset_name="fundamentals_flat_normalized",
            dataset_path=self.paths.normalized_parquet,
            dataframe=normalized.flat_normalized,
            pipeline_version=self.config.pipeline_version,
            schema_version=self.config.schema_version,
            source_name=normalized.raw_bundle.source_name,
            source_timestamp=normalized.raw_bundle.fetched_at,
            input_fingerprint=merged.input_fingerprint,
            quality_metrics=merged.quality_metrics,
            audit_log_path=self.paths.audit_log_jsonl,
        )
        write_dataset_provenance(
            dataset_name="fundamentals_staleness_report",
            dataset_path=self.paths.freshness_report_csv,
            dataframe=merged.staleness_report,
            pipeline_version=self.config.pipeline_version,
            schema_version=self.config.schema_version,
            source_name=normalized.raw_bundle.source_name,
            source_timestamp=normalized.raw_bundle.fetched_at,
            input_fingerprint=merged.input_fingerprint,
            quality_metrics=merged.quality_metrics,
            audit_log_path=self.paths.audit_log_jsonl,
        )

        self._write_cache_state(
            merged=merged,
            output_paths=outputs,
            source_name=normalized.raw_bundle.source_name,
            source_timestamp=normalized.raw_bundle.fetched_at,
        )
        log_event(self.logger, "pipeline_outputs_written", outputs={k: str(v) for k, v in outputs.items()})
        return outputs

    def run_diagnostics(self) -> PipelineRunResult:
        consolidated = self._load_existing_consolidated()
        if consolidated.empty:
            raise FileNotFoundError(
                f"Consolidated fundamentals parquet not found: {self.paths.consolidated_parquet}"
            )
        staleness = compute_staleness_report(consolidated)
        quality_metrics = summarize_quality_metrics(staleness)
        warnings = evaluate_freshness(quality_metrics, self.thresholds)
        merged_bundle = MergedDataBundle(
            merged_consolidated=consolidated,
            staleness_report=staleness,
            quality_metrics=quality_metrics,
            merge_stats={"cache_hit": True, "mode": "diagnostics_only"},
            input_fingerprint=self._compute_input_fingerprint(
                RawDataBundle(
                    raw_by_ticker={},
                    errors=[],
                    source_name="diagnostics_only",
                    fetched_at=datetime.now(timezone.utc),
                )
            ),
            warnings=warnings,
        )
        freshness_passed = self.validate_freshness(merged_bundle)
        weights = build_staleness_weights(staleness)
        self.paths.freshness_report_csv.write_text(staleness.to_csv(index=False), encoding="utf-8")
        self.paths.quality_metrics_json.write_text(
            json.dumps(quality_metrics, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        self.paths.staleness_weights_json.write_text(
            json.dumps(weights, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        log_event(
            self.logger,
            "diagnostics_complete",
            quality_metrics=quality_metrics,
            freshness_violations=warnings,
        )
        return PipelineRunResult(
            merged_bundle=merged_bundle,
            outputs={
                "freshness_report_csv": self.paths.freshness_report_csv,
                "quality_metrics_json": self.paths.quality_metrics_json,
                "staleness_weights_json": self.paths.staleness_weights_json,
            },
            freshness_passed=freshness_passed,
        )

    def process_raw_bundle(
        self,
        *,
        raw_bundle: RawDataBundle,
        force_merge: bool = False,
    ) -> PipelineRunResult:
        validated = self.validate_schema(raw_bundle)
        normalized = self.normalize_data(validated)
        merged = self.merge_data(normalized, force_merge=force_merge)
        freshness_passed = self.validate_freshness(merged)
        outputs = self.save_data(normalized=normalized, merged=merged)
        return PipelineRunResult(
            raw_bundle=raw_bundle,
            normalized_bundle=normalized,
            merged_bundle=merged,
            outputs=outputs,
            freshness_passed=freshness_passed,
        )

    def run(
        self,
        *,
        tickers: list[str] | None = None,
        force: bool = False,
        fetch_only: bool = False,
        merge_only: bool = False,
        diagnostics_only: bool = False,
        max_tickers: int | None = None,
    ) -> PipelineRunResult:
        log_event(
            self.logger,
            "pipeline_start",
            tickers=tickers if tickers is None else len(tickers),
            force=force,
            fetch_only=fetch_only,
            merge_only=merge_only,
            diagnostics_only=diagnostics_only,
            max_tickers=max_tickers,
        )
        if diagnostics_only:
            return self.run_diagnostics()

        if merge_only:
            raw_bundle = self.load_cached_raw_data(tickers=tickers, max_tickers=max_tickers)
            return self.process_raw_bundle(raw_bundle=raw_bundle, force_merge=force)

        raw_bundle = self.fetch_data(tickers=tickers, force=force, max_tickers=max_tickers)
        if fetch_only:
            self.validate_schema(raw_bundle)
            return PipelineRunResult(raw_bundle=raw_bundle, freshness_passed=True)

        return self.process_raw_bundle(raw_bundle=raw_bundle, force_merge=force)

    def _load_existing_consolidated(self) -> pd.DataFrame:
        if not self.paths.consolidated_parquet.exists():
            empty_index = pd.MultiIndex.from_arrays(
                [[], [], []],
                names=["ticker", "sheet_name", "row_name"],
            )
            return pd.DataFrame(index=empty_index)
        existing = pd.read_parquet(self.paths.consolidated_parquet)
        if isinstance(existing.index, pd.MultiIndex) and existing.index.has_duplicates:
            # Legacy parquet snapshots contain duplicate triples for some rows.
            existing = existing.groupby(level=["ticker", "sheet_name", "row_name"]).first()
            log_event(
                self.logger,
                "existing_panel_deduplicated",
                deduplicated_rows=int(existing.shape[0]),
            )
        return validate_consolidated_panel(existing)

    def _compute_input_fingerprint(self, raw_bundle: RawDataBundle) -> str:
        ticker_payload_summary: dict[str, Any] = {}
        for ticker, payload in sorted(raw_bundle.raw_by_ticker.items()):
            items = payload.get("items", [])
            ticker_payload_summary[ticker] = {
                "symbol": payload.get("symbol"),
                "periods_requested": payload.get("periods_requested"),
                "item_count": len(items) if isinstance(items, list) else 0,
                "items_checksum": stable_json_sha256(items if isinstance(items, list) else []),
                "fetch_timestamp": payload.get("fetch_timestamp"),
            }
        payload = {
            "source_name": raw_bundle.source_name,
            "fetched_at": raw_bundle.fetched_at.isoformat(),
            "pipeline_version": self.config.pipeline_version,
            "schema_version": self.config.schema_version,
            "periods": list(self.config.periods),
            "tickers": ticker_payload_summary,
        }
        return stable_json_sha256(payload)

    def _load_cache_state(self) -> dict[str, Any]:
        if not self.paths.cache_state_json.exists():
            return {}
        try:
            return json.loads(self.paths.cache_state_json.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_cache_state(
        self,
        *,
        merged: MergedDataBundle,
        output_paths: dict[str, Path],
        source_name: str,
        source_timestamp: datetime,
    ) -> None:
        state = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_name": source_name,
            "source_timestamp": source_timestamp.isoformat(),
            "pipeline_version": self.config.pipeline_version,
            "schema_version": self.config.schema_version,
            "input_fingerprint": merged.input_fingerprint,
            "quality_metrics": merged.quality_metrics,
            "outputs": {
                key: {
                    "path": str(path),
                    "checksum_sha256": (
                        file_checksum_sha256(path)
                        if path.exists() and path.is_file()
                        else None
                    ),
                }
                for key, path in output_paths.items()
            },
        }
        self.paths.cache_state_json.write_text(
            json.dumps(state, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
