from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bist_quant.data_pipeline.errors import ProvenanceError
from bist_quant.data_pipeline.logging_utils import append_jsonl


@dataclass(frozen=True)
class DatasetProvenance:
    """Provenance metadata persisted next to output datasets."""

    dataset_name: str
    dataset_path: str
    generated_at: str
    pipeline_version: str
    schema_version: str
    row_count: int
    column_count: int
    checksum_sha256: str
    source_name: str
    source_timestamp: str
    input_fingerprint: str
    quality_metrics: dict[str, Any]


def dataframe_checksum_sha256(frame: pd.DataFrame) -> str:
    """Deterministic DataFrame checksum based on values + index + columns."""
    normalized = frame.sort_index().sort_index(axis=1)
    hashed = pd.util.hash_pandas_object(normalized, index=True).to_numpy(dtype=np.uint64, copy=False)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def file_checksum_sha256(path: Path) -> str:
    """Compute file SHA256 checksum."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(payload: Any) -> str:
    """Compute deterministic SHA256 for nested JSON-serializable payloads."""
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_dataset_provenance(
    *,
    dataset_name: str,
    dataset_path: Path,
    dataframe: pd.DataFrame,
    pipeline_version: str,
    schema_version: str,
    source_name: str,
    source_timestamp: datetime,
    input_fingerprint: str,
    quality_metrics: dict[str, Any],
    audit_log_path: Path,
) -> DatasetProvenance:
    """Persist provenance metadata and append audit entry."""
    try:
        checksum = dataframe_checksum_sha256(dataframe)
        meta = DatasetProvenance(
            dataset_name=dataset_name,
            dataset_path=str(dataset_path),
            generated_at=datetime.now(timezone.utc).isoformat(),
            pipeline_version=pipeline_version,
            schema_version=schema_version,
            row_count=int(dataframe.shape[0]),
            column_count=int(dataframe.shape[1]),
            checksum_sha256=checksum,
            source_name=source_name,
            source_timestamp=source_timestamp.isoformat(),
            input_fingerprint=input_fingerprint,
            quality_metrics=dict(quality_metrics),
        )
        meta_path = dataset_path.with_suffix(dataset_path.suffix + ".meta.json")
        meta_path.write_text(json.dumps(asdict(meta), ensure_ascii=False, indent=2), encoding="utf-8")
        append_jsonl(
            audit_log_path,
            {
                "event": "dataset_provenance_written",
                "dataset": dataset_name,
                "path": str(dataset_path),
                "meta_path": str(meta_path),
                "checksum": checksum,
                "rows": meta.row_count,
                "columns": meta.column_count,
            },
        )
        return meta
    except Exception as exc:  # pragma: no cover - defensive path
        raise ProvenanceError(f"Failed to write provenance for {dataset_name}: {exc}") from exc
