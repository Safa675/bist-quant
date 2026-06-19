"""
Run history and artifact persistence.

Migrated from bist_quant.ai/api/run_store.py.

Provides a JSON-backed run store with atomic writes
and in-process mutual exclusion via threading.Lock.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

_RUN_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,160}$")
_ARTIFACT_ID_RE = re.compile(r"^[a-z0-9_]{1,180}$")


class RunStoreError(RuntimeError):
    """Raised when run/artifact store operations fail."""


class RunStore:
    """JSON-backed run store with atomic writes and in-process mutual exclusion."""

    def __init__(self, store_path: Path, artifacts_dir: Path) -> None:
        self._store_path = Path(store_path).resolve()
        self._artifacts_dir = Path(artifacts_dir).resolve()
        self._lock = Lock()
        self._ensure_storage()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _sanitize_kind(value: str) -> str:
        text = str(value).strip().lower()
        text = re.sub(r"[^a-z0-9_]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        return text[:64] or "run"

    @staticmethod
    def _sanitize_artifact_id(value: str) -> str | None:
        text = str(value or "").strip().lower()
        if not text:
            return None
        if not _ARTIFACT_ID_RE.match(text):
            return None
        return text

    @staticmethod
    def _normalize_run_id(value: str | None, *, fallback_kind: str) -> str:
        raw = str(value or "").strip()
        if raw and _RUN_ID_RE.match(raw):
            return raw
        prefix = RunStore._sanitize_kind(fallback_kind)
        return f"{prefix}_{uuid4().hex[:12]}"

    def _ensure_storage(self) -> None:
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        if self._store_path.exists():
            return
        self._write_state({"runs": []})

    def _load_state(self) -> dict[str, Any]:
        if not self._store_path.exists():
            return {"runs": []}
        try:
            payload = json.loads(self._store_path.read_text(encoding="utf-8"))
        except Exception:
            return {"runs": []}

        if not isinstance(payload, dict):
            return {"runs": []}
        runs = payload.get("runs")
        if not isinstance(runs, list):
            payload["runs"] = []
        return payload

    def _write_state(self, payload: dict[str, Any]) -> None:
        tmp = self._store_path.with_name(f"{self._store_path.name}.{uuid4().hex}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self._store_path)

    @staticmethod
    def _sort_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            runs,
            key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""),
            reverse=True,
        )

    @staticmethod
    def _merge_meta(
        current: dict[str, Any] | None,
        patch: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if patch is None:
            return current
        merged: dict[str, Any] = {}
        if isinstance(current, dict):
            merged.update(current)
        merged.update(patch)
        return merged

    def create_or_update_run(
        self,
        *,
        run_id: str | None,
        kind: str,
        request_payload: dict[str, Any],
        status: str,
        meta: dict[str, Any] | None = None,
        artifact_id: str | None = None,
        error: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_kind = self._sanitize_kind(kind)
        normalized_run_id = self._normalize_run_id(run_id, fallback_kind=normalized_kind)

        with self._lock:
            state = self._load_state()
            runs = state.setdefault("runs", [])
            now = self._now_iso()

            existing_idx = -1
            for idx, row in enumerate(runs):
                if isinstance(row, dict) and row.get("id") == normalized_run_id:
                    existing_idx = idx
                    break

            if existing_idx < 0:
                record = {
                    "id": normalized_run_id,
                    "kind": normalized_kind,
                    "status": str(status),
                    "created_at": now,
                    "updated_at": now,
                    "started_at": now if status == "running" else None,
                    "finished_at": now if status in {"succeeded", "failed", "cancelled"} else None,
                    "request": request_payload,
                    "meta": meta or {},
                    "artifact_id": artifact_id,
                    "error": error,
                }
                runs.append(record)
            else:
                current = runs[existing_idx]
                if not isinstance(current, dict):
                    raise RunStoreError(f"Run store record at index {existing_idx} is invalid")

                next_status = str(status or current.get("status") or "queued")
                started_at = current.get("started_at")
                finished_at = current.get("finished_at")
                if next_status == "running" and not started_at:
                    started_at = now
                if next_status in {"succeeded", "failed", "cancelled"} and not finished_at:
                    finished_at = now

                record = {
                    "id": normalized_run_id,
                    "kind": str(current.get("kind") or normalized_kind),
                    "status": next_status,
                    "created_at": str(current.get("created_at") or now),
                    "updated_at": now,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "request": request_payload or current.get("request") or {},
                    "meta": self._merge_meta(
                        current.get("meta") if isinstance(current.get("meta"), dict) else None,
                        meta,
                    )
                    or {},
                    "artifact_id": artifact_id if artifact_id is not None else current.get("artifact_id"),
                    "error": error if error is not None else current.get("error"),
                }
                runs[existing_idx] = record

            state["runs"] = self._sort_runs([row for row in runs if isinstance(row, dict)])
            self._write_state(state)
            return record

    def list_runs(
        self,
        *,
        kind: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        with self._lock:
            state = self._load_state()

        rows: list[dict[str, Any]] = []
        for row in state.get("runs", []):
            if not isinstance(row, dict):
                continue
            rows.append(row)

        if kind:
            normalized_kind = self._sanitize_kind(kind)
            rows = [row for row in rows if str(row.get("kind")) == normalized_kind]
        if status:
            target_status = str(status).strip().lower()
            rows = [row for row in rows if str(row.get("status", "")).strip().lower() == target_status]

        rows = self._sort_runs(rows)
        total = len(rows)
        safe_offset = max(0, int(offset))
        safe_limit = max(1, min(int(limit), 500))

        return {
            "runs": rows[safe_offset : safe_offset + safe_limit],
            "total": total,
        }

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            state = self._load_state()

        for row in state.get("runs", []):
            if isinstance(row, dict) and row.get("id") == run_id:
                return row
        return None

    def find_run_by_meta(self, *, key: str, value: str) -> dict[str, Any] | None:
        target_key = str(key).strip()
        target_value = str(value).strip()
        if not target_key or not target_value:
            return None

        with self._lock:
            state = self._load_state()

        for row in state.get("runs", []):
            if not isinstance(row, dict):
                continue
            meta = row.get("meta")
            if not isinstance(meta, dict):
                continue
            if str(meta.get(target_key, "")).strip() == target_value:
                return row
        return None

    def save_artifact(
        self,
        *,
        kind: str,
        run_id: str,
        payload: Any,
    ) -> dict[str, Any]:
        normalized_kind = self._sanitize_kind(kind)
        run_fragment = self._sanitize_kind(run_id)
        artifact_id = f"{normalized_kind}_{run_fragment}_{uuid4().hex[:10]}"
        file_path = self._artifacts_dir / f"{artifact_id}.json"
        tmp = file_path.with_name(f"{file_path.name}.{uuid4().hex}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        tmp.replace(file_path)

        return {
            "id": artifact_id,
            "kind": normalized_kind,
            "path": str(file_path),
            "created_at": self._now_iso(),
            "size_bytes": file_path.stat().st_size,
        }

    def read_artifact(self, artifact_id: str) -> tuple[Path, bytes] | None:
        normalized = self._sanitize_artifact_id(artifact_id)
        if normalized is None:
            return None
        path = (self._artifacts_dir / f"{normalized}.json").resolve()
        if not path.exists() or not path.is_file():
            return None

        try:
            path.relative_to(self._artifacts_dir)
        except ValueError:
            return None

        return path, path.read_bytes()
