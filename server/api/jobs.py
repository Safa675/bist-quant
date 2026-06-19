"""Async job manager with optional JSON persistence."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from threading import Lock
from typing import Any, Callable
from uuid import uuid4


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobRecord:
    id: str
    kind: str
    status: str
    created_at: str
    updated_at: str
    error: str | None = None
    result: Any = None
    future: Future[Any] | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    request: dict[str, Any] = field(default_factory=dict)


class JobManager:
    def __init__(self, max_workers: int = 4, store_path: Path | None = None) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="bq-job")
        self._jobs: dict[str, JobRecord] = {}
        self._lock = Lock()
        self._store_path = Path(store_path).resolve() if store_path is not None else None
        self._load_store()

    def create(
        self,
        kind: str,
        fn: Callable[[], Any],
        meta: dict[str, Any] | None = None,
        request: dict[str, Any] | None = None,
    ) -> JobRecord:
        job_id = str(uuid4())
        now = _iso_now()
        record = JobRecord(
            id=job_id,
            kind=kind,
            status="queued",
            created_at=now,
            updated_at=now,
            meta=meta or {},
            request=request or {},
        )
        with self._lock:
            self._jobs[job_id] = record
            self._persist_locked()

        def _run() -> Any:
            self._set_status(job_id, "running")
            try:
                result = fn()
                self._set_result(job_id, result)
                return result
            except Exception as exc:
                self._set_error(job_id, str(exc))
                raise

        future = self._executor.submit(_run)
        record.future = future
        return record

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 50) -> list[JobRecord]:
        with self._lock:
            rows = sorted(self._jobs.values(), key=lambda r: r.created_at, reverse=True)
        return rows[: max(1, min(limit, 200))]

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            record = self._jobs.get(job_id)
        if record is None or record.future is None:
            return False
        cancelled = record.future.cancel()
        if cancelled:
            self._set_status(job_id, "cancelled")
        return cancelled

    def to_dict(self, record: JobRecord) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": record.id,
            "kind": record.kind,
            "status": record.status,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "error": record.error,
            "meta": record.meta,
            "request": record.request,
        }
        if record.status == "completed":
            payload["result"] = record.result
        return payload

    def _set_status(self, job_id: str, status: str) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            record.status = status
            record.updated_at = _iso_now()
            self._persist_locked()

    def _set_result(self, job_id: str, result: Any) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            record.status = "completed"
            record.result = result
            record.updated_at = _iso_now()
            self._persist_locked()

    def _set_error(self, job_id: str, error: str) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            record.status = "failed"
            record.error = error
            record.updated_at = _iso_now()
            self._persist_locked()

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            return None if math.isnan(value) or math.isinf(value) else value
        if isinstance(value, dict):
            return {str(k): JobManager._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [JobManager._json_safe(v) for v in value]
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    def _persist_locked(self) -> None:
        if self._store_path is None:
            return
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for record in self._jobs.values():
            rows.append(
                {
                    "id": record.id,
                    "kind": record.kind,
                    "status": record.status,
                    "created_at": record.created_at,
                    "updated_at": record.updated_at,
                    "error": record.error,
                    "meta": self._json_safe(record.meta),
                    "request": self._json_safe(record.request),
                    "result": self._json_safe(record.result),
                }
            )
        payload = {"jobs": rows, "updated_at": _iso_now()}
        tmp_path = self._store_path.with_name(f"{self._store_path.name}.{uuid4().hex}.tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
        )
        tmp_path.replace(self._store_path)

    def _load_store(self) -> None:
        if self._store_path is None or not self._store_path.exists():
            return
        try:
            payload = json.loads(self._store_path.read_text(encoding="utf-8"))
        except Exception:
            return
        rows = payload.get("jobs", []) if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            return
        for item in rows:
            if not isinstance(item, dict):
                continue
            job_id = str(item.get("id") or "").strip()
            kind = str(item.get("kind") or "").strip()
            if not job_id or not kind:
                continue
            self._jobs[job_id] = JobRecord(
                id=job_id,
                kind=kind,
                status=str(item.get("status") or "failed"),
                created_at=str(item.get("created_at") or _iso_now()),
                updated_at=str(item.get("updated_at") or _iso_now()),
                error=str(item.get("error")) if item.get("error") is not None else None,
                result=item.get("result"),
                future=None,
                meta=item.get("meta") if isinstance(item.get("meta"), dict) else {},
                request=item.get("request") if isinstance(item.get("request"), dict) else {},
            )
