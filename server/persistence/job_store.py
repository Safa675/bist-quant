"""Persistent storage for async job records."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
import json
from pathlib import Path
from uuid import uuid4

from ..jobs.models import Job, JobProgress, JobResult, JobStatus, JobType


class JobStore:
    """JSON-backed persistence for jobs with atomic writes."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path or Path("data/jobs.json")).resolve()
        self._lock = asyncio.Lock()
        self._cache: dict[str, Job] = {}
        self._loaded = False

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        async with self._lock:
            if self._loaded:
                return

            if self.path.exists():
                try:
                    raw = await asyncio.to_thread(self.path.read_text, encoding="utf-8")
                    data = json.loads(raw)
                except Exception:
                    data = {}
                jobs = data.get("jobs", []) if isinstance(data, dict) else []
                if isinstance(jobs, list):
                    for item in jobs:
                        if not isinstance(item, dict):
                            continue
                        job = self._deserialize(item)
                        self._cache[job.id] = job

            self._loaded = True

    async def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "jobs": [self._serialize(job) for job in self._cache.values()],
            "updated_at": datetime.utcnow().isoformat(),
        }

        tmp_path = self.path.with_name(f"{self.path.name}.{uuid4().hex}.tmp")
        text = json.dumps(payload, ensure_ascii=True, indent=2, default=str) + "\n"
        await asyncio.to_thread(tmp_path.write_text, text, "utf-8")
        await asyncio.to_thread(tmp_path.replace, self.path)

    def _serialize(self, job: Job) -> dict:
        return {
            "id": job.id,
            "type": job.type.value,
            "status": job.status.value,
            "payload": job.payload,
            "progress": {
                "current": job.progress.current,
                "total": job.progress.total,
                "message": job.progress.message,
            }
            if job.progress
            else None,
            "result": {
                "success": job.result.success,
                "data": job.result.data,
                "error": job.result.error,
                "artifacts": list(job.result.artifacts),
            }
            if job.result
            else None,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "error": job.error,
            "metadata": job.metadata,
        }

    def _deserialize(self, data: dict) -> Job:
        progress = None
        raw_progress = data.get("progress")
        if isinstance(raw_progress, dict):
            progress = JobProgress(
                current=int(raw_progress.get("current", 0)),
                total=int(raw_progress.get("total", 0)),
                message=str(raw_progress.get("message", "")),
            )

        result = None
        raw_result = data.get("result")
        if isinstance(raw_result, dict):
            artifacts = raw_result.get("artifacts", [])
            result = JobResult(
                success=bool(raw_result.get("success", False)),
                data=raw_result.get("data"),
                error=raw_result.get("error"),
                artifacts=[str(item) for item in artifacts] if isinstance(artifacts, list) else [],
            )

        def _parse_dt(value: object) -> datetime | None:
            if not value:
                return None
            try:
                return datetime.fromisoformat(str(value))
            except Exception:
                return None

        return Job(
            id=str(data.get("id", "")),
            type=JobType(str(data.get("type", JobType.CUSTOM.value))),
            status=JobStatus(str(data.get("status", JobStatus.PENDING.value))),
            payload=data.get("payload", {}) if isinstance(data.get("payload"), dict) else {},
            progress=progress,
            result=result,
            created_at=_parse_dt(data.get("created_at")) or datetime.utcnow(),
            started_at=_parse_dt(data.get("started_at")),
            completed_at=_parse_dt(data.get("completed_at")),
            error=str(data.get("error")) if data.get("error") is not None else None,
            metadata=data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {},
        )

    async def save(self, job: Job) -> None:
        await self._ensure_loaded()
        async with self._lock:
            self._cache[job.id] = job
            await self._persist()

    async def get(self, job_id: str) -> Job | None:
        await self._ensure_loaded()
        return self._cache.get(job_id)

    async def list(
        self,
        status: JobStatus | None = None,
        job_type: JobType | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Job]:
        await self._ensure_loaded()
        rows = list(self._cache.values())

        if status is not None:
            rows = [job for job in rows if job.status == status]
        if job_type is not None:
            rows = [job for job in rows if job.type == job_type]

        rows.sort(key=lambda job: job.created_at, reverse=True)
        safe_offset = max(0, int(offset))
        safe_limit = max(1, int(limit))
        return rows[safe_offset : safe_offset + safe_limit]

    async def update_progress(self, job_id: str, progress: JobProgress) -> None:
        await self._ensure_loaded()
        job = self._cache.get(job_id)
        if job is not None:
            job.progress = progress

    async def cleanup(self, days: int = 30) -> int:
        await self._ensure_loaded()
        cutoff = datetime.utcnow() - timedelta(days=max(0, int(days)))
        remove_ids = [
            job_id
            for job_id, job in self._cache.items()
            if job.is_terminal and job.completed_at is not None and job.completed_at < cutoff
        ]

        async with self._lock:
            for job_id in remove_ids:
                self._cache.pop(job_id, None)
            await self._persist()
        return len(remove_ids)

