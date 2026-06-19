"""High-level job lifecycle manager."""

from __future__ import annotations

import asyncio
from datetime import datetime
import traceback
from typing import AsyncIterator

from .executor import JobExecutor, JobHandler
from .models import Job, JobProgress, JobResult, JobStatus, JobType
from .progress import ProgressBroker
from ..persistence.job_store import JobStore


class JobManager:
    """Manage job persistence, execution, and progress subscriptions."""

    def __init__(
        self,
        store: JobStore | None = None,
        executor: JobExecutor | None = None,
        max_concurrent: int = 4,
    ):
        self.store = store or JobStore()
        self.executor = executor or JobExecutor(max_concurrent=max_concurrent)
        self._progress = ProgressBroker()

    def register_handler(
        self,
        job_type: JobType,
        handler: JobHandler | None = None,
    ):
        """Register a handler directly or as a decorator."""

        def _register(fn: JobHandler) -> JobHandler:
            self.executor.register_handler(job_type.value, fn)
            return fn

        if handler is None:
            return _register
        return _register(handler)

    async def submit(
        self,
        job_type: JobType,
        payload: dict,
        metadata: dict | None = None,
    ) -> Job:
        job = Job(
            type=job_type,
            payload=dict(payload or {}),
            metadata=dict(metadata or {}),
        )
        await self.store.save(job)
        asyncio.create_task(self._execute_job(job))
        return job

    async def _execute_job(self, job: Job) -> None:
        async def progress_callback(progress: JobProgress) -> None:
            await self.store.update_progress(job.id, progress)
            await self._progress.publish(job.id, progress)

        try:
            await self.executor.execute(job, progress_callback=progress_callback)
        except Exception as exc:
            job.status = JobStatus.FAILED
            job.error = str(exc)
            job.completed_at = datetime.utcnow()
            job.result = JobResult(
                success=False,
                error=str(exc),
                data={"traceback": traceback.format_exc()},
            )
        finally:
            await self.store.save(job)
            await self._progress.close(job.id)

    async def get(self, job_id: str) -> Job | None:
        return await self.store.get(job_id)

    async def list(
        self,
        status: JobStatus | None = None,
        job_type: JobType | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Job]:
        return await self.store.list(
            status=status,
            job_type=job_type,
            limit=limit,
            offset=offset,
        )

    async def cancel(self, job_id: str) -> bool:
        success = await self.executor.cancel(job_id)
        if success:
            job = await self.store.get(job_id)
            if job is not None:
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.utcnow()
                job.result = JobResult(success=False, error="Job cancelled")
                await self.store.save(job)
                await self._progress.close(job_id)
        return success

    async def subscribe_progress(self, job_id: str) -> AsyncIterator[JobProgress]:
        current = await self.store.get(job_id)
        if current is not None and current.progress is not None:
            yield current.progress
        if current is not None and current.is_terminal:
            return

        async for progress in self._progress.subscribe(job_id):
            yield progress

    async def cleanup_old_jobs(self, days: int = 30) -> int:
        return await self.store.cleanup(days=days)

