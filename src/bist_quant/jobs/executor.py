"""Asynchronous job execution primitives."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
import traceback
from typing import Awaitable, Callable

from .models import Job, JobProgress, JobResult, JobStatus


@dataclass
class ExecutionContext:
    """Execution context passed to job handlers."""

    job: Job
    progress_callback: Callable[[JobProgress], Awaitable[None]] | None = None
    cancellation_token: asyncio.Event | None = None

    async def report_progress(self, current: int, total: int, message: str = "") -> None:
        progress = JobProgress(current=current, total=total, message=message)
        self.job.progress = progress
        if self.progress_callback is not None:
            await self.progress_callback(progress)

    def is_cancelled(self) -> bool:
        return self.cancellation_token is not None and self.cancellation_token.is_set()


JobHandler = Callable[[ExecutionContext], Awaitable[JobResult]]


class JobExecutor:
    """Execute jobs with bounded concurrency and cancellation support."""

    def __init__(self, max_concurrent: int = 4):
        self.max_concurrent = max(1, int(max_concurrent))
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._handlers: dict[str, JobHandler] = {}
        self._running_jobs: dict[str, asyncio.Task[JobResult]] = {}
        self._cancellation_tokens: dict[str, asyncio.Event] = {}

    def register_handler(self, job_type: str, handler: JobHandler) -> None:
        self._handlers[str(job_type)] = handler

    async def execute(
        self,
        job: Job,
        progress_callback: Callable[[JobProgress], Awaitable[None]] | None = None,
    ) -> JobResult:
        handler = self._handlers.get(job.type.value)
        if handler is None:
            raise ValueError(f"No handler registered for job type: {job.type.value}")

        async with self._semaphore:
            cancellation_token = asyncio.Event()
            self._cancellation_tokens[job.id] = cancellation_token
            context = ExecutionContext(
                job=job,
                progress_callback=progress_callback,
                cancellation_token=cancellation_token,
            )

            job.status = JobStatus.RUNNING
            job.started_at = datetime.utcnow()

            try:
                task: asyncio.Task[JobResult] = asyncio.create_task(handler(context))
                self._running_jobs[job.id] = task
                result = await task
                if not isinstance(result, JobResult):
                    result = JobResult(success=True, data=result)
                job.status = JobStatus.COMPLETED
                job.result = result
            except asyncio.CancelledError:
                job.status = JobStatus.CANCELLED
                job.result = JobResult(success=False, error="Job cancelled")
            except Exception as exc:
                job.status = JobStatus.FAILED
                job.error = str(exc)
                job.result = JobResult(
                    success=False,
                    error=str(exc),
                    data={"traceback": traceback.format_exc()},
                )
            finally:
                job.completed_at = datetime.utcnow()
                self._running_jobs.pop(job.id, None)
                self._cancellation_tokens.pop(job.id, None)

            return job.result or JobResult(success=False, error="Unknown execution failure")

    async def cancel(self, job_id: str) -> bool:
        token = self._cancellation_tokens.get(job_id)
        task = self._running_jobs.get(job_id)
        if token is None or task is None:
            return False
        token.set()
        task.cancel()
        return True

    def get_running_jobs(self) -> list[str]:
        return list(self._running_jobs.keys())

