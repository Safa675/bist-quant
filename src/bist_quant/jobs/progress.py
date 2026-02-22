"""Progress pub/sub utilities for job streaming."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from .models import JobProgress


class ProgressBroker:
    """In-process async pub/sub broker keyed by job id."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue[JobProgress | None]]] = {}
        self._lock = asyncio.Lock()

    async def publish(self, job_id: str, progress: JobProgress) -> None:
        async with self._lock:
            subscribers = list(self._subscribers.get(job_id, []))
        for queue in subscribers:
            await queue.put(progress)

    async def close(self, job_id: str) -> None:
        async with self._lock:
            subscribers = list(self._subscribers.pop(job_id, []))
        for queue in subscribers:
            await queue.put(None)

    async def subscribe(self, job_id: str) -> AsyncIterator[JobProgress]:
        queue: asyncio.Queue[JobProgress | None] = asyncio.Queue()
        async with self._lock:
            self._subscribers.setdefault(job_id, []).append(queue)

        try:
            while True:
                item = await queue.get()
                if item is None:
                    return
                yield item
        finally:
            async with self._lock:
                subscribers = self._subscribers.get(job_id, [])
                if queue in subscribers:
                    subscribers.remove(queue)
                if not subscribers and job_id in self._subscribers:
                    del self._subscribers[job_id]

