"""Rate limiting utilities."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: int
    remaining: int


class InMemoryRateLimiter:
    """Simple sliding-window limiter keyed by client and route."""

    def __init__(self, limit_per_minute: int):
        self._limit = max(1, limit_per_minute)
        self._window_seconds = 60.0
        self._buckets: dict[str, deque[float]] = {}
        self._lock = Lock()

    def check(self, key: str) -> RateLimitDecision:
        now = time.monotonic()
        window_start = now - self._window_seconds
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = deque()
                self._buckets[key] = bucket

            while bucket and bucket[0] < window_start:
                bucket.popleft()

            if len(bucket) >= self._limit:
                retry_after = max(1, int(self._window_seconds - (now - bucket[0])))
                return RateLimitDecision(
                    allowed=False,
                    retry_after_seconds=retry_after,
                    remaining=0,
                )

            bucket.append(now)
            remaining = max(0, self._limit - len(bucket))
            return RateLimitDecision(
                allowed=True,
                retry_after_seconds=0,
                remaining=remaining,
            )
