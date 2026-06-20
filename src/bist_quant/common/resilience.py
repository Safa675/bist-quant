"""Generic resilience primitives for external service calls.

Provides circuit-breaker, retry-with-backoff, and logging configuration
utilities used by ``BorsapyClient`` and potentially other I/O-bound
adapters.
"""

from __future__ import annotations

import logging
import random
import time
from enum import Enum
from functools import wraps
from pathlib import Path
from threading import Lock
from typing import Any, Callable


logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerError(RuntimeError):
    """Raised when the circuit breaker blocks a call."""


class CircuitBreaker:
    """Simple thread-safe circuit breaker."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = int(failure_threshold)
        self.recovery_timeout = int(recovery_timeout)

        self.failure_count = 0
        self.last_failure_time: float | None = None
        self.state = CircuitState.CLOSED
        self.lock = Lock()

    def call(self, func: Callable[..., Any], *args, **kwargs):
        with self.lock:
            if self.state == CircuitState.OPEN:
                elapsed = time.time() - (self.last_failure_time or 0)
                if elapsed > self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                else:
                    raise CircuitBreakerError("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except Exception:
            self.on_failure()
            raise

    def on_success(self):
        with self.lock:
            self.failure_count = 0
            self.state = CircuitState.CLOSED
            self.last_failure_time = None

    def on_failure(self):
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
):
    """Retry decorator with exponential backoff and optional jitter."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = int(max_retries)
            delay = float(base_delay)
            cap = float(max_delay)
            factor = float(backoff_factor)
            use_jitter = bool(jitter)

            if args:
                policy = getattr(args[0], "_retry_policy", None)
                if isinstance(policy, dict):
                    retries = int(policy.get("max_retries", retries))
                    delay = float(policy.get("base_delay", delay))
                    cap = float(policy.get("max_delay", cap))
                    factor = float(policy.get("backoff_factor", factor))
                    use_jitter = bool(policy.get("jitter", use_jitter))

            current_delay = max(0.0, delay)
            last_exception = None

            for attempt in range(retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exception = exc
                    if attempt == retries:
                        break

                    sleep_time = current_delay
                    if use_jitter and current_delay > 0:
                        sleep_time = random.uniform(0, current_delay)

                    logger.warning(
                        f"Attempt {attempt + 1} failed: {exc}. Retrying in {sleep_time:.2f}s..."
                    )
                    time.sleep(max(0.0, sleep_time))
                    current_delay = min(max(0.0, current_delay * factor), cap)

            raise last_exception

        return wrapper

    return decorator


def configure_borsapy_logging(
    log_file: str | Path = "borsapy_integration.log",
    level: int = logging.INFO,
) -> None:
    """
    Configure default logging for borsapy integrations.

    This only configures the root logger if no handlers are present.
    """
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
