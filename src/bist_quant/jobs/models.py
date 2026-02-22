"""Job data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import uuid


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(Enum):
    BACKTEST = "backtest"
    OPTIMIZATION = "optimization"
    SIGNAL_GENERATION = "signal_generation"
    DATA_REFRESH = "data_refresh"
    CUSTOM = "custom"


@dataclass
class JobProgress:
    """Progress update for a running job."""

    current: int
    total: int
    message: str
    percentage: float = field(init=False)

    def __post_init__(self) -> None:
        self.percentage = (self.current / self.total * 100.0) if self.total > 0 else 0.0


@dataclass
class JobResult:
    """Result of a completed job."""

    success: bool
    data: Any | None = None
    error: str | None = None
    artifacts: list[str] = field(default_factory=list)


@dataclass
class Job:
    """Job definition and runtime state."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: JobType = JobType.CUSTOM
    status: JobStatus = JobStatus.PENDING
    payload: dict[str, Any] = field(default_factory=dict)
    progress: JobProgress | None = None
    result: JobResult | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.completed_at or datetime.utcnow()
        return (end - self.started_at).total_seconds()

    @property
    def is_terminal(self) -> bool:
        return self.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}

