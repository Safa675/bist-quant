"""Job management package."""

from .executor import ExecutionContext, JobExecutor, JobHandler

from .models import Job, JobProgress, JobResult, JobStatus, JobType

__all__ = [
    "Job",
    "JobStatus",
    "JobType",
    "JobProgress",
    "JobResult",
    "JobExecutor",
    "ExecutionContext",
    "JobHandler",
    "JobManager",
]


def __getattr__(name: str):
    if name == "JobManager":
        from .manager import JobManager

        return JobManager
    raise AttributeError(name)
