# `jobs/` — Async Job Queue

## Purpose

A complete async job lifecycle system: typed job definitions, an executor with bounded concurrency and cancellation, a high-level manager with persistence and progress streaming, and a pub/sub broker for real-time progress delivery to multiple subscribers.

## Files

```
jobs/
├── models.py     # Job data models: Job, JobStatus, JobType, JobProgress, JobResult
├── executor.py   # Low-level async executor with semaphore + cancellation
├── manager.py    # High-level lifecycle manager: submit, cancel, query, cleanup
└── progress.py   # In-process async pub/sub broker for progress streaming
```

---

## Data Models (`models.py`)

| Class | Description |
|---|---|
| `JobStatus(Enum)` | `PENDING, RUNNING, COMPLETED, FAILED, CANCELLED` |
| `JobType(Enum)` | `BACKTEST, OPTIMIZATION, SIGNAL_GENERATION, DATA_REFRESH, CUSTOM` |
| `JobProgress` | `current, total, message` — auto-computes `percentage` |
| `JobResult` | `success, data, error, artifacts` |
| `Job` | Full record: auto-UUID id, lifecycle timestamps, `duration_seconds`, `is_terminal` |

`Job.is_terminal` is `True` when status is `COMPLETED`, `FAILED`, or `CANCELLED`.

---

## Executor (`executor.py`)

Low-level async runner with:
- `asyncio.Semaphore(max_concurrent)` for bounded concurrency.
- Per-job `asyncio.Event` for cancellation signaling.
- Typed `ExecutionContext` passed to handlers (with `report_progress()` and `is_cancelled()`).

```python
# Register a handler
executor.register_handler(JobType.BACKTEST, my_backtest_handler)

# Execute
await executor.execute(job, progress_callback=my_callback)

# Cancel
await executor.cancel(job_id)
```

**`job.result` is always set** on all exit paths (success, error, cancellation).

**`JobHandler` type alias:** `Callable[[ExecutionContext], Awaitable[JobResult]]`

---

## Manager (`manager.py`)

High-level orchestrator. Fires-and-forgets execution via `asyncio.create_task`.

```python
# Submit a job (returns immediately)
job = await manager.submit(JobType.BACKTEST, payload={"signal": "momentum"})

# Query
job = await manager.get(job_id)
jobs = await manager.list(status=JobStatus.RUNNING, limit=20)

# Cancel
await manager.cancel(job_id)

# Stream progress
async for progress in manager.subscribe_progress(job_id):
    print(f"{progress.percentage:.0f}% — {progress.message}")

# Cleanup old jobs
await manager.cleanup_old_jobs(days=30)
```

Progress is forwarded to both the `JobStore` (persistence) and the `ProgressBroker` (pub/sub). Terminal-state jobs are always saved to the store in a `finally` block.

---

## Progress Broker (`progress.py`)

In-process async pub/sub for streaming progress to multiple subscribers:

```python
# Publish
await broker.publish(job_id, JobProgress(current=50, total=100, message="Running..."))

# Subscribe (async generator)
async for progress in broker.subscribe(job_id):
    ...  # None sentinel signals end-of-stream

# Close (sends sentinel to all subscribers)
await broker.close(job_id)
```

Thread/task safety via `asyncio.Lock`. `None` sentinel signals stream end.

---

## Local Rules for Contributors

1. **Always set `job.result`** in your handler, even on failure paths — the executor guarantees this in the finally block but handlers should also set it explicitly.
2. **Use `ExecutionContext` for progress reporting** — do not accept a raw callback in your handler. The context is injected by the executor.
3. **Check `ctx.is_cancelled()` periodically** in long-running handlers to support cooperative cancellation.
4. **`JobType.CUSTOM`** is for one-off jobs. If a new job type recurs, add a named entry to `JobType`.
5. **Never block the event loop** in job handlers. Use `asyncio.to_thread()` for CPU-intensive work or synchronous I/O.
