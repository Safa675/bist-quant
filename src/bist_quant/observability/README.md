# `observability/` — Logging, Metrics & Telemetry

## Purpose

Production-grade observability stack. Provides structured JSON logging, in-memory metrics collection with optional Prometheus mirroring, and crash telemetry capture.

## Files

```
observability/
├── logging.py    # Structured JSON logging configuration
├── metrics.py    # In-memory metrics collector (counters, gauges, histograms)
└── telemetry.py  # Crash telemetry writer (JSONL file)
```

---

### `logging.py` — Structured Logging Configuration

```python
from bist_quant.observability.logging import configure_logging

configure_logging(
    level="INFO",
    json_format=True,    # emit JSON lines
    log_file="app.log",  # optional rotating file (10 MB, 5 backups)
)
```

`JsonLogFormatter` adds `ts`, `level`, `logger`, `message`, and optional `trace_id` / `request_path` fields to every log record.

**Important:** `disable_existing_loggers: False` — third-party library loggers are never silenced.

---

### `metrics.py` — In-Memory Metrics

```python
from bist_quant.observability.metrics import MetricsCollector

metrics = MetricsCollector()
metrics.increment("backtest.runs", labels={"strategy": "momentum"})
metrics.set_gauge("active_jobs", 3)
metrics.observe("backtest.duration_ms", 1250.0)
snapshot = metrics.snapshot()   # returns serializable dict
```

**Prometheus mirroring:** If `prometheus_client` is installed, each metric is automatically mirrored to the corresponding Prometheus `Counter`/`Gauge`/`Histogram`. Silently skipped if not installed.

Labels are encoded as sorted tuples for dict keys — always pass labels as dicts.

---

### `telemetry.py` — Crash Telemetry

```python
from bist_quant.observability.telemetry import emit_crash_telemetry

try:
    ...
except Exception as exc:
    emit_crash_telemetry(exception=exc, context={"endpoint": "/api/backtest"})
    # Writes to data/crash_telemetry.jsonl — never raises
```

`emit_crash_telemetry` **never raises**. Returns `True` on success, `False` if disabled or on write failure.

---

## Local Rules for Contributors

1. **Structured logging everywhere.** Use `log_event(logger, event, **kwargs)` (from `data_pipeline/logging_utils.py`) for pipeline code. Use `structlog` or the configured logger for service code. Never use `print()`.
2. **Metrics labels must be dicts.** Do not pass label strings directly — always use `labels={"key": "value"}`.
3. **`emit_crash_telemetry` is a best-effort fire-and-forget.** Do not block on it or check its return value in critical paths.
4. **`json_logs` is a backward-compatible alias** for `json_format` in `configure_logging`. Accept both in new code.
