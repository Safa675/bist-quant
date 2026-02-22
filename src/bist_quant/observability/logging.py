"""Structured logging helpers for production workloads."""

from __future__ import annotations

import json
import logging
import logging.config
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonLogFormatter(logging.Formatter):
    """Serialize logs as line-delimited JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if hasattr(record, "trace_id"):
            payload["trace_id"] = getattr(record, "trace_id")
        if hasattr(record, "request_path"):
            payload["request_path"] = getattr(record, "request_path")
        return json.dumps(payload, ensure_ascii=True)


def configure_logging(
    level: str = "INFO",
    *,
    json_format: bool | None = None,
    log_file: str | None = None,
    json_logs: bool | None = None,
) -> None:
    """
    Configure root logging handlers.

    `json_logs` is a backward-compatible alias for `json_format`.
    """
    use_json = bool(json_logs) if json_format is None else bool(json_format)

    formatters = {
        "json": {"()": "bist_quant.observability.logging.JsonLogFormatter"},
        "text": {
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S%z",
        },
    }

    handlers: dict[str, dict[str, Any]] = {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json" if use_json else "text",
        }
    }

    root_handlers = ["console"]
    if log_file:
        file_path = Path(log_file).expanduser().resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(file_path),
            "maxBytes": 10 * 1024 * 1024,
            "backupCount": 5,
            "formatter": "json" if use_json else "text",
            "encoding": "utf-8",
        }
        root_handlers.append("file")

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": formatters,
            "handlers": handlers,
            "root": {"level": level, "handlers": root_handlers},
        }
    )
