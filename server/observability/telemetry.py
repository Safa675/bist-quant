"""Crash telemetry helpers."""

from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def emit_crash_telemetry(
    exception: Exception | None = None,
    context: dict[str, Any] | None = None,
    endpoint: str | None = None,
    *,
    enabled: bool = True,
    telemetry_file: str | None = None,
    error: Exception | None = None,
) -> bool:
    """
    Emit crash telemetry for error tracking.

    Returns True if telemetry was written successfully.
    Returns False if telemetry is disabled or writing fails.
    """
    if not enabled:
        return False

    crash = error or exception
    if crash is None:
        return False

    if crash.__traceback__:
        trace = "".join(traceback.format_exception(type(crash), crash, crash.__traceback__))
    else:
        # Preserve legacy behavior when called outside an active exception context.
        trace = traceback.format_exc()

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "error_type": type(crash).__name__,
        "error_message": str(crash),
        "traceback": trace,
        "context": context or {},
    }

    target_path = telemetry_file or endpoint
    target = Path(target_path).expanduser().resolve() if target_path else Path.cwd() / "data" / "crash_telemetry.jsonl"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=True))
            fh.write("\n")
    except Exception:
        return False
    return True
