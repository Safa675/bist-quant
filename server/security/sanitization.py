"""Payload sanitization helpers."""

from __future__ import annotations

import re
from typing import Any


_CTRL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


def sanitize_payload(value: Any, *, max_string_length: int = 1_000) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        cleaned = _CTRL_CHAR_RE.sub("", value).strip()
        if len(cleaned) > max_string_length:
            return cleaned[:max_string_length]
        return cleaned
    if isinstance(value, list):
        return [sanitize_payload(item, max_string_length=max_string_length) for item in value]
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for k, v in value.items():
            key = str(k).strip()
            if not key:
                continue
            sanitized[key[:128]] = sanitize_payload(v, max_string_length=max_string_length)
        return sanitized
    # Non-serializable objects are stringified to avoid runtime surprises.
    return str(value)
