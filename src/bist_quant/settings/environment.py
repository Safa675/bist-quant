"""Environment variable parsing helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping


def parse_env_str(name: str, default: str = "", *, environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = source.get(name)
    if value is None:
        return default
    return str(value).strip()


def parse_env_bool(
    name: str,
    default: bool = False,
    *,
    environ: Mapping[str, str] | None = None,
) -> bool:
    raw = parse_env_str(name, "", environ=environ)
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on", "y"}


def parse_env_int(
    name: str,
    default: int,
    minimum: int,
    maximum: int,
    *,
    environ: Mapping[str, str] | None = None,
) -> int:
    raw = parse_env_str(name, "", environ=environ)
    if not raw:
        return default
    try:
        parsed = int(raw)
    except Exception:
        return default
    return max(minimum, min(parsed, maximum))


def parse_env_list(name: str, *, environ: Mapping[str, str] | None = None) -> list[str]:
    raw = parse_env_str(name, "", environ=environ)
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]
