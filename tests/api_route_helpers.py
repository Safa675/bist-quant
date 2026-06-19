"""Helpers for FastAPI route assertions."""

from __future__ import annotations

from typing import Any


def collect_route_paths(routes: list[Any]) -> set[str]:
    paths: set[str] = set()
    for route in routes:
        path = getattr(route, "path", None)
        if isinstance(path, str):
            paths.add(path)

        original_router = getattr(route, "original_router", None)
        if original_router is not None:
            paths.update(collect_route_paths(list(original_router.routes)))
            continue

        nested = getattr(route, "routes", None)
        if nested:
            paths.update(collect_route_paths(list(nested)))
    return paths
