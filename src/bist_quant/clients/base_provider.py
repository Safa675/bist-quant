"""Base class for Borsa data providers.

Manages caching initialization and dynamic borsapy imports uniformly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from bist_quant.clients.utils import (
    as_frame,
    call_if_callable,
    get_borsapy_module,
    pick_column,
    to_float,
)


class BaseProvider:
    """Base class for all domain-specific synchronous data providers wrapping borsapy."""

    _to_float = staticmethod(to_float)
    _as_frame = staticmethod(as_frame)
    _pick_column = staticmethod(pick_column)
    _call_if_callable = staticmethod(call_if_callable)

    def __init__(
        self,
        borsapy_module: Any | None = None,
        cache_dir: Any = None,
    ) -> None:
        self._bp = borsapy_module
        self._import_attempted = borsapy_module is not None
        self._disk_cache: Any | None = None
        if cache_dir is not None:
            try:
                self._disk_cache = self._get_disk_cache(cache_dir)
            except Exception:
                pass

    @staticmethod
    def _get_disk_cache(cache_dir: Any) -> Any | None:
        from bist_quant.common.disk_cache import DiskCache
        return DiskCache(Path(cache_dir))

    def _get_bp(self) -> Any | None:
        if self._bp is not None:
            return self._bp
        if self._import_attempted:
            return None
        self._bp, self._import_attempted = get_borsapy_module(self.__class__.__name__, self._import_attempted)
        return self._bp
