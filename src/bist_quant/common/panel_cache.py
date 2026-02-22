from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd


def _normalize_cache_param(value: Any) -> Any:
    """Convert cache params into deterministic JSON-serializable shapes."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Index):
        return [str(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _normalize_cache_param(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize_cache_param(v) for v in value]
    return value


class PanelCache:
    """Session-scoped LRU cache for computed Date x Ticker panels."""

    def __init__(self, max_entries: int = 32) -> None:
        self.max_entries = max(1, int(max_entries))
        self._store: OrderedDict[str, pd.DataFrame] = OrderedDict()

    def make_key(self, panel_name: str, **params: Any) -> str:
        payload = {
            "panel": str(panel_name),
            "params": _normalize_cache_param(params),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        digest = hashlib.sha1(raw).hexdigest()
        return f"{panel_name}:{digest}"

    def get(self, key: str) -> pd.DataFrame | None:
        value = self._store.pop(key, None)
        if value is None:
            return None
        self._store[key] = value
        return value

    def set(self, key: str, panel: pd.DataFrame) -> None:
        if key in self._store:
            self._store.pop(key, None)
        self._store[key] = panel
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
