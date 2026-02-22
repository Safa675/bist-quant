"""
Persistent disk cache for borsapy data with per-category TTL.

Each cached item consists of:
- A data file (Parquet preferred, CSV.GZ fallback if pyarrow is unavailable)
- A ``.meta.json`` sidecar with creation time, expiry, source, and row count.

Design principles:
- Thread-safe (file-level locking via ``threading.Lock``)
- Graceful degradation on corrupt / missing files
- Categories map to subdirectories under ``cache_dir``
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd

from bist_quant.common.cache_config import CacheTTL

logger = logging.getLogger(__name__)

_META_SUFFIX = ".meta.json"

# Detect parquet engine availability at import time.
_PARQUET_AVAILABLE = False
try:
    import io as _io

    pd.DataFrame({"_t": [1]}).to_parquet(_io.BytesIO(), index=False)
    _PARQUET_AVAILABLE = True
except Exception:
    pass

# Preferred DataFrame extension.
_DF_EXT = ".parquet" if _PARQUET_AVAILABLE else ".csv.gz"
# When reading, try both extensions (handles cache created with a different engine).
_DF_READ_EXTS = [".parquet", ".csv.gz"] if _PARQUET_AVAILABLE else [".csv.gz", ".parquet"]


class DiskCache:
    """Thread-safe, TTL-aware disk cache backed by Parquet/CSV + JSON.

    Args:
        cache_dir: Root directory for all cached data.
        ttl: Per-category TTL configuration.
    """

    def __init__(
        self,
        cache_dir: Path,
        ttl: CacheTTL | None = None,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._ttl = ttl or CacheTTL.from_env()
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_dataframe(self, category: str, key: str) -> pd.DataFrame | None:
        """Return a cached DataFrame, or *None* if missing / expired."""
        for ext in _DF_READ_EXTS:
            path = self._data_path(category, key, ext=ext)
            if not self._is_valid(category, key, path):
                continue
            try:
                if ext == ".parquet":
                    return pd.read_parquet(path)
                else:
                    return pd.read_csv(path, compression="gzip")
            except Exception as exc:
                logger.warning("Cache read error for %s/%s: %s", category, key, exc)
                self._remove(path)
        return None

    def set_dataframe(
        self,
        category: str,
        key: str,
        data: pd.DataFrame,
        ttl_seconds: int | None = None,
        source: str = "borsapy",
    ) -> None:
        """Persist a DataFrame to disk with metadata sidecar."""
        if data is None or data.empty:
            return
        path = self._data_path(category, key, ext=_DF_EXT)
        path.parent.mkdir(parents=True, exist_ok=True)
        resolved_ttl = ttl_seconds if ttl_seconds is not None else self._ttl.ttl_for(category)
        with self._lock:
            if _PARQUET_AVAILABLE:
                data.to_parquet(path, index=True)
            else:
                data.to_csv(path, index=True, compression="gzip")
            self._write_meta(path, resolved_ttl, source, row_count=len(data))

    def get_json(self, category: str, key: str) -> Any | None:
        """Return cached JSON-serializable data, or *None*."""
        path = self._data_path(category, key, ext=".json")
        if not self._is_valid(category, key, path):
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Cache read error for %s/%s: %s", category, key, exc)
            self._remove(path)
            return None

    def set_json(
        self,
        category: str,
        key: str,
        data: Any,
        ttl_seconds: int | None = None,
        source: str = "borsapy",
    ) -> None:
        """Persist JSON-serializable data to disk."""
        if data is None:
            return
        path = self._data_path(category, key, ext=".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        resolved_ttl = ttl_seconds if ttl_seconds is not None else self._ttl.ttl_for(category)
        row_count = len(data) if isinstance(data, (list, dict)) else 0
        with self._lock:
            path.write_text(json.dumps(data, default=str, ensure_ascii=False), encoding="utf-8")
            self._write_meta(path, resolved_ttl, source, row_count=row_count)

    def is_valid(self, category: str, key: str) -> bool:
        """Check if an entry exists and hasn't expired (any extension)."""
        for ext in [*_DF_READ_EXTS, ".json"]:
            path = self._data_path(category, key, ext=ext)
            if self._is_valid(category, key, path):
                return True
        return False

    def clear(self, category: str | None = None) -> int:
        """Remove cached entries.  Returns number of files removed.

        Args:
            category: If given, only clear that subdirectory.
                      If *None*, wipe everything.
        """
        removed = 0
        with self._lock:
            if category is not None:
                target = self._cache_dir / category
                if target.exists():
                    removed = sum(1 for _ in target.rglob("*") if _.is_file())
                    shutil.rmtree(target)
            else:
                for child in self._cache_dir.iterdir():
                    if child.is_dir():
                        removed += sum(1 for _ in child.rglob("*") if _.is_file())
                        shutil.rmtree(child)
        logger.info("Cache cleared: %d files removed (category=%s)", removed, category)
        return removed

    def clear_expired(self) -> int:
        """Remove only expired entries.  Returns number removed."""
        removed = 0
        now = datetime.now(timezone.utc)
        with self._lock:
            for meta_path in self._cache_dir.rglob(f"*{_META_SUFFIX}"):
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    expires = datetime.fromisoformat(meta["expires_at"])
                    if now >= expires:
                        # Find the actual data file (could be any extension)
                        for ext in [*_DF_READ_EXTS, ".json"]:
                            candidate = meta_path.parent / (
                                meta_path.stem.replace(".meta", "") + ext
                            )
                            if candidate.exists():
                                candidate.unlink()
                                removed += 1
                        meta_path.unlink()
                        removed += 1
                except Exception:
                    continue
        logger.info("Cleared %d expired cache files", removed)
        return removed

    def inspect(self) -> dict[str, list[dict[str, Any]]]:
        """Return a summary of all cached entries grouped by category."""
        result: dict[str, list[dict[str, Any]]] = {}
        now = datetime.now(timezone.utc)
        for category_dir in sorted(self._cache_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            category = category_dir.name
            entries: list[dict[str, Any]] = []
            for meta_path in sorted(category_dir.glob(f"*{_META_SUFFIX}")):
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    expires = datetime.fromisoformat(meta["expires_at"])
                    meta["expired"] = now >= expires
                    meta["key"] = meta_path.stem.replace(".meta", "")
                    entries.append(meta)
                except Exception:
                    continue
            if entries:
                result[category] = entries
        return result

    # ------------------------------------------------------------------
    # Helpers for batch operations
    # ------------------------------------------------------------------

    def partition_symbols(
        self,
        category: str,
        symbols: list[str],
    ) -> tuple[pd.DataFrame, list[str]]:
        """Split symbols into cached (valid) and missing.

        Returns:
            (cached_dataframe, missing_symbols)
        """
        cached_frames: list[pd.DataFrame] = []
        missing: list[str] = []
        for symbol in symbols:
            frame = self.get_dataframe(category, symbol)
            if frame is not None and not frame.empty:
                cached_frames.append(frame)
            else:
                missing.append(symbol)
        combined = pd.concat(cached_frames, ignore_index=True) if cached_frames else pd.DataFrame()
        return combined, missing

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _data_path(self, category: str, key: str, ext: str = ".parquet") -> Path:
        return self._cache_dir / category / f"{key}{ext}"

    def _meta_path(self, data_path: Path) -> Path:
        return data_path.with_name(data_path.stem + _META_SUFFIX)

    def _is_valid(self, category: str, key: str, data_path: Path) -> bool:
        if not data_path.exists():
            return False
        meta_path = self._meta_path(data_path)
        if not meta_path.exists():
            return False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            expires_at = datetime.fromisoformat(meta["expires_at"])
            return datetime.now(timezone.utc) < expires_at
        except Exception:
            return False

    def _write_meta(
        self,
        data_path: Path,
        ttl_seconds: int,
        source: str,
        row_count: int = 0,
    ) -> None:
        now = datetime.now(timezone.utc)
        meta = {
            "created_at": now.isoformat(),
            "expires_at": (
                now + timedelta(seconds=ttl_seconds)
            ).isoformat(),
            "ttl_seconds": ttl_seconds,
            "source": source,
            "row_count": row_count,
        }
        meta_path = self._meta_path(data_path)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _remove(self, data_path: Path) -> None:
        """Remove a data file and its meta sidecar if they exist."""
        try:
            if data_path.exists():
                data_path.unlink()
            meta = self._meta_path(data_path)
            if meta.exists():
                meta.unlink()
        except Exception:
            pass
