"""System services for settings persistence, backups, and diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import sys
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd

from bist_quant.settings import ProductionSettings
from bist_quant.security import sanitize_payload

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_label(label: str | None) -> str:
    if not label:
        return ""
    cleaned = "".join(ch for ch in str(label).strip() if ch.isalnum() or ch in {"_", "-", "."})
    return cleaned[:40]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _which(binary: str) -> str | None:
    return shutil.which(binary)


@dataclass(frozen=True)
class BackupManifest:
    filename: str
    path: str
    size_bytes: int
    sha256: str
    created_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "created_at": self.created_at,
        }


class SystemService:
    def __init__(
        self,
        *,
        project_root: Path | None = None,
        app_data_dir: str | None = None,
        settings: ProductionSettings | None = None,
    ):
        self.project_root = (project_root or Path.cwd()).resolve()
        default_data_dir = Path.home() / ".bist-quant-ai"
        raw = app_data_dir or os.getenv("BIST_APP_DATA_DIR", "")
        self.app_data_dir = Path(raw).expanduser().resolve() if raw else default_data_dir
        self.backup_dir = self.app_data_dir / "backups"
        self.settings_path = self.app_data_dir / "settings.json"
        self.diagnostics_dir = self.app_data_dir / "diagnostics"
        self.project_data_dir = self.project_root / "data"
        self.settings = settings or ProductionSettings.from_env()
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.app_data_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)

    def _default_settings(self) -> dict[str, Any]:
        return {
            "first_run_complete": False,
            "theme": "system",
            "language": "en",
            "backend": {
                "base_url": "http://127.0.0.1:8000",
                "timeout_seconds": 30,
            },
            "data": {
                "directory": str(self.project_data_dir),
                "auto_cleanup_days": 30,
                "retention_policy": "local",
            },
            "backup": {
                "enabled": True,
                "frequency_hours": 24,
                "retain_count": 14,
            },
            "telemetry": {
                "enabled": True,
                "crash_reporting": True,
            },
            "updated_at": _iso_now(),
        }

    def load_settings(self) -> dict[str, Any]:
        defaults = self._default_settings()
        if not self.settings_path.exists():
            return defaults
        try:
            raw = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except Exception:
            return defaults
        if not isinstance(raw, dict):
            return defaults
        merged = _deep_merge(defaults, sanitize_payload(raw))
        merged["updated_at"] = str(merged.get("updated_at") or _iso_now())
        return merged

    def save_settings(self, patch: dict[str, Any]) -> dict[str, Any]:
        current = self.load_settings()
        sanitized = sanitize_payload(patch)
        if not isinstance(sanitized, dict):
            raise ValueError("Settings payload must be an object.")
        merged = _deep_merge(current, sanitized)
        merged["updated_at"] = _iso_now()
        self.settings_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        return merged

    def list_backups(self, limit: int = 30) -> list[dict[str, Any]]:
        files = sorted(
            self.backup_dir.glob("*.zip"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        results: list[dict[str, Any]] = []
        for item in files[: max(1, min(limit, 200))]:
            stat = item.stat()
            created = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
            results.append(
                {
                    "filename": item.name,
                    "path": str(item),
                    "size_bytes": stat.st_size,
                    "created_at": created,
                }
            )
        return results

    def _add_optional_file(self, zf: zipfile.ZipFile, source: Path, arcname: str) -> None:
        if source.exists() and source.is_file():
            zf.write(source, arcname=arcname)

    def _add_optional_tree(self, zf: zipfile.ZipFile, source: Path, arc_prefix: str, file_limit: int = 500) -> None:
        if not source.exists() or not source.is_dir():
            return
        count = 0
        for path in source.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(source)
            zf.write(path, arcname=str(Path(arc_prefix) / rel))
            count += 1
            if count >= file_limit:
                break

    def create_backup(self, label: str | None = None) -> dict[str, Any]:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = _safe_label(label)
        filename = f"backup_{stamp}.zip" if not suffix else f"backup_{stamp}_{suffix}.zip"
        target = self.backup_dir / filename

        with zipfile.ZipFile(target, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            self._add_optional_file(zf, self.settings_path, "settings.json")
            self._add_optional_file(zf, self.project_data_dir / "run_store.json", "project_data/run_store.json")
            self._add_optional_file(zf, self.project_data_dir / "signal_store.json", "project_data/signal_store.json")
            self._add_optional_tree(
                zf,
                self.project_data_dir / "artifacts",
                "project_data/artifacts",
                file_limit=2_000,
            )

        manifest = BackupManifest(
            filename=target.name,
            path=str(target),
            size_bytes=target.stat().st_size,
            sha256=_sha256_file(target),
            created_at=_iso_now(),
        )
        return manifest.as_dict()

    def restore_backup(self, filename: str) -> dict[str, Any]:
        if not filename or "/" in filename or "\\" in filename:
            raise ValueError("Backup filename is invalid.")

        source = (self.backup_dir / filename).resolve()
        if not source.exists():
            raise FileNotFoundError(f"Backup not found: {filename}")

        self._ensure_dirs()
        restored_files: list[str] = []

        with TemporaryDirectory(prefix="bist_restore_") as tmp:
            tmp_dir = Path(tmp)
            with zipfile.ZipFile(source, mode="r") as zf:
                zf.extractall(tmp_dir)

            restored_files.extend(self._restore_extracted(tmp_dir))

        return {
            "backup": filename,
            "restored_files": restored_files,
            "restored_at": _iso_now(),
        }

    def _restore_extracted(self, extracted_root: Path) -> list[str]:
        restored: list[str] = []

        src_settings = extracted_root / "settings.json"
        if src_settings.exists() and src_settings.is_file():
            self.settings_path.write_bytes(src_settings.read_bytes())
            restored.append(str(self.settings_path))

        project_data_root = extracted_root / "project_data"
        if project_data_root.exists() and project_data_root.is_dir():
            self.project_data_dir.mkdir(parents=True, exist_ok=True)
            for item in project_data_root.rglob("*"):
                if not item.is_file():
                    continue
                rel = item.relative_to(project_data_root)
                target = (self.project_data_dir / rel).resolve()
                if not str(target).startswith(str(self.project_data_dir.resolve())):
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(item.read_bytes())
                restored.append(str(target))

        return restored

    def resource_usage(self) -> dict[str, Any]:
        disk = shutil.disk_usage(self.app_data_dir)
        payload: dict[str, Any] = {
            "cpu": {"load_average_1m": None, "logical_cores": os.cpu_count()},
            "memory": {"total_bytes": None, "available_bytes": None, "process_rss_bytes": None},
            "disk": {
                "path": str(self.app_data_dir),
                "total_bytes": disk.total,
                "used_bytes": disk.used,
                "free_bytes": disk.free,
            },
            "timestamp": _iso_now(),
        }

        if hasattr(os, "getloadavg"):
            try:
                payload["cpu"]["load_average_1m"] = float(os.getloadavg()[0])
            except Exception:
                payload["cpu"]["load_average_1m"] = None

        if psutil is not None:
            try:
                vm = psutil.virtual_memory()
                payload["memory"]["total_bytes"] = int(vm.total)
                payload["memory"]["available_bytes"] = int(vm.available)
                process = psutil.Process(os.getpid())
                payload["memory"]["process_rss_bytes"] = int(process.memory_info().rss)
            except Exception:
                pass

        return payload

    def requirements_report(self) -> dict[str, Any]:
        usage = self.resource_usage()
        free_disk = int(usage.get("disk", {}).get("free_bytes") or 0)
        total_memory = int(usage.get("memory", {}).get("total_bytes") or 0)

        checks = [
            {
                "name": "python_version",
                "required": ">=3.11",
                "actual": ".".join(str(part) for part in sys.version_info[:3]),
                "ok": sys.version_info >= (3, 11),
            },
            {
                "name": "cpu_cores",
                "required": ">=2",
                "actual": int(os.cpu_count() or 0),
                "ok": int(os.cpu_count() or 0) >= 2,
            },
            {
                "name": "free_disk",
                "required": ">=2GB",
                "actual": free_disk,
                "ok": free_disk >= 2 * 1024 * 1024 * 1024,
            },
            {
                "name": "total_memory",
                "required": ">=4GB (recommended)",
                "actual": total_memory if total_memory > 0 else "unknown",
                "ok": total_memory == 0 or total_memory >= 4 * 1024 * 1024 * 1024,
            },
            {
                "name": "node_runtime",
                "required": "node on PATH for desktop packaging",
                "actual": _which("node") or "missing",
                "ok": bool(_which("node")),
            },
        ]

        return {
            "ok": all(bool(item["ok"]) for item in checks),
            "checks": checks,
            "timestamp": _iso_now(),
        }

    @staticmethod
    def _json_safe_value(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (pd.Timestamp, datetime, date)):
            return value.isoformat()
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        return value

    @classmethod
    def _frame_to_records(cls, frame: pd.DataFrame) -> list[dict[str, Any]]:
        if frame.empty:
            return []
        records: list[dict[str, Any]] = []
        for row in frame.to_dict(orient="records"):
            records.append({str(key): cls._json_safe_value(value) for key, value in row.items()})
        return records

    def get_macro_calendar(
        self,
        period: str = "1w",
        country: str | list[str] | None = None,
        importance: str | None = None,
    ) -> dict[str, Any]:
        """Return live economic-calendar payload for API clients."""
        from bist_quant.common.data_loader import DataLoader

        try:
            loader = DataLoader(data_dir=self.project_data_dir)
            events = loader.economic_calendar.get_events(
                period=period,
                country=country,
                importance=importance,
            )
        except Exception as exc:
            return {
                "timestamp": _iso_now(),
                "period": period,
                "country": country,
                "importance": importance,
                "count": 0,
                "events": [],
                "error": str(exc),
            }

        return {
            "timestamp": _iso_now(),
            "period": period,
            "country": country,
            "importance": importance,
            "count": int(len(events)),
            "events": self._frame_to_records(events),
        }

    def diagnostics_snapshot(self) -> dict[str, Any]:
        payload = {
            "timestamp": _iso_now(),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "python": sys.version,
            },
            "paths": {
                "project_root": str(self.project_root),
                "app_data_dir": str(self.app_data_dir),
                "project_data_dir": str(self.project_data_dir),
            },
            "settings": self.load_settings(),
            "resource_usage": self.resource_usage(),
            "requirements": self.requirements_report(),
            "tools": {
                "python": sys.executable,
                "node": _which("node"),
                "npm": _which("npm"),
                "uvicorn": _which("uvicorn"),
            },
        }
        return payload

    def write_diagnostics_report(self) -> dict[str, Any]:
        snapshot = self.diagnostics_snapshot()
        filename = f"diagnostics_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        target = self.diagnostics_dir / filename
        target.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        return {
            "report_file": str(target),
            "report_name": filename,
            "generated_at": _iso_now(),
        }


# Backward-compatible alias for API shim imports.
ProductionSystemService = SystemService


__all__ = ["BackupManifest", "SystemService", "ProductionSystemService"]
