#!/usr/bin/env python3
"""Verify Phase 0 contract-freeze artifacts."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import re
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "docs" / "plans" / "artifacts" / "phase0-contract-manifest.json"
SAMPLES_PATH = PROJECT_ROOT / "docs" / "plans" / "artifacts" / "phase0-canonical-samples.json"

REQUIRED_ROUTE_FIELDS = {
    "route",
    "page",
    "streamlit_source",
    "frontend_calls",
    "endpoints",
    "execution_mode",
    "status",
}

REQUIRED_ENDPOINT_FIELDS = {
    "method",
    "path",
    "request_schema",
    "response_keys",
    "status_codes",
    "error_contract",
    "sample_refs",
}

REQUIRED_DTOS = {
    "BacktestUiResult",
    "AnalyticsUiResult",
    "OptimizationUiResult",
    "ComplianceUiResult",
    "ScreenerUiResult",
}


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _route_from_page_file(path: Path) -> str:
    rel = path.relative_to(PROJECT_ROOT / "frontend" / "src" / "app")
    if rel.as_posix() == "page.tsx":
        return "/"
    return "/" + rel.parent.as_posix()


def _normalize_path(path: str) -> str:
    # Strip query string and normalize any placeholder name to {param}
    clean = path.split("?", 1)[0]
    return re.sub(r"\{[^}]+\}", "{param}", clean)


def _normalize_endpoint_key(method: str, path: str) -> str:
    return f"{method.upper()} {_normalize_path(path)}"


def _extract_consumed_endpoints() -> set[str]:
    api_ts = (PROJECT_ROOT / "frontend" / "src" / "lib" / "api.ts").read_text(encoding="utf-8")

    endpoint_methods: set[str] = set()
    for line in api_ts.splitlines():
        if "fetchJson" not in line and "postJson" not in line:
            continue
        match = re.search(r"([`'\"])(/api[^`'\"]*)\\1", line)
        if not match:
            continue
        raw_path = match.group(2)
        path = re.sub(r"\$\{[^}]+\}", "{param}", raw_path.split("?", 1)[0])
        if "postJson" in line:
            method = "POST"
        elif "method: \"DELETE\"" in line:
            method = "DELETE"
        else:
            method = "GET"
        endpoint_methods.add(_normalize_endpoint_key(method, path))

    # AppShell dependency
    endpoint_methods.add(_normalize_endpoint_key("GET", "/api/dashboard/overview"))
    return endpoint_methods


def _verify() -> tuple[bool, list[str]]:
    manifest = _load_json(MANIFEST_PATH)
    samples = _load_json(SAMPLES_PATH)

    errors: list[str] = []

    # 1) Any Next route missing
    page_files = sorted((PROJECT_ROOT / "frontend" / "src" / "app").glob("**/page.tsx"))
    expected_routes = {_route_from_page_file(path) for path in page_files}
    manifest_routes = {row.get("route") for row in manifest.get("routes", [])}
    missing_routes = sorted(expected_routes - manifest_routes)
    if missing_routes:
        errors.append(f"missing routes in manifest: {missing_routes}")

    # Validate route schema
    for row in manifest.get("routes", []):
        missing = REQUIRED_ROUTE_FIELDS - set(row.keys())
        if missing:
            errors.append(f"route entry missing fields for {row.get('route')}: {sorted(missing)}")

    # 2) Any consumed endpoint missing
    consumed = _extract_consumed_endpoints()
    manifest_endpoints = {
        _normalize_endpoint_key(row.get("method", ""), row.get("path", ""))
        for row in manifest.get("endpoints", [])
    }
    missing_endpoints = sorted(consumed - manifest_endpoints)
    if missing_endpoints:
        errors.append(f"missing consumed endpoints in manifest: {missing_endpoints}")

    # Validate endpoint schema
    for row in manifest.get("endpoints", []):
        missing = REQUIRED_ENDPOINT_FIELDS - set(row.keys())
        if missing:
            key = f"{row.get('method')} {row.get('path')}"
            errors.append(f"endpoint entry missing fields for {key}: {sorted(missing)}")

    # 3) Any DTO mapping unresolved
    dto_rows = manifest.get("dto_mappings", [])
    present_dtos = {row.get("ui_dto") for row in dto_rows}
    missing_dtos = sorted(REQUIRED_DTOS - present_dtos)
    if missing_dtos:
        errors.append(f"missing required dto mappings: {missing_dtos}")

    for row in dto_rows:
        for field in ("ui_dto", "backend_field", "ui_field", "transform", "notes"):
            if not row.get(field):
                errors.append(f"dto mapping has empty {field}: {row}")

    # 4) Unknown-field register must be empty
    unknown = manifest.get("unknown_field_register", [])
    if unknown:
        errors.append("unknown_field_register is not empty")

    # 5) Canonical sample missing for consumed endpoint
    sample_keys = {
        _normalize_endpoint_key(*key.split(" ", 1))
        for key in samples.get("samples", {}).keys()
        if " " in key
    }
    missing_samples = sorted(consumed - sample_keys)
    if missing_samples:
        errors.append(f"missing canonical samples for consumed endpoints: {missing_samples}")

    # Additional required job error samples
    required_error_samples = {
        "POST /api/jobs::unsupported_kind",
        "POST /api/jobs::invalid_backtest_payload",
        "POST /api/jobs::optimize_missing_parameter_space",
    }
    existing_error_samples = set(samples.get("errors", {}).keys())
    missing_error_samples = sorted(required_error_samples - existing_error_samples)
    if missing_error_samples:
        errors.append(f"missing required job error samples: {missing_error_samples}")

    ok = len(errors) == 0
    return ok, errors


def main() -> None:
    started_at = datetime.now(UTC).isoformat()
    ok, errors = _verify()
    finished_at = datetime.now(UTC).isoformat()

    print(f"phase0_verify_started_at={started_at}")
    print(f"phase0_verify_finished_at={finished_at}")

    if ok:
        print("phase0_verify_status=PASS")
        return

    print("phase0_verify_status=FAIL")
    for idx, err in enumerate(errors, start=1):
        print(f"[{idx}] {err}")
    sys.exit(1)


if __name__ == "__main__":
    main()
