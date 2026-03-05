#!/usr/bin/env python3
"""Check or update the committed OpenAPI snapshot."""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_PATH = PROJECT_ROOT / "docs" / "plans" / "artifacts" / "openapi.snapshot.json"


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n"


def _generate_openapi_snapshot_text() -> str:
    from bist_quant.api.main import create_app

    app = create_app()
    return _canonical_json(app.openapi())


def _write_snapshot(text: str) -> None:
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(text, encoding="utf-8")


def _print_diff(expected: str, actual: str) -> None:
    diff = difflib.unified_diff(
        expected.splitlines(),
        actual.splitlines(),
        fromfile=str(SNAPSHOT_PATH),
        tofile="generated_openapi",
        lineterm="",
    )
    for line in diff:
        print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify committed OpenAPI snapshot.")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Regenerate and overwrite the committed snapshot.",
    )
    args = parser.parse_args()

    generated = _generate_openapi_snapshot_text()

    if args.update:
        _write_snapshot(generated)
        print(f"openapi_snapshot=UPDATED path={SNAPSHOT_PATH}")
        return 0

    if not SNAPSHOT_PATH.exists():
        print(f"openapi_snapshot=FAIL missing snapshot: {SNAPSHOT_PATH}")
        print("Run: python scripts/contract/check_openapi_snapshot.py --update")
        return 1

    expected = SNAPSHOT_PATH.read_text(encoding="utf-8")
    if expected == generated:
        print(f"openapi_snapshot=PASS path={SNAPSHOT_PATH}")
        return 0

    print("openapi_snapshot=FAIL snapshot drift detected")
    _print_diff(expected, generated)
    print("Update snapshot intentionally with:")
    print("python scripts/contract/check_openapi_snapshot.py --update")
    return 1


if __name__ == "__main__":
    sys.exit(main())
