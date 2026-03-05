#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TMP_FILE="$(mktemp)"
trap 'rm -f "$TMP_FILE"' EXIT

if rg --glob '*.tsx' -n "<(pre|select|textarea|table)\\b" src/app >"$TMP_FILE"; then
  echo "[ui-primitives] Found banned raw tags in src/app."
  cat "$TMP_FILE"
  echo
  echo "Use shared components instead: SelectInput, Textarea, DataTable, KeyValueList."
  exit 1
fi

echo "[ui-primitives] PASS"
