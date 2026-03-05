#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8001}"
FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

cleanup() {
  if [[ -n "${BACK_PID:-}" ]]; then
    kill "${BACK_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${FRONT_PID:-}" ]]; then
    kill "${FRONT_PID}" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT INT TERM

echo "[bist] building frontend"
( cd "${ROOT_DIR}/frontend" && npm install >/dev/null )
( cd "${ROOT_DIR}/frontend" && NEXT_PUBLIC_API_URL="http://${BACKEND_HOST}:${BACKEND_PORT}" npm run build )

echo "[bist] starting FastAPI backend"
python -m uvicorn bist_quant.api.main:app --host "${BACKEND_HOST}" --port "${BACKEND_PORT}" &
BACK_PID=$!

echo "[bist] starting Next.js production server"
( cd "${ROOT_DIR}/frontend" && NEXT_PUBLIC_API_URL="http://${BACKEND_HOST}:${BACKEND_PORT}" npm run start -- --hostname "${FRONTEND_HOST}" --port "${FRONTEND_PORT}" ) &
FRONT_PID=$!

wait "${BACK_PID}" "${FRONT_PID}"
