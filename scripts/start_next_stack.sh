#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8001}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

if [[ ! -d "${ROOT_DIR}/frontend" ]]; then
  echo "frontend directory not found at ${ROOT_DIR}/frontend"
  exit 1
fi

if [[ ! -f "${ROOT_DIR}/frontend/package.json" ]]; then
  echo "frontend/package.json not found"
  exit 1
fi

cleanup() {
  if [[ -n "${BACK_PID:-}" ]]; then
    kill "${BACK_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${FRONT_PID:-}" ]]; then
    kill "${FRONT_PID}" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT INT TERM

echo "[bist] starting FastAPI backend on ${BACKEND_HOST}:${BACKEND_PORT}"
python -m uvicorn bist_quant.api.main:app --host "${BACKEND_HOST}" --port "${BACKEND_PORT}" --reload &
BACK_PID=$!

echo "[bist] ensuring frontend dependencies"
( cd "${ROOT_DIR}/frontend" && npm install >/dev/null )

echo "[bist] starting Next.js dev server on 0.0.0.0:${FRONTEND_PORT}"
( cd "${ROOT_DIR}/frontend" && NEXT_PUBLIC_API_URL="http://${BACKEND_HOST}:${BACKEND_PORT}" npm run dev -- --hostname 0.0.0.0 --port "${FRONTEND_PORT}" ) &
FRONT_PID=$!

wait "${BACK_PID}" "${FRONT_PID}"
