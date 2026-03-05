#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8001}"
FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
RUN_E2E="${RUN_E2E:-1}"

STAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
ARTIFACT_DIR="${ROOT_DIR}/docs/plans/artifacts/phase5-smoke-${STAMP}"
mkdir -p "${ARTIFACT_DIR}"

BACKEND_LOG="${ARTIFACT_DIR}/backend.log"
FRONTEND_LOG="${ARTIFACT_DIR}/frontend.log"
API_SMOKE_LOG="${ARTIFACT_DIR}/api-smoke.log"
E2E_LOG="${ARTIFACT_DIR}/e2e-smoke.log"
SUMMARY_LOG="${ARTIFACT_DIR}/summary.log"

cleanup() {
  if [[ -n "${FRONT_PID:-}" ]]; then
    kill "${FRONT_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${BACK_PID:-}" ]]; then
    kill "${BACK_PID}" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT INT TERM

wait_for_url() {
  local url="$1"
  local name="$2"
  local retries="${3:-60}"
  local delay="${4:-2}"

  for _ in $(seq 1 "${retries}"); do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      echo "[smoke] ${name} ready: ${url}" | tee -a "${SUMMARY_LOG}"
      return 0
    fi
    sleep "${delay}"
  done

  echo "[smoke] ${name} failed to become ready: ${url}" | tee -a "${SUMMARY_LOG}"
  return 1
}

echo "[smoke] artifact_dir=${ARTIFACT_DIR}" | tee "${SUMMARY_LOG}"
echo "[smoke] building frontend" | tee -a "${SUMMARY_LOG}"
NEXT_PUBLIC_API_URL="http://${BACKEND_HOST}:${BACKEND_PORT}" npm --prefix "${ROOT_DIR}/frontend" run build >>"${SUMMARY_LOG}" 2>&1

echo "[smoke] starting backend" | tee -a "${SUMMARY_LOG}"
(
  cd "${ROOT_DIR}"
  python -m uvicorn bist_quant.api.main:app --host "${BACKEND_HOST}" --port "${BACKEND_PORT}" >"${BACKEND_LOG}" 2>&1
) &
BACK_PID=$!

echo "[smoke] starting frontend" | tee -a "${SUMMARY_LOG}"
(
  cd "${ROOT_DIR}"
  NEXT_PUBLIC_API_URL="http://${BACKEND_HOST}:${BACKEND_PORT}" npm --prefix frontend run start -- --hostname "${FRONTEND_HOST}" --port "${FRONTEND_PORT}" >"${FRONTEND_LOG}" 2>&1
) &
FRONT_PID=$!

wait_for_url "http://${BACKEND_HOST}:${BACKEND_PORT}/api/health/live" "backend" 90 2
wait_for_url "http://${FRONTEND_HOST}:${FRONTEND_PORT}/dashboard" "frontend" 90 2

echo "[smoke] running API smoke suite" | tee -a "${SUMMARY_LOG}"
(
  cd "${ROOT_DIR}"
  SMOKE_API_URL="http://${BACKEND_HOST}:${BACKEND_PORT}" \
    python scripts/release/run_api_smoke.py
) | tee "${API_SMOKE_LOG}"

if [[ "${RUN_E2E}" == "1" ]]; then
  echo "[smoke] running real-backend Playwright smoke" | tee -a "${SUMMARY_LOG}"
  (
    cd "${ROOT_DIR}/frontend"
    PLAYWRIGHT_EXTERNAL_BASE_URL="http://${FRONTEND_HOST}:${FRONTEND_PORT}" \
      NEXT_PUBLIC_API_URL="http://${BACKEND_HOST}:${BACKEND_PORT}" \
      npm run test:e2e -- e2e/navigation.spec.ts e2e/critical-flows.spec.ts
  ) | tee "${E2E_LOG}"
fi

echo "[smoke] complete" | tee -a "${SUMMARY_LOG}"
echo "[smoke] logs:" | tee -a "${SUMMARY_LOG}"
echo "  - ${BACKEND_LOG}" | tee -a "${SUMMARY_LOG}"
echo "  - ${FRONTEND_LOG}" | tee -a "${SUMMARY_LOG}"
echo "  - ${API_SMOKE_LOG}" | tee -a "${SUMMARY_LOG}"
if [[ "${RUN_E2E}" == "1" ]]; then
  echo "  - ${E2E_LOG}" | tee -a "${SUMMARY_LOG}"
fi

echo "  - ${SUMMARY_LOG}" | tee -a "${SUMMARY_LOG}"
