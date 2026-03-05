#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# daily_data_refresh.sh — Automated post-market data refresh for BIST Quant
#
# Fetches latest BIST stock prices, index data, and gold prices after market
# close, then warms the borsapy panel cache so the frontend serves current
# data.
#
# Designed to run via systemd timer at 18:45 Istanbul time (Mon-Fri).
# Can also be run manually: bash scripts/daily_data_refresh.sh
#
# Exit codes:
#   0 — all steps succeeded
#   1 — one or more steps failed (check logs/daily_refresh.log)
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
LOG_FILE="${LOG_DIR}/daily_refresh.log"

mkdir -p "${LOG_DIR}"

# ─── Logging helper ──────────────────────────────────────────────────────────

log() {
  local level="$1"; shift
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [${level}] $*" | tee -a "${LOG_FILE}"
}

# ─── Activate Python environment ─────────────────────────────────────────────

if [[ -f "${ROOT_DIR}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.venv/bin/activate"
elif command -v python3 &>/dev/null; then
  log WARN "No .venv found at ${ROOT_DIR}/.venv — using system python3"
else
  log ERROR "No Python environment available"
  exit 1
fi

# Ensure the bist_quant package is importable
python -c "import bist_quant" 2>/dev/null || {
  log WARN "bist_quant not importable — running pip install -e ."
  pip install -e "${ROOT_DIR}" --quiet 2>>"${LOG_FILE}"
}

# ─── Run all refresh steps ───────────────────────────────────────────────────

FAILURES=0

log INFO "════════════════════════════════════════════════════════════"
log INFO "BIST QUANT DAILY DATA REFRESH — $(date '+%Y-%m-%d %H:%M %Z')"
log INFO "════════════════════════════════════════════════════════════"

# Step 1: Update BIST stock prices + XU100 + XAU/TRY (incremental)
log INFO "[1/4] Updating BIST stock prices, XU100, and XAU/TRY..."
if python -m bist_quant.clients.update_prices --source auto 2>&1 | tee -a "${LOG_FILE}"; then
  log INFO "[1/4] ✓ Price update complete"
else
  log ERROR "[1/4] ✗ Price update FAILED (exit $?)"
  FAILURES=$((FAILURES + 1))
fi

# Step 2: Fetch index data (XU030, XU100, XUTUM full history refresh)
log INFO "[2/4] Refreshing index historical data..."
if python -m bist_quant.fetchers.fetch_indices --data-dir "${ROOT_DIR}/data" 2>&1 | tee -a "${LOG_FILE}"; then
  log INFO "[2/4] ✓ Index data refresh complete"
else
  log ERROR "[2/4] ✗ Index data refresh FAILED (exit $?)"
  FAILURES=$((FAILURES + 1))
fi

# Step 3: Fetch gold prices (XAU/TRY cache refresh)
log INFO "[3/4] Refreshing gold price cache..."
if python -m bist_quant.fetchers.fetch_gold_prices 2>&1 | tee -a "${LOG_FILE}"; then
  log INFO "[3/4] ✓ Gold price refresh complete"
else
  log ERROR "[3/4] ✗ Gold price refresh FAILED (exit $?)"
  FAILURES=$((FAILURES + 1))
fi

# Step 4: Warm borsapy panel cache (close_panel, prices_panel)
log INFO "[4/4] Warming borsapy panel cache (XUTUM universe)..."
if python -m bist_quant.cli.cache_cli warm --index XUTUM --period 5y 2>&1 | tee -a "${LOG_FILE}"; then
  log INFO "[4/4] ✓ Panel cache warm complete"
else
  log ERROR "[4/4] ✗ Panel cache warm FAILED (exit $?)"
  FAILURES=$((FAILURES + 1))
fi

# ─── Summary ─────────────────────────────────────────────────────────────────

log INFO "════════════════════════════════════════════════════════════"
if [[ "${FAILURES}" -eq 0 ]]; then
  log INFO "ALL REFRESH STEPS COMPLETED SUCCESSFULLY"
else
  log ERROR "${FAILURES}/4 STEPS FAILED — review ${LOG_FILE}"
fi
log INFO "════════════════════════════════════════════════════════════"

exit "${FAILURES}"
