#!/usr/bin/env bash
# unified script to update all bist quantitative dependencies
set -e

# Change down to project root
cd "$(dirname "$0")/.."

echo "============================================================"
echo " Starting Data Fetching Pipeline (Borsapy)"
echo "============================================================"

echo ""
echo "[1/3] Updating Prices via Borsapy..."
PYTHONPATH=src python src/bist_quant/fetcher/update_prices.py "$@"
echo "  ↳ Price update complete."

echo ""
echo "[2/3] Updating Macro Indicators (TCMB)..."
PYTHONPATH=src python src/bist_quant/fetcher/tcmb_data_fetcher.py
echo "  ↳ Macro update complete."

echo ""
echo "[3/3] Updating Fundamentals (ISYatirım)..."
PYTHONPATH=src python src/bist_quant/fetcher/fetch_integrate_fundamentals.py --allow-stale-override "$@"
echo "  ↳ Fundamentals update complete."

echo ""
echo "============================================================"
echo " Pipeline Complete ✅"
echo "============================================================"
