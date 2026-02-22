#!/usr/bin/env bash
# unified script to execute and verify all data fetcher clients
set -e

# Change down to project root
cd "$(dirname "$0")/.."

echo "============================================================"
echo " Starting Market Data Fetchers Pipeline"
echo "============================================================"

echo ""
echo "Running All Data Clients..."
PYTHONPATH=src python scripts/run_all_clients.py "$@"

echo ""
echo "============================================================"
echo " Data Clients Initialization Complete ✅"
echo "============================================================"
