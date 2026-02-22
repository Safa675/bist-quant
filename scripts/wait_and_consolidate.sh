#!/usr/bin/env bash

echo "Monitoring bist_quant.cli.cache_cli warm process..."
# Wait while the cache warming script is still running
while pgrep -f "cache_cli warm --index XUTUM --period max" > /dev/null; do
    sleep 30
done

echo "Cache warming complete. Proceeding with cache consolidation..."
PYTHONPATH=src python -m bist_quant.cli.cache_cli consolidate
echo "Consolidation complete!"
