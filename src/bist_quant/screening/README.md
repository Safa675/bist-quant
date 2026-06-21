# Multi-dimensional BIST stock screener

The `bist_quant.screening` package provides a local/hybrid/İş Yatırım stock screener
with template presets, numeric filters, percentile filters, and optional technical scans.

## Quick start

```python
from bist_quant.screening import get_screener_metadata, run_screener

meta = get_screener_metadata()
result = run_screener({
    "index": "XU100",
    "template": "high_dividend",
    "limit": 25,
    "sort_by": "upside_potential",
})
```

## CLI

```bash
bist-quant screener metadata
bist-quant screener run --index XU100 --template high_dividend --limit 25
bist-quant screener run --index XU100 --technical-scan oversold --json
bist-quant scan --template oversold --universe XU100
bist-quant scan --condition "rsi < 30" --symbols THYAO,GARAN
```

## Technical scans

Predefined scan names live in `bist_quant.clients.technical_scan.PREDEFINED_SCANS`.
Use them in screener payloads (`technical_scan`, `technical_scan_name`) or via
`DataLoader.technical_scan()` / `bist-quant scan`.

## Data sources

| Source | Description |
|--------|-------------|
| `local` | Build universe from cached price + fundamentals panels |
| `isyatirim` | İş Yatırım screener via borsapy |
| `hybrid` | Local base metrics + İş Yatırım analyst/foreign enrichment |

Set `BIST_DATA_DIR` (or use default user-scoped data paths) before running local/hybrid modes.

## Deprecated aliases

`run_stock_filter` and `get_stock_filter_metadata` remain available through 0.4.x.
Prefer `run_screener` and `get_screener_metadata`.

See `docs/design/screening-extraction-phase0.md` for the migration ADR.
