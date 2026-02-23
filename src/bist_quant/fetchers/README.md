# `fetchers/` — Data Seeding Scripts

## Purpose

One-off CLI scripts for seeding the local data cache. These scripts are not part of the main library API — they are run manually or via cron to initialize or refresh baseline price/index data files.

## Files

```
fetchers/
├── fetch_gold_prices.py    # Seeds 10 years of XAU/TRY and USD/TRY prices
└── fetch_indices.py        # Seeds 10 years of XU030, XU100, XUTUM index prices
```

---

## `fetch_gold_prices.py`

Fetches 10 years of daily XAU/TRY, USD/TRY, and derived XAU/USD cross-rate.

**Run:**
```bash
python -m bist_quant.fetchers.fetch_gold_prices [--years 10] [--output-csv path]
```

**Output:** Cached to `data/borsapy_cache/gold/xau_try_daily` via `DiskCache`.

**Source:** `borsapy.FX` for both `ons-altin` (XAU/TRY) and USD/TRY. XAU/USD is computed as a cross rate.

**Cache TTL:** 86400 seconds (1 day). Cache key: `"gold"/"xau_try_daily"`.

**Columns:** `XAU_TRY`, `USD_TRY`, `XAU_USD`. Index is date-only (no time component).

---

## `fetch_indices.py`

Fetches 10 years of daily OHLCV for XU030, XU100, and XUTUM indices.

**Run:**
```bash
python -m bist_quant.fetchers.fetch_indices [--data-dir path/to/data]
```

**Data flow:**
1. Try yfinance with `.IS` suffix tickers (`XU030.IS`, `XU100.IS`, `XUTUM.IS`).
2. If fewer than 100 rows returned → fall back to `BorsapyAdapter`.

**Output:** Both parquet and CSV saved under `data/borsapy_cache/index_components/`.

**yfinance MultiIndex handling:** yfinance `> 0.2.x` returns MultiIndex columns. Script calls `droplevel("Ticker")` to flatten.

---

## Local Rules for Contributors

1. **`fetchers/` is the only place yfinance is used.** All other data access goes through borsapy clients.
2. **Both parquet and CSV must be written** — `_append_and_persist()` and `fetch_index_data()` both maintain dual-format files. Do not write only one format.
3. **These scripts are idempotent** — running them multiple times is safe. They append new rows and deduplicate on the `Date` column.
4. **Do not add API-dependent business logic here.** Fetchers are infrastructure — signal construction and analytics belong in other modules.
