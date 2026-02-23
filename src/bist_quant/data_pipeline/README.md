# `data_pipeline/` — Fundamentals Data Ingestion Pipeline

## Purpose

An end-to-end pipeline for fetching, validating, normalizing, merging, and persisting Turkish financial statement data from isyatirim.com.tr. This package is fully self-contained with its own error hierarchy, typed data contracts, structured logging, and provenance tracking.

The pipeline has **five sequential stages**:
```
Fetch → Validate Schema → Normalize → Merge → Freshness Gate → Persist
```

## Files

```
data_pipeline/
├── pipeline.py        # Orchestrator: FundamentalsPipeline — single public façade
├── fetcher.py         # HTTP fetch from isyatirim.com.tr with retry + circuit breaker
├── normalize.py       # Raw API payload → flat DataFrame + consolidated panel
├── merge.py           # Merge new panel into existing panel with combine_first semantics
├── freshness.py       # Per-ticker staleness metrics + configurable freshness gate
├── schemas.py         # Schema validation for all pipeline DataFrames (pandera or manual)
├── types.py           # All typed data contracts (frozen dataclasses)
├── errors.py          # Exception hierarchy — one exception type per stage
├── provenance.py      # SHA256 checksums + .meta.json sidecar + audit JSONL
└── logging_utils.py   # Structured JSON logger + append_jsonl
```

---

## Stage-by-Stage Reference

### Stage 1: Fetch (`fetcher.py`)

`FundamentalsFetcher` fetches per-ticker financial statement JSON from isyatirim.com.tr.

**Resilience mechanisms:**
- `CircuitBreaker` opens after N consecutive failures, recovers after a timeout window.
- `retry_with_backoff` decorates each request with exponential backoff.
- Raw JSON results are persisted per ticker under `paths.raw_dir` for re-use without re-fetching.

**UFRS endpoint routing:** Some bank/financial tickers (`BANK_TICKERS`, `FINANCE_TICKERS`, `UFRS_TICKERS`) require a different API group. `classify_fetch_error()` returns stable error codes (`"timeout"`, `"rate_limited"`, etc.) used by the circuit breaker.

Output: `RawDataBundle(raw_by_ticker, errors, source_name, fetched_at)`

---

### Stage 2: Schema Validation (`schemas.py`)

Validates all pipeline DataFrames. Works **with or without `pandera`** — all validators include manual fallbacks.

| Validator | Validates |
|---|---|
| `validate_raw_payload_structure(raw)` | Required dict keys on raw JSON |
| `validate_flat_normalized(df)` | Required columns, dtypes, date validity, quarter range |
| `validate_consolidated_panel(df)` | MultiIndex `(ticker, sheet_name, row_name)` + period columns |
| `validate_staleness_report(df)` | Staleness report schema |

Raises `SchemaValidationError` on any violation.

---

### Stage 3: Normalize (`normalize.py`)

Transforms raw payloads into structured DataFrames:

| Function | Output |
|---|---|
| `build_flat_normalized(raw_by_ticker)` | Flat DataFrame: one row per `(ticker, period)`, ~20 standardized financial metrics |
| `build_consolidated_panel(raw_by_ticker)` | Wide multi-index panel: `(ticker, sheet_name, row_name)` × `period` columns |

**Period column format:** `"YYYY/MM"` strings (e.g. `"2024/12"`). Period end dates use `pd.offsets.MonthEnd(0)` snapping.

**FIELD_KEYS** maps canonical English metric names to their Turkish label variants from the source API. `_extract_field_value()` does fuzzy label matching.

---

### Stage 4: Merge (`merge.py`)

`merge_consolidated_panels(existing, new_data, prefer_existing_values)`:
- Normalizes row names via `_normalize_row_name_series()` (strip/lowercase/collapse whitespace) for fuzzy matching.
- Aligns on `(ticker, sheet_name, normalized_row_name)`.
- Resolves duplicates via `.groupby().first()`.
- Validates output via `validate_consolidated_panel`.
- Returns `(merged_df, stats_dict)` with per-category fill/overwrite counts.

Raises `MergeError` on failure.

---

### Stage 5: Freshness Gate (`freshness.py`)

Blocks the pipeline if data quality falls below configured thresholds:

| Function | Description |
|---|---|
| `compute_staleness_report(consolidated, reference_date)` | Per-ticker: latest quarter + staleness days (vectorized NumPy) |
| `summarize_quality_metrics(staleness_report)` | Scalar aggregates: coverage %, median/max staleness, pct > 120 days |
| `evaluate_freshness(quality_metrics, thresholds)` | Returns human-readable violation strings |
| `enforce_freshness_gate(...)` | Writes alert to JSONL then raises `FreshnessGateError` unless `allow_override=True` |

**`FreshnessThresholds` fields:**

| Field | Default | Description |
|---|---|---|
| `max_median_staleness_days` | 90 | Blocks if median > 90 days |
| `max_pct_over_120_days` | 30.0 | Blocks if > 30% of tickers have stale data |
| `min_q4_coverage_pct` | 60.0 | Blocks if < 60% of tickers have Q4 |
| `max_max_staleness_days` | 365 | Blocks if any ticker exceeds 1 year |
| `grace_days` | 5 | Buffer on all comparisons |

---

## Data Contracts (`types.py`)

All intermediate pipeline outputs are typed frozen dataclasses:

```
RawDataBundle
  → ValidatedDataBundle
    → NormalizedDataBundle (consolidated_like + flat_normalized)
      → MergedDataBundle (+ staleness_report + quality_metrics + fingerprint)
```

`PipelinePaths` (16 path fields) and `PipelineConfig` (retry params, schema version) are also frozen dataclasses.

---

## Exception Hierarchy (`errors.py`)

```
FundamentalsPipelineError
├── FetchError
│   └── CircuitBreakerOpenError
├── SchemaValidationError
├── MergeError
├── FreshnessGateError
└── ProvenanceError
```

One exception type per pipeline stage — never raise `FundamentalsPipelineError` directly.

---

## Provenance (`provenance.py`)

Every output file gets a `.meta.json` sidecar with:
- `sha256` checksum (deterministic: sort → `pd.util.hash_pandas_object` → SHA256)
- `shape`, `pipeline_version`, `schema_version`
- `generated_at` timestamp
- Quality metrics summary

All writes also append to an audit JSONL log at `paths.audit_log`.

---

## Structured Logging (`logging_utils.py`)

All events are logged as structured JSON:
```python
log_event(logger, "fetch_complete", ticker="THYAO", rows=40, elapsed_ms=320)
# → {"event": "fetch_complete", "ticker": "THYAO", "rows": 40, "elapsed_ms": 320, "ts": "2026-..."}
```

`append_jsonl(path, payload)` always prepends a UTC ISO timestamp.

---

## Local Rules for Contributors

1. **One exception type per stage.** Raise the specific exception (`FetchError`, `SchemaValidationError`, etc.) not the base class.
2. **Period columns must be `"YYYY/MM"` strings.** Do not use `pd.Period` objects or any other date format for the consolidated panel columns.
3. **Row name normalization is critical.** Any new row added to the consolidated panel must go through `_normalize_row_name_series()` to ensure it can be matched during merge.
4. **`pandera` is optional.** All validators in `schemas.py` must work without `pandera` installed. The `pandera` checks provide additional depth but are never required for correctness.
5. **All logging uses `log_event()`.** Do not use `print()` or `logger.info(f"...")` directly — always use the structured `log_event` helper for pipeline events.
6. **Provenance on every write.** Every time a pipeline output file is written, `write_dataset_provenance()` must be called to create the `.meta.json` sidecar.
7. **Freshness gate is not optional in production.** `allow_override=True` is only for development/testing. Production runs must enforce the gate.
