# ADR: Screening & Technical Scan Library Extraction (Phase 0)

**Status:** Completed (2026-06-19) — `server/` removed from this repo in v0.5.0; screener and technical scan live in `bist_quant`.  
**Scope:** Move `server/engines/stock_filter.py` → `bist_quant.screening`; fold `server/engines/technical_scanner.py` → `bist_quant.clients.technical_scan`

---

## Context

The library currently depends on the optional `server/` package in two places:

| Consumer | Server dependency | Problem |
|----------|-------------------|---------|
| `DataLoader.technical_scan()` | `server.engines.technical_scanner.TechnicalScannerEngine` | Wheel users need server on `PYTHONPATH` for a core data API |
| API screener routes | `server.engines.stock_filter` | Screener logic is app-layer but has no library equivalent |

`stock_filter.py` (~1,740 LOC) is the highest unique-value engine. `technical_scanner.py` (~218 LOC) is a thin borsapy wrapper duplicated in spirit by `BorsapyClient.technical_scan()`.

---

## Decisions

### D1 — Package location

**`bist_quant.screening` lives in the main repo** (`src/bist_quant/screening/`).

Rationale: Screener is research tooling (local cache + hybrid fundamentals), not HTTP infrastructure. A separate PyPI package adds versioning overhead without clear benefit.

### D2 — Technical scan location

**New module:** `src/bist_quant/clients/technical_scan.py`

- Class renamed: `TechnicalScannerEngine` → **`TechnicalScanner`**
- `BorsapyClient.technical_scan()` upgraded to delegate here (predefined templates, `scan_multi`, borsapy version shims)
- `DataLoader.technical_scan()` calls library code directly — **no server import**

Alternative considered: inline everything in `borsapy_client.py`. Rejected — scanner normalization/shims are ~200 LOC and would bloat an already large client file.

### D3 — Error hierarchy

**New names in library** (`bist_quant.screening.errors`):

| New class | Replaces | `code` constant |
|-----------|----------|-----------------|
| `ScreeningError` | `QuantEngineError` | `SCREENING_ERROR` |
| `ScreeningValidationError` | `QuantEngineValidationError` | `VALIDATION_ERROR` |
| `ScreeningDataError` | `QuantEngineDataError` | `DATA_ERROR` |
| `ScreeningExecutionError` | `QuantEngineExecutionError` | `EXECUTION_ERROR` |

Server shims (Phase 5) will re-export aliases:

```python
QuantEngineValidationError = ScreeningValidationError  # deprecated
```

Technical scan raises `ValueError` today — **keep that** in Phase 1; optional migration to `ScreeningValidationError` later.

### D4 — Public API naming

Prefer library-native names; keep server names as deprecated aliases for one minor release (`0.4.x`).

| New (library) | Deprecated alias | Notes |
|---------------|------------------|-------|
| `run_screener()` | `run_stock_filter()` | Same signature |
| `get_screener_metadata()` | `get_stock_filter_metadata()` | Same return shape |
| `TechnicalScanner` | `TechnicalScannerEngine` | Server shim only |

Constants keep existing names (already neutral): `FILTER_FIELD_DEFS`, `INDEX_OPTIONS`, `RECOMMENDATION_OPTIONS`, `TEMPLATE_PRESETS`, `DATA_SOURCE_OPTIONS`, `DEFAULT_TEMPLATES`.

### D5 — Response schema (unchanged)

`StockScreenerResult` TypedDict moves to `bist_quant.screening.types`. JSON shape is **frozen** — API and frontend contracts must not change.

Required top-level keys from `run_screener()`:

```
meta, columns, rows, applied_filters, chart
```

Optional keys preserved: `applied_percentile_filters`.

Metadata response keys (`get_screener_metadata()`):

```
templates, technical_scans, filters, indexes, recommendations,
default_sort_by, default_sort_desc, filter_mode, data_sources, default_data_source
```

### D6 — Deprecation timeline

| Release | Action |
|---------|--------|
| **0.4.0** | Library modules ship; server re-exports with `DeprecationWarning` |
| **0.4.x** | Docs/CLI point to `bist_quant.screening` |
| **0.5.0** | Remove `server/engines/stock_filter.py` and `technical_scanner.py` bodies; keep thin re-export stubs or delete if server repo split complete |

### D7 — Cache module attribute

`_SCREEN_CACHE` becomes `bist_quant.screening.cache.SCREEN_CACHE` (leading underscore removed for test monkeypatching, documented as internal).

Tests currently patch `stock_filter._SCREEN_CACHE` — update to `bist_quant.screening.cache.SCREEN_CACHE`.

---

## Target module tree

```
src/bist_quant/
├── clients/
│   ├── borsapy_client.py       # technical_scan() delegates to technical_scan.py
│   └── technical_scan.py       # NEW — Phase 1
└── screening/
    ├── __init__.py             # public exports
    ├── types.py                # StockScreenerResult
    ├── errors.py               # ScreeningError hierarchy
    ├── presets.py              # constants + TEMPLATE_PRESETS
    ├── parsing.py              # payload coercion helpers
    ├── data_sources.py         # local / isyatirim / hybrid loading
    ├── filters.py              # template, numeric, percentile, technical
    ├── cache.py                # SCREEN_CACHE, TTL, state token
    └── screener.py             # _run_response, run_screener, get_screener_metadata
```

---

## Function inventory → module mapping

Source: `server/engines/stock_filter.py` (49 symbols: 4 public-ish + 45 internal).

### `presets.py` (~100 LOC)

| Symbol | Type |
|--------|------|
| `FILTER_FIELD_DEFS` | constant |
| `FIELD_LABELS` | derived constant |
| `DISPLAY_COLUMNS_DEFAULT` | constant |
| `INDEX_OPTIONS` | constant |
| `RECOMMENDATION_OPTIONS` | constant |
| `DEFAULT_TEMPLATES` | constant |
| `DATA_SOURCE_OPTIONS` | constant |
| `TEMPLATE_PRESETS` | constant |

### `cache.py` (~40 LOC)

| Symbol | Type |
|--------|------|
| `SCREEN_CACHE_TTL_SEC` | constant |
| `SCREEN_CACHE` | mutable cache dict (was `_SCREEN_CACHE`) |
| `_screen_cache_state_token()` | function |

### `parsing.py` (~200 LOC)

| Symbol | Type |
|--------|------|
| `_as_int`, `_as_float`, `_safe_bool` | coercion |
| `_normalize_data_source` | validation |
| `_as_symbol_list`, `_normalize_condition_list` | lists |
| `_jsonable`, `_normalize_text`, `_normalize_column_name` | serialization |
| `_extract_alias_series`, `_extract_numeric_from_dict`, `_extract_text_from_dict` | frame helpers |
| `_normalize_recommendation_value` | domain |
| `_normalize_filters`, `_normalize_percentile_filters` | filter parsing |
| `_safe_divide`, `_quarter_sort_key`, `_extract_metric_pair` | math/helpers |

### `data_sources.py` (~550 LOC)

| Symbol | Type |
|--------|------|
| `_ensure_screen_columns` | normalization |
| `_normalize_isyatirim_frame` | normalization |
| `_merge_hybrid_fields` | hybrid merge |
| `_extract_fundamental_snapshot` | local fundamentals |
| `_load_sector_map` | sector lookup |
| `_get_index_components` | index membership |
| `_load_local_screen_frame` | local cache build |
| `_resolve_isyatirim_as_of` | timestamp |
| `_create_loader_for_runtime` | DataLoader factory |
| `_load_isyatirim_screen_frame` | İş Yatırım fetch |
| `_build_isyatirim_enrichment_frame` | enrichment |
| `_apply_isyatirim_enrichment` | enrichment apply |

### `filters.py` (~150 LOC)

| Symbol | Type |
|--------|------|
| `_resolve_technical_conditions` | technical payload → conditions |
| `_apply_technical_scan` | intersect screener rows with scan hits |
| `_apply_template` | named template presets |

Uses `TechnicalScanner` from `bist_quant.clients.technical_scan`.

### `screener.py` (~400 LOC)

| Symbol | Type |
|--------|------|
| `_resolve_paths` | runtime validation |
| `_meta_response` | metadata builder |
| `_run_response` | core orchestration |
| `_friendly_error` | SSL / user messages |
| `get_screener_metadata` | **public** |
| `run_screener` | **public** |

### `__init__.py` exports

```python
__all__ = [
    # API
    "run_screener",
    "get_screener_metadata",
    "run_stock_filter",           # deprecated alias
    "get_stock_filter_metadata",  # deprecated alias
    # Constants
    "FILTER_FIELD_DEFS",
    "INDEX_OPTIONS",
    "RECOMMENDATION_OPTIONS",
    "TEMPLATE_PRESETS",
    "DATA_SOURCE_OPTIONS",
    "DEFAULT_TEMPLATES",
    # Types & errors
    "StockScreenerResult",
    "ScreeningError",
    "ScreeningValidationError",
    "ScreeningDataError",
    "ScreeningExecutionError",
]
```

---

## Request payload schema (frozen)

Documented for API parity and CLI design (Phase 6).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `data_source` | str | `"local"` | `local` \| `isyatirim` \| `hybrid` |
| `template` | str | `""` | Named preset (see `DEFAULT_TEMPLATES`) |
| `index` | str | `""` | `XU030` \| `XU050` \| `XU100` \| `XUTUM` \| `CUSTOM` |
| `symbols` | str \| list | — | Custom universe (required when `index=CUSTOM`) |
| `sector` | str | `""` | Sector name filter |
| `recommendation` | str | `""` | `AL` \| `TUT` \| `SAT` |
| `filters` | dict | `{}` | `{field: {min, max}}` |
| `percentile_filters` | dict | `{}` | `{field: {min_pct, max_pct}}` |
| `technical_scan` | str \| list | — | Expression or predefined name |
| `technical_scan_name` | str | — | Predefined scan alias |
| `technical_template` | str | — | Same as predefined |
| `technical_condition` | str | — | Raw expression |
| `technical_conditions` | list | — | Multiple expressions (AND) |
| `technical_interval` | str | `"1d"` | Scan interval |
| `sort_by` | str | `"upside_potential"` | Column key |
| `sort_desc` | bool | `true` | Descending sort |
| `limit` | int | `100` | Page size (1–2000) |
| `page` | int | `1` | Page number |
| `offset` | int | — | Alternative to page |
| `chart_points` | int | — | Sparkline length |
| `refresh_cache` | bool | `false` | Bypass screen cache |
| `_refresh_cache` | bool | `false` | Internal alias |

---

## Technical scan module (Phase 1)

Source: `server/engines/technical_scanner.py` → `clients/technical_scan.py`

| Symbol | Notes |
|--------|-------|
| `PREDEFINED_SCANS` | was `TechnicalScannerEngine.PREDEFINED` |
| `TechnicalScanner` | class |
| `.scan()` | single condition |
| `.scan_multi()` | AND merge |
| `.predefined_scans()` | copy of predefined dict |
| `_normalize_symbol()` | module-level helper |

**Integration points:**

```python
# DataLoader (after Phase 1)
def technical_scan(self, condition=None, universe="XU100", interval="1d", conditions=None):
    from bist_quant.clients.technical_scan import TechnicalScanner
    ...

# BorsapyClient — add template= kwarg resolving PREDEFINED_SCANS keys
```

---

## Dependency graph (after migration)

```
bist_quant.screening.screener
  ├── bist_quant.screening.presets
  ├── bist_quant.screening.cache
  ├── bist_quant.screening.data_sources
  │     └── bist_quant.common.data_loader
  ├── bist_quant.screening.filters
  │     └── bist_quant.clients.technical_scan
  ├── bist_quant.screening.parsing
  ├── bist_quant.screening.errors
  ├── bist_quant.screening.types
  ├── bist_quant.runtime
  └── bist_quant.signals.borsapy_indicators  (RSI/MACD columns in local frame)

bist_quant.clients.technical_scan
  └── borsapy (optional)

server/engines/stock_filter.py  (Phase 5 shim)
  └── bist_quant.screening  (re-export only)
```

**Invariant:** No `src/bist_quant/**` file may import `server.*` after Phase 3.

---

## Test migration map

| Current file | New location | Phase |
|--------------|--------------|-------|
| `tests/unit/test_technical_scanner_engine.py` | `tests/unit/clients/test_technical_scan.py` | 1 |
| `tests/unit/test_data_loader.py` (`test_technical_scan_*`) | update imports only | 1 |
| `tests/unit/test_stock_filter_data_source.py` | `tests/unit/screening/test_data_sources.py` | 4 |
| `tests/unit/test_stock_filter_technical.py` | `tests/unit/screening/test_technical_filters.py` | 4 |
| `tests/test_screener_api.py` | unchanged (via server shim) | 5 |
| `tests/test_frontend_api_contracts.py` | unchanged (via server shim) | 5 |

---

## Implementation phases (reference)

| Phase | Deliverable | Exit criterion |
|-------|-------------|----------------|
| **0** | This ADR | Design locked ✓ |
| **1** | `clients/technical_scan.py` | `DataLoader.technical_scan()` works without server | **Done** |
| **2** | `screening/errors.py`, `types.py` | Importable skeleton | **Done** |
| **3** | Full screener move | `run_screener()` end-to-end | **Done** |
| **4** | Test migration | All unit tests use library imports | **Done** |
| **5** | Server shims | API tests pass | **Done** |
| **6** | CLI `bist-quant screener` | Terminal access | **Done** |
| **7** | Delete engine files | No stale `server.engines` screener/scanner | **Done** |
| **8** | Docs + 0.4.0 release | README examples | **Done** |

---

## Out of scope

- Moving `factor_lab`, `signal_construction`, `CoreBackendService`
- New `bist-quant-server` repo split (do after Phase 7)
- Changing screener JSON response shape
- Companion PyPI package

---

## Open questions (none blocking Phase 1)

1. **Split vs monolith for Phase 3:** Start with single `screener.py` copy, refactor into submodules in a follow-up commit if diff is too large. Submodule layout above is the target, not a hard requirement for first landing.
2. **`DataLoader.run_screener()`:** Optional convenience wrapper — defer to Phase 6 unless needed earlier.
