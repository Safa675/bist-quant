# `persistence/` — Data Persistence Layer

## Purpose

Two minimal JSON-backed stores: an async store for long-running job records and a synchronous store for backtest run history. Both use atomic writes and mutual exclusion.

## Files

```
persistence/
├── job_store.py   # Async JSON-backed store for Job records (jobs system)
└── run_store.py   # Synchronous JSON-backed store for backtest run history
```

---

### `job_store.py` — Async Job Store

Used by `jobs/manager.py`. Persists `Job` records to a single JSON file.

**Key methods:**

| Method | Description |
|---|---|
| `save(job)` | Upsert a job record |
| `get(job_id)` | Fetch single job |
| `list(status?, job_type?, limit, offset)` | Filtered + paginated listing |
| `update_progress(job_id, progress)` | **In-memory only** — no disk flush |
| `cleanup(days)` | Remove terminal jobs older than N days |

**Implementation details:**
- File I/O via `asyncio.to_thread` (non-blocking).
- Atomic writes: write to `.tmp` file → `os.replace()`.
- Lazy loading: `_ensure_loaded()` guard on every access.
- Format: `{"jobs": [...], "updated_at": "ISO string"}`, `datetime` fields as ISO strings.

---

### `run_store.py` — Synchronous Run Store

Used by `services/core_service.py` for backtest run history. Synchronous with `threading.Lock`.

**Key methods:**

| Method | Description |
|---|---|
| `create_or_update_run(run_id, kind, ...)` | Upsert a run; auto-generates IDs; tracks `started_at`/`finished_at` |
| `list_runs(kind?, status?, limit, offset)` | Filtered listing, sorted by `updated_at` descending |
| `get_run(run_id)` | Fetch single run |
| `delete_run(run_id)` | Remove run |

**Security:** Input IDs validated against `_RUN_ID_RE` and `_ARTIFACT_ID_RE` regex allowlists. `kind` sanitized to lowercase alphanumeric (capped at 64 chars).

**Atomic writes:** `.tmp` → `os.replace()`, same as job store.

---

## Local Rules for Contributors

1. **Atomic writes always.** All file writes must use the `.tmp → replace` pattern. Direct `open(path, "w")` is not acceptable.
2. **`update_progress` is in-memory only.** Do not add disk writes to `update_progress` — it is called at high frequency and disk I/O would be a bottleneck.
3. **Validate IDs before persistence.** Any new ID field must be validated against a regex allowlist to prevent path traversal or injection attacks.
4. **Do not add binary persistence.** These stores are intentionally JSON for human readability and debuggability. Do not switch to SQLite or binary formats without broader discussion.
