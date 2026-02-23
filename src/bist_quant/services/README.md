# `services/` — Application Service Layer

## Purpose

High-level service classes that expose the library's capabilities to API handlers and UI pages. Services handle serialization, error recovery, path resolution, and progress reporting. They do not contain business logic — they orchestrate it.

## Files

```
services/
├── core_service.py        # Core portfolio + backtest functionality for API handlers
├── realtime_service.py    # Typed realtime market data (Decimal-precise, market status)
├── realtime_stream.py     # Simpler thread-safe realtime quotes for dashboard integration
└── system_service.py      # Settings persistence, backups, system diagnostics
```

---

## `core_service.py` — Core Backend Service

The primary service used by API handlers and the Streamlit backend.

**Initialization:**
```python
from bist_quant.services.core_service import CoreBackendService
service = CoreBackendService()  # resolves RuntimePaths automatically
```

**Key responsibilities:**
- Signal config discovery (`load_signal_configs()`, `list_available_signals()`)
- Backtest execution (`run_backtest()`, `run_factor()`)
- Optimization runs (`run_optimization()`)
- JSON-serializable output building (`_serialize_equity_curve()`, `_serialize_drawdown_curve()`, `_serialize_monthly_returns()`)
- Holdings extraction (`_extract_latest_holdings(holdings_history, top_n)`)

**Progress reporting:** Uses `ProgressCallback = Callable[[float, str, int | None], None]` for reporting to job system or Streamlit.

**`FactorRunSpec(frozen=True)`** — specifies a factor to run: `name`, `weight`, `signal_params`.

---

## `realtime_service.py` — Typed Realtime Service

Production-grade service with `Decimal`-precision financials and proper BIST market session handling.

**Key types:**

| Class | Description |
|---|---|
| `MarketStatus(Enum)` | `OPEN, CLOSED, PRE_MARKET, POST_MARKET, UNKNOWN` |
| `Quote` | Decimal-precision quote snapshot |
| `IndexData` | Index level, change, and component info |
| `FXRate` | Currency pair rate |
| `PortfolioValuation` | Mark-to-market with `Decimal` precision |
| `QuoteCache` | TTL-based in-memory cache (default 5 seconds, auto-evicts on read) |

**Key methods:**

| Method | Description |
|---|---|
| `get_quote(symbol)` | Cached quote fetch |
| `get_quotes(symbols)` | Batch cached quote fetch |
| `get_index_data()` | XU100/XU030/XU050 overview |
| `get_fx_rates()` | USD/TRY, EUR/TRY, GBP/TRY rates |
| `get_market_summary()` | Combined market overview |
| `get_portfolio_valuation(positions, cash)` | Decimal-precise mark-to-market |
| `get_market_status()` | BIST session check (10:00–18:00 IST weekdays) |

**Exception hierarchy:**
```
RealtimeServiceError
├── SymbolNotFoundError
└── MarketDataUnavailableError
```

---

## `realtime_stream.py` — Dashboard Quote Service

Simpler, synchronous, `float`-based alternative to `realtime_service.py` for Streamlit dashboard use.

- `QuoteCache` — Thread-safe TTL dict with `threading.Lock` (60 s default).
- `RealtimeQuoteService.get_quote(symbol, use_cache=True)` — `bp.Ticker.fast_info` lookup.
- `get_quotes_batch(symbols)` — Parallel batch fetch.
- `get_portfolio_snapshot(holdings)` — Quick mark-to-market from holdings dict.
- `get_index_snapshot(indices)` — Index-level data via `bp.Index`.

**Differs from `realtime_service.py`:** Uses `float` (not `Decimal`), is synchronous, and uses `fast_info` attribute access (not `.get()`). Prefer this for Streamlit widgets; prefer `realtime_service.py` for API endpoints.

---

## `system_service.py` — System Operations

Manages application-level settings and maintenance:

| Method | Description |
|---|---|
| `load_settings()` | Reads settings JSON from `~/.bist-quant-ai/settings.json` |
| `save_settings(patch)` | Deep-merges patch into settings; sanitizes input via `sanitize_payload()` |
| `list_backups(limit)` | Lists recent ZIP backups |
| `create_backup(label)` | Creates ZIP backup of settings + run store + project data |
| `cleanup_old_backups(retain_count)` | Removes oldest backups beyond retain count |
| `get_system_diagnostics()` | Platform info, Python version, disk usage, optional psutil metrics |

**App data location:** `~/.bist-quant-ai/`. Backup filenames use UTC ISO timestamps.

---

## Local Rules for Contributors

1. **All user-facing inputs go through `sanitize_payload()`** before persistence in `SystemService`. Do not skip this.
2. **Use `Decimal` in `realtime_service.py`.** Financial values must be `Decimal` for precision. Do not change them to `float`.
3. **BIST market hours are hardcoded** as 10:00–18:00 IST weekdays. Update `get_market_status()` if BIST changes its trading hours.
4. **`CoreBackendService.__init__` validates `RuntimePaths`** — if the paths are wrong, it raises immediately. Do not catch `RuntimePathError` and continue silently.
5. **Progress callbacks must be non-blocking.** If a `ProgressCallback` is slow, it can block backtest execution. Keep callbacks lightweight — just enqueue to an asyncio queue or update a Streamlit state variable.
