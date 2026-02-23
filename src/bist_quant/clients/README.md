# `clients/` — Data Access Layer

## Purpose

All external data connectors live here. Each client wraps a specific data source (borsapy, Borsa MCP, isyatirim, TradingView, TEFAS) with caching, retry, circuit-breaking, and defensive normalization. The `DataLoader` in `common/` uses these clients internally via injected adapter properties.

## Files

```
clients/
├── borsapy_client.py              # Primary BIST client — price download, index composition, financials
├── borsapy_adapter.py             # Thin adapter integrating BorsapyClient into DataLoader
├── fx_enhanced_provider.py        # FX bank rates, institution rates (gold/silver), intraday bars
├── derivatives_provider.py        # VIOP futures/options data
├── fixed_income_provider.py       # Bond yields, TCMB policy rate, Eurobonds
├── economic_calendar_provider.py  # Macro event calendar (TR, US, EU, DE, GB, JP, CN)
├── macro_adapter.py               # Two-tier adapter: EconomicCalendarProvider → MacroEventsClient
├── macro_events.py                # Standalone macro events client via borsapy
├── crypto_client.py               # Async crypto market data via Borsa MCP endpoint
├── us_stock_client.py             # Async US equity data via Borsa MCP endpoint
├── fund_analyzer.py               # Async TEFAS mutual fund analytics via Borsa MCP
├── fx_commodities_client.py       # FX and commodities price client
└── update_prices.py               # CLI script: incremental price updates for BIST/XU100/XAU/TRY
```

---

### `borsapy_client.py` — Primary BIST Client

The main workhorse for all BIST market data. Adds:
- **Exponential-backoff retry** via `@retry_with_backoff()` decorator.
- **Thread-safe circuit breaker** (`CircuitBreaker`) that opens after N failures and recovers after a timeout.
- **Disk-caching** via `DiskCache` / `CacheTTL`.
- **MCP fallback** via `https://borsamcp.fastmcp.app/mcp`.

**Key classes:**

| Class | Description |
|---|---|
| `CircuitBreaker` | Opens after `failure_threshold` consecutive failures; recovers after `recovery_timeout` seconds |
| `CircuitBreakerError` | Raised when circuit is open |
| `BorsapyClient` | Main client: `batch_download_to_long()`, `get_index_components()`, `get_financial_statements()` |

**Important constant:**  `UFRS_TICKERS` — ~31 bank/financial tickers that require the UFRS accounting endpoint on İş Yatırım (not the standard endpoint). Always check this set before choosing the fetch endpoint.

---

### `borsapy_adapter.py` — DataLoader Integration Adapter

Wraps `BorsapyClient` into the `DataLoader` infrastructure. Lazily initializes the client on first access (`.client` property). Config loaded from `configs/borsapy_config.yaml`.

**Key class:** `BorsapyAdapter` — methods: `load_prices()`, `get_index_components()`, `get_financials()`

---

### Provider Classes (FX, Derivatives, Fixed Income, Calendar)

All four providers share the same structural pattern:

```
class XxxProvider:
    _import_attempted: bool = False   # guards repeated try/except
    _bp = None                        # lazy borsapy module reference

    def __init__(self, cache_dir=None):
        self._cache = DiskCache(cache_dir) if cache_dir else None

    def _get_bp(self):
        # One-time guarded import of borsapy
```

**Turkish number format handling:** All four providers include `_to_float()` helpers that strip `%` signs and replace `,` with `.` for Turkish numeric formatting (e.g. `"1.234,56"` → `1234.56`).

**`FixedIncomeProvider` specifics:**
- `DEFAULT_FALLBACK_RISK_FREE_RATE = 0.28` (28% TRY-denominated, used when live rate unavailable).
- `_to_percent_rate()` auto-converts decimal annualized rates (`0.28` → `28.0`).

**`EconomicCalendarProvider` specifics:**
- Accepts injectable `now_fn` for testability.
- Canonical columns: `Date, Time, Country, Importance, Event, Forecast, Previous`.
- Supported countries: `TR, US, EU, DE, GB, JP, CN`.

---

### Async Clients (Crypto, US Stocks, TEFAS)

`CryptoClient`, `USStockClient`, and `TEFASAnalyzer` are all **async** (`httpx.AsyncClient`) and call the Borsa MCP endpoint via JSON-RPC.

Common pattern:
```python
async def _call_mcp_async(self, tool_name, params) -> str:
    # POST to MCP endpoint, parse JSON-RPC response
```

All three use: in-memory TTL cache + optional `DiskCache`.

---

### `macro_adapter.py` — Two-Tier Fallback

**Priority**: `EconomicCalendarProvider` (structured) → `MacroEventsClient` (legacy path, dynamically loaded).

Returns empty `dict`/`list`/`DataFrame` if both are unavailable — **never raises** to callers.

---

### `update_prices.py` — Price Update CLI

Run as: `python -m bist_quant.clients.update_prices [--source {auto,borsapy,yfinance}] [--dry-run]`

Appends new OHLCV rows to:
- `bist_prices_full.csv/.parquet`
- `xu100_prices.csv/.parquet`
- `xau_try_2013_2026.csv/.parquet`

Uses `_append_and_persist()` which deduplicates on `Date` before writing both CSV and Parquet in sync.

---

## Local Rules for Contributors

1. **borsapy is primary; never use yfinance directly here.** yfinance belongs only in `fetchers/`.
2. **Guard all borsapy imports with `try/except`** and set a module-level `_import_attempted` flag so the import failure only logs once.
3. **Always inject `DiskCache`** via constructor `cache_dir` param rather than hardcoding paths. The cache directory is resolved by `DataLoader`.
4. **Turkish numeric formats.** All providers dealing with İş Yatırım / borsapy data must use `_to_float()` handling for `%` and `,` separators.
5. **`UFRS_TICKERS` check.** When adding any financial statement fetch logic in `borsapy_client.py`, verify whether the ticker uses the UFRS endpoint by checking against the `UFRS_TICKERS` set.
6. **Circuit breaker threshold.** Do not lower the failure threshold below 3 or the recovery timeout below 30 seconds — this would cause instability with the borsapy API.
7. **Async clients must not be called synchronously.** `CryptoClient`, `USStockClient`, and `TEFASAnalyzer` are async. Synchronous adapters (`PortfolioAnalyticsAdapter`) are the bridge layer.
