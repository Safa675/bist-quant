# `clients/` — Data Access Layer

## Purpose

All external data connectors live here. Each client wraps a specific data source (borsapy, Borsa MCP, isyatirim, TradingView, TEFAS) with caching, retry, circuit-breaking, and defensive normalization. 

The Data Access Layer is structured around:
1. **Base Classes**: `BaseProvider` for synchronous domain-specific providers, and `BaseMCPClient` for asynchronous Model Context Protocol (MCP) clients.
2. **Modular Clients**: `BorsapyClient` (decomposed into specialized helper files: `borsapy_prices.py`, `borsapy_financials.py`, and `borsapy_indices.py` for maximum maintainability).
3. **Adapters**: The `DataLoader` in `common/` uses these clients internally via injected adapter properties (such as `BorsapyAdapter` and `MacroAdapter`).

## Files

```
clients/
├── base_provider.py               # Base class for domain-specific synchronous providers
├── base_mcp.py                    # Base class for asynchronous JSON-RPC MCP clients
├── utils.py                       # Shared data parsers (to_float, as_frame, pick_column, etc.)
├── borsapy_client.py              # Primary BIST client wrapper (facade delegating to helper modules)
├── borsapy_prices.py              # Price downloading, long formatting, and caching logic
├── borsapy_financials.py          # Financial statements, ratios, and UFRS endpoint routing
├── borsapy_indices.py             # Index components, predefined scan presets, and indicator calcs
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

### Configuration & Paths

Default configuration for `BorsapyClient` is loaded dynamically via `get_borsapy_config_path()` defined in `bist_quant.settings`.

- **Environment Override**: Setting the environment variable `BIST_BORSAPY_CONFIG_PATH` to a custom YAML file path overrides the project-level default config.
- **Default Resolution**: Resolves to `PROJECT_ROOT / "configs" / "borsapy_config.yaml"`.

---

### `borsapy_client.py` — Primary BIST Client Facade

The main workhorse for all BIST market data. It uses a facade design pattern, delegating major responsibilities to specialized sub-modules:
- **`borsapy_prices.py`**: Price history, TradingView downloading, and CSV/Parquet price caching.
- **`borsapy_financials.py`**: Auto UFRS detection, quarterly fiscal statement formatting, and ratios.
- **`borsapy_indices.py`**: Predefined scanner presets, index constituents lookup, and technical indicator metrics (RSI, MACD, Supertrend).

**Key classes:**

| Class | Description |
|---|---|
| `CircuitBreaker` | Thread-safe breaker: opens after consecutive failures; recovers after a timeout |
| `BorsapyClient` | Main client interface: `batch_download_to_long()`, `get_index_components()`, `get_financial_statements()` |

**UFRS Endpoint Routing**: Governance of tickers requiring alternate accounting formats is centralized in `bist_quant.common.ticker_sets.UFRS_TICKERS`.

---

### Provider Classes (FX, Derivatives, Fixed Income, Calendar)

All synchronous providers inherit from `BaseProvider`. The base class manages dynamic `borsapy` importing and exposes shared static utility methods (`_to_float`, `_as_frame`, `_pick_column`, and `_call_if_callable`) to its subclasses, eliminating code redundancy.

**Macro Overlaps**: Some macro methods on `BorsapyClient` (e.g. `get_inflation_data`, `get_bond_yields`, `get_economic_calendar`) overlap with domain-specific providers like `FixedIncomeProvider` and `EconomicCalendarProvider`. Callers should prefer utilizing the dedicated providers where possible for richer parsing and structured outputs.

---

### Async MCP Clients (Crypto, US Stocks, TEFAS)

Async clients inherit from `BaseMCPClient`, which encapsulates async HTTP session management and JSON-RPC messaging format.

**Key Base Features**:
- **Async Client**: Handled automatically via `httpx.AsyncClient`.
- **Caching**: Shared cache lookup helper (`_cache_get` / `_cache_set`) utilizing in-memory TTL maps combined with persistent local `DiskCache` saves.

---

## Local Rules for Contributors

1. **borsapy is primary; never use yfinance directly here.** yfinance belongs only in `fetchers/`.
2. **Inherit from base classes.** Synchronous providers must inherit from `BaseProvider`. Asynchronous MCP clients must inherit from `BaseMCPClient`.
3. **Use shared utilities.** Do not re-define or duplicate parsers. Use helper functions exported from `bist_quant.clients.utils` directly or via `BaseProvider` aliases.
4. **Use centralized ticker sets.** Import `UFRS_TICKERS`, `BANK_TICKERS`, and `FINANCE_TICKERS` from `bist_quant.common.ticker_sets` instead of re-defining them locally.
5. **Config paths.** Do not hardcode YAML configuration paths. Always use `get_borsapy_config_path()` to ensure environment overrides work.
