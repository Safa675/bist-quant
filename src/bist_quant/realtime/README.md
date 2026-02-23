# `realtime/` — Live Market Data

## Purpose

Real-time quote fetching, streaming tick overlay, and portfolio valuation snapshots. Uses borsapy as the primary source with TradingView streaming as an overlay and synthetic generation as a last-resort fallback.

## Files

```
realtime/
├── quotes.py     # Single and batch quote fetching with streaming overlay
├── streaming.py  # TradingView stream wrapper (lazy init, symbol management)
├── portfolio.py  # Real-time portfolio valuation snapshot
└── ticks.py      # Tick extraction + streaming overlay + synthetic fallback
```

---

### `quotes.py` — Quote API

**Data flow:** Streaming provider (TradingView) → borsapy `Ticker.fast_info` → error

```python
from bist_quant.realtime.quotes import get_quote, get_quote_streaming

quote = get_quote("THYAO")                         # direct borsapy lookup
quote = get_quote_streaming("THYAO", auth_config)  # streaming first, polling fallback
quotes = get_quotes_streaming(["THYAO", "GARAN"], auth_config)  # batch
```

**Module-level singleton:** `_STREAMING_PROVIDER` — re-initialized when auth config fingerprint changes.

A streaming quote is only trusted over polling if `last_price is not None`.

---

### `streaming.py` — TradingView Stream Wrapper

`StreamingProvider` wraps `borsapy.TradingViewStream` with:
- Lazy initialization on first `subscribe()` call.
- Multiple constructor signature compatibility (different borsapy versions).
- Symbol subscription management.
- **Normalized quote dict** (14 canonical fields) via `_normalize_quote()`.

Auth token resolved from multiple config keys: `auth_token`, `token`, `tradingview_auth_token`.

**No exceptions escape public methods.** All errors are caught and logged internally.

---

### `portfolio.py` — Realtime Portfolio Valuation

```python
from bist_quant.realtime.portfolio import get_portfolio

result = get_portfolio({
    "holdings": {"THYAO": 100, "GARAN": 50},
    "cost_basis": {"THYAO": 280.0, "GARAN": 45.0}
})
# Returns: positions sorted by market_value desc + aggregate totals
```

Tolerates missing quotes gracefully — records `error` per position, continues with remaining.

---

### `ticks.py` — Tick Extraction

**Data flow:** Streaming → borsapy download → raises `RuntimeError` → caller uses synthetic fallback

```python
from bist_quant.realtime.ticks import fetch_realtime_ticks, fallback_realtime_ticks

ticks = fetch_realtime_ticks(["THYAO", "GARAN"], streaming=True, auth_config=cfg)
# If both streaming and borsapy fail:
ticks = fallback_realtime_ticks(["THYAO"], previous=last_ticks)  # synthetic drift
```

**Canonical tick dict fields:** `symbol`, `price`, `change_pct`, `volume`, `timestamp`.

`normalize_realtime_symbols()` — deduplicates, uppercases, caps at 20 symbols, defaults to `["XU100"]`.

---

## Local Rules for Contributors

1. **Never let exceptions escape public methods** in `StreamingProvider`. All errors must be caught and produce empty/default return values.
2. **Streaming quote validation.** A streaming quote is only used if `last_price is not None`. Do not relax this check.
3. **Synthetic fallback is last resort only.** `fallback_realtime_ticks()` is for UI continuity when all data sources fail — never call it as a primary path.
4. **Auth token resolution order** must stay: `auth_token` → `token` → `tradingview_auth_token`. Do not add new auth key names — update the existing resolution chain.
5. **All float values must be `math.isfinite()`** before being included in normalized quotes. Replace non-finite with `None`.
