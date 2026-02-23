# Multi-Asset Data

Data access hierarchy:

1. **borsapy** (primary) — all BIST data: prices, fundamentals, FX, gold, derivatives
2. **Borsa MCP** (`https://borsamcp.fastmcp.app/mcp`) — async: crypto, US stocks, TEFAS funds
3. **yfinance** — only in `fetchers/` scripts, never inside the library core

## BIST Data via borsapy

```python
from bist_quant import DataLoader, DataPaths

loader = DataLoader(data_paths=DataPaths())

close   = loader.build_close_panel()        # (dates × tickers) BIST prices
fx      = loader.load_fx_rates()            # USD/TRY, EUR/TRY series
gold    = loader.load_gold_prices()         # XAU/TRY series
findex  = loader.load_index_prices("XU100") # index history
funda   = loader.load_fundamentals()        # financial statements
derivs  = loader.load_derivatives()         # VIOP open interest
```

## Async Clients (Borsa MCP)

```python
import asyncio
from bist_quant.clients.mcp_client import BorsaMCPClient

async def main():
    async with BorsaMCPClient() as client:
        btc = await client.get_crypto_prices(["BTC", "ETH"], days=365)
        spy = await client.get_us_stock_prices(["SPY", "QQQ"])
        tefas = await client.get_tefas_fund("AKP")  # TEFAS fund code
    return btc, spy, tefas

btc, spy, tefas = asyncio.run(main())
```

## FX and Commodities

```python
from bist_quant.clients.fx_client import FXClient

client = FXClient()
usd_try = client.get_rate("USD", "TRY")
eur_try = client.get_rate("EUR", "TRY")
xau_try = client.get_gold_price()    # same as loader.load_gold_prices()
```

## Fixed Income & Macro

```python
from bist_quant.clients.fixed_income_client import FixedIncomeClient
from bist_quant.clients.macro_calendar_client import MacroCalendarClient

bonds  = FixedIncomeClient().get_yield_curve()       # TCMB yield curve
macro  = MacroCalendarClient().get_upcoming_events() # TCMB / TUIK releases
```

## Realtime Streaming

TradingView WebSocket is the primary realtime feed; borsapy polling is the fallback:

```python
from bist_quant.realtime.streaming_provider import StreamingProvider

async def on_tick(tick):
    print(tick.symbol, tick.price, tick.timestamp)

provider = StreamingProvider(symbols=["THYAO", "AKBNK", "KCHOL"])
await provider.start(callback=on_tick)
```
