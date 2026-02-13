# Borsapy Integration vs Borsa-MCP Comparison Report

**Date**: February 2026
**Purpose**: Evaluate overlap and integration opportunities between your existing borsapy implementation and borsa-mcp

---

## Executive Summary

| Aspect | Your Borsapy Integration | Borsa-MCP |
|--------|--------------------------|-----------|
| **Implementation Status** | 5/7 phases complete, production-ready | Complete, hosted MCP server |
| **Data Source** | TradingView via borsapy library | Same (TradingView via borsapy) |
| **Architecture** | Direct Python library integration | MCP protocol wrapper for LLMs |
| **Best For** | Backtesting, signals, dashboards | LLM agent tool calling |

**Key Finding**: Both use the same underlying data source (borsapy/TradingView). Borsa-MCP is essentially a **tool-calling wrapper** around similar capabilities you already have. The value-add is its **LLM-ready interface**.

---

## Feature Comparison Matrix

### 1. Stock Data & Quotes

| Feature | Your Implementation | Borsa-MCP | Overlap |
|---------|---------------------|-----------|---------|
| Real-time quotes | ✅ `RealtimeQuoteService` | ✅ `get_quick_info` | 100% |
| Historical OHLCV | ✅ `batch_download_to_long()` | ✅ `get_historical_data` | 100% |
| Company profiles | ✅ `get_fast_info_borsapy()` | ✅ `get_profile` | 100% |
| Symbol search | ❌ Not implemented | ✅ `search_symbol` | **Gap** |
| 758 BIST stocks | ✅ Full access | ✅ Full access | 100% |
| US stocks (NYSE/NASDAQ) | ❌ Not implemented | ✅ Supported | **Gap** |

### 2. Technical Analysis

| Feature | Your Implementation | Borsa-MCP | Overlap |
|---------|---------------------|-----------|---------|
| RSI | ✅ `build_rsi_panel()` | ✅ `get_technical_analysis` | 100% |
| MACD | ✅ `build_macd_panel()` | ✅ `get_technical_analysis` | 100% |
| Bollinger Bands | ✅ `build_bollinger_panel()` | ✅ `get_technical_analysis` | 100% |
| ATR | ✅ `build_atr_panel()` | ✅ Included | 100% |
| Stochastic | ✅ `build_stochastic_panel()` | ✅ Included | 100% |
| ADX | ✅ `build_adx_panel()` | ✅ Included | 100% |
| Supertrend | ✅ `build_supertrend_panel()` | ✅ `scan_stocks` | 100% |
| Pivot Points | ❌ Not implemented | ✅ `get_pivot_points` | **Gap** |
| Multi-indicator batch | ✅ `build_multi_indicator_panel()` | ❌ Single calls | **You're ahead** |

### 3. Stock Screening

| Feature | Your Implementation | Borsa-MCP | Overlap |
|---------|---------------------|-----------|---------|
| Fundamental screening | ⚠️ Blocked by SSL | ✅ `screen_securities` (23 presets) | **Gap** |
| Technical scanning | ⚠️ Blocked by SSL | ✅ `scan_stocks` | **Gap** |
| Custom filters | ⚠️ Blocked | ✅ Supported | **Gap** |

### 4. Fundamental Data

| Feature | Your Implementation | Borsa-MCP | Overlap |
|---------|---------------------|-----------|---------|
| Financial statements | ⚠️ Returns empty | ✅ `get_financial_statements` | **Gap** |
| Financial ratios | ❌ Not implemented | ✅ `get_financial_ratios` | **Gap** |
| Dividends | ⚠️ Returns empty | ✅ `get_dividends` | **Gap** |
| Earnings calendar | ✅ `get_earnings_calendar()` | ✅ `get_earnings` | 100% |
| Corporate actions | ❌ Not implemented | ✅ `get_corporate_actions` | **Gap** |
| Analyst ratings | ✅ `get_analyst_recommendations()` | ✅ `get_analyst_data` | 100% |

### 5. Macro & Economic Data

| Feature | Your Implementation | Borsa-MCP | Overlap |
|---------|---------------------|-----------|---------|
| Economic calendar | ✅ `get_economic_calendar()` | ✅ `get_economic_calendar` | 100% |
| TCMB inflation | ✅ `get_inflation_data()` | ✅ `get_macro_data` | 100% |
| Bond yields | ✅ `get_bond_yields()` | ✅ `get_bond_yields` | 100% |
| TCMB rates | ✅ `get_tcmb_rates()` | ✅ Included | 100% |
| Eurobonds | ✅ `get_eurobonds()` | ❌ Not listed | **You're ahead** |
| Sector comparison | ❌ Not implemented | ✅ `get_sector_comparison` | **Gap** |

### 6. Funds & Indices

| Feature | Your Implementation | Borsa-MCP | Overlap |
|---------|---------------------|-----------|---------|
| Index components | ✅ `get_index_components()` | ✅ `get_index_data` | 100% |
| 81 BIST indices | ✅ Full access | ✅ Full access | 100% |
| TEFAS funds (836+) | ❌ Not implemented | ✅ `get_fund_data` | **Gap** |

### 7. News & Announcements

| Feature | Your Implementation | Borsa-MCP | Overlap |
|---------|---------------------|-----------|---------|
| KAP news | ✅ `get_stock_news()` | ✅ `get_news` | 100% |
| News detail lookup | ❌ Basic only | ✅ Detailed lookup | Partial |

### 8. Crypto & FX

| Feature | Your Implementation | Borsa-MCP | Overlap |
|---------|---------------------|-----------|---------|
| USD/TRY | ✅ `get_market_summary()` | ✅ `get_fx_data` | 100% |
| 65 currency pairs | ❌ Limited | ✅ Full access | **Gap** |
| Commodities | ❌ Not implemented | ✅ `get_fx_data` | **Gap** |
| BtcTurk crypto | ❌ Not implemented | ✅ `get_crypto_market` | **Gap** |
| Coinbase crypto | ❌ Not implemented | ✅ `get_crypto_market` | **Gap** |

---

## Summary: Coverage Analysis

### What You Already Have (No Need to Integrate)

| Category | Coverage |
|----------|----------|
| Real-time quotes | ✅ Complete |
| Historical prices | ✅ Complete |
| Technical indicators | ✅ Complete (7 indicators + batch) |
| Economic calendar | ✅ Complete |
| TCMB data (inflation, rates) | ✅ Complete |
| Bond yields | ✅ Complete |
| KAP news | ✅ Complete |
| Portfolio analytics | ✅ **You're ahead** (borsa-mcp doesn't have this) |
| Multi-indicator panels | ✅ **You're ahead** |
| Eurobonds | ✅ **You're ahead** |

### Gaps That Borsa-MCP Could Fill

| Feature | Priority | Reason |
|---------|----------|--------|
| **Stock screener** | 🔴 High | Your SSL issue blocks this; MCP works |
| **TEFAS funds** | 🟡 Medium | 836+ funds for diversification analysis |
| **Crypto data** | 🟡 Medium | BtcTurk + Coinbase integration |
| **US stocks** | 🟡 Medium | NYSE/NASDAQ for comparison |
| **Pivot points** | 🟢 Low | Easy to implement locally |
| **Sector comparison** | 🟢 Low | Nice-to-have for analysis |
| **Financial statements** | 🔴 High | Your borsapy returns empty; MCP may work |
| **FX/Commodities (65 pairs)** | 🟡 Medium | Gold, oil, more currencies |

---

## Integration Recommendations

### Option 1: Direct MCP Integration for LLM Agents (Recommended)

**Use Case**: Your Vercel app has LLM agents that need to query financial data

**How It Works**:
```
User Query → LLM Agent → MCP Client → borsamcp.fastmcp.app → Data Response
```

**Implementation**:
```typescript
// In your Next.js API route
const MCP_ENDPOINT = "https://borsamcp.fastmcp.app/mcp";

// Your LLM agent calls MCP tools
const response = await fetch(MCP_ENDPOINT, {
  method: "POST",
  body: JSON.stringify({
    tool: "get_quick_info",
    params: { symbol: "THYAO" }
  })
});
```

**Pros**:
- No code changes to your borsapy integration
- 26 tools ready for LLM agents
- Fills your gaps (screener, funds, crypto)

**Cons**:
- External dependency
- Network latency
- May have rate limits

### Option 2: Selective Feature Adoption

**Cherry-pick only the missing features**:

1. **Stock Screener** - Use MCP's `screen_securities` when your SSL is blocked
2. **TEFAS Funds** - Use MCP's `get_fund_data` for fund analysis
3. **Crypto** - Use MCP's `get_crypto_market` for BtcTurk/Coinbase

**Keep your existing**:
- Real-time quotes (faster, local cache)
- Technical indicators (batch processing)
- Portfolio analytics (not in MCP)

### Option 3: Fork and Self-Host Borsa-MCP

**If you want full control**:
1. Fork `github.com/saidsurucu/borsa-mcp`
2. Deploy to your own infrastructure
3. Add custom tools for your factor signals
4. Integrate with your existing borsapy client

---

## LLM Agent Integration Architecture

### Current State (Your App)
```
┌─────────────────────────────────────────────────┐
│                 Vercel App                       │
│  ┌───────────┐    ┌───────────────────────────┐ │
│  │ LLM Agent │────│ /api/realtime (Python)    │ │
│  └───────────┘    │ /api/factor-lab (Python)  │ │
│                   └───────────────────────────┘ │
│                              │                   │
│                   ┌──────────▼──────────┐       │
│                   │  borsapy_client.py  │       │
│                   │  realtime_stream.py │       │
│                   │  macro_events.py    │       │
│                   └─────────────────────┘       │
└─────────────────────────────────────────────────┘
```

### With Borsa-MCP Integration
```
┌─────────────────────────────────────────────────┐
│                 Vercel App                       │
│  ┌───────────┐                                  │
│  │ LLM Agent │──────────┬───────────────────┐  │
│  └───────────┘          │                   │  │
│        │                │                   │  │
│        ▼                ▼                   ▼  │
│  ┌──────────┐    ┌────────────┐    ┌─────────┐│
│  │ Your API │    │ Borsa-MCP  │    │ Factor  ││
│  │ (quotes, │    │ (screener, │    │ Lab API ││
│  │ technicals)   │ funds,     │    │         ││
│  │          │    │ crypto)    │    │         ││
│  └──────────┘    └────────────┘    └─────────┘│
└─────────────────────────────────────────────────┘
```

### Tool Routing Logic
```typescript
// In your LLM agent's tool handler
function routeToolCall(tool: string, params: any) {
  // Use local implementation (faster, cached)
  const LOCAL_TOOLS = [
    'get_quote', 'get_historical', 'get_rsi',
    'get_macd', 'portfolio_analytics', 'factor_signals'
  ];

  // Use MCP for gaps
  const MCP_TOOLS = [
    'screen_securities', 'get_fund_data',
    'get_crypto_market', 'get_pivot_points'
  ];

  if (LOCAL_TOOLS.includes(tool)) {
    return callLocalAPI(tool, params);
  } else if (MCP_TOOLS.includes(tool)) {
    return callBorsaMCP(tool, params);
  }
}
```

---

## Implementation Priority

### Phase 1: Quick Wins (1-2 days)
- [ ] Add MCP client utility for Vercel
- [ ] Integrate `screen_securities` to bypass SSL issue
- [ ] Test MCP endpoint reliability

### Phase 2: LLM Agent Enhancement (3-5 days)
- [ ] Define tool schemas for your LLM agents
- [ ] Route calls between local borsapy and MCP
- [ ] Add TEFAS fund queries via MCP

### Phase 3: Full Integration (1 week)
- [ ] Add crypto data via MCP
- [ ] Implement FX/commodities queries
- [ ] Create unified tool catalog for agents

---

## Conclusion

**Should you integrate borsa-mcp?**

| Scenario | Recommendation |
|----------|----------------|
| LLM agents need real-time tool calling | ✅ Yes, use MCP for agent tools |
| Need stock screener (your SSL is broken) | ✅ Yes, use MCP's screener |
| Need TEFAS/crypto/FX data | ✅ Yes, fills your gaps |
| Just need quotes/technicals for dashboard | ❌ No, your borsapy is sufficient |
| Need portfolio analytics | ❌ No, MCP doesn't have this |
| Need batch technical indicators | ❌ No, your implementation is better |

**Bottom Line**: Use a **hybrid approach** - keep your borsapy for what it does well (quotes, technicals, portfolio), and use borsa-mcp for what you're missing (screener, funds, crypto, LLM tool interface).

---

## Appendix: Borsa-MCP Tools Reference

### All 26 Tools

| # | Tool | Description |
|---|------|-------------|
| 1 | `search_symbol` | Search stocks, indices, funds, crypto |
| 2 | `get_profile` | Company info, sector, description |
| 3 | `get_quick_info` | P/E, P/B, ROE, 52-week range |
| 4 | `get_historical_data` | OHLCV price data |
| 5 | `get_technical_analysis` | RSI, MACD, Bollinger, MAs |
| 6 | `get_pivot_points` | Support/resistance levels |
| 7 | `get_analyst_data` | Ratings and price targets |
| 8 | `get_dividends` | Dividend history and yield |
| 9 | `get_earnings` | Earnings calendar, EPS |
| 10 | `get_financial_statements` | Balance sheet, income, cash flow |
| 11 | `get_financial_ratios` | Valuation & health metrics |
| 12 | `get_corporate_actions` | Capital increases, dividends |
| 13 | `get_news` | KAP news with detail lookup |
| 14 | `screen_securities` | 23 presets + custom filters |
| 15 | `scan_stocks` | Technical scanner |
| 16 | `get_crypto_market` | BtcTurk + Coinbase data |
| 17 | `get_fx_data` | 65 currency pairs, commodities |
| 18 | `get_economic_calendar` | Events for 7 countries |
| 19 | `get_bond_yields` | TR government bonds |
| 20 | `get_sector_comparison` | Sector average metrics |
| 21 | `get_fund_data` | TEFAS funds (836+) |
| 22 | `get_index_data` | BIST and US indices |
| 23 | `get_macro_data` | TCMB inflation data |
| 24 | `get_screener_help` | Screener documentation |
| 25 | `get_scanner_help` | Scanner documentation |
| 26 | `get_regulations` | Fund regulations |
