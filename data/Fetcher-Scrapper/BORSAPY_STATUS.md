# Borsapy Integration Status

## Overview

This document tracks the integration status of the `borsapy` library into the BIST quantitative trading stack.

**Library Version**: 0.7.2
**Integration Date**: 2026-02-12
**Status**: In Progress

---

## Phase 1: Core Installation & Data Layer ✅

### Files Created
- `data/Fetcher-Scrapper/borsapy_client.py` - Unified wrapper with caching
- `data/Fetcher-Scrapper/test_borsapy_integration.py` - Test suite

### Files Modified
- `Models/common/data_loader.py` - Added borsapy integration methods

### Feature Status

| Feature | Status | Notes |
|---------|--------|-------|
| Index components (XU100, XU030) | ✅ Working | Returns 100/30 stocks correctly |
| Price history (OHLCV) | ✅ Working | All intervals supported |
| Fast info (current quotes) | ✅ Working | 15-min delay by default |
| Technical indicators (RSI, MACD) | ✅ Working | 12+ indicators available |
| Batch download | ✅ Working | Multi-ticker support |
| All indices list | ✅ Working | 81 indices found |
| Stock screener | ⚠️ SSL Issue | Certificate verification failed |
| Financial statements | ⚠️ Empty | Returns empty for tested stocks |
| Dividends | ⚠️ Empty | Returns empty for tested stocks |

### Known Issues

#### 1. SSL Certificate Verification (Stock Screener)
```
SSL: CERTIFICATE_VERIFY_FAILED - unable to get local issuer certificate
```
**Affected**: `screen_stocks()` function
**Cause**: Network/environment SSL configuration
**Workaround**: TBD - may need to configure SSL certificates or use `verify=False`

#### 2. Financial Statements Empty
**Affected**: `balance_sheet`, `income_stmt`, `cashflow` properties
**Tested Stocks**: EREGL, SISE
**Cause**: Unknown - may require different API access or data not available for all stocks
**Workaround**: Continue using existing İş Yatırım scraper for fundamentals

#### 3. Dividends Empty
**Affected**: `dividends` property
**Tested Stocks**: TUPRS
**Cause**: Unknown
**Workaround**: Continue using existing data sources

---

## Phase 2: Technical Analysis Integration ✅

### Files Created
- `Models/signals/borsapy_indicators.py` - Technical indicator wrapper with panel builders
- `Models/signals/test_borsapy_indicators.py` - Test suite

### Available Indicators

| Indicator | Panel Builder | Single Calc | API Fetch |
|-----------|--------------|-------------|-----------|
| RSI | `build_rsi_panel()` | `calculate_rsi()` | ✅ |
| MACD | `build_macd_panel()` | `calculate_macd()` | ✅ |
| Bollinger Bands | `build_bollinger_panel()` | `calculate_bollinger_bands()` | ✅ |
| ATR | `build_atr_panel()` | `calculate_atr()` | ✅ |
| Stochastic | `build_stochastic_panel()` | `calculate_stochastic()` | ✅ |
| ADX | `build_adx_panel()` | `calculate_adx()` | ✅ |
| Supertrend | `build_supertrend_panel()` | `calculate_supertrend()` | ✅ |

### Feature Status

| Feature | Status | Notes |
|---------|--------|-------|
| RSI panel | ✅ Working | 95.9% coverage on real data |
| MACD panel | ✅ Working | histogram/signal/line outputs |
| Bollinger %B | ✅ Working | upper/middle/lower/pct_b |
| ATR panel | ✅ Working | For volatility sizing |
| Stochastic | ✅ Working | %K and %D outputs |
| ADX panel | ✅ Working | Trend strength |
| Supertrend | ✅ Working | Direction signals |
| Multi-indicator builder | ✅ Working | `build_multi_indicator_panel()` |
| API fetch (single) | ✅ Working | `fetch_indicators_for_ticker()` |
| API fetch (batch) | ✅ Working | `fetch_indicators_batch()` |

### Usage Example

```python
from Models.signals.borsapy_indicators import BorsapyIndicators, build_multi_indicator_panel

# Build RSI panel for all tickers
rsi_panel = BorsapyIndicators.build_rsi_panel(close_df, period=14)

# Build multiple indicators at once
panels = build_multi_indicator_panel(
    close_df, high_df, low_df,
    indicators=["rsi", "macd", "bb", "atr", "stoch"]
)

# Fetch via borsapy API
df = BorsapyIndicators.fetch_indicators_for_ticker(
    "THYAO", indicators=["rsi", "macd"], period="2y"
)
```

### Test Results (2026-02-12)

```
RSI panel: 252 days × 603 tickers, 95.9% coverage
Top RSI: IEYHO (98.0), AYES (94.7), DSTKF (93.5)
Bottom RSI: SODSN (12.0), BIGCH (10.3), GENIL (5.0)
```

---

## Phase 3: Stock Screener Integration ⏳

**Status**: Blocked by SSL issue

---

## Phase 4: Real-Time Data & Streaming ✅

### Files Created
- `data/Fetcher-Scrapper/realtime_stream.py` - Real-time quote service with caching
- `data/Fetcher-Scrapper/realtime_api.py` - CLI for dashboard integration
- `data/Fetcher-Scrapper/test_realtime.py` - Test suite
- `bist-quant-ai/src/app/api/realtime/route.ts` - Next.js API endpoint

### Features

| Feature | Status | Notes |
|---------|--------|-------|
| Single quote | ✅ Working | `get_quote(symbol)` |
| Batch quotes | ✅ Working | `get_quotes_batch(symbols)` |
| Index quotes | ✅ Working | `get_index_quotes("XU100")` |
| Portfolio snapshot | ✅ Working | With P&L calculation |
| Market summary | ✅ Working | XU100, XU030, USD/TRY |
| Quote caching | ✅ Working | TTL-based cache |
| Background watcher | ✅ Working | `RealtimeWatcher` class |
| Next.js API | ✅ Working | `/api/realtime` endpoint |

### API Endpoint Usage

```bash
# Single quote
GET /api/realtime?type=quote&symbols=THYAO

# Multiple quotes
GET /api/realtime?type=quotes&symbols=THYAO,AKBNK,GARAN

# Index quotes
GET /api/realtime?type=index&index=XU030

# Market summary
GET /api/realtime?type=market

# Portfolio snapshot (POST)
POST /api/realtime
Body: {"holdings": {"THYAO": 100, "AKBNK": 200}}
```

### Python Usage

```python
from realtime_stream import RealtimeQuoteService

service = RealtimeQuoteService(cache_ttl=60)

# Get quote
quote = service.get_quote("THYAO")
print(f"{quote['symbol']}: {quote['last_price']} TRY")

# Get portfolio snapshot
snapshot = service.get_portfolio_snapshot(
    holdings={"THYAO": 100, "AKBNK": 200},
    cost_basis={"THYAO": 250.0, "AKBNK": 45.0}
)
print(f"Total Value: {snapshot['total_value']:,.2f} TRY")
```

### Known Limitations

1. **15-minute delay**: Default TradingView data has ~15min delay
   - Real-time requires TradingView Pro + BIST package
2. **Market hours**: Some attributes (previous_close) may be None outside trading hours
3. **Rate limiting**: Cache helps reduce API calls (default 60s TTL)

---

## Phase 5: Portfolio Analytics Enhancement ✅

### Files Created
- `Models/analytics/__init__.py` - Module exports
- `Models/analytics/portfolio_metrics.py` - Comprehensive risk/return metrics
- `Models/analytics/test_portfolio_metrics.py` - Test suite

### Files Modified
- `Models/common/data_loader.py` - Added `create_portfolio_analytics()` and `analyze_strategy_performance()`

### Available Metrics

| Metric | Function | Description |
|--------|----------|-------------|
| Sharpe Ratio | `calculate_sharpe_ratio()` | Risk-adjusted return vs risk-free |
| Sortino Ratio | `calculate_sortino_ratio()` | Downside risk-adjusted return |
| Max Drawdown | `calculate_max_drawdown()` | Largest peak-to-trough decline |
| Beta | `calculate_beta()` | Market sensitivity |
| Alpha | `calculate_alpha()` | Risk-adjusted excess return (Jensen's) |
| Calmar Ratio | `calculate_calmar_ratio()` | CAGR / Max Drawdown |
| Information Ratio | `calculate_information_ratio()` | Active return / Tracking error |
| VaR (95%) | `calculate_var()` | Value at Risk |
| CVaR (95%) | `calculate_cvar()` | Expected Shortfall |
| Correlation Matrix | `calculate_correlation_matrix()` | Asset correlations |
| Rolling Metrics | `calculate_rolling_metrics()` | Time-varying metrics |

### PortfolioAnalytics Class

```python
from Models.analytics import PortfolioAnalytics

# From returns series
analytics = PortfolioAnalytics(
    returns=portfolio_returns,
    benchmark_returns=benchmark_returns,
    name="My Portfolio"
)

# From holdings and price data
analytics = PortfolioAnalytics.from_holdings(
    holdings={"THYAO": 100, "AKBNK": 200},
    close_df=price_data,
    benchmark_col="XU100"
)

# Get all metrics
metrics = analytics.get_all_metrics()

# Get formatted summary
print(analytics.summary())
```

### DataLoader Integration

```python
# Create analytics from holdings
analytics = loader.create_portfolio_analytics(
    holdings={"THYAO": 100, "AKBNK": 200},
    benchmark="XU100",
    name="My Portfolio"
)

# Analyze strategy performance
analytics = loader.analyze_strategy_performance(
    equity_curve=strategy_equity,
    benchmark_curve=xu100_equity
)
```

### Test Results (2026-02-12)

```
Real BIST Portfolio (5 stocks, 2 years):
  CAGR: 29.02%
  Volatility: 26.71%
  Sharpe: -0.17 (high Turkish risk-free rate ~40%)
  Max Drawdown: -22.39%
```

---

## Phase 6: Derivatives & Fixed Income ⏳

**Status**: Not started

---

## Phase 7: Economic Calendar & News ⏳

**Status**: Not started

---

## Quick Reference

### Usage Examples

```python
# Via DataLoader
from Models.common.data_loader import DataLoader

loader = DataLoader(data_dir="data", regime_model_dir="Regime Filter")

# Get index components
xu100 = loader.get_index_components_borsapy("XU100")

# Load prices
prices = loader.load_prices_borsapy(symbols=["THYAO", "AKBNK"], period="1y")

# Get indicators
df = loader.get_history_with_indicators_borsapy("GARAN", indicators=["rsi", "macd"])

# Direct client access
client = loader.borsapy
quote = client.get_fast_info("THYAO")
```

### Running Tests
```bash
python data/Fetcher-Scrapper/test_borsapy_integration.py
```

---

## Post-Integration TODO

- [ ] Investigate SSL certificate issue for stock screener
- [ ] Test financial statements with more stocks
- [ ] Test dividend data with known dividend-paying stocks
- [ ] Compare borsapy prices vs yfinance for validation
- [ ] Benchmark performance (borsapy vs existing fetchers)
- [ ] Add retry logic for transient API failures
