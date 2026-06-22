# Midas Portfolio Risk Metrics

Personal portfolio risk analytics built on top of `bist-quant`. Parses
**Midas Hesap Ekstresi** PDF statements to reconstruct trades, cash flows,
and dividends, then computes return and risk metrics.

## What it does

1. **Parses Midas PDFs** (`Hesap Hareketleri` + `Hesap Ekstresi` table
   layouts) using `pdfplumber`. Extracts per-month:
   - Executed trades (with exact fill price, time, fees, order type)
   - Cash flows (deposits, withdrawals, nema, promosyon, FX, stopaj)
   - Dividends (gross/net)
   - End-of-month portfolio snapshot

2. **Fetches prices** via `borsapy.Ticker.history()` for mark-to-market
   valuation. Cached as parquet for fast re-runs.

3. **Replays the portfolio** day-by-day with real TRY cash flows,
   handling weekend execution, deduplicating month-spanning trades.

4. **Computes metrics**:
   - **Performance**: absolute return, MWR/IRR (annualized)
   - **Per-trade stats** (FIFO-matched): win rate, profit factor, expectancy
   - **Risk metrics** (holdings-only daily series, winsorized ±50%):
     volatility, Sharpe, Sortino, max drawdown, VaR, CVaR, beta vs XU100

## Usage

```bash
python -m scripts.midas_risk_metrics.risk_report \
    --midas-dir /path/to/Midas \
    --cache-dir /path/to/.price_cache \
    --output /path/to/risk_metrics_midas.json
```

## Files

- `risk_report.py` — main entry point, metric computations, equity curve
- `midas_pdf_parser.py` — PDF table extraction (Yatırım İşlemleri,
  Hesap İşlemleri, Temettü İşlemleri, Portföy Özeti)

## Honest caveats

- **Risk metrics are unreliable for high-turnover portfolios.** The
  holdings-only series is dominated by position-change noise (selling
  80% of holdings looks like an -80% daily return, but you just took
  money out). The `±50%` winsorization is a band-aid, not a fix.
  The absolute return and MWR/IRR are the reliable headline numbers.

- **Per-trade stats are FIFO-matched**, which approximates but does not
  exactly match the broker's cost basis method. For tax reporting
  use the broker's official statements.

- **BIST has a hard ±10% daily price limit.** The Midas fill prices
  are exact; the borsapy close on the same day can differ by a few
  percent because the close is one tick at 17:58.

- **Fund trades (TEFAS)** can be parsed from PDFs but cannot be priced
  reliably because `borsapy.Fund()` requires TEFAS codes (3-letter),
  not the ISINs in Midas statements.

## What this is NOT

- Not a trading tool. No order placement, no live data.
- Not a tax report. Use your broker's official statements for taxes.
- Not a GIPS-compliant performance report. GIPS requires vendor data,
  composite construction, and verification.
