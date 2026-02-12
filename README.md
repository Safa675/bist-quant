# BIST Quant Research Repository

A comprehensive Borsa Istanbul (BIST) quantitative research stack for systematic trading — combining multi-factor portfolio construction, regime-adaptive risk management, and an AI-powered web dashboard.

## Repository Structure

```text
BIST/
├── Models/                          # Multi-factor portfolio engine
│   ├── portfolio_engine.py          # Core backtest engine
│   ├── build_factor_construction_cache.py
│   ├── analyze_2026_daily.py
│   ├── common/
│   │   ├── data_loader.py           # Unified data loading (parquet + xlsx)
│   │   └── utils.py                 # Shared helpers (TTM, lag, z-score, etc.)
│   ├── configs/                     # 34 strategy configurations
│   │   ├── breakout_value.py        # Donchian breakout × value (102% CAGR)
│   │   ├── small_cap_momentum.py    # Size × momentum (95% CAGR)
│   │   ├── five_factor_rotation.py  # Dynamic 13-axis rotation (88% CAGR)
│   │   ├── trend_value.py           # Trend following × value (88% CAGR)
│   │   ├── ...                      # 30 more strategies
│   │   └── xu100.py                 # Benchmark (buy & hold XU100)
│   ├── signals/                     # Signal generation modules
│   │   ├── five_factor_rotation_signals.py  # 13-axis multi-factor engine
│   │   ├── factor_builders.py       # Quality, liquidity, sentiment panels
│   │   ├── factor_axes.py           # Quintile scoring, MWU weighting
│   │   └── ...                      # Per-strategy signal generators
│   └── results/                     # Backtest outputs (gitignored)
│
├── Regime Filter/                   # Market regime detection
│   ├── simple_regime.py             # 2D classifier (trend × volatility)
│   ├── run_evaluation.py            # Backtest & evaluation harness
│   ├── tune_variants.py             # Parameter tuning
│   └── outputs/                     # Regime outputs (gitignored)
│
├── data/                            # Market & fundamental data
│   ├── Fetcher-Scrapper/            # Data scrapers & fetchers
│   │   ├── update_prices.py         # Daily price updater (İş Yatırım)
│   │   ├── is_yatirim_fetcher.py    # İş Yatırım price scraper
│   │   ├── bist_fundamental_scrapper.py  # Fundamental data scraper
│   │   ├── fintables_fetch.py       # Fintables.com scraper
│   │   ├── tcmb_data_fetcher.py     # TCMB macro data (EVDS API)
│   │   ├── tcmb_rates.py            # Exchange rate fetcher
│   │   ├── fongetiri_gold_funds_fetcher.py  # Gold fund NAV fetcher
│   │   └── xu100_fetcher.py         # XU100 index data
│   ├── fundamental_data/            # Raw XLSX fundamental files (gitignored)
│   ├── xu100_prices.csv             # XU100 index prices
│   ├── xu100_prices.parquet
│   ├── fundamental_metrics.parquet  # Consolidated fundamental metrics
│   ├── gold_funds_daily_prices.parquet
│   ├── usdtry_data.csv             # USD/TRY exchange rates
│   ├── xau_try_2013_2026.csv       # Gold in TRY
│   └── tcmb_indicators.csv         # TCMB macro indicators
│
├── bist-quant-ai/                   # Next.js web dashboard & AI agents
│
├── cache/                           # Runtime caches (gitignored)
└── .gitignore
```

### Large Data Files (gitignored, regenerable)

| File | Size | Description |
|------|------|-------------|
| `data/bist_prices_full.parquet` | ~26 MB | Full BIST daily OHLCV prices |
| `data/consolidated_isyatirim_prices.parquet` | ~90 MB | İş Yatırım consolidated prices |
| `data/fundamental_data_consolidated.parquet` | ~12 MB | Consolidated fundamental data |
| `data/multi_factor_axis_construction.parquet` | ~140 MB | Factor axis cache (auto-rebuilt) |
| `data/five_factor_axis_construction.parquet` | ~4.5 MB | Five-factor axis cache |

## Setup

```bash
cd /home/safa/Documents/Markets/BIST
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy pyarrow openpyxl scipy
```

## Factor Backtest Engine

The portfolio engine supports 34 factor strategies. Run any strategy by name:

```bash
cd Models
python portfolio_engine.py breakout_value    # Run single strategy
python portfolio_engine.py all               # Run all 34 strategies
```

Backtest outputs are written to `Models/results/<factor_name>/` and include:
- `summary.txt` — Performance metrics (CAGR, Sharpe, drawdown, CAPM alpha/beta)
- `yearly_metrics.csv` — Year-by-year returns breakdown
- `returns.csv` — Daily returns vs XU100 and XAU/TRY benchmarks

### Results Summary (as of February 2026)

34 strategies backtested across 2017–2026 on BIST. Top performers:

| Factor | Total Return | CAGR | Max Drawdown | Sharpe | Category |
| --- | ---: | ---: | ---: | ---: | --- |
| `breakout_value` | 65,373% | 102.62% | -31.47% | 2.93 | Breakout × Value |
| `small_cap_momentum` | 46,472% | 95.24% | -26.92% | 2.47 | Size × Momentum |
| `five_factor_rotation` | 4,554% | 88.00% | -29.09% | 2.47 | Multi-Factor |
| `trend_value` | 23,896% | 88.17% | -35.96% | 2.66 | Trend × Value |
| `small_cap` | 26,616% | 83.77% | -35.19% | 2.48 | Size |
| `trend_following` | 33,096% | 81.78% | -28.58% | 2.43 | Trend |
| `donchian` | 156,766% | 81.98% | -26.90% | 2.49 | Trend (Channel) |
| `quality_momentum` | 20,491% | 78.63% | -33.48% | 2.37 | Quality × Momentum |
| `quality_value` | 16,677% | 74.69% | -39.90% | 2.32 | Quality × Value |
| `asset_growth` | 16,604% | 74.61% | -37.00% | 2.33 | Growth |
| `profitability` | 11,590% | 67.95% | -35.34% | 2.18 | Fundamental |
| `roa` | 11,957% | 68.52% | -36.77% | 2.16 | Fundamental |
| `size_rotation` | 47,737% | 65.22% | -33.02% | 2.02 | Size Rotation |
| `accrual` | 8,508% | 62.45% | -36.65% | 2.05 | Fundamental |
| `macro_hedge` | 6,532% | 57.90% | -30.19% | 2.04 | Macro |
| `dividend_rotation` | 5,517% | 55.07% | -34.41% | 1.95 | Income |
| `value` | 4,908% | 53.14% | -35.60% | 1.86 | Value |
| `momentum` | 6,701% | 40.97% | -40.61% | 1.54 | Momentum |
| `xu100` (benchmark) | 4,099% | 36.44% | -34.01% | 1.47 | Benchmark |

**Average across all 34 strategies**: ~63% CAGR, ~2.06 Sharpe

For complete details inspect:
- `Models/results/factor_correlation_matrix.csv`
- `Models/results/capm_summary.csv`
- `Models/results/<factor_name>/summary.txt`

### Strategy Categories

| Category | Strategies |
|----------|-----------|
| **Value** | `value`, `quality_value`, `breakout_value`, `trend_value` |
| **Momentum** | `momentum`, `consistent_momentum`, `residual_momentum`, `small_cap_momentum` |
| **Quality** | `profitability`, `quality_momentum`, `roa`, `accrual`, `earnings_quality` |
| **Size** | `small_cap`, `size_rotation`, `size_rotation_momentum`, `size_rotation_quality` |
| **Trend** | `trend_following`, `donchian`, `sma` |
| **Fundamental** | `investment`, `asset_growth`, `fscore_reversal` |
| **Multi-Factor** | `five_factor_rotation` (13 axes), `macro_hedge`, `dividend_rotation` |
| **Risk/Vol** | `low_volatility`, `betting_against_beta`, `momentum_reversal_volatility` |
| **Sector/Pairs** | `sector_rotation`, `pairs_trading` |
| **Short-Term** | `short_term_reversal`, `momentum_asset_growth` |
| **Benchmark** | `xu100` (buy & hold) |

## Regime Filter

A simple 2-dimensional regime classifier for market state detection:

- **Dimension 1**: Trend — Price vs moving average
- **Dimension 2**: Volatility — Realized vol percentile

Produces 4 regimes: `Bull`, `Bear`, `Recovery`, `Stress`

The regime filter is integrated into `macro_hedge` and other strategies through the portfolio engine, enabling automatic rotation between equities and gold based on market conditions.

```bash
cd "Regime Filter"
python simple_regime.py          # Run regime classification
python run_evaluation.py         # Evaluate regime strategy
python tune_variants.py          # Tune parameters
```

## Data Pipeline

Update market data:

```bash
cd data/Fetcher-Scrapper
python update_prices.py          # Update daily BIST prices
python xu100_fetcher.py          # Update XU100 index
python tcmb_data_fetcher.py      # Update TCMB macro data
```

All parquet files use **zstd compression** for efficient storage.

## Web Dashboard

See [`bist-quant-ai/README.md`](bist-quant-ai/README.md) for the Next.js web platform with AI-powered multi-agent trading intelligence.

## License

Proprietary — All rights reserved.
