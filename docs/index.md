# BIST Quant

**Quantitative research and backtesting library for Borsa Istanbul (BIST).** Version 0.3.1.

`bist_quant` is a library-first toolkit for BIST research workflows: factor construction, event-loop backtesting, portfolio analytics, and provider integrations that can be used from notebooks, scripts, dashboards, or external applications.

## What's Inside

- **40+ signal builders** — momentum, value, quality, technical, and composite multi-factor strategies
- **Event-loop backtester** — day-by-day with monthly/quarterly rebalancing, regime filter, vol-targeting, stop-loss, and realistic slippage
- **Fundamentals pipeline** — automated fetch → validate → merge → freshness-gate cycle for Turkish financial statements
- **Regime classifier** — 2D (trend × volatility) model producing Bull / Bear / Recovery / Stress labels for BIST XU100
- **Professional analytics** — GARCH volatility, walk-forward analysis, Monte Carlo, Kelly sizing, factor exposure, compliance checks
- **Realtime quotes** — TradingView streaming overlay with borsapy polling fallback
- **Multi-asset data** — borsapy (BIST), Borsa MCP (crypto, US stocks, TEFAS funds), FX, gold, derivatives, fixed income

## Installation

```bash
pip install bist-quant
```

## Quick Example

```python
from bist_quant import PortfolioEngine, DataLoader, DataPaths

paths = DataPaths()
loader = DataLoader(data_paths=paths)
engine = PortfolioEngine(
    data_loader=loader,
    options={"use_regime_filter": False},
)

result = engine.run_factor("momentum")  # runs full backtest
print(f"Sharpe: {result['sharpe']:.2f}")
print(f"CAGR:   {result['cagr']:.1%}")
```

Default library directories:

- data: `~/.local/share/bist-quant/data`
- regime outputs: `~/.local/share/bist-quant/regime/simple_regime`
- cache: `~/.cache/bist-quant`

## Documentation

- [Installation & Setup](getting-started/installation.md)
- [Quick Start](getting-started/quickstart.md)
- [Configuration](getting-started/configuration.md)
- [Signals Guide](user-guide/signals.md)
- [Backtesting Guide](user-guide/backtesting.md)
- [Portfolio & Risk](user-guide/portfolio.md)
- [Multi-Asset Data](user-guide/multi-asset.md)
- [Examples](examples/basic-backtest.md)
- [Contributing](contributing.md)

## Source Code Documentation

Every folder under `src/bist_quant/` has a `README.md` with file-level API reference and local rules:

| Module | README |
|---|---|
| Core package | [src/bist_quant/README.md](../src/bist_quant/README.md) |
| Analytics | [src/bist_quant/analytics/README.md](../src/bist_quant/analytics/README.md) |
| Clients | [src/bist_quant/clients/README.md](../src/bist_quant/clients/README.md) |
| Common | [src/bist_quant/common/README.md](../src/bist_quant/common/README.md) |
| Data Pipeline | [src/bist_quant/data_pipeline/README.md](../src/bist_quant/data_pipeline/README.md) |
| Engines | [src/bist_quant/engines/README.md](../src/bist_quant/engines/README.md) |
| Regime | [src/bist_quant/regime/README.md](../src/bist_quant/regime/README.md) |
| Signals | [src/bist_quant/signals/README.md](../src/bist_quant/signals/README.md) |

## License

Apache License 2.0 - see [LICENSE](https://github.com/Safa675/bist-quant/blob/main/LICENSE).
