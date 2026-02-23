# Portfolio Management

## PortfolioEngine

`PortfolioEngine` is the main orchestration class. It wraps `Backtester`, `RiskManager`, and `RegimeClassifier`.

```python
from bist_quant import PortfolioEngine, DataLoader, DataPaths

loader = DataLoader(data_paths=DataPaths())
engine = PortfolioEngine(loader=loader)
```

## Running a Strategy

```python
# By registered name (loads strategy params from strategies.yaml)
result = engine.run_factor("composite")

# With custom portfolio_options override
result = engine.run_factor("momentum", portfolio_options={
    "top_n": 15,
    "use_regime_filter": True,
    "use_vol_targeting": True,
    "target_downside_vol": 0.18,
    "rebalance_frequency": "quarterly",
})
```

## Position Sizing Methods

| Setting | Behavior |
|---|---|
| `use_inverse_vol_sizing=False` (default) | Equal weight across top-N |
| `use_inverse_vol_sizing=True` | 1/σ weighting (inverse realized vol) |
| `use_vol_targeting=True` | Scale entire portfolio to hit `target_downside_vol` |

`max_position_weight` caps any single position regardless of sizing method.
Weight sum is checked post-sizing and any excess is redistributed.

## Rebalancing

Available: `monthly` or `quarterly`.
No daily or weekly rebalancing — BIST transaction costs make high-frequency impractical.

## Liquidity Filter

```python
options["use_liquidity_filter"] = True
options["liquidity_quantile"] = 0.25   # remove bottom 25% by 30-day volume
```

## Multi-Factor Composite

```python
# composite = equal-weight average of all factor scores before ranking
result = engine.run_factor("composite")

# AHP-weighted composite (weights from strategies.yaml or custom)
result = engine.run_factor("ahp_composite", portfolio_options={
    "factor_weights": {"momentum": 0.4, "value": 0.3, "quality": 0.3}
})
```

## Benchmark Comparison

By default, results include the BIST XU100 total return index as a benchmark.
The benchmark is loaded from `loader.load_index_prices("XU100")`.
```
