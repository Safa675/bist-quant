# Portfolio Options Example

This example demonstrates configuring portfolio construction controls via `portfolio_options`.

## Equal Weight (Default)

```python
from bist_quant import PortfolioEngine, DataLoader, DataPaths

loader = DataLoader(data_paths=DataPaths())
engine = PortfolioEngine(loader=loader)

# Default: equal-weight top-20, monthly rebalance, no regime filter
result_ew = engine.run_factor("composite")
print(f"Equal-weight Sharpe: {result_ew.metrics['sharpe']:.2f}")
```

## Inverse Volatility Weighting

```python
result_vol = engine.run_factor("composite", portfolio_options={
    "use_inverse_vol_sizing": True,
    "max_position_weight": 0.20,
})
print(f"Inverse-vol Sharpe: {result_vol.metrics['sharpe']:.2f}")
```

## Volatility-Targeted Portfolio

```python
result_vt = engine.run_factor("momentum", portfolio_options={
    "use_vol_targeting": True,
    "target_downside_vol": 0.15,     # target 15% annualized downside vol
    "use_regime_filter": True,       # switch to gold in Bear/Stress
    "use_slippage": True,
    "slippage_bps": 5.0,
    "top_n": 20,
    "rebalance_frequency": "monthly",
})
print(f"Vol-targeted Sharpe: {result_vt.metrics['sharpe']:.2f}")
print(f"Vol-targeted CAGR:   {result_vt.metrics['cagr']:.1%}")
```

## Regime-Filtered All-Weather

```python
result_regime = engine.run_factor("momentum", portfolio_options={
    "use_regime_filter": True,        # Bear/Stress → gold (XAU/TRY)
    "top_n": 15,
    "max_position_weight": 0.15,
})
print(result_regime.regime_history.value_counts())
```

## Comparing Configurations

```python
configs = {
    "equal_weight":    {},
    "inverse_vol":     {"use_inverse_vol_sizing": True},
    "vol_targeted":    {"use_vol_targeting": True, "target_downside_vol": 0.15},
    "regime_filtered": {"use_regime_filter": True},
}

for label, opts in configs.items():
    r = engine.run_factor("composite", portfolio_options=opts)
    print(f"{label:20s}  Sharpe={r.metrics['sharpe']:.2f}  DD={r.metrics['max_drawdown']:.1%}")
```
