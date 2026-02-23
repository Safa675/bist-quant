# Configuration

Configuration is layered: environment variables → strategies.yaml → runtime `portfolio_options` overrides (deepest wins).

## Environment Variables

Set in a `.env` file or your shell before running:

```bash
# Data locations
export BIST_DATA_DIR=/path/to/data           # prices, fundamentals
export BIST_REGIME_DIR=/path/to/regime       # regime output files
export BIST_CACHE_DIR=/path/to/.cache        # disk cache

# Data source
export BIST_DATA_SOURCE=borsapy              # borsapy (default) or local

# Fundamentals freshness gate
export BIST_ENFORCE_FUNDAMENTAL_FRESHNESS=0  # 1 = enforce gate
export BIST_ALLOW_STALE_FUNDAMENTALS=1       # 1 = bypass gate
export BIST_MAX_MEDIAN_STALENESS_DAYS=90     # default threshold

# Auth (API server)
export BIST_AUTH_MODE=none                   # none | api_key | jwt | either
export BIST_API_KEYS=key1,key2               # comma-separated valid keys
export BIST_JWT_SECRET=your-secret           # JWT signing secret

# Debug
export DEBUG=1                               # verbose per-day backtester logs
```

## Strategy Configuration (`strategies.yaml`)

All registered strategies are defined in `src/bist_quant/configs/strategies.yaml`.
Each entry specifies:

```yaml
strategies:
  momentum:
    description: "12-1 month price momentum factor"
    enabled: true
    rebalance_frequency: monthly
    timeline:
      start_date: "2014-01-01"
      end_date:   "2026-12-31"
    portfolio_options:
      top_n: 20
      max_position_weight: 0.25
      liquidity_quantile: 0.25
      target_downside_vol: 0.20
      use_regime_filter: true
      use_liquidity_filter: true
      use_slippage: true
      slippage_bps: 5.0
      use_stop_loss: false
      stop_loss_threshold: 0.15
      use_vol_targeting: false
```

## Configure via ConfigManager

```python
from bist_quant.common.config_manager import ConfigManager

manager = ConfigManager.from_default_paths()
configs = manager.load_signal_configs()
print(list(configs.keys()))   # all enabled strategy names
```

## Runtime Option Overrides

```python
from bist_quant import get_default_options

# Start from defaults and override specific fields
options = get_default_options()
options["top_n"] = 15
options["use_regime_filter"] = True
options["slippage_bps"] = 10.0

result = engine.run_factor("momentum", portfolio_options=options)
```

## Cache TTL Configuration

All cache TTLs can be overridden via environment variables:

```bash
export BIST_CACHE_TTL_PRICES=14400          # seconds (4 hours default)
export BIST_CACHE_TTL_FINANCIALS=604800     # seconds (7 days default)
export BIST_CACHE_TTL_FX=900               # seconds (15 min default)
export BIST_CACHE_TTL_GOLD=86400           # seconds (1 day default)
```

## Portfolio Options Reference

| Option | Type | Default | Description |
|---|---|---|---|
| `top_n` | int | 20 | Number of stocks to hold |
| `max_position_weight` | float | 0.25 | Maximum single position weight |
| `liquidity_quantile` | float | 0.25 | Filter below this volume quantile |
| `use_regime_filter` | bool | true | Use Bull/Bear regime allocation |
| `use_liquidity_filter` | bool | true | Apply volume liquidity filter |
| `use_inverse_vol_sizing` | bool | false | 1/vol position weighting |
| `use_vol_targeting` | bool | false | Target downside vol via leverage |
| `target_downside_vol` | float | 0.20 | Annual downside vol target |
| `use_slippage` | bool | true | Apply size-aware slippage |
| `slippage_bps` | float | 5.0 | Base slippage (minimum for BIST) |
| `use_stop_loss` | bool | false | Apply stop-loss per position |
| `stop_loss_threshold` | float | 0.15 | Max drawdown from entry before exit |
| `rebalance_frequency` | str | `monthly` | `monthly` or `quarterly` |
| `signal_lag_days` | int | 1 | Days to lag signal (anti-lookahead) |
