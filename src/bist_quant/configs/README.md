# `configs/` — Strategy Registry

## Purpose

Holds the master YAML configuration registry for all trading strategies. This is the single source of truth for which strategies are enabled, their parameters, and their default portfolio engineering settings.

## Files

```
configs/
└── strategies.yaml    # Master strategy registry (~850 lines, ~40+ strategies)
```

---

## `strategies.yaml` — Strategy Registry

### Structure

Each strategy entry follows this schema:

```yaml
strategies:
  <strategy_name>:
    description: "Human-readable description"
    enabled: true                     # false = skipped by all runners
    rebalance_frequency: monthly      # monthly | quarterly | weekly
    timeline:
      start_date: "YYYY-MM-DD"
      end_date:   "YYYY-MM-DD"
    parameters:                       # optional, strategy-specific
      fast_window: 12
      slow_window: 26
    portfolio_options:
      top_n: 20
      max_position_weight: 0.25
      liquidity_quantile: 0.25
      target_downside_vol: 0.2
      use_regime_filter: true
      use_liquidity_filter: true
      use_inverse_vol_sizing: false
      use_slippage: true
      slippage_bps: 5.0
      use_stop_loss: false
      stop_loss_threshold: 0.15
      use_vol_targeting: false
      vol_lookback: 63
      vol_floor: 0.1
      vol_cap: 1.0
      inverse_vol_lookback: 60
```

### Registered Strategies (Partial)

| Strategy | Category | Key parameters |
|---|---|---|
| `momentum` | Momentum | 12-1 month window |
| `consistent_momentum` | Momentum | Multi-period |
| `residual_momentum` | Momentum | Beta-adjusted |
| `value` | Value | P/E, P/B, EV/EBITDA weights |
| `profitability` | Quality | Operating + gross margin |
| `earnings_quality` | Quality | Accruals |
| `fscore_reversal` | Quality | Piotroski F-score |
| `five_factor_rotation` | Composite | Fama-French 5 factors |
| `betting_against_beta` | Composite | Low-beta premium |
| `breakout_value` | Composite | Value + breakout |
| `quality_momentum` | Composite | Quality + momentum |
| `sma` | Technical | SMA crossover |
| `donchian` | Technical | Donchian channel |
| `macd` | Technical | MACD histogram |
| `supertrend` | Technical | ATR-based trend |
| `size_rotation` | Composite | Large → small cap rotation |
| (25+ more…) | | |

### Timeline Conventions

- Most strategies start at `2014-01-01` or `2017-01-01` depending on fundamentals data availability.
- End date is typically `2026-12-31` (open-ended).
- Strategies using earnings/accrual signals start later (`2017+`) due to fundamentals data depth.

---

## `common/config_manager.py` Integration

`ConfigManager.load_signal_configs()` loads this file:
- Python-format config modules (`configs/*.py`) take precedence.
- Falls back to YAML if no Python module exists.
- `ConfigManager.deep_merge(defaults, overrides)` applies `DEFAULT_PORTFOLIO_OPTIONS` as the base.

---

## Local Rules for Contributors

1. **`use_regime_filter: true` is the default** for almost all strategies. Only override to `false` if the strategy is specifically designed for all-market conditions.
2. **All new strategies must have a `description` field.** This is shown in the UI and the API.
3. **`enabled: false` is a soft disable** — the strategy stays in the registry but is skipped. Do not delete strategies — disable them.
4. **`top_n` should be between 10 and 30** for most strategies. Values below 10 produce highly concentrated portfolios; values above 50 dilute alpha.
5. **Timeline start dates matter.** If a strategy depends on fundamental data (P/E, ROE, accruals, etc.), the start date must be at least `2017-01-01`. Price-only signals can start from `2014-01-01`.
6. **`slippage_bps: 5.0` is the minimum.** Do not set slippage below 5 bps for BIST — transaction costs are higher than US markets.
7. **`max_position_weight: 0.25` is a hard cap** — no single position should ever exceed 25% of the portfolio. Do not exceed this without explicit justification.
