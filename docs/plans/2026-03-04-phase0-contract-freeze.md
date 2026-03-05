# Phase 0 Contract Freeze — Streamlit -> Next.js (2026-03-04)

## Summary
- Purpose: freeze frontend↔backend contracts before Wave 1 fixes.
- Scope: all Next.js routes (`/` + 10 product routes) and global `AppShell` API dependency.
- Backend source of truth: FastAPI route implementations and OpenAPI metadata.
- Canonical sample strategy: mocked live harness (deterministic route invocation with patched heavy services).

## Artifacts
- Manifest: [`artifacts/phase0-contract-manifest.json`](./artifacts/phase0-contract-manifest.json)
- Samples: [`artifacts/phase0-canonical-samples.json`](./artifacts/phase0-canonical-samples.json)

## Route Coverage Matrix
| Route | Execution | Status | Next Page | Streamlit Source | Consumed Endpoints |
| --- | --- | --- | --- | --- | --- |
| `/` | `redirect` | `match` | `page.tsx` | `1_Dashboard.py` | - |
| `/dashboard` | `sync` | `match` | `page.tsx` | `1_Dashboard.py` | `GET /api/dashboard/overview` |
| `/backtest` | `async_job` | `adapter` | `page.tsx` | `2_Backtest.py` | `DELETE /api/jobs/{job_id}`<br>`GET /api/jobs`<br>`GET /api/jobs/{job_id}`<br>`GET /api/meta/signals`<br>`POST /api/jobs` |
| `/factor-lab` | `async_job` | `mismatch` | `page.tsx` | `3_Factor_Lab.py` | `GET /api/factors/{name}`<br>`GET /api/jobs/{job_id}`<br>`GET /api/meta/signals`<br>`POST /api/jobs` |
| `/signal-construction` | `async_job` | `adapter` | `page.tsx` | `4_Signal_Construction.py` | `GET /api/jobs/{job_id}`<br>`GET /api/meta/signals`<br>`POST /api/jobs` |
| `/screener` | `sync` | `adapter` | `page.tsx` | `5_Screener.py` | `POST /api/screener/run` |
| `/analytics` | `async_job_noncanonical` | `mismatch` | `page.tsx` | `6_Analytics.py` | `GET /api/jobs/{job_id}`<br>`POST /api/jobs` |
| `/optimization` | `async_job` | `mismatch` | `page.tsx` | `7_Optimization.py` | `GET /api/jobs/{job_id}`<br>`GET /api/meta/signals`<br>`POST /api/jobs` |
| `/professional` | `sync` | `match` | `page.tsx` | `8_Professional.py` | `POST /api/professional/crypto-sizing`<br>`POST /api/professional/greeks`<br>`POST /api/professional/stress` |
| `/compliance` | `sync` | `adapter` | `page.tsx` | `9_Compliance.py` | `GET /api/compliance/rules`<br>`POST /api/compliance/check`<br>`POST /api/compliance/position-limits` |
| `/agents` | `placeholder` | `placeholder` | `page.tsx` | `10_Agents.py` | - |

### Shared Surface (AppShell)
- `frontend/src/components/shared/app-shell.tsx` -> `GET /api/dashboard/overview` (lookback=30) for regime label hydration.

## Endpoint Contract Matrix
| Endpoint | Request Schema | Status Codes | Response Keys | Structured Error |
| --- | --- | --- | --- | --- |
| `DELETE /api/jobs/{job_id}` | `-` | 200, 422 | `cancelled`, `id` | no |
| `GET /api/compliance/rules` | `-` | 200 | `rules` | no |
| `GET /api/dashboard/macro` | `-` | 200, 422 | `changes`, `series` | no |
| `GET /api/dashboard/overview` | `-` | 200, 422 | `date_range`, `defaults`, `kpi`, `lookback`, `macro`, `regime`, `timeline` | no |
| `GET /api/dashboard/regime-history` | `-` | 200, 422 | `current`, `distribution`, `label`, `series` | no |
| `GET /api/factors/{name}` | `-` | 200, 422 | `category`, `description`, `name`, `parameters` | no |
| `GET /api/health/live` | `-` | 200 | `ok`, `service` | no |
| `GET /api/jobs` | `-` | 200, 422 | `count`, `jobs` | no |
| `GET /api/jobs/{job_id}` | `-` | 200, 422 | `created_at`, `error`, `id`, `kind`, `meta`, `request`, `result`, `status`, `updated_at` | no |
| `GET /api/macro/calendar` | `-` | 200, 422 | `country`, `events`, `importance`, `period` | no |
| `GET /api/meta/signals` | `-` | 200 | `count`, `signals` | no |
| `GET /api/meta/system` | `-` | 200 | `data_dir`, `python`, `services`, `status` | no |
| `POST /api/analytics/run` | `#/components/schemas/AnalyticsRunRequest` | 200, 422 | `methods`, `performance`, `risk`, `rolling`, `stress`, `transaction_costs` | no |
| `POST /api/backtest/run` | `#/components/schemas/BacktestRequest` | 200, 422 | `drawdown_curve`, `equity_curve`, `metrics`, `monthly_returns`, `top_holdings` | no |
| `POST /api/compliance/check` | `#/components/schemas/ComplianceTransactionRequest` | 200, 422 | `hits`, `passed`, `transaction_id` | no |
| `POST /api/compliance/position-limits` | `#/components/schemas/PositionLimitsRequest` | 200, 422 | `breach_count`, `breaches`, `total_checked` | no |
| `POST /api/jobs` | `#/components/schemas/JobCreateRequest` | 200, 400, 422 | `created_at`, `error`, `id`, `kind`, `meta`, `request`, `result`, `status`, `updated_at` | yes |
| `POST /api/jobs/{job_id}/retry` | `-` | 200, 422 | `created_at`, `error`, `id`, `kind`, `meta`, `request`, `status`, `updated_at` | no |
| `POST /api/professional/crypto-sizing` | `#/components/schemas/CryptoSizingRequest` | 200, 422 | `estimated_fees`, `liquidation_price`, `margin_required`, `max_loss`, `notional`, `pair`, `quantity`, `side` | no |
| `POST /api/professional/greeks` | `#/components/schemas/GreeksRequest` | 200, 422 | `delta`, `gamma`, `rho_per_1pct`, `theoretical_price`, `theta_per_day`, `vega_per_1pct` | no |
| `POST /api/professional/stress` | `#/components/schemas/StressTestRequest` | 200, 422 | `by_factor`, `scenario_loss_pct`, `scenario_loss_value` | no |
| `POST /api/screener/run` | `#/components/schemas/ScreenerRunRequest` | 200, 422 | `meta`, `rows` | no |
| `POST /api/screener/sparklines` | `Payload` | 200, 422 | `GARAN`, `THYAO` | no |

## DTO Adapter Mapping Matrix
| UI DTO | Backend Field(s) | UI Field | Transform | Notes |
| --- | --- | --- | --- | --- |
| `BacktestUiResult` | `equity_curve[].strategy | equity_curve[].value` | `equity_curve[].strategy` | prefer strategy, fallback to value | Supports mixed backend payload shapes. |
| `BacktestUiResult` | `drawdown_curve[].drawdown | drawdown_curve[].value` | `drawdown_curve[].drawdown` | prefer drawdown, fallback to value | Normalizes historical drawdown formats. |
| `BacktestUiResult` | `monthly_returns (array | object)` | `monthly_returns[year][month]` | month-key normalization | Accepts both keyed and row-oriented payloads. |
| `BacktestUiResult` | `holdings[] | top_holdings[]` | `holdings[]` | symbol/ticker aliasing | Unifies holdings shape for table/chart consumers. |
| `AnalyticsUiResult` | `performance.* + risk.*` | `metrics.*` | merged metrics map | Single KPI source in UI. |
| `AnalyticsUiResult` | `rolling[].rolling_sharpe_63d | rolling[].rolling_sharpe` | `rolling[].rolling_sharpe` | alias fallback | Backward compatibility with legacy rolling keys. |
| `OptimizationUiResult` | `best_trial.params` | `best_params` | direct projection | Primary sweep summary. |
| `OptimizationUiResult` | `best_trial.metrics[metricKey] | best_trial.score` | `best_metric` | metric-key fallback to score | Handles metric-key drift. |
| `OptimizationUiResult` | `trials[]` | `sweep_results[]` | numeric params + metric extraction | Feeds heatmap/scatter visuals. |
| `ScreenerUiResult` | `count | meta.total_matches | rows.length` | `count` | fallback cascade | Stabilizes KPI count across schema variants. |
| `ComplianceUiResult` | `status | passed` | `status` | passed=false -> FAIL else PASS | Wave 1 frozen alias rule: passed -> status. |
| `ComplianceUiResult` | `rule.comparator | rule.operator` | `rule.operator` | comparator alias to operator | Wave 1 frozen alias rule: comparator -> operator. |
| `ComplianceUiResult` | `hit.limit | rule.threshold` | `hits[].limit` | fallback to rule threshold | Ensures deterministic limit display. |
| `ScreenerUiResult` | `rows[].return_1m (fraction)` | `table cell 1M Ret %` | display multiply by 100 | Frozen metric unit annotation for Wave 1. |

## Canonical Sample Index
| Sample Key | Status | Response Keys |
| --- | --- | --- |
| `DELETE /api/jobs/{job_id}` | 200 | `cancelled`, `id` |
| `GET /api/compliance/rules` | 200 | `rules` |
| `GET /api/dashboard/macro` | 200 | `changes`, `series` |
| `GET /api/dashboard/overview` | 200 | `date_range`, `defaults`, `kpi`, `lookback`, `macro`, `regime`, `timeline` |
| `GET /api/dashboard/regime-history` | 200 | `current`, `distribution`, `label`, `series` |
| `GET /api/factors/{name}` | 200 | `category`, `description`, `name`, `parameters` |
| `GET /api/health/live` | 200 | `ok`, `service` |
| `GET /api/jobs` | 200 | `count`, `jobs` |
| `GET /api/jobs/{job_id}` | 200 | `created_at`, `error`, `id`, `kind`, `meta`, `request`, `result`, `status`, `updated_at` |
| `GET /api/macro/calendar` | 200 | `country`, `events`, `importance`, `period` |
| `GET /api/meta/signals` | 200 | `count`, `signals` |
| `GET /api/meta/system` | 200 | `data_dir`, `python`, `services`, `status` |
| `POST /api/analytics/run` | 200 | `methods`, `performance`, `risk`, `rolling`, `stress`, `transaction_costs` |
| `POST /api/backtest/run` | 200 | `drawdown_curve`, `equity_curve`, `metrics`, `monthly_returns`, `top_holdings` |
| `POST /api/compliance/check` | 200 | `hits`, `passed`, `transaction_id` |
| `POST /api/compliance/position-limits` | 200 | `breach_count`, `breaches`, `total_checked` |
| `POST /api/factors/combine` | 200 | `attribution`, `backtest`, `factor_correlation` |
| `POST /api/factors/snapshot` | 200 | `date`, `scores` |
| `POST /api/jobs` | 200 | `created_at`, `error`, `id`, `kind`, `meta`, `request`, `result`, `status`, `updated_at` |
| `POST /api/jobs/{job_id}/retry` | 200 | `created_at`, `error`, `id`, `kind`, `meta`, `request`, `status`, `updated_at` |
| `POST /api/optimize/run` | 200 | `best_trial`, `method`, `trials` |
| `POST /api/professional/crypto-sizing` | 200 | `estimated_fees`, `liquidation_price`, `margin_required`, `max_loss`, `notional`, `pair`, `quantity`, `side` |
| `POST /api/professional/greeks` | 200 | `delta`, `gamma`, `rho_per_1pct`, `theoretical_price`, `theta_per_day`, `vega_per_1pct` |
| `POST /api/professional/stress` | 200 | `by_factor`, `scenario_loss_pct`, `scenario_loss_value` |
| `POST /api/screener/run` | 200 | `meta`, `rows` |
| `POST /api/screener/sparklines` | 200 | `GARAN`, `THYAO` |

### Structured Error Samples
| Error Sample Key | Status | detail.code |
| --- | --- | --- |
| `POST /api/jobs::invalid_backtest_payload` | 422 | `job_validation_error` |
| `POST /api/jobs::optimize_missing_parameter_space` | 422 | `job_validation_error` |
| `POST /api/jobs::unsupported_kind` | 400 | `unsupported_job_kind` |

## Drift Register
| Surface | Class | Detail | Target Wave |
| --- | --- | --- | --- |
| Factor Lab | `mismatch` | combine payload currently sends signals:string[] while backend contract expects signals:[{name, weight}]. | Wave 1 |
| Analytics | `mismatch` | UI uses async jobs endpoint with non-canonical method labels; canonical path is POST /api/analytics/run. | Wave 1 |
| Optimization | `mismatch` | UI payload diverges from OptimizationRunRequest parameter_space contract. | Wave 1 |
| Backtest + Signal Construction | `adapter` | Adapter normalization required for value/strategy keys and drawdown/value variants. | Wave 1 |
| Compliance | `adapter` | Alias mapping required for comparator/operator and passed/status. | Wave 1 |
| Screener | `adapter` | Count fallback and metric-unit display normalization required. | Wave 1 |

## Unknown-Field Register
- None. Unknown-field register is intentionally empty at Phase 0 gate.

## Wave 1 Interface Freeze
- Frozen UI DTOs:
  - `BacktestUiResult`
  - `AnalyticsUiResult`
  - `OptimizationUiResult`
  - `ComplianceUiResult`
  - `ScreenerUiResult`
- Frozen alias rules:
  - `comparator -> operator`
  - `passed -> status`
- Frozen unit annotations:
  - Screener `return_1m` remains fractional in backend and is rendered as percentage in UI.
  - Drawdown series are ratio-scale values unless explicitly formatted in UI components.
- Frozen analytics execution path:
  - Canonical: `POST /api/analytics/run`
  - Non-canonical fallback: `POST /api/jobs` with `kind=analytics` only if schema parity is guaranteed.

## Phase 0 Gate Criteria
The verification script must fail if any of the following occur:
- Any Next route is missing from the manifest.
- Any endpoint consumed by `frontend/src/lib/api.ts` or `AppShell` is missing.
- Any required DTO mapping is unresolved.
- Unknown-field register is not empty.
- Canonical sample is missing for a consumed endpoint.
