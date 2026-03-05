# Phase 4 — Streamlit Parity Review

Date: 2026-03-05
Scope: Feature parity completion for all 10 migrated pages, with `/agents` intentionally beta/stub.

## Outcome
- Parity matrix delivered: `docs/plans/2026-03-05-phase4-streamlit-parity-matrix.md`
- In-scope parity rows complete: `10 / 10`
- Deferred rows: `0`
- `/agents` remains explicit beta placeholder with local stub interactions only.

## Backend Contract Closures
- Extended backtest request surface in `src/bist_quant/api/main.py::BacktestRequest` and wiring into run service.
- Expanded analytics schema/method support in `src/bist_quant/api/schemas.py::AnalyticsRunRequest`.
- Added deep analytics sections and benchmark endpoint in `src/bist_quant/api/routers/analytics.py` (`/api/analytics/run`, `GET /api/analytics/benchmark/xu100`).
- Added compliance anomalies endpoint in `src/bist_quant/api/routers/compliance.py` (`POST /api/compliance/activity-anomalies`).
- Added professional pip-value endpoint in `src/bist_quant/api/routers/professional.py` (`POST /api/professional/pip-value`).
- Added signal construction router/endpoints in `src/bist_quant/api/routers/signal_construction.py` and mounted in API main/router exports.

## Frontend Parity Closures
- Backtest parity tabs/risk/export: `frontend/src/app/backtest/backtest-content.tsx`
- Analytics depth and benchmark controls: `frontend/src/app/analytics/analytics-content.tsx`
- Optimization controls + export/handoff: `frontend/src/app/optimization/optimization-content.tsx`
- Screener advanced filters/sparklines/results: `frontend/src/app/screener/screener-content.tsx`
- Factor Lab advanced combine workflow: `frontend/src/app/factor-lab/factor-lab-content.tsx`
- Signal Construction 3-tab workflow: `frontend/src/app/signal-construction/signal-construction-content.tsx`
- Professional extras incl. pip-value: `frontend/src/app/professional/professional-content.tsx`
- Compliance extras incl. anomalies/history/rule editor: `frontend/src/app/compliance/compliance-content.tsx`
- Agents beta/stub log/export/clear: `frontend/src/app/agents/agents-content.tsx`
- Dashboard macro detail controls: `frontend/src/app/dashboard/dashboard-content.tsx`

## Type/API/Adapter Hardening
- DTO and request/response expansions in:
  - `frontend/src/lib/types.ts`
  - `frontend/src/lib/api.ts`
  - `frontend/src/lib/adapters.ts`
- Added/extended tests for alias and payload normalization:
  - `frontend/src/lib/api.test.ts`
  - `frontend/src/lib/adapters.test.ts`
  - `tests/test_phase4_api.py`

## Verification Evidence
Full gate evidence is recorded in:
- `docs/plans/artifacts/2026-03-05-phase4-regression-evidence.md`

Executed gate commands:
1. `pytest` -> `280 passed, 4 skipped`
2. `ruff check src tests` -> `All checks passed`
3. `npm --prefix frontend run lint` -> `PASS`
4. `npm --prefix frontend run test` -> `17 files, 92 tests passed`
5. `npm --prefix frontend run build` -> `PASS`
6. `npm --prefix frontend exec playwright test` -> `17 passed`

## Notes
- Added root `playwright.config.js` so the required root-invoked command `npm --prefix frontend exec playwright test` reliably targets `frontend/e2e`.
- Playwright parity flows are API-mocked in `frontend/e2e/critical-flows.spec.ts` for deterministic CI behavior independent of local backend uptime.
