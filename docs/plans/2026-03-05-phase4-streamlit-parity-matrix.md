# Phase 4 — Streamlit Feature Parity Matrix (All 10 Pages)

Date: 2026-03-05
Scope: Feature parity against `app/pages/1..10` (not pixel parity)

## Status Key
- `Next.js Status`: `Complete` means parity closure implemented and validated.
- `Gap Severity`: pre-fix impact level; all listed gaps are now closed.
- `Decision`: `Implemented` unless explicitly deferred (none deferred).

| Page | Streamlit Feature | Next.js Status | Gap Severity | Decision | Implementation Ref | Test Ref | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dashboard | Macro detail controls: lookback switch + detail charts/table; maintain regime/timeline coverage | Complete | Medium -> Closed | Implemented | `frontend/src/app/dashboard/dashboard-content.tsx` | `frontend/e2e/ui-consistency.spec.ts`, `frontend/e2e/navigation.spec.ts` | [Regression Gate](artifacts/2026-03-05-phase4-regression-evidence.md) |
| Backtest | Full tabs (`equity`, `monthly`, `rolling`, `risk`, `holdings`, `export`), risk-control panel, CSV/JSON export | Complete | High -> Closed | Implemented | `frontend/src/app/backtest/backtest-content.tsx`, `src/bist_quant/api/main.py` | `frontend/e2e/critical-flows.spec.ts`, `tests/test_phase4_api.py` | [Regression Gate](artifacts/2026-03-05-phase4-regression-evidence.md) |
| Factor Lab | Category/search filtering, quick backtest, multi-factor weighting, combine schemes, result tabs (`curve/breakdown/correlation`) | Complete | High -> Closed | Implemented | `frontend/src/app/factor-lab/factor-lab-content.tsx`, `frontend/src/lib/api.ts` | `frontend/e2e/critical-flows.spec.ts`, `frontend/src/lib/api.test.ts` | [Regression Gate](artifacts/2026-03-05-phase4-regression-evidence.md) |
| Signal Construction | Indicator Builder + Snapshot/Backtest, Five-Factor tab, Orthogonalization tab + diagnostics | Complete | High -> Closed | Implemented | `frontend/src/app/signal-construction/signal-construction-content.tsx`, `src/bist_quant/api/routers/signal_construction.py` | `frontend/e2e/critical-flows.spec.ts`, `tests/test_phase4_api.py` | [Regression Gate](artifacts/2026-03-05-phase4-regression-evidence.md) |
| Screener | Universe/sector/preset/recommendation/advanced filters, applied chips, richer summaries, sparklines, watchlist/extended panels | Complete | High -> Closed | Implemented | `frontend/src/app/screener/screener-content.tsx`, `frontend/src/lib/api.ts` | `frontend/e2e/critical-flows.spec.ts`, `tests/test_screener_api.py`, `frontend/src/lib/adapters.test.ts` | [Regression Gate](artifacts/2026-03-05-phase4-regression-evidence.md) |
| Analytics | Session/upload source, benchmark include/toggle, deep metrics + `rolling/walk-forward/monte-carlo/attribution/risk/stress/costs` panels | Complete | High -> Closed | Implemented | `frontend/src/app/analytics/analytics-content.tsx`, `src/bist_quant/api/routers/analytics.py`, `src/bist_quant/api/schemas.py` | `frontend/e2e/critical-flows.spec.ts`, `tests/test_phase4_api.py`, `frontend/src/lib/api.test.ts` | [Regression Gate](artifacts/2026-03-05-phase4-regression-evidence.md) |
| Optimization | Dynamic sweep controls, optimizer controls (`method/max_trials/train_ratio`), trials/export, best-config handoff | Complete | High -> Closed | Implemented | `frontend/src/app/optimization/optimization-content.tsx`, `frontend/src/lib/adapters.ts` | `frontend/e2e/critical-flows.spec.ts`, `frontend/src/lib/adapters.test.ts` | [Regression Gate](artifacts/2026-03-05-phase4-regression-evidence.md) |
| Professional | Stress template loader/editor, IV-smile section, pip-value utility, preserve Greeks/stress/crypto core | Complete | Medium -> Closed | Implemented | `frontend/src/app/professional/professional-content.tsx`, `src/bist_quant/api/routers/professional.py` | `frontend/e2e/critical-flows.spec.ts`, `tests/test_phase4_api.py` | [Regression Gate](artifacts/2026-03-05-phase4-regression-evidence.md) |
| Compliance | Rule checklist detail, editable rules UX, position-limit monitor, anomaly detection, run-history log | Complete | High -> Closed | Implemented | `frontend/src/app/compliance/compliance-content.tsx`, `src/bist_quant/api/routers/compliance.py` | `frontend/e2e/critical-flows.spec.ts`, `tests/test_phase4_api.py` | [Regression Gate](artifacts/2026-03-05-phase4-regression-evidence.md) |
| Agents | Explicit beta/placeholder, planned cards, stub prompt/log flow, export + clear log, no external LLM execution | Complete | Medium -> Closed | Implemented | `frontend/src/app/agents/agents-content.tsx` | `frontend/e2e/critical-flows.spec.ts` | [Regression Gate](artifacts/2026-03-05-phase4-regression-evidence.md) |

## Decision Summary
- Deferred rows: `0`
- All in-scope parity rows: `Complete`
