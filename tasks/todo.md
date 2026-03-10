# Tasks — BIST Next.js Migration

Date: 2026-03-03
Scope: Phase 1 (planning and specification)

## Checklist

- [x] Extract Phase 1 requirements from current Streamlit app pages and service layer.
- [x] Define architecture target using Acibadem-style frontend/backend split.
- [x] Build full parity matrix for pages 1-10 with complexity and execution mode.
- [x] Draft API blueprint including async job contract for heavy workflows.
- [x] Define rollout waves (P1: pages 1-7, P2: pages 8-10).
- [x] Document risks, edge cases, and mitigation strategy.
- [x] Publish implementation-ready Phase 1 spec under `docs/plans/`.

## Review

Phase 1 is complete.

Primary output:

- `docs/plans/2026-03-03-bist-nextjs-migration-phase-1-spec.md`

Validation summary:

- All current Streamlit pages are mapped.
- Integration points to existing Python service/engine modules are specified.
- API and async execution patterns are defined with clear boundaries.
- Implementation can begin without unresolved architectural ambiguity.

## Phase 2 — Build Foundation

Checklist:

- [x] Create `src/bist_quant/api` package with `create_app`, `app`, and `quant_router` exports.
- [x] Add foundational endpoints: health, metadata, macro calendar, dashboard overview.
- [x] Scaffold Next.js frontend app in `frontend/` with TypeScript and basic app shell.
- [x] Add typed frontend API client and first dashboard slice.
- [x] Validate API router wiring via targeted test run.

Review:

- Backend API scaffold is now available under `src/bist_quant/api/main.py`.
- Frontend scaffold is now available under `frontend/`.
- Macro calendar API route compatibility is confirmed by `tests/test_macro_calendar_api.py`.

## Phase 3 — Dashboard Vertical Slice

Checklist:

- [x] Extend dashboard backend payload with macro series and regime timeline/distribution for UI parity.
- [x] Refactor frontend into reusable dashboard sections (shell, KPI, timeline chart, regime distribution, macro table).
- [x] Add responsive page layout and controls for lookback windows.
- [x] Validate backend routes with targeted API tests and smoke checks.

Definition of done:

- Dashboard page renders core metrics, timeline points, regime context, and macro change table from API data.
- No Streamlit dependency in the new UI slice.
- Existing API tests keep passing.

Review:

- Dashboard payload now includes `regime.series`, `regime.distribution`, and `macro.series/changes` in `src/bist_quant/api/main.py`.
- Frontend now renders a dashboard vertical slice with reusable components in `frontend/src/components/`.
- Lookback controls are implemented via query params on `frontend/src/app/page.tsx`.
- Verification completed with `pytest tests/test_macro_calendar_api.py tests/test_dashboard_api.py`.

## Phase 4 — Backtest Vertical Slice

Checklist:

- [x] Add async job orchestration endpoints in API (`/api/jobs`).
- [x] Add backtest execution endpoint wired to `CoreBackendService.run_backtest`.
- [x] Add frontend Backtest panel with run + polling + result summary.
- [x] Validate new API routes and background execution with tests.

Definition of done:

- User can trigger backtest from Next.js UI.
- Backtest runs asynchronously and can be polled by job id.
- Result metrics render in frontend without Streamlit.

Review:

- Added in-memory async job manager in `src/bist_quant/api/jobs.py`.
- Added endpoints in `src/bist_quant/api/main.py`:
  - `POST /api/backtest/run` (supports `async_job=true`)
  - `POST /api/jobs`
  - `GET /api/jobs`
  - `GET /api/jobs/{job_id}`
  - `DELETE /api/jobs/{job_id}`
- Added browser-driven Backtest panel in `frontend/src/components/BacktestPanel.tsx` and integrated into `frontend/src/app/page.tsx`.
- Added API contract tests in `tests/test_jobs_api.py`.
- Validation completed with `pytest tests/test_macro_calendar_api.py tests/test_dashboard_api.py tests/test_jobs_api.py` (6 passed).

## Phase 5 — Job Persistence and UX Hardening

Checklist:

- [x] Persist async job records to disk to survive API restarts.
- [x] Add job retry endpoint and include request payload metadata for replay.
- [x] Expand frontend Backtest panel with job history + cancel/retry controls.
- [x] Add tests for persistence/retry behavior and re-run API suite.

Definition of done:

- Job list remains available after process restart from persisted store.
- Failed/cancelled/completed jobs can be retried from API and UI.
- Backtest UI surfaces active/history jobs with operational controls.

Review:

- Job persistence implemented in `src/bist_quant/api/jobs.py` with JSON store at `data/api_jobs.json`.
- API retry route added in `src/bist_quant/api/main.py` at `POST /api/jobs/{job_id}/retry`.
- Job create payload now persists request metadata for replay.
- Frontend backtest panel now supports history refresh, cancel, and retry in `frontend/src/components/BacktestPanel.tsx`.
- Frontend API client expanded with `listJobs`, `cancelJob`, and `retryJob` in `frontend/src/lib/api.ts`.
- Added tests:
  - `tests/test_jobs_api.py` (retry endpoint)
  - `tests/test_job_manager_persistence.py` (store reload)
- Verification completed with:
  - `pytest tests/test_jobs_api.py tests/test_job_manager_persistence.py`
  - `pytest tests/test_macro_calendar_api.py tests/test_dashboard_api.py tests/test_jobs_api.py tests/test_job_manager_persistence.py`

## Phase 6 — Testing and Quality Gates

Checklist:

- [x] Run backend regression tests for migrated API slices.
- [x] Run backend static checks on touched API/test files.
- [x] Run frontend quality gates (`lint`, `build`) and resolve issues.
- [x] Document verification output and any remaining risks.

Definition of done:

- All migration-related backend tests pass.
- Backend lint checks pass for changed API/test modules.
- Frontend lint and production build pass.
- Phase 6 verification results are captured in this document.

Review:

- Backend regression checks passed:
  - `pytest tests/test_macro_calendar_api.py tests/test_dashboard_api.py tests/test_jobs_api.py tests/test_job_manager_persistence.py` (8 passed)
- Backend static checks passed:
  - `ruff check src/bist_quant/api tests/test_dashboard_api.py tests/test_jobs_api.py tests/test_job_manager_persistence.py`
- Frontend quality gates passed:
  - `npm install`
  - `npm run lint`
  - `npm run build`
- Frontend lint setup was hardened with a flat ESLint config in `frontend/eslint.config.mjs` and package script update in `frontend/package.json`.
- Next.js build updated `frontend/tsconfig.json` with Next-required defaults (`jsx: react-jsx`, include `.next/dev/types/**/*.ts`).
- Remaining non-blocking warning: Next build reports multiple lockfiles in parent directories; does not fail build.

## Phase 7 — Deployment and Cutover

Checklist:

- [x] Add local dual-run script for FastAPI backend + Next.js frontend.
- [x] Add production deployment templates (systemd + nginx) for the new stack.
- [x] Add deployment/cutover runbook documenting fallback path to Streamlit.
- [x] Validate script syntax and confirm migration verification suite remains green.

Definition of done:

- Engineers can start backend/frontend together locally using one command.
- Production templates exist for reverse proxy and process supervision.
- Cutover plan includes rollback path and smoke checks.

Review:

- Added local startup scripts:
  - `scripts/start_next_stack.sh` (dev, dual-run)
  - `scripts/start_next_stack_prod.sh` (production-like local run)
- Added deployment templates:
  - `deploy/nginx.conf`
  - `deploy/bist-quant-backend.service`
  - `deploy/bist-quant-frontend.service`
- Added runbook:
  - `deploy/README.md` (cutover steps, smoke checks, rollback path)
- Updated root docs:
  - `README.md` now includes Next.js + FastAPI stack run commands.
- Validation completed:
  - `bash -n scripts/start_next_stack.sh`
  - `bash -n scripts/start_next_stack_prod.sh`
  - `pytest tests/test_macro_calendar_api.py tests/test_dashboard_api.py tests/test_jobs_api.py tests/test_job_manager_persistence.py` (8 passed)

## Phase 8 — Recovery Plan Execution (Streamlit → Next.js Stabilization + Redesign)

Date: 2026-03-04

Checklist:
- [x] Create frontend↔backend contract matrix with canonical payload/response examples.
- [ ] Add frontend DTO adapters (`BacktestUiResult`, `AnalyticsUiResult`, `OptimizationUiResult`, `ComplianceUiResult`, `ScreenerUiResult`).
- [x] Harden backend `POST /api/jobs` validation and error payloads (no uncaught 500 on contract errors).
- [ ] Decide and implement single analytics execution path (direct `/api/analytics/run` from frontend).
- [ ] Implement or remove `/api/screener/sparklines` (implement endpoint for parity).
- [ ] Repair critical frontend flows:
  - [ ] Factor Lab combine payload and result rendering
  - [ ] Analytics payload + result mapping
  - [ ] Optimization payload + result mapping
  - [ ] Backtest + Signal Construction backend-shape adapters
  - [ ] Compliance API alias mapping (`comparator -> operator`, `passed -> status`)
  - [ ] Screener request shape + value-unit corrections
- [ ] UI coherence pass for core screens (remove debug/raw JSON blocks, unify controls)
- [x] Add OpenAPI snapshot contract test.
- [ ] Replace heading-only E2E with functional route scenarios for core pages.
- [ ] Fix lint/type/test baseline and document verification outputs.

Definition of done:
- Core P1 routes (Backtest, Factor Lab, Signal Construction, Screener, Analytics, Optimization) operate end-to-end against FastAPI without schema/runtime mismatches.
- Lint, typecheck, unit tests, and E2E test suite pass.
- Contract matrix and canonical examples are published under `docs/plans/`.

Review:
- Phase 0 complete: contract freeze artifacts generated under `docs/plans/` and `docs/plans/artifacts/`.
- Generated outputs:
  - `docs/plans/2026-03-04-phase0-contract-freeze.md`
  - `docs/plans/artifacts/phase0-contract-manifest.json`
  - `docs/plans/artifacts/phase0-canonical-samples.json`
  - `scripts/contract/generate_phase0_contract_artifacts.py`
  - `scripts/contract/verify_phase0_contract.py`
- Verification log:
  - `python scripts/contract/generate_phase0_contract_artifacts.py`
  - `python scripts/contract/verify_phase0_contract.py`
  - Result: `phase0_verify_status=PASS` at `2026-03-04T18:56:39.924915+00:00`.

## Phase 8A — Critical Flow Repair Execution (Six-Page E2E Gate)

Date: 2026-03-04

Checklist:
- [x] Add execution checklist and track completion in this file.
- [x] Fix Factor Lab combine payload to use `signals: [{name, weight}]` and support attribution aliases.
- [x] Migrate Analytics page to canonical `/api/analytics/run` flow with method mapping and `{date, value}` input.
- [x] Fix Optimization payload to send valid `parameter_space` and parse `best_trial/trials`.
- [x] Harden Backtest/Signal Construction adapters for `value/strategy`, `drawdown/value`, and monthly shape normalization.
- [x] Fix Compliance request mapping (`operator -> comparator`, `description -> message`) and response hit fallbacks.
- [x] Ensure Screener request builder emits nested `filters` schema and normalize return formatting.
- [x] Update frontend unit/contract tests, including stale `api.test.ts` backtest expectation and new adapter tests.
- [x] Replace heading-only core route checks with functional Playwright critical-flow scenarios.
- [x] Verify all gates (typecheck, unit tests, backend tests, functional Playwright).

Definition of done:
- Backtest, Factor Lab, Signal Construction, Screener, Analytics, and Optimization flows execute without schema/runtime mismatches.
- Frontend typecheck and unit tests pass.
- Backend jobs/screener API tests pass.
- Functional Playwright critical flows pass against live FastAPI backend.

Review:
- Frontend API/adapters/pages updated:
  - `frontend/src/lib/api.ts`
  - `frontend/src/lib/adapters.ts`
  - `frontend/src/app/factor-lab/factor-lab-content.tsx`
  - `frontend/src/app/analytics/analytics-content.tsx`
  - `frontend/src/app/optimization/optimization-content.tsx`
  - `frontend/src/app/signal-construction/signal-construction-content.tsx`
  - `frontend/src/app/compliance/compliance-content.tsx`
  - `frontend/src/app/screener/screener-content.tsx`
- Tests added/updated:
  - `frontend/src/lib/api.test.ts`
  - `frontend/src/lib/adapters.test.ts`
  - `frontend/e2e/critical-flows.spec.ts`
  - `frontend/e2e/navigation.spec.ts` (core heading-only checks removed)

Verification log:
- `cd frontend && npm run typecheck` -> PASS
- `cd frontend && npm run test` -> PASS (`71 passed`)
- `pytest tests/test_jobs_api.py tests/test_screener_api.py -q` -> PASS (`8 passed`)
- `cd frontend && NEXT_PUBLIC_API_URL=http://127.0.0.1:8001 npm run test:e2e -- e2e/critical-flows.spec.ts` -> PASS (`7 passed`, ~2.7m)

## Phase 8B — Jobs API Hardening + OpenAPI Snapshot Gate

Date: 2026-03-04

Checklist:
- [x] Add shared jobs error envelope utilities and jobs-route request-validation handler.
- [x] Centralize job-kind request validation wrappers for all supported kinds.
- [x] Normalize jobs-family 4xx responses to structured `{detail:{code,detail,hint,errors?}}`.
- [x] Add typed frontend API error parsing and render `detail/hint/code` in error UI.
- [x] Add committed OpenAPI snapshot + checker script and wire CI step.
- [x] Fix CI backend lint/type paths and include `api` extra dependency install.

Review:
- Backend changes:
  - `src/bist_quant/api/errors.py`
  - `src/bist_quant/api/main.py`
- Frontend changes:
  - `frontend/src/lib/api-error.ts`
  - `frontend/src/lib/api.ts`
  - `frontend/src/components/shared/api-error.tsx`
  - `frontend/src/components/shared/api-error.test.tsx`
  - `frontend/src/lib/api.test.ts`
- Contract/CI changes:
  - `scripts/contract/check_openapi_snapshot.py`
  - `docs/plans/artifacts/openapi.snapshot.json`
  - `.github/workflows/ci.yml`
- Tests expanded in:
  - `tests/test_jobs_api.py`

Verification log:
- `pytest tests/test_jobs_api.py tests/test_screener_api.py -q` -> PASS (`17 passed`)
- `ruff check src/bist_quant/api/main.py src/bist_quant/api/errors.py tests/test_jobs_api.py` -> PASS
- `python scripts/contract/check_openapi_snapshot.py` -> PASS
- `cd frontend && npm run test` -> PASS (`75 passed`)
- `cd frontend && npm run typecheck` -> PASS

## Phase 9 — UI System Redesign (Light-First + Density + 10-Route Consistency)

Date: 2026-03-05

Checklist:
- [x] Define and apply one light-first design language (typography, spacing, surfaces, semantic colors).
- [x] Add global density system (`comfortable`/`compact`) with persistence and sidebar toggle.
- [x] Introduce shared scaffold and form primitives; migrate all 10 routes.
- [x] Remove raw debug JSON blocks from product routes.
- [x] Enforce shared-control-only route markup via static check (`check-ui-primitives.sh`).
- [x] Add visual consistency checklist artifact for all 10 routes.
- [x] Add Playwright UI consistency assertions and keep critical flows functional.

Definition of done:
- All 10 product routes render with shared scaffold + shared header grammar.
- Product routes contain no raw `<pre>/<select>/<textarea>/<table>` markup.
- Density toggle updates root density tokens and persists preference.
- UI consistency gate passes for all 10 routes.
- Existing critical flows remain green with backend running.

Review:
- Added light-first tokenized system in `frontend/src/app/globals.css` and synchronized chart tokens in `frontend/src/lib/tokens.ts`.
- Added density infra:
  - `frontend/src/hooks/use-density.tsx`
  - `frontend/src/components/shared/density-provider.tsx`
  - `frontend/src/components/shared/density-toggle.tsx`
- Added shared primitives:
  - `frontend/src/components/shared/page-scaffold.tsx`
  - `frontend/src/components/shared/form-field.tsx`
  - `frontend/src/components/shared/key-value-list.tsx`
  - `frontend/src/components/ui/select-input.tsx`
  - `frontend/src/components/ui/textarea.tsx`
  - `frontend/src/components/ui/checkbox.tsx`
- Migrated all 10 route content files under `frontend/src/app/**` to scaffold and shared controls.
- Removed Factor Lab debug JSON `<pre>` and replaced with structured key/value rendering.
- Added static enforcement script:
  - `frontend/scripts/check-ui-primitives.sh`
  - wired into `frontend/package.json` (`lint` and `test` scripts).
- Added CI-ready consistency tests and artifacts:
  - `frontend/e2e/ui-consistency.spec.ts`
  - `docs/plans/artifacts/phase3-visual-consistency-checklist.md`
- Updated Playwright dev-server config to use isolated port and avoid cross-project reuse collisions:
  - `frontend/playwright.config.ts`

Verification:
- `cd frontend && npm run check:ui-primitives`
- `cd frontend && npm run typecheck`
- `cd frontend && npm run lint`
- `cd frontend && npm run test`
- `cd frontend && npm run test:e2e -- ui-consistency.spec.ts`
- `python -m uvicorn bist_quant.api.main:app --host 127.0.0.1 --port 8001` + `cd frontend && npm run test:e2e -- critical-flows.spec.ts`
  - Result: `7 passed` for `critical-flows.spec.ts`.

## Phase 10 — Streamlit Parity Completion (All 10 Pages)

Date: 2026-03-05

Checklist:
- [x] Build and publish parity matrix for `app/pages/1..10` with explicit completion state.
- [x] Close Backtest parity gaps (risk/holdings/export tabs + risk controls + export actions).
- [x] Close Analytics depth parity gaps (source switcher, benchmark toggle, deep analysis sections).
- [x] Close Optimization/Screener/Compliance parity gaps (advanced controls, sparklines, anomalies/history).
- [x] Close Dashboard/Factor Lab/Signal Construction/Professional parity gaps.
- [x] Keep Agents explicitly beta/stub with local session-log interactions only.
- [x] Update backend and frontend contracts/types/adapters in lockstep.
- [x] Extend backend/frontend/e2e tests for parity surfaces.
- [x] Run full regression gate and record evidence.
- [x] Publish final parity review document.

Definition of done:
- Parity matrix rows are all `Complete` (or blocker-documented deferral with sign-off).
- Full regression suite passes across backend + frontend + e2e.
- `tasks/todo.md` and phase review docs are updated with evidence references.

Review:
- Parity matrix: `docs/plans/2026-03-05-phase4-streamlit-parity-matrix.md`
- Phase review: `docs/plans/2026-03-05-phase4-parity-review.md`
- Evidence log: `docs/plans/artifacts/2026-03-05-phase4-regression-evidence.md`

Verification log:
- `pytest` -> `280 passed, 4 skipped`
- `ruff check src tests` -> `All checks passed`
- `npm --prefix frontend run lint` -> `PASS`
- `npm --prefix frontend run test` -> `17 files / 92 tests passed`
- `npm --prefix frontend run build` -> `PASS`
- `npm --prefix frontend exec playwright test` -> `17 passed`

## Phase 5 — Verification + Release Readiness (Real-Backend E2E + Dark-First Redesign)

Date: 2026-03-05

Checklist:
- [x] Add Phase 5 implementation checklist and review slots in this file.
- [x] Apply dark-first design system updates across shared tokens and all 10 routes.
- [x] Replace heading-only E2E checks with real-backend functional journeys.
- [x] Update Playwright config to run against FastAPI + Next dual stack by default, with external-base override for smoke mode.
- [x] Add backend contract tests for all frontend-consumed API surfaces missing direct coverage.
- [x] Add production smoke automation for dual-stack startup and key API/UI workflows.
- [x] Refresh OpenAPI snapshot baseline and re-verify contract gate.
- [x] Run full verification bundle and capture reproducible logs under `docs/plans/artifacts/`.

Definition of done:
- Lint/type/unit/e2e gates are green with real-backend Playwright flows.
- Contract tests cover frontend-consumed endpoints and malformed payload behavior.
- Smoke runner starts production dual stack and validates key workflows end-to-end.
- OpenAPI snapshot check passes from committed baseline.
- Verification evidence is committed and reproducible.

Review:
- UI/theming updates:
  - `frontend/src/app/globals.css`
  - `frontend/src/lib/tokens.ts`
  - `frontend/src/components/shared/sidebar.tsx`
  - `frontend/src/components/shared/page-header.tsx`
  - `frontend/src/components/shared/section-card.tsx`
  - `frontend/src/components/shared/data-table.tsx`
- Functional E2E updates:
  - `frontend/e2e/navigation.spec.ts`
  - `frontend/e2e/critical-flows.spec.ts`
  - `frontend/playwright.config.ts`
  - `frontend/next.config.ts`
- Stable selector hooks added on core action buttons:
  - `frontend/src/app/backtest/backtest-content.tsx`
  - `frontend/src/app/factor-lab/factor-lab-content.tsx`
  - `frontend/src/app/analytics/analytics-content.tsx`
  - `frontend/src/app/optimization/optimization-content.tsx`
  - `frontend/src/app/screener/screener-content.tsx`
  - `frontend/src/app/signal-construction/signal-construction-content.tsx`
  - `frontend/src/app/compliance/compliance-content.tsx`
  - `frontend/src/app/professional/professional-content.tsx`
- Contract and smoke automation:
  - `tests/test_frontend_api_contracts.py`
  - `scripts/release/run_api_smoke.py`
  - `scripts/release/smoke_dual_stack.sh`
- Contract baseline:
  - `docs/plans/artifacts/openapi.snapshot.json`

Verification log:
- `ruff check src tests` -> PASS (`All checks passed!`)
- `npm --prefix frontend run lint` -> PASS (`[ui-primitives] PASS`)
- `npm --prefix frontend run typecheck` -> PASS
- `pytest -q` -> PASS (`292 passed, 4 skipped`)
- `npm --prefix frontend run test` -> PASS (`17 files, 92 tests`)
- `npm --prefix frontend run test:e2e` -> PASS (`13 passed`)
- `python scripts/contract/check_openapi_snapshot.py --update` -> PASS (`openapi_snapshot=UPDATED`)
- `python scripts/contract/check_openapi_snapshot.py` -> PASS (`openapi_snapshot=PASS`)
- `python scripts/contract/verify_phase0_contract.py` -> PASS (`phase0_verify_status=PASS`)
- `bash scripts/release/smoke_dual_stack.sh` -> PASS (API smoke + Playwright smoke `11 passed`)
- Evidence artifacts:
  - `docs/plans/artifacts/2026-03-05-phase5-verification-evidence.md`
  - `docs/plans/artifacts/phase5-verification-20260305/`
  - `docs/plans/artifacts/phase5-smoke-20260305T071319Z/`

## Library Publish Readiness — Task 1 (Package Boundary)

Date: 2026-03-09

Checklist:
- [x] Add a failing package-boundary test that inspects the built wheel.
- [x] Exclude app-facing Python packages from the published distribution.
- [x] Remove app-facing exports from the top-level `bist_quant` public API.
- [x] Run targeted package-boundary tests and package build verification.

Definition of done:
- Built wheel does not include app-facing packages such as `bist_quant.api` and `bist_quant.services`.
- `import bist_quant` still works from the source tree and from the built wheel.
- Top-level package surface no longer advertises app-only helpers.

Review:
- Added `tests/test_package_boundary.py` to build the wheel, inspect wheel contents, and smoke-install it into a clean virtualenv.
- Updated `pyproject.toml` to exclude app-facing packages from the published artifact:
  - `src/bist_quant/api`
  - `src/bist_quant/engines`
  - `src/bist_quant/jobs`
  - `src/bist_quant/observability`
  - `src/bist_quant/persistence`
  - `src/bist_quant/security`
  - `src/bist_quant/services`
- Pruned app-facing exports from `src/bist_quant/__init__.py` so the top-level library surface no longer advertises API/router helpers or engine compatibility exports.
- Hardened `src/bist_quant/common/__init__.py` and `src/bist_quant/common/data_loader.py` so `import bist_quant` succeeds from a clean wheel install with only core dependencies installed.

Verification:
- `pytest tests/test_package_boundary.py -v` -> PASS (`2 passed`)
- `python -c "import bist_quant; print(bist_quant.__version__)"` -> PASS (`0.3.0`)

## Library Publish Readiness — Task 2 (Version Coherence)

Date: 2026-03-09

Checklist:
- [x] Add a failing regression test for version coherence.
- [x] Align package metadata version with the library version.
- [x] Remove hardcoded CLI/API version drift.
- [x] Re-run version and package-boundary verification.

Definition of done:
- `pyproject.toml`, `bist_quant.__version__`, docs version text, and remaining API metadata agree.
- CLI and API version surfaces derive from the package version instead of separate hardcoded literals.

Review:
- Added `tests/test_version_coherence.py` to pin version agreement across package metadata, docs, CLI source, and API source.
- Updated `pyproject.toml` package version from `2.0.0` to `0.3.0` to match the library’s canonical version.
- Updated `src/bist_quant/cli.py` to derive `--version` output from `bist_quant.__version__`.
- Updated `src/bist_quant/api/main.py` to derive FastAPI metadata version from the package version.

Verification:
- `pytest tests/test_package_boundary.py tests/test_version_coherence.py -v` -> PASS (`5 passed`)
- `python -c "import bist_quant; print(bist_quant.__version__)"` -> PASS (`0.3.0`)

## Library Publish Readiness — Task 3 (CLI and Public Export Integrity)

Date: 2026-03-09

Checklist:
- [x] Add failing regressions for the console script and top-level exports.
- [x] Fix the CLI entry point so the installed `bist-quant` script resolves a real `main()`.
- [x] Fix top-level optional client exports to import from `bist_quant.clients.*`.
- [x] Re-run publish-readiness regression verification.

Definition of done:
- Installed wheel exposes a working `bist-quant --version` command.
- `bist_quant.BorsapyAdapter` resolves from the `clients` package instead of silently becoming `None` due to stale import paths.

Review:
- Added `tests/test_public_api_surface.py` to verify:
  - `bist_quant.BorsapyAdapter` is a real export from `bist_quant.clients.borsapy_adapter`
  - a clean wheel install provides a working `bist-quant --version` console script
- Added canonical CLI implementation module `src/bist_quant/_cli.py`.
- Updated `src/bist_quant/cli/__init__.py` to export `main` from the canonical CLI implementation, making the existing `bist_quant.cli:main` console-script target valid.
- Removed the dead conflicting file `src/bist_quant/cli.py`.
- Fixed stale top-level optional client imports in `src/bist_quant/__init__.py` to import from `src/bist_quant/clients/`.
- Updated `tests/test_version_coherence.py` to track the canonical CLI implementation file.

Verification:
- `pytest tests/test_package_boundary.py tests/test_version_coherence.py tests/test_public_api_surface.py -v` -> PASS (`7 passed`)
- `python -c "import bist_quant, bist_quant.cli; print(bist_quant.__version__); print(hasattr(bist_quant.cli, 'main')); print(bist_quant.BorsapyAdapter.__module__)"` -> PASS (`0.3.0`, `True`, `bist_quant.clients.borsapy_adapter`)

## Library Publish Readiness — Task 4 (Runtime Path Decoupling)

Date: 2026-03-09

Checklist:
- [x] Add failing regressions for user-scoped runtime defaults.
- [x] Remove repo-root auto-detection from default `DataPaths` and runtime resolution.
- [x] Keep explicit env/argument overrides working while defaulting installed-library usage to XDG/home directories.
- [x] Re-run publish-readiness regression verification.

Definition of done:
- Default `DataPaths()` no longer points at the checked-out repo.
- Default `resolve_runtime_paths()` no longer depends on current working directory or source-tree layout.
- Clean wheel installs default to user-scoped directories.

Review:
- Added `tests/test_runtime_defaults.py` to verify source-tree and installed-wheel defaults for `DataPaths` and `resolve_runtime_paths()`.
- Updated `src/bist_quant/runtime.py` with user-scoped default directory helpers based on XDG/home conventions:
  - project root: `~/.local/share/bist-quant`
  - data dir: `~/.local/share/bist-quant/data`
  - regime dir: `~/.local/share/bist-quant/regime/simple_regime`
  - cache dir: `~/.cache/bist-quant`
- Removed parent-folder repository discovery from `src/bist_quant/common/data_paths.py` and switched its fallback behavior to the user-scoped defaults.
- Preserved explicit override behavior via `BIST_PROJECT_ROOT`, `BIST_DATA_DIR`, `BIST_REGIME_DIR`, `BIST_CACHE_DIR`, and direct function/class arguments.

Verification:
- `pytest tests/test_package_boundary.py tests/test_version_coherence.py tests/test_public_api_surface.py tests/test_runtime_defaults.py -v` -> PASS (`10 passed`)
- `python -c "from bist_quant.common.data_paths import DataPaths; from bist_quant.runtime import resolve_runtime_paths; p=DataPaths(); r=resolve_runtime_paths(); print(p.data_dir); print(p.cache_dir); print(r.project_root)"` -> PASS (`/home/safa/.local/share/bist-quant/data`, `/home/safa/.cache/bist-quant`, `/home/safa/.local/share/bist-quant`)

## Library Publish Readiness — Task 5 (Docs Coherence)

Date: 2026-03-09

Checklist:
- [x] Add failing regressions for public docs coherence.
- [x] Rewrite top-level docs for `pip install` and library-first usage.
- [x] Update install and quickstart docs for user-scoped runtime defaults.
- [x] Remove broken MkDocs nav entries that referenced missing API-reference pages.
- [x] Re-run publish-readiness regression verification.

Definition of done:
- README and docs home present `bist_quant` as a standalone Python library.
- Public installation docs lead with `pip install bist-quant`.
- Runtime defaults described in docs match the implemented XDG/home-scoped behavior.
- MkDocs nav no longer contains dead API-reference links.

Review:
- Added `tests/test_docs_public_story.py` to pin the public docs contract.
- Rewrote `README.md` for a library-first story:
  - `pip install bist-quant` is now the primary install path
  - app-stack material is removed from the top-level package README
  - default library directories now match runtime behavior
  - architecture/examples focus on the publishable library surface
- Updated `docs/getting-started/installation.md` to:
  - lead with PyPI install
  - move editable install to contributor context
  - document default user-scoped data/regime/cache directories
  - remove the old repo auto-detection description
- Updated `docs/getting-started/quickstart.md` and `docs/index.md` to reflect the new runtime defaults and library-first positioning.
- Removed broken API Reference nav entries from `mkdocs.yml` so docs navigation no longer points at missing files.

Verification:
- `pytest tests/test_package_boundary.py tests/test_version_coherence.py tests/test_public_api_surface.py tests/test_runtime_defaults.py tests/test_docs_public_story.py -v` -> PASS (`14 passed`)
- `python -c "from pathlib import Path; print('pip install bist-quant' in Path('README.md').read_text()); print('pip install bist-quant' in Path('docs/getting-started/installation.md').read_text())"` -> PASS (`True`, `True`)

## Library Publish Readiness — Task 6 (Dependency and Extras Strategy)

Date: 2026-03-09

Checklist:
- [x] Add failing regressions for the published extras contract.
- [x] Remove app-only extras from the published library package.
- [x] Add a library-facing provider extra that covers shipped optional client dependencies.
- [x] Narrow the `full` extra to library-relevant extras only.
- [x] Re-run publish-readiness regression verification.

Definition of done:
- The published package no longer advertises extras for excluded app modules.
- Optional shipped provider clients are represented by library-facing extras.
- `full` expands to research-library extras only.

Review:
- Added `tests/test_dependency_strategy.py` to pin the extras/dependency contract.
- Updated `pyproject.toml` optional dependencies:
  - removed app-only extras: `api`, `security`, `observability`, `services`, `app`
  - added `providers` extra for shipped optional client integrations (`httpx`, `yfinance`, `borsapy`)
  - kept library-facing extras: `borsapy`, `multi-asset`, `ml`, `dev`
  - narrowed `full` to `bist-quant[providers,multi-asset,borsapy,ml]`
- Updated `README.md` and `docs/getting-started/installation.md` to mention the new `providers` extra.

Verification:
- `pytest tests/test_package_boundary.py tests/test_version_coherence.py tests/test_public_api_surface.py tests/test_runtime_defaults.py tests/test_docs_public_story.py tests/test_dependency_strategy.py -v` -> PASS (`17 passed`)
- `python -c "import tomllib, pathlib; data=tomllib.loads(pathlib.Path('pyproject.toml').read_text()); extras=data['project']['optional-dependencies']; print(sorted(extras)); print(extras['full'][0])"` -> PASS (`['borsapy', 'dev', 'full', 'ml', 'multi-asset', 'providers']`, `bist-quant[providers,multi-asset,borsapy,ml]`)

## Library Publish Readiness — Final Release Validation

Date: 2026-03-09

Checklist:
- [x] Build wheel and sdist from the current tree.
- [x] Run `twine check` on built artifacts.
- [x] Install the wheel into a fresh virtualenv.
- [x] Verify import surface, CLI, and user-scoped defaults from the installed wheel.
- [x] Verify a corrected quickstart-style run from the installed wheel.
- [x] Capture any remaining non-blocking cleanup.

Definition of done:
- Artifacts build cleanly.
- Metadata passes `twine check`.
- Fresh-wheel install works.
- Installed wheel exposes working import/CLI behavior.
- A library-first example runs successfully from the installed artifact with explicit external data configuration.

Review:
- Installed `build` and `twine` in the local environment to perform artifact validation.
- Built clean release artifacts: `dist/bist_quant-0.3.0.tar.gz` and `dist/bist_quant-0.3.0-py3-none-any.whl`.
- Validated fresh-wheel install and CLI in isolated virtualenvs.
- Found and fixed one runtime blocker during validation: `DataManager.load_all()` always required regime outputs even when `use_regime_filter=False`.
- Updated `src/bist_quant/common/data_manager.py` and `src/bist_quant/portfolio.py` so minimal backtests can skip regime files when regime filtering is disabled.
- Updated surfaced quickstart snippets in `README.md`, `docs/index.md`, and `docs/getting-started/quickstart.md` to match the real API:
  - `PortfolioEngine(data_loader=loader, ...)`
  - `result['sharpe']` / `result['cagr']` instead of `result.metrics[...]`

Verification:
- `pytest tests/test_quickstart_runtime.py tests/test_docs_public_story.py tests/test_dependency_strategy.py tests/test_package_boundary.py tests/test_version_coherence.py tests/test_public_api_surface.py tests/test_runtime_defaults.py -v` -> PASS (`18 passed`)
- `python -m build && python -m twine check dist/*` -> PASS
- Fresh installed wheel checks:
  - `bist-quant --version` -> PASS (`bist-quant 0.3.0`)
  - installed import defaults -> PASS (user-scoped data/cache/project-root paths)
  - corrected quickstart-style example -> PASS (`prices_shape=(1360918, 7)`, `sharpe=1.7774`, `cagr=0.3294`)

## Library Publish Readiness — Final Cleanup and Release Prep

Date: 2026-03-09

Checklist:
- [x] Clean remaining stale docs/examples that referenced old constructor/result shapes.
- [x] Re-run focused docs and release-readiness verification.
- [x] Create a library-release preparation commit without frontend/app changes.
- [x] Prepare the PyPI publish command checklist.

Definition of done:
- Public docs/examples consistently use the validated library API.
- Release-prep work is committed.
- Publish commands are documented and ready to run.

Review:
- Updated the remaining stale examples and guides to replace:
  - `PortfolioEngine(loader=loader)` -> `PortfolioEngine(data_loader=loader, options={"use_regime_filter": False})`
  - `result.metrics[...]` -> `result[...]`
- Cleaned the following docs/examples:
  - `docs/examples/basic-backtest.md`
  - `docs/examples/custom-signals.md`
  - `docs/examples/portfolio-optimization.md`
  - `docs/user-guide/backtesting.md`
  - `docs/user-guide/portfolio.md`
  - `docs/user-guide/signals.md`
  - `src/bist_quant/README.md`
- Created commit `c80ce94` with message: `chore: prepare bist_quant for standalone library release`
- Left unrelated frontend/app working tree changes uncommitted.

Verification:
- `pytest tests/test_docs_public_story.py tests/test_quickstart_runtime.py tests/test_dependency_strategy.py -v` -> PASS (`8 passed`)
- stale-pattern grep across `docs/` for old quickstart forms -> PASS (no matches)
- `git status --short` after commit -> only unrelated frontend/app changes remain

## Library Publish Readiness — Comprehensive Module Verification

Date: 2026-03-10

Checklist:
- [x] Add a regression that imports every published `bist_quant` module.
- [x] Run the import-smoke regression and capture failures.
- [x] Fix any import/runtime blockers surfaced by full-module verification.
- [x] Re-run the full published-library verification suite.
- [x] Rebuild artifacts and confirm package integrity after the verification pass.

Definition of done:
- Every module included in the published wheel imports successfully in the source tree.
- Library regression suite passes.
- Artifacts still build and pass `twine check`.

Review:
- Added `tests/test_all_published_modules_import.py` to enumerate and import every module included in the published wheel, excluding the app-facing packages already removed from release artifacts.
- Ran the import-smoke verification and confirmed all published modules import successfully.
- Ran a broad published-library verification bundle spanning unit tests, integration workflow tests, provider tests, package-boundary tests, runtime/default-path tests, docs/install contract tests, and the new all-modules import smoke.
- Found one real verification blocker during the broad suite: `tests/test_borsapy_indicators.py` was still importing the old legacy top-level module path `borsapy_indicators`.
- Updated `tests/test_borsapy_indicators.py` to use the packaged import paths under `bist_quant.signals.borsapy_indicators` and `bist_quant.DataLoader`.

Verification:
- `pytest tests/test_all_published_modules_import.py -v` -> PASS (`1 passed`)
- `pytest tests/unit tests/integration/test_backtest_workflow.py tests/test_borsapy_indicators.py tests/test_signals_batch.py tests/test_fixed_income_provider.py tests/test_fx_enhanced_provider.py tests/test_economic_calendar_provider.py tests/test_derivatives_provider.py tests/test_streaming_provider.py tests/test_all_published_modules_import.py tests/test_quickstart_runtime.py tests/test_docs_public_story.py tests/test_dependency_strategy.py tests/test_package_boundary.py tests/test_version_coherence.py tests/test_public_api_surface.py tests/test_runtime_defaults.py -v` -> PASS (`271 passed`, `4 skipped`)
- `python -m build && python -m twine check dist/*` -> PASS
- Installed-wheel behavior smoke (`bist_quant==0.3.1` from TestPyPI with local data mode) -> PASS:
  - strategy registry lookup
  - signal registry lookup
  - `DataPaths` default/runtime behavior
  - `DataLoader.load_prices()` + close panel build
  - `signals.factory.build_signal()` with runtime context
  - `signals.borsapy_indicators` local indicator calculations
  - `PortfolioEngine.run_factor("momentum")` with `use_regime_filter=False`
