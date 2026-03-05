# Phase 4 Regression Evidence

Date: 2026-03-05

## 1) `pytest`

- Result: `280 passed, 4 skipped, 2 warnings`
- Command: `pytest`

```
================= 280 passed, 4 skipped, 2 warnings in 15.31s ==================
```

## 2) `ruff check src tests`

- Result: `PASS`
- Command: `ruff check src tests`

```
All checks passed!
```

## 3) `npm --prefix frontend run lint`

- Result: `PASS`
- Command: `npm --prefix frontend run lint`

```
[ui-primitives] PASS
```

## 4) `npm --prefix frontend run test`

- Result: `PASS`
- Command: `npm --prefix frontend run test`

```
Test Files  17 passed (17)
Tests       92 passed (92)
```

## 5) `npm --prefix frontend run build`

- Result: `PASS`
- Command: `npm --prefix frontend run build`

```
Route (app)
├ ○ /agents
├ ○ /analytics
├ ○ /backtest
├ ○ /compliance
├ ○ /dashboard
├ ○ /factor-lab
├ ○ /optimization
├ ○ /professional
├ ○ /screener
└ ○ /signal-construction
```

## 6) `npm --prefix frontend exec playwright test`

- Result: `PASS`
- Command: `npm --prefix frontend exec playwright test`

```
17 passed (21.7s)
```
