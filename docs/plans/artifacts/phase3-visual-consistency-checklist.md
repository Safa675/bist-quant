# Phase 3 Visual Consistency Checklist

Date: 2026-03-04
Scope: `/dashboard`, `/backtest`, `/factor-lab`, `/signal-construction`, `/screener`, `/analytics`, `/optimization`, `/professional`, `/compliance`, `/agents`

## Criteria

- Exactly one `h1` page title rendered via shared `PageHeader`.
- Shared scaffold markers present: `data-ui-scaffold`, `data-ui-main`.
- Spacing/typography uses tokenized classes and CSS variables.
- No raw debug JSON blocks (`<pre>`) in product views.
- No route-level raw `<select>`, `<textarea>`, `<table>` tags in `src/app/**`.
- Tables render through shared `DataTable` styling.
- Empty/loading/error states use shared card and token styling.

## Route Status

| Route | H1 + Header | Scaffold | Shared Controls | No Raw Debug JSON | Table Consistency | Empty/Error States | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `/dashboard` | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `/backtest` | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `/factor-lab` | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `/signal-construction` | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `/screener` | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `/analytics` | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `/optimization` | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `/professional` | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `/compliance` | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `/agents` | PASS | PASS | PASS | PASS | PASS | PASS | PASS |

## Gate Outcome

- Visual consistency checklist status: **PASS (10/10 routes)**.
