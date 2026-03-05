# BIST Quant Frontend

Phase 2 scaffold for replacing Streamlit with Next.js.

## Run

```bash
npm install
npm run dev
```

Set backend URL if needed:

```bash
export NEXT_PUBLIC_API_URL=http://127.0.0.1:8001
```

## Current scope

- Dashboard foundation page (`src/app/page.tsx`)
- Typed API client (`src/lib/api.ts`)
- KPI component (`src/components/KpiGrid.tsx`)
