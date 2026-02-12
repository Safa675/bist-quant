# BIST Daily Portfolio Dashboard

## Overview

Legacy dashboard app for daily portfolio monitoring, consisting of:
- `frontend/` — Next.js dashboard UI (TailwindCSS + Framer Motion)
- `backend/` — FastAPI serving dashboard data

> **Note**: This is the original standalone dashboard. The primary web platform is now the parent `bist-quant-ai/` Next.js app with integrated AI agents. This legacy dashboard may be deprecated.

## Tech Stack
- Frontend: Next.js 14 + TailwindCSS + Framer Motion
- Backend: FastAPI + Pydantic
- Data source: `generate_dashboard_data.py` → `dashboard_data.json`

## Frontend Routes
- `/` → redirects to `/dashboard`
- `/dashboard` → full daily portfolio dashboard

## Backend Endpoints
- `GET /api/health`
- `GET /api/dashboard`
- `GET /api/regime`
- `GET /api/signals`
- `GET /api/signals/{signal_name}`
- `GET /api/portfolio/holdings`
- `GET /api/portfolio/trades`
- `GET /api/portfolio/daily`
- `GET /api/portfolio/summary`
- `GET /api/stats`

## Run
1. Backend:
   ```bash
   cd bist-quant-ai/dashboard/app/backend
   uvicorn main:app --reload --port 8000
   ```
2. Frontend:
   ```bash
   cd bist-quant-ai/dashboard/app/frontend
   npm install
   npm run dev
   ```

## Data Refresh

Regenerate dashboard payload from latest backtest results:
```bash
cd bist-quant-ai/dashboard
python generate_dashboard_data.py
```

This reads from `Models/results/` and produces `dashboard_data.json` with signal metrics, equity curves, and portfolio analytics.
