# Quant AI Platform

> **AI-Powered Multi-Agent Trading Intelligence for Emerging Markets**

[![Next.js](https://img.shields.io/badge/Next.js-16-black?style=flat-square&logo=next.js)](https://nextjs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue?style=flat-square&logo=typescript)](https://typescriptlang.org)
[![Vercel AI SDK](https://img.shields.io/badge/Vercel-AI%20SDK-black?style=flat-square&logo=vercel)](https://sdk.vercel.ai)

## Overview

Quant AI is an institutional-grade quantitative trading platform that combines **34 proven factor models** with a **multi-agent AI system** for portfolio management, risk monitoring, and market analysis. Built for emerging markets, starting with Borsa Istanbul (BIST).

### Key Metrics (February 2026)

| Metric | Value |
|--------|-------|
| Factor Strategies | **34** |
| Top Signal CAGR | **102.62%** (breakout_value) |
| Best Sharpe Ratio | **2.93** |
| Average CAGR (34 signals) | **~63%** |
| Top Signal Alpha | **83.03%** ann. |
| Top Signal Beta | **0.36** |
| Lowest Max Drawdown | **-26.90%** (donchian) |
| Backtest Period | **2017–2026** |

## Architecture

```
┌──────────────────────────────────────────┐
│           Next.js Frontend               │
│  Landing Page • Dashboard • AI Agents    │
├──────────────────────────────────────────┤
│        AI Agent Orchestrator             │
│  Portfolio Manager • Risk Manager        │
│        Market Analyst                    │
├──────────────────────────────────────────┤
│        Quant Engine (34 Factors)         │
│  Value • Momentum • Quality • Breakout   │
│  Trend • Size • Multi-Factor • Macro     │
├──────────────────────────────────────────┤
│        Regime Detection (Simple 2D)      │
│  Trend (MA) × Volatility (Percentile)    │
│  4 States: Bull / Bear / Recovery / Stress│
└──────────────────────────────────────────┘
```

## Multi-Agent AI System

Three specialized agents collaborate to manage your portfolio:

- 🎯 **Portfolio Manager** — Factor allocations, rebalancing, position sizing
- 🛡️ **Risk Manager** — Drawdown monitoring, vol-targeting, regime-based risk management
- 🧠 **Market Analyst** — BIST trends, sector rotation, macro analysis

## Factor Models

The platform runs **34 factor strategies** across 10 categories:

| Category | Top Strategy | CAGR | Sharpe |
|----------|-------------|------|--------|
| Breakout × Value | `breakout_value` | 102.62% | 2.93 |
| Size × Momentum | `small_cap_momentum` | 95.24% | 2.47 |
| Multi-Factor | `five_factor_rotation` (13 axes) | 88.00% | 2.47 |
| Trend × Value | `trend_value` | 88.17% | 2.66 |
| Trend (Channel) | `donchian` | 81.98% | 2.49 |
| Quality × Momentum | `quality_momentum` | 78.63% | 2.37 |
| Growth | `asset_growth` | 74.61% | 2.33 |
| Size Rotation | `size_rotation` | 65.22% | 2.02 |
| Macro Hedge | `macro_hedge` | 57.90% | 2.04 |
| Low Volatility | `low_volatility` | 49.76% | 2.06 |

All strategies include:
- Transaction costs and slippage
- Liquidity filters
- Survivorship bias treatment
- CAPM alpha/beta vs XU100 and XAU/TRY benchmarks

## Tech Stack

- **Frontend**: Next.js 16 (App Router), React 19, TypeScript
- **Styling**: Custom CSS design system (glassmorphism, dark theme)
- **Charts**: Recharts
- **AI**: Vercel AI SDK (multi-agent orchestration)
- **Backend**: Python quantitative engine (pandas, numpy, scipy)
- **Regime Detection**: Simple 2D classifier (trend × volatility → 4 regimes)
- **Data**: Parquet files with zstd compression
- **Icons**: Lucide React

## Getting Started

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Start production server
npm run start
```

The app runs at `http://localhost:3000`.

### Required Environment Variables (Agent APIs)

Create `.env.local` with:

```bash
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com
AZURE_OPENAI_API_KEY=<your-azure-openai-key>
AZURE_OPENAI_DEPLOYMENT=<your-deployment-name>
AZURE_OPENAI_API_VERSION=2024-10-21
```

The app performs startup checks and will fail fast if required Azure OpenAI vars are missing.

### Agent Diagnostics

- `GET /api/agents/health` runs a lightweight live Azure OpenAI connectivity/deployment check.
- Agent APIs emit structured JSON logs (request, Azure call, response, errors) with request IDs and latency.

## Project Structure

```
bist-quant-ai/
├── src/
│   ├── app/
│   │   ├── page.tsx              # Landing page
│   │   ├── dashboard/page.tsx    # Trading dashboard
│   │   ├── agents/page.tsx       # AI Agents showcase
│   │   ├── globals.css           # Design system
│   │   ├── layout.tsx            # Root layout
│   │   └── api/
│   │       ├── dashboard/route.ts  # Dashboard data API
│   │       ├── signals/route.ts    # Signal data API
│   │       └── agents/
│   │           ├── _handler.ts     # Shared agent handler
│   │           ├── _shared.ts      # Agent config & types
│   │           ├── health/route.ts # Health check endpoint
│   │           ├── portfolio/route.ts
│   │           ├── risk/route.ts
│   │           └── analyst/route.ts
│   ├── components/
│   │   ├── Navbar.tsx            # Glass navigation
│   │   ├── SignalTable.tsx       # Sortable signal table
│   │   ├── EquityChart.tsx       # Interactive equity curves
│   │   ├── RegimeIndicator.tsx   # Market regime badge
│   │   ├── PortfolioView.tsx     # Holdings grid
│   │   └── AgentChat.tsx         # AI multi-agent chat
│   └── lib/
│       ├── agents/
│       │   ├── orchestrator.ts   # Agent coordination logic
│       │   └── logging.ts        # Structured JSON logging
│       └── server/
│           └── dashboardData.ts  # Server-side data loading
├── public/
│   └── data/
│       ├── dashboard_data.json   # Aggregated signal metrics
│       └── equity_curves.json    # Historical equity curves
├── dashboard/                    # Legacy dashboard (FastAPI + Next.js)
│   ├── generate_dashboard_data.py  # Data pipeline for dashboard JSON
│   ├── dashboard_data.json
│   └── app/
│       ├── backend/              # FastAPI backend
│       └── frontend/             # Legacy Next.js frontend
├── package.json
├── next.config.ts
├── tsconfig.json
└── vercel.json
```

## Deployment

Deployed on [Vercel](https://vercel.com):

```bash
npx vercel
```

## Parent Repository

This web dashboard is part of the [BIST Quant Research Repository](../README.md), which contains the Python quantitative engine, 34 factor strategies, and the regime detection pipeline.

## License

Proprietary — All rights reserved.

---

*Built with ❤️ for emerging markets*
