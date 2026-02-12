# Vercel AI Accelerator Application — Quant AI Platform

## Company / Project Name
**Quant AI Platform**

## Tagline
AI-Powered Multi-Agent Trading Intelligence for Emerging Markets

---

## What are you building? (250 words)

We're building the first AI-native quantitative trading platform designed for emerging markets. Our system combines 34 proven factor models with a multi-agent AI architecture that autonomously manages portfolios, monitors risk, and explains every decision in natural language.

The platform has three AI agents that collaborate:
- **Portfolio Manager** — Manages factor allocations and rebalancing using signals with up to 102.62% CAGR (backtested on BIST, 2017–2026)
- **Risk Manager** — Monitors drawdowns, regime shifts, and volatility targeting using a simple 2D regime classifier (trend × volatility → 4 states)
- **Market Analyst** — Analyzes sector rotations, macro drivers, and market conditions

Unlike traditional quant platforms that are black boxes, our agents explain WHY they make decisions using plain language — democratizing institutional-grade tools for retail investors and small prop trading firms in emerging markets.

Our core engine is proven: 34 factor models across value, momentum, quality, breakout, trend, size, and macro strategies. The top signal (breakout_value) achieves 102.62% CAGR with a 2.93 Sharpe ratio across 9+ years of backtesting on Borsa Istanbul, including transaction costs, slippage, and regime-based risk management.

The web platform is built on Next.js with the Vercel AI SDK powering agent orchestration. The regime detection pipeline automatically rotates between equities and gold during market stress — the macro_hedge strategy achieves 57.90% CAGR with only -30.19% max drawdown.

We're starting with BIST (Turkey) and expanding to other emerging markets where retail investors lack access to sophisticated quantitative tools.

---

## How does AI feature in your product?

AI is the core of our product across three layers:

### 1. Multi-Agent AI System (Vercel AI SDK)
Three specialized agents coordinate using the Vercel AI SDK:
- Agents share context about market state, holdings, and risk metrics
- Each agent has a distinct persona and expertise domain
- Natural language explanations make complex quant decisions accessible

### 2. Regime Detection
A simple but effective 2-dimensional regime classifier:
- **Dimension 1**: Trend — Price vs moving average
- **Dimension 2**: Volatility — Realized vol percentile (20d)
- **Output**: 4 regimes (Bull, Bear, Recovery, Stress)
- Hysteresis filter prevents regime-flipping noise

### 3. Factor Signal Generation
Quantitative factor scoring with multi-axis construction:
- 13-axis multi-factor rotation (size, value, profitability, momentum, quality, liquidity, trading intensity, sentiment, fundamental momentum, carry, defensive, risk)
- Quintile-based bucket selection with multi-lookback ensemble
- Exponentially-weighted factor selection favoring recent performance
- Dynamic signal combination based on regime context

---

## What stage are you at?

**Working Product with Real Data**

- ✅ 34 factor models backtested on 9+ years of BIST data
- ✅ Top signal: 102.62% CAGR, 2.93 Sharpe, 0.36 Beta
- ✅ Average across all 34 strategies: ~63% CAGR, ~2.06 Sharpe
- ✅ Regime detection pipeline integrated into portfolio engine
- ✅ Next.js web platform with interactive dashboard
- ✅ Multi-agent AI chat interface (Vercel AI SDK + Azure OpenAI)
- ✅ Real-time signal monitoring and portfolio analytics
- ✅ Agent health checks and structured logging
- 🔜 Live trading integration (paper trading → live)
- 🔜 Multi-market expansion (beyond BIST)

---

## What makes you different?

### 1. Explainable AI for Quant Trading
No other platform combines factor investing with conversational AI agents that explain reasoning. Traditional quant is a black box.

### 2. Emerging Market Focus
Most quant platforms target US/EU markets. We built from scratch for emerging markets where data is messier, markets are less efficient (more alpha), and retail investors have fewer tools.

### 3. Regime-Adaptive Risk Management
Our simple 2D regime classifier detects market state changes and rotates to gold in Bear/Stress markets — achieving 57.90% CAGR on macro_hedge with only -30.19% max drawdown.

### 4. Proven Performance
Not theoretical — our backtests include:
- Transaction costs (realistic slippage)
- Liquidity filters (only tradeable stocks)
- Survivorship bias treatment
- 9+ years of data across multiple market cycles

### 5. Multi-Agent Architecture
Three specialized agents (portfolio, risk, analyst) that share context and collaborate. Each has domain expertise and provides explainable, actionable insights.

---

## Team

**Founder / Solo Developer**

- Built the entire quantitative research framework from scratch
- 10+ years of market data processing and analysis
- Full-stack development: Python backend + Next.js/React frontend
- Deep expertise in factor investing, portfolio construction, and risk management
- Passionate about democratizing finance in emerging markets

---

## Why Vercel AI Accelerator?

1. **AI SDK Integration** — Our multi-agent system is built directly on the Vercel AI SDK. We need credits and optimization guidance to scale LLM calls for real-time portfolio management.

2. **Deployment & Scale** — Vercel is our deployment platform. As we expand to multi-market coverage, we need the infrastructure to handle real-time data streams and agent coordination.

3. **Community & Mentorship** — We're a solo founder building something ambitious. Access to AI engineering expertise and the Vercel ecosystem would accelerate our path to production.

4. **Vision Alignment** — We're using AI to make complex financial tools accessible. This aligns with Vercel's mission of making powerful technology accessible to everyone.

---

## Technical Architecture

```
┌─────────────────────────────────────────┐
│              Next.js Frontend           │
│  ┌─────────┐ ┌──────────┐ ┌──────────┐ │
│  │Dashboard │ │ Signals  │ │ Holdings │ │
│  └─────────┘ └──────────┘ └──────────┘ │
│                    │                     │
│         ┌──────────┴──────────┐         │
│         │  AI Agent Orchestrator│         │
│         │   (Vercel AI SDK)    │         │
│         └──────────┬──────────┘         │
│    ┌───────────────┼───────────────┐    │
│    │               │               │    │
│ ┌──┴──┐      ┌────┴────┐    ┌────┴──┐ │
│ │Port.│      │  Risk   │    │Market │ │
│ │Mgr  │      │  Mgr    │    │Analyst│ │
│ └──┬──┘      └────┬────┘    └────┬──┘ │
│    └───────────────┼───────────────┘    │
│                    │                     │
│         ┌──────────┴──────────┐         │
│         │   Quant Engine      │         │
│         │  (34 Factor Models) │         │
│         └──────────┬──────────┘         │
│                    │                     │
│         ┌──────────┴──────────┐         │
│         │  Regime Detection   │         │
│         │ (Trend × Volatility)│         │
│         └─────────────────────┘         │
└─────────────────────────────────────────┘
```

---

## Key Metrics (February 2026)

| Metric | Value |
|--------|-------|
| Top Signal CAGR | 102.62% |
| Best Sharpe Ratio | 2.93 |
| Average CAGR (34 signals) | ~63% |
| Average Sharpe (34 signals) | ~2.06 |
| Top Signal Beta | 0.36 |
| Top Signal Alpha | 83.03% ann. |
| Backtest Period | 2017–2026 (9+ years) |
| Max Drawdown (top signal) | -31.47% |
| Market Covered | BIST (Borsa Istanbul) |
| Factor Models | 34 |
| AI Agents | 3 (Portfolio, Risk, Analyst) |

---

## Links

- **Live Demo**: [To be deployed on Vercel]
- **GitHub**: [Repository link]
- **Dashboard**: /dashboard
- **Landing Page**: /

---

*Application prepared for the Vercel AI Accelerator — February 2026*
