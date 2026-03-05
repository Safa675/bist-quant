#!/usr/bin/env python3
"""Generate Phase 0 contract-freeze artifacts for Streamlit -> Next.js migration.

Outputs:
- docs/plans/artifacts/phase0-contract-manifest.json
- docs/plans/artifacts/phase0-canonical-samples.json
- docs/plans/2026-03-04-phase0-contract-freeze.md

The canonical samples are produced via a mocked in-process harness by calling
FastAPI route handlers directly (no HTTP server required).
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack
from datetime import UTC, datetime
import json
from pathlib import Path
from time import sleep
from typing import Any
from unittest.mock import patch

import pandas as pd

from fastapi import HTTPException

from bist_quant.api import main as api_main
from bist_quant.api.jobs import JobManager
from bist_quant.api.routers.analytics import analytics_run
from bist_quant.api.routers.compliance import (
    compliance_check,
    compliance_position_limits,
    compliance_rules,
)
from bist_quant.api.routers.factors import factors_combine, factors_detail, factors_snapshot
from bist_quant.api.routers.optimization import optimization_run
from bist_quant.api.routers.professional import (
    professional_crypto_sizing,
    professional_greeks,
    professional_stress,
)
from bist_quant.api.routers.screener import screener_run, screener_sparklines
from bist_quant.api.schemas import (
    AnalyticsRunRequest,
    ComplianceTransactionRequest,
    CryptoSizingRequest,
    FactorCombineRequest,
    FactorSnapshotRequest,
    GreeksRequest,
    OptimizationRunRequest,
    PositionLimitsRequest,
    ScreenerRunRequest,
    StressTestRequest,
)
from bist_quant.services import CoreBackendService, SystemService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "docs" / "plans" / "artifacts"
MANIFEST_PATH = ARTIFACT_DIR / "phase0-contract-manifest.json"
SAMPLES_PATH = ARTIFACT_DIR / "phase0-canonical-samples.json"
DOC_PATH = PROJECT_ROOT / "docs" / "plans" / "2026-03-04-phase0-contract-freeze.md"


API_FUNCTION_ENDPOINTS: dict[str, list[dict[str, str]]] = {
    "getDashboardOverview": [{"method": "GET", "path": "/api/dashboard/overview"}],
    "getDashboardRegimeHistory": [{"method": "GET", "path": "/api/dashboard/regime-history"}],
    "getDashboardMacro": [{"method": "GET", "path": "/api/dashboard/macro"}],
    "getFactorCatalog": [{"method": "GET", "path": "/api/meta/signals"}],
    "getFactorDetail": [{"method": "GET", "path": "/api/factors/{name}"}],
    "getMacroCalendar": [{"method": "GET", "path": "/api/macro/calendar"}],
    "runBacktest": [{"method": "POST", "path": "/api/backtest/run"}],
    "createJob": [{"method": "POST", "path": "/api/jobs"}],
    "getJob": [{"method": "GET", "path": "/api/jobs/{job_id}"}],
    "listJobs": [{"method": "GET", "path": "/api/jobs"}],
    "cancelJob": [{"method": "DELETE", "path": "/api/jobs/{job_id}"}],
    "retryJob": [{"method": "POST", "path": "/api/jobs/{job_id}/retry"}],
    "runScreener": [{"method": "POST", "path": "/api/screener/run"}],
    "getSparklines": [{"method": "POST", "path": "/api/screener/sparklines"}],
    "combineFactors": [{"method": "POST", "path": "/api/jobs"}],
    "runAnalytics": [{"method": "POST", "path": "/api/analytics/run"}],
    "runOptimization": [{"method": "POST", "path": "/api/jobs"}],
    "calculateGreeks": [{"method": "POST", "path": "/api/professional/greeks"}],
    "runStressTest": [{"method": "POST", "path": "/api/professional/stress"}],
    "calculateCryptoSizing": [{"method": "POST", "path": "/api/professional/crypto-sizing"}],
    "runComplianceCheck": [{"method": "POST", "path": "/api/compliance/check"}],
    "getDefaultComplianceRules": [{"method": "GET", "path": "/api/compliance/rules"}],
    "checkPositionLimits": [{"method": "POST", "path": "/api/compliance/position-limits"}],
    "getSystemMeta": [{"method": "GET", "path": "/api/meta/system"}],
    "getHealthStatus": [{"method": "GET", "path": "/api/health/live"}],
}


ROUTE_DEFINITIONS: list[dict[str, Any]] = [
    {
        "route": "/",
        "page": "frontend/src/app/page.tsx",
        "streamlit_source": "app/pages/1_Dashboard.py",
        "frontend_calls": [],
        "execution_mode": "redirect",
        "status": "match",
        "notes": "Root route redirects to /dashboard.",
    },
    {
        "route": "/dashboard",
        "page": "frontend/src/app/dashboard/page.tsx",
        "streamlit_source": "app/pages/1_Dashboard.py",
        "frontend_calls": ["getDashboardOverview"],
        "execution_mode": "sync",
        "status": "match",
        "notes": "Core dashboard overview path is aligned.",
    },
    {
        "route": "/backtest",
        "page": "frontend/src/app/backtest/page.tsx",
        "streamlit_source": "app/pages/2_Backtest.py",
        "frontend_calls": ["getFactorCatalog", "createJob", "getJob", "listJobs", "cancelJob"],
        "execution_mode": "async_job",
        "status": "adapter",
        "notes": "Backtest job result is normalized by toBacktestUiResult inside getJob().",
    },
    {
        "route": "/factor-lab",
        "page": "frontend/src/app/factor-lab/page.tsx",
        "streamlit_source": "app/pages/3_Factor_Lab.py",
        "frontend_calls": ["getFactorCatalog", "getFactorDetail", "createJob", "getJob"],
        "execution_mode": "async_job",
        "status": "mismatch",
        "notes": "Current UI sends signals as string[] while backend expects [{name, weight}] for factor_combine jobs.",
    },
    {
        "route": "/signal-construction",
        "page": "frontend/src/app/signal-construction/page.tsx",
        "streamlit_source": "app/pages/4_Signal_Construction.py",
        "frontend_calls": ["getFactorCatalog", "createJob", "getJob"],
        "execution_mode": "async_job",
        "status": "adapter",
        "notes": "Uses backtest job path; requires consistent BacktestUiResult mapping for all chart paths.",
    },
    {
        "route": "/screener",
        "page": "frontend/src/app/screener/page.tsx",
        "streamlit_source": "app/pages/5_Screener.py",
        "frontend_calls": ["runScreener"],
        "execution_mode": "sync",
        "status": "adapter",
        "notes": "Result count/rows normalized by toScreenerUiResult.",
    },
    {
        "route": "/analytics",
        "page": "frontend/src/app/analytics/page.tsx",
        "streamlit_source": "app/pages/6_Analytics.py",
        "frontend_calls": ["createJob", "getJob"],
        "execution_mode": "async_job_noncanonical",
        "status": "mismatch",
        "notes": "Canonical target is direct POST /api/analytics/run; current UI uses lightweight analytics job payload.",
    },
    {
        "route": "/optimization",
        "page": "frontend/src/app/optimization/page.tsx",
        "streamlit_source": "app/pages/7_Optimization.py",
        "frontend_calls": ["getFactorCatalog", "createJob", "getJob"],
        "execution_mode": "async_job",
        "status": "mismatch",
        "notes": "UI request shape diverges from backend OptimizationRunRequest (parameter_space expected).",
    },
    {
        "route": "/professional",
        "page": "frontend/src/app/professional/page.tsx",
        "streamlit_source": "app/pages/8_Professional.py",
        "frontend_calls": ["calculateGreeks", "runStressTest", "calculateCryptoSizing"],
        "execution_mode": "sync",
        "status": "match",
        "notes": "Direct endpoint mapping is aligned.",
    },
    {
        "route": "/compliance",
        "page": "frontend/src/app/compliance/page.tsx",
        "streamlit_source": "app/pages/9_Compliance.py",
        "frontend_calls": ["getDefaultComplianceRules", "runComplianceCheck", "checkPositionLimits"],
        "execution_mode": "sync",
        "status": "adapter",
        "notes": "UI aliases comparator->operator and passed->status via adapters.",
    },
    {
        "route": "/agents",
        "page": "frontend/src/app/agents/page.tsx",
        "streamlit_source": "app/pages/10_Agents.py",
        "frontend_calls": [],
        "execution_mode": "placeholder",
        "status": "placeholder",
        "notes": "Intentional beta placeholder (no backend dependencies).",
    },
]


SHARED_SURFACES: list[dict[str, Any]] = [
    {
        "surface": "AppShell",
        "path": "frontend/src/components/shared/app-shell.tsx",
        "frontend_calls": [
            {
                "method": "GET",
                "path": "/api/dashboard/overview",
                "purpose": "Regime label badge (lookback=30)",
            }
        ],
        "status": "match",
    }
]


DTO_MAPPINGS: list[dict[str, str]] = [
    {
        "ui_dto": "BacktestUiResult",
        "backend_field": "equity_curve[].strategy | equity_curve[].value",
        "ui_field": "equity_curve[].strategy",
        "transform": "prefer strategy, fallback to value",
        "notes": "Supports mixed backend payload shapes.",
    },
    {
        "ui_dto": "BacktestUiResult",
        "backend_field": "drawdown_curve[].drawdown | drawdown_curve[].value",
        "ui_field": "drawdown_curve[].drawdown",
        "transform": "prefer drawdown, fallback to value",
        "notes": "Normalizes historical drawdown formats.",
    },
    {
        "ui_dto": "BacktestUiResult",
        "backend_field": "monthly_returns (array | object)",
        "ui_field": "monthly_returns[year][month]",
        "transform": "month-key normalization",
        "notes": "Accepts both keyed and row-oriented payloads.",
    },
    {
        "ui_dto": "BacktestUiResult",
        "backend_field": "holdings[] | top_holdings[]",
        "ui_field": "holdings[]",
        "transform": "symbol/ticker aliasing",
        "notes": "Unifies holdings shape for table/chart consumers.",
    },
    {
        "ui_dto": "AnalyticsUiResult",
        "backend_field": "performance.* + risk.*",
        "ui_field": "metrics.*",
        "transform": "merged metrics map",
        "notes": "Single KPI source in UI.",
    },
    {
        "ui_dto": "AnalyticsUiResult",
        "backend_field": "rolling[].rolling_sharpe_63d | rolling[].rolling_sharpe",
        "ui_field": "rolling[].rolling_sharpe",
        "transform": "alias fallback",
        "notes": "Backward compatibility with legacy rolling keys.",
    },
    {
        "ui_dto": "OptimizationUiResult",
        "backend_field": "best_trial.params",
        "ui_field": "best_params",
        "transform": "direct projection",
        "notes": "Primary sweep summary.",
    },
    {
        "ui_dto": "OptimizationUiResult",
        "backend_field": "best_trial.metrics[metricKey] | best_trial.score",
        "ui_field": "best_metric",
        "transform": "metric-key fallback to score",
        "notes": "Handles metric-key drift.",
    },
    {
        "ui_dto": "OptimizationUiResult",
        "backend_field": "trials[]",
        "ui_field": "sweep_results[]",
        "transform": "numeric params + metric extraction",
        "notes": "Feeds heatmap/scatter visuals.",
    },
    {
        "ui_dto": "ScreenerUiResult",
        "backend_field": "count | meta.total_matches | rows.length",
        "ui_field": "count",
        "transform": "fallback cascade",
        "notes": "Stabilizes KPI count across schema variants.",
    },
    {
        "ui_dto": "ComplianceUiResult",
        "backend_field": "status | passed",
        "ui_field": "status",
        "transform": "passed=false -> FAIL else PASS",
        "notes": "Wave 1 frozen alias rule: passed -> status.",
    },
    {
        "ui_dto": "ComplianceUiResult",
        "backend_field": "rule.comparator | rule.operator",
        "ui_field": "rule.operator",
        "transform": "comparator alias to operator",
        "notes": "Wave 1 frozen alias rule: comparator -> operator.",
    },
    {
        "ui_dto": "ComplianceUiResult",
        "backend_field": "hit.limit | rule.threshold",
        "ui_field": "hits[].limit",
        "transform": "fallback to rule threshold",
        "notes": "Ensures deterministic limit display.",
    },
    {
        "ui_dto": "ScreenerUiResult",
        "backend_field": "rows[].return_1m (fraction)",
        "ui_field": "table cell 1M Ret %",
        "transform": "display multiply by 100",
        "notes": "Frozen metric unit annotation for Wave 1.",
    },
]


DRIFT_REGISTER: list[dict[str, str]] = [
    {
        "surface": "Factor Lab",
        "classification": "mismatch",
        "detail": "combine payload currently sends signals:string[] while backend contract expects signals:[{name, weight}].",
        "resolution_wave": "Wave 1",
    },
    {
        "surface": "Analytics",
        "classification": "mismatch",
        "detail": "UI uses async jobs endpoint with non-canonical method labels; canonical path is POST /api/analytics/run.",
        "resolution_wave": "Wave 1",
    },
    {
        "surface": "Optimization",
        "classification": "mismatch",
        "detail": "UI payload diverges from OptimizationRunRequest parameter_space contract.",
        "resolution_wave": "Wave 1",
    },
    {
        "surface": "Backtest + Signal Construction",
        "classification": "adapter",
        "detail": "Adapter normalization required for value/strategy keys and drawdown/value variants.",
        "resolution_wave": "Wave 1",
    },
    {
        "surface": "Compliance",
        "classification": "adapter",
        "detail": "Alias mapping required for comparator/operator and passed/status.",
        "resolution_wave": "Wave 1",
    },
    {
        "surface": "Screener",
        "classification": "adapter",
        "detail": "Count fallback and metric-unit display normalization required.",
        "resolution_wave": "Wave 1",
    },
]


UNKNOWN_FIELD_REGISTER: list[dict[str, str]] = []


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _normalize_endpoint(method: str, path: str) -> str:
    return f"{method.upper()} {path}"


def _endpoint_set_from_calls(calls: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for call in calls:
        rows.extend(API_FUNCTION_ENDPOINTS.get(call, []))
    uniq = {(r["method"], r["path"]): r for r in rows}
    return [uniq[k] for k in sorted(uniq)]


def _mock_xu100_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=90, freq="B")
    close = [10000 + (idx * 12.5) for idx in range(len(dates))]
    return pd.DataFrame({"Date": dates, "Close": close})


def _mock_regime_payload(_: Any) -> dict[str, Any]:
    return {
        "label": "Bull",
        "current": {
            "date": "2024-05-03",
            "above_ma": True,
            "allocation": 0.85,
            "realized_vol": 0.21,
            "vol_percentile": 0.62,
        },
        "series": [
            {"date": "2024-04-29", "regime": "Bull"},
            {"date": "2024-04-30", "regime": "Bull"},
            {"date": "2024-05-02", "regime": "Sideways"},
            {"date": "2024-05-03", "regime": "Bull"},
        ],
        "distribution": [
            {"regime": "Bull", "count": 150, "percent": 55.0},
            {"regime": "Sideways", "count": 80, "percent": 29.3},
            {"regime": "Bear", "count": 43, "percent": 15.7},
        ],
    }


def _mock_macro_payload(_: Any, lookback: int) -> dict[str, Any]:
    tail = max(5, min(lookback, 30))
    base_dates = pd.date_range("2024-04-01", periods=tail, freq="B")
    usd = [31.0 + i * 0.04 for i in range(tail)]
    gold = [2100.0 + i * 3.0 for i in range(tail)]
    return {
        "series": {
            "usdtry": [
                {"date": str(base_dates[i].date()), "value": float(usd[i])}
                for i in range(tail)
            ],
            "xau_try": [
                {"date": str(base_dates[i].date()), "value": float(gold[i])}
                for i in range(tail)
            ],
        },
        "changes": [
            {"asset": "USD/TRY", "current": usd[-1], "d1_pct": 0.21, "w1_pct": 0.64, "m1_pct": 1.44},
            {"asset": "Gold (TRY/oz)", "current": gold[-1], "d1_pct": 0.18, "w1_pct": 0.73, "m1_pct": 1.80},
        ],
    }


def _fake_run_backtest(self: CoreBackendService, **_: Any) -> dict[str, Any]:
    return {
        "metrics": {
            "cagr": 0.124,
            "sharpe": 1.46,
            "sortino": 1.92,
            "max_drawdown": -0.214,
            "annualized_volatility": 0.22,
            "win_rate": 56.2,
        },
        "equity_curve": [
            {"date": "2024-01-02", "value": 100.0, "benchmark": 100.0, "drawdown": 0.0},
            {"date": "2024-01-03", "value": 101.2, "benchmark": 100.3, "drawdown": -0.004},
            {"date": "2024-01-04", "value": 99.8, "benchmark": 99.9, "drawdown": -0.018},
            {"date": "2024-01-05", "value": 102.1, "benchmark": 100.5, "drawdown": -0.002},
        ],
        "drawdown_curve": [
            {"date": "2024-01-02", "value": 0.0},
            {"date": "2024-01-03", "value": -0.004},
            {"date": "2024-01-04", "value": -0.018},
            {"date": "2024-01-05", "value": -0.002},
        ],
        "monthly_returns": [
            {"month": "2024-01", "strategy_return": 2.1, "benchmark_return": 0.9},
            {"month": "2024-02", "strategy_return": 1.3, "benchmark_return": 0.7},
        ],
        "top_holdings": [
            {"ticker": "THYAO", "weight": 0.12},
            {"ticker": "GARAN", "weight": 0.10},
            {"ticker": "ASELS", "weight": 0.08},
        ],
    }


def _fake_combine_factors(self: CoreBackendService, **_: Any) -> dict[str, Any]:
    return {
        "backtest": _fake_run_backtest(self),
        "attribution": {"momentum": 0.52, "value": 0.33, "quality": 0.15},
        "factor_correlation": {
            "momentum": {"momentum": 1.0, "value": 0.21, "quality": 0.18},
            "value": {"momentum": 0.21, "value": 1.0, "quality": 0.14},
            "quality": {"momentum": 0.18, "value": 0.14, "quality": 1.0},
        },
    }


def _fake_optimize_strategy(self: CoreBackendService, **_: Any) -> dict[str, Any]:
    return {
        "method": "grid",
        "best_trial": {
            "trial_id": 3,
            "params": {"top_n": 15, "lookback": 63},
            "metrics": {"sharpe": 1.71, "cagr": 0.15},
            "score": 1.71,
            "feasible": True,
        },
        "trials": [
            {
                "trial_id": 1,
                "params": {"top_n": 10, "lookback": 42},
                "metrics": {"sharpe": 1.42, "cagr": 0.11},
                "score": 1.42,
                "feasible": True,
            },
            {
                "trial_id": 2,
                "params": {"top_n": 10, "lookback": 63},
                "metrics": {"sharpe": 1.63, "cagr": 0.14},
                "score": 1.63,
                "feasible": True,
            },
            {
                "trial_id": 3,
                "params": {"top_n": 15, "lookback": 63},
                "metrics": {"sharpe": 1.71, "cagr": 0.15},
                "score": 1.71,
                "feasible": True,
            },
        ],
    }


def _fake_signal_snapshot(_: dict[str, Any]) -> dict[str, Any]:
    return {
        "date": "2024-05-03",
        "scores": [
            {"symbol": "THYAO", "score": 0.82, "rank": 1},
            {"symbol": "GARAN", "score": 0.67, "rank": 2},
            {"symbol": "ASELS", "score": 0.61, "rank": 3},
        ],
    }


def _fake_run_stock_filter(_: dict[str, Any]) -> dict[str, Any]:
    return {
        "meta": {"total_matches": 2, "returned_rows": 2},
        "rows": [
            {
                "symbol": "THYAO",
                "sector": "Transportation",
                "market_cap_usd": 11800,
                "pe": 8.2,
                "pb": 1.12,
                "rsi_14": 61,
                "return_1m": 0.12,
                "return_1y": 0.44,
                "upside_potential": 0.18,
                "recommendation": "AL",
            },
            {
                "symbol": "GARAN",
                "sector": "Banking",
                "market_cap_usd": 9400,
                "pe": 6.4,
                "pb": 0.96,
                "rsi_14": 48,
                "return_1m": 0.04,
                "return_1y": 0.29,
                "upside_potential": 0.10,
                "recommendation": "TUT",
            },
        ],
    }


def _fake_list_signals(self: CoreBackendService) -> list[str]:
    return ["momentum", "value", "quality"]


def _fake_signal_detail(self: CoreBackendService, name: str) -> dict[str, Any] | None:
    catalog = {
        "momentum": {
            "name": "momentum",
            "category": "trend",
            "description": "12-1 momentum signal",
            "parameters": {"lookback": {"type": "int", "default": 252}},
        },
        "value": {
            "name": "value",
            "category": "valuation",
            "description": "Composite value z-score",
            "parameters": {"winsor": {"type": "float", "default": 0.01}},
        },
        "quality": {
            "name": "quality",
            "category": "fundamental",
            "description": "Profitability + balance-sheet quality",
            "parameters": {},
        },
    }
    return catalog.get(name)


def _fake_macro_calendar(self: SystemService, period: str, country: Any, importance: Any) -> dict[str, Any]:
    return {
        "period": period,
        "country": country,
        "importance": importance,
        "events": [
            {
                "date": "2024-05-06",
                "country": "TR",
                "title": "CPI",
                "importance": "high",
            }
        ],
    }


def _fake_system_init(self: SystemService, project_root: Path | str | None = None) -> None:
    # Keep init side-effect free for sandbox-safe artifact generation.
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    self.project_root = root
    self.app_data_dir = PROJECT_ROOT / "data" / "phase0_system_service"


def _fake_system_diag(self: SystemService) -> dict[str, Any]:
    return {
        "status": "ok",
        "python": "3.13",
        "data_dir": "data",
        "services": {"core": True, "regime": True, "realtime": False},
    }


def _build_sparkline_cache() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    return pd.DataFrame(
        {
            "THYAO": [100.0 + i * 0.5 for i in range(len(dates))],
            "GARAN": [50.0 + i * 0.2 for i in range(len(dates))],
            "ASELS": [75.0 + i * 0.35 for i in range(len(dates))],
        },
        index=dates,
    )


def _http_error_sample(exc: HTTPException) -> dict[str, Any]:
    return {"status_code": exc.status_code, "detail": exc.detail}


def _top_level_keys(value: Any) -> list[str]:
    return sorted(list(value.keys())) if isinstance(value, dict) else []


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _build_samples() -> dict[str, Any]:
    # These imports are intentionally local: patched symbols are resolved at call time.
    import bist_quant.engines.signal_construction as signal_engine
    import bist_quant.engines.stock_filter as stock_engine

    sample_rows: dict[str, dict[str, Any]] = {}
    error_rows: dict[str, dict[str, Any]] = {}

    with ExitStack() as stack:
        tmp_store = Path("/tmp/phase0_contract_jobs_tmp.json")
        if tmp_store.exists():
            tmp_store.unlink()
        stack.enter_context(patch.object(api_main, "job_manager", JobManager(max_workers=2, store_path=tmp_store)))

        stack.enter_context(patch.object(api_main, "_load_xu100_frame", _mock_xu100_frame))
        stack.enter_context(patch.object(api_main, "_regime_payload", _mock_regime_payload))
        stack.enter_context(patch.object(api_main, "_macro_payload", _mock_macro_payload))

        stack.enter_context(patch.object(CoreBackendService, "run_backtest", _fake_run_backtest))
        stack.enter_context(patch.object(CoreBackendService, "combine_factors", _fake_combine_factors))
        stack.enter_context(patch.object(CoreBackendService, "optimize_strategy", _fake_optimize_strategy))
        stack.enter_context(patch.object(CoreBackendService, "list_available_signals", _fake_list_signals))
        stack.enter_context(patch.object(CoreBackendService, "get_signal_details", _fake_signal_detail))

        stack.enter_context(patch.object(SystemService, "get_macro_calendar", _fake_macro_calendar))
        stack.enter_context(patch.object(SystemService, "diagnostics_snapshot", _fake_system_diag))
        stack.enter_context(patch.object(SystemService, "__init__", _fake_system_init))

        stack.enter_context(patch.object(signal_engine, "run_signal_snapshot", _fake_signal_snapshot))
        stack.enter_context(patch.object(stock_engine, "run_stock_filter", _fake_run_stock_filter))

        spark_cache = _build_sparkline_cache()
        stock_engine._SCREEN_CACHE["close_df"] = spark_cache

        # Health + meta + dashboard + macro
        sample_rows[_normalize_endpoint("GET", "/api/health/live")] = {
            "status_code": 200,
            "request": None,
            "response": api_main.health_live(),
        }
        sample_rows[_normalize_endpoint("GET", "/api/meta/signals")] = {
            "status_code": 200,
            "request": None,
            "response": api_main.meta_signals(),
        }
        sample_rows[_normalize_endpoint("GET", "/api/meta/system")] = {
            "status_code": 200,
            "request": None,
            "response": api_main.meta_system(),
        }
        sample_rows[_normalize_endpoint("GET", "/api/dashboard/overview")] = {
            "status_code": 200,
            "request": {"query": {"lookback": 30}},
            "response": api_main.dashboard_overview(lookback=30),
        }
        sample_rows[_normalize_endpoint("GET", "/api/dashboard/regime-history")] = {
            "status_code": 200,
            "request": {"query": {"lookback": 30}},
            "response": api_main.dashboard_regime_history(lookback=30),
        }
        sample_rows[_normalize_endpoint("GET", "/api/dashboard/macro")] = {
            "status_code": 200,
            "request": {"query": {"lookback": 30}},
            "response": api_main.dashboard_macro(lookback=30),
        }
        sample_rows[_normalize_endpoint("GET", "/api/macro/calendar")] = {
            "status_code": 200,
            "request": {"query": {"period": "1w", "country": ["TR"], "importance": "high"}},
            "response": api_main.macro_calendar(period="1w", country=["TR"], importance="high"),
        }

        # Factors + backtest
        sample_rows[_normalize_endpoint("GET", "/api/factors/{name}")] = {
            "status_code": 200,
            "request": {"path_params": {"name": "momentum"}},
            "response": factors_detail("momentum"),
        }
        sample_rows[_normalize_endpoint("POST", "/api/factors/snapshot")] = {
            "status_code": 200,
            "request": {
                "body": {
                    "indicators": ["momentum_20d"],
                    "universe": "XU100",
                    "period": "1y",
                    "top_n": 10,
                }
            },
            "response": factors_snapshot(
                FactorSnapshotRequest(
                    indicators=["momentum_20d"],
                    universe="XU100",
                    period="1y",
                    top_n=10,
                )
            ),
        }
        sample_rows[_normalize_endpoint("POST", "/api/factors/combine")] = {
            "status_code": 200,
            "request": {
                "body": {
                    "signals": [
                        {"name": "momentum", "weight": 0.6},
                        {"name": "value", "weight": 0.4},
                    ],
                    "method": "custom",
                    "start_date": "2020-01-01",
                    "end_date": "2020-12-31",
                }
            },
            "response": factors_combine(
                FactorCombineRequest(
                    signals=[
                        {"name": "momentum", "weight": 0.6},
                        {"name": "value", "weight": 0.4},
                    ],
                    method="custom",
                    start_date="2020-01-01",
                    end_date="2020-12-31",
                )
            ),
        }
        sample_rows[_normalize_endpoint("POST", "/api/backtest/run")] = {
            "status_code": 200,
            "request": {
                "body": {
                    "factor_name": "momentum",
                    "start_date": "2020-01-01",
                    "end_date": "2020-12-31",
                    "rebalance_frequency": "monthly",
                    "top_n": 20,
                    "max_position_weight": 0.25,
                }
            },
            "response": api_main.backtest_run(
                api_main.BacktestRequest(
                    factor_name="momentum",
                    start_date="2020-01-01",
                    end_date="2020-12-31",
                    rebalance_frequency="monthly",
                    top_n=20,
                    max_position_weight=0.25,
                ),
                async_job=False,
            ),
        }

        # Jobs
        created_job = api_main.create_job(
            api_main.JobCreateRequest(
                kind="backtest",
                request={
                    "factor_name": "momentum",
                    "start_date": "2020-01-01",
                    "end_date": "2020-12-31",
                    "top_n": 20,
                },
            )
        )
        # Allow short async completion window.
        sleep(0.08)

        sample_rows[_normalize_endpoint("POST", "/api/jobs")] = {
            "status_code": 200,
            "request": {
                "body": {
                    "kind": "backtest",
                    "request": {
                        "factor_name": "momentum",
                        "start_date": "2020-01-01",
                        "end_date": "2020-12-31",
                        "top_n": 20,
                    },
                }
            },
            "response": created_job,
        }

        sample_rows[_normalize_endpoint("GET", "/api/jobs/{job_id}")] = {
            "status_code": 200,
            "request": {"path_params": {"job_id": created_job["id"]}},
            "response": api_main.get_job(created_job["id"]),
        }

        sample_rows[_normalize_endpoint("GET", "/api/jobs")] = {
            "status_code": 200,
            "request": {"query": {"limit": 20}},
            "response": api_main.list_jobs(limit=20),
        }

        sample_rows[_normalize_endpoint("DELETE", "/api/jobs/{job_id}")] = {
            "status_code": 200,
            "request": {"path_params": {"job_id": created_job["id"]}},
            "response": api_main.cancel_job(created_job["id"]),
        }

        sample_rows[_normalize_endpoint("POST", "/api/jobs/{job_id}/retry")] = {
            "status_code": 200,
            "request": {"path_params": {"job_id": created_job["id"]}, "body": {}},
            "response": api_main.retry_job(created_job["id"]),
        }

        # Screener
        sample_rows[_normalize_endpoint("POST", "/api/screener/run")] = {
            "status_code": 200,
            "request": {"body": {"limit": 50, "filters": {"pe": {"min": 0, "max": 20}}}},
            "response": screener_run(ScreenerRunRequest(limit=50, filters={"pe": {"min": 0, "max": 20}})),
        }
        sample_rows[_normalize_endpoint("POST", "/api/screener/sparklines")] = {
            "status_code": 200,
            "request": {"body": {"symbols": ["THYAO", "GARAN"], "points": 30}},
            "response": screener_sparklines({"symbols": ["THYAO", "GARAN"], "points": 30}),
        }

        # Analytics
        analytics_body = {
            "equity_curve": [
                {"date": "2024-01-01", "value": 100.0},
                {"date": "2024-01-02", "value": 101.5},
                {"date": "2024-01-03", "value": 100.8},
                {"date": "2024-01-04", "value": 102.2},
                {"date": "2024-01-05", "value": 103.1},
            ],
            "methods": ["performance", "rolling", "risk", "stress", "transaction_costs"],
        }
        sample_rows[_normalize_endpoint("POST", "/api/analytics/run")] = {
            "status_code": 200,
            "request": {"body": analytics_body},
            "response": analytics_run(AnalyticsRunRequest(**analytics_body)),
        }

        # Optimization
        opt_body = {
            "signal": "momentum",
            "params": {"start_date": "2020-01-01", "end_date": "2020-12-31"},
            "method": "grid",
            "parameter_space": [{"key": "top_n", "type": "int", "min": 5, "max": 20, "step": 5}],
            "max_trials": 10,
        }
        sample_rows[_normalize_endpoint("POST", "/api/optimize/run")] = {
            "status_code": 200,
            "request": {"body": opt_body, "query": {"async_job": False}},
            "response": optimization_run(OptimizationRunRequest(**opt_body), async_job=False),
        }

        # Professional
        greeks_body = {
            "option_type": "call",
            "spot": 100,
            "strike": 100,
            "time_years": 0.5,
            "volatility": 0.3,
            "risk_free_rate": 0.05,
        }
        sample_rows[_normalize_endpoint("POST", "/api/professional/greeks")] = {
            "status_code": 200,
            "request": {"body": greeks_body},
            "response": professional_greeks(GreeksRequest(**greeks_body)),
        }

        stress_body = {
            "portfolio_value": 1_000_000,
            "shocks": [{"factor": "equity", "shock_pct": -20, "beta": 1.0}],
        }
        sample_rows[_normalize_endpoint("POST", "/api/professional/stress")] = {
            "status_code": 200,
            "request": {"body": stress_body},
            "response": professional_stress(StressTestRequest(**stress_body)),
        }

        crypto_body = {
            "pair": "BTC/USDT",
            "side": "long",
            "entry_price": 60000,
            "equity": 10000,
            "risk_pct": 2,
            "leverage": 5,
            "stop_distance_pct": 3,
        }
        sample_rows[_normalize_endpoint("POST", "/api/professional/crypto-sizing")] = {
            "status_code": 200,
            "request": {"body": crypto_body},
            "response": professional_crypto_sizing(CryptoSizingRequest(**crypto_body)),
        }

        # Compliance
        sample_rows[_normalize_endpoint("GET", "/api/compliance/rules")] = {
            "status_code": 200,
            "request": None,
            "response": compliance_rules(),
        }

        compliance_body = {
            "transaction": {
                "id": "tx-1",
                "timestamp": "2026-03-04T10:00:00Z",
                "user_id": "usr-1",
                "order_id": "ord-1",
                "symbol": "THYAO",
                "side": "buy",
                "quantity": 1000,
                "price": 150.0,
                "venue": "BIST",
                "strategy_id": "mom",
            },
            "rules": [],
        }
        sample_rows[_normalize_endpoint("POST", "/api/compliance/check")] = {
            "status_code": 200,
            "request": {"body": compliance_body},
            "response": compliance_check(ComplianceTransactionRequest(**compliance_body)),
        }

        pos_body = {
            "positions": [
                {"symbol": "THYAO", "value": 600000, "limit": 500000},
                {"symbol": "GARAN", "value": 300000, "limit": 450000},
            ]
        }
        sample_rows[_normalize_endpoint("POST", "/api/compliance/position-limits")] = {
            "status_code": 200,
            "request": {"body": pos_body},
            "response": compliance_position_limits(PositionLimitsRequest(**pos_body)),
        }

        # Structured error contract samples for POST /api/jobs
        try:
            api_main.create_job(api_main.JobCreateRequest(kind="nope", request={}))
        except HTTPException as exc:
            error_rows["POST /api/jobs::unsupported_kind"] = _http_error_sample(exc)

        try:
            api_main.create_job(
                api_main.JobCreateRequest(
                    kind="backtest",
                    request={
                        "factor_name": "",
                        "start_date": "2020-01-01",
                        "end_date": "2020-12-31",
                    },
                )
            )
        except HTTPException as exc:
            error_rows["POST /api/jobs::invalid_backtest_payload"] = _http_error_sample(exc)

        try:
            api_main.create_job(
                api_main.JobCreateRequest(
                    kind="optimize",
                    request={
                        "signal": "momentum",
                        "params": {"start_date": "2020-01-01", "end_date": "2020-12-31"},
                        "method": "grid",
                        "parameter_space": [],
                    },
                )
            )
        except HTTPException as exc:
            error_rows["POST /api/jobs::optimize_missing_parameter_space"] = _http_error_sample(exc)

    if tmp_store.exists():
        tmp_store.unlink()

    return {
        "generated_at": _now_iso(),
        "source": "mocked_live_harness",
        "notes": [
            "Samples are generated by direct invocation of FastAPI handlers with deterministic stubs.",
            "Heavy service paths are monkeypatched for reproducibility.",
        ],
        "samples": sample_rows,
        "errors": error_rows,
    }


def _build_endpoint_entries(samples: dict[str, Any]) -> list[dict[str, Any]]:
    app = api_main.create_app()
    openapi = app.openapi()
    paths = openapi.get("paths", {})

    # Build lookup keyed by normalized method/path.
    openapi_lookup: dict[str, tuple[str, str, dict[str, Any]]] = {}
    for openapi_path, path_spec in paths.items():
        for method, op_spec in path_spec.items():
            if method.lower() not in {"get", "post", "delete", "put", "patch"}:
                continue
            key = _normalize_endpoint(method.upper(), openapi_path)
            openapi_lookup[key] = (method.upper(), openapi_path, op_spec)

    consumed: dict[str, dict[str, str]] = {}
    for endpoints in API_FUNCTION_ENDPOINTS.values():
        for endpoint in endpoints:
            consumed[_normalize_endpoint(endpoint["method"], endpoint["path"])] = endpoint
    consumed[_normalize_endpoint("GET", "/api/dashboard/overview")] = {
        "method": "GET",
        "path": "/api/dashboard/overview",
    }

    endpoint_entries: list[dict[str, Any]] = []
    for key, endpoint in sorted(consumed.items()):
        method = endpoint["method"]
        path = endpoint["path"]

        # Match against OpenAPI with normalized path placeholders.
        matched_spec: dict[str, Any] | None = None
        matched_openapi_path: str | None = None
        for o_key, (o_method, o_path, o_spec) in openapi_lookup.items():
            if o_method != method:
                continue
            # Placeholder-insensitive match
            def _p_norm(value: str) -> str:
                out = []
                in_brace = False
                for ch in value:
                    if ch == "{":
                        in_brace = True
                        out.append("{param}")
                        continue
                    if ch == "}":
                        in_brace = False
                        continue
                    if not in_brace:
                        out.append(ch)
                normalized = "".join(out)
                return normalized.replace("{param}{param}", "{param}")

            if _p_norm(o_path) == _p_norm(path):
                matched_spec = o_spec
                matched_openapi_path = o_path
                break

        request_schema = None
        status_codes: list[int] = []
        if matched_spec is not None:
            req_body = matched_spec.get("requestBody", {})
            req_schema = (
                req_body.get("content", {})
                .get("application/json", {})
                .get("schema", {})
            )
            request_schema = req_schema.get("$ref") or req_schema.get("title") or req_schema.get("type")
            status_codes = sorted(
                int(code)
                for code in matched_spec.get("responses", {}).keys()
                if str(code).isdigit()
            )

        sample_key = _normalize_endpoint(method, path)
        sample_row = samples.get("samples", {}).get(sample_key)
        response_keys = _top_level_keys(sample_row.get("response") if sample_row else None)

        error_contract: dict[str, Any] | None = None
        sample_refs: list[str] = []
        if sample_row:
            sample_refs.append(f"samples.{sample_key}")

        if method == "POST" and path == "/api/jobs":
            error_contract = {
                "shape": {
                    "detail": {
                        "code": "string",
                        "detail": "string",
                        "hint": "string",
                        "errors": "list[object] (optional)",
                    }
                },
                "notable_statuses": [400, 422],
            }
            for err_key in (
                "POST /api/jobs::unsupported_kind",
                "POST /api/jobs::invalid_backtest_payload",
                "POST /api/jobs::optimize_missing_parameter_space",
            ):
                if err_key in samples.get("errors", {}):
                    sample_refs.append(f"errors.{err_key}")
                    status_codes.append(int(samples["errors"][err_key]["status_code"]))

        endpoint_entries.append(
            {
                "method": method,
                "path": path,
                "openapi_path": matched_openapi_path,
                "request_schema": request_schema,
                "response_keys": response_keys,
                "status_codes": sorted(set(status_codes)),
                "error_contract": error_contract,
                "sample_refs": sample_refs,
            }
        )

    return endpoint_entries


def _build_manifest(samples_payload: dict[str, Any]) -> dict[str, Any]:
    route_rows: list[dict[str, Any]] = []
    for row in ROUTE_DEFINITIONS:
        endpoints = _endpoint_set_from_calls(row["frontend_calls"])
        route_rows.append(
            {
                "route": row["route"],
                "page": row["page"],
                "streamlit_source": row["streamlit_source"],
                "frontend_calls": row["frontend_calls"],
                "endpoints": endpoints,
                "execution_mode": row["execution_mode"],
                "status": row["status"],
                "notes": row["notes"],
            }
        )

    endpoint_rows = _build_endpoint_entries(samples_payload)

    dto_by_name: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in DTO_MAPPINGS:
        dto_by_name[row["ui_dto"]].append(row)

    manifest = {
        "generated_at": _now_iso(),
        "phase": "Phase 0 Contract Freeze",
        "scope": {
            "route_coverage": "all_next_routes_plus_app_shell",
            "sample_mode": "mocked_live_harness",
            "backend_source_of_truth": "FastAPI",
        },
        "routes": route_rows,
        "shared_surfaces": SHARED_SURFACES,
        "frontend_api_surface": {
            name: rows for name, rows in sorted(API_FUNCTION_ENDPOINTS.items())
        },
        "endpoints": endpoint_rows,
        "dto_mappings": DTO_MAPPINGS,
        "dto_freeze": {
            "frozen_for_wave1": [
                "BacktestUiResult",
                "AnalyticsUiResult",
                "OptimizationUiResult",
                "ComplianceUiResult",
                "ScreenerUiResult",
            ],
            "alias_rules": [
                "comparator -> operator",
                "passed -> status",
            ],
            "unit_annotations": [
                "Screener return_1m is fractional in backend and rendered as percentage in UI.",
                "Backtest/analytics drawdown values remain ratio-scale unless explicitly formatted in UI.",
            ],
            "analytics_canonical_execution": {
                "canonical": "POST /api/analytics/run",
                "non_canonical": "POST /api/jobs (kind=analytics) unless schema parity is guaranteed",
            },
        },
        "drift_register": DRIFT_REGISTER,
        "unknown_field_register": UNKNOWN_FIELD_REGISTER,
        "verification_targets": {
            "must_fail_if": [
                "any_next_route_missing",
                "any_consumed_endpoint_missing",
                "any_dto_mapping_unresolved",
                "unknown_field_register_not_empty",
                "canonical_sample_missing_for_consumed_endpoint",
            ]
        },
    }
    return manifest


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def _build_markdown_doc(manifest: dict[str, Any], samples_payload: dict[str, Any]) -> str:
    route_rows = []
    for row in manifest["routes"]:
        endpoint_text = "<br>".join(
            f"`{item['method']} {item['path']}`" for item in row["endpoints"]
        ) or "-"
        route_rows.append(
            [
                f"`{row['route']}`",
                f"`{row['execution_mode']}`",
                f"`{row['status']}`",
                f"`{Path(row['page']).name}`",
                f"`{Path(row['streamlit_source']).name}`",
                endpoint_text,
            ]
        )

    endpoint_rows = []
    for row in manifest["endpoints"]:
        status_codes = ", ".join(str(code) for code in row.get("status_codes", [])) or "-"
        response_keys = ", ".join(f"`{k}`" for k in row.get("response_keys", [])) or "-"
        endpoint_rows.append(
            [
                f"`{row['method']} {row['path']}`",
                f"`{row.get('request_schema') or '-'}`",
                status_codes,
                response_keys,
                "yes" if row.get("error_contract") else "no",
            ]
        )

    dto_rows = []
    for row in manifest["dto_mappings"]:
        dto_rows.append(
            [
                f"`{row['ui_dto']}`",
                f"`{row['backend_field']}`",
                f"`{row['ui_field']}`",
                row["transform"],
                row["notes"],
            ]
        )

    drift_rows = []
    for row in manifest["drift_register"]:
        drift_rows.append([
            row["surface"],
            f"`{row['classification']}`",
            row["detail"],
            row["resolution_wave"],
        ])

    unknown_rows = manifest.get("unknown_field_register", [])

    sample_index_rows = []
    for key in sorted(samples_payload.get("samples", {}).keys()):
        item = samples_payload["samples"][key]
        sample_index_rows.append([
            f"`{key}`",
            str(item.get("status_code", "-")),
            ", ".join(f"`{k}`" for k in _top_level_keys(item.get("response"))) or "-",
        ])

    error_index_rows = []
    for key in sorted(samples_payload.get("errors", {}).keys()):
        item = samples_payload["errors"][key]
        detail = item.get("detail", {})
        code = detail.get("code") if isinstance(detail, dict) else None
        error_index_rows.append([
            f"`{key}`",
            str(item.get("status_code", "-")),
            f"`{code or '-'}`",
        ])

    route_table = _markdown_table(
        ["Route", "Execution", "Status", "Next Page", "Streamlit Source", "Consumed Endpoints"],
        route_rows,
    )
    endpoint_table = _markdown_table(
        ["Endpoint", "Request Schema", "Status Codes", "Response Keys", "Structured Error"],
        endpoint_rows,
    )
    dto_table = _markdown_table(
        ["UI DTO", "Backend Field(s)", "UI Field", "Transform", "Notes"],
        dto_rows,
    )
    drift_table = _markdown_table(
        ["Surface", "Class", "Detail", "Target Wave"],
        drift_rows,
    )
    sample_table = _markdown_table(["Sample Key", "Status", "Response Keys"], sample_index_rows)
    error_table = _markdown_table(["Error Sample Key", "Status", "detail.code"], error_index_rows)

    if unknown_rows:
        unknown_block = "\n".join(
            f"- `{row.get('field')}` from `{row.get('source')}`: {row.get('reason')}"
            for row in unknown_rows
        )
    else:
        unknown_block = "- None. Unknown-field register is intentionally empty at Phase 0 gate."

    return f"""# Phase 0 Contract Freeze — Streamlit -> Next.js (2026-03-04)

## Summary
- Purpose: freeze frontend↔backend contracts before Wave 1 fixes.
- Scope: all Next.js routes (`/` + 10 product routes) and global `AppShell` API dependency.
- Backend source of truth: FastAPI route implementations and OpenAPI metadata.
- Canonical sample strategy: mocked live harness (deterministic route invocation with patched heavy services).

## Artifacts
- Manifest: [`artifacts/phase0-contract-manifest.json`](./artifacts/phase0-contract-manifest.json)
- Samples: [`artifacts/phase0-canonical-samples.json`](./artifacts/phase0-canonical-samples.json)

## Route Coverage Matrix
{route_table}

### Shared Surface (AppShell)
- `frontend/src/components/shared/app-shell.tsx` -> `GET /api/dashboard/overview` (lookback=30) for regime label hydration.

## Endpoint Contract Matrix
{endpoint_table}

## DTO Adapter Mapping Matrix
{dto_table}

## Canonical Sample Index
{sample_table}

### Structured Error Samples
{error_table}

## Drift Register
{drift_table}

## Unknown-Field Register
{unknown_block}

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
"""


def main() -> None:
    samples_payload = _build_samples()
    manifest_payload = _build_manifest(samples_payload)
    doc_text = _build_markdown_doc(manifest_payload, samples_payload)

    _json_dump(SAMPLES_PATH, samples_payload)
    _json_dump(MANIFEST_PATH, manifest_payload)
    DOC_PATH.write_text(doc_text, encoding="utf-8")

    print(f"wrote {SAMPLES_PATH.relative_to(PROJECT_ROOT)}")
    print(f"wrote {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    print(f"wrote {DOC_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
