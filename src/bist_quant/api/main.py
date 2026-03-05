"""FastAPI application for BIST Quant frontend clients."""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Callable, cast

import pandas as pd

try:
    from fastapi import APIRouter, FastAPI, Query
    from fastapi.exceptions import RequestValidationError
    from fastapi.middleware.cors import CORSMiddleware
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "FastAPI is required for bist_quant.api. Install with `pip install -e .[api]`."
    ) from exc

from bist_quant.common.data_paths import get_data_paths
from bist_quant.api.errors import (
    ApiErrorEnvelope,
    job_not_found_http_exception,
    job_request_missing_http_exception,
    job_validation_http_exception,
    jobs_request_validation_exception_handler,
    retry_unsupported_job_kind_http_exception,
    unsupported_job_kind_http_exception,
)
from bist_quant.api.jobs import JobManager
from bist_quant.api.routers import (
    analytics_router,
    compliance_router,
    factors_router,
    optimization_router,
    professional_router,
    screener_router,
    signal_construction_router,
)
from bist_quant.services import CoreBackendService, SystemService
from bist_quant.services.core_service import DEFAULT_ENGINE_END_DATE, DEFAULT_ENGINE_START_DATE

try:
    from pydantic import BaseModel, Field, ValidationError
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "pydantic is required for bist_quant.api. Install with `pip install -e .[api]`."
    ) from exc


logger = logging.getLogger(__name__)

quant_router = APIRouter(prefix="/api", tags=["bist-quant"])
job_manager = JobManager(max_workers=4, store_path=Path.cwd() / "data" / "api_jobs.json")


class BacktestRequest(BaseModel):
    factor_name: str = Field(..., min_length=1)
    start_date: str = Field(default=DEFAULT_ENGINE_START_DATE)
    end_date: str = Field(default=DEFAULT_ENGINE_END_DATE)
    rebalance_frequency: str = Field(default="monthly")
    top_n: int = Field(default=20, ge=5, le=100)
    max_position_weight: float = Field(default=0.25, ge=0.05, le=1.0)
    use_regime_filter: bool | None = Field(default=True)
    use_liquidity_filter: bool | None = Field(default=True)
    use_slippage: bool | None = Field(default=True)
    slippage_bps: float | None = Field(default=5.0, ge=0, le=1000)
    use_stop_loss: bool | None = Field(default=False)
    stop_loss_threshold: float | None = Field(default=0.15, ge=0.01, le=1.0)
    use_vol_targeting: bool | None = Field(default=False)
    target_downside_vol: float | None = Field(default=0.20, ge=0.01, le=2.0)
    benchmark: str | None = Field(default="XU100")
    signal_lag_days: int | None = Field(default=None, ge=0, le=30)
    use_inverse_vol_sizing: bool | None = Field(default=None)


class JobCreateRequest(BaseModel):
    kind: str
    request: dict[str, Any]


def _validate_job_request_model(model: type[BaseModel], raw: dict[str, Any], *, kind: str) -> BaseModel:
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        raise job_validation_http_exception(
            kind=kind,
            detail=f"Invalid request payload for job kind '{kind}'.",
            errors=exc.errors(),
        ) from exc


def _validate_backtest_job_request(raw: dict[str, Any]) -> BacktestRequest:
    return cast(BacktestRequest, _validate_job_request_model(BacktestRequest, raw, kind="backtest"))


def _validate_factor_combine_job_request(raw: dict[str, Any]) -> BaseModel:
    from bist_quant.api.schemas import FactorCombineRequest

    return _validate_job_request_model(FactorCombineRequest, raw, kind="factor_combine")


def _validate_screener_job_request(raw: dict[str, Any]) -> BaseModel:
    from bist_quant.api.schemas import ScreenerRunRequest

    return _validate_job_request_model(ScreenerRunRequest, raw, kind="screener")


def _validate_analytics_job_request(raw: dict[str, Any]) -> BaseModel:
    from bist_quant.api.schemas import AnalyticsRunRequest

    return _validate_job_request_model(AnalyticsRunRequest, raw, kind="analytics")


def _validate_optimize_job_request(raw: dict[str, Any]) -> BaseModel:
    from bist_quant.api.schemas import OptimizationRunRequest

    opt_req = _validate_job_request_model(OptimizationRunRequest, raw, kind="optimize")
    parameter_space = cast(Any, getattr(opt_req, "parameter_space", None))
    if not parameter_space:
        raise job_validation_http_exception(
            kind="optimize",
            detail="Missing required optimization field 'parameter_space'.",
            hint=(
                "Provide a non-empty 'parameter_space' list. "
                "Each item should include at least: key, type, and bounds/values."
            ),
        )
    return opt_req


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and (pd.isna(value) or pd.isnull(value)):
        return None
    return value


def _iso_date(value: Any) -> str | None:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return str(ts.date())


def _latest_from_series(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.iloc[-1])


def _pct_change(series: pd.Series, periods: int = 1) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < periods + 1:
        return None
    prev = float(clean.iloc[-(periods + 1)])
    curr = float(clean.iloc[-1])
    if prev == 0:
        return None
    return (curr / prev - 1.0) * 100.0


def _load_xu100_frame() -> pd.DataFrame:
    data_paths = get_data_paths()
    xu100_path = data_paths.xu100_prices
    if xu100_path.suffix == ".parquet":
        frame = pd.read_parquet(xu100_path) if xu100_path.exists() else pd.DataFrame()
    else:
        frame = pd.read_csv(xu100_path) if xu100_path.exists() else pd.DataFrame()
    if frame.empty:
        return frame
    date_col = next((c for c in frame.columns if "date" in c.lower()), frame.columns[0])
    close_col = "Close" if "Close" in frame.columns else frame.columns[-1]
    frame["Date"] = pd.to_datetime(frame[date_col], errors="coerce")
    frame["Close"] = pd.to_numeric(frame[close_col], errors="coerce")
    return frame.dropna(subset=["Date", "Close"]).sort_values("Date")


def _load_gold_fx_frame(data_paths: Any) -> pd.DataFrame:
    gold_cache = Path(data_paths.data_dir) / "borsapy_cache" / "gold" / "xau_try_daily.parquet"
    if not gold_cache.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_parquet(gold_cache).reset_index()
    except Exception:
        return pd.DataFrame()
    if frame.empty:
        return frame
    date_col = next((c for c in frame.columns if "date" in c.lower()), frame.columns[0])
    frame["Date"] = pd.to_datetime(frame[date_col], errors="coerce")
    if "USD_TRY" in frame.columns:
        frame["USD_TRY"] = pd.to_numeric(frame["USD_TRY"], errors="coerce")
    if "XAU_TRY" in frame.columns:
        frame["XAU_TRY"] = pd.to_numeric(frame["XAU_TRY"], errors="coerce")
    return frame.dropna(subset=["Date"]).sort_values("Date")


def _regime_payload(data_paths: Any) -> dict[str, Any]:
    try:
        from bist_quant.regime.simple_regime import DataLoader as RegimeDataLoader
        from bist_quant.regime.simple_regime import SimpleRegimeClassifier

        loader = RegimeDataLoader(data_dir=str(data_paths.data_dir))
        xu100 = loader.load_xu100()
        clf = SimpleRegimeClassifier()
        series = clf.classify(xu100["Close"])
        current = clf.get_current_regime() or {}
        distribution = clf.get_distribution()

        series_rows: list[dict[str, str]] = []
        for idx, value in series.items():
            d = _iso_date(idx)
            if d:
                series_rows.append({"date": d, "regime": str(value)})

        dist_rows: list[dict[str, Any]] = []
        if hasattr(distribution, "iterrows"):
            for regime_name, row in distribution.iterrows():
                dist_rows.append(
                    {
                        "regime": str(regime_name),
                        "count": int(row.get("Count", 0)),
                        "percent": float(row.get("Percent", 0.0)),
                    }
                )

        return {
            "label": str(current.get("regime", "Unknown")),
            "current": {
                "date": _iso_date(current.get("date")),
                "above_ma": bool(current.get("above_ma", False)),
                "allocation": _json_safe(current.get("allocation")),
                "realized_vol": _json_safe(current.get("realized_vol")),
                "vol_percentile": _json_safe(current.get("vol_percentile")),
            },
            "series": series_rows,
            "distribution": dist_rows,
        }
    except Exception:
        return {
            "label": "Unknown",
            "current": {},
            "series": [],
            "distribution": [],
        }


def _macro_payload(data_paths: Any, lookback: int) -> dict[str, Any]:
    gold_df = _load_gold_fx_frame(data_paths)
    if gold_df.empty:
        return {"series": {"usdtry": [], "xau_try": []}, "changes": []}

    tail = gold_df.tail(lookback)
    usdtry_series: list[dict[str, Any]] = []
    xau_series: list[dict[str, Any]] = []

    if "USD_TRY" in tail.columns:
        for row in tail[["Date", "USD_TRY"]].itertuples(index=False):
            if pd.notna(row.USD_TRY):
                usdtry_series.append({"date": str(row.Date.date()), "value": float(row.USD_TRY)})
    if "XAU_TRY" in tail.columns:
        for row in tail[["Date", "XAU_TRY"]].itertuples(index=False):
            if pd.notna(row.XAU_TRY):
                xau_series.append({"date": str(row.Date.date()), "value": float(row.XAU_TRY)})

    changes: list[dict[str, Any]] = []
    if "USD_TRY" in gold_df.columns:
        series = gold_df["USD_TRY"].dropna()
        if not series.empty:
            changes.append(
                {
                    "asset": "USD/TRY",
                    "current": float(series.iloc[-1]),
                    "d1_pct": _json_safe(_pct_change(series, 1)),
                    "w1_pct": _json_safe(_pct_change(series, 5)),
                    "m1_pct": _json_safe(_pct_change(series, 21)),
                }
            )
    if "XAU_TRY" in gold_df.columns:
        series = gold_df["XAU_TRY"].dropna()
        if not series.empty:
            changes.append(
                {
                    "asset": "Gold (TRY/oz)",
                    "current": float(series.iloc[-1]),
                    "d1_pct": _json_safe(_pct_change(series, 1)),
                    "w1_pct": _json_safe(_pct_change(series, 5)),
                    "m1_pct": _json_safe(_pct_change(series, 21)),
                }
            )

    return {
        "series": {"usdtry": usdtry_series, "xau_try": xau_series},
        "changes": changes,
    }


@quant_router.get("/health/live")
def health_live() -> dict[str, Any]:
    return {"ok": True, "service": "bist-quant-api"}


@quant_router.get("/health/ready")
def health_ready() -> dict[str, Any]:
    data_paths = get_data_paths()
    return {
        "ok": True,
        "data_dir": str(data_paths.data_dir),
        "prices_path": str(data_paths.prices_file),
        "xu100_prices": str(data_paths.xu100_prices),
    }


@quant_router.get("/meta/signals")
def meta_signals() -> dict[str, Any]:
    core = CoreBackendService(strict_paths=False)
    signals = core.list_available_signals()
    return {"count": len(signals), "signals": signals}


@quant_router.get("/meta/system")
def meta_system() -> dict[str, Any]:
    system = SystemService(project_root=Path.cwd())
    return system.diagnostics_snapshot()


@quant_router.get("/macro/calendar")
def macro_calendar(
    period: str = Query(default="1w"),
    country: list[str] | None = Query(default=None),
    importance: str | None = Query(default=None),
) -> dict[str, Any]:
    system = SystemService(project_root=Path.cwd())
    parsed_country: str | list[str] | None
    if country is None:
        parsed_country = None
    elif len(country) == 1:
        parsed_country = country[0]
    else:
        parsed_country = country
    return system.get_macro_calendar(period=period, country=parsed_country, importance=importance)


@quant_router.get("/dashboard/overview")
def dashboard_overview(lookback: int = Query(default=504, ge=30, le=3000)) -> dict[str, Any]:
    data_paths = get_data_paths()
    prices = _load_xu100_frame()

    if prices.empty:
        return {
            "kpi": {
                "xu100_last": None,
                "xu100_daily_pct": None,
                "usdtry_last": None,
                "usdtry_daily_pct": None,
                "xau_try_last": None,
                "xau_try_daily_pct": None,
            },
            "regime": {"label": "Unknown"},
            "timeline": [],
            "lookback": lookback,
            "error": "XU100 price data not available",
        }

    prices = prices.tail(lookback)

    regime = _regime_payload(data_paths)
    macro = _macro_payload(data_paths, lookback)

    timeline = [
        {"date": str(row.Date.date()), "close": float(row.Close)}
        for row in prices.itertuples(index=False)
    ]

    return {
        "kpi": {
            "xu100_last": _json_safe(_latest_from_series(prices["Close"])),
            "xu100_daily_pct": _json_safe(_pct_change(prices["Close"])),
            "usdtry_last": (
                _json_safe(macro["series"]["usdtry"][-1]["value"])
                if macro["series"]["usdtry"]
                else None
            ),
            "usdtry_daily_pct": (
                _json_safe(macro["changes"][0]["d1_pct"]) if macro.get("changes") else None
            ),
            "xau_try_last": (
                _json_safe(macro["series"]["xau_try"][-1]["value"])
                if macro["series"]["xau_try"]
                else None
            ),
            "xau_try_daily_pct": (
                _json_safe(
                    next(
                        (
                            row["d1_pct"]
                            for row in macro.get("changes", [])
                            if row["asset"] == "Gold (TRY/oz)"
                        ),
                        None,
                    )
                )
            ),
        },
        "regime": regime,
        "timeline": timeline,
        "macro": macro,
        "lookback": lookback,
        "date_range": {
            "start": str(prices["Date"].min().date()),
            "end": str(prices["Date"].max().date()),
        },
        "defaults": {
            "start_date": DEFAULT_ENGINE_START_DATE,
            "end_date": DEFAULT_ENGINE_END_DATE,
        },
    }


@quant_router.get("/dashboard/regime-history")
def dashboard_regime_history(lookback: int = Query(default=504, ge=30, le=3000)) -> dict[str, Any]:
    payload = _regime_payload(get_data_paths())
    series = payload.get("series", [])
    if isinstance(series, list):
        payload["series"] = series[-lookback:]
    return payload


@quant_router.get("/dashboard/macro")
def dashboard_macro(lookback: int = Query(default=252, ge=30, le=3000)) -> dict[str, Any]:
    return _macro_payload(get_data_paths(), lookback)


def _run_backtest_request(req: BacktestRequest) -> dict[str, Any]:
    core = CoreBackendService(strict_paths=False)
    return core.run_backtest(
        factor_name=req.factor_name,
        start_date=req.start_date,
        end_date=req.end_date,
        rebalance_frequency=req.rebalance_frequency,
        top_n=req.top_n,
        max_position_weight=req.max_position_weight,
        use_regime_filter=req.use_regime_filter,
        use_liquidity_filter=req.use_liquidity_filter,
        use_slippage=req.use_slippage,
        slippage_bps=req.slippage_bps,
        use_stop_loss=req.use_stop_loss,
        stop_loss_threshold=req.stop_loss_threshold,
        use_vol_targeting=req.use_vol_targeting,
        target_downside_vol=req.target_downside_vol,
        benchmark=req.benchmark,
        signal_lag_days=req.signal_lag_days,
        use_inverse_vol_sizing=req.use_inverse_vol_sizing,
    )


@quant_router.post("/backtest/run")
def backtest_run(
    payload: BacktestRequest, async_job: bool = Query(default=False)
) -> dict[str, Any]:
    if not async_job:
        return _run_backtest_request(payload)

    record = job_manager.create(
        kind="backtest",
        fn=lambda: _run_backtest_request(payload),
        meta={"factor_name": payload.factor_name},
        request=payload.model_dump(),
    )
    return job_manager.to_dict(record)


VALID_JOB_KINDS = {
    "backtest",
    "factor_combine",
    "screener",
    "analytics",
    "optimize",
}


JOB_KIND_VALIDATORS: dict[str, Callable[[dict[str, Any]], BaseModel]] = {
    "backtest": _validate_backtest_job_request,
    "factor_combine": _validate_factor_combine_job_request,
    "screener": _validate_screener_job_request,
    "analytics": _validate_analytics_job_request,
    "optimize": _validate_optimize_job_request,
}

JOBS_CREATE_RESPONSES = {
    400: {"model": ApiErrorEnvelope, "description": "Unsupported job kind."},
    422: {"model": ApiErrorEnvelope, "description": "Request validation or payload validation error."},
}
JOBS_LIST_RESPONSES = {
    422: {"model": ApiErrorEnvelope, "description": "Request validation error."},
}
JOBS_ITEM_RESPONSES = {
    404: {"model": ApiErrorEnvelope, "description": "Job not found."},
    422: {"model": ApiErrorEnvelope, "description": "Request validation error."},
}
JOBS_RETRY_RESPONSES = {
    400: {"model": ApiErrorEnvelope, "description": "Retry request is invalid for this job."},
    404: {"model": ApiErrorEnvelope, "description": "Job not found."},
    422: {"model": ApiErrorEnvelope, "description": "Request validation or payload validation error."},
}


@quant_router.post("/jobs", responses=JOBS_CREATE_RESPONSES)
def create_job(payload: JobCreateRequest) -> dict[str, Any]:
    if payload.kind not in VALID_JOB_KINDS:
        raise unsupported_job_kind_http_exception(payload.kind, VALID_JOB_KINDS)

    validator = JOB_KIND_VALIDATORS[payload.kind]
    validated_request = validator(payload.request)

    if payload.kind == "backtest":
        req = cast(BacktestRequest, validated_request)
        record = job_manager.create(
            kind="backtest",
            fn=lambda: _run_backtest_request(req),
            meta={"factor_name": req.factor_name},
            request=req.model_dump(),
        )
    elif payload.kind == "factor_combine":
        from bist_quant.api.schemas import FactorCombineRequest

        combine_req = cast(FactorCombineRequest, validated_request)

        def _run_combine() -> dict[str, Any]:
            core = CoreBackendService(strict_paths=False)
            factors = []
            for sig in combine_req.signals:
                spec: dict[str, Any] = {"name": sig.get("name", ""), "weight": sig.get("weight", 1.0)}
                for k, v in sig.items():
                    if k not in ("name", "weight"):
                        spec[k] = v
                factors.append(spec)
            return core.combine_factors(
                factors=factors,
                start_date=combine_req.start_date,
                end_date=combine_req.end_date,
                weighting_scheme=combine_req.method,
                timing_enabled=combine_req.timing_enabled,
                benchmark=combine_req.benchmark,
                rebalance_frequency=combine_req.rebalance_frequency,
                top_n=combine_req.top_n,
                max_position_weight=combine_req.max_position_weight,
            )

        record = job_manager.create(
            kind="factor_combine",
            fn=_run_combine,
            meta={"method": combine_req.method},
            request=combine_req.model_dump(),
        )
    elif payload.kind == "screener":
        from bist_quant.api.schemas import ScreenerRunRequest

        scr_req = cast(ScreenerRunRequest, validated_request)

        def _run_screener() -> dict[str, Any]:
            from bist_quant.engines.stock_filter import run_stock_filter
            return run_stock_filter(scr_req.model_dump())

        record = job_manager.create(
            kind="screener",
            fn=_run_screener,
            meta={"template": scr_req.template or "custom"},
            request=scr_req.model_dump(),
        )
    elif payload.kind == "analytics":
        analytics_req = validated_request
        analytics_payload = analytics_req.model_dump()
        record = job_manager.create(
            kind="analytics",
            fn=lambda: analytics_payload,  # Lightweight; real work is in /api/analytics/run
            meta={},
            request=analytics_payload,
        )
    elif payload.kind == "optimize":
        from bist_quant.api.schemas import OptimizationRunRequest

        opt_req = cast(OptimizationRunRequest, validated_request)

        def _run_optimize() -> dict[str, Any]:
            core = CoreBackendService(strict_paths=False)
            base_request = {"factor_name": opt_req.signal, **opt_req.params}
            return core.optimize_strategy(
                base_request=base_request,
                method=opt_req.method,
                parameter_space=opt_req.parameter_space,
                max_trials=opt_req.max_trials,
                random_seed=opt_req.random_seed,
                train_ratio=opt_req.train_ratio,
                walk_forward_splits=opt_req.walk_forward_splits,
                constraints=opt_req.constraints,
                objective=opt_req.objective,
            )

        record = job_manager.create(
            kind="optimize",
            fn=_run_optimize,
            meta={"signal": opt_req.signal, "method": opt_req.method},
            request=opt_req.model_dump(),
        )
    else:
        raise unsupported_job_kind_http_exception(payload.kind, VALID_JOB_KINDS)

    return job_manager.to_dict(record)


@quant_router.get("/jobs", responses=JOBS_LIST_RESPONSES)
def list_jobs(limit: int = Query(default=20, ge=1, le=200)) -> dict[str, Any]:
    rows = [job_manager.to_dict(item) for item in job_manager.list(limit)]
    return {"count": len(rows), "jobs": rows}


@quant_router.get("/jobs/{job_id}", responses=JOBS_ITEM_RESPONSES)
def get_job(job_id: str) -> dict[str, Any]:
    record = job_manager.get(job_id)
    if record is None:
        raise job_not_found_http_exception(job_id)
    return job_manager.to_dict(record)


@quant_router.delete("/jobs/{job_id}", responses=JOBS_ITEM_RESPONSES)
def cancel_job(job_id: str) -> dict[str, Any]:
    record = job_manager.get(job_id)
    if record is None:
        raise job_not_found_http_exception(job_id)
    cancelled = job_manager.cancel(job_id)
    return {"id": job_id, "cancelled": cancelled}


@quant_router.post("/jobs/{job_id}/retry", responses=JOBS_RETRY_RESPONSES)
def retry_job(job_id: str) -> dict[str, Any]:
    record = job_manager.get(job_id)
    if record is None:
        raise job_not_found_http_exception(job_id)
    if record.kind not in VALID_JOB_KINDS:
        raise retry_unsupported_job_kind_http_exception(record.kind)
    if not isinstance(record.request, dict) or not record.request:
        raise job_request_missing_http_exception(job_id)

    # Re-create the job via the generic create_job path
    retry_payload = JobCreateRequest(kind=record.kind, request=record.request)
    return create_job(retry_payload)


def create_app() -> FastAPI:
    app = FastAPI(title="BIST Quant API", version="0.1.0")
    app.add_exception_handler(RequestValidationError, jobs_request_validation_exception_handler)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(quant_router)
    app.include_router(factors_router)
    app.include_router(screener_router)
    app.include_router(analytics_router)
    app.include_router(optimization_router)
    app.include_router(professional_router)
    app.include_router(compliance_router)
    app.include_router(signal_construction_router)
    return app


app = create_app()
