"""
Unified BIST Engine API for Vercel

Combines Factor Lab and Signal Construction into a single Python serverless function.
This is the main entry point for all Python-based API calls.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Setup paths
ROOT = Path(__file__).resolve().parent.parent
FACTOR_MODULE_PATH = ROOT / "dashboard" / "factor_lab_api.py"
SIGNAL_MODULE_PATH = ROOT / "dashboard" / "signal_construction_api.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(name: str, path: Path):
    """Dynamically load a Python module from file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Lazy load modules to reduce cold start time
_factor_lab_api = None
_signal_construction_api = None


def get_factor_lab_api():
    global _factor_lab_api
    if _factor_lab_api is None:
        _factor_lab_api = _load_module("factor_lab_api", FACTOR_MODULE_PATH)
    return _factor_lab_api


def get_signal_construction_api():
    global _signal_construction_api
    if _signal_construction_api is None:
        _signal_construction_api = _load_module("signal_construction_api", SIGNAL_MODULE_PATH)
    return _signal_construction_api


# Create FastAPI app
app = FastAPI(title="BIST Quant Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


def _factor_catalog() -> dict[str, Any]:
    """Build factor catalog from the engine."""
    factor_lab_api = get_factor_lab_api()
    with redirect_stdout(io.StringIO()):
        engine = factor_lab_api.PortfolioEngine(
            data_dir=ROOT / "data",
            regime_model_dir=ROOT / "Regime Filter",
            start_date="2014-01-01",
            end_date="2026-12-31",
        )

    return {
        "factors": factor_lab_api._build_factor_catalog(engine),
        "default_portfolio_options": factor_lab_api.DEFAULT_PORTFOLIO_OPTIONS,
    }


# ====================
# Health & Status
# ====================

@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "service": "bist-quant-engine",
        "version": "1.0.0",
        "endpoints": [
            "/api/health",
            "/api/factor_lab",
            "/api/signal_construction",
        ]
    })


@app.get("/api/health")
async def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "service": "bist-quant-engine",
        "python_version": sys.version,
    })


# ====================
# Factor Lab API
# ====================

@app.get("/api/factor_lab")
async def get_factor_catalog() -> JSONResponse:
    """Get available factors and portfolio options."""
    try:
        return JSONResponse(_factor_catalog())
    except Exception as exc:
        return _error(str(exc), status_code=500)


@app.post("/api/factor_lab")
async def post_factor_run(request: Request) -> JSONResponse:
    """Run factor analysis or backtest."""
    try:
        payload: Any = await request.json()
    except Exception:
        return _error("Request body must be valid JSON.")

    if not isinstance(payload, dict):
        return _error("Request body must be a JSON object.")

    mode = str(payload.get("_mode", "run")).strip().lower()

    # Block heavy computation on Vercel
    factor_key = payload.get("factor_key", "")
    if factor_key == "five_factor_rotation":
        return _error(
            "five_factor_rotation is too computationally intensive for serverless. "
            "Use a different factor or deploy to a dedicated server.",
            status_code=400
        )

    try:
        factor_lab_api = get_factor_lab_api()
        if mode == "catalog":
            return JSONResponse(_factor_catalog())

        with redirect_stdout(io.StringIO()):
            response = factor_lab_api._build_response(payload)
        return JSONResponse(response)
    except ValueError as exc:
        return _error(str(exc), status_code=400)
    except Exception as exc:
        return _error(str(exc), status_code=500)


# ====================
# Signal Construction API
# ====================

@app.get("/api/signal_construction")
async def get_signal_metadata() -> JSONResponse:
    """Get available indicators and options for signal construction."""
    return JSONResponse({
        "universes": ["XU030", "XU100", "XUTUM", "CUSTOM"],
        "periods": ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        "intervals": ["1d"],
        "indicators": [
            {"key": "rsi", "label": "RSI"},
            {"key": "macd", "label": "MACD Histogram"},
            {"key": "bollinger", "label": "Bollinger %B"},
            {"key": "atr", "label": "ATR (Cross-Sectional)"},
            {"key": "stochastic", "label": "Stochastic %K"},
            {"key": "adx", "label": "ADX (+DI/-DI trend)"},
            {"key": "supertrend", "label": "Supertrend Direction"},
        ],
    })


@app.post("/api/signal_construction")
async def post_signal_run(request: Request) -> JSONResponse:
    """Construct signals or run backtest."""
    try:
        payload: Any = await request.json()
    except Exception:
        return _error("Request body must be valid JSON.")

    if not isinstance(payload, dict):
        return _error("Request body must be a JSON object.")

    mode = str(payload.get("_mode", "construct")).strip().lower()

    try:
        signal_construction_api = get_signal_construction_api()
        if mode == "backtest":
            with redirect_stdout(io.StringIO()):
                response = signal_construction_api._build_backtest_response(payload)
        else:
            with redirect_stdout(io.StringIO()):
                response = signal_construction_api._build_response(payload)

        return JSONResponse(response)
    except ValueError as exc:
        return _error(str(exc), status_code=400)
    except Exception as exc:
        return _error(str(exc), status_code=500)
