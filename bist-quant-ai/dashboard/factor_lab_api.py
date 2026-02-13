#!/usr/bin/env python3
"""
Factor Lab API for Quant AI.

Exposes model-based factor construction/backtest using the production
Models/portfolio_engine pipeline.

Input: JSON via stdin
Output: JSON via stdout
"""

from __future__ import annotations

import copy
import io
import json
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


APP_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_ROOT.parent
MODELS_DIR = PROJECT_ROOT / "Models"

for candidate in (PROJECT_ROOT, MODELS_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from Models.portfolio_engine import DEFAULT_PORTFOLIO_OPTIONS, PortfolioEngine  # noqa: E402


PARAM_SCHEMAS: dict[str, list[dict[str, Any]]] = {
    "momentum": [
        {"key": "lookback", "label": "Lookback Days", "type": "int", "default": 252, "min": 21, "max": 756},
        {"key": "skip", "label": "Skip Days", "type": "int", "default": 21, "min": 0, "max": 252},
        {"key": "vol_lookback", "label": "Vol Lookback", "type": "int", "default": 252, "min": 21, "max": 756},
    ],
    "profitability": [
        {"key": "operating_income_weight", "label": "Op Inc Weight", "type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
        {"key": "gross_profit_weight", "label": "Gross Profit Weight", "type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
    ],
    "value": [
        {"key": "metric_weights.ep", "label": "E/P Weight", "type": "float", "default": 1.0, "min": 0.0, "max": 5.0},
        {"key": "metric_weights.fcfp", "label": "FCF/P Weight", "type": "float", "default": 1.0, "min": 0.0, "max": 5.0},
        {"key": "metric_weights.ocfev", "label": "OCF/EV Weight", "type": "float", "default": 1.0, "min": 0.0, "max": 5.0},
        {"key": "metric_weights.sp", "label": "S/P Weight", "type": "float", "default": 1.0, "min": 0.0, "max": 5.0},
        {"key": "metric_weights.ebitdaev", "label": "EBITDA/EV Weight", "type": "float", "default": 1.0, "min": 0.0, "max": 5.0},
        {
            "key": "enabled_metrics",
            "label": "Enabled Metrics",
            "type": "multi_select",
            "default": ["ep", "fcfp", "ocfev", "sp", "ebitdaev"],
            "options": [
                {"value": "ep", "label": "E/P"},
                {"value": "fcfp", "label": "FCF/P"},
                {"value": "ocfev", "label": "OCF/EV"},
                {"value": "sp", "label": "S/P"},
                {"value": "ebitdaev", "label": "EBITDA/EV"},
            ],
        },
    ],
}


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _parse_payload() -> dict[str, Any]:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Request body must be a JSON object.")
    return parsed


def _as_int(value: Any, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _as_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
        if np.isnan(parsed):
            return default
        return parsed
    except Exception:
        return default


def _as_date(value: Any, default: str) -> pd.Timestamp:
    ts = pd.to_datetime(value if value else default, errors="coerce")
    if pd.isna(ts):
        return pd.Timestamp(default)
    return pd.Timestamp(ts)


def _cross_sectional_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    row_mean = panel.mean(axis=1)
    row_std = panel.std(axis=1).replace(0.0, np.nan)
    out = panel.sub(row_mean, axis=0).div(row_std, axis=0)
    return out.replace([np.inf, -np.inf], np.nan)


def _normalize_factor_entries(raw: Any, available: set[str]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        raw = []

    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip().lower()
        if not name or name not in available:
            continue
        enabled = bool(item.get("enabled", True))
        weight = _as_float(item.get("weight"), 1.0)
        signal_params = item.get("signal_params", {})
        if not isinstance(signal_params, dict):
            signal_params = {}

        if enabled and weight > 0:
            entries.append(
                {
                    "name": name,
                    "weight": float(weight),
                    "signal_params": signal_params,
                }
            )

    if not entries:
        for default_name in ("momentum", "value"):
            if default_name in available:
                entries.append({"name": default_name, "weight": 1.0, "signal_params": {}})
    return entries


def _normalize_weights(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = sum(max(0.0, float(item.get("weight", 0.0))) for item in entries)
    if total <= 0:
        n = len(entries)
        if n == 0:
            return entries
        equal = 1.0 / float(n)
        for item in entries:
            item["weight"] = equal
        return entries

    for item in entries:
        item["weight"] = max(0.0, float(item.get("weight", 0.0))) / total
    return entries


def _extract_current_holdings(holdings_history: list[dict[str, Any]], limit: int = 30) -> list[str]:
    if not holdings_history:
        return []
    frame = pd.DataFrame(holdings_history)
    if frame.empty or "date" not in frame.columns or "ticker" not in frame.columns:
        return []

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"])
    if frame.empty:
        return []

    latest_date = frame["date"].max()
    latest = frame[frame["date"] == latest_date].copy()
    latest = latest[latest["ticker"] != "XAU/TRY"]
    if latest.empty:
        return []

    if "weight" in latest.columns:
        latest = latest.sort_values("weight", ascending=False)
    tickers = latest["ticker"].astype(str).tolist()

    out: list[str] = []
    seen: set[str] = set()
    for ticker in tickers:
        t = ticker.upper().split(".")[0]
        if t and t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= limit:
            break
    return out


def _build_factor_catalog(engine: PortfolioEngine) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in sorted(engine.signal_configs.keys()):
        cfg = engine.signal_configs[name]
        row = {
            "name": name,
            "label": str(name).replace("_", " ").title(),
            "description": cfg.get("description", ""),
            "rebalance_frequency": cfg.get("rebalance_frequency", "quarterly"),
            "timeline": cfg.get("timeline", {}),
            "portfolio_options": cfg.get("portfolio_options", {}),
            "parameter_schema": PARAM_SCHEMAS.get(name, []),
        }
        rows.append(row)
    return rows


def _build_response(payload: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()

    data_dir = PROJECT_ROOT / "data"
    regime_dir = PROJECT_ROOT / "Regime Filter"

    engine = PortfolioEngine(
        data_dir=data_dir,
        regime_model_dir=regime_dir,
        start_date="2014-01-01",
        end_date="2026-12-31",
    )

    with redirect_stdout(io.StringIO()):
        engine.load_all_data()

    factors = _normalize_factor_entries(payload.get("factors"), set(engine.signal_configs.keys()))
    factors = _normalize_weights(factors)

    start_date = _as_date(payload.get("start_date"), "2018-01-01")
    end_date = _as_date(payload.get("end_date"), "2026-12-31")
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    rebalance_frequency = str(payload.get("rebalance_frequency", "monthly")).strip().lower()
    if rebalance_frequency not in {"monthly", "quarterly"}:
        rebalance_frequency = "monthly"

    top_n = _as_int(payload.get("top_n"), 20, minimum=5, maximum=200)

    base_portfolio_options = copy.deepcopy(DEFAULT_PORTFOLIO_OPTIONS)
    overrides = payload.get("portfolio_options", {})
    if isinstance(overrides, dict):
        for key, value in overrides.items():
            if key in base_portfolio_options:
                base_portfolio_options[key] = value
    base_portfolio_options["top_n"] = top_n

    composite: pd.DataFrame | None = None
    used_factors: list[dict[str, Any]] = []
    factor_top_symbols: dict[str, list[dict[str, Any]]] = {}

    for factor in factors:
        name = factor["name"]
        weight = float(factor["weight"])
        cfg = copy.deepcopy(engine.signal_configs.get(name, {}))
        cfg["enabled"] = True

        timeline = cfg.get("timeline", {}) if isinstance(cfg.get("timeline", {}), dict) else {}
        timeline["start_date"] = start_date.date().isoformat()
        timeline["end_date"] = end_date.date().isoformat()
        cfg["timeline"] = timeline

        cfg["signal_params"] = factor.get("signal_params", {})

        with redirect_stdout(io.StringIO()):
            panel, _ = engine._build_signals_for_factor(name, engine.close_df.index, cfg)

        if panel is None or panel.empty:
            continue

        panel = panel.reindex(index=engine.close_df.index, columns=engine.close_df.columns)
        panel_z = _cross_sectional_zscore(panel)
        weighted = panel_z * weight

        if composite is None:
            composite = weighted
        else:
            composite = composite.add(weighted, fill_value=0.0)

        latest = panel.iloc[-1].dropna().sort_values(ascending=False).head(8)
        factor_top_symbols[name] = [
            {"symbol": str(symbol), "score": _safe_float(score)}
            for symbol, score in latest.items()
        ]

        used_factors.append(
            {
                "name": name,
                "weight": round(weight, 4),
                "signal_params": factor.get("signal_params", {}),
            }
        )

    if composite is None or composite.empty or not used_factors:
        raise ValueError("No usable factors were selected. Check your factor list and data availability.")

    composite = composite.replace([np.inf, -np.inf], np.nan)

    with redirect_stdout(io.StringIO()):
        results = engine._run_backtest(
            signals=composite,
            factor_name="factor_lab_custom",
            rebalance_freq=rebalance_frequency,
            start_date=start_date,
            end_date=end_date,
            portfolio_options=base_portfolio_options,
        )

    returns = results["returns"].dropna()
    equity = results["equity"].dropna()

    xu100_curve: list[dict[str, Any]] = []
    beta = None
    if engine.xu100_prices is not None and not returns.empty:
        xu100 = engine.xu100_prices.reindex(returns.index).ffill()
        xu100_returns = xu100.pct_change().fillna(0.0)
        xu100_equity = (1.0 + xu100_returns).cumprod()
        xu100_curve = [
            {"date": idx.date().isoformat(), "value": round(float(val), 6)}
            for idx, val in xu100_equity.items()
        ]

        bench_var = float(xu100_returns.var())
        if bench_var > 0:
            beta = float(returns.cov(xu100_returns) / bench_var)

    equity_curve = [
        {"date": idx.date().isoformat(), "value": round(float(val), 6)}
        for idx, val in equity.items()
    ]

    latest_scores = composite.iloc[-1].dropna().sort_values(ascending=False)
    composite_top = [
        {"symbol": str(symbol), "score": _safe_float(score)}
        for symbol, score in latest_scores.head(top_n).items()
    ]

    as_of = engine.close_df.index.max()
    as_of_iso = as_of.isoformat() if hasattr(as_of, "isoformat") else str(as_of)

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    return {
        "meta": {
            "mode": "factor_lab_backtest",
            "as_of": as_of_iso,
            "start_date": start_date.date().isoformat(),
            "end_date": end_date.date().isoformat(),
            "rebalance_frequency": rebalance_frequency,
            "top_n": top_n,
            "symbols_used": int(composite.shape[1]),
            "rows_used": int(composite.shape[0]),
            "execution_ms": elapsed_ms,
            "factors": used_factors,
        },
        "metrics": {
            "cagr": round(float(results["cagr"]) * 100.0, 2),
            "sharpe": round(float(results["sharpe"]), 3),
            "sortino": round(float(results["sortino"]), 3),
            "max_dd": round(float(results["max_drawdown"]) * 100.0, 2),
            "total_return": round(float(results["total_return"]) * 100.0, 2),
            "win_rate": round(float(results["win_rate"]) * 100.0, 2),
            "beta": None if beta is None else round(float(beta), 3),
            "rebalance_count": int(results.get("rebalance_count", 0)),
            "trade_count": int(results.get("trade_count", 0)),
        },
        "composite_top": composite_top,
        "factor_top_symbols": factor_top_symbols,
        "current_holdings": _extract_current_holdings(results.get("holdings_history", []), limit=top_n),
        "equity_curve": equity_curve,
        "benchmark_curve": xu100_curve,
    }


def _main() -> int:
    try:
        payload = _parse_payload()
        mode = str(payload.get("_mode", "run")).strip().lower()

        if mode == "catalog":
            engine = PortfolioEngine(
                data_dir=PROJECT_ROOT / "data",
                regime_model_dir=PROJECT_ROOT / "Regime Filter",
                start_date="2014-01-01",
                end_date="2026-12-31",
            )
            response = {
                "factors": _build_factor_catalog(engine),
                "default_portfolio_options": DEFAULT_PORTFOLIO_OPTIONS,
            }
        else:
            response = _build_response(payload)

        print(json.dumps(response, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 0


if __name__ == "__main__":
    raise SystemExit(_main())
