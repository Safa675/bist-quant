"""Factor lab computation engine."""

from __future__ import annotations

import copy
import io
import hashlib
import json
import logging
import time
import unicodedata
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bist_quant.common.config_manager import DEFAULT_PORTFOLIO_OPTIONS
from bist_quant.engines.errors import (
    QuantEngineDataError,
    QuantEngineError,
    QuantEngineExecutionError,
    QuantEngineValidationError,
)
from bist_quant.engines.types import FactorCatalogResult
from bist_quant.portfolio import PortfolioEngine as _BasePortfolioEngine
from bist_quant.runtime import RuntimePathError, RuntimePaths, resolve_runtime_paths, validate_runtime_paths
from bist_quant.signals.ta_consensus_signals import TAConsensusSignals

LOGGER = logging.getLogger("bist_quant.engines.factor_lab")
RESPONSE_CACHE_TTL_SEC = 600
_RESPONSE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def _resolve_paths(runtime_paths: RuntimePaths | None) -> RuntimePaths:
    resolved = runtime_paths or resolve_runtime_paths()
    validate_runtime_paths(resolved, require_price_data=True)
    return resolved


class PortfolioEngine:
    """Thin wrapper around ``Models.portfolio_engine.PortfolioEngine``.

    The wrapper keeps backward compatibility for callers expecting direct access
    to attributes/methods while giving us a stable integration seam.
    """

    def __init__(self, data_dir: Path, regime_model_dir: Path, start_date: str, end_date: str):
        self._engine = _BasePortfolioEngine(
            data_dir=data_dir,
            regime_model_dir=regime_model_dir,
            start_date=start_date,
            end_date=end_date,
        )
        self.signal_configs = self._engine.signal_configs

    def load_all_data(self) -> None:
        self._engine.load_all_data()

    def run_factor(self, factor_name: str, override_config: dict[str, Any] | None = None) -> Any:
        return self._engine.run_factor(factor_name, override_config=override_config)

    def _build_signals_for_factor(self, factor_name: str, dates: pd.DatetimeIndex, config: dict[str, Any]):
        return self._engine._build_signals_for_factor(factor_name, dates, config)

    def _run_backtest(
        self,
        signals: pd.DataFrame,
        factor_name: str,
        rebalance_freq: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        portfolio_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._engine._run_backtest(
            signals=signals,
            factor_name=factor_name,
            rebalance_freq=rebalance_freq,
            start_date=start_date,
            end_date=end_date,
            portfolio_options=portfolio_options,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)


class FactorLabEngine:
    """Service-layer wrapper used by the Streamlit app.

    Accepts a ``CoreBackendService`` instance and delegates factor/backtest
    operations to it, keeping the app decoupled from low-level engine details.
    """

    def __init__(self, backend_service: Any) -> None:
        self._service = backend_service

    def __getattr__(self, name: str) -> Any:
        return getattr(self._service, name)


PARAM_SCHEMAS: dict[str, list[dict[str, Any]]] = {
    "momentum": [
        {"key": "lookback", "label": "Lookback Days", "type": "int", "default": 252, "min": 21, "max": 756},
        {"key": "skip", "label": "Skip Days", "type": "int", "default": 21, "min": 0, "max": 252},
        {"key": "vol_lookback", "label": "Vol Lookback", "type": "int", "default": 252, "min": 21, "max": 756},
    ],
    "profitability": [
        {
            "key": "operating_income_weight",
            "label": "Op Inc Weight",
            "type": "float",
            "default": 0.5,
            "min": 0.0,
            "max": 1.0,
        },
        {
            "key": "gross_profit_weight",
            "label": "Gross Profit Weight",
            "type": "float",
            "default": 0.5,
            "min": 0.0,
            "max": 1.0,
        },
    ],
    "value": [
        {"key": "metric_weights.ep", "label": "E/P Weight", "type": "float", "default": 1.0, "min": 0.0, "max": 5.0},
        {
            "key": "metric_weights.fcfp",
            "label": "FCF/P Weight",
            "type": "float",
            "default": 1.0,
            "min": 0.0,
            "max": 5.0,
        },
        {
            "key": "metric_weights.ocfev",
            "label": "OCF/EV Weight",
            "type": "float",
            "default": 1.0,
            "min": 0.0,
            "max": 5.0,
        },
        {"key": "metric_weights.sp", "label": "S/P Weight", "type": "float", "default": 1.0, "min": 0.0, "max": 5.0},
        {
            "key": "metric_weights.ebitdaev",
            "label": "EBITDA/EV Weight",
            "type": "float",
            "default": 1.0,
            "min": 0.0,
            "max": 5.0,
        },
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


FIVE_FACTOR_AXIS_FACTORS: dict[str, dict[str, str]] = {
    "five_factor_axis_size": {
        "axis": "size",
        "label": "13-Axis Rotation · Size",
        "description": "Standalone size axis from five-factor rotation (small vs big).",
    },
    "five_factor_axis_value": {
        "axis": "value",
        "label": "13-Axis Rotation · Value",
        "description": "Standalone value-style axis (value level vs value growth).",
    },
    "five_factor_axis_profitability": {
        "axis": "profitability",
        "label": "13-Axis Rotation · Profitability",
        "description": "Standalone profitability-style axis (margin level vs growth).",
    },
    "five_factor_axis_investment": {
        "axis": "investment",
        "label": "13-Axis Rotation · Investment",
        "description": "Standalone investment-style axis (conservative vs reinvestment).",
    },
    "five_factor_axis_momentum": {
        "axis": "momentum",
        "label": "13-Axis Rotation · Momentum",
        "description": "Standalone momentum axis (winner vs loser).",
    },
    "five_factor_axis_risk": {
        "axis": "risk",
        "label": "13-Axis Rotation · Risk",
        "description": "Standalone risk axis (high beta vs low beta).",
    },
    "five_factor_axis_quality": {
        "axis": "quality",
        "label": "13-Axis Rotation · Quality",
        "description": "Standalone quality axis (ROE/ROA/accrual/F-score composite).",
    },
    "five_factor_axis_liquidity": {
        "axis": "liquidity",
        "label": "13-Axis Rotation · Liquidity",
        "description": "Standalone liquidity axis (Amihud/turnover/spread proxy composite).",
    },
    "five_factor_axis_trading_intensity": {
        "axis": "trading_intensity",
        "label": "13-Axis Rotation · Trading Intensity",
        "description": "Standalone trading-intensity axis (relative volume/activity).",
    },
    "five_factor_axis_sentiment": {
        "axis": "sentiment",
        "label": "13-Axis Rotation · Sentiment",
        "description": "Standalone sentiment axis (52w proximity/price action).",
    },
    "five_factor_axis_fundmom": {
        "axis": "fundmom",
        "label": "13-Axis Rotation · Fundamental Momentum",
        "description": "Standalone fundamental-momentum axis (margin/sales acceleration).",
    },
    "five_factor_axis_carry": {
        "axis": "carry",
        "label": "13-Axis Rotation · Carry",
        "description": "Standalone carry axis (dividend/shareholder yield).",
    },
    "five_factor_axis_defensive": {
        "axis": "defensive",
        "label": "13-Axis Rotation · Defensive",
        "description": "Standalone defensive axis (stability and low-beta profile).",
    },
}

EXTERNAL_AXIS_FACTORS: dict[str, dict[str, str]] = {
    "external_consensus": {
        "label": "External Consensus · TradingView TA",
        "description": (
            "TradingView technical-analysis consensus mapped to [-1, +1] from "
            "BUY/SELL/NEUTRAL aggregate votes."
        ),
    },
}


def _normalize_ticker(value: Any) -> str:
    raw = str(value or "").strip().upper()
    if not raw:
        return ""
    return raw.split(".")[0]


def _normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    return "".join(ch for ch in text if not unicodedata.combining(ch)).lower().strip()


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        parsed = float(value)
        if not np.isfinite(parsed):
            return None
        if pd.isna(parsed):
            return None
        return parsed
    except Exception:
        return None


def _to_series(values: pd.Series | None) -> pd.Series:
    if values is None:
        return pd.Series(dtype="float64")
    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.empty:
        return pd.Series(dtype="float64")
    series = series.astype("float64")
    if isinstance(series.index, pd.DatetimeIndex):
        series = series.sort_index()
    return series


def _yearly_metrics(returns: pd.Series) -> list[dict[str, Any]]:
    if returns.empty or not isinstance(returns.index, pd.DatetimeIndex):
        return []

    rows: list[dict[str, Any]] = []
    for year, group in returns.groupby(returns.index.year):
        if group.empty:
            continue
        equity = (1.0 + group).cumprod()
        total_return = equity.iloc[-1] - 1.0
        std = group.std()
        vol = std * np.sqrt(252.0) if std > 0 else np.nan
        sharpe = (group.mean() / std) * np.sqrt(252.0) if std > 0 else np.nan
        drawdown = equity / equity.cummax() - 1.0
        max_dd = drawdown.min() if not drawdown.empty else np.nan
        win_rate = (group > 0).mean() if len(group) > 0 else np.nan

        rows.append(
            {
                "year": int(year),
                "return": _safe_float(total_return * 100.0),
                "volatility": _safe_float(vol * 100.0),
                "sharpe": _safe_float(sharpe),
                "max_dd": _safe_float(max_dd * 100.0),
                "win_rate": _safe_float(win_rate * 100.0),
                "trading_days": int(len(group)),
            }
        )

    return rows


def _monthly_returns(returns: pd.Series) -> list[dict[str, Any]]:
    if returns.empty or not isinstance(returns.index, pd.DatetimeIndex):
        return []

    monthly = (1.0 + returns).resample("ME").prod() - 1.0
    return [
        {
            "year": int(dt.year),
            "month": int(dt.month),
            "return": _safe_float(value * 100.0),
        }
        for dt, value in monthly.items()
    ]


def _drawdown_metrics(equity: pd.Series) -> dict[str, Any]:
    if equity.empty:
        return {
            "current_dd": None,
            "max_dd": None,
            "avg_dd": None,
            "max_duration_days": 0,
        }

    dd = equity / equity.cummax() - 1.0
    negative = dd[dd < 0]

    max_streak = 0
    streak = 0
    for value in dd:
        if value < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    return {
        "current_dd": _safe_float(dd.iloc[-1] * 100.0),
        "max_dd": _safe_float(dd.min() * 100.0),
        "avg_dd": _safe_float(negative.mean() * 100.0) if not negative.empty else 0.0,
        "max_duration_days": int(max_streak),
    }


def _tail_risk_metrics(returns: pd.Series) -> dict[str, Any]:
    if returns.empty:
        return {
            "var_95": None,
            "cvar_95": None,
            "var_99": None,
            "cvar_99": None,
            "best_day": None,
            "worst_day": None,
        }

    var_95 = returns.quantile(0.05)
    var_99 = returns.quantile(0.01)
    cvar_95 = returns[returns <= var_95].mean() if not returns[returns <= var_95].empty else np.nan
    cvar_99 = returns[returns <= var_99].mean() if not returns[returns <= var_99].empty else np.nan

    return {
        "var_95": _safe_float(var_95 * 100.0),
        "cvar_95": _safe_float(cvar_95 * 100.0),
        "var_99": _safe_float(var_99 * 100.0),
        "cvar_99": _safe_float(cvar_99 * 100.0),
        "best_day": _safe_float(returns.max() * 100.0),
        "worst_day": _safe_float(returns.min() * 100.0),
    }


def _benchmark_metrics(returns: pd.Series, benchmark_returns: pd.Series) -> dict[str, Any]:
    if returns.empty or benchmark_returns.empty:
        return {
            "beta": None,
            "correlation": None,
            "tracking_error": None,
            "information_ratio": None,
        }

    aligned = pd.concat([returns.rename("strategy"), benchmark_returns.rename("benchmark")], axis=1).dropna()
    if aligned.empty:
        return {
            "beta": None,
            "correlation": None,
            "tracking_error": None,
            "information_ratio": None,
        }

    strategy = aligned["strategy"]
    bench = aligned["benchmark"]
    correlation = strategy.corr(bench)
    bench_var = bench.var()
    beta = strategy.cov(bench) / bench_var if bench_var > 0 else np.nan

    active = strategy - bench
    active_std = active.std()
    tracking_error = active_std * np.sqrt(252.0) if active_std > 0 else np.nan
    information_ratio = (active.mean() / active_std) * np.sqrt(252.0) if active_std > 0 else np.nan

    return {
        "beta": _safe_float(beta),
        "correlation": _safe_float(correlation),
        "tracking_error": _safe_float(tracking_error * 100.0),
        "information_ratio": _safe_float(information_ratio),
    }


def _turnover_metrics(holdings_history: Any) -> dict[str, Any]:
    if not isinstance(holdings_history, list) or not holdings_history:
        return {
            "avg_positions": None,
            "avg_turnover": None,
            "rebalance_events": 0,
        }

    frame = pd.DataFrame(holdings_history)
    if frame.empty or "date" not in frame.columns or "ticker" not in frame.columns:
        return {
            "avg_positions": None,
            "avg_turnover": None,
            "rebalance_events": 0,
        }

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", "ticker"])
    if frame.empty:
        return {
            "avg_positions": None,
            "avg_turnover": None,
            "rebalance_events": 0,
        }

    frame["ticker"] = frame["ticker"].map(_normalize_ticker)
    frame = frame[frame["ticker"] != ""]
    if frame.empty:
        return {
            "avg_positions": None,
            "avg_turnover": None,
            "rebalance_events": 0,
        }

    has_weight_col = "weight" in frame.columns
    if has_weight_col:
        frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")

    per_day: dict[pd.Timestamp, dict[str, float]] = {}
    for dt, group in frame.groupby(frame["date"].dt.normalize()):
        day = pd.Timestamp(dt)
        tickers = sorted(set(group["ticker"].tolist()))
        if not tickers:
            per_day[day] = {}
            continue

        if has_weight_col:
            weighted = group.dropna(subset=["weight"]).groupby("ticker")["weight"].sum().astype("float64")
            weighted = weighted.replace([np.inf, -np.inf], np.nan).dropna()
            weighted = weighted[weighted > 0]
            total_weight = float(weighted.sum())
            if total_weight > 0:
                weights = {str(t): float(w / total_weight) for t, w in weighted.items()}
            else:
                equal = 1.0 / float(len(tickers))
                weights = {ticker: equal for ticker in tickers}
        else:
            equal = 1.0 / float(len(tickers))
            weights = {ticker: equal for ticker in tickers}

        per_day[day] = weights

    ordered_days = sorted(per_day.keys())
    if not ordered_days:
        return {
            "avg_positions": None,
            "avg_turnover": None,
            "rebalance_events": 0,
        }

    pos_counts = [len(per_day[day]) for day in ordered_days]
    turnovers: list[float] = []

    prev = per_day[ordered_days[0]]
    for day in ordered_days[1:]:
        current = per_day[day]
        if not current and not prev:
            prev = current
            continue
        universe = set(prev.keys()) | set(current.keys())
        one_way_turnover = 0.5 * sum(
            abs(float(current.get(ticker, 0.0)) - float(prev.get(ticker, 0.0))) for ticker in universe
        )
        one_way_turnover = min(max(float(one_way_turnover), 0.0), 1.0)
        turnovers.append(one_way_turnover)
        prev = current

    return {
        "avg_positions": _safe_float(np.mean(pos_counts)) if pos_counts else None,
        "avg_turnover": _safe_float(np.mean(turnovers)) if turnovers else 0.0,
        "rebalance_events": int(len(turnovers)),
    }


def _build_backtest_analytics_v2(
    returns: pd.Series | None,
    equity: pd.Series | None,
    benchmark_returns: pd.Series | None = None,
    holdings_history: Any = None,
) -> dict[str, Any]:
    strategy_returns = _to_series(returns)
    strategy_equity = _to_series(equity)
    benchmark = _to_series(benchmark_returns)

    if strategy_returns.empty or strategy_equity.empty:
        return {
            "summary": {
                "observations": 0,
                "start": None,
                "end": None,
                "positive_days": None,
                "negative_days": None,
                "flat_days": None,
            },
            "yearly": [],
            "monthly": [],
            "drawdown": _drawdown_metrics(pd.Series(dtype="float64")),
            "tail_risk": _tail_risk_metrics(pd.Series(dtype="float64")),
            "benchmark": _benchmark_metrics(pd.Series(dtype="float64"), pd.Series(dtype="float64")),
            "turnover": _turnover_metrics(holdings_history),
        }

    if isinstance(strategy_returns.index, pd.DatetimeIndex):
        start = strategy_returns.index.min().date().isoformat()
        end = strategy_returns.index.max().date().isoformat()
    else:
        start = None
        end = None

    summary = {
        "observations": int(len(strategy_returns)),
        "start": start,
        "end": end,
        "positive_days": int((strategy_returns > 0).sum()),
        "negative_days": int((strategy_returns < 0).sum()),
        "flat_days": int((strategy_returns == 0).sum()),
    }

    return {
        "summary": summary,
        "yearly": _yearly_metrics(strategy_returns),
        "monthly": _monthly_returns(strategy_returns),
        "drawdown": _drawdown_metrics(strategy_equity),
        "tail_risk": _tail_risk_metrics(strategy_returns),
        "benchmark": _benchmark_metrics(strategy_returns, benchmark),
        "turnover": _turnover_metrics(holdings_history),
    }


def _available_factor_names(engine: PortfolioEngine) -> set[str]:
    names = set(engine.signal_configs.keys())
    names.update(FIVE_FACTOR_AXIS_FACTORS.keys())
    names.update(EXTERNAL_AXIS_FACTORS.keys())
    return names


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
        if not np.isfinite(parsed):
            return default
        return parsed
    except Exception:
        return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
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


def _build_external_consensus_panel(
    engine: PortfolioEngine,
    signal_params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    params = signal_params if isinstance(signal_params, dict) else {}
    interval = str(params.get("interval", "1d"))
    batch_size = _as_int(params.get("batch_size"), 20, minimum=1, maximum=500)
    request_sleep_seconds = _as_float(params.get("request_sleep_seconds"), 0.0)
    batch_pause_seconds = _as_float(params.get("batch_pause_seconds"), 0.0)

    raw_fillna = params.get("fillna_value", 0.0)
    fillna_value = None if raw_fillna is None else _as_float(raw_fillna, 0.0)

    builder = TAConsensusSignals(
        batch_size=batch_size,
        request_sleep_seconds=request_sleep_seconds,
        batch_pause_seconds=batch_pause_seconds,
    )
    panel = builder.build_signal_panel(
        symbols=[str(symbol) for symbol in engine.close_df.columns],
        dates=engine.close_df.index,
        interval=interval,
        fillna_value=fillna_value,
    )
    return panel.reindex(index=engine.close_df.index, columns=engine.close_df.columns)


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

        if enabled and np.isfinite(weight) and weight > 0:
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
    cleaned_weights: list[float] = []
    for item in entries:
        try:
            weight = float(item.get("weight", 0.0))
        except Exception:
            weight = 0.0
        if not np.isfinite(weight) or weight < 0.0:
            weight = 0.0
        cleaned_weights.append(weight)

    total = float(sum(cleaned_weights))
    if total <= 0:
        n = len(entries)
        if n == 0:
            return entries
        equal = 1.0 / float(n)
        for item in entries:
            item["weight"] = equal
        return entries

    for item, weight in zip(entries, cleaned_weights):
        item["weight"] = weight / total
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

    out: list[str] = []
    seen: set[str] = set()
    for raw in latest["ticker"].astype(str).tolist():
        ticker = _normalize_ticker(raw)
        if ticker and ticker not in seen:
            seen.add(ticker)
            out.append(ticker)
        if len(out) >= limit:
            break
    return out


def _cache_key(payload: dict[str, Any]) -> str:
    sanitized = dict(payload)
    sanitized.pop("run_id", None)
    sanitized.pop("trace_id", None)
    sanitized.pop("refresh_cache", None)
    sanitized.pop("_refresh_cache", None)
    encoded = json.dumps(sanitized, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _get_cached_response(key: str) -> dict[str, Any] | None:
    cached = _RESPONSE_CACHE.get(key)
    if not cached:
        return None
    ts, payload = cached
    if time.time() - ts > RESPONSE_CACHE_TTL_SEC:
        _RESPONSE_CACHE.pop(key, None)
        return None
    return copy.deepcopy(payload)


def _store_cached_response(key: str, payload: dict[str, Any]) -> None:
    _RESPONSE_CACHE[key] = (time.time(), copy.deepcopy(payload))


def _build_factor_catalog(engine: PortfolioEngine) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in sorted(engine.signal_configs.keys()):
        cfg = engine.signal_configs[name]
        rows.append(
            {
                "name": name,
                "label": str(name).replace("_", " ").title(),
                "description": cfg.get("description", ""),
                "rebalance_frequency": cfg.get("rebalance_frequency", "quarterly"),
                "timeline": cfg.get("timeline", {}),
                "portfolio_options": cfg.get("portfolio_options", {}),
                "parameter_schema": PARAM_SCHEMAS.get(name, []),
            }
        )

    base_five_factor_cfg = engine.signal_configs.get("five_factor_rotation", {})
    for virtual_name, spec in FIVE_FACTOR_AXIS_FACTORS.items():
        rows.append(
            {
                "name": virtual_name,
                "label": spec["label"],
                "description": spec["description"],
                "rebalance_frequency": base_five_factor_cfg.get("rebalance_frequency", "monthly"),
                "timeline": base_five_factor_cfg.get("timeline", {}),
                "portfolio_options": base_five_factor_cfg.get("portfolio_options", {}),
                "parameter_schema": [],
            }
        )

    for virtual_name, spec in EXTERNAL_AXIS_FACTORS.items():
        rows.append(
            {
                "name": virtual_name,
                "label": spec["label"],
                "description": spec["description"],
                "rebalance_frequency": "monthly",
                "timeline": {},
                "portfolio_options": {},
                "parameter_schema": [
                    {"key": "interval", "label": "TA Interval", "type": "text", "default": "1d"},
                    {"key": "batch_size", "label": "Batch Size", "type": "int", "default": 20, "min": 1, "max": 500},
                    {
                        "key": "request_sleep_seconds",
                        "label": "Request Sleep (sec)",
                        "type": "float",
                        "default": 0.0,
                        "min": 0.0,
                        "max": 10.0,
                    },
                    {
                        "key": "batch_pause_seconds",
                        "label": "Batch Pause (sec)",
                        "type": "float",
                        "default": 0.0,
                        "min": 0.0,
                        "max": 60.0,
                    },
                ],
            }
        )

    rows.sort(key=lambda row: str(row.get("name", "")))
    return rows


def _build_response(
    payload: dict[str, Any],
    *,
    runtime_paths: RuntimePaths | None = None,
) -> dict[str, Any]:
    paths = _resolve_paths(runtime_paths)
    data_dir = paths.data_dir
    regime_dir = paths.regime_outputs_dir
    refresh_requested = _as_bool(payload.get("refresh_cache"), False) or _as_bool(payload.get("_refresh_cache"), False)

    cache_paths = [
        data_dir / "bist_prices_full.parquet",
        data_dir / "bist_prices_full.csv",
        data_dir / "bist_prices_full.csv.gz",
        data_dir / "fundamental_data_consolidated.parquet",
        data_dir / "fundamental_data_consolidated.csv",
        data_dir / "fundamental_data_consolidated.csv.gz",
        data_dir / "shares_outstanding_consolidated.parquet",
        data_dir / "shares_outstanding_consolidated.csv",
        data_dir / "shares_outstanding_consolidated.csv.gz",
        data_dir / "bist_sector_classification.parquet",
        data_dir / "bist_sector_classification.csv",
    ]
    cache_rows: list[str] = []
    for cache_path in cache_paths:
        try:
            stat = cache_path.stat()
            cache_rows.append(f"{cache_path}:{stat.st_mtime_ns}:{stat.st_size}")
        except FileNotFoundError:
            cache_rows.append(f"{cache_path}:missing")
        except OSError:
            cache_rows.append(f"{cache_path}:error")
    cache_state_token = hashlib.sha256("\n".join(cache_rows).encode("utf-8")).hexdigest()

    cache_key = _cache_key(
        {
            **payload,
            "_runtime_data_dir": str(data_dir),
            "_runtime_regime_dir": str(regime_dir),
            "_cache_state_token": cache_state_token,
        }
    )
    cached_response = None if refresh_requested else _get_cached_response(cache_key)
    if cached_response is not None:
        meta = cached_response.get("meta") if isinstance(cached_response.get("meta"), dict) else {}
        meta["cache"] = "hit"
        cached_response["meta"] = meta
        return cached_response

    started = time.perf_counter()

    engine = PortfolioEngine(
        data_dir=data_dir,
        regime_model_dir=regime_dir,
        start_date="2014-01-01",
        end_date="2026-12-31",
    )

    with redirect_stdout(io.StringIO()):
        engine.load_all_data()

    factors = _normalize_factor_entries(payload.get("factors"), _available_factor_names(engine))
    factors = _normalize_weights(factors)

    start_date = _as_date(payload.get("start_date"), "2018-01-01")
    end_date = _as_date(payload.get("end_date"), "2026-12-31")
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    rebalance_frequency = str(payload.get("rebalance_frequency", "monthly")).strip().lower()
    if rebalance_frequency not in {"monthly", "quarterly"}:
        raise ValueError("rebalance_frequency must be 'monthly' or 'quarterly'.")

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
    factor_time_series: dict[str, pd.Series] = {}
    five_factor_bundle: dict[str, Any] | None = None

    def _get_five_factor_bundle(signal_params: dict[str, Any] | None = None) -> dict[str, Any]:
        nonlocal five_factor_bundle
        if five_factor_bundle is not None:
            return five_factor_bundle

        cfg = copy.deepcopy(engine.signal_configs.get("five_factor_rotation", {}))
        if not cfg:
            raise ValueError("five_factor_rotation config is required for axis factors.")

        cfg["enabled"] = True
        timeline_cfg = cfg.get("timeline", {}) if isinstance(cfg.get("timeline", {}), dict) else {}
        timeline_cfg["start_date"] = start_date.date().isoformat()
        timeline_cfg["end_date"] = end_date.date().isoformat()
        cfg["timeline"] = timeline_cfg
        if isinstance(signal_params, dict):
            cfg["signal_params"] = signal_params

        with redirect_stdout(io.StringIO()):
            panel, details = engine._build_signals_for_factor("five_factor_rotation", engine.close_df.index, cfg)

        if panel is None or panel.empty:
            raise ValueError("five_factor_rotation returned an empty signal panel.")

        axis_components: dict[str, pd.DataFrame] = {}
        if isinstance(details, dict):
            raw_components = details.get("axis_components", {})
            if isinstance(raw_components, dict):
                for axis_name, axis_panel in raw_components.items():
                    if isinstance(axis_panel, pd.DataFrame) and not axis_panel.empty:
                        axis_components[str(axis_name)] = axis_panel.reindex(
                            index=engine.close_df.index,
                            columns=engine.close_df.columns,
                        )

        five_factor_bundle = {
            "panel": panel.reindex(index=engine.close_df.index, columns=engine.close_df.columns),
            "axis_components": axis_components,
        }
        return five_factor_bundle

    for factor in factors:
        name = factor["name"]
        weight = float(factor["weight"])
        signal_params = factor.get("signal_params", {})

        if name in FIVE_FACTOR_AXIS_FACTORS:
            axis_name = FIVE_FACTOR_AXIS_FACTORS[name]["axis"]
            bundle = _get_five_factor_bundle()
            axis_panel = bundle.get("axis_components", {}).get(axis_name)
            panel = axis_panel if isinstance(axis_panel, pd.DataFrame) else None
        elif name in EXTERNAL_AXIS_FACTORS:
            panel = _build_external_consensus_panel(
                engine=engine,
                signal_params=signal_params if isinstance(signal_params, dict) else {},
            )
        elif name == "five_factor_rotation":
            panel = _get_five_factor_bundle(signal_params if isinstance(signal_params, dict) else {}).get("panel")
        else:
            cfg = copy.deepcopy(engine.signal_configs.get(name, {}))
            cfg["enabled"] = True

            timeline = cfg.get("timeline", {}) if isinstance(cfg.get("timeline", {}), dict) else {}
            timeline["start_date"] = start_date.date().isoformat()
            timeline["end_date"] = end_date.date().isoformat()
            cfg["timeline"] = timeline

            cfg["signal_params"] = signal_params if isinstance(signal_params, dict) else {}

            with redirect_stdout(io.StringIO()):
                panel, _ = engine._build_signals_for_factor(name, engine.close_df.index, cfg)

        if panel is None or panel.empty:
            LOGGER.warning("factor_lab.empty_panel", extra={"factor": name})
            continue

        panel = panel.reindex(index=engine.close_df.index, columns=engine.close_df.columns)
        panel_z = _cross_sectional_zscore(panel)
        factor_time_series[name] = (
            panel_z.mean(axis=1, skipna=True).replace([np.inf, -np.inf], np.nan).astype("float64")
        )
        weighted = panel_z * weight

        if composite is None:
            composite = weighted
        else:
            composite = composite.add(weighted, fill_value=0.0)

        latest = panel.iloc[-1].dropna().sort_values(ascending=False).head(8)
        factor_top_symbols[name] = [
            {"symbol": str(symbol), "score": _safe_float(score)} for symbol, score in latest.items()
        ]

        used_factors.append(
            {
                "name": name,
                "weight": round(weight, 4),
                "signal_params": signal_params if isinstance(signal_params, dict) else {},
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

    returns = results.get("returns", pd.Series(dtype="float64"))
    equity = results.get("equity", pd.Series(dtype="float64"))
    if not isinstance(returns, pd.Series) or not isinstance(equity, pd.Series):
        raise RuntimeError("Factor backtest returned malformed result payload.")

    returns = returns.dropna().astype("float64")
    equity = equity.dropna().astype("float64")

    xu100_curve: list[dict[str, Any]] = []
    xu100_returns = pd.Series(dtype="float64")
    beta = None

    if engine.xu100_prices is not None and not returns.empty:
        xu100 = engine.xu100_prices.reindex(returns.index).ffill()
        xu100_returns = xu100.pct_change().fillna(0.0)
        xu100_equity = (1.0 + xu100_returns).cumprod()
        xu100_curve = [
            {"date": idx.date().isoformat(), "value": round(float(val), 6)} for idx, val in xu100_equity.items()
        ]

        bench_var = float(xu100_returns.var())
        if bench_var > 0:
            beta = float(returns.cov(xu100_returns) / bench_var)

    equity_curve = [{"date": idx.date().isoformat(), "value": round(float(val), 6)} for idx, val in equity.items()]

    latest_scores = composite.iloc[-1].dropna().sort_values(ascending=False)
    composite_top = [
        {"symbol": str(symbol), "score": _safe_float(score)} for symbol, score in latest_scores.head(top_n).items()
    ]

    as_of = engine.close_df.index.max()
    as_of_iso = as_of.isoformat() if hasattr(as_of, "isoformat") else str(as_of)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    analytics_v2 = _build_backtest_analytics_v2(
        returns=returns,
        equity=equity,
        benchmark_returns=xu100_returns,
        holdings_history=results.get("holdings_history", []),
    )

    factor_correlation: dict[str, dict[str, float | None]] = {}
    if factor_time_series:
        corr_input = pd.DataFrame(factor_time_series).dropna(how="all")
        if not corr_input.empty:
            corr_matrix = corr_input.corr()
            for row_name in corr_matrix.index:
                row: dict[str, float | None] = {}
                for col_name in corr_matrix.columns:
                    value = corr_matrix.at[row_name, col_name]
                    if pd.isna(value):
                        row[str(col_name)] = None
                    else:
                        row[str(col_name)] = round(float(value), 4)
                factor_correlation[str(row_name)] = row

    response = {
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
            "cache": "miss",
            "factors": used_factors,
        },
        "metrics": {
            "cagr": round(float(results.get("cagr", 0.0)) * 100.0, 2),
            "sharpe": round(float(results.get("sharpe", 0.0)), 3),
            "sortino": round(float(results.get("sortino", 0.0)), 3),
            "max_dd": round(float(results.get("max_drawdown", 0.0)) * 100.0, 2),
            "total_return": round(float(results.get("total_return", 0.0)) * 100.0, 2),
            "win_rate": round(float(results.get("win_rate", 0.0)) * 100.0, 2),
            "beta": None if beta is None else round(float(beta), 3),
            "rebalance_count": int(results.get("rebalance_count", 0)),
            "trade_count": int(results.get("trade_count", 0)),
        },
        "composite_top": composite_top,
        "factor_top_symbols": factor_top_symbols,
        "current_holdings": _extract_current_holdings(results.get("holdings_history", []), limit=top_n),
        "equity_curve": equity_curve,
        "benchmark_curve": xu100_curve,
        "factor_correlation": factor_correlation,
        "analytics_v2": analytics_v2,
    }
    _store_cached_response(cache_key, response)
    return response


def build_factor_catalog(
    *,
    runtime_paths: RuntimePaths | None = None,
) -> FactorCatalogResult:
    try:
        paths = _resolve_paths(runtime_paths)
        engine = PortfolioEngine(
            data_dir=paths.data_dir,
            regime_model_dir=paths.regime_outputs_dir,
            start_date="2014-01-01",
            end_date="2026-12-31",
        )
        payload: FactorCatalogResult = {
            "factors": _build_factor_catalog(engine),
            "default_portfolio_options": DEFAULT_PORTFOLIO_OPTIONS,
        }
        if not isinstance(payload["factors"], list) or not isinstance(
            payload["default_portfolio_options"], dict
        ):
            raise QuantEngineExecutionError("Malformed factor catalog payload.")
        return payload
    except QuantEngineError:
        raise
    except RuntimePathError as exc:
        raise QuantEngineDataError(
            str(exc),
            user_message=(
                "Price data is not available. "
                "Place bist_prices_full.csv (or .parquet / .csv.gz) in your data directory, "
                "or set the BIST_DATA_DIR environment variable to a directory that contains it."
            ),
        ) from exc
    except FileNotFoundError as exc:
        raise QuantEngineDataError(str(exc)) from exc
    except ValueError as exc:
        raise QuantEngineValidationError(str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("factor_lab.catalog_failed")
        raise QuantEngineExecutionError(
            "Factor catalog generation failed.",
            user_message="Factor catalog generation failed.",
        ) from exc


def run_factor_lab(
    payload: dict[str, Any],
    *,
    runtime_paths: RuntimePaths | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise QuantEngineValidationError("Request payload must be a JSON object.")
    try:
        response = _build_response(payload, runtime_paths=runtime_paths)
        required_keys = {
            "meta",
            "metrics",
            "composite_top",
            "equity_curve",
            "benchmark_curve",
            "analytics_v2",
        }
        missing = sorted(required_keys.difference(response.keys()))
        if missing:
            raise QuantEngineExecutionError(
                f"Factor lab result is missing required keys: {', '.join(missing)}."
            )
        return response
    except QuantEngineError:
        raise
    except RuntimePathError as exc:
        raise QuantEngineDataError(
            str(exc),
            user_message=(
                "Price data is not available. "
                "Place bist_prices_full.csv (or .parquet / .csv.gz) in your data directory, "
                "or set the BIST_DATA_DIR environment variable to a directory that contains it."
            ),
        ) from exc
    except FileNotFoundError as exc:
        raise QuantEngineDataError(str(exc)) from exc
    except ValueError as exc:
        raise QuantEngineValidationError(str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("factor_lab.run_failed")
        raise QuantEngineExecutionError(
            "Factor lab execution failed.",
            user_message="Factor lab execution failed.",
        ) from exc
