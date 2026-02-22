"""Signal construction computation engine."""

from __future__ import annotations

import io
import json
import hashlib
import logging
import os
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bist_quant.engines.errors import (
    QuantEngineDataError,
    QuantEngineError,
    QuantEngineExecutionError,
    QuantEngineValidationError,
)
from bist_quant.engines.types import SignalBacktestResult, SignalConstructionSnapshotResult
from bist_quant.runtime import RuntimePathError, RuntimePaths, resolve_runtime_paths, validate_runtime_paths

LOGGER = logging.getLogger("bist_quant.engines.signal_construction")

try:  # noqa: E402
    import borsapy as bp
except Exception:  # pragma: no cover - optional dependency fallback
    bp = None

from bist_quant.common.data_loader import DataLoader  # noqa: E402
from bist_quant.signals.borsapy_indicators import BorsapyIndicators  # noqa: E402
from bist_quant.signals.ta_consensus_signals import TAConsensusSignals  # noqa: E402


DEFAULT_INDICATORS: dict[str, dict[str, Any]] = {
    "rsi": {"period": 14, "oversold": 30.0, "overbought": 70.0},
    "macd": {"fast": 12, "slow": 26, "signal": 9, "threshold": 0.0},
    "bollinger": {"period": 20, "std_dev": 2.0, "lower": 0.2, "upper": 0.8},
    "atr": {"period": 14, "lower_pct": 0.3, "upper_pct": 0.7},
    "stochastic": {"k_period": 14, "d_period": 3, "oversold": 20.0, "overbought": 80.0},
    "adx": {"period": 14, "trend_threshold": 25.0},
    "supertrend": {"period": 10, "multiplier": 3.0},
    "ta_consensus": {
        "interval": "1d",
        "batch_size": 20,
        "request_sleep_seconds": 0.0,
        "batch_pause_seconds": 0.0,
    },
}


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except Exception:
        return default
    return max(minimum, value)


DEFAULT_CACHE_DIR = (
    Path("/tmp/bist-quant-signal-cache")
    if os.environ.get("VERCEL")
    else Path.cwd() / ".cache" / "bist_quant" / "signal_construction"
)
CACHE_DIR = Path(os.environ.get("SIGNAL_CACHE_DIR", str(DEFAULT_CACHE_DIR)))
PRICE_CACHE_TTL_SEC = _env_int("SIGNAL_PRICE_CACHE_TTL_SEC", 900, minimum=0)
INDEX_CACHE_TTL_SEC = _env_int("SIGNAL_INDEX_CACHE_TTL_SEC", 21600, minimum=0)
DOWNLOAD_BATCH_SIZE = _env_int("SIGNAL_DOWNLOAD_BATCH_SIZE", 25, minimum=1)


def _resolve_paths(runtime_paths: RuntimePaths | None) -> RuntimePaths:
    resolved = runtime_paths or resolve_runtime_paths()
    validate_runtime_paths(resolved, require_price_data=True)
    return resolved


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


def _build_backtest_analytics_v2(
    returns: pd.Series | None,
    equity: pd.Series | None,
    benchmark_returns: pd.Series | None = None,
    holdings_history: Any = None,
) -> dict[str, Any]:
    del holdings_history
    strategy_returns = _to_series(returns)
    strategy_equity = _to_series(equity)
    benchmark = _to_series(benchmark_returns)

    if strategy_returns.empty or strategy_equity.empty:
        return {
            "summary": {"observations": 0, "start": None, "end": None},
            "yearly": [],
            "monthly": [],
            "drawdown": {"current_dd": None, "max_dd": None},
            "tail_risk": {"var_95": None, "cvar_95": None},
            "benchmark": {"beta": None, "correlation": None},
            "turnover": {"avg_positions": None, "avg_turnover": None, "rebalance_events": 0},
        }

    yearly: list[dict[str, Any]] = []
    if isinstance(strategy_returns.index, pd.DatetimeIndex):
        for year, group in strategy_returns.groupby(strategy_returns.index.year):
            if group.empty:
                continue
            eq = (1.0 + group).cumprod()
            dd = eq / eq.cummax() - 1.0
            std = group.std()
            sharpe = (group.mean() / std) * np.sqrt(252.0) if std and std > 0 else 0.0
            yearly.append(
                {
                    "year": int(year),
                    "return": _safe_float((eq.iloc[-1] - 1.0) * 100.0),
                    "sharpe": _safe_float(sharpe),
                    "max_dd": _safe_float(dd.min() * 100.0),
                }
            )

    monthly_rows: list[dict[str, Any]] = []
    if isinstance(strategy_returns.index, pd.DatetimeIndex):
        monthly = (1.0 + strategy_returns).resample("ME").prod() - 1.0
        for dt, value in monthly.items():
            monthly_rows.append(
                {
                    "year": int(dt.year),
                    "month": int(dt.month),
                    "return": _safe_float(value * 100.0),
                }
            )

    drawdown = strategy_equity / strategy_equity.cummax() - 1.0
    var_95 = strategy_returns.quantile(0.05)
    cvar_95 = strategy_returns[strategy_returns <= var_95].mean() if (strategy_returns <= var_95).any() else np.nan

    bench_metrics = {"beta": None, "correlation": None}
    if not benchmark.empty:
        aligned = pd.concat([strategy_returns.rename("s"), benchmark.rename("b")], axis=1).dropna()
        if not aligned.empty:
            b_var = aligned["b"].var()
            beta = aligned["s"].cov(aligned["b"]) / b_var if b_var > 0 else np.nan
            bench_metrics = {
                "beta": _safe_float(beta),
                "correlation": _safe_float(aligned["s"].corr(aligned["b"])),
            }

    start = strategy_returns.index.min().date().isoformat() if isinstance(strategy_returns.index, pd.DatetimeIndex) else None
    end = strategy_returns.index.max().date().isoformat() if isinstance(strategy_returns.index, pd.DatetimeIndex) else None

    return {
        "summary": {
            "observations": int(len(strategy_returns)),
            "start": start,
            "end": end,
            "positive_days": int((strategy_returns > 0).sum()),
            "negative_days": int((strategy_returns < 0).sum()),
        },
        "yearly": yearly,
        "monthly": monthly_rows,
        "drawdown": {
            "current_dd": _safe_float(drawdown.iloc[-1] * 100.0),
            "max_dd": _safe_float(drawdown.min() * 100.0),
        },
        "tail_risk": {
            "var_95": _safe_float(var_95 * 100.0),
            "cvar_95": _safe_float(cvar_95 * 100.0),
        },
        "benchmark": bench_metrics,
        "turnover": {"avg_positions": None, "avg_turnover": None, "rebalance_events": 0},
    }


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        if pd.isna(value):
            return None
        parsed = float(value)
        if not np.isfinite(parsed):
            return None
        return parsed
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def _as_symbol_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = [p.strip().upper() for p in raw.split(",")]
    elif isinstance(raw, list):
        parts = [str(p).strip().upper() for p in raw]
    else:
        parts = []

    cleaned = []
    for symbol in parts:
        base = symbol.split(".")[0]
        if base and base not in cleaned:
            cleaned.append(base)
    return cleaned


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


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _is_cache_fresh(path: Path, ttl_seconds: int) -> bool:
    if ttl_seconds <= 0:
        return False
    if not path.exists() or not path.is_file():
        return False
    try:
        age_seconds = time.time() - path.stat().st_mtime
    except Exception:
        return False
    return age_seconds <= ttl_seconds and path.stat().st_size > 0


def _index_cache_path(index_name: str) -> Path:
    safe = "".join(ch if ch.isalnum() else "_" for ch in index_name.upper())
    return CACHE_DIR / f"index_components_{safe}.json"


def _load_cached_index_components(index_name: str) -> list[str] | None:
    cache_path = _index_cache_path(index_name)
    if not _is_cache_fresh(cache_path, INDEX_CACHE_TTL_SEC):
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    symbols = payload.get("symbols") if isinstance(payload, dict) else payload
    if not isinstance(symbols, list):
        return None
    cleaned: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        base = str(symbol).upper().split(".")[0]
        if not base or base in seen:
            continue
        seen.add(base)
        cleaned.append(base)
    return cleaned or None


def _save_cached_index_components(index_name: str, symbols: list[str]) -> None:
    if not symbols:
        return
    try:
        _ensure_cache_dir()
        cache_path = _index_cache_path(index_name)
        payload = {
            "symbols": symbols,
            "cached_at": time.time(),
        }
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
    except Exception:
        # Cache write failures should never break live API flow.
        return


def _price_cache_path(universe: str, period: str, interval: str, symbols: list[str]) -> Path:
    digest = hashlib.sha256(
        json.dumps(
            {
                "universe": universe,
                "period": period,
                "interval": interval,
                "symbols": symbols,
            },
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()[:24]
    return CACHE_DIR / f"prices_{digest}.pkl"


def _load_cached_prices(cache_path: Path, ttl_seconds: int | None) -> pd.DataFrame | None:
    if ttl_seconds is not None:
        if ttl_seconds <= 0:
            return None
        if not _is_cache_fresh(cache_path, ttl_seconds):
            return None
    elif not cache_path.exists():
        return None

    try:
        df = pd.read_pickle(cache_path)
    except Exception:
        return None

    if not isinstance(df, pd.DataFrame) or df.empty:
        return None

    required = {"Date", "Ticker", "Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(set(df.columns)):
        return None

    return df


def _save_cached_prices(cache_path: Path, prices: pd.DataFrame) -> None:
    if prices.empty:
        return
    try:
        _ensure_cache_dir()
        prices.to_pickle(cache_path)
    except Exception:
        # Cache write failures should never break live API flow.
        return


def _normalize_download_to_long(raw: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    required_cols = ["Open", "High", "Low", "Close", "Volume"]

    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0)
        # Handle "column-first" shape by swapping levels.
        if {"Open", "High", "Low", "Close"}.issubset(set(lvl0)):
            raw = raw.swaplevel(0, 1, axis=1).sort_index(axis=1)

        for ticker in dict.fromkeys(raw.columns.get_level_values(0)):
            sub = raw[ticker]
            if not isinstance(sub, pd.DataFrame) or sub.empty:
                continue
            sub = sub.rename_axis("Date").reset_index()
            sub["Ticker"] = str(ticker).upper().split(".")[0]
            for col in required_cols:
                if col not in sub.columns:
                    sub[col] = np.nan
            frames.append(sub[["Date", "Ticker", *required_cols]])
    else:
        sub = raw.rename_axis("Date").reset_index()
        ticker = symbols[0] if symbols else "UNKNOWN"
        sub["Ticker"] = str(ticker).upper().split(".")[0]
        for col in required_cols:
            if col not in sub.columns:
                sub[col] = np.nan
        frames.append(sub[["Date", "Ticker", *required_cols]])

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _normalize_indicator_payload(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        name = str(key).strip().lower()
        if name not in DEFAULT_INDICATORS:
            continue
        if not isinstance(value, dict):
            out[name] = {"enabled": False, "params": {}}
            continue
        enabled = bool(value.get("enabled", False))
        params = value.get("params", {})
        out[name] = {
            "enabled": enabled,
            "params": params if isinstance(params, dict) else {},
        }
    return out


def _resolve_runtime_config(payload: dict[str, Any]) -> dict[str, Any]:
    universe = str(payload.get("universe", "XU100")).upper()
    period = str(payload.get("period", "6mo"))
    interval = str(payload.get("interval", "1d"))
    max_symbols = _as_int(payload.get("max_symbols"), 50, minimum=5, maximum=200)
    top_n = _as_int(payload.get("top_n"), 30, minimum=1, maximum=200)
    buy_threshold = _as_float(payload.get("buy_threshold"), 0.2)
    sell_threshold = _as_float(payload.get("sell_threshold"), -0.2)
    if buy_threshold <= sell_threshold:
        raise ValueError("buy_threshold must be greater than sell_threshold.")

    custom_symbols = _as_symbol_list(payload.get("symbols"))
    indicator_payload = _normalize_indicator_payload(payload.get("indicators"))

    selected_indicators = [
        name for name in DEFAULT_INDICATORS.keys()
        if indicator_payload.get(name, {}).get("enabled", False)
    ]
    if not selected_indicators:
        selected_indicators = ["rsi", "macd", "supertrend"]

    return {
        "universe": universe,
        "period": period,
        "interval": interval,
        "max_symbols": max_symbols,
        "top_n": top_n,
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        "custom_symbols": custom_symbols,
        "indicator_payload": indicator_payload,
        "selected_indicators": selected_indicators,
    }


def _period_to_rows(period: str) -> int | None:
    key = str(period).strip().lower()
    mapping = {
        "1mo": 21,
        "3mo": 63,
        "6mo": 126,
        "1y": 252,
        "2y": 504,
        "5y": 1260,
        "max": None,
    }
    return mapping.get(key, 126)


def _load_local_price_panels(
    universe: str,
    period: str,
    interval: str,
    max_symbols: int,
    custom_symbols: list[str],
    runtime_paths: RuntimePaths,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if str(interval).strip().lower() != "1d":
        raise ValueError("Only interval='1d' is supported in local mode.")

    data_dir = runtime_paths.data_dir
    prices_file = data_dir / "bist_prices_full.csv"
    if not prices_file.exists() and not prices_file.with_suffix(".parquet").exists():
        gz_candidate = data_dir / "bist_prices_full.csv.gz"
        if gz_candidate.exists():
            prices_file = gz_candidate
    if not data_dir.exists():
        raise FileNotFoundError(f"Local data directory not found: {data_dir}")

    loader = DataLoader(data_dir=data_dir, regime_model_dir=runtime_paths.regime_outputs_dir)
    with redirect_stdout(io.StringIO()):
        prices = loader.load_prices(prices_file)
    if prices is None or prices.empty:
        raise ValueError("Local price dataset is empty.")

    local = prices.copy()
    local["Date"] = pd.to_datetime(local["Date"], errors="coerce")
    local = local.dropna(subset=["Date"]).copy()
    local["Ticker"] = local["Ticker"].astype(str).str.split(".").str[0].str.upper()

    close_df = local.pivot_table(index="Date", columns="Ticker", values="Close", aggfunc="last").sort_index()
    high_df = local.pivot_table(index="Date", columns="Ticker", values="High", aggfunc="last").sort_index()
    low_df = local.pivot_table(index="Date", columns="Ticker", values="Low", aggfunc="last").sort_index()
    vol_df = local.pivot_table(index="Date", columns="Ticker", values="Volume", aggfunc="last").sort_index()

    if close_df.empty:
        raise ValueError("No local close prices available after pivot.")

    available_symbols = [str(c).upper() for c in close_df.columns]
    symbol_set = set(available_symbols)

    if universe == "CUSTOM":
        symbols = [s for s in custom_symbols if s in symbol_set]
    elif universe == "XUTUM":
        symbols = available_symbols
    else:
        try:
            candidates = _get_index_components(universe)
        except Exception:
            candidates = []
        symbols = [s for s in candidates if s in symbol_set]
        if not symbols:
            # Fallback when index components are unavailable: pick most liquid names.
            if vol_df.empty:
                liq = pd.Series(index=available_symbols, data=np.nan, dtype="float64")
            else:
                liq = vol_df.rolling(63, min_periods=20).mean().iloc[-1]
            liq = liq.dropna().sort_values(ascending=False)
            if liq.empty:
                liq = pd.Series(index=available_symbols, data=1.0, dtype="float64")
            if universe == "XU030":
                symbols = [str(s) for s in liq.head(30).index.tolist()]
            elif universe == "XU100":
                symbols = [str(s) for s in liq.head(100).index.tolist()]
            else:
                symbols = [str(s) for s in liq.head(max_symbols).index.tolist()]

    if not symbols:
        raise ValueError(f"No symbols resolved for universe={universe}.")

    symbols = symbols[:max_symbols]

    close_df = close_df.reindex(columns=symbols).dropna(axis=1, how="all").ffill()
    high_df = high_df.reindex(columns=close_df.columns).dropna(axis=1, how="all").ffill()
    low_df = low_df.reindex(columns=close_df.columns).dropna(axis=1, how="all").ffill()

    row_budget = _period_to_rows(period)
    if row_budget is not None and row_budget > 0:
        close_df = close_df.tail(row_budget)
        high_df = high_df.reindex(index=close_df.index).tail(row_budget)
        low_df = low_df.reindex(index=close_df.index).tail(row_budget)

    close_df = close_df.dropna(axis=1, how="all")
    high_df = high_df.reindex(columns=close_df.columns)
    low_df = low_df.reindex(columns=close_df.columns)

    if close_df.empty:
        raise ValueError("Local dataset returned no usable rows for requested period/universe.")
    return close_df, high_df, low_df


def _get_index_components(index_name: str) -> list[str]:
    cached = _load_cached_index_components(index_name)
    if cached:
        return cached

    if bp is None:
        raise ValueError("borsapy dependency is unavailable for index component lookup.")

    try:
        idx = bp.index(index_name)
    except Exception as exc:
        raise ValueError(f"Failed to load index components for {index_name}: {exc}") from exc

    symbols: list[str] = []
    component_symbols = getattr(idx, "component_symbols", None)
    if isinstance(component_symbols, list):
        symbols = [str(s) for s in component_symbols if s]

    if not symbols:
        components = getattr(idx, "components", None)
        if isinstance(components, list):
            for item in components:
                if isinstance(item, dict):
                    symbol = str(item.get("symbol", "")).strip()
                    if symbol:
                        symbols.append(symbol)

    cleaned: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        base = symbol.upper().split(".")[0]
        if not base or base in seen:
            continue
        seen.add(base)
        cleaned.append(base)

    _save_cached_index_components(index_name, cleaned)
    return cleaned


def _download_to_long(symbols: list[str], period: str, interval: str) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    if bp is None:
        return pd.DataFrame()

    def _download(batch: list[str]) -> pd.DataFrame:
        with redirect_stdout(io.StringIO()):
            return bp.download(
                batch,
                period=period,
                interval=interval,
                group_by="ticker",
                progress=False,
            )

    # Fast path: single call.
    try:
        raw = _download(symbols)
        normalized = _normalize_download_to_long(raw, symbols)
        if not normalized.empty:
            return normalized
    except Exception:
        pass

    # Robust fallback: request smaller batches and merge successful chunks.
    frames: list[pd.DataFrame] = []
    for i in range(0, len(symbols), DOWNLOAD_BATCH_SIZE):
        batch = symbols[i:i + DOWNLOAD_BATCH_SIZE]
        try:
            raw_batch = _download(batch)
            normalized_batch = _normalize_download_to_long(raw_batch, batch)
            if not normalized_batch.empty:
                frames.append(normalized_batch)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_price_panels(
    universe: str,
    period: str,
    interval: str,
    max_symbols: int,
    custom_symbols: list[str],
    runtime_paths: RuntimePaths,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Primary path: local BIST pipeline data (offline and deterministic).
    try:
        return _load_local_price_panels(
            universe=universe,
            period=period,
            interval=interval,
            max_symbols=max_symbols,
            custom_symbols=custom_symbols,
            runtime_paths=runtime_paths,
        )
    except Exception as local_exc:
        LOGGER.warning("signal_construction.local_prices_unavailable", extra={"error": str(local_exc)})

    # Fallback path: borsapy download/cache.
    if universe == "CUSTOM":
        symbols = custom_symbols
    else:
        symbols = _get_index_components(universe)

    symbols = [s.upper().split(".")[0] for s in symbols if s]
    if not symbols:
        raise ValueError("No symbols found for the selected universe.")

    symbols = symbols[:max_symbols]
    cache_path = _price_cache_path(universe, period, interval, symbols)
    prices = _load_cached_prices(cache_path, PRICE_CACHE_TTL_SEC)

    if prices is None:
        downloaded = _download_to_long(symbols=symbols, period=period, interval=interval)
        if downloaded.empty:
            # Robust fallback: allow stale cache if network fetch fails.
            stale = _load_cached_prices(cache_path, ttl_seconds=None)
            if stale is not None and not stale.empty:
                prices = stale
            else:
                raise ValueError("No price data returned from borsapy.")
        else:
            prices = downloaded
            _save_cached_prices(cache_path, downloaded)

    for col in ("Date", "Ticker", "Open", "High", "Low", "Close", "Volume"):
        if col not in prices.columns:
            raise ValueError(f"Required price column is missing: {col}")

    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
    prices = prices.dropna(subset=["Date"]).sort_values(["Date", "Ticker"]).copy()

    close_df = prices.pivot_table(index="Date", columns="Ticker", values="Close").sort_index()
    high_df = prices.pivot_table(index="Date", columns="Ticker", values="High").sort_index()
    low_df = prices.pivot_table(index="Date", columns="Ticker", values="Low").sort_index()

    if close_df.empty:
        raise ValueError("Insufficient close prices after pivoting.")

    return close_df, high_df, low_df


def _build_signal_for_indicator(
    name: str,
    params: dict[str, Any],
    close_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """
    Returns:
      (latest_indicator_values, discrete_signal_series)
      where signal is in {-1, 0, 1}.
    """
    values_panel, signal_panel = _build_signal_panel_for_indicator(name, params, close_df, high_df, low_df)
    latest_values = values_panel.iloc[-1]
    latest_signal = signal_panel.iloc[-1].astype("int64")
    return latest_values, latest_signal


def _build_signal_panel_for_indicator(
    name: str,
    params: dict[str, Any],
    close_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      (indicator_values_panel, discrete_signal_panel)
      where signal panel values are in {-1, 0, 1}.
    """
    name = name.lower()

    if name == "rsi":
        period = _as_int(params.get("period"), 14, minimum=2, maximum=200)
        oversold = _as_float(params.get("oversold"), 30.0)
        overbought = _as_float(params.get("overbought"), 70.0)
        panel = BorsapyIndicators.build_rsi_panel(close_df, period=period)
        signal_panel = pd.DataFrame(
            np.where(panel <= oversold, 1, np.where(panel >= overbought, -1, 0)),
            index=panel.index,
            columns=panel.columns,
            dtype="int64",
        )
        return panel, signal_panel

    if name == "macd":
        fast = _as_int(params.get("fast"), 12, minimum=2, maximum=200)
        slow = _as_int(params.get("slow"), 26, minimum=2, maximum=300)
        signal_period = _as_int(params.get("signal"), 9, minimum=2, maximum=200)
        threshold = _as_float(params.get("threshold"), 0.0)
        panel = BorsapyIndicators.build_macd_panel(
            close_df,
            fast=fast,
            slow=slow,
            signal=signal_period,
            output="histogram",
        )
        signal_panel = pd.DataFrame(
            np.where(panel > threshold, 1, np.where(panel < -threshold, -1, 0)),
            index=panel.index,
            columns=panel.columns,
            dtype="int64",
        )
        return panel, signal_panel

    if name == "bollinger":
        period = _as_int(params.get("period"), 20, minimum=2, maximum=300)
        std_dev = _as_float(params.get("std_dev"), 2.0)
        lower = _as_float(params.get("lower"), 0.2)
        upper = _as_float(params.get("upper"), 0.8)
        panel = BorsapyIndicators.build_bollinger_panel(
            close_df,
            period=period,
            std_dev=std_dev,
            output="pct_b",
        )
        signal_panel = pd.DataFrame(
            np.where(panel <= lower, 1, np.where(panel >= upper, -1, 0)),
            index=panel.index,
            columns=panel.columns,
            dtype="int64",
        )
        return panel, signal_panel

    if name == "atr":
        period = _as_int(params.get("period"), 14, minimum=2, maximum=300)
        lower_pct = _as_float(params.get("lower_pct"), 0.3)
        upper_pct = _as_float(params.get("upper_pct"), 0.7)
        panel = BorsapyIndicators.build_atr_panel(high_df, low_df, close_df, period=period)
        ranks = panel.rank(axis=1, pct=True)
        signal_panel = pd.DataFrame(
            np.where(ranks <= lower_pct, 1, np.where(ranks >= upper_pct, -1, 0)),
            index=panel.index,
            columns=panel.columns,
            dtype="int64",
        )
        return panel, signal_panel

    if name == "stochastic":
        k_period = _as_int(params.get("k_period"), 14, minimum=2, maximum=200)
        d_period = _as_int(params.get("d_period"), 3, minimum=1, maximum=100)
        oversold = _as_float(params.get("oversold"), 20.0)
        overbought = _as_float(params.get("overbought"), 80.0)
        panel = BorsapyIndicators.build_stochastic_panel(
            high_df,
            low_df,
            close_df,
            k_period=k_period,
            d_period=d_period,
            output="k",
        )
        signal_panel = pd.DataFrame(
            np.where(panel <= oversold, 1, np.where(panel >= overbought, -1, 0)),
            index=panel.index,
            columns=panel.columns,
            dtype="int64",
        )
        return panel, signal_panel

    if name == "adx":
        period = _as_int(params.get("period"), 14, minimum=2, maximum=300)
        threshold = _as_float(params.get("trend_threshold"), 25.0)
        adx_panel = BorsapyIndicators.build_adx_panel(high_df, low_df, close_df, period=period, output="adx")
        plus_panel = BorsapyIndicators.build_adx_panel(
            high_df, low_df, close_df, period=period, output="di_plus"
        )
        minus_panel = BorsapyIndicators.build_adx_panel(
            high_df, low_df, close_df, period=period, output="di_minus"
        )

        is_trending = adx_panel >= threshold
        bullish = is_trending & (plus_panel > minus_panel)
        bearish = is_trending & (plus_panel < minus_panel)

        signal_panel = pd.DataFrame(
            np.where(bullish, 1, np.where(bearish, -1, 0)),
            index=adx_panel.index,
            columns=adx_panel.columns,
            dtype="int64",
        )
        return adx_panel, signal_panel

    if name == "supertrend":
        period = _as_int(params.get("period"), 10, minimum=2, maximum=200)
        multiplier = _as_float(params.get("multiplier"), 3.0)
        panel = BorsapyIndicators.build_supertrend_panel(
            high_df,
            low_df,
            close_df,
            period=period,
            multiplier=multiplier,
            output="direction",
        )
        filled = panel.fillna(0.0)
        signal_panel = (
            filled.gt(0).astype("int64")
            - filled.lt(0).astype("int64")
        )
        return panel, signal_panel

    if name == "ta_consensus":
        interval = str(params.get("interval", "1d"))
        batch_size = _as_int(params.get("batch_size"), 20, minimum=1, maximum=500)
        request_sleep_seconds = _as_float(params.get("request_sleep_seconds"), 0.0)
        batch_pause_seconds = _as_float(params.get("batch_pause_seconds"), 0.0)

        consensus_builder = TAConsensusSignals(
            batch_size=batch_size,
            request_sleep_seconds=request_sleep_seconds,
            batch_pause_seconds=batch_pause_seconds,
        )
        consensus = consensus_builder.build_consensus_panel(
            symbols=[str(symbol) for symbol in close_df.columns],
            interval=interval,
        )

        latest_scores = pd.Series(index=close_df.columns, dtype="float64")
        if not consensus.empty:
            latest_scores = pd.to_numeric(
                consensus.set_index("symbol")["consensus_score"].reindex(close_df.columns),
                errors="coerce",
            )

        panel = pd.DataFrame(
            np.tile(latest_scores.to_numpy(dtype="float64"), (len(close_df.index), 1)),
            index=close_df.index,
            columns=close_df.columns,
            dtype="float64",
        )
        signal_panel = pd.DataFrame(
            np.where(panel > 0.0, 1, np.where(panel < 0.0, -1, 0)),
            index=panel.index,
            columns=panel.columns,
            dtype="int64",
        )
        return panel, signal_panel

    raise ValueError(f"Unsupported indicator: {name}")


def _build_response(
    payload: dict[str, Any],
    *,
    runtime_paths: RuntimePaths | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    paths = _resolve_paths(runtime_paths)
    runtime = _resolve_runtime_config(payload)

    close_df, high_df, low_df = _load_price_panels(
        universe=runtime["universe"],
        period=runtime["period"],
        interval=runtime["interval"],
        max_symbols=runtime["max_symbols"],
        custom_symbols=runtime["custom_symbols"],
        runtime_paths=paths,
    )

    selected_indicators: list[str] = runtime["selected_indicators"]
    indicator_payload: dict[str, dict[str, Any]] = runtime["indicator_payload"]

    latest_values_by_indicator: dict[str, pd.Series] = {}
    discrete_signals_by_indicator: dict[str, pd.Series] = {}

    for indicator_name in selected_indicators:
        user_params = indicator_payload.get(indicator_name, {}).get("params", {})
        merged_params = dict(DEFAULT_INDICATORS.get(indicator_name, {}))
        if isinstance(user_params, dict):
            merged_params.update(user_params)

        with redirect_stdout(io.StringIO()):
            latest_values, discrete_signal = _build_signal_for_indicator(
                indicator_name,
                merged_params,
                close_df,
                high_df,
                low_df,
            )

        latest_values_by_indicator[indicator_name] = latest_values
        discrete_signals_by_indicator[indicator_name] = discrete_signal

    signal_df = pd.DataFrame(discrete_signals_by_indicator).fillna(0.0)
    value_df = pd.DataFrame(latest_values_by_indicator)

    combined_score = signal_df.mean(axis=1)
    buy_votes = (signal_df == 1).sum(axis=1)
    sell_votes = (signal_df == -1).sum(axis=1)
    hold_votes = (signal_df == 0).sum(axis=1)

    actions = pd.Series("HOLD", index=combined_score.index)
    actions.loc[combined_score >= runtime["buy_threshold"]] = "BUY"
    actions.loc[combined_score <= runtime["sell_threshold"]] = "SELL"

    result_frame = pd.DataFrame(
        {
            "symbol": combined_score.index,
            "combined_score": combined_score.values,
            "action": actions.values,
            "buy_votes": buy_votes.values,
            "sell_votes": sell_votes.values,
            "hold_votes": hold_votes.values,
        }
    ).set_index("symbol")

    result_frame = result_frame.join(value_df, how="left").join(signal_df.add_suffix("_signal"), how="left")
    result_frame = result_frame.sort_values(["combined_score", "buy_votes"], ascending=[False, False])

    rows: list[dict[str, Any]] = []
    for symbol, row in result_frame.head(runtime["top_n"]).iterrows():
        indicator_values = {
            name: _safe_float(row.get(name))
            for name in selected_indicators
        }
        indicator_signals = {
            name: _safe_int(row.get(f"{name}_signal"))
            for name in selected_indicators
        }

        rows.append(
            {
                "symbol": str(symbol),
                "action": str(row["action"]),
                "combined_score": _safe_float(row["combined_score"]),
                "buy_votes": _safe_int(row["buy_votes"]),
                "sell_votes": _safe_int(row["sell_votes"]),
                "hold_votes": _safe_int(row["hold_votes"]),
                "indicator_values": indicator_values,
                "indicator_signals": indicator_signals,
            }
        )

    indicator_summaries: list[dict[str, Any]] = []
    for name in selected_indicators:
        series = signal_df[name]
        indicator_summaries.append(
            {
                "name": name,
                "buy_count": int((series == 1).sum()),
                "sell_count": int((series == -1).sum()),
                "hold_count": int((series == 0).sum()),
            }
        )

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    as_of = close_df.index.max()
    as_of_iso = as_of.isoformat() if hasattr(as_of, "isoformat") else str(as_of)

    return {
        "meta": {
            "mode": "construct",
            "universe": runtime["universe"],
            "period": runtime["period"],
            "interval": runtime["interval"],
            "symbols_used": int(close_df.shape[1]),
            "rows_used": int(close_df.shape[0]),
            "as_of": as_of_iso,
            "indicators": selected_indicators,
            "execution_ms": elapsed_ms,
        },
        "indicator_summaries": indicator_summaries,
        "signals": rows,
    }


def _calculate_performance_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> dict[str, Any]:
    if portfolio_returns.empty:
        return {
            "cagr": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "ytd": 0.0,
            "total_return": 0.0,
            "volatility": 0.0,
            "win_rate": 0.0,
            "beta": None,
        }

    portfolio_returns = portfolio_returns.fillna(0.0)
    benchmark_returns = benchmark_returns.reindex(portfolio_returns.index).fillna(0.0)

    equity = (1.0 + portfolio_returns).cumprod()
    bench_equity = (1.0 + benchmark_returns).cumprod()

    n_years = len(portfolio_returns) / 252.0
    if n_years > 0 and equity.iloc[-1] > 0:
        cagr = float(equity.iloc[-1] ** (1.0 / n_years) - 1.0)
    else:
        cagr = 0.0

    std = float(portfolio_returns.std())
    sharpe = float(portfolio_returns.mean() / std * np.sqrt(252.0)) if std > 0 else 0.0

    drawdown = equity / equity.cummax() - 1.0
    max_dd = float(drawdown.min())

    last_date = portfolio_returns.index.max()
    ytd_mask = portfolio_returns.index.year == last_date.year
    ytd_returns = portfolio_returns[ytd_mask]
    ytd = float((1.0 + ytd_returns).prod() - 1.0) if len(ytd_returns) else 0.0

    total_return = float(equity.iloc[-1] - 1.0)
    volatility = float(std * np.sqrt(252.0))
    win_rate = float((portfolio_returns > 0).mean())

    bench_var = float(benchmark_returns.var())
    if bench_var > 0:
        beta = float(portfolio_returns.cov(benchmark_returns) / bench_var)
    else:
        beta = None

    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "ytd": ytd,
        "total_return": total_return,
        "volatility": volatility,
        "win_rate": win_rate,
        "beta": beta,
        "equity": equity,
        "benchmark_equity": bench_equity,
    }


def _build_portfolio_and_benchmark_returns(
    returns_df: pd.DataFrame,
    lagged_actions: pd.DataFrame,
    lagged_scores: pd.DataFrame,
    max_positions: int,
) -> tuple[pd.Series, pd.Series]:
    """
    Vectorized daily portfolio construction:
    - Select lagged BUY universe
    - Pick top-N by lagged combined score
    - Equal-weight selected names
    """
    aligned_actions = lagged_actions.reindex_like(returns_df).fillna(0).astype("int8")
    aligned_scores = lagged_scores.reindex_like(returns_df).astype("float64")

    eligible = (aligned_actions == 1) & aligned_scores.notna()

    # Rank BUY-eligible scores cross-sectionally by date.
    ranked_scores = aligned_scores.where(eligible, np.nan).rank(
        axis=1,
        ascending=False,
        method="first",
    )
    selected_mask = eligible & ranked_scores.le(max_positions)

    selected_counts = selected_mask.sum(axis=1)
    weighted_sum = returns_df.where(selected_mask, 0.0).sum(axis=1)
    portfolio_series = weighted_sum.div(selected_counts.replace(0, np.nan)).fillna(0.0)
    benchmark_series = returns_df.mean(axis=1).fillna(0.0)

    # Keep behavior equivalent to prior implementation (starts from second row).
    if len(portfolio_series) > 1:
        portfolio_series = portfolio_series.iloc[1:]
        benchmark_series = benchmark_series.iloc[1:]
    else:
        portfolio_series = pd.Series(dtype="float64")
        benchmark_series = pd.Series(dtype="float64")

    return portfolio_series.astype("float64"), benchmark_series.astype("float64")


def _build_backtest_response(
    payload: dict[str, Any],
    *,
    runtime_paths: RuntimePaths | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    paths = _resolve_paths(runtime_paths)
    runtime = _resolve_runtime_config(payload)

    max_positions = _as_int(
        payload.get("max_positions", payload.get("top_n", runtime["top_n"])),
        runtime["top_n"],
        minimum=1,
        maximum=200,
    )

    close_df, high_df, low_df = _load_price_panels(
        universe=runtime["universe"],
        period=runtime["period"],
        interval=runtime["interval"],
        max_symbols=runtime["max_symbols"],
        custom_symbols=runtime["custom_symbols"],
        runtime_paths=paths,
    )

    selected_indicators: list[str] = runtime["selected_indicators"]
    indicator_payload: dict[str, dict[str, Any]] = runtime["indicator_payload"]

    indicator_value_panels: dict[str, pd.DataFrame] = {}
    indicator_signal_panels: dict[str, pd.DataFrame] = {}

    for indicator_name in selected_indicators:
        user_params = indicator_payload.get(indicator_name, {}).get("params", {})
        merged_params = dict(DEFAULT_INDICATORS.get(indicator_name, {}))
        if isinstance(user_params, dict):
            merged_params.update(user_params)

        with redirect_stdout(io.StringIO()):
            values_panel, signal_panel = _build_signal_panel_for_indicator(
                indicator_name,
                merged_params,
                close_df,
                high_df,
                low_df,
            )

        indicator_value_panels[indicator_name] = values_panel
        indicator_signal_panels[indicator_name] = signal_panel

    combined_score_df: pd.DataFrame | None = None
    for panel in indicator_signal_panels.values():
        panel_float = panel.astype("float64")
        if combined_score_df is None:
            combined_score_df = panel_float.copy()
        else:
            combined_score_df = combined_score_df.add(panel_float, fill_value=0.0)

    if combined_score_df is None:
        raise ValueError("No indicator signals available for backtest.")

    combined_score_df = combined_score_df / float(len(selected_indicators))

    action_df = pd.DataFrame(
        np.where(
            combined_score_df >= runtime["buy_threshold"],
            1,
            np.where(combined_score_df <= runtime["sell_threshold"], -1, 0),
        ),
        index=combined_score_df.index,
        columns=combined_score_df.columns,
        dtype="int64",
    )

    returns_df = close_df.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    lagged_actions = action_df.shift(1).fillna(0).astype("int64")
    lagged_scores = combined_score_df.shift(1)

    portfolio_series, benchmark_series = _build_portfolio_and_benchmark_returns(
        returns_df=returns_df,
        lagged_actions=lagged_actions,
        lagged_scores=lagged_scores,
        max_positions=max_positions,
    )

    metrics = _calculate_performance_metrics(portfolio_series, benchmark_series)
    equity = metrics.get("equity", pd.Series(dtype="float64"))
    benchmark_equity = metrics.get("benchmark_equity", pd.Series(dtype="float64"))

    latest_actions = action_df.iloc[-1]
    latest_scores = combined_score_df.iloc[-1]
    latest_buy_scores = latest_scores[latest_actions == 1].dropna().sort_values(ascending=False)
    current_holdings = list(latest_buy_scores.head(max_positions).index)

    latest_rows = []
    latest_signal_df = pd.DataFrame({name: panel.iloc[-1] for name, panel in indicator_signal_panels.items()}).fillna(0.0)
    latest_values_df = pd.DataFrame({name: panel.iloc[-1] for name, panel in indicator_value_panels.items()})

    combined_latest = latest_signal_df.mean(axis=1)
    buy_votes = (latest_signal_df == 1).sum(axis=1)
    sell_votes = (latest_signal_df == -1).sum(axis=1)
    hold_votes = (latest_signal_df == 0).sum(axis=1)

    latest_actions_label = pd.Series("HOLD", index=combined_latest.index)
    latest_actions_label.loc[combined_latest >= runtime["buy_threshold"]] = "BUY"
    latest_actions_label.loc[combined_latest <= runtime["sell_threshold"]] = "SELL"

    latest_result_frame = pd.DataFrame(
        {
            "symbol": combined_latest.index,
            "combined_score": combined_latest.values,
            "action": latest_actions_label.values,
            "buy_votes": buy_votes.values,
            "sell_votes": sell_votes.values,
            "hold_votes": hold_votes.values,
        }
    ).set_index("symbol")
    latest_result_frame = latest_result_frame.join(latest_values_df, how="left").join(
        latest_signal_df.add_suffix("_signal"), how="left"
    )
    latest_result_frame = latest_result_frame.sort_values(["combined_score", "buy_votes"], ascending=[False, False])

    for symbol, row in latest_result_frame.head(runtime["top_n"]).iterrows():
        latest_rows.append(
            {
                "symbol": str(symbol),
                "action": str(row["action"]),
                "combined_score": _safe_float(row["combined_score"]),
                "buy_votes": _safe_int(row["buy_votes"]),
                "sell_votes": _safe_int(row["sell_votes"]),
                "hold_votes": _safe_int(row["hold_votes"]),
                "indicator_values": {name: _safe_float(row.get(name)) for name in selected_indicators},
                "indicator_signals": {name: _safe_int(row.get(f"{name}_signal")) for name in selected_indicators},
            }
        )

    indicator_summaries: list[dict[str, Any]] = []
    for name in selected_indicators:
        latest_series = indicator_signal_panels[name].iloc[-1]
        indicator_summaries.append(
            {
                "name": name,
                "buy_count": int((latest_series == 1).sum()),
                "sell_count": int((latest_series == -1).sum()),
                "hold_count": int((latest_series == 0).sum()),
            }
        )

    as_of = close_df.index.max()
    as_of_iso = as_of.isoformat() if hasattr(as_of, "isoformat") else str(as_of)

    equity_curve = [
        {
            "date": idx.date().isoformat(),
            "value": round(float(val), 6),
        }
        for idx, val in equity.items()
    ]
    benchmark_curve = [
        {
            "date": idx.date().isoformat(),
            "value": round(float(val), 6),
        }
        for idx, val in benchmark_equity.items()
    ]

    focus_symbol_raw = str(payload.get("focus_symbol") or "").strip().upper().split(".")[0]
    focus_symbol = ""
    if focus_symbol_raw and focus_symbol_raw in close_df.columns:
        focus_symbol = focus_symbol_raw
    elif current_holdings:
        for symbol in current_holdings:
            candidate = str(symbol).strip().upper().split(".")[0]
            if candidate in close_df.columns:
                focus_symbol = candidate
                break
    if not focus_symbol and latest_rows:
        candidate = str(latest_rows[0].get("symbol", "")).strip().upper().split(".")[0]
        if candidate in close_df.columns:
            focus_symbol = candidate
    if not focus_symbol and len(close_df.columns) > 0:
        focus_symbol = str(close_df.columns[0])

    overlay_payload: dict[str, Any] = {
        "symbol": focus_symbol or None,
        "points": [],
        "markers": [],
    }
    if focus_symbol:
        overlay_close = pd.to_numeric(close_df[focus_symbol], errors="coerce")
        overlay_score = pd.to_numeric(combined_score_df[focus_symbol], errors="coerce")
        overlay_action = pd.to_numeric(action_df[focus_symbol], errors="coerce").fillna(0).astype("int64")
        overlay_frame = pd.DataFrame(
            {
                "close": overlay_close,
                "combined_score": overlay_score,
                "action": overlay_action,
            }
        ).dropna(subset=["close"])
        overlay_frame = overlay_frame.tail(400)

        points: list[dict[str, Any]] = []
        markers: list[dict[str, Any]] = []
        prev_action = 0
        for idx, row in overlay_frame.iterrows():
            action_value = int(row["action"])
            points.append(
                {
                    "date": idx.date().isoformat(),
                    "close": round(float(row["close"]), 6),
                    "combined_score": _safe_float(row["combined_score"]),
                    "action": action_value,
                }
            )
            if action_value != 0 and action_value != prev_action:
                markers.append(
                    {
                        "date": idx.date().isoformat(),
                        "close": round(float(row["close"]), 6),
                        "action": "BUY" if action_value > 0 else "SELL",
                    }
                )
            prev_action = action_value
        overlay_payload = {
            "symbol": focus_symbol,
            "points": points,
            "markers": markers,
        }

    def _validation_metrics(view_metrics: dict[str, Any]) -> dict[str, float | None]:
        return {
            "cagr": round(float(view_metrics.get("cagr", 0.0)) * 100.0, 2),
            "sharpe": round(float(view_metrics.get("sharpe", 0.0)), 3),
            "max_dd": round(float(view_metrics.get("max_dd", 0.0)) * 100.0, 2),
            "total_return": round(float(view_metrics.get("total_return", 0.0)) * 100.0, 2),
            "volatility": round(float(view_metrics.get("volatility", 0.0)) * 100.0, 2),
            "win_rate": round(float(view_metrics.get("win_rate", 0.0)) * 100.0, 2),
            "beta": None
            if view_metrics.get("beta") is None
            else round(float(view_metrics.get("beta", 0.0)), 3),
        }

    validation_payload: dict[str, Any] = {"available": False}
    if len(portfolio_series) >= 40:
        split_idx = int(len(portfolio_series) * 0.7)
        split_idx = max(20, min(split_idx, len(portfolio_series) - 10))

        train_returns = portfolio_series.iloc[:split_idx]
        test_returns = portfolio_series.iloc[split_idx:]
        train_benchmark = benchmark_series.iloc[:split_idx]
        test_benchmark = benchmark_series.iloc[split_idx:]

        train_metrics = _calculate_performance_metrics(train_returns, train_benchmark)
        test_metrics = _calculate_performance_metrics(test_returns, test_benchmark)

        walk_forward: list[dict[str, Any]] = []
        wf_splits = 3
        segment = max(15, len(portfolio_series) // (wf_splits + 1))
        for split in range(1, wf_splits + 1):
            train_end = min(len(portfolio_series) - segment, split * segment)
            test_end = min(len(portfolio_series), train_end + segment)
            if train_end < 15 or test_end - train_end < 10:
                continue
            wf_train = portfolio_series.iloc[:train_end]
            wf_test = portfolio_series.iloc[train_end:test_end]
            wf_train_bench = benchmark_series.iloc[:train_end]
            wf_test_bench = benchmark_series.iloc[train_end:test_end]
            wf_train_metrics = _calculate_performance_metrics(wf_train, wf_train_bench)
            wf_test_metrics = _calculate_performance_metrics(wf_test, wf_test_bench)
            walk_forward.append(
                {
                    "split": split,
                    "train_start": wf_train.index.min().date().isoformat(),
                    "train_end": wf_train.index.max().date().isoformat(),
                    "test_start": wf_test.index.min().date().isoformat(),
                    "test_end": wf_test.index.max().date().isoformat(),
                    "train_sharpe": round(float(wf_train_metrics.get("sharpe", 0.0)), 3),
                    "test_sharpe": round(float(wf_test_metrics.get("sharpe", 0.0)), 3),
                    "train_return": round(float(wf_train_metrics.get("total_return", 0.0)) * 100.0, 2),
                    "test_return": round(float(wf_test_metrics.get("total_return", 0.0)) * 100.0, 2),
                }
            )

        validation_payload = {
            "available": True,
            "train_ratio": 0.7,
            "train_period": {
                "start": train_returns.index.min().date().isoformat(),
                "end": train_returns.index.max().date().isoformat(),
            },
            "test_period": {
                "start": test_returns.index.min().date().isoformat(),
                "end": test_returns.index.max().date().isoformat(),
            },
            "train_metrics": _validation_metrics(train_metrics),
            "test_metrics": _validation_metrics(test_metrics),
            "stability": {
                "sharpe_delta": round(
                    float(test_metrics.get("sharpe", 0.0)) - float(train_metrics.get("sharpe", 0.0)),
                    3,
                ),
                "return_delta": round(
                    (float(test_metrics.get("total_return", 0.0)) - float(train_metrics.get("total_return", 0.0)))
                    * 100.0,
                    2,
                ),
            },
            "walk_forward": walk_forward,
        }

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    analytics_v2 = _build_backtest_analytics_v2(
        returns=portfolio_series,
        equity=equity,
        benchmark_returns=benchmark_series,
        holdings_history=None,
    )

    return {
        "meta": {
            "mode": "backtest",
            "universe": runtime["universe"],
            "period": runtime["period"],
            "interval": runtime["interval"],
            "symbols_used": int(close_df.shape[1]),
            "rows_used": int(close_df.shape[0]),
            "as_of": as_of_iso,
            "indicators": selected_indicators,
            "buy_threshold": runtime["buy_threshold"],
            "sell_threshold": runtime["sell_threshold"],
            "max_positions": max_positions,
            "execution_ms": elapsed_ms,
        },
        "metrics": {
            "cagr": round(float(metrics["cagr"]) * 100.0, 2),
            "sharpe": round(float(metrics["sharpe"]), 3),
            "max_dd": round(float(metrics["max_dd"]) * 100.0, 2),
            "ytd": round(float(metrics["ytd"]) * 100.0, 2),
            "total_return": round(float(metrics["total_return"]) * 100.0, 2),
            "volatility": round(float(metrics["volatility"]) * 100.0, 2),
            "win_rate": round(float(metrics["win_rate"]) * 100.0, 2),
            "beta": None if metrics["beta"] is None else round(float(metrics["beta"]), 3),
            "last_rebalance": as_of_iso[:10],
        },
        "signals": latest_rows,
        "indicator_summaries": indicator_summaries,
        "current_holdings": current_holdings,
        "equity_curve": equity_curve,
        "benchmark_curve": benchmark_curve,
        "price_overlay": overlay_payload,
        "validation": validation_payload,
        "analytics_v2": analytics_v2,
    }


def run_signal_snapshot(
    payload: dict[str, Any],
    *,
    runtime_paths: RuntimePaths | None = None,
) -> SignalConstructionSnapshotResult:
    if not isinstance(payload, dict):
        raise QuantEngineValidationError("Request payload must be a JSON object.")
    try:
        response = _build_response(payload, runtime_paths=runtime_paths)
        required = {"meta", "indicator_summaries", "signals"}
        missing = sorted(required.difference(response.keys()))
        if missing:
            raise QuantEngineExecutionError(
                f"Signal snapshot result is missing required keys: {', '.join(missing)}."
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
        LOGGER.exception("signal_construction.snapshot_failed")
        raise QuantEngineExecutionError(
            "Signal construction failed.",
            user_message="Signal construction failed.",
        ) from exc


def get_signal_metadata() -> dict[str, Any]:
    return {
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
            {"key": "ta_consensus", "label": "TradingView TA Consensus"},
        ],
    }


def run_signal_backtest(
    payload: dict[str, Any],
    *,
    runtime_paths: RuntimePaths | None = None,
) -> SignalBacktestResult:
    if not isinstance(payload, dict):
        raise QuantEngineValidationError("Request payload must be a JSON object.")
    try:
        response = _build_backtest_response(payload, runtime_paths=runtime_paths)
        required = {
            "meta",
            "metrics",
            "signals",
            "current_holdings",
            "equity_curve",
            "benchmark_curve",
            "analytics_v2",
        }
        missing = sorted(required.difference(response.keys()))
        if missing:
            raise QuantEngineExecutionError(
                f"Signal backtest result is missing required keys: {', '.join(missing)}."
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
        LOGGER.exception("signal_construction.backtest_failed")
        raise QuantEngineExecutionError(
            "Signal backtest failed.",
            user_message="Signal backtest failed.",
        ) from exc
