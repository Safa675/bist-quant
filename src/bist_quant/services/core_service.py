from __future__ import annotations

import copy
import itertools
import math
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from bist_quant.common.data_loader import DataLoader
from bist_quant.common.report_generator import compute_capm_metrics
from bist_quant.portfolio import PortfolioEngine, load_signal_configs
from bist_quant.runtime import (
    BackendPaths,
    RuntimePaths,
    resolve_runtime_paths,
    validate_runtime_paths,
)

DEFAULT_ENGINE_START_DATE = "2018-01-01"
DEFAULT_ENGINE_END_DATE = "2024-12-31"

ProgressCallback = Callable[[float, str, int | None], None]
CancelCheck = Callable[[], bool]


@dataclass(frozen=True)
class FactorRunSpec:
    name: str
    weight: float
    signal_params: dict[str, Any]


class CoreBackendService:
    """Service layer to expose core BIST functionality to the API."""

    def __init__(
        self,
        project_root: Path | str | None = None,
        data_dir: Path | str | None = None,
        regime_dir: Path | str | None = None,
        runtime_paths: RuntimePaths | None = None,
        strict_paths: bool = True,
    ) -> None:
        self.runtime = runtime_paths or resolve_runtime_paths(
            project_root=project_root,
            data_dir=data_dir,
            regime_dir=regime_dir,
        )
        if strict_paths:
            validate_runtime_paths(self.runtime, require_price_data=False)
        self.paths = self.runtime.to_backend_paths()

    def load_signal_configs(self) -> dict[str, dict]:
        configs = load_signal_configs()
        if not isinstance(configs, dict):
            raise TypeError(f"Expected dict for signal configs, got {type(configs).__name__}")
        return configs

    def list_available_signals(self) -> list[str]:
        return sorted(self.load_signal_configs().keys())

    @staticmethod
    def _normalize_factor_name(factor_name: str) -> str:
        normalized = str(factor_name).strip()
        if not normalized:
            raise ValueError("factor_name must be a non-empty string")
        return normalized

    @staticmethod
    def _to_date_string(value: str | pd.Timestamp) -> str:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            raise ValueError(f"Invalid date: {value}")
        return str(ts.date())

    @staticmethod
    def _serialize_equity_curve(
        equity: pd.Series,
        benchmark_equity: pd.Series | None,
        drawdown: pd.Series,
    ) -> list[dict[str, float | str | None]]:
        if not isinstance(equity, pd.Series) or equity.empty:
            return []

        benchmark_aligned = benchmark_equity.reindex(equity.index) if benchmark_equity is not None else None
        drawdown_aligned = drawdown.reindex(equity.index)

        payload: list[dict[str, float | str | None]] = []
        for idx, value in equity.items():
            date_key = str(pd.Timestamp(idx).date())
            bench_value = None
            if benchmark_aligned is not None:
                raw_bench = benchmark_aligned.loc[idx]
                if pd.notna(raw_bench):
                    bench_value = float(raw_bench)

            dd_value = drawdown_aligned.loc[idx]
            payload.append(
                {
                    "date": date_key,
                    "value": float(value),
                    "benchmark": bench_value,
                    "drawdown": float(dd_value) if pd.notna(dd_value) else None,
                }
            )
        return payload

    @staticmethod
    def _serialize_drawdown_curve(drawdown: pd.Series) -> list[dict[str, float | str]]:
        if not isinstance(drawdown, pd.Series) or drawdown.empty:
            return []

        payload: list[dict[str, float | str]] = []
        for idx, value in drawdown.items():
            payload.append(
                {
                    "date": str(pd.Timestamp(idx).date()),
                    "value": float(value),
                }
            )
        return payload

    @staticmethod
    def _extract_latest_holdings(
        holdings_history: Any,
        top_n: int = 10,
    ) -> list[dict[str, float | str]]:
        if not isinstance(holdings_history, list) or len(holdings_history) == 0:
            return []

        holdings_df = pd.DataFrame(holdings_history)
        required_columns = {"date", "ticker", "weight"}
        if holdings_df.empty or not required_columns.issubset(set(holdings_df.columns)):
            return []

        holdings_df["date"] = pd.to_datetime(holdings_df["date"], errors="coerce")
        holdings_df = holdings_df.dropna(subset=["date", "ticker", "weight"])
        if holdings_df.empty:
            return []

        latest_date = holdings_df["date"].max()
        latest = holdings_df[holdings_df["date"] == latest_date].copy()
        if latest.empty:
            return []

        latest = latest.groupby("ticker", as_index=False)["weight"].sum()
        latest = latest.sort_values("weight", ascending=False).head(max(top_n, 1))

        weight_sum = float(latest["weight"].sum())
        if weight_sum > 0:
            latest["weight"] = latest["weight"] / weight_sum

        return [
            {
                "ticker": str(row["ticker"]),
                "weight": float(row["weight"]),
            }
            for _, row in latest.iterrows()
        ]

    @staticmethod
    def _serialize_monthly_returns(
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series | None,
    ) -> list[dict[str, float | str | None]]:
        if strategy_returns.empty:
            return []

        strategy_monthly = (1.0 + strategy_returns).resample("ME").prod() - 1.0
        benchmark_monthly = None
        if benchmark_returns is not None and not benchmark_returns.empty:
            benchmark_monthly = (1.0 + benchmark_returns).resample("ME").prod() - 1.0
            benchmark_monthly = benchmark_monthly.reindex(strategy_monthly.index)

        rows: list[dict[str, float | str | None]] = []
        for idx, strat_value in strategy_monthly.items():
            benchmark_value = None
            if benchmark_monthly is not None:
                raw = benchmark_monthly.loc[idx]
                benchmark_value = float(raw) if pd.notna(raw) else None

            excess = None
            if benchmark_value is not None:
                excess = float(strat_value) - benchmark_value

            rows.append(
                {
                    "month": pd.Timestamp(idx).strftime("%Y-%m"),
                    "strategy_return": float(strat_value),
                    "benchmark_return": benchmark_value,
                    "excess_return": excess,
                }
            )

        return rows

    @staticmethod
    def _serialize_rolling_metrics(strategy_returns: pd.Series) -> list[dict[str, float | str | None]]:
        if strategy_returns.empty:
            return []

        rolling_mean = strategy_returns.rolling(63, min_periods=20).mean()
        rolling_std = strategy_returns.rolling(63, min_periods=20).std()
        rolling_sharpe = (rolling_mean / rolling_std.replace(0.0, np.nan)) * np.sqrt(252)
        rolling_vol = rolling_std * np.sqrt(252)

        rolling_equity = (1.0 + strategy_returns).cumprod()
        rolling_peak = rolling_equity.rolling(126, min_periods=40).max()
        rolling_dd = rolling_equity / rolling_peak - 1.0
        rolling_max_dd = rolling_dd.rolling(126, min_periods=40).min()

        rows: list[dict[str, float | str | None]] = []
        for idx in strategy_returns.index:
            sharpe_val = rolling_sharpe.loc[idx]
            vol_val = rolling_vol.loc[idx]
            max_dd_val = rolling_max_dd.loc[idx]

            rows.append(
                {
                    "date": str(pd.Timestamp(idx).date()),
                    "rolling_sharpe_63d": float(sharpe_val) if pd.notna(sharpe_val) else None,
                    "rolling_volatility_63d": float(vol_val) if pd.notna(vol_val) else None,
                    "rolling_max_drawdown_126d": float(max_dd_val) if pd.notna(max_dd_val) else None,
                }
            )

        return rows

    @staticmethod
    def _build_factor_breakdown(
        raw_results: dict[str, dict[str, Any]],
        normalized_weights: dict[str, float],
    ) -> dict[str, Any]:
        breakdown: dict[str, Any] = {}
        for factor_name, raw in raw_results.items():
            returns = raw.get("returns")
            total_return = None
            if isinstance(returns, pd.Series) and not returns.empty:
                total_return = float((1.0 + returns).prod() - 1.0)

            breakdown[factor_name] = {
                "weight": float(normalized_weights.get(factor_name, 0.0)),
                "cagr": float(raw.get("cagr", 0.0)),
                "sharpe": float(raw.get("sharpe", 0.0)),
                "max_drawdown": float(raw.get("max_drawdown", 0.0)),
                "total_return": total_return,
                "rebalance_count": int(raw.get("rebalance_count", 0)),
                "trade_count": int(raw.get("trade_count", 0)),
            }
        return breakdown

    @staticmethod
    def _build_performance_attribution(
        factor_returns: pd.DataFrame,
        normalized_weights: dict[str, float],
    ) -> list[dict[str, float | str]]:
        if factor_returns.empty:
            return []

        rows: list[dict[str, float | str]] = []
        for factor_name in factor_returns.columns:
            series = factor_returns[factor_name].dropna()
            if series.empty:
                continue
            weight = float(normalized_weights.get(factor_name, 0.0))
            contribution = float((1.0 + series * weight).prod() - 1.0)
            rows.append(
                {
                    "key": factor_name,
                    "contribution": contribution,
                }
            )

        rows.sort(key=lambda row: row["contribution"], reverse=True)
        return rows

    @staticmethod
    def _compute_metrics(
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series | None,
    ) -> dict[str, float | None]:
        if strategy_returns.empty:
            return {
                "total_return": 0.0,
                "cagr": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "calmar": 0.0,
                "max_drawdown": 0.0,
                "annualized_volatility": 0.0,
                "win_rate": 0.0,
                "beta": None,
                "alpha_annual": None,
                "tracking_error": None,
                "information_ratio": None,
            }

        equity = (1.0 + strategy_returns).cumprod()
        total_return = float(equity.iloc[-1] - 1.0)

        n_years = max(len(strategy_returns) / 252.0, 1.0 / 252.0)
        cagr = float((1.0 + total_return) ** (1.0 / n_years) - 1.0)

        returns_std = strategy_returns.std()
        annualized_volatility = float(returns_std * np.sqrt(252)) if pd.notna(returns_std) else 0.0
        sharpe = (
            float(strategy_returns.mean() / returns_std * np.sqrt(252))
            if pd.notna(returns_std) and returns_std > 0
            else 0.0
        )

        downside = strategy_returns[strategy_returns < 0]
        downside_std = downside.std()
        sortino = (
            float(strategy_returns.mean() / downside_std * np.sqrt(252))
            if pd.notna(downside_std) and downside_std > 0
            else 0.0
        )

        drawdown = equity / equity.cummax() - 1.0
        max_drawdown = float(drawdown.min())
        calmar = float(cagr / abs(max_drawdown)) if max_drawdown < 0 else 0.0
        win_rate = float((strategy_returns > 0).sum() / len(strategy_returns))

        beta = None
        alpha_annual = None
        tracking_error = None
        information_ratio = None

        if benchmark_returns is not None and not benchmark_returns.empty:
            aligned = pd.concat(
                [strategy_returns.rename("strategy"), benchmark_returns.rename("benchmark")],
                axis=1,
            ).dropna()
            if not aligned.empty:
                capm = compute_capm_metrics(aligned["strategy"], aligned["benchmark"])
                beta = float(capm.get("beta")) if pd.notna(capm.get("beta")) else None
                alpha_annual = (
                    float(capm.get("alpha_annual"))
                    if pd.notna(capm.get("alpha_annual"))
                    else None
                )

                excess = aligned["strategy"] - aligned["benchmark"]
                excess_std = excess.std()
                if pd.notna(excess_std) and excess_std > 0:
                    tracking_error = float(excess_std * np.sqrt(252))
                    information_ratio = float(excess.mean() / excess_std * np.sqrt(252))
                else:
                    tracking_error = 0.0
                    information_ratio = 0.0

        return {
            "total_return": total_return,
            "cagr": cagr,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown": max_drawdown,
            "annualized_volatility": annualized_volatility,
            "win_rate": win_rate,
            "beta": beta,
            "alpha_annual": alpha_annual,
            "tracking_error": tracking_error,
            "information_ratio": information_ratio,
        }

    @staticmethod
    def _serialize_daily_returns(
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series | None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if strategy_returns.empty:
            return rows

        benchmark_aligned = (
            benchmark_returns.reindex(strategy_returns.index)
            if benchmark_returns is not None
            else None
        )

        for idx, value in strategy_returns.items():
            row: dict[str, Any] = {
                "date": str(pd.Timestamp(idx).date()),
                "strategy_return": float(value),
                "benchmark_return": None,
            }
            if benchmark_aligned is not None:
                bench = benchmark_aligned.loc[idx]
                if pd.notna(bench):
                    row["benchmark_return"] = float(bench)
            rows.append(row)
        return rows

    @staticmethod
    def _compute_tail_risk(strategy_returns: pd.Series) -> dict[str, float | None]:
        clean = strategy_returns.dropna()
        if clean.empty:
            return {
                "var_95": None,
                "cvar_95": None,
                "var_99": None,
                "cvar_99": None,
            }

        var_95 = float(clean.quantile(0.05))
        var_99 = float(clean.quantile(0.01))
        cvar_95 = float(clean[clean <= var_95].mean()) if (clean <= var_95).any() else None
        cvar_99 = float(clean[clean <= var_99].mean()) if (clean <= var_99).any() else None
        return {
            "var_95": var_95,
            "cvar_95": cvar_95,
            "var_99": var_99,
            "cvar_99": cvar_99,
        }

    @staticmethod
    def _compute_mae_mfe(strategy_returns: pd.Series) -> dict[str, float | None]:
        clean = strategy_returns.dropna()
        if clean.empty:
            return {
                "mae_1d": None,
                "mfe_1d": None,
                "worst_5d": None,
                "best_5d": None,
            }

        rolling_5 = clean.rolling(5, min_periods=5).sum()
        return {
            "mae_1d": float(clean.min()),
            "mfe_1d": float(clean.max()),
            "worst_5d": float(rolling_5.min()) if not rolling_5.dropna().empty else None,
            "best_5d": float(rolling_5.max()) if not rolling_5.dropna().empty else None,
        }

    def _load_sector_mapping(self) -> dict[str, str]:
        sector_csv = self.paths.data_dir / "bist_sector_classification.csv"
        if not sector_csv.exists():
            return {}

        try:
            frame = pd.read_csv(sector_csv)
        except Exception:
            return {}

        if frame.empty or "ticker" not in frame.columns or "sector" not in frame.columns:
            return {}

        mapping: dict[str, str] = {}
        for _, row in frame.iterrows():
            ticker = str(row.get("ticker", "")).strip().upper()
            if not ticker:
                continue
            sector = str(row.get("sector", "")).strip() or "Unknown"
            mapping[ticker] = sector
        return mapping

    def _estimate_sector_exposure(self, top_holdings: list[dict[str, Any]]) -> dict[str, float]:
        if not top_holdings:
            return {}

        sector_map = self._load_sector_mapping()
        exposure: dict[str, float] = {}
        for row in top_holdings:
            ticker = str(row.get("ticker", "")).strip().upper()
            if not ticker:
                continue
            sector = sector_map.get(ticker, "Unknown")
            weight = float(row.get("weight", 0.0) or 0.0)
            if not math.isfinite(weight):
                continue
            exposure[sector] = exposure.get(sector, 0.0) + weight

        total = sum(exposure.values())
        if total > 0:
            exposure = {k: float(v / total) for k, v in exposure.items()}

        return dict(sorted(exposure.items(), key=lambda item: item[1], reverse=True))

    @staticmethod
    def _raise_if_cancelled(cancellation_check: CancelCheck | None) -> None:
        if cancellation_check is not None and cancellation_check():
            raise RuntimeError("BACKTEST_CANCELLED")

    @staticmethod
    def _progress(
        callback: ProgressCallback | None,
        percent: float,
        step: str,
        eta_seconds: int | None = None,
    ) -> None:
        if callback is None:
            return
        callback(float(percent), step, eta_seconds)

    @staticmethod
    def _factor_runs_from_request(
        factor_name: str | None,
        factor_names: list[str] | None,
        factors: list[dict[str, Any]] | None,
        factor_weights: dict[str, float] | None,
    ) -> list[FactorRunSpec]:
        ordered_names: list[str] = []
        seen: set[str] = set()

        def add_name(value: str | None) -> None:
            if not value:
                return
            candidate = str(value).strip()
            if not candidate:
                return
            if candidate in seen:
                return
            seen.add(candidate)
            ordered_names.append(candidate)

        add_name(factor_name)

        for value in factor_names or []:
            add_name(value)

        factor_specific_weights: dict[str, float] = {}
        factor_specific_params: dict[str, dict[str, Any]] = {}
        for item in factors or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            add_name(name)

            raw_weight = item.get("weight")
            if raw_weight is not None:
                weight = float(raw_weight)
                if weight <= 0:
                    raise ValueError(f"Factor weight must be > 0 for '{name}'")
                factor_specific_weights[name] = weight

            params = item.get("signal_params")
            if isinstance(params, dict):
                factor_specific_params[name] = dict(params)

        normalized_map: dict[str, float] = {}
        for key, value in (factor_weights or {}).items():
            candidate = str(key).strip()
            if not candidate:
                continue
            weight = float(value)
            if weight <= 0:
                raise ValueError(f"factor_weights value must be > 0 for '{candidate}'")
            normalized_map[candidate] = weight

        runs: list[FactorRunSpec] = []
        for name in ordered_names:
            if name in normalized_map:
                weight = normalized_map[name]
            elif name in factor_specific_weights:
                weight = factor_specific_weights[name]
            else:
                weight = 1.0

            runs.append(
                FactorRunSpec(
                    name=name,
                    weight=weight,
                    signal_params=factor_specific_params.get(name, {}),
                )
            )

        if not runs:
            raise ValueError("At least one factor must be provided")

        return runs

    @staticmethod
    def _normalized_weights(factor_runs: list[FactorRunSpec]) -> dict[str, float]:
        raw = {run.name: float(run.weight) for run in factor_runs}
        total = sum(raw.values())
        if total <= 0:
            raise ValueError("Total factor weight must be > 0")
        return {key: value / total for key, value in raw.items()}

    @staticmethod
    def _merge_portfolio_options(
        base_options: dict[str, Any],
        overrides: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(base_options)
        for key, value in overrides.items():
            if value is not None:
                merged[key] = value
        return merged

    @staticmethod
    @contextmanager
    def _temporary_env(name: str, value: str):
        previous = os.environ.get(name)
        os.environ[name] = value
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous

    @staticmethod
    def _api_stale_override_enabled() -> bool:
        raw = os.getenv("BIST_API_ALLOW_STALE_FUNDAMENTALS", "1").strip().lower()
        return raw not in {"0", "false", "no", "off"}

    def create_portfolio_engine(
        self,
        start_date: str = DEFAULT_ENGINE_START_DATE,
        end_date: str = DEFAULT_ENGINE_END_DATE,
    ) -> PortfolioEngine:
        return PortfolioEngine(
            data_dir=self.paths.data_dir,
            regime_model_dir=self.paths.regime_outputs_dir,
            start_date=start_date,
            end_date=end_date,
        )

    def load_prices(self, prices_path: Path | None = None) -> pd.DataFrame:
        loader = DataLoader(
            data_dir=self.paths.data_dir,
            regime_model_dir=self.paths.regime_outputs_dir,
        )

        if prices_path is None:
            resolved_path = self.paths.data_dir / "bist_prices_full.csv"
            if not resolved_path.exists() and not resolved_path.with_suffix(".parquet").exists():
                raise FileNotFoundError(
                    f"Price file not found: {resolved_path} or {resolved_path.with_suffix('.parquet')}"
                )
            return loader.load_prices(resolved_path)

        resolved_path = Path(prices_path).expanduser().resolve()
        if not resolved_path.exists() and not resolved_path.with_suffix(".parquet").exists():
            raise FileNotFoundError(
                f"Price file not found: {resolved_path} or {resolved_path.with_suffix('.parquet')}"
            )
        return loader.load_prices(resolved_path)

    def get_signal_details(self, signal_name: str) -> dict[str, Any]:
        normalized_name = self._normalize_factor_name(signal_name)
        signal_configs = self.load_signal_configs()
        config = signal_configs.get(normalized_name)
        if not isinstance(config, dict):
            raise ValueError(f"Unknown signal: {normalized_name}")

        timeline_raw = config.get("timeline", {})
        timeline = timeline_raw if isinstance(timeline_raw, dict) else {}

        signal_params_raw = config.get("signal_params", {})
        signal_params = signal_params_raw if isinstance(signal_params_raw, dict) else {}

        legacy_params_raw = config.get("parameters", {})
        legacy_params = legacy_params_raw if isinstance(legacy_params_raw, dict) else {}

        portfolio_options_raw = config.get("portfolio_options", {})
        portfolio_options = (
            portfolio_options_raw if isinstance(portfolio_options_raw, dict) else {}
        )

        parameter_keys = sorted(set(signal_params.keys()).union(set(legacy_params.keys())))

        return {
            "signal_name": normalized_name,
            "description": (
                str(config.get("description"))
                if isinstance(config.get("description"), str)
                else None
            ),
            "enabled": bool(config.get("enabled", True)),
            "rebalance_frequency": (
                str(config.get("rebalance_frequency"))
                if config.get("rebalance_frequency") is not None
                else None
            ),
            "timeline": {
                key: str(value)
                for key, value in timeline.items()
                if value is not None
            },
            "parameter_keys": parameter_keys,
            "signal_params": signal_params,
            "portfolio_options": portfolio_options,
        }

    def run_backtest(
        self,
        *,
        factor_name: str | None,
        start_date: str,
        end_date: str,
        factor_names: list[str] | None = None,
        factors: list[dict[str, Any]] | None = None,
        factor_weights: dict[str, float] | None = None,
        rebalance_frequency: str | None = None,
        top_n: int | None = None,
        signal_lag_days: int | None = None,
        use_stop_loss: bool | None = None,
        stop_loss_threshold: float | None = None,
        use_vol_targeting: bool | None = None,
        target_downside_vol: float | None = None,
        vol_lookback: int | None = None,
        vol_floor: float | None = None,
        vol_cap: float | None = None,
        use_regime_filter: bool | None = None,
        use_liquidity_filter: bool | None = None,
        use_inverse_vol_sizing: bool | None = None,
        max_position_weight: float | None = None,
        use_slippage: bool | None = None,
        slippage_bps: float | None = None,
        benchmark: str | None = "XU100",
        progress_callback: ProgressCallback | None = None,
        cancellation_check: CancelCheck | None = None,
    ) -> dict[str, Any]:
        start_date_str = self._to_date_string(start_date)
        end_date_str = self._to_date_string(end_date)

        start_ts = pd.Timestamp(start_date_str)
        end_ts = pd.Timestamp(end_date_str)
        if end_ts <= start_ts:
            raise ValueError("end_date must be after start_date")

        factor_runs = self._factor_runs_from_request(
            factor_name=factor_name,
            factor_names=factor_names,
            factors=factors,
            factor_weights=factor_weights,
        )
        normalized_weights = self._normalized_weights(factor_runs)

        available_signals = self.load_signal_configs()
        missing = [run.name for run in factor_runs if run.name not in available_signals]
        if missing:
            available = ", ".join(self.list_available_signals()[:20])
            missing_label = ", ".join(missing)
            raise ValueError(
                f"Unknown factor_name(s) '{missing_label}'. Available signals (first 20): {available}"
            )

        portfolio_overrides = {
            "top_n": top_n,
            "signal_lag_days": signal_lag_days,
            "use_stop_loss": use_stop_loss,
            "stop_loss_threshold": stop_loss_threshold,
            "use_vol_targeting": use_vol_targeting,
            "target_downside_vol": target_downside_vol,
            "vol_lookback": vol_lookback,
            "vol_floor": vol_floor,
            "vol_cap": vol_cap,
            "use_regime_filter": use_regime_filter,
            "use_liquidity_filter": use_liquidity_filter,
            "use_inverse_vol_sizing": use_inverse_vol_sizing,
            "max_position_weight": max_position_weight,
            "use_slippage": use_slippage,
            "slippage_bps": slippage_bps,
        }

        self._raise_if_cancelled(cancellation_check)
        self._progress(progress_callback, 5.0, "Initializing portfolio engine", None)

        if self._api_stale_override_enabled() and "BIST_ALLOW_STALE_FUNDAMENTALS" not in os.environ:
            with self._temporary_env("BIST_ALLOW_STALE_FUNDAMENTALS", "1"):
                engine = self.create_portfolio_engine(start_date=start_date_str, end_date=end_date_str)
                self._progress(progress_callback, 15.0, "Loading market and fundamentals data", None)
                engine.load_all_data()
        else:
            engine = self.create_portfolio_engine(start_date=start_date_str, end_date=end_date_str)
            self._progress(progress_callback, 15.0, "Loading market and fundamentals data", None)
            engine.load_all_data()

        self._raise_if_cancelled(cancellation_check)

        raw_results: dict[str, dict[str, Any]] = {}
        factor_returns_map: dict[str, pd.Series] = {}

        total_factors = len(factor_runs)
        for idx, run in enumerate(factor_runs):
            self._raise_if_cancelled(cancellation_check)

            start_pct = 25.0 + (idx / total_factors) * 55.0
            end_pct = 25.0 + ((idx + 1) / total_factors) * 55.0
            self._progress(
                progress_callback,
                start_pct,
                f"Running factor {idx + 1}/{total_factors}: {run.name}",
                None,
            )

            base_config = engine.signal_configs.get(run.name)
            if not isinstance(base_config, dict):
                raise ValueError(f"No runtime signal config found for factor '{run.name}'")

            override_config = copy.deepcopy(base_config)
            timeline = dict(override_config.get("timeline", {}))
            timeline["start_date"] = start_date_str
            timeline["end_date"] = end_date_str
            override_config["timeline"] = timeline

            if rebalance_frequency:
                override_config["rebalance_frequency"] = str(rebalance_frequency)

            base_options = override_config.get("portfolio_options", {})
            if not isinstance(base_options, dict):
                base_options = {}
            override_config["portfolio_options"] = self._merge_portfolio_options(
                base_options=base_options,
                overrides=portfolio_overrides,
            )

            if run.signal_params:
                existing_signal_params = override_config.get("signal_params", {})
                if not isinstance(existing_signal_params, dict):
                    existing_signal_params = {}
                existing_signal_params.update(run.signal_params)
                override_config["signal_params"] = existing_signal_params

            raw_result = engine.run_factor(run.name, override_config=override_config)
            if not isinstance(raw_result, dict):
                raise RuntimeError(f"Backtest returned no payload for factor '{run.name}'")

            raw_results[run.name] = raw_result

            returns_series = raw_result.get("returns")
            if isinstance(returns_series, pd.Series):
                factor_returns_map[run.name] = returns_series.astype(float)

            self._progress(progress_callback, end_pct, f"Completed factor: {run.name}", None)

        self._raise_if_cancelled(cancellation_check)
        self._progress(progress_callback, 85.0, "Aggregating multi-factor performance", None)

        factor_returns = pd.DataFrame(factor_returns_map).sort_index()
        if factor_returns.empty:
            raise RuntimeError("No factor return series were produced by the backtest")

        factor_returns = factor_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        weight_series = pd.Series(normalized_weights).reindex(factor_returns.columns).fillna(0.0)
        strategy_returns = factor_returns.mul(weight_series, axis=1).sum(axis=1)
        strategy_returns = strategy_returns.replace([np.inf, -np.inf], np.nan).dropna()
        if strategy_returns.empty:
            raise RuntimeError("Backtest produced an empty strategy return series")

        benchmark_key = str(benchmark or "XU100").strip().upper() or "XU100"
        benchmark_returns: pd.Series | None = None
        if benchmark_key == "XU100" and engine.xu100_prices is not None:
            benchmark_returns = (engine.xu100_prices.shift(-1) / engine.xu100_prices) - 1.0
            benchmark_returns = benchmark_returns.reindex(strategy_returns.index)
        elif benchmark_key == "XAUTRY":
            first_key = factor_runs[0].name
            xautry = raw_results[first_key].get("xautry_returns")
            if isinstance(xautry, pd.Series):
                benchmark_returns = xautry.reindex(strategy_returns.index)

        metrics = self._compute_metrics(strategy_returns, benchmark_returns)

        strategy_equity = (1.0 + strategy_returns).cumprod()
        strategy_drawdown = strategy_equity / strategy_equity.cummax() - 1.0

        benchmark_equity = None
        if benchmark_returns is not None and not benchmark_returns.empty:
            bench_clean = benchmark_returns.fillna(0.0)
            benchmark_equity = (1.0 + bench_clean).cumprod().reindex(strategy_equity.index)

        self._raise_if_cancelled(cancellation_check)
        self._progress(progress_callback, 92.0, "Building analytics payload", None)

        top_holdings_map: dict[str, float] = {}
        for run in factor_runs:
            latest_holdings = self._extract_latest_holdings(raw_results[run.name].get("holdings_history"))
            run_weight = normalized_weights.get(run.name, 0.0)
            for row in latest_holdings:
                ticker = str(row["ticker"])
                holding_weight = float(row["weight"])
                top_holdings_map[ticker] = top_holdings_map.get(ticker, 0.0) + run_weight * holding_weight

        top_holdings = [
            {"ticker": ticker, "weight": weight}
            for ticker, weight in sorted(top_holdings_map.items(), key=lambda item: item[1], reverse=True)
        ]
        top_holdings = top_holdings[:10]

        top_holdings_total = sum(float(item["weight"]) for item in top_holdings)
        if top_holdings_total > 0:
            for item in top_holdings:
                item["weight"] = float(item["weight"]) / top_holdings_total

        rebalance_count = int(sum(int(raw.get("rebalance_count", 0)) for raw in raw_results.values()))
        trade_count = int(sum(int(raw.get("trade_count", 0)) for raw in raw_results.values()))

        summary = {
            "factor_count": len(factor_runs),
            "trading_days": int(len(strategy_returns)),
            "total_return": metrics.get("total_return"),
            "cagr": metrics.get("cagr"),
            "sharpe": metrics.get("sharpe"),
            "max_drawdown": metrics.get("max_drawdown"),
            "benchmark": benchmark_key,
        }

        risk_metrics = {
            "tail_risk": self._compute_tail_risk(strategy_returns),
            "mae_mfe": self._compute_mae_mfe(strategy_returns),
        }

        scenario_analysis = {
            "stress_1d_minus_2sigma": float(strategy_returns.mean() - 2.0 * strategy_returns.std()),
            "stress_1d_minus_3sigma": float(strategy_returns.mean() - 3.0 * strategy_returns.std()),
            "best_day": float(strategy_returns.max()),
            "worst_day": float(strategy_returns.min()),
        }

        payload = {
            "factor_name": factor_runs[0].name if len(factor_runs) == 1 else "multi_factor",
            "factor_names": [run.name for run in factor_runs],
            "benchmark": benchmark_key,
            "start_date": start_date_str,
            "end_date": end_date_str,
            "metrics": metrics,
            "summary": summary,
            "equity_curve": self._serialize_equity_curve(
                strategy_equity,
                benchmark_equity,
                strategy_drawdown,
            ),
            "drawdown_curve": self._serialize_drawdown_curve(strategy_drawdown),
            "monthly_returns": self._serialize_monthly_returns(strategy_returns, benchmark_returns),
            "rolling_metrics": self._serialize_rolling_metrics(strategy_returns),
            "performance_attribution": self._build_performance_attribution(factor_returns, normalized_weights),
            "top_holdings": top_holdings,
            "rebalance_count": rebalance_count,
            "trade_count": trade_count,
            "factor_breakdown": self._build_factor_breakdown(raw_results, normalized_weights),
            "daily_returns": self._serialize_daily_returns(strategy_returns, benchmark_returns),
            "sector_exposure": self._estimate_sector_exposure(top_holdings),
            "risk_metrics": risk_metrics,
            "scenario_analysis": scenario_analysis,
        }

        self._progress(progress_callback, 100.0, "Backtest completed", 0)
        return payload

    @staticmethod
    def _extract_strategy_series_from_payload(payload: dict[str, Any]) -> pd.Series:
        rows = payload.get("daily_returns")
        if not isinstance(rows, list):
            return pd.Series(dtype="float64")

        data: list[tuple[pd.Timestamp, float]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            date_raw = row.get("date")
            value_raw = row.get("strategy_return")
            if not isinstance(date_raw, str):
                continue
            if not isinstance(value_raw, (int, float)):
                continue
            ts = pd.to_datetime(date_raw, errors="coerce")
            if pd.isna(ts):
                continue
            data.append((pd.Timestamp(ts), float(value_raw)))

        if not data:
            return pd.Series(dtype="float64")

        index = [item[0] for item in data]
        values = [item[1] for item in data]
        return pd.Series(values, index=index, dtype="float64").sort_index()

    @staticmethod
    def _normalize_weight_map(weight_map: dict[str, float]) -> dict[str, float]:
        cleaned = {
            str(name): float(value)
            for name, value in weight_map.items()
            if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0.0
        }
        total = sum(cleaned.values())
        if total <= 0:
            return {}
        return {name: float(value / total) for name, value in cleaned.items()}

    @staticmethod
    def _compute_covariance_weights(
        returns_df: pd.DataFrame,
        scheme: str,
    ) -> dict[str, float]:
        if returns_df.empty:
            return {}

        columns = list(returns_df.columns)
        n = len(columns)
        if n == 0:
            return {}
        if n == 1:
            return {columns[0]: 1.0}

        cov = returns_df.cov().replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        eps = 1e-8
        cov += np.eye(n) * eps

        if scheme == "risk_parity":
            vol = np.sqrt(np.clip(np.diag(cov), eps, None))
            inv_vol = 1.0 / vol
            raw = inv_vol / inv_vol.sum()
            return {columns[i]: float(raw[i]) for i in range(n)}

        inv_cov = np.linalg.pinv(cov)
        ones = np.ones(n, dtype=float)

        if scheme == "min_variance":
            raw = inv_cov @ ones
        else:
            mu = returns_df.mean().to_numpy(dtype=float) * 252.0
            raw = inv_cov @ mu

        raw = np.clip(raw, 0.0, None)
        total = float(raw.sum())
        if total <= 0:
            raw = np.ones(n, dtype=float) / float(n)
        else:
            raw = raw / total
        return {columns[i]: float(raw[i]) for i in range(n)}

    @staticmethod
    def _compute_risk_contribution(returns_df: pd.DataFrame, weights: dict[str, float]) -> dict[str, float]:
        if returns_df.empty or not weights:
            return {}

        cols = [col for col in returns_df.columns if col in weights]
        if not cols:
            return {}

        cov = returns_df[cols].cov().replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        w = np.array([weights[col] for col in cols], dtype=float)
        portfolio_var = float(w.T @ cov @ w)
        if portfolio_var <= 0:
            return {col: 0.0 for col in cols}

        marginal = cov @ w
        contrib = w * marginal / portfolio_var
        return {cols[i]: float(contrib[i]) for i in range(len(cols))}

    @staticmethod
    def _compute_factor_contribution(returns_df: pd.DataFrame, weights: dict[str, float]) -> dict[str, float]:
        if returns_df.empty or not weights:
            return {}

        out: dict[str, float] = {}
        for col in returns_df.columns:
            if col not in weights:
                continue
            series = returns_df[col].dropna()
            if series.empty:
                continue
            weighted = series * float(weights[col])
            out[col] = float((1.0 + weighted).prod() - 1.0)
        return out

    @staticmethod
    def _compute_timing_signals(
        returns_df: pd.DataFrame,
        lookback: int,
        threshold: float,
    ) -> dict[str, Any]:
        if returns_df.empty:
            return {
                "latest": {},
                "history": [],
            }

        latest: dict[str, bool] = {}
        history: list[dict[str, Any]] = []

        rolling_mean = returns_df.rolling(lookback, min_periods=max(10, lookback // 3)).mean()
        rolling_std = returns_df.rolling(lookback, min_periods=max(10, lookback // 3)).std()
        rolling_sharpe = (rolling_mean / rolling_std.replace(0.0, np.nan)) * np.sqrt(252)

        for name in returns_df.columns:
            series = rolling_sharpe[name].dropna()
            latest[name] = bool(series.iloc[-1] > threshold) if not series.empty else True

        sampled = rolling_sharpe.tail(120)
        for idx, row in sampled.iterrows():
            row_payload: dict[str, Any] = {"date": str(pd.Timestamp(idx).date())}
            for name in returns_df.columns:
                value = row.get(name)
                row_payload[name] = bool(value > threshold) if pd.notna(value) else False
            history.append(row_payload)

        return {
            "latest": latest,
            "history": history,
        }

    @staticmethod
    def _set_path_value(target: Any, path: str, value: Any) -> None:
        parts = [part for part in str(path).split(".") if part]
        if not parts:
            return

        cursor = target
        for idx, part in enumerate(parts):
            is_last = idx == len(parts) - 1
            next_is_index = (idx + 1 < len(parts)) and parts[idx + 1].isdigit()

            if part.isdigit():
                index = int(part)
                if not isinstance(cursor, list):
                    return
                while len(cursor) <= index:
                    cursor.append([] if next_is_index else {})
                if is_last:
                    cursor[index] = value
                    return
                next_cursor = cursor[index]
                if next_is_index and not isinstance(next_cursor, list):
                    cursor[index] = []
                elif not next_is_index and not isinstance(next_cursor, dict):
                    cursor[index] = {}
                cursor = cursor[index]
                continue

            if not isinstance(cursor, dict):
                return
            if is_last:
                cursor[part] = value
                return

            if part not in cursor or cursor[part] is None:
                cursor[part] = [] if next_is_index else {}
            elif next_is_index and not isinstance(cursor[part], list):
                cursor[part] = []
            elif not next_is_index and not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]

    @staticmethod
    def _parameter_candidates(
        spec: dict[str, Any],
        method: str,
        rng: random.Random,
    ) -> list[Any]:
        raw_values = spec.get("values")
        if isinstance(raw_values, list) and len(raw_values) > 0:
            return raw_values

        ptype = str(spec.get("type", "float")).lower()
        min_value = spec.get("min")
        max_value = spec.get("max")
        step_value = spec.get("step")

        if ptype == "bool":
            return [False, True]

        if min_value is None or max_value is None:
            return []

        lower = float(min_value)
        upper = float(max_value)
        if upper < lower:
            lower, upper = upper, lower

        if method == "grid":
            if step_value is not None and float(step_value) > 0:
                step = float(step_value)
                values: list[float] = []
                current = lower
                max_points = 50
                while current <= upper + 1e-12 and len(values) < max_points:
                    values.append(float(current))
                    current += step
                if len(values) == 0:
                    values = [lower, upper]
                return values
            return np.linspace(lower, upper, 6).tolist()

        if ptype == "int":
            if step_value is not None and int(step_value) > 0:
                step = int(step_value)
                values = list(range(int(round(lower)), int(round(upper)) + 1, step))
                return values if values else [int(round(lower))]
            return [int(round(rng.uniform(lower, upper)))]

        return [float(rng.uniform(lower, upper))]

    @staticmethod
    def _cast_parameter_value(value: Any, ptype: str) -> Any:
        normalized = ptype.lower()
        if normalized == "int":
            return int(round(float(value)))
        if normalized == "float":
            return float(value)
        if normalized == "bool":
            return bool(value)
        return value

    @staticmethod
    def _score_trial(metrics: dict[str, Any], objective: dict[str, Any]) -> float:
        sharpe = float(metrics.get("sharpe", 0.0) or 0.0)
        cagr = float(metrics.get("cagr", 0.0) or 0.0)
        total_return = float(metrics.get("total_return", 0.0) or 0.0)
        max_drawdown = abs(float(metrics.get("max_drawdown", 0.0) or 0.0))
        volatility = float(metrics.get("annualized_volatility", 0.0) or 0.0)

        return (
            float(objective.get("weight_sharpe", 1.0)) * sharpe
            + float(objective.get("weight_cagr", 0.5)) * cagr
            + float(objective.get("weight_total_return", 0.25)) * total_return
            - float(objective.get("weight_max_drawdown", 1.0)) * max_drawdown
            - float(objective.get("weight_volatility", 0.25)) * volatility
        )

    @staticmethod
    def _constraints_satisfied(metrics: dict[str, Any], constraints: dict[str, Any]) -> bool:
        max_drawdown = constraints.get("max_drawdown")
        if isinstance(max_drawdown, (int, float)):
            if abs(float(metrics.get("max_drawdown", 0.0) or 0.0)) > float(max_drawdown):
                return False

        max_volatility = constraints.get("max_volatility")
        if isinstance(max_volatility, (int, float)):
            if float(metrics.get("annualized_volatility", 0.0) or 0.0) > float(max_volatility):
                return False

        min_sharpe = constraints.get("min_sharpe")
        if isinstance(min_sharpe, (int, float)):
            if float(metrics.get("sharpe", 0.0) or 0.0) < float(min_sharpe):
                return False

        min_cagr = constraints.get("min_cagr")
        if isinstance(min_cagr, (int, float)):
            if float(metrics.get("cagr", 0.0) or 0.0) < float(min_cagr):
                return False

        return True

    @staticmethod
    def _pareto_front(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
        candidates = [
            trial for trial in trials
            if isinstance(trial.get("metrics"), dict) and trial.get("feasible") is True
        ]
        front: list[dict[str, Any]] = []
        for trial in candidates:
            metrics = trial["metrics"]
            sharpe = float(metrics.get("sharpe", 0.0) or 0.0)
            drawdown = abs(float(metrics.get("max_drawdown", 0.0) or 0.0))
            dominated = False
            for other in candidates:
                if other is trial:
                    continue
                om = other["metrics"]
                other_sharpe = float(om.get("sharpe", 0.0) or 0.0)
                other_drawdown = abs(float(om.get("max_drawdown", 0.0) or 0.0))
                if (other_sharpe >= sharpe and other_drawdown <= drawdown) and (
                    other_sharpe > sharpe or other_drawdown < drawdown
                ):
                    dominated = True
                    break
            if not dominated:
                front.append(trial)
        front.sort(key=lambda item: float(item.get("score", float("-inf"))), reverse=True)
        return front

    def _run_backtest_from_request(self, request_data: dict[str, Any]) -> dict[str, Any]:
        return self.run_backtest(
            factor_name=request_data.get("factor_name"),
            factor_names=request_data.get("factor_names"),
            factors=request_data.get("factors"),
            factor_weights=request_data.get("factor_weights"),
            start_date=str(request_data["start_date"]),
            end_date=str(request_data["end_date"]),
            rebalance_frequency=request_data.get("rebalance_frequency"),
            top_n=request_data.get("top_n"),
            signal_lag_days=request_data.get("signal_lag_days"),
            use_stop_loss=request_data.get("use_stop_loss"),
            stop_loss_threshold=request_data.get("stop_loss_threshold"),
            use_vol_targeting=request_data.get("use_vol_targeting"),
            target_downside_vol=request_data.get("target_downside_vol"),
            vol_lookback=request_data.get("vol_lookback"),
            vol_floor=request_data.get("vol_floor"),
            vol_cap=request_data.get("vol_cap"),
            use_regime_filter=request_data.get("use_regime_filter"),
            use_liquidity_filter=request_data.get("use_liquidity_filter"),
            use_inverse_vol_sizing=request_data.get("use_inverse_vol_sizing"),
            max_position_weight=request_data.get("max_position_weight"),
            use_slippage=request_data.get("use_slippage"),
            slippage_bps=request_data.get("slippage_bps"),
            benchmark=request_data.get("benchmark"),
            progress_callback=None,
            cancellation_check=None,
        )

    def combine_factors(
        self,
        *,
        factors: list[dict[str, Any]],
        start_date: str,
        end_date: str,
        weighting_scheme: str = "custom",
        timing_enabled: bool = False,
        timing_lookback: int = 63,
        timing_threshold: float = 0.0,
        benchmark: str | None = "XU100",
        rebalance_frequency: str | None = None,
        top_n: int | None = None,
        signal_lag_days: int | None = None,
        use_stop_loss: bool | None = None,
        stop_loss_threshold: float | None = None,
        use_vol_targeting: bool | None = None,
        target_downside_vol: float | None = None,
        vol_lookback: int | None = None,
        vol_floor: float | None = None,
        vol_cap: float | None = None,
        use_regime_filter: bool | None = None,
        use_liquidity_filter: bool | None = None,
        use_inverse_vol_sizing: bool | None = None,
        max_position_weight: float | None = None,
        use_slippage: bool | None = None,
        slippage_bps: float | None = None,
    ) -> dict[str, Any]:
        factor_runs = self._factor_runs_from_request(
            factor_name=None,
            factor_names=None,
            factors=factors,
            factor_weights=None,
        )
        custom_weights = self._normalized_weights(factor_runs)

        shared_payload = {
            "start_date": start_date,
            "end_date": end_date,
            "benchmark": benchmark,
            "rebalance_frequency": rebalance_frequency,
            "top_n": top_n,
            "signal_lag_days": signal_lag_days,
            "use_stop_loss": use_stop_loss,
            "stop_loss_threshold": stop_loss_threshold,
            "use_vol_targeting": use_vol_targeting,
            "target_downside_vol": target_downside_vol,
            "vol_lookback": vol_lookback,
            "vol_floor": vol_floor,
            "vol_cap": vol_cap,
            "use_regime_filter": use_regime_filter,
            "use_liquidity_filter": use_liquidity_filter,
            "use_inverse_vol_sizing": use_inverse_vol_sizing,
            "max_position_weight": max_position_weight,
            "use_slippage": use_slippage,
            "slippage_bps": slippage_bps,
        }

        factor_series_map: dict[str, pd.Series] = {}
        for run in factor_runs:
            single_payload = {
                **shared_payload,
                "factor_name": None,
                "factor_names": None,
                "factor_weights": None,
                "factors": [
                    {
                        "name": run.name,
                        "weight": 1.0,
                        "signal_params": run.signal_params,
                    }
                ],
            }
            result = self._run_backtest_from_request(single_payload)
            factor_series_map[run.name] = self._extract_strategy_series_from_payload(result)

        factor_returns = pd.DataFrame(factor_series_map).sort_index().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if factor_returns.empty:
            raise ValueError("Failed to build factor return matrix for combination.")

        candidate_weights: dict[str, dict[str, float]] = {
            "custom": custom_weights,
            "equal": self._normalize_weight_map({name: 1.0 for name in factor_returns.columns}),
            "risk_parity": self._compute_covariance_weights(factor_returns, "risk_parity"),
            "mean_variance": self._compute_covariance_weights(factor_returns, "mean_variance"),
            "min_variance": self._compute_covariance_weights(factor_returns, "min_variance"),
        }

        scheme = str(weighting_scheme or "custom").strip().lower()
        if scheme not in candidate_weights:
            scheme = "custom"

        selected_weights = dict(candidate_weights[scheme])
        timing_signals = self._compute_timing_signals(factor_returns, lookback=timing_lookback, threshold=timing_threshold)
        if timing_enabled and timing_signals.get("latest"):
            latest = timing_signals["latest"]
            for name in list(selected_weights.keys()):
                if name in latest and latest[name] is False:
                    selected_weights[name] = 0.0
            selected_weights = self._normalize_weight_map(selected_weights)

        combined_factors: list[dict[str, Any]] = []
        for run in factor_runs:
            weight = float(selected_weights.get(run.name, 0.0))
            if weight <= 0:
                continue
            combined_factors.append(
                {
                    "name": run.name,
                    "weight": weight,
                    "signal_params": run.signal_params,
                }
            )

        if len(combined_factors) == 0:
            combined_factors = [
                {
                    "name": run.name,
                    "weight": float(selected_weights.get(run.name, 0.0)),
                    "signal_params": run.signal_params,
                }
                for run in factor_runs
            ]
            combined_factors = [item for item in combined_factors if item["weight"] > 0]
            if len(combined_factors) == 0:
                raise ValueError("No active factors remain after timing filter.")

        combined_payload = {
            **shared_payload,
            "factor_name": None,
            "factor_names": None,
            "factor_weights": None,
            "factors": combined_factors,
        }
        combined_result = self._run_backtest_from_request(combined_payload)

        corr_matrix = factor_returns.corr().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        corr_payload: dict[str, dict[str, float]] = {}
        for row_name in corr_matrix.index:
            corr_payload[str(row_name)] = {
                str(col_name): float(corr_matrix.loc[row_name, col_name])
                for col_name in corr_matrix.columns
            }

        return {
            "status": "ok",
            "weighting_scheme": scheme,
            "optimized_weights": selected_weights,
            "factor_correlation": corr_payload,
            "factor_contribution": self._compute_factor_contribution(factor_returns, selected_weights),
            "risk_contribution": self._compute_risk_contribution(factor_returns, selected_weights),
            "sector_exposure": combined_result.get("sector_exposure", {}),
            "timing_signals": timing_signals,
            "optimization_candidates": candidate_weights,
            "backtest": combined_result,
        }

    def optimize_strategy(
        self,
        *,
        base_request: dict[str, Any],
        method: str = "grid",
        parameter_space: list[dict[str, Any]] | None = None,
        max_trials: int = 50,
        random_seed: int | None = None,
        train_ratio: float = 0.7,
        walk_forward_splits: int = 0,
        constraints: dict[str, Any] | None = None,
        objective: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(base_request, dict):
            raise ValueError("base_request must be a dictionary")
        if not parameter_space:
            raise ValueError("parameter_space must include at least one parameter")

        method_normalized = str(method or "grid").strip().lower()
        if method_normalized not in {"grid", "random"}:
            raise ValueError("method must be one of: grid, random")

        constraints_dict = constraints if isinstance(constraints, dict) else {}
        objective_dict = objective if isinstance(objective, dict) else {}

        rng = random.Random(random_seed)
        specs = [dict(spec) for spec in parameter_space if isinstance(spec, dict) and spec.get("key")]
        if not specs:
            raise ValueError("parameter_space entries are invalid")

        trial_sets: list[dict[str, Any]] = []
        if method_normalized == "grid":
            candidate_lists: list[list[Any]] = []
            for spec in specs:
                candidates = self._parameter_candidates(spec, "grid", rng)
                if len(candidates) == 0:
                    raise ValueError(f"No candidates generated for parameter '{spec.get('key')}'")
                candidate_lists.append(candidates)
            for values in itertools.product(*candidate_lists):
                params: dict[str, Any] = {}
                for spec, value in zip(specs, values):
                    ptype = str(spec.get("type", "float"))
                    params[str(spec["key"])] = self._cast_parameter_value(value, ptype)
                trial_sets.append(params)
                if len(trial_sets) >= max_trials:
                    break
        else:
            for _ in range(max_trials):
                params: dict[str, Any] = {}
                for spec in specs:
                    candidates = self._parameter_candidates(spec, "random", rng)
                    if not candidates:
                        continue
                    picked = candidates[0] if len(candidates) == 1 else rng.choice(candidates)
                    ptype = str(spec.get("type", "float"))
                    params[str(spec["key"])] = self._cast_parameter_value(picked, ptype)
                trial_sets.append(params)

        if len(trial_sets) == 0:
            raise ValueError("No trial parameter sets generated")

        trials: list[dict[str, Any]] = []
        feasible_count = 0
        best_trial: dict[str, Any] | None = None

        start_ts = pd.Timestamp(str(base_request.get("start_date")))
        end_ts = pd.Timestamp(str(base_request.get("end_date")))
        if pd.isna(start_ts) or pd.isna(end_ts) or end_ts <= start_ts:
            raise ValueError("base_request must include a valid start_date and end_date")

        all_days = pd.bdate_range(start_ts, end_ts)
        split_index = max(1, min(len(all_days) - 1, int(len(all_days) * float(train_ratio))))
        split_date = all_days[split_index - 1]
        validation_start = all_days[split_index]

        for trial_id, params in enumerate(trial_sets, start=1):
            trial_payload = copy.deepcopy(base_request)
            trial_payload["async_mode"] = False

            for key, value in params.items():
                self._set_path_value(trial_payload, key, value)

            try:
                train_payload = copy.deepcopy(trial_payload)
                train_payload["start_date"] = str(start_ts.date())
                train_payload["end_date"] = str(split_date.date())

                validation_payload = copy.deepcopy(trial_payload)
                validation_payload["start_date"] = str(validation_start.date())
                validation_payload["end_date"] = str(end_ts.date())

                train_result = self._run_backtest_from_request(train_payload)
                validation_result = self._run_backtest_from_request(validation_payload)

                train_metrics = dict(train_result.get("metrics", {}))
                validation_metrics = dict(validation_result.get("metrics", {}))
                score = self._score_trial(validation_metrics, objective_dict)
                feasible = self._constraints_satisfied(validation_metrics, constraints_dict)
                if feasible:
                    feasible_count += 1

                trial = {
                    "trial_id": trial_id,
                    "params": params,
                    "feasible": feasible,
                    "score": float(score),
                    "metrics": validation_metrics,
                    "train_metrics": train_metrics,
                    "validation_metrics": validation_metrics,
                    "error": None,
                }
            except Exception as exc:  # pragma: no cover - safeguard
                trial = {
                    "trial_id": trial_id,
                    "params": params,
                    "feasible": False,
                    "score": None,
                    "metrics": {},
                    "train_metrics": {},
                    "validation_metrics": {},
                    "error": str(exc),
                }

            trials.append(trial)

            if trial["score"] is not None:
                if best_trial is None:
                    best_trial = trial
                else:
                    current_score = float(trial["score"])
                    best_score = float(best_trial.get("score", float("-inf")) or float("-inf"))
                    if trial["feasible"] and not best_trial.get("feasible", False):
                        best_trial = trial
                    elif trial["feasible"] == best_trial.get("feasible", False) and current_score > best_score:
                        best_trial = trial

        if best_trial is None:
            best_trial = {
                "trial_id": 0,
                "params": {},
                "feasible": False,
                "score": None,
                "metrics": {},
                "train_metrics": {},
                "validation_metrics": {},
                "error": "No successful trials",
            }

        walk_forward: list[dict[str, Any]] = []
        scenario_analysis: dict[str, Any] = {}
        if walk_forward_splits > 0 and best_trial.get("params"):
            params = dict(best_trial["params"])
            n = len(all_days)
            split_size = max(20, n // (walk_forward_splits + 1))
            for split_idx in range(walk_forward_splits):
                test_start_idx = split_size * (split_idx + 1)
                test_end_idx = min(n - 1, test_start_idx + split_size - 1)
                if test_start_idx >= n or test_end_idx <= test_start_idx:
                    continue

                split_payload = copy.deepcopy(base_request)
                split_payload["async_mode"] = False
                for key, value in params.items():
                    self._set_path_value(split_payload, key, value)
                split_payload["start_date"] = str(all_days[test_start_idx].date())
                split_payload["end_date"] = str(all_days[test_end_idx].date())

                try:
                    split_result = self._run_backtest_from_request(split_payload)
                    split_metrics = dict(split_result.get("metrics", {}))
                except Exception as exc:  # pragma: no cover - safeguard
                    split_metrics = {"error": str(exc)}

                walk_forward.append(
                    {
                        "split": split_idx + 1,
                        "start_date": split_payload["start_date"],
                        "end_date": split_payload["end_date"],
                        "metrics": split_metrics,
                    }
                )

        if best_trial.get("params"):
            final_payload = copy.deepcopy(base_request)
            final_payload["async_mode"] = False
            for key, value in dict(best_trial["params"]).items():
                self._set_path_value(final_payload, key, value)
            best_result = self._run_backtest_from_request(final_payload)
            scenario_analysis = dict(best_result.get("scenario_analysis", {}))
            best_trial = dict(best_trial)
            best_trial["backtest"] = best_result

        pareto = self._pareto_front(trials)
        trials_sorted = sorted(
            trials,
            key=lambda item: float(item.get("score", float("-inf")) or float("-inf")),
            reverse=True,
        )

        return {
            "status": "ok",
            "method": method_normalized,
            "total_trials": len(trials),
            "feasible_trials": feasible_count,
            "best_trial": best_trial,
            "pareto_front": pareto[:30],
            "trials": trials_sorted[:200],
            "constraints": constraints_dict,
            "objective": objective_dict,
            "walk_forward": walk_forward,
            "scenario_analysis": scenario_analysis,
        }

    def collect_system_info(self) -> dict[str, object]:
        signal_configs = self.load_signal_configs()
        available_signals = sorted(signal_configs.keys())
        portfolio_engine = self.create_portfolio_engine()

        prices_csv = self.paths.data_dir / "bist_prices_full.csv"
        prices_parquet = prices_csv.with_suffix(".parquet")

        return {
            "project_root": str(self.paths.project_root),
            "data_dir": str(self.paths.data_dir),
            "regime_outputs_dir": str(self.paths.regime_outputs_dir),
            "signal_count": len(available_signals),
            "available_signals": available_signals,
            "portfolio_engine_accessible": isinstance(portfolio_engine, PortfolioEngine),
            "portfolio_engine_signal_count": len(portfolio_engine.signal_configs),
            "prices_csv_exists": prices_csv.exists(),
            "prices_parquet_exists": prices_parquet.exists(),
        }
