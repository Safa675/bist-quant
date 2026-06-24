"""Bridge ``FactorSignal`` subclasses to the ``SignalBuilder`` protocol.

Research builders are intentionally **not** merged into ``factory.BUILDERS``.
Use ``build_research_signal()`` for ablation / notebook workflows; keep
``build_signal()`` for production backtests and ``strategies.yaml``.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Type

import pandas as pd

from bist_quant.signals._context import get_runtime_context, require_context
from bist_quant.signals.protocol import SignalBuilder
from bist_quant.signals.standalone_factors.base import (
    FactorData,
    FactorParams,
    FactorSignal,
    NormalizationMethod,
)
from bist_quant.signals.standalone_factors.defensive_signal import LowVolatilitySignal
from bist_quant.signals.standalone_factors.momentum_signal import (
    MomentumSignal,
    VolatilityAdjustedMomentumSignal,
)

ConfigDict = dict[str, Any]
ScoreOutput = Literal["raw", "normalized"]


def factor_data_from_config(
    signal_name: str,
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
) -> FactorData:
    """Build ``FactorData`` from backtest runtime context."""
    context = get_runtime_context(config)
    close_df = require_context(signal_name, context, "close_df")
    volume_df = context.get("volume_df")
    if not isinstance(volume_df, pd.DataFrame):
        volume_df = None

    fundamentals = context.get("fundamentals")
    if not isinstance(fundamentals, dict):
        fundamentals = None

    return FactorData(
        close=close_df.reindex(index=dates),
        volume=volume_df.reindex(index=dates) if volume_df is not None else None,
        fundamentals=fundamentals,
        dates=dates,
        tickers=pd.Index(close_df.columns),
        data_loader=loader,
    )


def factor_params_from_signal_params(
    signal_cls: Type[FactorSignal],
    signal_params: dict[str, Any],
) -> FactorParams:
    """Merge class defaults with registry ``signal_params``."""
    instance = signal_cls()
    params = instance.get_default_params() if hasattr(instance, "get_default_params") else FactorParams()
    merged_custom = dict(params.custom)
    merged_custom.update(signal_params or {})

    norm = merged_custom.pop("normalization", None)
    normalization = params.normalization
    if isinstance(norm, str):
        try:
            normalization = NormalizationMethod(norm)
        except ValueError:
            pass

    lookback = int(merged_custom.get("lookback_days", params.lookback_days))
    lag_days = int(merged_custom.get("lag_days", params.lag_days))
    winsorize = float(merged_custom.get("winsorize_pct", params.winsorize_pct))

    return FactorParams(
        normalization=normalization,
        winsorize_pct=winsorize,
        min_observations=params.min_observations,
        lookback_days=lookback,
        decay_halflife=params.decay_halflife,
        lag_days=lag_days,
        custom=merged_custom,
    )


def factor_signal_to_builder(
    signal_cls: Type[FactorSignal],
    *,
    score_output: ScoreOutput = "raw",
) -> SignalBuilder:
    """Wrap a ``FactorSignal`` as a ``SignalBuilder``-compatible callable."""

    def _build(
        dates: pd.DatetimeIndex,
        loader: Any,
        config: ConfigDict,
        signal_params: dict[str, Any],
    ) -> pd.DataFrame:
        instance = signal_cls()
        signal_name = instance.name
        data = factor_data_from_config(signal_name, dates, loader, config)
        params = factor_params_from_signal_params(signal_cls, signal_params)

        if score_output == "normalized":
            output = instance.compute_signal(data, params)
            panel = output.scores
        else:
            panel, _meta = instance.compute_raw_signal(data, params)

        return panel.reindex(index=dates, columns=data.tickers)

    return _build


# Only factors with passing BUILDERS parity tests (see tests/signals/test_standalone_parity.py)
RESEARCH_BUILDERS: dict[str, SignalBuilder] = {
    "momentum_research": factor_signal_to_builder(
        VolatilityAdjustedMomentumSignal,
        score_output="raw",
    ),
    "low_volatility_research": factor_signal_to_builder(
        LowVolatilitySignal,
        score_output="raw",
    ),
    "momentum_legacy": factor_signal_to_builder(
        MomentumSignal,
        score_output="raw",
    ),
}


def get_research_signals() -> list[str]:
    return sorted(RESEARCH_BUILDERS.keys())


def build_research_signal(
    name: str,
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
) -> pd.DataFrame:
    """Build a research signal by name (not in production ``BUILDERS``)."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Research signal name must be a non-empty string")
    if not isinstance(dates, pd.DatetimeIndex):
        raise TypeError(f"'dates' must be pd.DatetimeIndex, got {type(dates).__name__}")
    if not isinstance(config, dict):
        raise TypeError(f"'config' must be dict, got {type(config).__name__}")

    builder = RESEARCH_BUILDERS.get(name)
    if builder is None:
        available = ", ".join(get_research_signals())
        raise ValueError(f"Unknown research signal: {name}. Available: {available}")

    signal_params = dict(config.get("signal_params") or {})
    legacy = config.get("parameters") or {}
    if isinstance(legacy, dict):
        signal_params = {**legacy, **signal_params}

    result = builder(dates=dates, loader=loader, config=config, signal_params=signal_params)
    if not isinstance(result, pd.DataFrame):
        raise TypeError(f"Research builder '{name}' must return pd.DataFrame")
    return result


__all__ = [
    "RESEARCH_BUILDERS",
    "build_research_signal",
    "factor_data_from_config",
    "factor_params_from_signal_params",
    "factor_signal_to_builder",
    "get_research_signals",
]
