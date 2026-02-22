import os
from typing import Any, Mapping

import numpy as np
import pandas as pd

from bist_quant.signals._context import (
    build_high_low_panels,
    get_runtime_context,
    parse_float_param,
    parse_int_param,
    require_context,
)
from bist_quant.signals.betting_against_beta_signals import build_betting_against_beta_signals
from bist_quant.signals.breakout_value_signals import build_breakout_value_signals
from bist_quant.signals.five_factor_rotation_signals import build_five_factor_rotation_signals
from bist_quant.signals.momentum_asset_growth_signals import build_momentum_asset_growth_signals
from bist_quant.signals.pairs_trading_signals import build_pairs_trading_signals
from bist_quant.signals.quality_momentum_signals import build_quality_momentum_signals
from bist_quant.signals.quality_value_signals import build_quality_value_signals
from bist_quant.signals.size_rotation_momentum_signals import build_size_rotation_momentum_signals
from bist_quant.signals.size_rotation_quality_signals import build_size_rotation_quality_signals
from bist_quant.signals.size_rotation_signals import build_size_rotation_signals
from bist_quant.signals.small_cap_momentum_signals import build_small_cap_momentum_signals
from bist_quant.signals.ta_consensus_signals import TAConsensusSignals
from bist_quant.signals.trend_value_signals import build_trend_value_signals

ConfigDict = dict[str, Any]
SignalParams = dict[str, Any]


def weighted_sum(
    panels: Mapping[str, pd.DataFrame],
    weights: Mapping[str, float],
) -> pd.DataFrame:
    if not panels:
        raise ValueError("weighted_sum requires at least one panel")

    missing = [name for name in panels if name not in weights]
    if missing:
        raise ValueError(f"weighted_sum missing weights for panels: {missing}")

    result = None
    total_weight = 0.0
    for name, panel in panels.items():
        weight = float(weights[name])
        total_weight += weight
        weighted_panel = panel * weight
        result = weighted_panel if result is None else result.add(weighted_panel, fill_value=np.nan)

    if result is None:
        raise ValueError("weighted_sum received no valid panels")
    if total_weight == 0:
        raise ValueError("weighted_sum total weight cannot be zero")
    return result / total_weight


def zscore_blend(
    panels: Mapping[str, pd.DataFrame],
    weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    if not panels:
        raise ValueError("zscore_blend requires at least one panel")

    z_panels = {}
    for name, panel in panels.items():
        mean = panel.mean(axis=1)
        std = panel.std(axis=1).replace(0, np.nan)
        z_panels[name] = panel.sub(mean, axis=0).div(std, axis=0)

    if weights is None:
        equal_weight = 1.0 / len(z_panels)
        weights = {name: equal_weight for name in z_panels}
    return weighted_sum(z_panels, weights)


def rank_blend(
    panels: Mapping[str, pd.DataFrame],
    weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    if not panels:
        raise ValueError("rank_blend requires at least one panel")

    rank_panels = {name: panel.rank(axis=1, pct=True) for name, panel in panels.items()}
    if weights is None:
        equal_weight = 1.0 / len(rank_panels)
        weights = {name: equal_weight for name in rank_panels}
    return weighted_sum(rank_panels, weights)


def _blend_optional_external_consensus(
    signal_name: str,
    panel: pd.DataFrame,
    signal_params: SignalParams,
) -> pd.DataFrame:
    if panel.empty:
        return panel

    raw_weight = parse_float_param(signal_name, signal_params, "external_consensus_weight", 0.0)
    external_weight = float(np.clip(raw_weight, 0.0, 1.0))
    if external_weight <= 0.0:
        return panel

    interval = str(signal_params.get("external_consensus_interval", "1d"))
    batch_size = parse_int_param(signal_name, signal_params, "external_consensus_batch_size", 20)
    request_sleep_seconds = parse_float_param(
        signal_name,
        signal_params,
        "external_consensus_request_sleep_seconds",
        0.0,
    )
    batch_pause_seconds = parse_float_param(
        signal_name,
        signal_params,
        "external_consensus_batch_pause_seconds",
        0.0,
    )
    fillna_value = parse_float_param(
        signal_name,
        signal_params,
        "external_consensus_fillna_value",
        0.0,
    )

    consensus_builder = TAConsensusSignals(
        batch_size=batch_size,
        request_sleep_seconds=request_sleep_seconds,
        batch_pause_seconds=batch_pause_seconds,
    )
    consensus_panel = consensus_builder.build_signal_panel(
        symbols=[str(symbol) for symbol in panel.columns],
        dates=panel.index,
        interval=interval,
        fillna_value=fillna_value,
    )
    if consensus_panel.empty:
        return panel

    base_weight = max(0.0, 1.0 - external_weight)
    if base_weight <= 0.0:
        return consensus_panel.astype("float64")

    return zscore_blend(
        {
            "base": panel.astype("float64"),
            "external_consensus": consensus_panel.astype("float64"),
        },
        weights={
            "base": base_weight,
            "external_consensus": external_weight,
        },
    )


def build_trend_value_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    close_df = require_context("trend_value", get_runtime_context(config), "close_df")
    panel = build_trend_value_signals(close_df, dates, loader)
    return _blend_optional_external_consensus("trend_value", panel, signal_params)


def build_breakout_value_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    context = get_runtime_context(config)
    close_df = require_context("breakout_value", context, "close_df")
    high_df, low_df = build_high_low_panels("breakout_value", context)
    panel = build_breakout_value_signals(close_df, high_df, low_df, dates, loader)
    return _blend_optional_external_consensus("breakout_value", panel, signal_params)


def build_quality_momentum_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    context = get_runtime_context(config)
    close_df = require_context("quality_momentum", context, "close_df")
    fundamentals = require_context("quality_momentum", context, "fundamentals")
    panel = build_quality_momentum_signals(close_df, fundamentals, dates, loader)
    return _blend_optional_external_consensus("quality_momentum", panel, signal_params)


def build_quality_value_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    context = get_runtime_context(config)
    close_df = require_context("quality_value", context, "close_df")
    fundamentals = require_context("quality_value", context, "fundamentals")
    panel = build_quality_value_signals(close_df, fundamentals, dates, loader)
    return _blend_optional_external_consensus("quality_value", panel, signal_params)


def build_small_cap_momentum_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    close_df = require_context(
        "small_cap_momentum",
        get_runtime_context(config),
        "close_df",
    )
    panel = build_small_cap_momentum_signals(close_df, dates, loader)
    return _blend_optional_external_consensus("small_cap_momentum", panel, signal_params)


def build_momentum_asset_growth_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    context = get_runtime_context(config)
    fundamentals = require_context("momentum_asset_growth", context, "fundamentals")
    close_df = require_context("momentum_asset_growth", context, "close_df")
    panel = build_momentum_asset_growth_signals(fundamentals, close_df, dates, loader)
    return _blend_optional_external_consensus("momentum_asset_growth", panel, signal_params)


def build_pairs_trading_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    close_df = require_context(
        "pairs_trading",
        get_runtime_context(config),
        "close_df",
    )
    panel = build_pairs_trading_signals(close_df, dates, loader)
    return _blend_optional_external_consensus("pairs_trading", panel, signal_params)


def build_betting_against_beta_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    close_df = require_context(
        "betting_against_beta",
        get_runtime_context(config),
        "close_df",
    )
    beta_window = parse_int_param(
        "betting_against_beta",
        signal_params,
        "beta_window",
        252,
    )
    panel = build_betting_against_beta_signals(close_df, dates, loader, beta_window=beta_window)
    return _blend_optional_external_consensus("betting_against_beta", panel, signal_params)


def build_size_rotation_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    close_df = require_context("size_rotation", get_runtime_context(config), "close_df")
    panel = build_size_rotation_signals(close_df, dates, loader)
    return _blend_optional_external_consensus("size_rotation", panel, signal_params)


def build_size_rotation_momentum_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    close_df = require_context(
        "size_rotation_momentum",
        get_runtime_context(config),
        "close_df",
    )
    panel = build_size_rotation_momentum_signals(close_df, dates, loader)
    return _blend_optional_external_consensus("size_rotation_momentum", panel, signal_params)


def build_size_rotation_quality_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    context = get_runtime_context(config)
    close_df = require_context("size_rotation_quality", context, "close_df")
    fundamentals = require_context("size_rotation_quality", context, "fundamentals")
    panel = build_size_rotation_quality_signals(close_df, fundamentals, dates, loader)
    return _blend_optional_external_consensus("size_rotation_quality", panel, signal_params)


def build_five_factor_rotation_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    context = get_runtime_context(config)
    close_df = require_context("five_factor_rotation", context, "close_df")
    fundamentals = require_context("five_factor_rotation", context, "fundamentals")
    volume_df = require_context("five_factor_rotation", context, "volume_df")

    cache_cfg = config.get("construction_cache", {})
    if not isinstance(cache_cfg, dict):
        cache_cfg = {}

    debug_cfg = config.get("debug", {})
    if not isinstance(debug_cfg, dict):
        debug_cfg = {}

    orth_cfg = config.get("axis_orthogonalization", {})
    if not isinstance(orth_cfg, dict):
        orth_cfg = {}

    walk_forward_cfg = config.get("walk_forward", {})
    if not isinstance(walk_forward_cfg, dict):
        walk_forward_cfg = {}

    debug_env = os.getenv("FIVE_FACTOR_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
    debug_enabled = bool(debug_cfg.get("enabled", False) or debug_env)

    signals, factor_details = build_five_factor_rotation_signals(
        close_df,
        dates,
        loader,
        fundamentals=fundamentals,
        volume_df=volume_df,
        use_construction_cache=cache_cfg.get("enabled", True),
        force_rebuild_construction_cache=cache_cfg.get("force_rebuild", False),
        construction_cache_path=cache_cfg.get("path"),
        mwu_walkforward_config=walk_forward_cfg,
        axis_orthogonalization_config=orth_cfg,
        return_details=True,
        debug=debug_enabled,
    )

    config["_factor_details"] = factor_details
    return _blend_optional_external_consensus("five_factor_rotation", signals, signal_params)


BUILDERS = {
    "trend_value": build_trend_value_from_config,
    "breakout_value": build_breakout_value_from_config,
    "quality_momentum": build_quality_momentum_from_config,
    "quality_value": build_quality_value_from_config,
    "small_cap_momentum": build_small_cap_momentum_from_config,
    "momentum_asset_growth": build_momentum_asset_growth_from_config,
    "pairs_trading": build_pairs_trading_from_config,
    "betting_against_beta": build_betting_against_beta_from_config,
    "size_rotation": build_size_rotation_from_config,
    "size_rotation_momentum": build_size_rotation_momentum_from_config,
    "size_rotation_quality": build_size_rotation_quality_from_config,
    "five_factor_rotation": build_five_factor_rotation_from_config,
}
