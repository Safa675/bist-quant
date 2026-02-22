from typing import Any

import pandas as pd

from bist_quant.signals._context import (
    get_runtime_context,
    parse_optional_dict,
    parse_optional_str_list,
    require_context,
)
from bist_quant.signals.asset_growth_signals import build_asset_growth_signals
from bist_quant.signals.dividend_rotation_signals import build_dividend_rotation_signals
from bist_quant.signals.investment_signals import build_investment_signals
from bist_quant.signals.macro_hedge_signals import build_macro_hedge_signals
from bist_quant.signals.small_cap_signals import build_small_cap_signals
from bist_quant.signals.sovereign_risk_signals import build_sovereign_risk_from_config
from bist_quant.signals.value_signals import build_value_signals

ConfigDict = dict[str, Any]
SignalParams = dict[str, Any]


def build_value_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    context = get_runtime_context(config)
    fundamentals = require_context("value", context, "fundamentals")
    close_df = require_context("value", context, "close_df")

    metric_weights = parse_optional_dict("value", signal_params, "metric_weights")
    enabled_metrics = parse_optional_str_list("value", signal_params, "enabled_metrics")

    return build_value_signals(
        fundamentals,
        close_df,
        dates,
        loader,
        metric_weights=metric_weights,
        enabled_metrics=enabled_metrics,
    )


def build_investment_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    context = get_runtime_context(config)
    fundamentals = require_context("investment", context, "fundamentals")
    close_df = require_context("investment", context, "close_df")
    return build_investment_signals(fundamentals, close_df, dates, loader)


def build_small_cap_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    context = get_runtime_context(config)
    fundamentals = require_context("small_cap", context, "fundamentals")
    close_df = require_context("small_cap", context, "close_df")
    volume_df = require_context("small_cap", context, "volume_df")
    return build_small_cap_signals(fundamentals, close_df, volume_df, dates, loader)


def build_asset_growth_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    fundamentals = require_context(
        "asset_growth",
        get_runtime_context(config),
        "fundamentals",
    )
    return build_asset_growth_signals(fundamentals, dates, loader)


def build_dividend_rotation_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    close_df = require_context(
        "dividend_rotation",
        get_runtime_context(config),
        "close_df",
    )
    return build_dividend_rotation_signals(close_df, dates, loader)


def build_macro_hedge_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    close_df = require_context(
        "macro_hedge",
        get_runtime_context(config),
        "close_df",
    )
    return build_macro_hedge_signals(close_df, dates, loader)


BUILDERS = {
    "value": build_value_from_config,
    "investment": build_investment_from_config,
    "small_cap": build_small_cap_from_config,
    "asset_growth": build_asset_growth_from_config,
    "dividend_rotation": build_dividend_rotation_from_config,
    "macro_hedge": build_macro_hedge_from_config,
    "sovereign_risk": build_sovereign_risk_from_config,
}
