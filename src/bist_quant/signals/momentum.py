from typing import Any

import pandas as pd

from bist_quant.signals._context import (
    get_runtime_context,
    parse_int_param,
    require_context,
)
from bist_quant.signals.consistent_momentum_signals import build_consistent_momentum_signals
from bist_quant.signals.low_volatility_signals import build_low_volatility_signals
from bist_quant.signals.momentum_reversal_volatility_signals import (
    build_momentum_reversal_volatility_signals,
)
from bist_quant.signals.momentum_signals import build_momentum_signals
from bist_quant.signals.residual_momentum_signals import build_residual_momentum_signals
from bist_quant.signals.sector_rotation_signals import build_sector_rotation_signals
from bist_quant.signals.short_term_reversal_signals import build_short_term_reversal_signals
from bist_quant.signals.trend_following_signals import build_trend_following_signals

ConfigDict = dict[str, Any]
SignalParams = dict[str, Any]


def build_momentum_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    context = get_runtime_context(config)
    close_df = require_context("momentum", context, "close_df")
    return build_momentum_signals(
        close_df,
        dates,
        loader,
        lookback=parse_int_param("momentum", signal_params, "lookback", 252),
        skip=parse_int_param("momentum", signal_params, "skip", 21),
        vol_lookback=parse_int_param("momentum", signal_params, "vol_lookback", 252),
    )


def build_consistent_momentum_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    close_df = require_context(
        "consistent_momentum",
        get_runtime_context(config),
        "close_df",
    )
    return build_consistent_momentum_signals(close_df, dates, loader)


def build_residual_momentum_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    close_df = require_context(
        "residual_momentum",
        get_runtime_context(config),
        "close_df",
    )
    return build_residual_momentum_signals(close_df, dates, loader)


def build_momentum_reversal_volatility_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    close_df = require_context(
        "momentum_reversal_volatility",
        get_runtime_context(config),
        "close_df",
    )
    return build_momentum_reversal_volatility_signals(close_df, dates, loader)


def build_low_volatility_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    close_df = require_context(
        "low_volatility",
        get_runtime_context(config),
        "close_df",
    )
    return build_low_volatility_signals(close_df, dates, loader)


def build_trend_following_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    close_df = require_context(
        "trend_following",
        get_runtime_context(config),
        "close_df",
    )
    return build_trend_following_signals(close_df, dates, loader)


def build_sector_rotation_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    close_df = require_context(
        "sector_rotation",
        get_runtime_context(config),
        "close_df",
    )
    return build_sector_rotation_signals(close_df, dates, loader)


def build_short_term_reversal_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    del signal_params
    close_df = require_context(
        "short_term_reversal",
        get_runtime_context(config),
        "close_df",
    )
    return build_short_term_reversal_signals(close_df, dates, loader)


BUILDERS = {
    "momentum": build_momentum_from_config,
    "consistent_momentum": build_consistent_momentum_from_config,
    "residual_momentum": build_residual_momentum_from_config,
    "momentum_reversal_volatility": build_momentum_reversal_volatility_from_config,
    "low_volatility": build_low_volatility_from_config,
    "trend_following": build_trend_following_from_config,
    "sector_rotation": build_sector_rotation_from_config,
    "short_term_reversal": build_short_term_reversal_from_config,
}
