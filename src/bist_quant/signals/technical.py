from typing import Any

import pandas as pd

from bist_quant.signals._context import (
    build_high_low_panels,
    get_runtime_context,
    parse_float_param,
    parse_int_param,
    require_context,
)
from bist_quant.signals.adx_signals import build_adx_signals
from bist_quant.signals.atr_signals import build_atr_signals
from bist_quant.signals.donchian_signals import build_donchian_signals
from bist_quant.signals.ema_signals import build_ema_signals
from bist_quant.signals.ichimoku_signals import build_ichimoku_signals
from bist_quant.signals.macd_signals import build_macd_signals
from bist_quant.signals.obv_signals import build_obv_signals
from bist_quant.signals.parabolic_sar_signals import build_parabolic_sar_signals
from bist_quant.signals.sma_signals import build_sma_signals
from bist_quant.signals.supertrend_signals import build_supertrend_signals
from bist_quant.signals.xu100_signals import build_xu100_signals

ConfigDict = dict[str, Any]
SignalParams = dict[str, Any]


def build_sma_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    close_df = require_context("sma", get_runtime_context(config), "close_df")
    return build_sma_signals(
        close_df,
        dates,
        loader,
        short_period=parse_int_param("sma", signal_params, "short_period", 10),
        long_period=parse_int_param("sma", signal_params, "long_period", 30),
    )


def build_donchian_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    context = get_runtime_context(config)
    close_df = require_context("donchian", context, "close_df")
    high_df, low_df = build_high_low_panels("donchian", context)
    return build_donchian_signals(
        close_df,
        high_df,
        low_df,
        dates,
        loader,
        lookback=parse_int_param("donchian", signal_params, "lookback", 20),
    )


def build_xu100_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    context = get_runtime_context(config)
    close_df = require_context("xu100", context, "close_df")
    result = build_xu100_signals(close_df, dates, loader)
    xu100_prices = context.get("xu100_prices")
    if "XU100" not in close_df.columns and xu100_prices is not None:
        close_df["XU100"] = xu100_prices.reindex(close_df.index)
    return result


def build_macd_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    close_df = require_context("macd", get_runtime_context(config), "close_df")
    return build_macd_signals(
        close_df,
        dates,
        loader,
        fast=parse_int_param("macd", signal_params, "fast", 12),
        slow=parse_int_param("macd", signal_params, "slow", 26),
        signal=parse_int_param("macd", signal_params, "signal", 9),
    )


def build_adx_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    context = get_runtime_context(config)
    close_df = require_context("adx", context, "close_df")
    high_df, low_df = build_high_low_panels("adx", context)
    return build_adx_signals(
        close_df,
        high_df,
        low_df,
        dates,
        loader,
        period=parse_int_param("adx", signal_params, "period", 14),
    )


def build_supertrend_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    context = get_runtime_context(config)
    close_df = require_context("supertrend", context, "close_df")
    high_df, low_df = build_high_low_panels("supertrend", context)
    return build_supertrend_signals(
        close_df,
        high_df,
        low_df,
        dates,
        loader,
        period=parse_int_param("supertrend", signal_params, "period", 10),
        multiplier=parse_float_param("supertrend", signal_params, "multiplier", 3.0),
    )


def build_ema_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    close_df = require_context("ema", get_runtime_context(config), "close_df")
    return build_ema_signals(
        close_df,
        dates,
        loader,
        short_period=parse_int_param("ema", signal_params, "short_period", 12),
        long_period=parse_int_param("ema", signal_params, "long_period", 26),
    )


def build_atr_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    context = get_runtime_context(config)
    close_df = require_context("atr", context, "close_df")
    high_df, low_df = build_high_low_panels("atr", context)
    return build_atr_signals(
        close_df,
        high_df,
        low_df,
        dates,
        loader,
        period=parse_int_param("atr", signal_params, "period", 14),
    )


def build_obv_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    context = get_runtime_context(config)
    close_df = require_context("obv", context, "close_df")
    volume_df = require_context("obv", context, "volume_df")
    return build_obv_signals(
        close_df,
        volume_df,
        dates,
        loader,
        momentum_lookback=parse_int_param("obv", signal_params, "momentum_lookback", 20),
    )


def build_ichimoku_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    context = get_runtime_context(config)
    close_df = require_context("ichimoku", context, "close_df")
    high_df, low_df = build_high_low_panels("ichimoku", context)
    return build_ichimoku_signals(
        close_df,
        high_df,
        low_df,
        dates,
        loader,
        conversion_period=parse_int_param("ichimoku", signal_params, "conversion_period", 9),
        base_period=parse_int_param("ichimoku", signal_params, "base_period", 26),
        span_b_period=parse_int_param("ichimoku", signal_params, "span_b_period", 52),
    )


def build_parabolic_sar_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    context = get_runtime_context(config)
    close_df = require_context("parabolic_sar", context, "close_df")
    high_df, low_df = build_high_low_panels("parabolic_sar", context)
    return build_parabolic_sar_signals(
        close_df,
        high_df,
        low_df,
        dates,
        loader,
        af_start=parse_float_param("parabolic_sar", signal_params, "af_start", 0.02),
        af_step=parse_float_param("parabolic_sar", signal_params, "af_step", 0.02),
        af_max=parse_float_param("parabolic_sar", signal_params, "af_max", 0.2),
    )


BUILDERS = {
    "sma": build_sma_from_config,
    "donchian": build_donchian_from_config,
    "xu100": build_xu100_from_config,
    "macd": build_macd_from_config,
    "adx": build_adx_from_config,
    "supertrend": build_supertrend_from_config,
    "ema": build_ema_from_config,
    "atr": build_atr_from_config,
    "obv": build_obv_from_config,
    "ichimoku": build_ichimoku_from_config,
    "parabolic_sar": build_parabolic_sar_from_config,
}
