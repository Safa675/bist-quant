from typing import Any

import pandas as pd

from bist_quant.signals._context import (
    get_runtime_context,
    parse_float_param,
    require_context,
)
from bist_quant.signals.accrual_signals import build_accrual_signals
from bist_quant.signals.earnings_quality_signals import build_earnings_quality_signals
from bist_quant.signals.fscore_reversal_signals import build_fscore_reversal_signals
from bist_quant.signals.profitability_signals import build_profitability_signals
from bist_quant.signals.roa_signals import build_roa_signals

ConfigDict = dict[str, Any]
SignalParams = dict[str, Any]


def build_profitability_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    fundamentals = require_context(
        "profitability",
        get_runtime_context(config),
        "fundamentals",
    )
    return build_profitability_signals(
        fundamentals,
        dates,
        loader,
        operating_income_weight=parse_float_param(
            "profitability",
            signal_params,
            "operating_income_weight",
            0.5,
        ),
        gross_profit_weight=parse_float_param(
            "profitability",
            signal_params,
            "gross_profit_weight",
            0.5,
        ),
    )


def build_earnings_quality_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    context = get_runtime_context(config)
    fundamentals = require_context("earnings_quality", context, "fundamentals")
    close_df = require_context("earnings_quality", context, "close_df")
    return build_earnings_quality_signals(fundamentals, close_df, dates, loader)


def build_fscore_reversal_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    context = get_runtime_context(config)
    fundamentals = require_context("fscore_reversal", context, "fundamentals")
    close_df = require_context("fscore_reversal", context, "close_df")
    return build_fscore_reversal_signals(fundamentals, close_df, dates, loader)


def build_roa_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    fundamentals = require_context("roa", get_runtime_context(config), "fundamentals")
    return build_roa_signals(fundamentals, dates, loader)


def build_accrual_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    fundamentals = require_context(
        "accrual",
        get_runtime_context(config),
        "fundamentals",
    )
    return build_accrual_signals(fundamentals, dates, loader)


BUILDERS = {
    "profitability": build_profitability_from_config,
    "earnings_quality": build_earnings_quality_from_config,
    "fscore_reversal": build_fscore_reversal_from_config,
    "roa": build_roa_from_config,
    "accrual": build_accrual_from_config,
}
