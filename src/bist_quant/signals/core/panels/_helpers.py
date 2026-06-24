"""Shared helpers for core panel builders (bridge during migration)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def finalize_builder_outputs(
    builder_name: str,
    raw_panels: Dict[str, pd.DataFrame],
    contract_key: str,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
) -> Dict[str, pd.DataFrame]:
    from bist_quant.signals.factor_builders import (
        FACTOR_PANEL_CONTRACT,
        _finalize_builder_outputs,
    )

    return _finalize_builder_outputs(
        builder_name,
        raw_panels,
        FACTOR_PANEL_CONTRACT[contract_key],
        dates,
        tickers,
    )


def load_shares_panel(
    data_loader,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
) -> tuple[pd.DataFrame, bool]:
    from bist_quant.signals.factor_builders import _load_shares_panel

    return _load_shares_panel(data_loader, dates, tickers)


def build_metric_panel(
    metrics_df: pd.DataFrame,
    metric_name: str,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
) -> pd.DataFrame:
    from bist_quant.signals.factor_builders import _build_metric_panel

    return _build_metric_panel(metrics_df, metric_name, dates, tickers)


def panel_constants() -> dict:
    from bist_quant.signals import factor_builders as fb

    return {
        "AMIHUD_LOOKBACK_DAYS": fb.AMIHUD_LOOKBACK_DAYS,
        "TURNOVER_LOOKBACK_DAYS": fb.TURNOVER_LOOKBACK_DAYS,
        "PRICE_ACCELERATION_FAST": fb.PRICE_ACCELERATION_FAST,
        "PRICE_ACCELERATION_SLOW": fb.PRICE_ACCELERATION_SLOW,
        "REVERSAL_LOOKBACK_DAYS": fb.REVERSAL_LOOKBACK_DAYS,
        "MARGIN_CHANGE_QUARTERS": fb.MARGIN_CHANGE_QUARTERS,
        "SALES_ACCEL_QUARTERS": fb.SALES_ACCEL_QUARTERS,
        "EARNINGS_STABILITY_QUARTERS": fb.EARNINGS_STABILITY_QUARTERS,
        "VOLATILITY_LOOKBACK_DAYS": fb.VOLATILITY_LOOKBACK_DAYS,
        "BETA_LOOKBACK_DAYS": fb.BETA_LOOKBACK_DAYS,
        "BETA_MIN_OBS": fb.BETA_MIN_OBS,
        "MIN_ROLLING_OBS_RATIO": fb.MIN_ROLLING_OBS_RATIO,
        "INCOME_SHEET": fb.INCOME_SHEET,
        "BALANCE_SHEET": fb.BALANCE_SHEET,
        "CASH_FLOW_SHEET": fb.CASH_FLOW_SHEET,
    }


__all__ = [
    "finalize_builder_outputs",
    "load_shares_panel",
    "build_metric_panel",
    "panel_constants",
]
