"""Sovereign-risk overlay built from Turkish USD eurobond stress."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bist_quant.regime.macro_features import MacroFeatures
from bist_quant.signals._context import get_runtime_context, require_context

LOGGER = logging.getLogger(__name__)


def _as_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    if not np.isfinite(parsed):
        return default
    return float(parsed)


class SovereignRiskSignals:
    """
    Use eurobond stress to modulate equity exposure.

    Wide sovereign spreads indicate risk aversion and reduce the equity
    allocation multiplier.
    """

    def __init__(
        self,
        *,
        data_dir: str | Path | None = None,
        macro_features: MacroFeatures | None = None,
        calm_multiplier: float = 1.0,
        stress_multiplier: float = 0.35,
    ) -> None:
        self._macro_features = macro_features
        self._data_dir = Path(data_dir) if data_dir is not None else None
        self.calm_multiplier = _as_float(calm_multiplier, default=1.0)
        self.stress_multiplier = _as_float(stress_multiplier, default=0.35)

        # Clamp to [0, 1] to keep multiplier semantics explicit.
        self.calm_multiplier = float(np.clip(self.calm_multiplier, 0.0, 1.0))
        self.stress_multiplier = float(np.clip(self.stress_multiplier, 0.0, 1.0))

    @property
    def macro_features(self) -> MacroFeatures:
        if self._macro_features is None:
            self._macro_features = MacroFeatures(data_dir=self._data_dir)
        return self._macro_features

    def compute_allocation_multiplier(self, dates: pd.DatetimeIndex) -> pd.Series:
        """
        Build an allocation multiplier series aligned to requested dates.

        When eurobond stress is active, returns ``stress_multiplier``.
        Otherwise returns ``calm_multiplier``.
        """
        macro = self.macro_features
        if not getattr(macro, "_loaded", False):
            macro.load()

        stress = macro.compute_eurobond_stress()
        if stress.empty:
            return pd.Series(self.calm_multiplier, index=dates, name="sovereign_risk_multiplier")

        aligned_stress = (
            stress.reindex(pd.DatetimeIndex(dates).sort_values().unique())
            .ffill()
            .fillna(False)
            .astype(bool)
        )
        values = np.where(
            aligned_stress.to_numpy(dtype=bool),
            self.stress_multiplier,
            self.calm_multiplier,
        )
        return pd.Series(values, index=aligned_stress.index, name="sovereign_risk_multiplier")

    def apply_to_signal_panel(
        self,
        signal_panel: pd.DataFrame,
        *,
        dates: pd.DatetimeIndex | None = None,
    ) -> pd.DataFrame:
        """
        Apply sovereign-risk multiplier to any existing signal panel.

        This keeps cross-sectional ranking intact while reducing gross
        equity exposure during sovereign stress windows.
        """
        if signal_panel.empty:
            return signal_panel.copy()

        target_dates = dates if dates is not None else pd.DatetimeIndex(signal_panel.index)
        multiplier = self.compute_allocation_multiplier(pd.DatetimeIndex(target_dates))
        aligned = multiplier.reindex(signal_panel.index).ffill().fillna(self.calm_multiplier)
        return signal_panel.mul(aligned, axis=0)

    def build_signal_panel(self, close_df: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Build a standalone sovereign-risk overlay panel (dates x tickers)."""
        if close_df.empty:
            return pd.DataFrame(index=dates)
        base = pd.DataFrame(1.0, index=dates, columns=close_df.columns, dtype=float)
        result = self.apply_to_signal_panel(base, dates=dates)
        LOGGER.info(
            "Sovereign risk panel built: %s days x %s tickers (calm=%.2f, stress=%.2f)",
            result.shape[0],
            result.shape[1],
            self.calm_multiplier,
            self.stress_multiplier,
        )
        return result


def build_sovereign_risk_signals(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader: Any = None,
    *,
    calm_multiplier: float = 1.0,
    stress_multiplier: float = 0.35,
) -> pd.DataFrame:
    """Functional wrapper for sovereign-risk overlay signal construction."""
    data_dir = getattr(data_loader, "data_dir", None) if data_loader is not None else None
    builder = SovereignRiskSignals(
        data_dir=data_dir,
        calm_multiplier=calm_multiplier,
        stress_multiplier=stress_multiplier,
    )
    return builder.build_signal_panel(close_df=close_df, dates=dates)


def build_sovereign_risk_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: dict[str, Any],
    signal_params: dict[str, Any],
) -> pd.DataFrame:
    """Config-driven sovereign risk builder for signal factory integration."""
    close_df = require_context(
        "sovereign_risk",
        get_runtime_context(config),
        "close_df",
    )

    calm_multiplier = _as_float(signal_params.get("calm_multiplier"), default=1.0)
    stress_multiplier = _as_float(signal_params.get("stress_multiplier"), default=0.35)
    return build_sovereign_risk_signals(
        close_df=close_df,
        dates=dates,
        data_loader=loader,
        calm_multiplier=calm_multiplier,
        stress_multiplier=stress_multiplier,
    )

