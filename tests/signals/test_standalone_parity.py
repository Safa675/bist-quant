"""BUILDERS ↔ standalone parity tests (Phase 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bist_quant.signals.core.momentum import compute_risk_adjusted_momentum
from bist_quant.signals.low_volatility_signals import build_low_volatility_signals
from bist_quant.signals.momentum_signals import build_momentum_signals
from bist_quant.signals.standalone_factors.base import FactorData, FactorParams
from bist_quant.signals.standalone_factors.defensive_signal import LowVolatilitySignal
from bist_quant.signals.standalone_factors.momentum_signal import VolatilityAdjustedMomentumSignal


class TestStandaloneParity:
    def test_momentum_prod_raw_matches_builders(self, signal_close_df, signal_dates) -> None:
        builders = build_momentum_signals(signal_close_df, signal_dates)
        raw = compute_risk_adjusted_momentum(signal_close_df, mode="prod").reindex(signal_dates)

        data = FactorData(
            close=signal_close_df,
            dates=signal_dates,
            tickers=pd.Index(signal_close_df.columns),
        )
        params = VolatilityAdjustedMomentumSignal().get_default_params()
        standalone_raw, _ = VolatilityAdjustedMomentumSignal().compute_raw_signal(data, params)
        standalone_raw = standalone_raw.reindex(signal_dates)

        assert np.allclose(builders, standalone_raw, rtol=1e-5, atol=1e-8, equal_nan=True)
        assert np.allclose(builders, raw, rtol=1e-5, atol=1e-8, equal_nan=True)

    def test_low_volatility_raw_matches_builders(self, signal_close_df, signal_dates) -> None:
        builders = build_low_volatility_signals(signal_close_df, signal_dates)

        data = FactorData(
            close=signal_close_df,
            dates=signal_dates,
            tickers=pd.Index(signal_close_df.columns),
        )
        params = FactorParams(lookback_days=252)
        standalone_raw, _ = LowVolatilitySignal().compute_raw_signal(data, params)
        standalone_raw = standalone_raw.reindex(signal_dates)

        # BUILDERS uses weekly vol path; compare via core weekly scores
        from bist_quant.signals.core.low_volatility import calculate_low_volatility_scores

        core_weekly = calculate_low_volatility_scores(signal_close_df).reindex(signal_dates)
        assert np.allclose(builders, core_weekly, rtol=1e-5, atol=1e-8, equal_nan=True)
        assert np.allclose(builders, standalone_raw, rtol=1e-5, atol=1e-8, equal_nan=True)
