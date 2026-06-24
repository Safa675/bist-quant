"""Pre-Phase-3 standalone factor baselines (separate from BUILDERS)."""

from __future__ import annotations

import pandas as pd

from bist_quant.signals.standalone_factors.base import FactorData, FactorParams
from bist_quant.signals.standalone_factors.momentum_signal import MomentumSignal

from tests.signals.conftest import panel_checksum


class TestStandaloneBaselines:
    def test_momentum_signal_rotation_indexing_baseline(self, signal_close_df, signal_dates) -> None:
        data = FactorData(
            close=signal_close_df,
            dates=signal_dates,
            tickers=pd.Index(signal_close_df.columns),
        )
        params = FactorParams(
            lookback_days=126,
            custom={"lookback_days": 126, "skip_days": 21, "volatility_adjust": False},
        )
        output = MomentumSignal().compute_signal(data, params)
        scores = output.scores.reindex(index=signal_dates, columns=signal_close_df.columns)
        assert scores.shape == signal_close_df.shape
        assert panel_checksum(scores) == "bb058e7a6b01404c3e084962bec72194"

    def test_momentum_signal_raw_matches_rotation_mode(self, signal_close_df, signal_dates) -> None:
        from bist_quant.signals.core.momentum import compute_price_momentum

        data = FactorData(
            close=signal_close_df,
            dates=signal_dates,
            tickers=pd.Index(signal_close_df.columns),
        )
        params = FactorParams(custom={"lookback_days": 126, "skip_days": 21})
        raw, _meta = MomentumSignal().compute_raw_signal(data, params)
        expected = compute_price_momentum(
            signal_close_df, lookback=126, skip=21, mode="rotation"
        )
        pd.testing.assert_frame_equal(raw, expected.reindex(index=signal_dates))
