"""Phase 5: legacy_standalone indexing mode for pre-refactor 126d momentum."""

from __future__ import annotations

import pandas as pd

from bist_quant.signals.core.momentum import compute_price_momentum
from bist_quant.signals.standalone_factors.base import FactorData, FactorParams
from bist_quant.signals.standalone_factors.momentum_signal import MomentumSignal


class TestLegacyStandaloneMomentum:
    def test_legacy_standalone_equals_prod_indexing(self, signal_close_df, signal_dates) -> None:
        legacy = compute_price_momentum(
            signal_close_df, lookback=126, skip=21, mode="legacy_standalone"
        )
        prod = compute_price_momentum(signal_close_df, lookback=126, skip=21, mode="prod")
        pd.testing.assert_frame_equal(legacy, prod)

    def test_momentum_signal_accepts_legacy_indexing_mode(
        self, signal_close_df, signal_dates
    ) -> None:
        data = FactorData(
            close=signal_close_df,
            dates=signal_dates,
            tickers=pd.Index(signal_close_df.columns),
        )
        params = FactorParams(
            custom={
                "lookback_days": 126,
                "skip_days": 21,
                "indexing_mode": "legacy_standalone",
            }
        )
        raw, meta = MomentumSignal().compute_raw_signal(data, params)
        expected = compute_price_momentum(
            signal_close_df, lookback=126, skip=21, mode="legacy_standalone"
        )
        pd.testing.assert_frame_equal(raw, expected.reindex(signal_dates))
        assert meta["indexing_mode"] == "legacy_standalone"
