"""Golden and unit tests for momentum indexing modes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bist_quant.signals.core.momentum import compute_price_momentum
from bist_quant.signals.five_factor_rotation_signals import (
    MOMENTUM_LOOKBACK_DAYS,
    MOMENTUM_SKIP_DAYS,
)
from bist_quant.signals.momentum_signals import build_momentum_signals

from tests.signals.conftest import panel_checksum


class TestMomentumIndexingModes:
    def test_prod_and_rotation_differ_at_same_lookback(self, signal_close_df: pd.DataFrame) -> None:
        lookback, skip = 126, 21
        prod = compute_price_momentum(signal_close_df, lookback=lookback, skip=skip, mode="prod")
        rotation = compute_price_momentum(
            signal_close_df, lookback=lookback, skip=skip, mode="rotation"
        )
        valid = prod.notna() & rotation.notna()
        assert valid.any().any()
        assert not np.allclose(
            prod.to_numpy()[valid.to_numpy()],
            rotation.to_numpy()[valid.to_numpy()],
            equal_nan=True,
        )

    def test_rotation_matches_legacy_inline_formula(self, signal_close_df: pd.DataFrame) -> None:
        core = compute_price_momentum(
            signal_close_df,
            lookback=MOMENTUM_LOOKBACK_DAYS,
            skip=MOMENTUM_SKIP_DAYS,
            mode="rotation",
        )
        legacy = (
            signal_close_df.shift(MOMENTUM_SKIP_DAYS)
            / signal_close_df.shift(MOMENTUM_LOOKBACK_DAYS + MOMENTUM_SKIP_DAYS)
            - 1.0
        )
        pd.testing.assert_frame_equal(core, legacy)

    def test_prod_uses_shift_lookback_not_lookback_plus_skip(
        self, signal_close_df: pd.DataFrame
    ) -> None:
        lookback, skip = 252, 21
        core = compute_price_momentum(signal_close_df, lookback=lookback, skip=skip, mode="prod")
        expected = signal_close_df.shift(skip) / signal_close_df.shift(lookback) - 1.0
        pd.testing.assert_frame_equal(core, expected)


class TestMomentumBuildersGolden:
    def test_build_momentum_signals_checksum(self, signal_close_df, signal_dates) -> None:
        result = build_momentum_signals(
            signal_close_df,
            signal_dates,
            lookback=252,
            skip=21,
            vol_lookback=252,
        )
        assert result.shape == (len(signal_dates), signal_close_df.shape[1])
        # Lock prod output; update only with intentional momentum math changes.
        assert panel_checksum(result) == "d90251041319da3941dd82cc3555ffa6"
