"""Phase 4: research bridge tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bist_quant.signals.low_volatility_signals import build_low_volatility_signals
from bist_quant.signals.momentum_signals import build_momentum_signals
from bist_quant.signals.standalone_bridge import (
    RESEARCH_BUILDERS,
    build_research_signal,
    get_research_signals,
)


def _runtime_config(close_df: pd.DataFrame) -> dict:
    return {"_runtime_context": {"close_df": close_df}}


class TestResearchBridge:
    def test_get_research_signals_lists_registered(self) -> None:
        names = get_research_signals()
        assert names == sorted(RESEARCH_BUILDERS.keys())
        assert "momentum_research" in names
        assert "low_volatility_research" in names

    def test_momentum_research_matches_builders(self, signal_close_df, signal_dates) -> None:
        config = _runtime_config(signal_close_df)
        research = build_research_signal(
            "momentum_research", signal_dates, loader=None, config=config
        )
        builders = build_momentum_signals(signal_close_df, signal_dates)
        assert np.allclose(research, builders, rtol=1e-5, atol=1e-8, equal_nan=True)

    def test_low_volatility_research_matches_builders(self, signal_close_df, signal_dates) -> None:
        config = _runtime_config(signal_close_df)
        research = build_research_signal(
            "low_volatility_research", signal_dates, loader=None, config=config
        )
        builders = build_low_volatility_signals(signal_close_df, signal_dates)
        assert np.allclose(research, builders, rtol=1e-5, atol=1e-8, equal_nan=True)

    def test_unknown_research_signal_raises(self, signal_dates) -> None:
        with pytest.raises(ValueError, match="Unknown research signal"):
            build_research_signal("not_a_signal", signal_dates, None, {})
