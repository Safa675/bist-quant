"""Golden tests for five-factor axis panel contract."""

from __future__ import annotations

import inspect

from bist_quant.signals import factor_builders
from bist_quant.signals._axis_cache import AXIS_PANEL_NAMES
from bist_quant.signals.core.panels import quality as core_quality


class TestFiveFactorAxisPanelsGolden:
    def test_axis_panel_names_count(self) -> None:
        assert len(AXIS_PANEL_NAMES) == 27

    def test_quality_panel_contract_keys(self) -> None:
        expected = (
            "quality_roe",
            "quality_roa",
            "quality_accruals",
            "quality_piotroski",
        )
        assert factor_builders.FACTOR_PANEL_CONTRACT["quality"] == expected

    def test_factor_builders_quality_delegates_to_core(self) -> None:
        source = inspect.getsource(factor_builders.build_quality_panels)
        assert "core.panels.quality" in source
        assert "_build_quality_panels" in source
        assert "quality_roe" not in source

    def test_core_quality_is_canonical_implementation(self) -> None:
        source = inspect.getsource(core_quality.build_quality_panels)
        assert "quality_roe" in source
        assert "piotroski_panel" in source
        assert len(source.splitlines()) > 40
