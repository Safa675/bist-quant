"""Unit tests for bist_quant.realtime helpers."""

from __future__ import annotations

import math
import sys
from unittest.mock import MagicMock

# Mock borsapy before any bist_quant import to prevent network hangs
if "borsapy" not in sys.modules:
    sys.modules["borsapy"] = MagicMock()

from bist_quant.realtime.quotes import (
    normalize_change_pct,
    normalize_symbol,
    to_float,
)
from bist_quant.realtime.ticks import (
    fallback_realtime_ticks,
    normalize_realtime_symbols,
)


class TestNormalizeSymbol:
    def test_basic(self):
        assert normalize_symbol("thyao") == "THYAO"

    def test_with_exchange(self):
        assert normalize_symbol("thyao.e") == "THYAO"

    def test_empty(self):
        assert normalize_symbol("") == ""

    def test_whitespace(self):
        assert normalize_symbol("  akbnk  ") == "AKBNK"


class TestToFloat:
    def test_valid(self):
        assert to_float("3.14") == 3.14

    def test_int(self):
        assert to_float(42) == 42.0

    def test_none(self):
        assert to_float(None) is None

    def test_nan(self):
        assert to_float(float("nan")) is None

    def test_inf(self):
        assert to_float(float("inf")) is None

    def test_garbage(self):
        assert to_float("not_a_number") is None


class TestNormalizeChangePct:
    def test_both_match(self):
        result = normalize_change_pct(5.0, 105.0, 100.0)
        assert result == 5.0

    def test_provided_only(self):
        result = normalize_change_pct(3.5, None, None)
        assert result == 3.5

    def test_computed_only(self):
        result = normalize_change_pct(None, 110.0, 100.0)
        assert math.isclose(result, 10.0, rel_tol=0.001)

    def test_decimal_correction(self):
        result = normalize_change_pct(0.05, 105.0, 100.0)
        assert result == 5.0

    def test_all_none(self):
        assert normalize_change_pct(None, None, None) is None

    def test_zero_prev_close(self):
        assert normalize_change_pct(None, 100.0, 0.0) is None


class TestNormalizeRealtimeSymbols:
    def test_string(self):
        assert normalize_realtime_symbols("THYAO,AKBNK") == ["THYAO", "AKBNK"]

    def test_list(self):
        assert normalize_realtime_symbols(["thyao", "garan"]) == ["THYAO", "GARAN"]

    def test_deduplicate(self):
        assert normalize_realtime_symbols("THYAO,THYAO,AKBNK") == ["THYAO", "AKBNK"]

    def test_empty(self):
        assert normalize_realtime_symbols([]) == ["XU100"]

    def test_max_20(self):
        syms = [f"SYM{i}" for i in range(30)]
        result = normalize_realtime_symbols(syms)
        assert len(result) == 20


class TestFallbackTicks:
    def test_basic(self):
        ticks = fallback_realtime_ticks(["THYAO", "AKBNK"], {})
        assert len(ticks) == 2
        assert ticks[0]["symbol"] == "THYAO"
        assert "price" in ticks[0]
        assert "change_pct" in ticks[0]
        assert "volume" in ticks[0]

    def test_with_previous(self):
        prev = {"THYAO": {"price": 50.0, "volume": 500000.0}}
        ticks = fallback_realtime_ticks(["THYAO"], prev)
        assert len(ticks) == 1
        assert 45.0 < ticks[0]["price"] < 55.0
