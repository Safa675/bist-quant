"""Unit tests for screening error hierarchy and types."""

from __future__ import annotations

from bist_quant.screening import (
    ScreeningDataError,
    ScreeningError,
    ScreeningExecutionError,
    ScreeningValidationError,
    StockScreenerResult,
)


def test_screening_error_codes() -> None:
    assert ScreeningError("x").code == "SCREENING_ERROR"
    assert ScreeningValidationError("x").code == "VALIDATION_ERROR"
    assert ScreeningDataError("x").code == "DATA_ERROR"
    assert ScreeningExecutionError("x").code == "EXECUTION_ERROR"


def test_screening_error_user_message_defaults_to_technical() -> None:
    err = ScreeningDataError("technical", user_message="friendly")
    assert str(err) == "technical"
    assert err.user_message == "friendly"


def test_stock_screener_result_is_typed_dict() -> None:
    payload: StockScreenerResult = {
        "meta": {},
        "columns": [],
        "rows": [],
        "applied_filters": [],
        "chart": {},
    }
    assert payload["meta"] == {}
