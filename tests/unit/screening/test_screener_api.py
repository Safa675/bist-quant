"""Unit tests for public screener API."""

from __future__ import annotations

import pytest

from bist_quant.screening import (
    FILTER_FIELD_DEFS,
    INDEX_OPTIONS,
    ScreeningValidationError,
    get_screener_metadata,
    run_screener,
)


def test_get_screener_metadata_shape() -> None:
    meta = get_screener_metadata()
    assert "filters" in meta
    assert meta["filters"] == FILTER_FIELD_DEFS
    assert "indexes" in meta
    assert meta["indexes"] == INDEX_OPTIONS
    assert "technical_scans" in meta


def test_run_screener_rejects_non_object_payload() -> None:
    with pytest.raises(ScreeningValidationError, match="JSON object"):
        run_screener([])  # type: ignore[arg-type]
