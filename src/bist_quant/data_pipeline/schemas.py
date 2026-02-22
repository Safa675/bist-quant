from __future__ import annotations

from typing import Any

import pandas as pd

from bist_quant.data_pipeline.errors import SchemaValidationError

try:
    import pandera as pa
    from pandera import Check, Column

    _PANDERA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency path
    pa = None
    Check = None
    Column = None
    _PANDERA_AVAILABLE = False


REQUIRED_FLAT_COLUMNS = [
    "ticker",
    "period_end",
    "fiscal_year",
    "fiscal_quarter",
    "reporting_type",
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "ebitda",
    "total_assets",
    "total_equity",
    "total_liabilities",
    "operating_cash_flow",
    "free_cash_flow",
    "short_term_debt",
    "long_term_debt",
    "cash",
]


def _ensure_columns(frame: pd.DataFrame, required_columns: list[str], context: str) -> None:
    missing = [col for col in required_columns if col not in frame.columns]
    if missing:
        raise SchemaValidationError(f"{context}: missing required columns: {missing}")


def validate_raw_payload_structure(raw_by_ticker: dict[str, dict[str, Any]]) -> None:
    """Validate minimal shape for raw fetch payloads."""
    if not isinstance(raw_by_ticker, dict):
        raise SchemaValidationError("raw payload must be dict[ticker, payload]")

    required_keys = {"symbol", "periods_requested", "items"}
    for ticker, payload in raw_by_ticker.items():
        if not isinstance(payload, dict):
            raise SchemaValidationError(f"raw payload for {ticker} must be dict")
        missing = [k for k in required_keys if k not in payload]
        if missing:
            raise SchemaValidationError(f"raw payload for {ticker} missing keys: {missing}")

        items = payload.get("items")
        if items is None:
            continue
        if not isinstance(items, list):
            raise SchemaValidationError(f"raw payload for {ticker} items must be list")

        if _PANDERA_AVAILABLE and items:
            items_df = pd.DataFrame(items)
            schema = pa.DataFrameSchema(
                {
                    "itemCode": Column(object, nullable=True, required=False),
                    "itemDescTr": Column(object, nullable=True, required=False),
                    "value1": Column(object, nullable=True, required=False),
                    "value2": Column(object, nullable=True, required=False),
                    "value3": Column(object, nullable=True, required=False),
                    "value4": Column(object, nullable=True, required=False),
                    "value5": Column(object, nullable=True, required=False),
                },
                strict=False,
            )
            try:
                schema.validate(items_df, lazy=True)
            except Exception as exc:
                raise SchemaValidationError(
                    f"raw items schema validation failed for {ticker}: {exc}"
                ) from exc


def validate_flat_normalized(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the flat normalized fundamentals dataset."""
    if frame is None:
        raise SchemaValidationError("flat normalized dataframe is None")
    if frame.empty:
        return frame

    _ensure_columns(frame, REQUIRED_FLAT_COLUMNS, "flat_normalized")

    validated = frame.copy()
    validated["ticker"] = validated["ticker"].astype(str)
    validated["period_end"] = pd.to_datetime(validated["period_end"], errors="coerce")
    if validated["period_end"].isna().any():
        raise SchemaValidationError("flat_normalized: period_end contains invalid dates")

    validated["fiscal_year"] = pd.to_numeric(validated["fiscal_year"], errors="coerce")
    validated["fiscal_quarter"] = pd.to_numeric(validated["fiscal_quarter"], errors="coerce")

    if _PANDERA_AVAILABLE:
        metric_columns = [
            c
            for c in REQUIRED_FLAT_COLUMNS
            if c not in {"ticker", "period_end", "fiscal_year", "fiscal_quarter", "reporting_type"}
        ]
        schema = pa.DataFrameSchema(
            {
                "ticker": Column(str, nullable=False),
                "period_end": Column(pa.DateTime, nullable=False),
                "fiscal_year": Column(float, nullable=False, checks=Check.in_range(1990, 2100)),
                "fiscal_quarter": Column(float, nullable=False, checks=Check.isin([1, 2, 3, 4])),
                "reporting_type": Column(str, nullable=False),
                **{col: Column(float, nullable=True, required=True) for col in metric_columns},
            },
            strict=False,
            coerce=True,
        )
        try:
            validated = schema.validate(validated, lazy=True)
        except Exception as exc:
            raise SchemaValidationError(f"flat normalized schema validation failed: {exc}") from exc

    return validated


def validate_consolidated_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate consolidated multi-index fundamentals panel."""
    if frame is None:
        raise SchemaValidationError("consolidated dataframe is None")
    if frame.empty:
        return frame
    if not isinstance(frame.index, pd.MultiIndex):
        raise SchemaValidationError("consolidated dataframe index must be MultiIndex")

    names = list(frame.index.names)
    if names[:3] != ["ticker", "sheet_name", "row_name"]:
        raise SchemaValidationError(
            "consolidated dataframe index names must start with ['ticker', 'sheet_name', 'row_name']"
        )

    period_cols = [c for c in frame.columns if "/" in str(c)]
    if not period_cols:
        raise SchemaValidationError("consolidated dataframe has no period columns")

    if frame.index.has_duplicates:
        raise SchemaValidationError("consolidated dataframe index contains duplicates")

    if _PANDERA_AVAILABLE:
        schema = pa.DataFrameSchema(
            {
                r"^\d{4}/\d{1,2}$": Column(float, nullable=True, required=False, regex=True),
            },
            index=pa.MultiIndex(
                [
                    pa.Index(str, name="ticker"),
                    pa.Index(str, name="sheet_name"),
                    pa.Index(str, name="row_name"),
                ]
            ),
            strict=False,
            coerce=True,
        )
        try:
            frame = schema.validate(frame, lazy=True)
        except Exception as exc:
            raise SchemaValidationError(f"consolidated schema validation failed: {exc}") from exc

    return frame


def validate_staleness_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate staleness/freshness report schema."""
    required = [
        "ticker",
        "latest_period",
        "period_end",
        "staleness_days",
        "has_q4_2025",
    ]
    _ensure_columns(frame, required, "staleness_report")

    validated = frame.copy()
    validated["ticker"] = validated["ticker"].astype(str)
    validated["period_end"] = pd.to_datetime(validated["period_end"], errors="coerce")
    validated["staleness_days"] = pd.to_numeric(validated["staleness_days"], errors="coerce")
    validated["has_q4_2025"] = validated["has_q4_2025"].fillna(False).astype(bool)

    if _PANDERA_AVAILABLE:
        schema = pa.DataFrameSchema(
            {
                "ticker": Column(str, nullable=False),
                "latest_period": Column(object, nullable=True),
                "period_end": Column(pa.DateTime, nullable=True),
                "staleness_days": Column(float, nullable=True, checks=Check.ge(0)),
                "has_q4_2025": Column(bool, nullable=False),
            },
            strict=False,
            coerce=True,
        )
        try:
            validated = schema.validate(validated, lazy=True)
        except Exception as exc:
            raise SchemaValidationError(f"staleness report schema validation failed: {exc}") from exc

    return validated


def pandera_available() -> bool:
    """Expose Pandera availability for diagnostics and tests."""
    return _PANDERA_AVAILABLE
