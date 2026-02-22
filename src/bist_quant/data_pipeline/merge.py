from __future__ import annotations

from typing import Any

import pandas as pd

from bist_quant.data_pipeline.errors import MergeError
from bist_quant.data_pipeline.schemas import validate_consolidated_panel


def _normalize_row_name_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.lower()
    )


def merge_consolidated_panels(
    *,
    existing: pd.DataFrame,
    new_data: pd.DataFrame,
    prefer_existing_values: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Merge two consolidated-format panels using vectorized row-key normalization.

    Behavior:
    - Align rows by (ticker, sheet_name, normalized(row_name)).
    - Use combine_first semantics to avoid iterative row-by-row merges.
    - Preserve full index/column union.
    """
    if new_data is None or new_data.empty:
        validated_existing = validate_consolidated_panel(existing)
        return validated_existing, {
            "existing_rows": int(existing.shape[0]),
            "new_rows": 0,
            "merged_rows": int(existing.shape[0]),
            "new_columns_added": 0,
            "cells_filled_from_new": 0,
            "cells_overwritten_from_new": 0,
        }

    if existing is None or existing.empty:
        validated_new = validate_consolidated_panel(new_data)
        return validated_new, {
            "existing_rows": 0,
            "new_rows": int(validated_new.shape[0]),
            "merged_rows": int(validated_new.shape[0]),
            "new_columns_added": int(validated_new.shape[1]),
            "cells_filled_from_new": int(validated_new.notna().sum().sum()),
            "cells_overwritten_from_new": 0,
        }

    try:
        existing = validate_consolidated_panel(existing)
        new_data = validate_consolidated_panel(new_data)

        existing = existing.groupby(level=["ticker", "sheet_name", "row_name"]).first()
        new_data = new_data.groupby(level=["ticker", "sheet_name", "row_name"]).first()

        existing_reset = existing.reset_index()
        existing_reset["_row_key"] = _normalize_row_name_series(existing_reset["row_name"])

        existing_key_map = (
            existing_reset[["ticker", "sheet_name", "_row_key", "row_name"]]
            .drop_duplicates(subset=["ticker", "sheet_name", "_row_key"], keep="first")
            .rename(columns={"row_name": "row_name_existing"})
        )

        new_reset = new_data.reset_index()
        new_reset["_row_key"] = _normalize_row_name_series(new_reset["row_name"])

        new_aligned = new_reset.merge(
            existing_key_map,
            on=["ticker", "sheet_name", "_row_key"],
            how="left",
        )
        new_aligned["row_name"] = new_aligned["row_name_existing"].fillna(new_aligned["row_name"])

        drop_cols = [col for col in ["row_name_existing", "_row_key"] if col in new_aligned.columns]
        new_aligned = new_aligned.drop(columns=drop_cols)
        new_aligned = new_aligned.set_index(["ticker", "sheet_name", "row_name"])
        new_aligned = new_aligned.groupby(level=["ticker", "sheet_name", "row_name"]).first()

        all_columns = sorted(set(existing.columns).union(new_aligned.columns), key=str)
        existing_aligned = existing.reindex(columns=all_columns)
        new_aligned = new_aligned.reindex(columns=all_columns)

        if prefer_existing_values:
            merged = existing_aligned.combine_first(new_aligned)
        else:
            merged = new_aligned.combine_first(existing_aligned)

        existing_notna = existing_aligned.notna()
        new_notna = new_aligned.notna()
        merged_notna = merged.notna()

        cells_filled_from_new = int((~existing_notna & new_notna & merged_notna).sum().sum())
        cells_overwritten_from_new = int(
            (
                existing_notna
                & new_notna
                & (existing_aligned != new_aligned)
                & (merged == new_aligned)
            ).sum().sum()
        )

        merged = validate_consolidated_panel(merged)

        stats = {
            "existing_rows": int(existing.shape[0]),
            "new_rows": int(new_data.shape[0]),
            "merged_rows": int(merged.shape[0]),
            "new_columns_added": int(len(set(all_columns) - set(existing.columns))),
            "cells_filled_from_new": cells_filled_from_new,
            "cells_overwritten_from_new": cells_overwritten_from_new,
        }
        return merged, stats
    except Exception as exc:  # pragma: no cover - defensive path
        raise MergeError(f"Vectorized fundamentals merge failed: {exc}") from exc
