"""Stock screener computation engine."""

from __future__ import annotations

import io
import logging
import time
import unicodedata
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bist_quant.engines.errors import (
    QuantEngineDataError,
    QuantEngineError,
    QuantEngineExecutionError,
    QuantEngineValidationError,
)
from bist_quant.engines.types import StockScreenerResult
from bist_quant.runtime import RuntimePathError, RuntimePaths, resolve_runtime_paths, validate_runtime_paths

LOGGER = logging.getLogger("bist_quant.engines.stock_filter")

try:  # optional dependency
    import borsapy as bp
except Exception:  # pragma: no cover
    bp = None

from bist_quant.common.data_loader import DataLoader  # noqa: E402
from bist_quant.engines.technical_scanner import TechnicalScannerEngine  # noqa: E402
from bist_quant.signals.borsapy_indicators import BorsapyIndicators  # noqa: E402


def _resolve_paths(runtime_paths: RuntimePaths | None) -> RuntimePaths:
    resolved = runtime_paths or resolve_runtime_paths()
    validate_runtime_paths(resolved, require_price_data=True)
    return resolved


FILTER_FIELD_DEFS: list[dict[str, Any]] = [
    {"key": "market_cap_usd", "label": "Market Cap (USD mn)", "group": "valuation"},
    {"key": "market_cap", "label": "Market Cap (TL mn)", "group": "valuation"},
    {"key": "pe", "label": "P/E", "group": "valuation"},
    {"key": "forward_pe", "label": "Forward P/E", "group": "valuation"},
    {"key": "pb", "label": "P/B", "group": "valuation"},
    {"key": "ev_ebitda", "label": "EV/EBITDA", "group": "valuation"},
    {"key": "ev_sales", "label": "EV/Sales", "group": "valuation"},
    {"key": "dividend_yield", "label": "Dividend Yield (%)", "group": "income"},
    {"key": "upside_potential", "label": "Upside Potential (%)", "group": "analyst"},
    {"key": "analyst_target_price", "label": "Analyst Target Price", "group": "analyst"},
    {"key": "roe", "label": "ROE (%)", "group": "quality"},
    {"key": "roa", "label": "ROA (%)", "group": "quality"},
    {"key": "net_margin", "label": "Net Margin (%)", "group": "quality"},
    {"key": "ebitda_margin", "label": "EBITDA Margin (%)", "group": "quality"},
    {"key": "foreign_ratio", "label": "Foreign Ownership (%)", "group": "flow"},
    {"key": "foreign_change_1w", "label": "Foreign Ownership 1W Change (pp)", "group": "flow"},
    {"key": "foreign_change_1m", "label": "Foreign Ownership 1M Change (pp)", "group": "flow"},
    {"key": "float_ratio", "label": "Free Float (%)", "group": "flow"},
    {"key": "volume_3m", "label": "Avg Volume 3M (mn)", "group": "liquidity"},
    {"key": "volume_12m", "label": "Avg Volume 12M (mn)", "group": "liquidity"},
    {"key": "return_1w", "label": "Return 1W (%)", "group": "momentum"},
    {"key": "return_1m", "label": "Return 1M (%)", "group": "momentum"},
    {"key": "return_1y", "label": "Return 1Y (%)", "group": "momentum"},
    {"key": "return_ytd", "label": "Return YTD (%)", "group": "momentum"},
    {"key": "rsi_14", "label": "RSI 14", "group": "technical"},
    {"key": "macd_hist", "label": "MACD Histogram", "group": "technical"},
    {"key": "atr_14_pct", "label": "ATR 14 (% of Price)", "group": "technical"},
    {"key": "revenue_growth_yoy", "label": "Revenue Growth YoY (%)", "group": "growth"},
    {"key": "net_income_growth_yoy", "label": "Net Income Growth YoY (%)", "group": "growth"},
]

FIELD_LABELS = {row["key"]: row["label"] for row in FILTER_FIELD_DEFS}

DISPLAY_COLUMNS_DEFAULT: list[str] = [
    "symbol",
    "name",
    "market_cap_usd",
    "pe",
    "forward_pe",
    "pb",
    "dividend_yield",
    "upside_potential",
    "roe",
    "net_margin",
    "rsi_14",
    "return_1m",
    "recommendation",
]

INDEX_OPTIONS = ["XU030", "XU050", "XU100", "XUTUM", "CUSTOM"]
RECOMMENDATION_OPTIONS = ["AL", "TUT", "SAT"]

DEFAULT_TEMPLATES = [
    "small_cap",
    "mid_cap",
    "large_cap",
    "high_dividend",
    "high_upside",
    "low_upside",
    "high_volume",
    "low_volume",
    "buy_recommendation",
    "sell_recommendation",
    "high_net_margin",
    "high_return",
    "low_pe",
    "high_roe",
    "high_foreign_ownership",
]

DATA_SOURCE_OPTIONS = ["local", "isyatirim", "hybrid"]

TEMPLATE_PRESETS: dict[str, dict[str, Any]] = {
    "small_cap": {"market_cap_usd": {"max": 1_000.0}},
    "mid_cap": {"market_cap_usd": {"min": 1_000.0, "max": 10_000.0}},
    "large_cap": {"market_cap_usd": {"min": 10_000.0}},
    "high_dividend": {"dividend_yield": {"min": 4.0}},
    "high_upside": {"upside_potential": {"min": 15.0}},
    "low_upside": {"upside_potential": {"max": 5.0}},
    "high_volume": {"volume_3m": {"min": 20.0}},
    "low_volume": {"volume_3m": {"max": 2.0}},
    "buy_recommendation": {"recommendation": "AL"},
    "sell_recommendation": {"recommendation": "SAT"},
    "high_net_margin": {"net_margin": {"min": 10.0}},
    "high_return": {"return_1y": {"min": 20.0}},
    "low_pe": {"pe": {"max": 12.0}},
    "high_roe": {"roe": {"min": 15.0}},
    "high_foreign_ownership": {"foreign_ratio": {"min": 50.0}},
}

SCREEN_CACHE_TTL_SEC = 600
_SCREEN_CACHE: dict[str, Any] = {
    "built_at": 0.0,
    "as_of": None,
    "frame": None,
    "sector_map": None,
    "close_df": None,
    "data_dir": None,
    "state_token": None,
}


def _as_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(minimum, min(maximum, parsed))


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
        if np.isnan(parsed) or np.isinf(parsed):
            return None
        return parsed
    except Exception:
        return None


def _safe_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _normalize_data_source(value: Any) -> str:
    source = str(value or "local").strip().lower()
    if source not in DATA_SOURCE_OPTIONS:
        raise ValueError(f"Unknown data_source: {source}")
    return source


def _as_symbol_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    values: list[str]
    if isinstance(raw, str):
        values = [item.strip() for item in raw.replace(";", ",").split(",")]
    elif isinstance(raw, list):
        values = [str(item).strip() for item in raw]
    else:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        base = item.upper().split(".")[0]
        if not base or base in seen:
            continue
        seen.add(base)
        out.append(base)
    return out


def _normalize_condition_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        values = [raw.strip()]
    elif isinstance(raw, list):
        values = [str(item).strip() for item in raw]
    else:
        return []

    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _resolve_technical_conditions(
    payload: dict[str, Any],
    scanner: TechnicalScannerEngine,
) -> tuple[list[str], list[str]]:
    predefined = scanner.predefined_scans()
    conditions: list[str] = []
    templates: list[str] = []

    def _add_condition(value: Any) -> None:
        for item in _normalize_condition_list(value):
            if item not in conditions:
                conditions.append(item)

    def _add_template(name: Any) -> None:
        key = str(name or "").strip().lower()
        if not key:
            return
        expr = predefined.get(key)
        if expr is None:
            raise ValueError(f"Unknown technical scan template: {key}")
        if key not in templates:
            templates.append(key)
        if expr not in conditions:
            conditions.append(expr)

    _add_template(payload.get("technical_scan_name"))
    _add_template(payload.get("technical_template"))

    scan_value = payload.get("technical_scan")
    if isinstance(scan_value, str):
        key = scan_value.strip()
        key_normalized = key.lower()
        if key_normalized in predefined:
            _add_template(key_normalized)
        else:
            _add_condition(key)
    else:
        for item in _normalize_condition_list(scan_value):
            item_normalized = item.lower()
            if item_normalized in predefined:
                _add_template(item_normalized)
            else:
                _add_condition(item)

    _add_condition(payload.get("technical_condition"))
    _add_condition(payload.get("technical_conditions"))
    return conditions, templates


def _apply_technical_scan(
    df: pd.DataFrame,
    payload: dict[str, Any],
    scanner: TechnicalScannerEngine,
) -> tuple[pd.DataFrame, list[str], list[str], str, list[str]]:
    conditions, templates = _resolve_technical_conditions(payload, scanner)
    if not conditions:
        return df, [], [], "1d", []

    interval = str(payload.get("technical_interval") or payload.get("interval") or "1d").strip() or "1d"
    if "symbol" in df.columns:
        universe = _as_symbol_list(df["symbol"].tolist())
    else:
        universe = []
    if not universe:
        return df.iloc[0:0], conditions, templates, interval, []

    if len(conditions) == 1:
        scan_df = scanner.scan(
            universe=universe,
            condition=conditions[0],
            interval=interval,
        )
    else:
        scan_df = scanner.scan_multi(
            universe=universe,
            conditions=conditions,
            interval=interval,
        )

    if scan_df is None or scan_df.empty or "symbol" not in scan_df.columns:
        return df.iloc[0:0], conditions, templates, interval, []

    out = scan_df.copy()
    out["symbol"] = out["symbol"].astype(str).str.upper().str.split(".").str[0]
    out = out[out["symbol"] != ""]
    out = out.drop_duplicates(subset=["symbol"], keep="last")
    if out.empty:
        return df.iloc[0:0], conditions, templates, interval, []

    technical_columns = [col for col in out.columns if col != "symbol"]
    merge_columns = ["symbol"] + [col for col in technical_columns if col not in df.columns]

    if len(merge_columns) > 1:
        merged = df.merge(out.loc[:, merge_columns], on="symbol", how="inner")
    else:
        merged = df[df["symbol"].isin(out["symbol"])].copy()

    visible_technical_columns = [col for col in technical_columns if col in merged.columns]
    return merged, conditions, templates, interval, visible_technical_columns


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return str(value)


def _normalize_text(value: Any) -> str:
    # Normalize Turkish-specific characters so keyword matching remains robust
    # across mixed-language financial statement labels.
    translation = str.maketrans(
        {
            "ı": "i",
            "İ": "i",
            "ş": "s",
            "Ş": "s",
            "ğ": "g",
            "Ğ": "g",
            "ç": "c",
            "Ç": "c",
            "ö": "o",
            "Ö": "o",
            "ü": "u",
            "Ü": "u",
        }
    )
    text = unicodedata.normalize("NFKD", str(value or "")).translate(translation)
    return "".join(ch for ch in text if not unicodedata.combining(ch)).lower().strip()


def _normalize_column_name(value: Any) -> str:
    raw = _normalize_text(value)
    chars = [ch if ch.isalnum() else "_" for ch in raw]
    normalized = "".join(chars)
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def _extract_alias_series(
    frame: pd.DataFrame,
    normalized_columns: dict[str, str],
    aliases: tuple[str, ...],
) -> pd.Series:
    for alias in aliases:
        if alias in normalized_columns:
            return frame[normalized_columns[alias]]
        if len(alias) >= 6:
            for key, column in normalized_columns.items():
                if alias in key:
                    return frame[column]
    return pd.Series(np.nan, index=frame.index, dtype="object")


def _extract_numeric_from_dict(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    if not payload:
        return None
    normalized = {_normalize_column_name(key): value for key, value in payload.items()}
    for key in keys:
        target = _normalize_column_name(key)
        if target in normalized:
            return _as_float(normalized[target])
        if len(target) >= 6:
            for item_key, item_value in normalized.items():
                if target in item_key:
                    return _as_float(item_value)
    return None


def _extract_text_from_dict(payload: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    if not payload:
        return None
    normalized = {_normalize_column_name(key): value for key, value in payload.items()}
    for key in keys:
        target = _normalize_column_name(key)
        if target in normalized:
            text = str(normalized[target] or "").strip()
            return text or None
        if len(target) >= 6:
            for item_key, item_value in normalized.items():
                if target in item_key:
                    text = str(item_value or "").strip()
                    return text or None
    return None


def _normalize_recommendation_value(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    if not text:
        return None
    mapping = {
        "BUY": "AL",
        "OUTPERFORM": "AL",
        "OVERWEIGHT": "AL",
        "AL": "AL",
        "HOLD": "TUT",
        "NEUTRAL": "TUT",
        "TUT": "TUT",
        "SELL": "SAT",
        "UNDERWEIGHT": "SAT",
        "SAT": "SAT",
    }
    return mapping.get(text, text)


def _ensure_screen_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    required = {"symbol", "name", "recommendation", "sector"}
    required.update(FIELD_LABELS.keys())
    required.update({"rsi_14", "macd_hist", "atr_14_pct"})
    for key in required:
        if key not in out.columns:
            out[key] = np.nan if key not in {"symbol", "name", "recommendation", "sector"} else ""
    out["symbol"] = out["symbol"].astype(str).str.upper().str.split(".").str[0]
    out = out[out["symbol"] != ""]
    out = out.drop_duplicates(subset=["symbol"], keep="last")
    name_text = out["name"].astype(str).str.strip()
    sector_text = out["sector"].astype(str).str.strip()
    out["name"] = out["name"].where((name_text != "") & (name_text.str.lower() != "nan"), out["symbol"])
    out["sector"] = out["sector"].where(
        (sector_text != "") & (sector_text.str.lower() != "nan"),
        "Unknown",
    )
    return out


def _normalize_isyatirim_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _ensure_screen_columns(pd.DataFrame(columns=["symbol", "name"]))

    source = frame.copy()
    source.columns = [str(col) for col in source.columns]
    normalized_columns = {_normalize_column_name(col): col for col in source.columns}

    out = pd.DataFrame(index=source.index)
    out["symbol"] = _extract_alias_series(
        source,
        normalized_columns,
        ("symbol", "ticker", "code", "hisse", "sembol", "kod"),
    )
    out["name"] = _extract_alias_series(
        source,
        normalized_columns,
        ("name", "company_name", "company", "title", "short_name", "unvan"),
    )
    out["market_cap_usd"] = _extract_alias_series(
        source,
        normalized_columns,
        ("market_cap_usd", "marketcap_usd", "piyasa_degeri_usd", "pd_usd"),
    )
    out["market_cap"] = _extract_alias_series(
        source,
        normalized_columns,
        ("market_cap", "market_cap_tl", "marketcap_tl", "piyasa_degeri_tl", "piyasa_degeri"),
    )
    out["pe"] = _extract_alias_series(
        source,
        normalized_columns,
        ("pe", "fk", "f_k", "price_earnings"),
    )
    out["forward_pe"] = _extract_alias_series(
        source,
        normalized_columns,
        ("forward_pe", "fwd_pe", "pe_forward", "ileri_fk", "ileri_f_k"),
    )
    out["pb"] = _extract_alias_series(
        source,
        normalized_columns,
        ("pb", "pd_dd", "price_to_book"),
    )
    out["ev_ebitda"] = _extract_alias_series(
        source,
        normalized_columns,
        ("ev_ebitda", "fd_favok"),
    )
    out["ev_sales"] = _extract_alias_series(
        source,
        normalized_columns,
        ("ev_sales", "fd_satis"),
    )
    out["dividend_yield"] = _extract_alias_series(
        source,
        normalized_columns,
        ("dividend_yield", "div_yield", "temettu_verimi"),
    )
    out["upside_potential"] = _extract_alias_series(
        source,
        normalized_columns,
        ("upside_potential", "upside", "prim_potansiyeli"),
    )
    out["analyst_target_price"] = _extract_alias_series(
        source,
        normalized_columns,
        ("analyst_target_price", "target_price", "hedef_fiyat", "consensus_target"),
    )
    out["recommendation"] = _extract_alias_series(
        source,
        normalized_columns,
        ("recommendation", "rating", "analyst_recommendation", "oneri"),
    )
    out["foreign_ratio"] = _extract_alias_series(
        source,
        normalized_columns,
        ("foreign_ratio", "foreign_ownership", "yabanci_orani", "yabanci_payi"),
    )
    out["foreign_change_1w"] = _extract_alias_series(
        source,
        normalized_columns,
        ("foreign_change_1w", "foreign_ratio_change_1w", "foreign_1w_change", "yabanci_degisim_1h"),
    )
    out["foreign_change_1m"] = _extract_alias_series(
        source,
        normalized_columns,
        ("foreign_change_1m", "foreign_ratio_change_1m", "foreign_1m_change", "yabanci_degisim_1a"),
    )
    out["float_ratio"] = _extract_alias_series(
        source,
        normalized_columns,
        ("float_ratio", "free_float", "fiili_dolasim_orani"),
    )
    out["volume_3m"] = _extract_alias_series(
        source,
        normalized_columns,
        ("volume_3m", "avg_volume_3m", "ortalama_hacim_3ay"),
    )
    out["volume_12m"] = _extract_alias_series(
        source,
        normalized_columns,
        ("volume_12m", "avg_volume_12m", "ortalama_hacim_12ay"),
    )
    out["return_1w"] = _extract_alias_series(source, normalized_columns, ("return_1w", "getiri_1h"))
    out["return_1m"] = _extract_alias_series(source, normalized_columns, ("return_1m", "getiri_1a"))
    out["return_1y"] = _extract_alias_series(source, normalized_columns, ("return_1y", "getiri_1y"))
    out["return_ytd"] = _extract_alias_series(source, normalized_columns, ("return_ytd", "ytd"))
    out["roe"] = _extract_alias_series(source, normalized_columns, ("roe",))
    out["roa"] = _extract_alias_series(source, normalized_columns, ("roa",))
    out["net_margin"] = _extract_alias_series(
        source,
        normalized_columns,
        ("net_margin", "net_kar_marji"),
    )
    out["ebitda_margin"] = _extract_alias_series(
        source,
        normalized_columns,
        ("ebitda_margin", "favok_marji"),
    )
    out["revenue_growth_yoy"] = _extract_alias_series(
        source,
        normalized_columns,
        ("revenue_growth_yoy", "ciro_buyume"),
    )
    out["net_income_growth_yoy"] = _extract_alias_series(
        source,
        normalized_columns,
        ("net_income_growth_yoy", "net_kar_buyume"),
    )
    out["sector"] = _extract_alias_series(
        source,
        normalized_columns,
        ("sector", "sektor", "industry"),
    )

    latest_price = pd.to_numeric(
        _extract_alias_series(
            source,
            normalized_columns,
            ("last_price", "price", "close", "kapanis", "son_fiyat"),
        ),
        errors="coerce",
    )
    target_price = pd.to_numeric(out["analyst_target_price"], errors="coerce")
    calculated_upside = _safe_divide(target_price - latest_price, latest_price) * 100.0
    out["upside_potential"] = pd.to_numeric(out["upside_potential"], errors="coerce").combine_first(
        calculated_upside
    )

    out["recommendation"] = out["recommendation"].map(_normalize_recommendation_value)
    out = _ensure_screen_columns(out)

    numeric_fields = [col for col in FIELD_LABELS.keys() if col in out.columns]
    for col in numeric_fields + ["rsi_14", "macd_hist", "atr_14_pct"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out.reset_index(drop=True)


def _merge_hybrid_fields(local_frame: pd.DataFrame, isy_frame: pd.DataFrame) -> pd.DataFrame:
    if isy_frame.empty:
        return local_frame

    local = _ensure_screen_columns(local_frame)
    remote = _ensure_screen_columns(isy_frame)
    merge_cols = [
        "symbol",
        "forward_pe",
        "analyst_target_price",
        "upside_potential",
        "recommendation",
        "foreign_ratio",
        "foreign_change_1w",
        "foreign_change_1m",
        "float_ratio",
        "sector",
    ]
    available_cols = [col for col in merge_cols if col in remote.columns]
    merged = local.merge(remote[available_cols], on="symbol", how="left", suffixes=("", "_isy"))

    numeric_overlay = [
        "forward_pe",
        "analyst_target_price",
        "foreign_ratio",
        "foreign_change_1w",
        "foreign_change_1m",
        "float_ratio",
    ]
    for col in numeric_overlay:
        extra_col = f"{col}_isy"
        if extra_col not in merged.columns:
            continue
        merged[col] = pd.to_numeric(merged[col], errors="coerce").combine_first(
            pd.to_numeric(merged[extra_col], errors="coerce")
        )

    if "upside_potential_isy" in merged.columns:
        merged["upside_potential"] = pd.to_numeric(
            merged["upside_potential_isy"], errors="coerce"
        ).combine_first(pd.to_numeric(merged["upside_potential"], errors="coerce"))

    if "recommendation_isy" in merged.columns:
        normalized = merged["recommendation_isy"].map(_normalize_recommendation_value)
        merged["recommendation"] = normalized.where(normalized.notna(), merged["recommendation"])

    if "sector_isy" in merged.columns:
        local_sector = merged["sector"].astype(str).str.strip().str.lower()
        merged["sector"] = merged["sector"].where(
            (local_sector != "") & (local_sector != "nan"),
            merged["sector_isy"],
        )

    merged = merged.drop(columns=[col for col in merged.columns if col.endswith("_isy")])
    return _ensure_screen_columns(merged).reset_index(drop=True)


def _normalize_filters(raw: Any, allowed: set[str]) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    if not isinstance(raw, dict):
        return out

    for key, value in raw.items():
        name = str(key).strip()
        if not name or name not in allowed:
            continue
        if not isinstance(value, dict):
            continue

        minimum = _as_float(value.get("min"))
        maximum = _as_float(value.get("max"))
        if minimum is None and maximum is None:
            continue
        if minimum is not None and maximum is not None and minimum > maximum:
            minimum, maximum = maximum, minimum

        out[name] = {"min": minimum, "max": maximum}
    return out


def _normalize_percentile_filters(raw: Any, allowed: set[str]) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    if not isinstance(raw, dict):
        return out

    for key, value in raw.items():
        name = str(key).strip()
        if not name or name not in allowed:
            continue
        if not isinstance(value, dict):
            continue

        minimum = _as_float(value.get("min_pct"))
        maximum = _as_float(value.get("max_pct"))

        if minimum is None and maximum is None:
            continue

        if minimum is not None:
            minimum = minimum / 100.0 if minimum > 1.0 else minimum
            minimum = max(0.0, min(1.0, minimum))
        if maximum is not None:
            maximum = maximum / 100.0 if maximum > 1.0 else maximum
            maximum = max(0.0, min(1.0, maximum))
        if minimum is not None and maximum is not None and minimum > maximum:
            minimum, maximum = maximum, minimum

        out[name] = {"min_pct": minimum, "max_pct": maximum}
    return out


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = denominator.replace(0, np.nan)
    out = numerator / den
    return out.replace([np.inf, -np.inf], np.nan)


def _quarter_sort_key(value: Any) -> tuple[int, int]:
    text = str(value)
    parts = text.split("/")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return int(parts[0]), int(parts[1])
    return (-1, -1)


def _extract_metric_pair(
    flat: pd.DataFrame,
    latest_col: str,
    prev_col: str | None,
    tokens: list[str],
) -> tuple[pd.Series, pd.Series]:
    if flat.empty:
        return pd.Series(dtype="float64"), pd.Series(dtype="float64")

    mask = pd.Series(True, index=flat.index)
    for token in tokens:
        mask = mask & flat["row_norm"].str.contains(token, regex=False, na=False)

    subset = flat.loc[mask].copy()
    if subset.empty:
        return pd.Series(dtype="float64"), pd.Series(dtype="float64")

    subset["_latest"] = pd.to_numeric(subset[latest_col], errors="coerce")
    latest_series = subset.dropna(subset=["_latest"]).groupby("ticker")["_latest"].first()

    if prev_col is None or prev_col not in subset.columns:
        return latest_series.astype("float64"), pd.Series(dtype="float64")

    subset["_prev"] = pd.to_numeric(subset[prev_col], errors="coerce")
    prev_series = subset.dropna(subset=["_prev"]).groupby("ticker")["_prev"].first()

    return latest_series.astype("float64"), prev_series.astype("float64")


def _extract_fundamental_snapshot(loader: DataLoader, symbols: list[str]) -> pd.DataFrame:
    try:
        with redirect_stdout(io.StringIO()):
            panel = loader.load_fundamentals_parquet()
    except Exception:
        panel = None

    out = pd.DataFrame(index=symbols)
    columns = [
        "net_income",
        "net_income_prev",
        "revenue",
        "revenue_prev",
        "equity",
        "assets",
        "liabilities",
        "ebitda",
        "dividends",
        "cash",
        "debt",
    ]
    for col in columns:
        out[col] = np.nan

    if panel is None or not isinstance(panel, pd.DataFrame) or panel.empty:
        return out

    if not isinstance(panel.index, pd.MultiIndex):
        return out

    flat = panel.reset_index()
    if "ticker" not in flat.columns or "row_name" not in flat.columns:
        return out

    flat["ticker"] = flat["ticker"].astype(str).str.upper().str.split(".").str[0]
    flat = flat[flat["ticker"].isin(symbols)].copy()
    if flat.empty:
        return out

    quarter_cols = [
        c
        for c in flat.columns
        if c not in {"ticker", "sheet_name", "row_name"}
    ]
    quarter_cols = sorted(quarter_cols, key=_quarter_sort_key, reverse=True)
    if not quarter_cols:
        return out

    latest_col = str(quarter_cols[0])
    latest_year, latest_month = _quarter_sort_key(latest_col)
    desired_prev = f"{latest_year - 1}/{latest_month}" if latest_year > 0 and latest_month > 0 else None
    prev_col: str | None = None
    if desired_prev and desired_prev in quarter_cols:
        prev_col = desired_prev
    elif len(quarter_cols) > 4:
        prev_col = str(quarter_cols[4])
    elif len(quarter_cols) > 1:
        prev_col = str(quarter_cols[1])

    flat["row_norm"] = flat["row_name"].map(_normalize_text)

    net_income, net_income_prev = _extract_metric_pair(flat, latest_col, prev_col, ["donem", "net", "kari"])
    if net_income.empty:
        net_income, net_income_prev = _extract_metric_pair(flat, latest_col, prev_col, ["donem", "kari"])

    revenue, revenue_prev = _extract_metric_pair(flat, latest_col, prev_col, ["satis", "gelir"])
    equity, _ = _extract_metric_pair(flat, latest_col, None, ["ozkaynak"])
    assets, _ = _extract_metric_pair(flat, latest_col, None, ["toplam", "varlik"])
    liabilities, _ = _extract_metric_pair(flat, latest_col, None, ["toplam", "yukumluluk"])

    if liabilities.empty:
        kv, _ = _extract_metric_pair(flat, latest_col, None, ["kisa", "vadeli", "yukumluluk"])
        uv, _ = _extract_metric_pair(flat, latest_col, None, ["uzun", "vadeli", "yukumluluk"])
        liabilities = kv.add(uv, fill_value=np.nan)

    ebitda, _ = _extract_metric_pair(flat, latest_col, None, ["favok"])
    dividends, _ = _extract_metric_pair(flat, latest_col, None, ["odenen", "temettu"])
    cash, _ = _extract_metric_pair(flat, latest_col, None, ["nakit", "nakit", "benzer"])
    if cash.empty:
        cash, _ = _extract_metric_pair(flat, latest_col, None, ["nakit", "nakit"])

    debt, _ = _extract_metric_pair(flat, latest_col, None, ["finansal", "borc"])
    if debt.empty:
        debt = liabilities

    out.loc[net_income.index, "net_income"] = net_income.values
    out.loc[net_income_prev.index, "net_income_prev"] = net_income_prev.values
    out.loc[revenue.index, "revenue"] = revenue.values
    out.loc[revenue_prev.index, "revenue_prev"] = revenue_prev.values
    out.loc[equity.index, "equity"] = equity.values
    out.loc[assets.index, "assets"] = assets.values
    out.loc[liabilities.index, "liabilities"] = liabilities.values
    out.loc[ebitda.index, "ebitda"] = ebitda.values
    out.loc[dividends.index, "dividends"] = dividends.values
    out.loc[cash.index, "cash"] = cash.values
    out.loc[debt.index, "debt"] = debt.values

    return out


def _load_sector_map(data_dir: Path) -> dict[str, str]:
    csv_path = data_dir / "bist_sector_classification.csv"
    if not csv_path.exists():
        return {}
    try:
        frame = pd.read_csv(csv_path)
    except Exception:
        return {}
    if frame.empty or "ticker" not in frame.columns or "sector" not in frame.columns:
        return {}

    out: dict[str, str] = {}
    for _, row in frame.iterrows():
        ticker = str(row.get("ticker", "")).upper().split(".")[0]
        sector = str(row.get("sector", "")).strip()
        if ticker:
            out[ticker] = sector or "Unknown"
    return out


def _get_index_components(index_name: str) -> list[str]:
    if bp is None:
        return []
    try:
        idx = bp.index(index_name)
    except Exception:
        return []

    symbols: list[str] = []
    component_symbols = getattr(idx, "component_symbols", None)
    if isinstance(component_symbols, list):
        symbols = [str(s) for s in component_symbols if s]

    if not symbols:
        components = getattr(idx, "components", None)
        if isinstance(components, list):
            for item in components:
                if isinstance(item, dict):
                    symbol = str(item.get("symbol", "")).strip()
                    if symbol:
                        symbols.append(symbol)

    out: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        base = symbol.upper().split(".")[0]
        if not base or base in seen:
            continue
        seen.add(base)
        out.append(base)
    return out


def _screen_cache_state_token(data_dir: Path) -> str:
    candidates = [
        data_dir / "bist_prices_full.parquet",
        data_dir / "bist_prices_full.csv",
        data_dir / "bist_prices_full.csv.gz",
        data_dir / "fundamental_data_consolidated.parquet",
        data_dir / "fundamental_data_consolidated.csv",
        data_dir / "fundamental_data_consolidated.csv.gz",
        data_dir / "shares_outstanding_consolidated.parquet",
        data_dir / "shares_outstanding_consolidated.csv",
        data_dir / "shares_outstanding_consolidated.csv.gz",
        data_dir / "bist_sector_classification.parquet",
        data_dir / "bist_sector_classification.csv",
    ]
    rows: list[str] = []
    for path in candidates:
        try:
            stat = path.stat()
            rows.append(f"{path}:{stat.st_mtime_ns}:{stat.st_size}")
        except FileNotFoundError:
            rows.append(f"{path}:missing")
        except OSError:
            rows.append(f"{path}:error")
    return str(hash(tuple(rows)))


def _load_local_screen_frame(
    runtime_paths: RuntimePaths,
    *,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, str, dict[str, str]]:
    now = time.time()
    expected_state_token = _screen_cache_state_token(runtime_paths.data_dir)
    cached_frame = _SCREEN_CACHE.get("frame")
    cached_as_of = _SCREEN_CACHE.get("as_of")
    cached_sector_map = _SCREEN_CACHE.get("sector_map")
    cached_close_df = _SCREEN_CACHE.get("close_df")
    cached_data_dir = _SCREEN_CACHE.get("data_dir")
    cached_state_token = _SCREEN_CACHE.get("state_token")
    built_at = float(_SCREEN_CACHE.get("built_at", 0.0) or 0.0)
    if (
        not force_refresh
        and isinstance(cached_frame, pd.DataFrame)
        and cached_as_of is not None
        and isinstance(cached_sector_map, dict)
        and isinstance(cached_close_df, pd.DataFrame)
        and str(cached_data_dir) == str(runtime_paths.data_dir)
        and str(cached_state_token) == expected_state_token
        and now - built_at <= SCREEN_CACHE_TTL_SEC
    ):
        return cached_frame.copy(), str(cached_as_of), dict(cached_sector_map)

    data_dir = runtime_paths.data_dir
    prices_file = data_dir / "bist_prices_full.csv"
    if not prices_file.exists() and not prices_file.with_suffix(".parquet").exists():
        gz_candidate = data_dir / "bist_prices_full.csv.gz"
        if gz_candidate.exists():
            prices_file = gz_candidate
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    loader = DataLoader(data_dir=data_dir, regime_model_dir=runtime_paths.regime_outputs_dir)
    with redirect_stdout(io.StringIO()):
        prices = loader.load_prices(prices_file)

    if prices is None or prices.empty:
        raise ValueError("Local price data is unavailable for stock screening.")

    local = prices.copy()
    local["Date"] = pd.to_datetime(local["Date"], errors="coerce")
    local = local.dropna(subset=["Date"]).copy()
    local["Ticker"] = local["Ticker"].astype(str).str.split(".").str[0].str.upper()

    close_df = local.pivot_table(index="Date", columns="Ticker", values="Close", aggfunc="last").sort_index().ffill()
    high_df = local.pivot_table(index="Date", columns="Ticker", values="High", aggfunc="last").sort_index().ffill()
    low_df = local.pivot_table(index="Date", columns="Ticker", values="Low", aggfunc="last").sort_index().ffill()
    volume_df = local.pivot_table(index="Date", columns="Ticker", values="Volume", aggfunc="last").sort_index().ffill()

    if close_df.empty:
        raise ValueError("No usable close price data for stock filter.")

    symbols = [str(c) for c in close_df.columns]
    as_of = close_df.index.max()
    as_of_iso = pd.Timestamp(as_of).date().isoformat()

    latest_close = close_df.iloc[-1]
    ret_1w = (close_df.pct_change(5).iloc[-1] * 100.0) if len(close_df) > 5 else pd.Series(index=symbols, dtype="float64")
    ret_1m = (close_df.pct_change(21).iloc[-1] * 100.0) if len(close_df) > 21 else pd.Series(index=symbols, dtype="float64")
    ret_1y = (close_df.pct_change(252).iloc[-1] * 100.0) if len(close_df) > 252 else pd.Series(index=symbols, dtype="float64")

    ytd_prices = close_df[close_df.index.year == pd.Timestamp(as_of).year]
    if not ytd_prices.empty:
        ret_ytd = (ytd_prices.iloc[-1] / ytd_prices.iloc[0] - 1.0) * 100.0
    else:
        ret_ytd = pd.Series(index=symbols, dtype="float64")

    vol_3m = volume_df.rolling(63, min_periods=20).mean().iloc[-1] / 1_000_000.0 if len(volume_df) > 0 else pd.Series(index=symbols, dtype="float64")
    vol_12m = volume_df.rolling(252, min_periods=40).mean().iloc[-1] / 1_000_000.0 if len(volume_df) > 0 else pd.Series(index=symbols, dtype="float64")

    lookback_close = close_df.tail(350)
    lookback_high = high_df.reindex(index=lookback_close.index, columns=lookback_close.columns)
    lookback_low = low_df.reindex(index=lookback_close.index, columns=lookback_close.columns)

    with redirect_stdout(io.StringIO()):
        rsi_panel = BorsapyIndicators.build_rsi_panel(lookback_close, period=14)
        macd_panel = BorsapyIndicators.build_macd_panel(lookback_close, output="histogram")
        atr_panel = BorsapyIndicators.build_atr_panel(lookback_high, lookback_low, lookback_close, period=14)

    rsi_14 = rsi_panel.iloc[-1] if not rsi_panel.empty else pd.Series(index=symbols, dtype="float64")
    macd_hist = macd_panel.iloc[-1] if not macd_panel.empty else pd.Series(index=symbols, dtype="float64")
    atr_14 = atr_panel.iloc[-1] if not atr_panel.empty else pd.Series(index=symbols, dtype="float64")
    atr_14_pct = _safe_divide(atr_14, latest_close) * 100.0

    high_1y = close_df.rolling(252, min_periods=40).max().iloc[-1] if len(close_df) > 40 else pd.Series(index=symbols, dtype="float64")
    upside_potential = _safe_divide(high_1y - latest_close, latest_close) * 100.0

    with redirect_stdout(io.StringIO()):
        shares_panel = loader.load_shares_outstanding_panel()

    if shares_panel is not None and isinstance(shares_panel, pd.DataFrame) and not shares_panel.empty:
        shares = shares_panel.copy()
        shares.index = pd.to_datetime(shares.index, errors="coerce")
        shares = shares.sort_index()
        shares.columns = [str(c).upper().split(".")[0] for c in shares.columns]
        shares = shares.reindex(columns=symbols)
        shares = shares.reindex(index=close_df.index, method="ffill")
        shares_latest = shares.iloc[-1]
    else:
        shares_latest = pd.Series(index=symbols, dtype="float64")

    market_cap_tl = latest_close * shares_latest
    market_cap_mn = market_cap_tl / 1_000_000.0

    usdtry_rate = np.nan
    usd_file = data_dir / "usdtry_data.csv"
    if usd_file.exists():
        try:
            usd = pd.read_csv(usd_file)
            if not usd.empty:
                if "USDTRY" in usd.columns:
                    usdtry_rate = float(pd.to_numeric(usd["USDTRY"], errors="coerce").dropna().iloc[-1])
                elif "Close" in usd.columns:
                    usdtry_rate = float(pd.to_numeric(usd["Close"], errors="coerce").dropna().iloc[-1])
        except Exception:
            usdtry_rate = np.nan

    market_cap_usd_mn = market_cap_mn / usdtry_rate if np.isfinite(usdtry_rate) and usdtry_rate > 0 else pd.Series(index=symbols, dtype="float64")

    fundamentals = _extract_fundamental_snapshot(loader, symbols)
    net_income_q = fundamentals["net_income"].reindex(symbols)
    net_income_prev_q = fundamentals["net_income_prev"].reindex(symbols)
    revenue_q = fundamentals["revenue"].reindex(symbols)
    revenue_prev_q = fundamentals["revenue_prev"].reindex(symbols)
    equity = fundamentals["equity"].reindex(symbols)
    assets = fundamentals["assets"].reindex(symbols)
    liabilities = fundamentals["liabilities"].reindex(symbols)
    ebitda_q = fundamentals["ebitda"].reindex(symbols)
    dividends_q = fundamentals["dividends"].reindex(symbols)
    cash = fundamentals["cash"].reindex(symbols)
    debt = fundamentals["debt"].reindex(symbols)

    net_income_ttm = net_income_q * 4.0
    revenue_ttm = revenue_q * 4.0
    ebitda_ttm = ebitda_q * 4.0
    dividends_ttm = dividends_q * 4.0

    pe = _safe_divide(market_cap_tl, net_income_ttm)
    pb = _safe_divide(market_cap_tl, equity)

    enterprise_value = market_cap_tl + debt - cash
    ev_ebitda = _safe_divide(enterprise_value, ebitda_ttm)
    ev_sales = _safe_divide(enterprise_value, revenue_ttm)

    dividend_yield = _safe_divide(dividends_ttm.abs(), market_cap_tl) * 100.0
    # Missing dividend statements are interpreted as zero payout when market cap exists.
    dividend_yield = dividend_yield.where(market_cap_tl.notna())
    dividend_yield = dividend_yield.fillna(0.0).where(market_cap_tl.notna())
    roe = _safe_divide(net_income_ttm, equity) * 100.0
    roa = _safe_divide(net_income_ttm, assets) * 100.0
    net_margin = _safe_divide(net_income_ttm, revenue_ttm) * 100.0
    ebitda_margin = _safe_divide(ebitda_ttm, revenue_ttm) * 100.0

    revenue_growth_yoy = _safe_divide(revenue_q - revenue_prev_q, revenue_prev_q.abs()) * 100.0
    net_income_growth_yoy = _safe_divide(net_income_q - net_income_prev_q, net_income_prev_q.abs()) * 100.0

    score = np.sign(ret_1m.fillna(0.0)) + np.sign(ret_1y.fillna(0.0))
    score = score + np.where(rsi_14 < 35, 1, np.where(rsi_14 > 70, -1, 0))
    score = score + np.where(macd_hist > 0, 1, np.where(macd_hist < 0, -1, 0))
    recommendation = pd.Series(np.where(score >= 2, "AL", np.where(score <= -2, "SAT", "TUT")), index=symbols)

    frame = pd.DataFrame(index=symbols)
    frame["symbol"] = frame.index
    frame["name"] = frame.index
    frame["market_cap_usd"] = market_cap_usd_mn.reindex(symbols)
    frame["market_cap"] = market_cap_mn.reindex(symbols)
    frame["pe"] = pe.reindex(symbols)
    frame["forward_pe"] = np.nan
    frame["pb"] = pb.reindex(symbols)
    frame["ev_ebitda"] = ev_ebitda.reindex(symbols)
    frame["ev_sales"] = ev_sales.reindex(symbols)
    frame["dividend_yield"] = dividend_yield.reindex(symbols)
    frame["upside_potential"] = upside_potential.reindex(symbols)
    frame["analyst_target_price"] = np.nan
    frame["roe"] = roe.reindex(symbols)
    frame["roa"] = roa.reindex(symbols)
    frame["net_margin"] = net_margin.reindex(symbols)
    frame["ebitda_margin"] = ebitda_margin.reindex(symbols)
    frame["foreign_ratio"] = np.nan
    frame["foreign_change_1w"] = np.nan
    frame["foreign_change_1m"] = np.nan
    frame["float_ratio"] = np.nan
    frame["volume_3m"] = vol_3m.reindex(symbols)
    frame["volume_12m"] = vol_12m.reindex(symbols)
    frame["return_1w"] = ret_1w.reindex(symbols)
    frame["return_1m"] = ret_1m.reindex(symbols)
    frame["return_1y"] = ret_1y.reindex(symbols)
    frame["return_ytd"] = ret_ytd.reindex(symbols)
    frame["revenue_growth_yoy"] = revenue_growth_yoy.reindex(symbols)
    frame["net_income_growth_yoy"] = net_income_growth_yoy.reindex(symbols)
    frame["rsi_14"] = rsi_14.reindex(symbols)
    frame["macd_hist"] = macd_hist.reindex(symbols)
    frame["atr_14_pct"] = atr_14_pct.reindex(symbols)
    frame["recommendation"] = recommendation.reindex(symbols)

    numeric_cols = [col["key"] for col in FILTER_FIELD_DEFS if col["key"] in frame.columns]
    for col in numeric_cols + ["rsi_14", "macd_hist", "atr_14_pct", "revenue_growth_yoy", "net_income_growth_yoy"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    sector_map = _load_sector_map(data_dir)
    frame["sector"] = frame["symbol"].map(lambda x: sector_map.get(str(x), "Unknown"))

    _SCREEN_CACHE["built_at"] = now
    _SCREEN_CACHE["as_of"] = as_of_iso
    _SCREEN_CACHE["frame"] = frame.copy()
    _SCREEN_CACHE["sector_map"] = dict(sector_map)
    _SCREEN_CACHE["close_df"] = close_df.tail(756).copy()
    _SCREEN_CACHE["data_dir"] = str(data_dir)
    _SCREEN_CACHE["state_token"] = expected_state_token

    return frame, as_of_iso, sector_map


def _resolve_isyatirim_as_of(raw_frame: pd.DataFrame) -> str:
    if raw_frame.empty:
        return pd.Timestamp.utcnow().date().isoformat()

    for col in raw_frame.columns:
        normalized = _normalize_column_name(col)
        if normalized not in {"as_of", "date", "tarih", "updated_at", "last_update"}:
            continue
        parsed = pd.to_datetime(raw_frame[col], errors="coerce").dropna()
        if not parsed.empty:
            return parsed.max().date().isoformat()
    return pd.Timestamp.utcnow().date().isoformat()


def _create_loader_for_runtime(runtime_paths: RuntimePaths) -> DataLoader:
    regime_dir = getattr(runtime_paths, "regime_outputs_dir", None) or getattr(runtime_paths, "regime_dir", None)
    return DataLoader(
        data_dir=runtime_paths.data_dir,
        regime_model_dir=regime_dir,
    )


def _load_isyatirim_screen_frame(
    runtime_paths: RuntimePaths,
    *,
    template: str | None = None,
    screen_filters: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, str, dict[str, str], Any | None]:
    loader = _create_loader_for_runtime(runtime_paths)
    adapter = loader.borsapy_adapter
    if adapter is None:
        return _ensure_screen_columns(pd.DataFrame(columns=["symbol", "name"])), "", {}, None

    filters = dict(screen_filters or {})
    raw = adapter.screen_stocks_isyatirim(template=template, **filters)
    if raw is None or raw.empty:
        return _ensure_screen_columns(pd.DataFrame(columns=["symbol", "name"])), "", {}, adapter

    normalized = _normalize_isyatirim_frame(raw)
    as_of_iso = _resolve_isyatirim_as_of(raw)
    sector_map = {
        str(row.symbol): str(row.sector).strip() or "Unknown"
        for row in normalized.loc[:, ["symbol", "sector"]].itertuples()
        if str(row.symbol).strip()
    }
    return normalized, as_of_iso, sector_map, adapter


def _build_isyatirim_enrichment_frame(adapter: Any, symbols: list[str]) -> pd.DataFrame:
    if adapter is None:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        normalized_symbol = str(symbol).upper().split(".")[0].strip()
        if not normalized_symbol:
            continue

        analyst_payload: dict[str, Any] = {}
        foreign_payload: dict[str, Any] = {}
        try:
            result = adapter.get_analyst_data(normalized_symbol)
            if isinstance(result, dict):
                analyst_payload = result
        except Exception:
            analyst_payload = {}
        try:
            result = adapter.get_foreign_ownership(normalized_symbol)
            if isinstance(result, dict):
                foreign_payload = result
        except Exception:
            foreign_payload = {}

        row: dict[str, Any] = {"symbol": normalized_symbol}
        row["analyst_target_price"] = _extract_numeric_from_dict(
            analyst_payload,
            ("analyst_target_price", "target_price", "price_target", "consensus_target"),
        )
        row["analyst_target_count"] = _extract_numeric_from_dict(
            analyst_payload,
            ("analyst_target_count", "target_count", "analyst_count", "number_of_analysts"),
        )
        row["forward_pe"] = _extract_numeric_from_dict(
            analyst_payload,
            ("forward_pe", "fwd_pe", "pe_forward", "ileri_fk"),
        )
        row["upside_potential"] = _extract_numeric_from_dict(
            analyst_payload,
            ("upside_potential", "upside", "target_upside"),
        )
        row["recommendation"] = _normalize_recommendation_value(
            _extract_text_from_dict(
                analyst_payload,
                ("recommendation", "rating", "to_grade", "consensus_recommendation"),
            )
        )
        row["recommendation_score"] = _extract_numeric_from_dict(
            analyst_payload,
            ("recommendation_score", "rating_score", "score"),
        )
        row["foreign_ratio"] = _extract_numeric_from_dict(
            foreign_payload,
            ("foreign_ratio", "foreign_ownership", "yabanci_orani", "yabanci_payi"),
        )
        row["foreign_change_1w"] = _extract_numeric_from_dict(
            foreign_payload,
            ("foreign_change_1w", "foreign_ratio_change_1w", "foreign_1w_change", "yabanci_degisim_1h"),
        )
        row["foreign_change_1m"] = _extract_numeric_from_dict(
            foreign_payload,
            ("foreign_change_1m", "foreign_ratio_change_1m", "foreign_1m_change", "yabanci_degisim_1a"),
        )
        row["float_ratio"] = _extract_numeric_from_dict(
            foreign_payload,
            ("float_ratio", "free_float", "fiili_dolasim_orani"),
        )
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    enriched = pd.DataFrame(rows).drop_duplicates(subset=["symbol"], keep="last")
    return enriched


def _apply_isyatirim_enrichment(frame: pd.DataFrame, adapter: Any | None) -> pd.DataFrame:
    if adapter is None or frame.empty or "symbol" not in frame.columns:
        return frame

    symbols = _as_symbol_list(frame["symbol"].tolist())
    if not symbols:
        return frame

    enriched = _build_isyatirim_enrichment_frame(adapter, symbols)
    if enriched.empty:
        return frame

    merged = frame.merge(enriched, on="symbol", how="left", suffixes=("", "_enrich"))
    numeric_fill_missing = [
        "forward_pe",
        "analyst_target_price",
        "analyst_target_count",
        "foreign_ratio",
        "foreign_change_1w",
        "foreign_change_1m",
        "float_ratio",
        "recommendation_score",
    ]
    for col in numeric_fill_missing:
        extra_col = f"{col}_enrich"
        if extra_col not in merged.columns:
            continue
        merged[col] = pd.to_numeric(merged[col], errors="coerce").combine_first(
            pd.to_numeric(merged[extra_col], errors="coerce")
        )

    if "upside_potential_enrich" in merged.columns:
        merged["upside_potential"] = pd.to_numeric(
            merged["upside_potential_enrich"], errors="coerce"
        ).combine_first(pd.to_numeric(merged["upside_potential"], errors="coerce"))

    if "recommendation_enrich" in merged.columns:
        normalized = merged["recommendation_enrich"].map(_normalize_recommendation_value)
        merged["recommendation"] = normalized.where(normalized.notna(), merged["recommendation"])

    merged = merged.drop(columns=[col for col in merged.columns if col.endswith("_enrich")])
    return _ensure_screen_columns(merged)


def _meta_response() -> dict[str, Any]:
    templates = sorted(getattr(bp.Screener, "TEMPLATES", DEFAULT_TEMPLATES)) if bp is not None else sorted(DEFAULT_TEMPLATES)
    scanner = TechnicalScannerEngine()
    return {
        "templates": templates,
        "technical_scans": scanner.predefined_scans(),
        "filters": FILTER_FIELD_DEFS,
        "indexes": INDEX_OPTIONS,
        "recommendations": RECOMMENDATION_OPTIONS,
        "default_sort_by": "upside_potential",
        "default_sort_desc": True,
        "filter_mode": "mixed",
        "data_sources": list(DATA_SOURCE_OPTIONS),
        "default_data_source": "local",
    }


def _apply_template(df: pd.DataFrame, template: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if not template:
        return df, []

    preset = TEMPLATE_PRESETS.get(template)
    if not isinstance(preset, dict):
        return df, []

    out = df
    applied: list[dict[str, Any]] = []

    for key, cfg in preset.items():
        if key == "recommendation":
            expected = str(cfg).upper()
            out = out[out["recommendation"].astype(str).str.upper() == expected]
            applied.append({"key": "recommendation", "label": "Recommendation", "min": None, "max": None})
            continue

        if key not in out.columns or not isinstance(cfg, dict):
            continue

        values = pd.to_numeric(out[key], errors="coerce")
        mask = values.notna()
        minimum = _as_float(cfg.get("min"))
        maximum = _as_float(cfg.get("max"))

        if minimum is not None:
            mask = mask & (values >= minimum)
        if maximum is not None:
            mask = mask & (values <= maximum)

        out = out.loc[mask]
        applied.append(
            {
                "key": key,
                "label": FIELD_LABELS.get(key, key),
                "min": minimum,
                "max": maximum,
            }
        )

    return out, applied


def _run_response(
    payload: dict[str, Any],
    *,
    runtime_paths: RuntimePaths | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    data_source = _normalize_data_source(payload.get("data_source"))
    paths = _resolve_paths(runtime_paths)
    refresh_requested = _safe_bool(payload.get("refresh_cache"), False) or _safe_bool(payload.get("_refresh_cache"), False)

    template = str(payload.get("template") or "").strip()
    available_templates = set(DEFAULT_TEMPLATES)
    if bp is not None:
        available_templates.update(set(getattr(bp.Screener, "TEMPLATES", DEFAULT_TEMPLATES)))
    if template and template not in available_templates:
        raise ValueError(f"Unknown template: {template}")

    local_frame = pd.DataFrame()
    local_as_of_iso = ""
    local_sector_map: dict[str, str] = {}
    if data_source in {"local", "hybrid"}:
        local_frame, local_as_of_iso, local_sector_map = _load_local_screen_frame(
            paths,
            force_refresh=refresh_requested,
        )

    isy_frame = pd.DataFrame()
    isy_as_of_iso = ""
    isy_sector_map: dict[str, str] = {}
    isy_adapter: Any | None = None
    if data_source in {"isyatirim", "hybrid"}:
        isy_template = (template or None) if data_source == "isyatirim" else None
        isy_frame, isy_as_of_iso, isy_sector_map, isy_adapter = _load_isyatirim_screen_frame(
            paths,
            template=isy_template,
        )

    if data_source == "local":
        frame = local_frame
        as_of_iso = local_as_of_iso
        sector_map = local_sector_map
    elif data_source == "isyatirim":
        frame = isy_frame
        as_of_iso = isy_as_of_iso or pd.Timestamp.utcnow().date().isoformat()
        sector_map = isy_sector_map
    else:
        frame = _merge_hybrid_fields(local_frame, isy_frame)
        as_of_iso = local_as_of_iso or isy_as_of_iso or pd.Timestamp.utcnow().date().isoformat()
        sector_map = dict(local_sector_map)
        sector_map.update(isy_sector_map)

    if frame.empty:
        raise ValueError(f"No screener data available for data_source={data_source}.")

    enrichment_applied = data_source in {"isyatirim", "hybrid"} and isy_adapter is not None
    if enrichment_applied:
        frame = _apply_isyatirim_enrichment(frame, isy_adapter)

    df = frame.copy()

    sector = str(payload.get("sector") or "").strip()
    index_name = str(payload.get("index") or "").strip().upper()
    recommendation = str(payload.get("recommendation") or "").strip().upper()
    custom_symbols = _as_symbol_list(payload.get("symbols"))

    if sector:
        if not sector_map:
            raise ValueError("Sector filter is unavailable because sector classification data is missing.")
        df = df[df["sector"].astype(str).str.lower() == sector.lower()]

    if index_name:
        if index_name not in INDEX_OPTIONS:
            raise ValueError(f"Unknown index: {index_name}")
        if index_name == "CUSTOM":
            if not custom_symbols:
                raise ValueError("CUSTOM universe requires a non-empty 'symbols' list.")
            df = df[df["symbol"].isin(custom_symbols)]
        elif index_name != "XUTUM":
            members = _get_index_components(index_name)
            if members:
                df = df[df["symbol"].isin(members)]
            else:
                # Fallback basket by market cap when static members are unavailable.
                rank_source = pd.to_numeric(df["market_cap_usd"], errors="coerce")
                if rank_source.dropna().empty:
                    rank_source = pd.to_numeric(df["market_cap"], errors="coerce")
                order = rank_source.sort_values(ascending=False).index
                n = 100
                if index_name == "XU030":
                    n = 30
                elif index_name == "XU050":
                    n = 50
                df = df.loc[order[:n]]
    elif custom_symbols:
        df = df[df["symbol"].isin(custom_symbols)]

    if recommendation:
        if recommendation not in RECOMMENDATION_OPTIONS:
            raise ValueError(f"Unknown recommendation: {recommendation}")
        df = df[df["recommendation"].astype(str).str.upper() == recommendation]

    df, template_filters = _apply_template(df, template)
    scanner = TechnicalScannerEngine()
    df, technical_conditions, technical_templates, technical_interval, technical_columns = _apply_technical_scan(
        df=df,
        payload=payload,
        scanner=scanner,
    )

    allowed_filters = set(FIELD_LABELS.keys())
    filters = _normalize_filters(payload.get("filters"), allowed_filters)
    percentile_filters = _normalize_percentile_filters(payload.get("percentile_filters"), allowed_filters)

    applied_filters: list[dict[str, Any]] = list(template_filters)
    applied_percentile_filters: list[dict[str, Any]] = []
    if technical_conditions:
        applied_filters.append(
            {
                "key": "technical_scan",
                "label": "Technical Scan",
                "conditions": list(technical_conditions),
                "templates": list(technical_templates),
                "interval": technical_interval,
            }
        )

    for key, bounds in filters.items():
        if key not in df.columns:
            continue

        values = pd.to_numeric(df[key], errors="coerce")
        mask = values.notna()

        minimum = bounds.get("min")
        maximum = bounds.get("max")
        if minimum is not None:
            mask = mask & (values >= minimum)
        if maximum is not None:
            mask = mask & (values <= maximum)

        df = df.loc[mask]
        applied_filters.append(
            {
                "key": key,
                "label": FIELD_LABELS.get(key, key),
                "min": minimum,
                "max": maximum,
            }
        )

    for key, bounds in percentile_filters.items():
        if key not in df.columns:
            continue

        values = pd.to_numeric(df[key], errors="coerce")
        pct_rank = values.rank(pct=True, method="average")

        min_pct = bounds.get("min_pct")
        max_pct = bounds.get("max_pct")

        mask = pct_rank.notna()
        if min_pct is not None:
            mask = mask & (pct_rank >= min_pct)
        if max_pct is not None:
            mask = mask & (pct_rank <= max_pct)

        df = df.loc[mask]
        applied_percentile_filters.append(
            {
                "key": key,
                "label": FIELD_LABELS.get(key, key),
                "min_pct": min_pct,
                "max_pct": max_pct,
            }
        )

    sort_by = str(payload.get("sort_by") or "upside_potential").strip()
    sort_desc = _safe_bool(payload.get("sort_desc"), default=True)

    if sort_by in df.columns:
        sort_values = pd.to_numeric(df[sort_by], errors="coerce")
        df = df.assign(_sort_key=sort_values).sort_values("_sort_key", ascending=not sort_desc, na_position="last")
        df = df.drop(columns=["_sort_key"])

    total_matches = int(len(df))
    limit = _as_int(payload.get("limit"), default=100, minimum=1, maximum=2000)
    page = _as_int(payload.get("page"), default=1, minimum=1, maximum=500_000)

    offset_raw = payload.get("offset")
    if offset_raw is None or offset_raw == "":
        offset = max(0, (page - 1) * limit)
    else:
        offset = _as_int(offset_raw, default=0, minimum=0, maximum=5_000_000)
        page = max(1, offset // limit + 1)

    total_pages = max(1, (total_matches + limit - 1) // limit) if total_matches > 0 else 1
    if page > total_pages:
        page = total_pages
        offset = max(0, (page - 1) * limit)

    paged_df = df.iloc[offset : offset + limit]

    requested_columns_raw = payload.get("fields")
    if not isinstance(requested_columns_raw, list):
        requested_columns_raw = payload.get("columns")
    requested_columns: list[str] = []
    if isinstance(requested_columns_raw, list):
        requested_columns = [str(c) for c in requested_columns_raw if isinstance(c, str)]

    if requested_columns:
        display_columns = [col for col in requested_columns if col in paged_df.columns]
    else:
        display_columns = [col for col in DISPLAY_COLUMNS_DEFAULT if col in paged_df.columns]
        for key in filters.keys():
            if key in paged_df.columns and key not in display_columns:
                display_columns.append(key)
        for key in technical_columns:
            if key in paged_df.columns and key not in display_columns:
                display_columns.append(key)

    if not display_columns:
        display_columns = [str(c) for c in paged_df.columns[:20]]

    out_df = paged_df.loc[:, display_columns].copy() if display_columns else paged_df.copy()

    rows = [
        {key: _jsonable(value) for key, value in row.items()}
        for row in out_df.replace([np.inf, -np.inf], np.nan).to_dict(orient="records")
    ]

    chart_symbol_raw = str(payload.get("chart_symbol") or "").strip().upper().split(".")[0]
    if not chart_symbol_raw and rows:
        chart_symbol_raw = str(rows[0].get("symbol", "")).strip().upper().split(".")[0]
    if not chart_symbol_raw and custom_symbols:
        chart_symbol_raw = custom_symbols[0]

    chart_payload: dict[str, Any] = {"symbol": None, "points": []}
    chart_points_limit = _as_int(payload.get("chart_points"), default=252, minimum=30, maximum=756)
    close_cache = _SCREEN_CACHE.get("close_df")
    if isinstance(close_cache, pd.DataFrame) and chart_symbol_raw and chart_symbol_raw in close_cache.columns:
        chart_series = pd.to_numeric(close_cache[chart_symbol_raw], errors="coerce").dropna().tail(chart_points_limit)
        chart_payload = {
            "symbol": chart_symbol_raw,
            "points": [
                {
                    "date": idx.date().isoformat(),
                    "close": round(float(value), 6),
                }
                for idx, value in chart_series.items()
            ],
        }

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    return {
        "meta": {
            "as_of": as_of_iso,
            "execution_ms": elapsed_ms,
            "total_matches": total_matches,
            "returned_rows": len(rows),
            "page": page,
            "page_size": limit,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
            "offset": offset,
            "template": template or None,
            "sector": sector or None,
            "index": index_name or None,
            "recommendation": recommendation or None,
            "technical_conditions": technical_conditions or None,
            "technical_templates": technical_templates or None,
            "technical_interval": technical_interval if technical_conditions else None,
            "sort_by": sort_by,
            "sort_desc": sort_desc,
            "data_source": data_source,
            "isyatirim_enrichment": enrichment_applied,
        },
        "columns": [
            {"key": col, "label": FIELD_LABELS.get(col, col.replace("_", " ").title())}
            for col in out_df.columns
        ],
        "rows": rows,
        "applied_filters": applied_filters,
        "applied_percentile_filters": applied_percentile_filters,
        "chart": chart_payload,
    }


def _friendly_error(exc: Exception) -> str:
    message = str(exc)
    if "CERTIFICATE_VERIFY_FAILED" in message:
        return (
            "Stock screener SSL verification failed in this runtime. "
            "Use deployment runtime or install CA certificates on this host."
        )
    return message


def get_stock_filter_metadata() -> dict[str, Any]:
    return _meta_response()


def run_stock_filter(
    payload: dict[str, Any],
    *,
    runtime_paths: RuntimePaths | None = None,
) -> StockScreenerResult:
    if not isinstance(payload, dict):
        raise QuantEngineValidationError("Request payload must be a JSON object.")
    try:
        response = _run_response(payload, runtime_paths=runtime_paths)
        required = {"meta", "columns", "rows", "applied_filters", "chart"}
        missing = sorted(required.difference(response.keys()))
        if missing:
            raise QuantEngineExecutionError(
                f"Stock screener result is missing required keys: {', '.join(missing)}."
            )
        return response
    except QuantEngineError:
        raise
    except RuntimePathError as exc:
        raise QuantEngineDataError(
            str(exc),
            user_message=(
                "Price data is not available. "
                "Place bist_prices_full.csv (or .parquet / .csv.gz) in your data directory, "
                "or set the BIST_DATA_DIR environment variable to a directory that contains it."
            ),
        ) from exc
    except FileNotFoundError as exc:
        raise QuantEngineDataError(_friendly_error(exc)) from exc
    except ValueError as exc:
        raise QuantEngineValidationError(_friendly_error(exc)) from exc
    except Exception as exc:
        LOGGER.exception("stock_filter.run_failed")
        raise QuantEngineExecutionError(
            "Stock filtering failed.",
            user_message=_friendly_error(exc),
        ) from exc
