from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ITEM_CODE_TO_SHEET = {
    "1": "Bilanço",
    "2": "Bilanço",
    "3": "Gelir Tablosu (Çeyreklik)",
    "4": "Nakit Akış (Çeyreklik)",
}

FIELD_KEYS = {
    "revenue": [
        "Satış Gelirleri",
        "Toplam Hasılat",
        "Hasılat",
        "Net Satışlar",
    ],
    "gross_profit": [
        "BRÜT KAR (ZARAR)",
        "Brüt Kar (Zarar)",
        "Ticari Faaliyetlerden Brüt Kar (Zarar)",
    ],
    "operating_income": [
        "Net Faaliyet Kar/Zararı",
        "Faaliyet Karı (Zararı)",
        "Finansman Geliri (Gideri) Öncesi Faaliyet Karı (Zararı)",
    ],
    "net_income": [
        "DÖNEM KARI (ZARARI)",
        "Dönem Net Karı (Zararı)",
        "Net Dönem Karı (Zararı)",
        "Ana Ortaklık Payları",
        "Dönem Karı (Zararı)",
    ],
    "ebitda": [
        "FAVÖK",
        "Faiz Amortisman ve Vergi Öncesi Kar",
    ],
    "total_assets": [
        "Toplam Varlıklar",
        "TOPLAM VARLIKLAR",
        "Toplam Aktifler",
    ],
    "total_equity": [
        "Özkaynaklar",
        "TOPLAM ÖZKAYNAKLAR",
        "Ana Ortaklığa Ait Özkaynaklar",
        "Toplam Özkaynaklar",
    ],
    "total_liabilities": [
        "Toplam Yükümlülükler",
    ],
    "operating_cash_flow": [
        "İşletme Faaliyetlerinden Kaynaklanan Net Nakit",
        "Faaliyetlerden Elde Edilen Nakit Akışları",
        "İşletme Faaliyetlerinden Nakit Akışları",
    ],
    "short_term_debt": [
        "Kısa Vadeli Yükümlülükler",
    ],
    "long_term_debt": [
        "Uzun Vadeli Yükümlülükler",
    ],
    "cash": [
        "Nakit ve Nakit Benzerleri",
    ],
    "depreciation": [
        "Amortisman Giderleri",
        "Amortisman ve İtfa Payı İle İlgili Düzeltmeler",
    ],
    "capex": [
        "Maddi ve Maddi Olmayan Duran Varlıkların Alımından Kaynaklanan Nakit Çıkışları",
        "Maddi Duran Varlık Alımları",
    ],
}


def _period_end_timestamp(year: int, month: int) -> pd.Timestamp:
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)


def _extract_numeric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)):
        value_float = float(value)
        if np.isnan(value_float):
            return None
        return value_float
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return float(raw.replace(",", ".").replace(" ", ""))
    except (TypeError, ValueError):
        return None


def _extract_field_value(
    *,
    items: list[dict[str, Any]],
    field_labels: Iterable[str],
    value_key: str,
) -> float | None:
    labels = {label.strip().lower() for label in field_labels}
    for item in items:
        desc = str(item.get("itemDescTr", "")).strip().lower()
        if desc and desc in labels:
            return _extract_numeric(item.get(value_key))
    return None


def build_flat_normalized(raw_by_ticker: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Build a flat, typed fundamentals dataset from raw ticker payloads."""
    rows: list[dict[str, Any]] = []
    for ticker, payload in raw_by_ticker.items():
        items = payload.get("items", [])
        periods = payload.get("periods_requested", [])
        if not isinstance(items, list) or not periods:
            continue

        for period_idx, period in enumerate(periods[:5]):
            year = int(period[0])
            month = int(period[1])
            quarter = max(1, min(4, month // 3))
            value_key = f"value{period_idx + 1}"

            has_any_data = any(_extract_numeric(item.get(value_key)) is not None for item in items)
            if not has_any_data:
                continue

            row: dict[str, Any] = {
                "ticker": ticker,
                "period_end": _period_end_timestamp(year, month),
                "fiscal_year": year,
                "fiscal_quarter": quarter,
                "reporting_type": "annual/q4" if quarter == 4 else "quarterly",
                "financial_group": payload.get("financial_group", "XI_29"),
                "is_annual_report": bool(quarter == 4),
            }
            for field, labels in FIELD_KEYS.items():
                row[field] = _extract_field_value(items=items, field_labels=labels, value_key=value_key)

            ocf = row.get("operating_cash_flow")
            capex = row.get("capex")
            row["free_cash_flow"] = (ocf - abs(capex)) if (ocf is not None and capex is not None) else None

            if row.get("total_liabilities") is None:
                st_debt = row.get("short_term_debt")
                lt_debt = row.get("long_term_debt")
                if st_debt is not None and lt_debt is not None:
                    row["total_liabilities"] = st_debt + lt_debt
            rows.append(row)

    flat = pd.DataFrame(rows)
    if flat.empty:
        return flat

    desired_columns = [
        "ticker",
        "period_end",
        "fiscal_year",
        "fiscal_quarter",
        "reporting_type",
        "is_annual_report",
        "financial_group",
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
        "depreciation",
        "capex",
    ]
    for column in desired_columns:
        if column not in flat.columns:
            flat[column] = np.nan
    flat = flat[desired_columns]
    return flat.sort_values(["ticker", "period_end"]).reset_index(drop=True)


def build_consolidated_panel(raw_by_ticker: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Build consolidated-format fundamentals panel from raw payloads."""
    rows: list[pd.Series] = []
    for ticker, payload in raw_by_ticker.items():
        items = payload.get("items", [])
        periods = payload.get("periods_requested", [])
        if not isinstance(items, list) or not periods:
            continue

        period_col_map = {f"value{i+1}": f"{int(y)}/{int(m)}" for i, (y, m) in enumerate(periods[:5])}

        for item in items:
            item_code = str(item.get("itemCode", ""))
            item_desc = str(item.get("itemDescTr", ""))
            if not item_code or not item_desc:
                continue
            sheet_name = ITEM_CODE_TO_SHEET.get(item_code[0])
            if sheet_name is None:
                continue

            stripped = item_desc.lstrip()
            leading_spaces = len(item_desc) - len(stripped)
            indent_level = leading_spaces // 2 if leading_spaces > 0 else 0
            normalized_row_name = ("    " * indent_level) + stripped

            row_values: dict[str, float | None] = {}
            for value_key, period_col in period_col_map.items():
                row_values[period_col] = _extract_numeric(item.get(value_key))

            if not any(value is not None for value in row_values.values()):
                continue
            rows.append(pd.Series(row_values, name=(ticker, sheet_name, normalized_row_name)))

    if not rows:
        return pd.DataFrame()

    panel = pd.DataFrame(rows)
    panel.index = pd.MultiIndex.from_tuples(
        panel.index.tolist(),
        names=["ticker", "sheet_name", "row_name"],
    )
    panel = panel.groupby(level=["ticker", "sheet_name", "row_name"]).first()
    return panel


def save_normalized_per_ticker_json(flat_frame: pd.DataFrame, output_dir: Path) -> None:
    """Persist flat normalized records as per-ticker JSON blobs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if flat_frame.empty:
        return
    for ticker, group in flat_frame.groupby("ticker", sort=True):
        records = group.to_dict(orient="records")
        for record in records:
            for key, value in list(record.items()):
                if isinstance(value, pd.Timestamp):
                    record[key] = value.isoformat()
                elif isinstance(value, float) and np.isnan(value):
                    record[key] = None
        (output_dir / f"{ticker}.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )


def build_staleness_weights(staleness_report: pd.DataFrame) -> dict[str, float]:
    """Generate deterministic ticker weights for staleness-aware signal scaling."""
    if staleness_report.empty:
        return {}
    weights: dict[str, float] = {}
    for _, row in staleness_report.iterrows():
        ticker = str(row.get("ticker"))
        days = row.get("staleness_days")
        if pd.isna(days):
            weights[ticker] = 0.0
        elif float(days) <= 60:
            weights[ticker] = 1.0
        elif float(days) <= 120:
            weights[ticker] = max(0.2, 1.0 - (float(days) - 60.0) / 60.0 * 0.8)
        else:
            weights[ticker] = 0.1
    return weights
