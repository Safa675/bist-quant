"""
Midas Hesap Ekstresi (PDF) parser.

Extracts from each PDF:
  * YATIRIM İŞLEMLERİ  -> trade records (only 'Gerçekleşti' executions)
  * HESAP İŞLEMLERİ     -> cash flows (Para Yatırma / Çekme / Nema / Promosyon)
  * TEMETTÜ İŞLEMLERİ   -> dividend payments
  * PORTFÖY ÖZETİ       -> end-of-month snapshot (qty, avg cost, P&L)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("midas_pdf_parser")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MidasTrade:
    exec_datetime: pd.Timestamp
    symbol: str
    side: str               # "BUY" or "SELL"
    quantity: float
    price: float
    fee: float
    total_amount: float
    order_type: str         # "Limit Emri", "Piyasa Emri", "Fon Emri"
    source_file: str

    @property
    def trade_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.exec_datetime).normalize()


@dataclass
class MidasCashFlow:
    flow_datetime: pd.Timestamp
    flow_type: str          # "DEPOSIT", "WITHDRAWAL", "NEMA", "PROMOTION"
    amount: float
    description: str
    source_file: str

    @property
    def flow_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.flow_datetime).normalize()


@dataclass
class MidasDividend:
    pay_date: pd.Timestamp
    symbol: str
    gross: float
    withholding: float
    net: float
    source_file: str


@dataclass
class MidasPortfolioSnapshot:
    snapshot_date: pd.Timestamp
    rows: list[dict] = field(default_factory=list)
    total_value: float | None = None
    cash_balance: float | None = None
    source_file: str = ""


@dataclass
class MidasStatement:
    period_start: pd.Timestamp
    period_end: pd.Timestamp
    trades: list[MidasTrade] = field(default_factory=list)
    cash_flows: list[MidasCashFlow] = field(default_factory=list)
    dividends: list[MidasDividend] = field(default_factory=list)
    snapshot: MidasPortfolioSnapshot | None = None
    source_file: str = ""


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_TR_MONTHS = {
    "Ocak": 1, "Şubat": 2, "Mart": 3, "Nisan": 4, "Mayıs": 5, "Haziran": 6,
    "Temmuz": 7, "Ağustos": 8, "Eylül": 9, "Ekim": 10, "Kasım": 11, "Aralık": 12,
}


def _parse_tr_datetime(s: str) -> pd.Timestamp:
    """Parse '02/03/26 13:21:14' style timestamps."""
    s = s.strip()
    for fmt in ("%d/%m/%y %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%d/%m/%y", "%d/%m/%Y"):
        try:
            return pd.Timestamp(datetime.strptime(s, fmt))
        except ValueError:
            continue
    raise ValueError(f"Cannot parse datetime: {s!r}")


def _parse_tr_date(s: str) -> pd.Timestamp:
    """Parse '27/03/26' style date-only."""
    s = s.strip()
    for fmt in ("%d/%m/%y", "%d/%m/%Y"):
        try:
            return pd.Timestamp(datetime.strptime(s, fmt))
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {s!r}")


def _parse_period_header(s: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Parse '01/03/26 - 31/03/26 HESAP EKSTRESİ'."""
    m = re.search(r"(\d{2}/\d{2}/\d{2,4})\s*-\s*(\d{2}/\d{2}/\d{2,4})", s)
    if not m:
        raise ValueError(f"Cannot parse period: {s!r}")
    return _parse_tr_date(m.group(1)), _parse_tr_date(m.group(2))


def _to_number(s: str | None) -> float:
    """Parse a TR-formatted number like '1.740,00' or '-' to float."""
    if s is None:
        return 0.0
    s = str(s).strip()
    if s in ("", "-", "—"):
        return 0.0
    # Handle TR-style decimals: '1.740,00' -> 1740.00 ; '0,00' -> 0.0
    # Strategy: if both '.' and ',' present, '.' is thousands, ',' is decimal
    if "." in s and "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s and "." not in s:
        # '11,00' -> 11.0 ; '0,89' -> 0.89
        s = s.replace(",", ".")
    # else: dot already present as decimal
    s = re.sub(r"[^\d\.\-]", "", s)
    try:
        return float(s)
    except ValueError:
        return 0.0


def _extract_ticker(name: str) -> str:
    """Extract ticker from 'SYMBOL - Long Name' or 'SYMBOL - Long Name...'."""
    if not name:
        return ""
    name = name.strip()
    # Skip if name doesn't look like a portfolio row
    if " - " not in name:
        return name.split()[0].upper() if name else ""
    return name.split(" - ")[0].strip().upper()


# ---------------------------------------------------------------------------
# Table-level parsers
# ---------------------------------------------------------------------------

def _parse_trades_table(rows: list[list[str]], source_file: str) -> list[MidasTrade]:
    """Parse a YATIRIM İŞLEMLERİ table (header at index 0, data follows)."""
    if len(rows) < 2:
        return []
    out: list[MidasTrade] = []
    for row in rows[1:]:
        # Skip the section header row that pdfplumber may emit at index 0
        if not row or not row[0]:
            continue
        first = str(row[0])
        if "YATIRIM" in first.upper() or not re.search(r"\d{2}/\d{2}/\d{2}", first):
            continue
        try:
            dt = _parse_tr_datetime(first)
        except ValueError:
            continue
        # Layout: [Tarih, İşlem Türü, Sembol, İşlem Tipi, İşlem Durumu, Para Birimi,
        #         Emir Adedi, Emir Tutarı, Gerçekleşen Adet, Ortalama İşlem Fiyatı,
        #         İşlem Ücreti, İşlem Tutarı]
        try:
            symbol = str(row[2]).strip() if row[2] else ""
            side_raw = str(row[3]).strip() if row[3] else ""
            status = str(row[4]).strip() if row[4] else ""
            order_type = str(row[1]).strip() if row[1] else ""
            qty = _to_number(row[8])
            price = _to_number(row[9])
            fee = _to_number(row[10])
            total = _to_number(row[11])
        except IndexError:
            continue

        # Only keep executed orders with non-zero quantity
        if status != "Gerçekleşti":
            continue
        if qty <= 0:
            continue
        side = "BUY" if side_raw == "Alış" else "SELL" if side_raw == "Satış" else side_raw
        out.append(
            MidasTrade(
                exec_datetime=dt,
                symbol=symbol,
                side=side,
                quantity=qty,
                price=price,
                fee=fee,
                total_amount=total,
                order_type=order_type,
                source_file=source_file,
            )
        )
    return out


def _parse_cashflows_table(rows: list[list[str]], source_file: str) -> list[MidasCashFlow]:
    if len(rows) < 2:
        return []
    out: list[MidasCashFlow] = []
    for row in rows[1:]:
        if not row or not row[0]:
            continue
        first = str(row[0])
        # Skip section header
        if "HESAP İŞLEM" in first.upper() and "TARİH" not in first.upper():
            continue
        # Two date columns: Talep Tarihi (col 0) and İşlem Tarihi (col 1)
        # For "Ücretsiz İşlem" rows, the date is just a month range like "01/03/26"
        try:
            if re.search(r"\d{2}/\d{2}/\d{2}", first):
                dt = _parse_tr_datetime(first) if re.search(r"\d{2}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}", first) else _parse_tr_date(first)
            else:
                continue
        except ValueError:
            continue

        try:
            flow_type_raw = str(row[2]).strip() if row[2] else ""
            desc = str(row[3]).strip() if row[3] else ""
            amount = _to_number(row[5])
        except IndexError:
            continue

        if flow_type_raw == "Para Yatırma":
            ft = "DEPOSIT"
        elif flow_type_raw == "Para Çekme":
            ft = "WITHDRAWAL"
        elif flow_type_raw == "Diğer Gelir":
            ft = "NEMA" if "Nema" in desc else "OTHER_INCOME"
        elif flow_type_raw == "Ücretsiz İşlem":
            ft = "PROMOTION"
        else:
            ft = flow_type_raw.upper().replace(" ", "_") or "UNKNOWN"

        out.append(
            MidasCashFlow(
                flow_datetime=dt,
                flow_type=ft,
                amount=amount,
                description=desc,
                source_file=source_file,
            )
        )
    return out


def _parse_dividends_table(rows: list[list[str]], source_file: str) -> list[MidasDividend]:
    if len(rows) < 2:
        return []
    out: list[MidasDividend] = []
    for row in rows[1:]:
        if not row or not row[0]:
            continue
        first = str(row[0])
        if "TEMETTÜ" in first.upper() and "TARİH" not in first.upper():
            continue
        try:
            dt = _parse_tr_date(first)
        except ValueError:
            continue
        try:
            symbol_raw = str(row[1]).strip() if row[1] else ""
            gross = _to_number(row[2])
            wht = _to_number(row[3])
            net = _to_number(row[4])
        except IndexError:
            continue
        symbol = _extract_ticker(symbol_raw)
        out.append(
            MidasDividend(
                pay_date=dt,
                symbol=symbol,
                gross=gross,
                withholding=wht,
                net=net,
                source_file=source_file,
            )
        )
    return out


def _parse_portfolio_table(rows: list[list[str]], source_file: str) -> MidasPortfolioSnapshot:
    if len(rows) < 2:
        return MidasPortfolioSnapshot(pd.Timestamp("NaT"), source_file=source_file)
    # Header: ['PORTFÖY ÖZETİ (31/03/26)', ...]
    header_first = str(rows[0][0] or "")
    m = re.search(r"\((\d{2}/\d{2}/\d{2,4})\)", header_first)
    snapshot_date = _parse_tr_date(m.group(1)) if m else pd.Timestamp("NaT")
    portfolio_rows: list[dict] = []
    for row in rows[1:]:
        if not row or not row[0]:
            continue
        first = str(row[0])
        if "Sermaya" in first:  # column header row
            continue
        if " - " not in first:
            continue
        try:
            portfolio_rows.append({
                "name": first,
                "ticker": _extract_ticker(first),
                "quantity": _to_number(row[1]),
                "avg_cost": _to_number(row[2]),
                "pnl": _to_number(row[3]),
                "value": _to_number(row[4]),
            })
        except IndexError:
            continue
    return MidasPortfolioSnapshot(
        snapshot_date=snapshot_date,
        rows=portfolio_rows,
        source_file=source_file,
    )


# ---------------------------------------------------------------------------
# Top-level PDF parser
# ---------------------------------------------------------------------------

def parse_midas_pdf(path: Path) -> MidasStatement:
    """Parse one Midas Hesap Ekstresi PDF."""
    import pdfplumber  # local import

    fname = path.name
    with pdfplumber.open(path) as pdf:
        all_tables: list[tuple[int, list[list[str]]]] = []
        for page_idx, page in enumerate(pdf.pages):
            for tbl in page.extract_tables() or []:
                all_tables.append((page_idx, tbl))

    period_start = period_end = pd.Timestamp("NaT")
    trades: list[MidasTrade] = []
    cash_flows: list[MidasCashFlow] = []
    dividends: list[MidasDividend] = []
    snapshot: MidasPortfolioSnapshot | None = None

    for _, tbl in all_tables:
        if not tbl or not tbl[0]:
            continue
        first = " ".join(str(c) for c in tbl[0] if c).strip()
        upper = first.upper()

        if "HESAP EKSTRESİ" in upper:
            try:
                period_start, period_end = _parse_period_header(first)
            except ValueError:
                pass
        elif "PORTFÖY ÖZETİ" in upper:
            snapshot = _parse_portfolio_table(tbl, fname)
        elif "YATIRIM İŞLEMLERİ" in upper:
            trades.extend(_parse_trades_table(tbl, fname))
        elif "HESAP İŞLEMLERİ" in upper:
            cash_flows.extend(_parse_cashflows_table(tbl, fname))
        elif "TEMETTÜ İŞLEMLERİ" in upper:
            dividends.extend(_parse_dividends_table(tbl, fname))

    return MidasStatement(
        period_start=period_start,
        period_end=period_end,
        trades=trades,
        cash_flows=cash_flows,
        dividends=dividends,
        snapshot=snapshot,
        source_file=fname,
    )


def parse_midas_directory(directory: Path) -> list[MidasStatement]:
    pdfs = sorted(directory.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No Midas PDF files found in {directory}")
    out: list[MidasStatement] = []
    for pdf in pdfs:
        st = parse_midas_pdf(pdf)
        logger.info(
            "  %s -> %d trades, %d cash flows, %d dividends",
            pdf.name, len(st.trades), len(st.cash_flows), len(st.dividends),
        )
        out.append(st)
    return out


def deduplicate_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate trades that appear in multiple monthly statements.

    Midas PDFs span calendar months, so a trade executed in late November
    appears in both the November and December statements. We keep the first
    occurrence (sorted by source_file).
    """
    if trades.empty:
        return trades
    # Sort by exec_date then source_file so the earlier month is first
    sorted_t = trades.sort_values(["exec_date", "source_file"]).reset_index(drop=True)
    # Drop exact duplicates (same symbol, side, qty, price, date)
    deduped = sorted_t.drop_duplicates(
        subset=["exec_date", "symbol", "side", "quantity", "price"],
        keep="first",
    ).reset_index(drop=True)
    return deduped


def flatten_to_frames(statements: list[MidasStatement]) -> dict[str, pd.DataFrame]:
    trades_df = pd.DataFrame(
        [asdict(t) for s in statements for t in s.trades]
    ) if any(s.trades for s in statements) else pd.DataFrame()
    if not trades_df.empty:
        trades_df["exec_date"] = pd.to_datetime(trades_df["exec_datetime"]).dt.normalize()
        # drop exec_datetime to keep schema clean (we keep normalized date)
        trades_df = trades_df.drop(columns=["exec_datetime"])

    cash_df = pd.DataFrame(
        [asdict(c) for s in statements for c in s.cash_flows]
    ) if any(s.cash_flows for s in statements) else pd.DataFrame()
    if not cash_df.empty:
        cash_df["flow_date"] = pd.to_datetime(cash_df["flow_datetime"]).dt.normalize()
        cash_df = cash_df.drop(columns=["flow_datetime"])

    div_df = pd.DataFrame(
        [asdict(d) for s in statements for d in s.dividends]
    ) if any(s.dividends for s in statements) else pd.DataFrame()

    snapshots = []
    for s in statements:
        if s.snapshot and s.snapshot.rows:
            for r in s.snapshot.rows:
                snapshots.append({**r, "snapshot_date": s.snapshot.snapshot_date})
    snap_df = pd.DataFrame(snapshots)

    return {
        "trades": trades_df,
        "cash_flows": cash_df,
        "dividends": div_df,
        "portfolio_snapshots": snap_df,
    }
