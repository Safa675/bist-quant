"""Helper module for fetching and formatting financial statements and ratios via borsapy."""

from __future__ import annotations

import logging
from ssl import SSLError
from typing import Any

import pandas as pd
from bist_quant.common.ticker_sets import UFRS_TICKERS

logger = logging.getLogger(__name__)


def _detect_financial_group(symbol: str) -> str | None:
    """Return UFRS for bank/financial tickers, None (=XI_29 default) otherwise."""
    return "UFRS" if symbol.upper() in UFRS_TICKERS else None


def _format_quarterly_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Format borsapy quarterly column names (e.g. 2026Q1, 2025Q4) to YYYY/MM."""
    if df is None or df.empty:
        return df

    new_cols = []
    for col in df.columns:
        col_str = str(col).strip()
        if len(col_str) == 6 and col_str[4] == "Q":
            try:
                year = int(col_str[:4])
                q = int(col_str[5])
                month = q * 3
                new_cols.append(f"{year}/{month:02d}")
                continue
            except ValueError:
                pass
        new_cols.append(col)

    df.columns = new_cols
    return df


def get_financials(client: Any, symbol: str) -> dict[str, pd.DataFrame]:
    """Backward-compatible alias for financial statements."""
    statements = get_financial_statements(client, symbol)
    cash_flow = statements.get("cash_flow", pd.DataFrame())
    return {
        "balance_sheet": statements.get("balance_sheet", pd.DataFrame()),
        "income_stmt": statements.get("income_stmt", pd.DataFrame()),
        "cashflow": cash_flow,
        "cash_flow": cash_flow,
    }


def get_financial_statements(client: Any, symbol: str, last_n: int = 20) -> dict[str, pd.DataFrame]:
    """Get financial statements with disk caching."""
    symbol = client._normalize_symbol(symbol)

    # Try disk cache first
    if client._disk_cache is not None:
        cached_bs = client._disk_cache.get_dataframe("financials", f"{symbol}/balance_sheet")
        cached_is = client._disk_cache.get_dataframe("financials", f"{symbol}/income_stmt")
        cached_cf = client._disk_cache.get_dataframe("financials", f"{symbol}/cash_flow")
        if cached_bs is not None or cached_is is not None or cached_cf is not None:
            logger.debug("  Cache hit for %s financials", symbol)
            return {
                "balance_sheet": cached_bs if cached_bs is not None else pd.DataFrame(),
                "income_stmt": cached_is if cached_is is not None else pd.DataFrame(),
                "cash_flow": cached_cf if cached_cf is not None else pd.DataFrame(),
            }

    try:
        ticker = client.get_ticker(symbol)
        fg = _detect_financial_group(symbol)

        # Use explicit method calls so we can pass financial_group for
        # bank/financial tickers that need UFRS instead of the default XI_29.
        statements: dict[str, pd.DataFrame] = {}
        for sheet_name, method_name in [
            ("balance_sheet", "get_balance_sheet"),
            ("income_stmt", "get_income_stmt"),
            ("cash_flow", "get_get_cashflow" if hasattr(ticker, "get_get_cashflow") else "get_cashflow"),
        ]:
            # Keep compat with dynamic method call
            real_method = method_name
            if method_name == "get_get_cashflow" and not hasattr(ticker, "get_get_cashflow"):
                real_method = "get_cashflow"

            try:
                raw = getattr(ticker, real_method)(
                    quarterly=True, financial_group=fg, last_n=last_n
                )
                statements[sheet_name] = _format_quarterly_columns(
                    client._coerce_dataframe(raw)
                )
            except Exception as inner_exc:
                logger.debug(
                    "  %s %s fetch failed: %s", symbol, sheet_name, inner_exc
                )
                statements[sheet_name] = pd.DataFrame()

        if all(frame.empty for frame in statements.values()):
            logger.warning(
                f"Empty financial statements from borsapy for {symbol}."
            )
        # Cache non-empty results
        if client._disk_cache is not None:
            for sheet_name, df in statements.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    client._disk_cache.set_dataframe(
                        "financials", f"{symbol}/{sheet_name}", df,
                    )
        return statements
    except SSLError as exc:
        logger.warning(f"SSL error in borsapy financial statements for {symbol}: {exc}")
        return {
            "balance_sheet": pd.DataFrame(),
            "income_stmt": pd.DataFrame(),
            "cash_flow": pd.DataFrame(),
        }
    except Exception as exc:
        logger.error(f"Unexpected error in borsapy financial statements for {symbol}: {exc}")
        return {
            "balance_sheet": pd.DataFrame(),
            "income_stmt": pd.DataFrame(),
            "cash_flow": pd.DataFrame(),
        }


def get_financial_ratios(client: Any, symbol: str) -> pd.DataFrame:
    """Get financial ratios."""
    symbol = client._normalize_symbol(symbol)
    try:
        ticker = client.get_ticker(symbol)
        ratios = client._coerce_dataframe(getattr(ticker, "financial_ratios", None))
        if ratios.empty:
            logger.warning(f"Empty financial ratios from borsapy for {symbol}.")
        return ratios
    except SSLError as exc:
        logger.warning(f"SSL error in borsapy financial ratios for {symbol}: {exc}")
        return pd.DataFrame()
    except Exception as exc:
        logger.error(f"Unexpected error in borsapy financial ratios for {symbol}: {exc}")
        return pd.DataFrame()
