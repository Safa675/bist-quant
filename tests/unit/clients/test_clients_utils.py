from __future__ import annotations

import math
import pandas as pd
import pytest

from bist_quant.clients.utils import (
    to_float,
    as_frame,
    pick_column,
    call_if_callable,
)


def test_to_float_formats() -> None:
    # Basic numeric conversion
    assert to_float(42) == 42.0
    assert to_float(3.14) == 3.14
    
    # Turkish separators and percentages
    assert to_float("1.234,56") == 1234.56
    assert to_float("% 25,4") == 25.4
    assert to_float(" -12.345,67 % ") == -12345.67
    
    # Scientific notations
    assert to_float("1.23e4") == 12300.0
    assert to_float("-3.14e-2") == -0.0314
    
    # Invalid/empty/bool values
    assert to_float(None) is None
    assert to_float(True) is None
    assert to_float(False) is None
    assert to_float("") is None
    assert to_float("not a number") is None
    assert to_float("NaN") is None
    assert to_float("inf") is None


def test_as_frame_conversion() -> None:
    # DataFrame returns copy
    df_in = pd.DataFrame({"a": [1, 2]})
    df_out = as_frame(df_in)
    assert df_out is not df_in
    assert df_out.equals(df_in)
    
    # Empty payloads
    assert as_frame(None).empty
    assert as_frame([]).empty
    assert as_frame({}).empty
    
    # List of dicts
    list_dicts = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    df_list = as_frame(list_dicts)
    assert not df_list.empty
    assert list(df_list["a"]) == [1, 3]
    
    # Simple dictionary (scalars) -> single row
    d_scalar = {"a": 1, "b": "hello"}
    df_scalar = as_frame(d_scalar)
    assert len(df_scalar) == 1
    assert df_scalar.loc[0, "a"] == 1
    
    # Dictionary with lists
    d_list = {"a": [1, 2], "b": [3, 4]}
    df_list_cols = as_frame(d_list)
    assert len(df_list_cols) == 2
    assert list(df_list_cols["a"]) == [1, 2]


def test_pick_column_matching() -> None:
    df = pd.DataFrame({"TickerSymbol": [1], "close_price": [2], "Date": [3]})
    
    assert pick_column(df, ["tickersymbol"]) == "TickerSymbol"
    assert pick_column(df, ["symbol", "ticker_symbol", "tickersymbol"]) == "TickerSymbol"
    assert pick_column(df, ["CLOSE_PRICE", "close"]) == "close_price"
    assert pick_column(df, ["not_present"]) is None
    assert pick_column(pd.DataFrame(), ["a"]) is None
    assert pick_column(None, ["a"]) is None


def test_call_if_callable() -> None:
    # Not callable
    assert call_if_callable(42) is None
    assert call_if_callable(None) is None
    
    # Simple function
    def add(x, y):
        return x + y
    
    assert call_if_callable(add, 2, 3) == 5
    
    # Signature discrepancy fallback
    def single_arg(x):
        return x * 2
        
    assert call_if_callable(single_arg, 5, a=10) == 10  # should fall back to single_arg(5)
    
    # Exceptions handled
    def raises_error(x):
        raise ValueError("Oops")
        
    assert call_if_callable(raises_error, 5) is None
