import pandas as pd

from bist_quant.fetcher.update_prices import _normalize_borsapy_xau


def test_normalize_borsapy_xau_handles_try_quoted_switch() -> None:
    idx = pd.to_datetime(
        [
            "2026-01-29 00:00:00",
            "2026-01-30 03:00:00",
            "2026-02-04 03:00:00",
        ]
    )
    xau_raw = pd.Series([5318.39990234375, 122248.03, 122267.13], index=idx)
    usd_try = pd.Series([43.40729904174805, 43.499, 43.5251], index=idx)

    xau_usd, xau_try, converted_try_rows = _normalize_borsapy_xau(xau_raw, usd_try, anchor_try=None)

    assert converted_try_rows == 2

    # First row behaves like USD-quoted ons-altin -> converted to TRY via USDTRY.
    assert xau_try.iloc[0] > 200_000
    assert xau_usd.iloc[0] > 5_000

    # Following rows behave like TRY-quoted ons-altin -> no second multiplication.
    assert abs(xau_try.iloc[1] - 122248.03) < 1e-9
    assert abs(xau_try.iloc[2] - 122267.13) < 1e-9
    assert xau_usd.iloc[1] < 4_000
    assert xau_usd.iloc[2] < 4_000
