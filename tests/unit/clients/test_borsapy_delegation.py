from __future__ import annotations

import pandas as pd
from unittest.mock import patch, MagicMock

# Mock out the BORSAPY_AVAILABLE constant or imports before loading the client so it imports smoothly
with patch("bist_quant.clients.borsapy_client.BORSAPY_AVAILABLE", True), \
     patch("bist_quant.clients.borsapy_client.bp") as mock_bp:
    
    from bist_quant.clients.borsapy_client import BorsapyClient


def test_borsapy_client_delegates_price_history() -> None:
    client = MagicMock(spec=BorsapyClient)
    
    with patch("bist_quant.clients.borsapy_prices.get_history") as mock_get_history:
        BorsapyClient.get_history(client, symbol="THYAO", period="1y", interval="1d")
        mock_get_history.assert_called_once_with(client, "THYAO", "1y", "1d", None, None, True)


def test_borsapy_client_delegates_batch_download() -> None:
    client = MagicMock(spec=BorsapyClient)
    
    with patch("bist_quant.clients.borsapy_prices.batch_download") as mock_download:
        BorsapyClient.batch_download(client, ["THYAO"], "1y", "1d", None, None, "ticker", True)
        mock_download.assert_called_once_with(client, ["THYAO"], "1y", "1d", None, None, "ticker", True)


def test_borsapy_client_delegates_to_long_ohlcv() -> None:
    client = MagicMock(spec=BorsapyClient)
    df = pd.DataFrame()
    
    with patch("bist_quant.clients.borsapy_prices.to_long_ohlcv") as mock_to_long:
        BorsapyClient.to_long_ohlcv(client, df, symbol_hint="THYAO", add_is_suffix=True)
        mock_to_long.assert_called_once_with(client, df, "THYAO", True)


def test_borsapy_client_delegates_batch_download_to_long() -> None:
    client = MagicMock(spec=BorsapyClient)
    
    with patch("bist_quant.clients.borsapy_prices.batch_download_to_long") as mock_download_long:
        BorsapyClient.batch_download_to_long(client, ["THYAO"], "1y", "1d", None, None, "ticker", True, False)
        mock_download_long.assert_called_once_with(client, ["THYAO"], "1y", "1d", None, None, "ticker", True, False)


def test_borsapy_client_delegates_cache_io() -> None:
    client = MagicMock(spec=BorsapyClient)
    df = pd.DataFrame()
    
    with patch("bist_quant.clients.borsapy_prices.save_to_cache") as mock_save, \
         patch("bist_quant.clients.borsapy_prices.load_from_cache") as mock_load:
         
        BorsapyClient.save_to_cache(client, df, "prices_cache", "csv")
        mock_save.assert_called_once_with(client, df, "prices_cache", "csv")
        
        BorsapyClient.load_from_cache(client, "prices_cache", "parquet", 12)
        mock_load.assert_called_once_with(client, "prices_cache", "parquet", 12)


def test_borsapy_client_delegates_financials() -> None:
    client = MagicMock(spec=BorsapyClient)
    
    with patch("bist_quant.clients.borsapy_financials.get_financials") as mock_get_fin:
        BorsapyClient.get_financials(client, symbol="THYAO")
        mock_get_fin.assert_called_once_with(client, "THYAO")


def test_borsapy_client_delegates_financial_statements() -> None:
    client = MagicMock(spec=BorsapyClient)
    
    with patch("bist_quant.clients.borsapy_financials.get_financial_statements") as mock_get_statements:
        BorsapyClient.get_financial_statements(client, symbol="THYAO", last_n=10)
        mock_get_statements.assert_called_once_with(client, "THYAO", 10)


def test_borsapy_client_delegates_financial_ratios() -> None:
    client = MagicMock(spec=BorsapyClient)
    
    with patch("bist_quant.clients.borsapy_financials.get_financial_ratios") as mock_get_ratios:
        BorsapyClient.get_financial_ratios(client, symbol="THYAO")
        mock_get_ratios.assert_called_once_with(client, "THYAO")


def test_borsapy_client_delegates_get_index() -> None:
    client = MagicMock(spec=BorsapyClient)
    
    with patch("bist_quant.clients.borsapy_indices.get_index") as mock_get_index:
        BorsapyClient.get_index(client, index="XU030")
        mock_get_index.assert_called_once_with(client, "XU030")


def test_borsapy_client_delegates_get_index_components() -> None:
    client = MagicMock(spec=BorsapyClient)
    
    with patch("bist_quant.clients.borsapy_indices.get_index_components") as mock_get_components:
        BorsapyClient.get_index_components(client, index="XU100")
        mock_get_components.assert_called_once_with(client, "XU100")


def test_borsapy_client_delegates_get_all_indices() -> None:
    client = MagicMock(spec=BorsapyClient)
    
    with patch("bist_quant.clients.borsapy_indices.get_all_indices") as mock_get_all:
        BorsapyClient.get_all_indices(client)
        mock_get_all.assert_called_once_with(client)


def test_borsapy_client_delegates_get_history_with_indicators() -> None:
    client = MagicMock(spec=BorsapyClient)
    
    with patch("bist_quant.clients.borsapy_indices.get_history_with_indicators") as mock_get_history_ind:
        BorsapyClient.get_history_with_indicators(client, symbol="THYAO", indicators=["rsi"], period="1y", interval="1d")
        mock_get_history_ind.assert_called_once_with(client, "THYAO", ["rsi"], "1y", "1d")


def test_borsapy_client_delegates_calculate_rsi() -> None:
    prices = pd.Series()
    with patch("bist_quant.clients.borsapy_indices.calculate_rsi") as mock_rsi:
        BorsapyClient.calculate_rsi(None, prices, 14)
        mock_rsi.assert_called_once_with(prices, 14)


def test_borsapy_client_delegates_calculate_macd() -> None:
    prices = pd.Series()
    with patch("bist_quant.clients.borsapy_indices.calculate_macd") as mock_macd:
        BorsapyClient.calculate_macd(None, prices, 12, 26, 9)
        mock_macd.assert_called_once_with(prices, 12, 26, 9)


def test_borsapy_client_delegates_calculate_supertrend() -> None:
    high, low, close = pd.Series(), pd.Series(), pd.Series()
    with patch("bist_quant.clients.borsapy_indices.calculate_supertrend") as mock_supertrend:
        BorsapyClient.calculate_supertrend(None, high, low, close, 10, 3.0)
        mock_supertrend.assert_called_once_with(high, low, close, 10, 3.0)


