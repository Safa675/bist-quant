from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from bist_quant.clients.base_provider import BaseProvider
from bist_quant.clients.base_mcp import BaseMCPClient
from bist_quant.clients.crypto_client import CryptoClient
from bist_quant.clients.us_stock_client import USStockClient


# =========================================================================
# BaseProvider Tests
# =========================================================================

def test_base_provider_lazy_import() -> None:
    # Construct base provider without pre-injected module
    provider = BaseProvider(borsapy_module=None)
    assert provider._bp is None
    assert not provider._import_attempted
    
    # Mock the get_borsapy_module function
    mock_module = SimpleNamespace(name="mock_borsapy")
    with patch("bist_quant.clients.base_provider.get_borsapy_module", return_value=(mock_module, True)) as mock_import:
        bp_module = provider._get_bp()
        assert bp_module == mock_module
        assert provider._bp == mock_module
        assert provider._import_attempted
        mock_import.assert_called_once_with("BaseProvider", False)
        
        # Second call should return cached without re-importing
        mock_import.reset_mock()
        assert provider._get_bp() == mock_module
        mock_import.assert_not_called()


def test_base_provider_caching_init(tmp_path: Path) -> None:
    # If cache_dir provided, DiskCache should be initialized
    provider = BaseProvider(cache_dir=tmp_path / "cache")
    assert provider._disk_cache is not None
    
    # If invalid cache_dir, handles exceptions gracefully
    provider_invalid = BaseProvider(cache_dir=object())
    assert provider_invalid._disk_cache is None


# =========================================================================
# BaseMCPClient Tests
# =========================================================================

class DummyMCPClient(BaseMCPClient):
    pass


@pytest.mark.asyncio
async def test_base_mcp_client_rpc_success() -> None:
    client = DummyMCPClient(mcp_endpoint="https://mock.mcp/rpc", cache_ttl=60)
    
    # Mock httpx.AsyncClient.post response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "id": "123",
        "result": {
            "content": [
                {"type": "text", "text": "Result from tool call"}
            ]
        }
    }
    
    with patch.object(client._session, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        
        result = await client._call_mcp_async("test_tool", {"param1": "val1"})
        
        # Assert parameters
        mock_post.assert_called_once()
        call_args, call_kwargs = mock_post.call_args
        assert call_args[0] == "https://mock.mcp/rpc"
        payload = call_kwargs["json"]
        assert payload["method"] == "tools/call"
        assert payload["params"]["name"] == "test_tool"
        assert payload["params"]["arguments"] == {"param1": "val1"}
        
        assert "content" in result
        assert client._extract_text_blocks(result) == ["Result from tool call"]
        
    await client.close()


@pytest.mark.asyncio
async def test_base_mcp_client_rpc_error() -> None:
    client = DummyMCPClient(mcp_endpoint="https://mock.mcp/rpc")
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "id": "123",
        "error": {
            "code": -32000,
            "message": "Tool execution failed"
        }
    }
    
    with patch.object(client._session, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        
        with pytest.raises(RuntimeError, match="Tool execution failed"):
            await client._call_mcp_async("test_tool", {})
            
    await client.close()


def test_base_mcp_client_caching() -> None:
    client = DummyMCPClient(cache_ttl=1, cache_dir=None)
    
    key = "cache_test"
    df = pd.DataFrame({"symbol": ["BTC"], "price": [60000.0]})
    
    # Cache miss
    assert client._cache_get(key) is None
    
    # Set cache
    client._cache_set(key, df)
    
    # Cache hit
    cached = client._cache_get(key)
    assert cached is not None
    assert cached.equals(df)
    
    # Expiry
    import time
    time.sleep(1.5)
    assert client._cache_get(key) is None


@pytest.mark.asyncio
async def test_concrete_mcp_clients_parameterized(tmp_path: Path) -> None:
    # Parameterized tests on caching and endpoints using concrete MCP subclasses
    for client_class in (CryptoClient, USStockClient):
        client = client_class(cache_dir=tmp_path / "cache", cache_ttl=10)
        assert client._disk_cache is not None
        
        # Test basic method calls with mocked network response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "result": {"data": []}
        }
        
        with patch.object(client._session, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            
            if isinstance(client, CryptoClient):
                res = await client.get_crypto_markets(exchange="btcturk")
                assert isinstance(res, pd.DataFrame)
            elif isinstance(client, USStockClient):
                res = await client.get_us_stock_history(symbol="AAPL", period="1d")
                assert isinstance(res, pd.DataFrame)
                
        await client.close()
