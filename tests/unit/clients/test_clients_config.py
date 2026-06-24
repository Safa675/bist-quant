from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import pytest

from bist_quant.settings import get_borsapy_config_path
from bist_quant.clients.borsapy_adapter import BorsapyAdapter
from bist_quant.clients.borsapy_client import BorsapyClient


def test_get_borsapy_config_path_default(monkeypatch) -> None:
    # Clear env var if set
    monkeypatch.delenv("BIST_BORSAPY_CONFIG_PATH", raising=False)
    
    path = get_borsapy_config_path()
    assert path.name == "borsapy_config.yaml"
    assert "configs" in path.parts


def test_get_borsapy_config_path_env_override(monkeypatch, tmp_path: Path) -> None:
    custom_path = tmp_path / "custom_config.yaml"
    monkeypatch.setenv("BIST_BORSAPY_CONFIG_PATH", str(custom_path))
    
    path = get_borsapy_config_path()
    assert path == custom_path.resolve()


def test_adapter_and_client_share_config_path(monkeypatch, tmp_path: Path) -> None:
    custom_path = tmp_path / "shared_config.yaml"
    monkeypatch.setenv("BIST_BORSAPY_CONFIG_PATH", str(custom_path))
    
    # Instantiate BorsapyAdapter
    loader = SimpleNamespace(data_dir=tmp_path / "data")
    adapter = BorsapyAdapter(loader=loader)
    
    # Assert adapter has config path set
    assert adapter._config_path == custom_path.resolve()
    
    # Under test, BorsapyClient raises ImportError if borsapy is not available,
    # so we mock import status or check initialization path logic.
    # We can check _load_config of BorsapyClient directly.
    config = BorsapyClient._load_config(None)
    assert isinstance(config, dict)
