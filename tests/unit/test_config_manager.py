"""Unit tests for ConfigManager."""

from __future__ import annotations

import pytest

from bist_quant import ConfigManager, load_config, load_signal_configs


class TestConfigManager:
    """Tests for ConfigManager class."""

    def test_config_manager_initialization(self) -> None:
        """Test ConfigManager can be initialized from default paths."""
        manager = ConfigManager.from_default_paths()
        assert manager is not None

    def test_load_config_function(self) -> None:
        """Test load_config returns valid configuration."""
        config = load_config("momentum")
        assert config is not None
        assert isinstance(config, dict)

    def test_load_invalid_config(self) -> None:
        """Test load_config raises error for invalid signal."""
        with pytest.raises(ValueError):
            load_config("nonexistent_signal_xyz")

    def test_load_signal_configs(self) -> None:
        """Test load_signal_configs returns all available configs."""
        configs = load_signal_configs()
        assert isinstance(configs, dict)
        assert len(configs) > 0
