"""
BIST Quant Strategy Configurations.

YAML-based configuration for 30+ trading strategies consolidated into strategies.yaml.
"""
from __future__ import annotations

from typing import Any
from bist_quant.common.config_manager import ConfigManager

_MANAGER = ConfigManager.from_default_paths()

def load_config(strategy_name: str) -> dict[str, Any]:
    return _MANAGER.load_config(strategy_name)

def load_param_ranges(strategy_name: str) -> dict[str, list[Any]] | None:
    return _MANAGER.load_param_ranges(strategy_name)

def list_strategies() -> list[str]:
    return _MANAGER.list_available(include_yaml=True)

def get_strategy_info(strategy_name: str) -> dict[str, Any]:
    config = load_config(strategy_name)
    param_ranges = load_param_ranges(strategy_name)
    return {
        "name": config.get("name", strategy_name),
        "display_name": config.get("display_name", strategy_name.replace("_", " ").title()),
        "description": config.get("description", ""),
        "has_param_ranges": param_ranges is not None,
    }

def get_all_configs() -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for strategy in list_strategies():
        try:
            configs[strategy] = load_config(strategy)
        except ValueError:
            continue
    return configs

__all__ = [
    "load_config",
    "load_param_ranges",
    "list_strategies",
    "get_strategy_info",
    "get_all_configs",
]
