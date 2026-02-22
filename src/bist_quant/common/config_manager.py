"""
Configuration Manager - BIST Quant

Supports loading strategy configurations from:
1. Python modules (Models/configs/*.py) - preferred
2. YAML files (configs/*.yaml) - legacy fallback

Python configs take precedence over YAML.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Iterable, Optional

import yaml

from bist_quant.common.enums import RegimeLabel

logger = logging.getLogger(__name__)

# Base paths for class-level convenience methods.
BIST_ROOT = Path(__file__).resolve().parents[2]
PYTHON_CONFIGS_DIR = BIST_ROOT / "Models" / "configs"
YAML_CONFIGS_DIR = BIST_ROOT / "bist_quant" / "configs"

REGIME_ALLOCATIONS: dict[RegimeLabel, float] = {
    RegimeLabel.BULL: 1.0,
    RegimeLabel.RECOVERY: 1.0,
    RegimeLabel.STRESS: 0.0,
    RegimeLabel.BEAR: 0.0,
}

DEFAULT_PORTFOLIO_OPTIONS = {
    "use_regime_filter": True,
    "use_vol_targeting": True,
    "target_downside_vol": 0.20,
    "vol_lookback": 63,
    "vol_floor": 0.10,
    "vol_cap": 1.0,
    "use_inverse_vol_sizing": True,
    "inverse_vol_lookback": 60,
    "max_position_weight": 0.25,
    "use_stop_loss": True,
    "stop_loss_threshold": 0.15,
    "use_liquidity_filter": True,
    "liquidity_quantile": 0.25,
    "use_slippage": True,
    "slippage_bps": 5.0,
    "use_mcap_slippage": True,
    "small_cap_slippage_bps": 20.0,
    "mid_cap_slippage_bps": 10.0,
    "top_n": 20,
    "signal_lag_days": 1,
}

TOP_N = 20
LIQUIDITY_QUANTILE = 0.25
POSITION_STOP_LOSS = 0.15
SLIPPAGE_BPS = 5.0
TARGET_DOWNSIDE_VOL = 0.20
VOL_LOOKBACK = 63
VOL_FLOOR = 0.10
VOL_CAP = 1.0
INVERSE_VOL_LOOKBACK = 60
MAX_POSITION_WEIGHT = 0.25

ConfigDict = dict[str, Any]


class ConfigError(ValueError):
    """Raised when strategy configuration is invalid."""


@dataclass(frozen=True)
class ConfigManager:
    """Load and validate strategy configurations from Python and YAML sources.

    The manager supports both a modern package layout (`bist_quant/configs`) and
    legacy layouts (`Models/configs`, `configs/strategies.yaml`) to preserve
    backward compatibility.

    Attributes:
        project_root: Repository root used for resolving config paths.
        models_dir: Package root that contains the `configs` module directory.
    """

    project_root: Path
    models_dir: Path

    _cache: ClassVar[dict[str, ConfigDict]] = {}
    _use_cache: ClassVar[bool] = True

    @classmethod
    def from_default_paths(cls) -> ConfigManager:
        models_dir = Path(__file__).resolve().parents[1]
        project_root = models_dir.parent
        return cls(project_root=project_root, models_dir=models_dir)

    @staticmethod
    def deep_merge(base: ConfigDict, override: ConfigDict) -> ConfigDict:
        result = deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigManager.deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        return result

    @staticmethod
    def _validate_strategy_config(name: str, config: ConfigDict) -> ConfigDict:
        if not isinstance(config, dict):
            raise ConfigError(f"Strategy '{name}' config must be a dict")

        validated = dict(config)
        validated["name"] = name

        description = validated.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ConfigError(f"Strategy '{name}' missing required non-empty 'description'")

        timeline = validated.get("timeline")
        if timeline is not None and not isinstance(timeline, dict):
            raise ConfigError(f"Strategy '{name}' field 'timeline' must be a dict when present")

        parameters = validated.get("parameters")
        if parameters is not None and not isinstance(parameters, dict):
            raise ConfigError(f"Strategy '{name}' field 'parameters' must be a dict when present")

        signal_params = validated.get("signal_params")
        if signal_params is not None and not isinstance(signal_params, dict):
            raise ConfigError(
                f"Strategy '{name}' field 'signal_params' must be a dict when present"
            )

        portfolio_options = validated.get("portfolio_options")
        if portfolio_options is not None and not isinstance(portfolio_options, dict):
            raise ConfigError(
                f"Strategy '{name}' field 'portfolio_options' must be a dict when present"
            )

        return validated

    @staticmethod
    def _coerce_strategy_config(name: str, config: ConfigDict) -> ConfigDict:
        coerced = dict(config)
        coerced.setdefault("name", name)
        return coerced

    @classmethod
    def _parse_yaml_payload(cls, payload: Any) -> dict[str, ConfigDict]:
        if not isinstance(payload, dict):
            return {}

        parsed: dict[str, ConfigDict] = {}

        # Legacy structure: {"defaults": ..., "strategies": {...}}
        if "strategies" in payload and isinstance(payload["strategies"], dict):
            defaults = payload.get("defaults", {})
            if not isinstance(defaults, dict):
                raise ConfigError("YAML field 'defaults' must be a mapping")

            for name, override in payload["strategies"].items():
                if override is None:
                    override = {}
                if not isinstance(override, dict):
                    raise ConfigError(f"YAML strategy '{name}' must be a mapping")
                merged = cls.deep_merge(defaults, override)
                try:
                    parsed[name] = cls._validate_strategy_config(name, merged)
                except ConfigError:
                    parsed[name] = cls._coerce_strategy_config(name, merged)
            return parsed

        # Simple structure: {"momentum": {...}, "value": {...}}
        for name, config in payload.items():
            if not isinstance(config, dict):
                continue
            try:
                parsed[name] = cls._validate_strategy_config(name, config)
            except ConfigError:
                parsed[name] = cls._coerce_strategy_config(name, config)
        return parsed

    def _yaml_path_candidates(self) -> list[Path]:
        return [
            self.project_root / "configs" / "strategies.yaml",
            self.models_dir / "configs" / "strategies.yaml",
        ]

    def _python_config_dir(self) -> Path:
        return self.models_dir / "configs"

    @classmethod
    def _default_yaml_file(cls) -> Path:
        return YAML_CONFIGS_DIR / "strategies.yaml"

    def load_yaml_configs(self) -> dict[str, ConfigDict]:
        yaml_path = next((path for path in self._yaml_path_candidates() if path.exists()), None)
        if yaml_path is None:
            return {}

        try:
            with open(yaml_path, "r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle)
            return self._parse_yaml_payload(payload)
        except Exception as exc:
            logger.warning("Failed to load YAML configs from %s: %s", yaml_path, exc)
            return {}

    def load_legacy_py_configs(self) -> dict[str, ConfigDict]:
        """
        Load Python strategy configs from `Models/configs`.

        Supports both:
        - CONFIG (preferred)
        - SIGNAL_CONFIG (legacy)
        """
        configs: dict[str, ConfigDict] = {}
        config_dir = self._python_config_dir()
        if not config_dir.exists():
            return configs

        for config_file in config_dir.glob("*.py"):
            if config_file.name.startswith("_"):
                continue

            module_name = f"_bist_cfg_{config_file.stem}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, config_file)
                if spec is None or spec.loader is None:
                    raise ConfigError(f"Cannot build module spec for '{config_file.name}'")

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                raw_config = getattr(module, "CONFIG", None)
                if raw_config is None:
                    raw_config = getattr(module, "SIGNAL_CONFIG", None)
                if not isinstance(raw_config, dict):
                    continue

                name = str(raw_config.get("name") or config_file.stem)
                configs[name] = self._coerce_strategy_config(name, raw_config)
            except Exception as exc:
                logger.warning("Failed to load Python config %s: %s", config_file.name, exc)

        return configs

    def load_signal_configs(self, prefer_yaml: bool = True) -> dict[str, ConfigDict]:
        """
        Load all strategy configs.

        Python configs are always loaded first. If `prefer_yaml=True`,
        YAML-only strategies are added as fallback entries.
        """
        configs = self.load_legacy_py_configs()

        if prefer_yaml:
            yaml_configs = self.load_yaml_configs()
            for name, cfg in yaml_configs.items():
                configs.setdefault(name, cfg)

        if not configs:
            logger.warning("No strategy configs loaded from Python or YAML")

        return configs

    @classmethod
    def _load_python_config(cls, strategy_name: str) -> Optional[ConfigDict]:
        module = None
        for module_path in (
            f"bist_quant.configs.{strategy_name}",
            f"Models.configs.{strategy_name}",
        ):
            try:
                module = importlib.import_module(module_path)
                break
            except ImportError:
                continue
        if module is None:
            return None

        config = getattr(module, "CONFIG", None)
        if config is None:
            config = getattr(module, "SIGNAL_CONFIG", None)
        if not isinstance(config, dict):
            return None

        return cls._coerce_strategy_config(strategy_name, config)

    @classmethod
    def _load_yaml_config(cls, strategy_name: str) -> Optional[ConfigDict]:
        yaml_file = cls._default_yaml_file()
        if not yaml_file.exists():
            return None

        try:
            with open(yaml_file, "r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle)
            all_configs = cls._parse_yaml_payload(payload)
            config = all_configs.get(strategy_name)
            if config is None:
                return None
            return cls._coerce_strategy_config(strategy_name, config)
        except Exception as exc:
            logger.warning("Error loading YAML config for %s: %s", strategy_name, exc)
            return None

    @classmethod
    def load_config(
        cls,
        strategy_name: str,
        fallback: Optional[ConfigDict] = None,
        use_cache: bool = True,
    ) -> ConfigDict:
        if use_cache and cls._use_cache and strategy_name in cls._cache:
            return deepcopy(cls._cache[strategy_name])

        config = cls._load_python_config(strategy_name)
        if config is None:
            config = cls._load_yaml_config(strategy_name)

        if config is None:
            if fallback is not None:
                return deepcopy(fallback)
            raise ValueError(f"Configuration not found for strategy: {strategy_name}")

        if cls._use_cache:
            cls._cache[strategy_name] = deepcopy(config)

        return deepcopy(config)

    @classmethod
    def load_param_ranges(cls, strategy_name: str) -> Optional[dict[str, list[Any]]]:
        module = None
        for module_path in (
            f"bist_quant.configs.{strategy_name}",
            f"Models.configs.{strategy_name}",
        ):
            try:
                module = importlib.import_module(module_path)
                break
            except ImportError:
                continue
        if module is None:
            return None

        ranges = getattr(module, "PARAM_RANGES", None)
        if isinstance(ranges, dict):
            return deepcopy(ranges)
        return None

    @classmethod
    def list_available(cls, include_yaml: bool = True) -> list[str]:
        strategies: set[str] = set()

        if PYTHON_CONFIGS_DIR.exists():
            for config_file in PYTHON_CONFIGS_DIR.glob("*.py"):
                if config_file.name.startswith("_") or config_file.name == "__init__.py":
                    continue
                strategies.add(config_file.stem)

        if include_yaml:
            yaml_file = cls._default_yaml_file()
            if yaml_file.exists():
                try:
                    with open(yaml_file, "r", encoding="utf-8") as handle:
                        payload = yaml.safe_load(handle)
                    strategies.update(cls._parse_yaml_payload(payload).keys())
                except Exception:
                    pass

        return sorted(strategies)

    @classmethod
    def get_config_source(cls, strategy_name: str) -> Optional[str]:
        if cls._load_python_config(strategy_name) is not None:
            return "python"
        if cls._load_yaml_config(strategy_name) is not None:
            return "yaml"
        return None

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache.clear()

    @classmethod
    def disable_cache(cls) -> None:
        cls._use_cache = False
        cls._cache.clear()

    @classmethod
    def enable_cache(cls) -> None:
        cls._use_cache = True


def load_config(strategy_name: str, **kwargs: Any) -> ConfigDict:
    """Load one strategy configuration by name.

    Args:
        strategy_name: Strategy identifier (for example, ``"momentum"``).
        **kwargs: Optional arguments forwarded to :meth:`ConfigManager.load_config`.

    Returns:
        Configuration dictionary for the requested strategy.
    """
    return ConfigManager.load_config(strategy_name, **kwargs)


def load_signal_configs(
    signal_names: Optional[Iterable[str]] = None,
    prefer_yaml: bool = True,
) -> dict[str, ConfigDict]:
    """
    Backward and forward compatible loader.

    Supported call styles:
    - load_signal_configs() -> all configs
    - load_signal_configs(prefer_yaml=False) -> all configs without YAML fallback
    - load_signal_configs(["momentum", "value"]) -> selected configs
    """
    # Backward-compat positional usage: load_signal_configs(False)
    if isinstance(signal_names, bool):
        prefer_yaml = signal_names
        signal_names = None

    if signal_names is None:
        manager = ConfigManager.from_default_paths()
        return manager.load_signal_configs(prefer_yaml=prefer_yaml)

    configs: dict[str, ConfigDict] = {}
    for name in signal_names:
        try:
            configs[str(name)] = ConfigManager.load_config(str(name))
        except ValueError:
            logger.warning("Config not found for signal: %s", name)
    return configs


def list_available_strategies() -> list[str]:
    """List all available strategy configuration names."""
    return ConfigManager.list_available()


__all__ = [
    "ConfigError",
    "ConfigManager",
    "ConfigDict",
    "REGIME_ALLOCATIONS",
    "DEFAULT_PORTFOLIO_OPTIONS",
    "TOP_N",
    "LIQUIDITY_QUANTILE",
    "POSITION_STOP_LOSS",
    "SLIPPAGE_BPS",
    "TARGET_DOWNSIDE_VOL",
    "VOL_LOOKBACK",
    "VOL_FLOOR",
    "VOL_CAP",
    "INVERSE_VOL_LOOKBACK",
    "MAX_POSITION_WEIGHT",
    "load_config",
    "load_signal_configs",
    "list_available_strategies",
]
