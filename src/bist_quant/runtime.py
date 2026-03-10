from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

APP_DIR_NAME = "bist-quant"
ENV_PROJECT_ROOT = "BIST_PROJECT_ROOT"
ENV_DATA_DIR = "BIST_DATA_DIR"
ENV_REGIME_DIR = "BIST_REGIME_DIR"
ENV_CACHE_DIR = "BIST_CACHE_DIR"


class RuntimePathError(RuntimeError):
    """Raised when the BIST runtime paths are not configured correctly."""


@dataclass(frozen=True)
class RuntimePaths:
    project_root: Path
    data_dir: Path
    regime_dir: Path
    regime_outputs_dir: Path

    def to_backend_paths(self) -> "BackendPaths":
        return BackendPaths(
            project_root=self.project_root,
            data_dir=self.data_dir,
            regime_outputs_dir=self.regime_outputs_dir,
        )


@dataclass(frozen=True)
class BackendPaths:
    project_root: Path
    data_dir: Path
    regime_outputs_dir: Path


def _expand_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def _xdg_data_home() -> Path:
    return (
        Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
        .expanduser()
        .resolve()
    )


def _xdg_cache_home() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser().resolve()


def default_project_root(*, create: bool = False) -> Path:
    root = _xdg_data_home() / APP_DIR_NAME
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def default_data_dir(*, create: bool = False) -> Path:
    data_dir = default_project_root(create=create) / "data"
    if create:
        data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def default_regime_dir(*, create: bool = False) -> Path:
    regime_dir = default_project_root(create=create) / "regime" / "simple_regime"
    if create:
        regime_dir.mkdir(parents=True, exist_ok=True)
    return regime_dir


def default_cache_dir(*, create: bool = False) -> Path:
    cache_dir = _xdg_cache_home() / APP_DIR_NAME
    if create:
        cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _default_project_root() -> Path:
    """Return the library's user-scoped home directory."""
    return default_project_root(create=True)


def _resolve_regime_dir(project_root: Path, explicit_regime_dir: Path | None) -> Path:
    if explicit_regime_dir is not None:
        return explicit_regime_dir
    regime_dir = project_root / "regime" / "simple_regime"
    regime_dir.mkdir(parents=True, exist_ok=True)
    return regime_dir


def _resolve_regime_outputs_dir(regime_dir: Path) -> Path:
    if regime_dir.name.lower() == "outputs":
        return regime_dir
    if (regime_dir / "regime_features.csv").exists():
        return regime_dir
    return regime_dir / "outputs"


def resolve_runtime_paths(
    *,
    project_root: str | Path | None = None,
    data_dir: str | Path | None = None,
    regime_dir: str | Path | None = None,
) -> RuntimePaths:
    explicit_project_root = _expand_path(project_root) or _expand_path(os.getenv(ENV_PROJECT_ROOT))
    explicit_data_dir = _expand_path(data_dir) or _expand_path(os.getenv(ENV_DATA_DIR))
    explicit_regime_dir = _expand_path(regime_dir) or _expand_path(os.getenv(ENV_REGIME_DIR))

    resolved_project_root = explicit_project_root or _default_project_root()
    if explicit_data_dir is not None:
        resolved_data_dir = explicit_data_dir
    elif explicit_project_root is not None:
        resolved_data_dir = resolved_project_root / "data"
        resolved_data_dir.mkdir(parents=True, exist_ok=True)
    else:
        resolved_data_dir = default_data_dir(create=True)
    resolved_regime_dir = _resolve_regime_dir(resolved_project_root, explicit_regime_dir)
    resolved_regime_outputs_dir = _resolve_regime_outputs_dir(resolved_regime_dir)

    return RuntimePaths(
        project_root=resolved_project_root,
        data_dir=resolved_data_dir,
        regime_dir=resolved_regime_dir,
        regime_outputs_dir=resolved_regime_outputs_dir,
    )


def validate_runtime_paths(
    paths: RuntimePaths,
    *,
    require_price_data: bool = True,
    require_regime_outputs: bool = False,
) -> None:
    errors: list[str] = []

    if not paths.project_root.exists():
        errors.append(
            "BIST project root does not exist: "
            f"{paths.project_root}. Set {ENV_PROJECT_ROOT} or pass project_root explicitly."
        )

    if not paths.data_dir.exists():
        errors.append(
            "BIST data directory does not exist: "
            f"{paths.data_dir}. Set {ENV_DATA_DIR} or pass data_dir explicitly."
        )

    if require_price_data:
        price_csv = paths.data_dir / "bist_prices_full.csv"
        price_parquet = price_csv.with_suffix(".parquet")
        price_csv_gz = paths.data_dir / "bist_prices_full.csv.gz"
        if not price_csv.exists() and not price_parquet.exists() and not price_csv_gz.exists():
            errors.append(
                "Missing price dataset. Expected one of: "
                f"{price_csv}, {price_parquet}, or {price_csv_gz}. "
                f"Set {ENV_DATA_DIR} to a valid data directory."
            )

    if require_regime_outputs and not paths.regime_outputs_dir.exists():
        errors.append(
            "BIST regime outputs directory does not exist: "
            f"{paths.regime_outputs_dir}. Set {ENV_REGIME_DIR} to a valid regime path."
        )

    if errors:
        raise RuntimePathError(" | ".join(errors))
