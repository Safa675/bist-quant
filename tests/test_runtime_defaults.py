from __future__ import annotations

import json
import subprocess
import sys
import venv
from pathlib import Path

from bist_quant.common.data_paths import DataPaths, reset_data_paths
from bist_quant.runtime import resolve_runtime_paths


REPO_ROOT = Path(__file__).resolve().parents[1]


def _expected_user_dirs(base: Path) -> dict[str, Path]:
    data_home = base / ".local" / "share"
    cache_home = base / ".cache"
    root = data_home / "bist-quant"
    return {
        "project_root": root,
        "data_dir": root / "data",
        "regime_dir": root / "regime" / "simple_regime",
        "cache_dir": cache_home / "bist-quant",
    }


def _build_wheel(dist_dir: Path) -> Path:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(REPO_ROOT),
            "--no-deps",
            "--wheel-dir",
            str(dist_dir),
        ],
        check=True,
        cwd=dist_dir,
    )
    wheels = sorted(dist_dir.glob("bist_quant-*.whl"))
    assert wheels, "expected pip wheel to produce a bist_quant wheel"
    return wheels[-1]


def test_source_tree_data_paths_default_to_user_scoped_dirs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delenv("BIST_DATA_DIR", raising=False)
    monkeypatch.delenv("BIST_REGIME_DIR", raising=False)
    monkeypatch.delenv("BIST_CACHE_DIR", raising=False)
    reset_data_paths()

    paths = DataPaths()
    expected = _expected_user_dirs(tmp_path)

    assert paths.data_dir == expected["data_dir"]
    assert paths.regime_dir == expected["regime_dir"]
    assert paths.cache_dir == expected["cache_dir"]


def test_source_tree_runtime_defaults_to_user_scoped_dirs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delenv("BIST_PROJECT_ROOT", raising=False)
    monkeypatch.delenv("BIST_DATA_DIR", raising=False)
    monkeypatch.delenv("BIST_REGIME_DIR", raising=False)

    runtime = resolve_runtime_paths()
    expected = _expected_user_dirs(tmp_path)

    assert runtime.project_root == expected["project_root"]
    assert runtime.data_dir == expected["data_dir"]
    assert runtime.regime_dir == expected["regime_dir"]


def test_installed_wheel_uses_user_scoped_runtime_defaults(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    wheel_path = _build_wheel(dist_dir)

    venv_dir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    python_bin = venv_dir / "bin" / "python"

    subprocess.run(
        [str(python_bin), "-m", "pip", "install", str(wheel_path)], check=True, cwd=tmp_path
    )

    home_dir = tmp_path / "home"
    work_dir = tmp_path / "workspace"
    home_dir.mkdir()
    work_dir.mkdir()
    env = {
        "HOME": str(home_dir),
        "PATH": str(venv_dir / "bin"),
        "VIRTUAL_ENV": str(venv_dir),
    }

    script = """
import json
from bist_quant.common.data_paths import DataPaths
from bist_quant.runtime import resolve_runtime_paths

paths = DataPaths()
runtime = resolve_runtime_paths()
print(json.dumps({
    'data_dir': str(paths.data_dir),
    'regime_dir': str(paths.regime_dir),
    'cache_dir': str(paths.cache_dir),
    'project_root': str(runtime.project_root),
    'runtime_data_dir': str(runtime.data_dir),
    'runtime_regime_dir': str(runtime.regime_dir),
}))
"""
    completed = subprocess.run(
        [str(python_bin), "-c", script],
        check=True,
        capture_output=True,
        text=True,
        cwd=work_dir,
        env=env,
    )

    result = json.loads(completed.stdout)
    expected = _expected_user_dirs(home_dir)

    assert result == {
        "data_dir": str(expected["data_dir"]),
        "regime_dir": str(expected["regime_dir"]),
        "cache_dir": str(expected["cache_dir"]),
        "project_root": str(expected["project_root"]),
        "runtime_data_dir": str(expected["data_dir"]),
        "runtime_regime_dir": str(expected["regime_dir"]),
    }
