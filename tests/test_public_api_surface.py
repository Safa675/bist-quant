from __future__ import annotations

import subprocess
import sys
import venv
from pathlib import Path

import bist_quant


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def _venv_bin(venv_dir: Path, name: str) -> Path:
    return venv_dir / "bin" / name


def test_borsapy_adapter_top_level_export_resolves_to_clients_module() -> None:
    assert bist_quant.BorsapyAdapter is not None
    assert bist_quant.BorsapyAdapter.__module__ == "bist_quant.clients.borsapy_adapter"


def test_installed_console_script_reports_package_version(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    wheel_path = _build_wheel(dist_dir)

    venv_dir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True).create(venv_dir)

    subprocess.run(
        [str(_venv_bin(venv_dir, "python")), "-m", "pip", "install", str(wheel_path)],
        check=True,
        cwd=tmp_path,
    )

    completed = subprocess.run(
        [str(_venv_bin(venv_dir, "bist-quant")), "--version"],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert completed.stdout.strip() == f"bist-quant {bist_quant.__version__}"
