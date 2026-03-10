from __future__ import annotations

import json
import subprocess
import sys
import venv
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PACKAGE_PREFIXES = [
    "bist_quant/api/",
    "bist_quant/engines/",
    "bist_quant/jobs/",
    "bist_quant/observability/",
    "bist_quant/persistence/",
    "bist_quant/security/",
    "bist_quant/services/",
]


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


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / "bin" / "python"


def test_built_wheel_excludes_app_facing_packages(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()

    wheel_path = _build_wheel(dist_dir)

    with zipfile.ZipFile(wheel_path) as wheel_zip:
        names = wheel_zip.namelist()

    unexpected = [
        prefix
        for prefix in EXCLUDED_PACKAGE_PREFIXES
        if any(name.startswith(prefix) for name in names)
    ]
    assert unexpected == [], f"wheel should not contain app-facing packages: {unexpected}"


def test_installed_wheel_hides_app_only_top_level_exports(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    wheel_path = _build_wheel(dist_dir)

    venv_dir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    python_bin = _venv_python(venv_dir)

    subprocess.run(
        [str(python_bin), "-m", "pip", "install", str(wheel_path)],
        check=True,
        cwd=tmp_path,
    )

    script = """
import importlib.util
import json

import bist_quant

result = {
    "has_get_api_app": hasattr(bist_quant, "get_api_app"),
    "has_get_quant_router": hasattr(bist_quant, "get_quant_router"),
    "api_spec": importlib.util.find_spec("bist_quant.api") is not None,
    "services_spec": importlib.util.find_spec("bist_quant.services") is not None,
    "jobs_spec": importlib.util.find_spec("bist_quant.jobs") is not None,
    "engines_spec": importlib.util.find_spec("bist_quant.engines") is not None,
}
print(json.dumps(result))
"""

    completed = subprocess.run(
        [str(python_bin), "-c", script],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    result = json.loads(completed.stdout)

    assert result == {
        "has_get_api_app": False,
        "has_get_quant_router": False,
        "api_spec": False,
        "services_spec": False,
        "jobs_spec": False,
        "engines_spec": False,
    }
