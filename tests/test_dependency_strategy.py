from __future__ import annotations

import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_published_library_does_not_advertise_app_only_extras() -> None:
    extras = _pyproject()["project"]["optional-dependencies"]

    assert sorted(extras) == ["dev"]


def test_core_dependencies_include_provider_stack() -> None:
    dependencies = _pyproject()["project"]["dependencies"]

    assert any(dep.startswith("httpx>=") for dep in dependencies)
    assert any(dep.startswith("yfinance>=") for dep in dependencies)
    assert any(dep.startswith("borsapy>=") for dep in dependencies)


def test_docs_use_single_install_story() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    install_doc = (REPO_ROOT / "docs/getting-started/installation.md").read_text(encoding="utf-8")

    assert 'pip install "bist-quant[' not in readme
    assert 'pip install "bist-quant[' not in install_doc
    assert "Optional extras:" not in readme
    assert "Optional extras:" not in install_doc
