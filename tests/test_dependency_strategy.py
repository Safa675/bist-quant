from __future__ import annotations

import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_published_library_does_not_advertise_app_only_extras() -> None:
    extras = _pyproject()["project"]["optional-dependencies"]

    for unexpected in ("api", "security", "observability", "services", "app"):
        assert unexpected not in extras


def test_provider_extras_cover_shipped_optional_clients() -> None:
    extras = _pyproject()["project"]["optional-dependencies"]

    assert "providers" in extras
    assert any(dep.startswith("httpx>=") for dep in extras["providers"])
    assert any(dep.startswith("yfinance>=") for dep in extras["providers"])
    assert any(dep.startswith("borsapy>=") for dep in extras["providers"])


def test_full_extra_only_references_library_relevant_extras() -> None:
    extras = _pyproject()["project"]["optional-dependencies"]

    assert extras["full"] == ["bist-quant[providers,multi-asset,borsapy,ml]"]
