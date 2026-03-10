from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_root_readme_leads_with_pip_install_and_not_app_stack() -> None:
    readme = _read("README.md")

    assert "pip install bist-quant" in readme
    assert "Development install (recommended)" not in readme
    assert "## Web App Stack (Next.js + FastAPI)" not in readme
    assert "The migration stack now runs as separate frontend/backend services." not in readme


def test_installation_doc_is_library_first() -> None:
    install_doc = _read("docs/getting-started/installation.md")

    assert "## Install from PyPI" in install_doc
    assert "pip install bist-quant" in install_doc
    assert "## Development Installation" in install_doc
    assert 'pip install -e ".[dev]"' in install_doc
    assert "auto-detects the `data/` directory by searching parent folders" not in install_doc
    assert "Python 3.10, 3.11, 3.12, or 3.13" not in install_doc


def test_docs_home_matches_runtime_defaults_and_library_scope() -> None:
    docs_index = _read("docs/index.md")

    assert "pip install bist-quant" in docs_index
    assert "auto-detects data/ from project root" not in docs_index
    assert "Job system" not in docs_index


def test_mkdocs_nav_does_not_reference_missing_api_reference_pages() -> None:
    mkdocs = _read("mkdocs.yml")

    assert "api-reference/core.md" not in mkdocs
    assert "api-reference/signals.md" not in mkdocs
    assert "api-reference/analytics.md" not in mkdocs
    assert "api-reference/common.md" not in mkdocs
