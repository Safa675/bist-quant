from __future__ import annotations

import re

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.10 fallback
from pathlib import Path

import bist_quant


REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_VERSION = bist_quant.__version__


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_pyproject_version_matches_package_version() -> None:
    pyproject = tomllib.loads(_read("pyproject.toml"))
    assert pyproject["project"]["version"] == CANONICAL_VERSION


def test_docs_versions_match_package_version() -> None:
    version_markers = {
        "README.md": f"Version {CANONICAL_VERSION}.",
        "docs/index.md": f"Version {CANONICAL_VERSION}.",
        "src/bist_quant/README.md": f"Version: **{CANONICAL_VERSION}**",
    }

    for path, expected in version_markers.items():
        assert expected in _read(path), f"expected {path} to contain {expected!r}"


def test_cli_and_api_versions_match_package_version() -> None:
    cli_source = _read("src/bist_quant/_cli.py")
    api_source = _read("src/bist_quant/api/main.py")

    api_match = re.search(r'FastAPI\(title="BIST Quant API", version=([^\)]+)\)', api_source)

    assert "from . import __version__" in cli_source
    assert 'version=f"%(prog)s {__version__}"' in cli_source
    assert api_match is not None
    assert api_match.group(1).strip() == "__version__"
