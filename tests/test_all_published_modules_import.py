from __future__ import annotations

import importlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "bist_quant"
EXCLUDED_TOP_LEVEL_PACKAGES = {
    "api",
    "engines",
    "jobs",
    "observability",
    "persistence",
    "security",
    "services",
}


def _iter_published_modules() -> list[str]:
    modules: list[str] = []

    for file_path in sorted(PACKAGE_ROOT.rglob("*.py")):
        relative = file_path.relative_to(PACKAGE_ROOT)
        parts = relative.parts
        if parts[0] in EXCLUDED_TOP_LEVEL_PACKAGES:
            continue
        if file_path.name == "__main__.py":
            continue

        module_parts = ["bist_quant", *parts]
        if module_parts[-1] == "__init__.py":
            module_parts = module_parts[:-1]
        else:
            module_parts[-1] = module_parts[-1][:-3]

        modules.append(".".join(module_parts))

    return sorted(set(modules))


def test_all_published_modules_import() -> None:
    failed: list[tuple[str, str]] = []

    for module_name in _iter_published_modules():
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - exercised on failure only
            failed.append((module_name, f"{type(exc).__name__}: {exc}"))

    assert not failed, "\n".join(f"{name} -> {error}" for name, error in failed)
