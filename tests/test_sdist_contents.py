from __future__ import annotations

import tarfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_sdist_includes_docs_and_contributing(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()

    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-m", "build", "--sdist", "--outdir", str(dist_dir)],
        check=True,
        cwd=REPO_ROOT,
    )

    sdist = next(dist_dir.glob("bist_quant-*.tar.gz"))
    with tarfile.open(sdist) as tf:
        names = tf.getnames()

    assert any(name.endswith("/CONTRIBUTING.md") for name in names)
    assert any("/docs/" in name for name in names)
