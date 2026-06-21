"""In-memory screener frame cache."""

from __future__ import annotations

from pathlib import Path

from typing import Any

SCREEN_CACHE_TTL_SEC = 600
SCREEN_CACHE: dict[str, Any] = {
    "built_at": 0.0,
    "as_of": None,
    "frame": None,
    "sector_map": None,
    "close_df": None,
    "data_dir": None,
    "state_token": None,
}



def screen_cache_state_token(data_dir: Path) -> str:
    candidates = [
        data_dir / "bist_prices_full.parquet",
        data_dir / "bist_prices_full.csv",
        data_dir / "bist_prices_full.csv.gz",
        data_dir / "fundamental_data_consolidated.parquet",
        data_dir / "fundamental_data_consolidated.csv",
        data_dir / "fundamental_data_consolidated.csv.gz",
        data_dir / "shares_outstanding_consolidated.parquet",
        data_dir / "shares_outstanding_consolidated.csv",
        data_dir / "shares_outstanding_consolidated.csv.gz",
        data_dir / "bist_sector_classification.parquet",
        data_dir / "bist_sector_classification.csv",
    ]
    rows: list[str] = []
    for path in candidates:
        try:
            stat = path.stat()
            rows.append(f"{path}:{stat.st_mtime_ns}:{stat.st_size}")
        except FileNotFoundError:
            rows.append(f"{path}:missing")
        except OSError:
            rows.append(f"{path}:error")
    return str(hash(tuple(rows)))


