import os
from pathlib import Path

from .settings import ProductionSettings
from .environment import parse_env_bool, parse_env_int, parse_env_list

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_output_dir(*subpaths: str) -> Path:
    """
    Get the default output directory for writing results.

    Priority:
    1. BIST_QUANT_OUTPUT_DIR environment variable
    2. 'outputs' folder in current working directory (os.getcwd() / "outputs")

    Args:
        *subpaths: Optional string parts to append to the base output directory
                  (e.g., 'signals', 'five_factor')

    This ensures the library never writes to its own directory or clutters
    the user's root directory, making it safe to use as an importable library.
    """
    default_dir = Path(os.getcwd()) / "outputs"
    base_dir = Path(os.environ.get("BIST_QUANT_OUTPUT_DIR", default_dir))
    
    if subpaths:
        target_dir = base_dir.joinpath(*subpaths)
    else:
        target_dir = base_dir
        
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


__all__ = [
    "ProductionSettings",
    "parse_env_bool",
    "parse_env_int",
    "parse_env_list",
    "get_output_dir",
    "PROJECT_ROOT",
]
