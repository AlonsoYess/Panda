"""I/O helpers for reproducible pipeline outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def ensure_dir(path: Path) -> Path:
    """Create a directory if missing and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_output_structure(paths: Dict[str, Path]) -> None:
    """Ensure expected output directories exist."""
    for key in ["outputs_dir", "metadata_dir", "logs_dir", "selected_tiles_dir"]:
        if key in paths:
            ensure_dir(paths[key])


def save_dataframe_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Persist DataFrame to CSV with parent directory creation."""
    ensure_dir(output_path.parent)
    df.to_csv(output_path, index=False)

